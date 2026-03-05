"""Agent worker thread for running Agent tasks."""

import asyncio
import concurrent.futures
import logging
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal

from ...agent.react_agent import ReActAgent, AgentState
from ...agent.enhanced_agent import EnhancedAgent, EnhancedAgentConfig
from ...memory.working_memory import WorkingMemory
from ...llm.client import LLMClient
from ..log_handler import LogEmitter

logger = logging.getLogger(__name__)


class AgentWorker(QThread):
    """Worker thread for running Agent."""

    progress_updated = pyqtSignal(dict)
    state_changed = pyqtSignal(str, str)
    log_message = pyqtSignal(str, str)
    completed = pyqtSignal(dict)
    error = pyqtSignal(str)
    paused = pyqtSignal()
    resumed = pyqtSignal()
    terminated = pyqtSignal()

    def __init__(
        self,
        llm_client: LLMClient,
        project_path: str,
        target_file: str,
        target_coverage: float = 0.8,
        max_iterations: int = 2,
        agent_config: Optional[EnhancedAgentConfig] = None,
        use_enhanced_agent: bool = True,
        incremental_mode: bool = False,
        preserve_passing_tests: bool = True,
        skip_test_analysis: bool = False
    ):
        super().__init__()
        self.llm_client = llm_client
        self.project_path = project_path
        self.target_file = target_file
        self.target_coverage = target_coverage
        self.max_iterations = max_iterations
        self.agent_config = agent_config
        self.use_enhanced_agent = use_enhanced_agent
        self.incremental_mode = incremental_mode
        self.preserve_passing_tests = preserve_passing_tests
        self.skip_test_analysis = skip_test_analysis
        self._is_running = True
        self._is_paused = False
        self.agent: Optional[ReActAgent] = None
        self._lock = asyncio.Lock()
        self._log_emitter: Optional[LogEmitter] = None

    def run(self):
        """Run the agent."""
        self._log_emitter = LogEmitter()
        self._log_emitter.log_message.connect(self._on_log)
        self._log_emitter.install_handler('pyutagent')
        
        try:
            working_memory = WorkingMemory(
                target_coverage=self.target_coverage,
                max_iterations=self.max_iterations,
                current_file=self.target_file
            )

            if self.use_enhanced_agent:
                config = self.agent_config or EnhancedAgentConfig(
                    model_name=self.llm_client.model,
                    enable_error_prediction=True,
                    enable_strategy_optimization=True,
                    enable_self_reflection=True,
                    enable_knowledge_graph=False,
                    enable_pattern_library=True,
                    enable_chain_of_thought=True,
                    enable_metrics=True,
                    incremental_mode=self.incremental_mode,
                    preserve_passing_tests=self.preserve_passing_tests,
                    skip_test_analysis=self.skip_test_analysis,
                )
                self.agent = EnhancedAgent(
                    llm_client=self.llm_client,
                    working_memory=working_memory,
                    project_path=self.project_path,
                    progress_callback=self._on_progress,
                    config=config
                )
                logger.info("[AgentWorker] Using EnhancedAgent with P0-P4 features enabled")
            else:
                self.agent = ReActAgent(
                    llm_client=self.llm_client,
                    working_memory=working_memory,
                    project_path=self.project_path,
                    progress_callback=self._on_progress,
                    model_name=self.llm_client.model,
                    incremental_mode=self.incremental_mode,
                    preserve_passing_tests=self.preserve_passing_tests,
                    skip_test_analysis=self.skip_test_analysis
                )
                logger.info("[AgentWorker] Using ReActAgent (basic mode)")

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._run_agent()
                    )
                    result = future.result()
            else:
                result = loop.run_until_complete(
                    self._run_agent()
                )

            if result.success:
                result_data = {
                    "success": True,
                    "message": result.message,
                    "test_file": result.test_file,
                    "coverage": result.coverage,
                    "iterations": result.iterations,
                    "incremental_mode": self.incremental_mode
                }
                if hasattr(result, 'metadata') and result.metadata:
                    result_data.update({
                        "preserved_tests": result.metadata.get("preserved_tests", 0),
                        "new_tests": result.metadata.get("new_tests", 0),
                        "fixed_tests": result.metadata.get("fixed_tests", 0)
                    })
                self.completed.emit(result_data)
            else:
                self.error.emit(result.message)

        except Exception as e:
            logger.exception("Agent worker failed")
            self.error.emit(str(e))
        finally:
            if self._log_emitter:
                self._log_emitter.uninstall_handler('pyutagent')

    async def _run_agent(self):
        """Run the agent with pause/resume support."""
        return await self.agent.generate_tests(self.target_file)
    
    def _on_log(self, message: str, level: str):
        """Handle log message from LogEmitter."""
        self.log_message.emit(message, level)

    def pause(self):
        """Pause the agent."""
        if self.agent and not self._is_paused:
            self.agent.pause()
            self._is_paused = True
            self.paused.emit()
            logger.info("[AgentWorker] Agent paused")

    def resume(self):
        """Resume the agent."""
        if self.agent and self._is_paused:
            self.agent.resume()
            self._is_paused = False
            self.resumed.emit()
            logger.info("[AgentWorker] Agent resumed")

    def terminate_agent(self):
        """Terminate the agent immediately with multiple strategies."""
        logger.info("[AgentWorker] Starting agent termination")
        
        if self.agent:
            logger.info("[AgentWorker] Calling agent.terminate()")
            self.agent.terminate()
        
        logger.info("[AgentWorker] Requesting thread interruption")
        self.requestInterruption()
        
        if self.isRunning():
            logger.info("[AgentWorker] Waiting for graceful shutdown (max 2 seconds)")
            self.wait(2000)
        
        if self.isRunning():
            logger.warning("[AgentWorker] Force terminating worker thread")
            super().terminate()
            self.wait(500)
        
        self._is_paused = False
        self._is_running = False
        self.terminated.emit()
        logger.info("[AgentWorker] Agent termination complete")

    def is_paused(self) -> bool:
        """Check if agent is paused."""
        return self._is_paused

    def _on_progress(self, progress_info: dict):
        """Handle progress updates."""
        self.progress_updated.emit(progress_info)
        self.state_changed.emit(
            progress_info.get("state", ""),
            progress_info.get("message", "")
        )

    def stop(self):
        """Stop the worker."""
        self._is_running = False
        if self.agent:
            self.agent.stop()
        self.wait(1000)
