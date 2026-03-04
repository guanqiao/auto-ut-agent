"""Core Agent - Base class and core state management."""

import logging
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import asyncio

from pyutagent.agent.base_agent import StepResult
from pyutagent.core.protocols import AgentState, AgentResult
from pyutagent.memory.working_memory import WorkingMemory
from pyutagent.llm.client import LLMClient

logger = logging.getLogger(__name__)


class AgentCore:
    """Core ReAct Agent functionality - state management and basic lifecycle.
    
    This class handles:
    - Basic agent state and lifecycle
    - Pause/Resume/Terminate control
    - Working memory management
    - Progress callbacks
    
    Note: This class does NOT inherit from BaseAgent to avoid abstract method requirements.
    It's designed to be used as a component within ReActAgent, not as a standalone agent.
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        working_memory: WorkingMemory,
        project_path: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """Initialize core agent.
        
        Args:
            llm_client: LLM client for generation
            working_memory: Working memory for context
            project_path: Path to the project
            progress_callback: Optional callback for progress updates
        """
        self.llm_client = llm_client
        self.working_memory = working_memory
        self.project_path = project_path
        self.progress_callback = progress_callback
        
        self.state = AgentState.IDLE
        self._pause_event = asyncio.Event()
        self._pause_event.set()
        
        # Delegate these properties to working_memory
        self.max_iterations = working_memory.max_iterations
        self.target_coverage = working_memory.target_coverage
        
        self.current_test_file: Optional[str] = None
        self.target_class_info: Optional[Dict[str, Any]] = None
        self._stop_requested = False
        self._terminated = False
        
        
    
    def stop(self):
        """Stop agent execution gracefully (legacy, use pause instead)."""
        logger.info("[AgentCore] Stopping agent execution")
        self._stop_requested = True
    
    def pause(self):
        """Pause agent execution.
        
        The agent will pause at the next checkpoint.
        """
        logger.info("[AgentCore] Pausing agent execution")
        self._pause_event.clear()
        self._stop_requested = True
    
    def resume(self):
        """Resume agent execution."""
        logger.info("[AgentCore] Resuming agent execution")
        self._pause_event.set()
        self._stop_requested = False
        self._terminated = False
    
    def terminate(self):
        """Terminate agent execution immediately."""
        logger.info("[AgentCore] Terminating agent execution")
        self._terminated = True
        self._stop_requested = True
    
    def reset(self):
        """Reset agent state."""
        logger.info("[AgentCore] Resetting agent state")
        self.state = AgentState.IDLE
        self._stop_requested = False
        self._terminated = False
        self._pause_event.set()
    
    def _update_state(self, state: AgentState | str, message: str):
        """Update agent state and notify progress.
        
        Args:
            state: New agent state (can be AgentState enum or string)
            message: Status message
        """
        # Handle both AgentState enum and string
        if isinstance(state, str):
            state_name = state
            # Try to convert string to AgentState enum
            try:
                state = AgentState[state]
            except (KeyError, AttributeError):
                pass  # Keep as string if conversion fails
        else:
            state_name = state.name
        
        self.state = state
        if self.progress_callback:
            self.progress_callback({
                "state": state_name,
                "message": message,
                "progress": {
                    "iteration": f"{self.working_memory.current_iteration}/{self.working_memory.max_iterations}",
                    "coverage": f"{self.working_memory.current_coverage:.1%}",
                    "target": f"{self.working_memory.target_coverage:.1%}"
                }
            })
    
    async def _check_pause(self) -> None:
        """Check if agent should pause and wait."""
        if self._pause_event.is_set():
            logger.info("[AgentCore] Waiting for pause event...")
            await self._pause_event.wait()
            logger.info("[AgentCore] Pause event cleared, resuming")
    
    def _create_terminated_result(self, context: str) -> AgentResult:
        """Create result when generation is terminated.
        
        Args:
            context: Context for the message
            
        Returns:
            AgentResult with terminated status
        """
        return AgentResult(
            success=False,
            message=f"Generation terminated by user {context}",
            state=AgentState.FAILED
        )
    
    def _create_success_result(self, coverage: float) -> AgentResult:
        """Create success result when target coverage is reached.
        
        Args:
            coverage: Final coverage percentage
            
        Returns:
            AgentResult with success status
        """
        logger.info(f"[AgentCore] 🎉 Target coverage reached! {coverage:.1%}")
        self._update_state(
            AgentState.COMPLETED,
            f"🎉 Target coverage reached: {coverage:.1%}"
        )
        return AgentResult(
            success=True,
            message=f"Successfully generated tests with {coverage:.1%} coverage",
            test_file=self.current_test_file,
            coverage=coverage,
            iterations=self.current_iteration
        )
    
    def _extract_java_code(self, response: str) -> str:
        """Extract Java code from LLM response.
        
        Args:
            response: LLM response text
            
        Returns:
            Extracted Java code
        """
        from pyutagent.agent.utils.code_extractor import CodeExtractor
        return CodeExtractor.extract_java_code(response)
