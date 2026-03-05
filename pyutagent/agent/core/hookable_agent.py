"""Hookable Agent Base Class.

This module provides an agent base class that integrates with the Hooks system.
"""

import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Awaitable
import asyncio

from ..core.agent_state import AgentState, StateManager
from ..core.agent_context import AgentContext, ContextKey
from ..core.hooks import HookManager, HookType, HookContext, HookResult, trigger_hook
from ..memory.working_memory import WorkingMemory
from ..llm.client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result of agent execution."""
    success: bool
    message: str
    coverage: float = 0.0
    iterations: int = 0
    test_file: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class HookableAgent:
    """Agent base class with Hooks integration.
    
    Features:
    - State management with StateManager
    - Context management with AgentContext
    - Hooks integration for lifecycle events
    - Pause/Resume/Stop support
    - Progress callbacks
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        working_memory: WorkingMemory,
        project_path: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        hook_manager: Optional[HookManager] = None,
    ):
        """Initialize hookable agent.
        
        Args:
            llm_client: LLM client for generation
            working_memory: Working memory for context
            project_path: Path to the project
            progress_callback: Optional callback for progress updates
            hook_manager: Optional hook manager (uses global if None)
        """
        self.llm_client = llm_client
        self.working_memory = working_memory
        self.project_path = Path(project_path)
        self.progress_callback = progress_callback
        
        self.state_manager = StateManager(
            initial_state=AgentState.IDLE,
            on_state_change=self._on_state_change,
        )
        
        self.context = AgentContext()
        self.context.set(ContextKey.PROJECT_PATH, str(project_path))
        self.context.set(ContextKey.TARGET_COVERAGE, working_memory.target_coverage)
        self.context.set(ContextKey.MAX_ITERATIONS, working_memory.max_iterations)
        
        self.hook_manager = hook_manager or HookManager()
        
        self._stop_requested = False
        self._pause_event = asyncio.Event()
        self._pause_event.set()
        self._terminated = False
        
        self._start_time: Optional[datetime] = None
        self._results: List[AgentResult] = []
    
    def _on_state_change(self, transition) -> None:
        """Handle state change callback."""
        logger.info(f"[Agent] State: {transition.from_state.name} -> {transition.to_state.name}")
        
        if self.progress_callback:
            self.progress_callback({
                "event": "state_change",
                "from_state": transition.from_state.name,
                "to_state": transition.to_state.name,
                "message": transition.message,
            })
    
    async def _trigger_hook(
        self,
        hook_type: HookType,
        data: Optional[Dict[str, Any]] = None,
    ) -> HookResult:
        """Trigger a hook and return result.
        
        Args:
            hook_type: Type of hook to trigger
            data: Data to pass to hook
            
        Returns:
            HookResult from hook execution
        """
        return await self.hook_manager.trigger(hook_type, data or {})
    
    async def _check_pause(self) -> bool:
        """Check if paused and wait for resume.
        
        Returns:
            True if should continue, False if stopped
        """
        if self._stop_requested:
            return False
        
        await self._pause_event.wait()
        return not self._stop_requested
    
    def pause(self) -> None:
        """Pause the agent."""
        self._pause_event.clear()
        logger.info("[Agent] Paused")
    
    def resume(self) -> None:
        """Resume the agent."""
        self._pause_event.set()
        self._stop_requested = False
        logger.info("[Agent] Resumed")
    
    def stop(self) -> None:
        """Request the agent to stop."""
        self._stop_requested = True
        self._pause_event.set()
        logger.info("[Agent] Stop requested")
    
    def terminate(self) -> None:
        """Immediately terminate the agent."""
        self._terminated = True
        self._stop_requested = True
        self._pause_event.set()
        logger.info("[Agent] Terminated")
    
    @property
    def is_paused(self) -> bool:
        """Check if agent is paused."""
        return not self._pause_event.is_set()
    
    @property
    def is_running(self) -> bool:
        """Check if agent is running."""
        return self.state_manager.state in (
            AgentState.INITIALIZING,
            AgentState.PLANNING,
            AgentState.PARSING,
            AgentState.GENERATING,
            AgentState.COMPILING,
            AgentState.TESTING,
            AgentState.ANALYZING,
            AgentState.FIXING,
            AgentState.OPTIMIZING,
        )
    
    @property
    def is_stopped(self) -> bool:
        """Check if agent is stopped."""
        return self._stop_requested or self._terminated
    
    def _report_progress(self, event: str, data: Dict[str, Any] = None) -> None:
        """Report progress to callback."""
        if self.progress_callback:
            self.progress_callback({
                "event": event,
                "state": self.state_manager.state.name,
                "iteration": self.context.get(ContextKey.CURRENT_ITERATION, 0),
                **(data or {}),
            })
    
    async def run(self, target_file: str, **kwargs) -> AgentResult:
        """Run the agent on a target file.
        
        Args:
            target_file: Path to target file
            **kwargs: Additional arguments
            
        Returns:
            AgentResult with execution result
        """
        self._start_time = datetime.now()
        self._stop_requested = False
        self._terminated = False
        
        self.context.set(ContextKey.TARGET_FILE, target_file)
        
        await self._trigger_hook(HookType.PRE_AGENT_START, {
            "target_file": target_file,
            "kwargs": kwargs,
        })
        
        self.state_manager.transition(AgentState.INITIALIZING, f"Starting for {target_file}")
        
        try:
            result = await self._execute(target_file, **kwargs)
            
            await self._trigger_hook(HookType.POST_AGENT_STOP, {
                "success": result.success,
                "coverage": result.coverage,
            })
            
            return result
            
        except Exception as e:
            logger.exception(f"[Agent] Error: {e}")
            
            await self._trigger_hook(HookType.ON_ERROR, {
                "error": str(e),
                "phase": self.state_manager.state.name,
            })
            
            self.state_manager.transition(AgentState.FAILED, str(e))
            
            return AgentResult(
                success=False,
                message=f"Error: {e}",
                error=str(e),
            )
    
    @abstractmethod
    async def _execute(self, target_file: str, **kwargs) -> AgentResult:
        """Execute the agent logic.
        
        Args:
            target_file: Path to target file
            **kwargs: Additional arguments
            
        Returns:
            AgentResult with execution result
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        duration_ms = 0
        if self._start_time:
            duration_ms = int((datetime.now() - self._start_time).total_seconds() * 1000)
        
        return {
            "state": self.state_manager.state.name,
            "is_paused": self.is_paused,
            "is_stopped": self.is_stopped,
            "duration_ms": duration_ms,
            "state_history": [t.to_dict() for t in self.state_manager.history],
            "context_keys": list(self.context.keys()),
        }
