"""Step Executor for Agent Tasks.

This module provides:
- StepExecutor: Execute individual steps with recovery
- ExecutionResult: Results from step execution
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, TypeVar
import logging
import traceback

from ..core.agent_state import AgentState, StateManager
from ..core.agent_context import AgentContext, ContextKey
from .execution_plan import Step, StepStatus, ExecutionPlan

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class ExecutionResult:
    """Result from step execution."""
    
    success: bool
    step_id: str
    step_name: str
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_ms: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "step_id": self.step_id,
            "step_name": self.step_name,
            "message": self.message,
            "data": self.data,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


class StepExecutor:
    """Executes steps with error handling and recovery.
    
    Features:
    - Step execution with timeout
    - Automatic retry on failure
    - Progress callbacks
    - Error recovery integration
    - Context management
    """
    
    def __init__(
        self,
        state_manager: StateManager,
        context: AgentContext,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """Initialize step executor.
        
        Args:
            state_manager: State manager for transitions
            context: Agent context for data sharing
            progress_callback: Optional callback for progress updates
        """
        self.state_manager = state_manager
        self.context = context
        self.progress_callback = progress_callback
        
        self._step_handlers: Dict[str, Callable] = {}
        self._recovery_handlers: Dict[str, Callable] = {}
        
        self._stop_requested = False
        self._pause_event = asyncio.Event()
        self._pause_event.set()
    
    def register_handler(
        self,
        step_type: str,
        handler: Callable,
        recovery_handler: Optional[Callable] = None,
    ) -> None:
        """Register a handler for a step type.
        
        Args:
            step_type: Type of step
            handler: Handler function
            recovery_handler: Optional recovery handler
        """
        self._step_handlers[step_type] = handler
        if recovery_handler:
            self._recovery_handlers[step_type] = recovery_handler
        logger.debug(f"Registered handler for step type: {step_type}")
    
    def request_stop(self) -> None:
        """Request execution to stop."""
        self._stop_requested = True
        logger.info("Stop requested")
    
    def pause(self) -> None:
        """Pause execution."""
        self._pause_event.clear()
        logger.info("Execution paused")
    
    def resume(self) -> None:
        """Resume execution."""
        self._pause_event.set()
        self._stop_requested = False
        logger.info("Execution resumed")
    
    async def _check_pause(self) -> None:
        """Check if execution should pause."""
        await self._pause_event.wait()
    
    def _report_progress(self, step: Step, status: str, message: str = "") -> None:
        """Report progress to callback."""
        if self.progress_callback:
            self.progress_callback({
                "step_id": step.id,
                "step_name": step.name,
                "status": status,
                "message": message,
                "progress": step.to_dict(),
            })
    
    async def execute_step(
        self,
        step: Step,
        timeout: Optional[float] = None,
    ) -> ExecutionResult:
        """Execute a single step.
        
        Args:
            step: Step to execute
            timeout: Optional timeout in seconds
            
        Returns:
            ExecutionResult
        """
        if self._stop_requested:
            return ExecutionResult(
                success=False,
                step_id=step.id,
                step_name=step.name,
                message="Execution stopped by request",
                error="STOP_REQUESTED",
            )
        
        await self._check_pause()
        
        handler = self._step_handlers.get(step.step_type.value)
        if not handler:
            logger.error(f"No handler registered for step type: {step.step_type}")
            return ExecutionResult(
                success=False,
                step_id=step.id,
                step_name=step.name,
                message=f"No handler for step type: {step.step_type}",
                error="NO_HANDLER",
            )
        
        step.start()
        self._report_progress(step, "started", f"Starting {step.name}")
        
        start_time = datetime.now()
        
        try:
            if asyncio.iscoroutinefunction(handler):
                if timeout:
                    result = await asyncio.wait_for(
                        handler(step.params, self.context),
                        timeout=timeout
                    )
                else:
                    result = await handler(step.params, self.context)
            else:
                result = handler(step.params, self.context)
            
            step.complete(result)
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            
            self._report_progress(step, "completed", f"Completed {step.name}")
            
            return ExecutionResult(
                success=True,
                step_id=step.id,
                step_name=step.name,
                message=f"Step completed successfully",
                data={"result": result} if result else {},
                duration_ms=int(duration),
            )
            
        except asyncio.TimeoutError:
            error_msg = f"Step timed out after {timeout}s"
            step.fail(error_msg)
            logger.error(f"Step {step.name} timed out")
            
            return ExecutionResult(
                success=False,
                step_id=step.id,
                step_name=step.name,
                message=error_msg,
                error="TIMEOUT",
                duration_ms=int(timeout * 1000) if timeout else 0,
            )
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            step.fail(error_msg)
            logger.exception(f"Step {step.name} failed")
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            
            return ExecutionResult(
                success=False,
                step_id=step.id,
                step_name=step.name,
                message=f"Step failed: {error_msg}",
                error=error_msg,
                duration_ms=int(duration),
            )
    
    async def execute_step_with_retry(
        self,
        step: Step,
        timeout: Optional[float] = None,
    ) -> ExecutionResult:
        """Execute a step with automatic retry.
        
        Args:
            step: Step to execute
            timeout: Optional timeout per attempt
            
        Returns:
            ExecutionResult
        """
        result = await self.execute_step(step, timeout)
        
        while not result.success and step.retry():
            logger.info(f"Retrying step {step.name} (attempt {step.retry_count})")
            result = await self.execute_step(step, timeout)
        
        return result
    
    async def execute_plan(
        self,
        plan: ExecutionPlan,
        parallel: bool = False,
        timeout_per_step: Optional[float] = None,
    ) -> List[ExecutionResult]:
        """Execute an entire plan.
        
        Args:
            plan: Execution plan to execute
            parallel: Whether to execute independent steps in parallel
            timeout_per_step: Optional timeout per step
            
        Returns:
            List of ExecutionResults
        """
        plan.start()
        results: List[ExecutionResult] = []
        
        while not plan.is_complete and not self._stop_requested:
            ready_steps = plan.get_ready_steps()
            
            if not ready_steps:
                if plan.has_failures:
                    logger.error("Plan execution stopped due to failures")
                    break
                logger.warning("No ready steps but plan not complete")
                break
            
            if parallel and len(ready_steps) > 1:
                tasks = [
                    self.execute_step_with_retry(step, timeout_per_step)
                    for step in ready_steps
                ]
                step_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for step, result in zip(ready_steps, step_results):
                    if isinstance(result, Exception):
                        step.fail(str(result))
                        results.append(ExecutionResult(
                            success=False,
                            step_id=step.id,
                            step_name=step.name,
                            error=str(result),
                        ))
                    else:
                        results.append(result)
            else:
                for step in ready_steps:
                    if self._stop_requested:
                        break
                    
                    result = await self.execute_step_with_retry(step, timeout_per_step)
                    results.append(result)
                    
                    if not result.success:
                        break
        
        plan.complete()
        return results
    
    async def execute_with_recovery(
        self,
        step: Step,
        timeout: Optional[float] = None,
    ) -> ExecutionResult:
        """Execute a step with recovery handling.
        
        Args:
            step: Step to execute
            timeout: Optional timeout
            
        Returns:
            ExecutionResult
        """
        result = await self.execute_step_with_retry(step, timeout)
        
        if not result.success:
            recovery_handler = self._recovery_handlers.get(step.step_type.value)
            if recovery_handler:
                logger.info(f"Attempting recovery for step {step.name}")
                
                try:
                    if asyncio.iscoroutinefunction(recovery_handler):
                        recovery_result = await recovery_handler(
                            step.params,
                            self.context,
                            result.error,
                        )
                    else:
                        recovery_result = recovery_handler(
                            step.params,
                            self.context,
                            result.error,
                        )
                    
                    if recovery_result:
                        step.retry_count = 0
                        result = await self.execute_step_with_retry(step, timeout)
                        
                        if result.success:
                            logger.info(f"Recovery successful for step {step.name}")
                
                except Exception as e:
                    logger.error(f"Recovery failed for step {step.name}: {e}")
        
        return result
