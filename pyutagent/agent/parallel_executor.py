"""Parallel Tool Executor - Execute independent tools in parallel.

.. deprecated::
    Use pyutagent.agent.execution.executor.StepExecutor with parallel=True instead.
    This module is kept for backward compatibility.

This module provides:
- ParallelExecutor: Execute tools concurrently (deprecated)
- DependencyResolver: Resolve tool dependencies (deprecated)
- ResultAggregator: Combine results (deprecated)
"""

import asyncio
import logging
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ..tools.tool import ToolResult

logger = logging.getLogger(__name__)

# Emit deprecation warning when module is imported
warnings.warn(
    "pyutagent.agent.parallel_executor is deprecated. "
    "Use pyutagent.agent.execution.executor.StepExecutor with parallel=True instead.",
    DeprecationWarning,
    stacklevel=2
)


@dataclass
class ParallelToolCall:
    """A tool call that can be executed in parallel."""
    tool_name: str
    parameters: Dict[str, Any]
    dependencies: Set[str] = field(default_factory=set)
    result: Optional[ToolResult] = None
    execution_time: float = 0.0


@dataclass
class ParallelExecutionResult:
    """Result of parallel execution."""
    success: bool
    results: Dict[str, ToolResult] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)
    total_time: float = 0.0
    errors: List[str] = field(default_factory=list)


class DependencyResolver:
    """Resolves dependencies between tool calls."""
    
    def __init__(self):
        self._tool_outputs = {
            "read_file": {"content", "file_path"},
            "write_file": {"file_path"},
            "edit_file": {"file_path"},
            "grep": {"matches", "results"},
            "glob": {"files"},
            "bash": {"stdout", "stderr", "exit_code"},
            "git_status": {"status", "changed_files"},
            "git_diff": {"diff"},
            "git_commit": {"commit_hash"},
            "git_branch": {"branches"},
            "git_log": {"commits"},
        }
    
    def resolve(self, tool_calls: List[Dict[str, Any]]) -> List[Set[str]]:
        """Resolve dependencies between tool calls.
        
        Args:
            tool_calls: List of tool call dictionaries
        
        Returns:
            List of dependency sets for each tool call
        """
        dependencies = []
        
        for i, call in enumerate(tool_calls):
            deps = set()
            tool_name = call.get("tool_name", "")
            params = call.get("parameters", {})
            
            for key, value in params.items():
                if isinstance(value, str) and "$" in value:
                    for j, prev_call in enumerate(tool_calls[:i]):
                        prev_tool = prev_call.get("tool_name", "")
                        if prev_tool in self._tool_outputs:
                            outputs = self._tool_outputs[prev_tool]
                            for output in outputs:
                                if output in value:
                                    deps.add(j)
            
            dependencies.append(deps)
        
        return dependencies
    
    def get_execution_layers(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> List[List[int]]:
        """Get execution layers (tools in same layer can run in parallel).
        
        Args:
            tool_calls: List of tool call dictionaries
        
        Returns:
            List of layers, each containing tool indices
        """
        dependencies = self.resolve(tool_calls)
        n = len(tool_calls)
        
        in_degree = [len(deps) for deps in dependencies]
        layers = []
        remaining = set(range(n))
        
        while remaining:
            current_layer = []
            
            for i in remaining:
                if in_degree[i] == 0:
                    current_layer.append(i)
            
            if not current_layer:
                break
            
            layers.append(current_layer)
            
            for i in current_layer:
                remaining.remove(i)
                for j in remaining:
                    if i in dependencies[j]:
                        in_degree[j] -= 1
        
        return layers


class ParallelExecutor:
    """Execute tools in parallel with dependency handling.
    
    Features:
    - Parallel execution of independent tools
    - Dependency resolution
    - Result aggregation
    - Timeout handling
    """
    
    def __init__(self, tool_service: Any):
        """Initialize parallel executor.
        
        Args:
            tool_service: AgentToolService for execution
        """
        self.tool_service = tool_service
        self.dependency_resolver = DependencyResolver()
        
        logger.info("[ParallelExecutor] Initialized")
    
    async def execute_parallel(
        self,
        tool_calls: List[Dict[str, Any]],
        timeout: float = 60.0
    ) -> ParallelExecutionResult:
        """Execute tools in parallel.
        
        Args:
            tool_calls: List of tool calls to execute
            timeout: Overall timeout in seconds
        
        Returns:
            ParallelExecutionResult
        """
        import time
        start_time = time.time()
        
        layers = self.dependency_resolver.get_execution_layers(tool_calls)
        
        logger.info(f"[ParallelExecutor] Executing {len(tool_calls)} tools in {len(layers)} layers")
        
        results = {}
        errors = []
        execution_order = []
        
        for layer_idx, layer in enumerate(layers):
            logger.info(f"[ParallelExecutor] Layer {layer_idx + 1}: {len(layer)} tools")
            
            tasks = []
            indices = []
            
            for tool_idx in layer:
                call = tool_calls[tool_idx]
                task = self._execute_tool(
                    call.get("tool_name", ""),
                    call.get("parameters", {})
                )
                tasks.append(task)
                indices.append(tool_idx)
            
            if tasks:
                layer_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for idx, result in zip(indices, layer_results):
                    tool_name = tool_calls[idx].get("tool_name", f"tool_{idx}")
                    execution_order.append(tool_name)
                    
                    if isinstance(result, Exception):
                        results[idx] = ToolResult(success=False, error=str(result))
                        errors.append(f"{tool_name}: {str(result)}")
                    else:
                        results[idx] = result
        
        total_time = time.time() - start_time
        
        success = len(errors) == 0
        
        return ParallelExecutionResult(
            success=success,
            results=results,
            execution_order=execution_order,
            total_time=total_time,
            errors=errors
        )
    
    async def execute_independent(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> Dict[int, ToolResult]:
        """Execute all independent tools in parallel.
        
        Args:
            tool_calls: List of tool calls
        
        Returns:
            Dictionary of results by index
        """
        tasks = []
        
        for call in tool_calls:
            task = self._execute_tool(
                call.get("tool_name", ""),
                call.get("parameters", {})
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        output = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                output[i] = ToolResult(success=False, error=str(result))
            else:
                output[i] = result
        
        return output
    
    async def _execute_tool(self, tool_name: str, params: Dict[str, Any]) -> ToolResult:
        """Execute a single tool.
        
        Args:
            tool_name: Tool name
            params: Parameters
        
        Returns:
            ToolResult
        """
        try:
            result = await self.tool_service.execute_tool(tool_name, params)
            return result
        except Exception as e:
            logger.error(f"[ParallelExecutor] {tool_name} failed: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def execute_batch(
        self,
        batch: List[Dict[str, Any]],
        concurrency: int = 3
    ) -> List[ToolResult]:
        """Execute batch of tools with concurrency limit.
        
        Args:
            batch: List of tool calls
            concurrency: Maximum concurrent executions
        
        Returns:
            List of results
        """
        semaphore = asyncio.Semaphore(concurrency)
        
        async def limited_exec(call):
            async with semaphore:
                return await self._execute_tool(
                    call.get("tool_name", ""),
                    call.get("parameters", {})
                )
        
        tasks = [limited_exec(call) for call in batch]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        output = []
        for result in results:
            if isinstance(result, Exception):
                output.append(ToolResult(success=False, error=str(result)))
            else:
                output.append(result)
        
        return output


class ResultAggregator:
    """Aggregate results from multiple tool executions."""
    
    @staticmethod
    def aggregate_text(results: Dict[str, ToolResult]) -> str:
        """Aggregate results as text.
        
        Args:
            results: Dictionary of results
        
        Returns:
            Aggregated text
        """
        outputs = []
        
        for key, result in results.items():
            if result.success and result.output:
                outputs.append(f"=== {key} ===\n{result.output}")
            elif result.error:
                outputs.append(f"=== {key} (error) ===\n{result.error}")
        
        return "\n\n".join(outputs)
    
    @staticmethod
    def aggregate_json(results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results as JSON.
        
        Args:
            results: Dictionary of results
        
        Returns:
            Aggregated dictionary
        """
        output = {}
        
        for key, result in results.items():
            if hasattr(result, 'success'):
                output[key] = {
                    "success": result.success,
                    "output": result.output,
                    "error": result.error
                }
            else:
                output[key] = result
        
        return output
    
    @staticmethod
    def get_summary(results: Dict[str, ToolResult]) -> Dict[str, Any]:
        """Get summary of results.
        
        Args:
            results: Dictionary of results
        
        Returns:
            Summary dictionary
        """
        total = len(results)
        successes = sum(1 for r in results.values() if r.success)
        failures = total - successes
        
        return {
            "total": total,
            "successes": successes,
            "failures": failures,
            "success_rate": successes / total if total > 0 else 0.0
        }


def create_parallel_executor(tool_service: Any) -> ParallelExecutor:
    """Create a parallel executor.
    
    Args:
        tool_service: Tool service instance
    
    Returns:
        ParallelExecutor
    """
    return ParallelExecutor(tool_service)
