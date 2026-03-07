"""Tool-Enabled ReAct Agent - Activates tool usage in agent feedback loop.

This module provides:
- ToolEnabledReActAgent: Agent with integrated tool usage
- Tool execution in feedback loop
- Context-aware tool selection
"""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .react_agent import ReActAgent
from .tool_integration import ToolIntegrationMixin
from .intelligent_selector import IntelligentToolSelector
from ..core.config import DEFAULT_MAX_ITERATIONS

logger = logging.getLogger(__name__)


@dataclass
class ToolExecutionStep:
    """One step of tool execution."""
    iteration: int
    tool_name: str
    parameters: Dict[str, Any]
    result: Any
    success: bool
    error: Optional[str] = None


class ToolEnabledReActAgent(ReActAgent, ToolIntegrationMixin):
    """ReAct Agent with integrated tool usage.
    
    This agent uses tools within the feedback loop to:
    - Read source files
    - Search code
    - Execute commands
    - Analyze errors
    
    Usage:
        agent = ToolEnabledReActAgent(
            project_path=".",
            llm_client=llm_client,
            tool_service=tool_service
        )
        result = await agent.run("Generate tests for src/Main.java")
    """
    
    def __init__(
        self,
        project_path: str,
        llm_client: Any,
        tool_service: Any,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        tool_selector: Optional[IntelligentToolSelector] = None,
        **kwargs
    ):
        """Initialize tool-enabled agent.
        
        Args:
            project_path: Project path
            llm_client: LLM client
            tool_service: Tool service
            max_iterations: Max feedback iterations
            tool_selector: Optional intelligent selector
        """
        self._project_path = Path(project_path)
        self._llm_client = llm_client
        self._tool_service = tool_service
        
        super().__init__(
            project_path=str(project_path),
            llm_client=llm_client,
            **kwargs
        )
        
        ToolIntegrationMixin._init_tool_integration(
            self,
            tool_service=tool_service,
            llm_client=llm_client,
            tool_selector=tool_selector,
            max_iterations=max_iterations
        )
        
        self._tool_execution_log: List[ToolExecutionStep] = []
        self._use_tools_in_loop = True
        
        logger.info("[ToolEnabledReActAgent] Initialized")
    
    async def run_with_tools(self, task: str) -> Dict[str, Any]:
        """Run task with tool integration.
        
        Args:
            task: Task description
        
        Returns:
            Result dictionary with tool execution log
        """
        self._tool_execution_log = []
        
        context = {
            "task": task,
            "project_path": str(self._project_path)
        }
        
        result = await self.execute_with_tools(
            task=task,
            context=context,
            available_tools=None
        )
        
        return {
            "success": result.success,
            "output": result.final_output,
            "tool_calls": len(result.calls),
            "iterations": result.iterations,
            "execution_log": [
                {
                    "iteration": step.iteration,
                    "tool": step.tool_name,
                    "params": step.parameters,
                    "success": step.success,
                    "error": step.error
                }
                for step in self._tool_execution_log
            ]
        }
    
    async def _execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Execute tool with error handling.
        
        Args:
            tool_name: Tool to execute
            params: Parameters
        
        Returns:
            Tool result
        """
        try:
            result = await self._tool_service.execute_tool(tool_name, params)
            
            self._tool_execution_log.append(ToolExecutionStep(
                iteration=len(self._tool_execution_log) + 1,
                tool_name=tool_name,
                parameters=params,
                result=result,
                success=result.success,
                error=result.error
            ))
            
            return result
            
        except Exception as e:
            logger.error(f"[ToolEnabledReActAgent] Tool {tool_name} failed: {e}")
            
            self._tool_execution_log.append(ToolExecutionStep(
                iteration=len(self._tool_execution_log) + 1,
                tool_name=tool_name,
                parameters=params,
                result=None,
                success=False,
                error=str(e)
            ))
            
            from ..tools.tool import ToolResult
            return ToolResult(success=False, error=str(e))
    
    async def analyze_and_fix(self, error: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze error and try to fix it using tools.
        
        Args:
            error: Error message
            context: Context dictionary
        
        Returns:
            Fix result
        """
        logger.info(f"[ToolEnabledReActAgent] Analyzing error: {error[:100]}")
        
        suggestions = []
        
        search_result = await self.execute_single_tool("grep", {
            "pattern": error.split()[0] if error else "",
            "path": str(self._project_path)
        })
        
        if search_result.success:
            suggestions.append(f"Found {search_result.output[:200]}")
        
        git_status = await self.execute_single_tool("git_status", {})
        if git_status.success:
            suggestions.append(f"Git status: {git_status.output[:200]}")
        
        return {
            "error": error,
            "suggestions": suggestions,
            "tools_used": [s.tool_name for s in self._tool_execution_log]
        }
    
    async def read_source_for_task(self, task: str) -> str:
        """Read relevant source files for a task.
        
        Args:
            task: Task description
        
        Returns:
            Combined source content
        """
        sources = []
        
        if "java" in task.lower():
            java_files = list(self._project_path.rglob("*.java"))[:5]
            for f in java_files:
                try:
                    content = f.read_text(encoding="utf-8")[:1000]
                    sources.append(f"// {f.name}\n{content}")
                except:
                    pass
        
        if not sources:
            sources.append("No source files found")
        
        return "\n\n".join(sources)
    
    def get_tool_execution_log(self) -> List[Dict[str, Any]]:
        """Get tool execution log.
        
        Returns:
            List of execution steps
        """
        return [
            {
                "iteration": step.iteration,
                "tool": step.tool_name,
                "params": step.parameters,
                "success": step.success,
                "error": step.error
            }
            for step in self._tool_execution_log
        ]
    
    def enable_tools(self):
        """Enable tool usage in agent."""
        self._use_tools_in_loop = True
        logger.info("[ToolEnabledReActAgent] Tools enabled")
    
    def disable_tools(self):
        """Disable tool usage in agent."""
        self._use_tools_in_loop = False
        logger.info("[ToolEnabledReActAgent] Tools disabled")
    
    @property
    def tool_service(self):
        """Get tool service."""
        return self._tool_service
    
    @property
    def available_tools(self) -> List[str]:
        """Get available tools."""
        if self._tool_service:
            return self._tool_service.list_available_tools()
        return []


def create_tool_enabled_agent(
    project_path: str,
    llm_client: Any,
    tool_service: Any,
    **kwargs
) -> ToolEnabledReActAgent:
    """Create a tool-enabled ReAct agent.
    
    Args:
        project_path: Project path
        llm_client: LLM client
        tool_service: Tool service
        **kwargs: Additional arguments
    
    Returns:
        ToolEnabledReActAgent instance
    """
    return ToolEnabledReActAgent(
        project_path=project_path,
        llm_client=llm_client,
        tool_service=tool_service,
        **kwargs
    )
