"""Tool Integration Mixin - Activates tool usage in agents.

This module provides:
- ToolIntegrationMixin: Mixin to add tool capabilities to agents
- Integrated with AgentToolService
- Enables ReAct-style tool execution
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..tools.tool import ToolResult

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Represents a tool call."""
    tool_name: str
    parameters: Dict[str, Any]
    result: Optional[ToolResult] = None
    error: Optional[str] = None


@dataclass
class ToolExecutionResult:
    """Result of tool execution."""
    success: bool
    calls: List[ToolCall] = field(default_factory=list)
    final_output: str = ""
    iterations: int = 0


class ToolIntegrationMixin:
    """Mixin to integrate tool usage into agents.
    
    Add this mixin to your agent class to enable tool usage.
    
    Usage:
        class MyAgent(ToolIntegrationMixin, BaseAgent):
            def __init__(self, ...):
                super().__init__(...)
                self._init_tool_integration(tool_service, llm_client)
    """
    
    _tool_service: Any = None
    _llm_client: Any = None
    _tool_selector: Any = None
    _tool_history: List[ToolCall] = []
    _max_tool_iterations: int = 5
    
    def _init_tool_integration(
        self,
        tool_service: Any,
        llm_client: Any,
        tool_selector: Optional[Any] = None,
        max_iterations: int = 5
    ):
        """Initialize tool integration.
        
        Args:
            tool_service: AgentToolService instance
            llm_client: LLM client for reasoning
            tool_selector: Optional intelligent tool selector
            max_iterations: Maximum tool call iterations
        """
        self._tool_service = tool_service
        self._llm_client = llm_client
        self._tool_selector = tool_selector
        self._max_tool_iterations = max_iterations
        self._tool_history = []
        
        logger.info("[ToolIntegrationMixin] Initialized")
    
    async def execute_with_tools(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        available_tools: Optional[List[str]] = None
    ) -> ToolExecutionResult:
        """Execute task using tools.
        
        Args:
            task: Task to execute
            context: Additional context
            available_tools: Optional list of tool names to use
        
        Returns:
            ToolExecutionResult
        """
        if not self._tool_service or not self._llm_client:
            return ToolExecutionResult(
                success=False,
                final_output="Tool service or LLM client not initialized"
            )
        
        context = context or {}
        self._tool_history = []
        
        tool_schemas = self._tool_service.get_schemas_json()
        
        prompt = self._build_tool_prompt(task, context, tool_schemas)
        
        for iteration in range(self._max_tool_iterations):
            logger.info(f"[ToolIntegrationMixin] Iteration {iteration + 1}")
            
            response = await self._llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            content = response.content if hasattr(response, 'content') else str(response)
            
            tool_call = self._parse_tool_call(content)
            
            if not tool_call:
                return ToolExecutionResult(
                    success=True,
                    calls=self._tool_history,
                    final_output=content,
                    iterations=iteration + 1
                )
            
            result = await self._execute_tool(tool_call.tool_name, tool_call.parameters)
            tool_call.result = result
            
            self._tool_history.append(tool_call)
            
            prompt += f"\n\nObservation: {result.output[:500] if result.success else result.error}"
            
            if self._check_completion(result):
                return ToolExecutionResult(
                    success=True,
                    calls=self._tool_history,
                    final_output=result.output if result.success else "Task completed with warnings",
                    iterations=iteration + 1
                )
        
        return ToolExecutionResult(
            success=False,
            calls=self._tool_history,
            final_output="Max iterations reached",
            iterations=self._max_tool_iterations
        )
    
    def _build_tool_prompt(
        self,
        task: str,
        context: Dict[str, Any],
        tool_schemas: str
    ) -> str:
        """Build prompt for tool execution.
        
        Args:
            task: Task description
            context: Context dictionary
            tool_schemas: Tool schemas JSON
        
        Returns:
            Formatted prompt
        """
        context_str = "\n".join(f"- {k}: {v}" for k, v in context.items())
        
        return f"""You are an agent that can use tools to complete tasks.

## Task
{task}

## Context
{context_str}

## Available Tools
{tool_schemas}

## Instructions
1. Think about the next step
2. If you need to use a tool, respond with:
   Action: <tool_name>
   Action Input: <JSON parameters>
3. If task is complete, respond with:
   Action: complete
   Action Input: {{"result": "what you accomplished"}}

## Format
Thought: <your reasoning>
Action: <tool_name or complete>
Action Input: <JSON parameters or result summary>"""
    
    def _parse_tool_call(self, content: str) -> Optional[ToolCall]:
        """Parse tool call from LLM response.
        
        Args:
            content: LLM response content
        
        Returns:
            ToolCall or None
        """
        import json
        import re
        
        lines = content.split("\n")
        
        tool_name = None
        params = {}
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("Action:"):
                action = line[7:].strip()
                if action.lower() == "complete":
                    return None
                tool_name = action
            
            elif line.startswith("Action Input:"):
                try:
                    json_str = line[13:].strip()
                    params = json.loads(json_str)
                except json.JSONDecodeError:
                    match = re.search(r'\{.*\}', line)
                    if match:
                        try:
                            params = json.loads(match.group())
                        except:
                            params = {"raw": line}
        
        if tool_name:
            return ToolCall(tool_name=tool_name, parameters=params)
        
        return None
    
    async def _execute_tool(self, tool_name: str, params: Dict[str, Any]) -> ToolResult:
        """Execute a tool.
        
        Args:
            tool_name: Tool to execute
            params: Parameters
        
        Returns:
            ToolResult
        """
        try:
            result = await self._tool_service.execute_tool(tool_name, params)
            logger.info(f"[ToolIntegrationMixin] Executed {tool_name}: {result.success}")
            return result
        except Exception as e:
            logger.error(f"[ToolIntegrationMixin] Tool {tool_name} failed: {e}")
            return ToolResult(success=False, error=str(e))
    
    def _check_completion(self, result: ToolResult) -> bool:
        """Check if task is complete.
        
        Args:
            result: Tool result
        
        Returns:
            True if complete
        """
        if result.success and result.output:
            output = str(result.output)
            if len(output) > 10 and "error" not in output.lower():
                return True
        return False
    
    async def execute_single_tool(
        self,
        tool_name: str,
        params: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Execute a single tool.
        
        Args:
            tool_name: Tool name
            params: Tool parameters
        
        Returns:
            ToolResult
        """
        params = params or {}
        return await self._execute_tool(tool_name, params)
    
    def get_tool_history(self) -> List[Dict[str, Any]]:
        """Get tool execution history.
        
        Returns:
            List of tool calls as dictionaries
        """
        return [
            {
                "tool": call.tool_name,
                "params": call.parameters,
                "success": call.result.success if call.result else False,
                "output": call.result.output if call.result else None,
                "error": call.error
            }
            for call in self._tool_history
        ]
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools.
        
        Returns:
            Tool names
        """
        if self._tool_service:
            return self._tool_service.list_available_tools()
        return []
    
    async def suggest_tool(self, task: str) -> Optional[Dict[str, Any]]:
        """Suggest best tool for task.
        
        Args:
            task: Task description
        
        Returns:
            Dict with tool_name and parameters
        """
        if self._tool_selector:
            tools = self.get_available_tools()
            selections = self._tool_selector.select_tools(task, tools, limit=1)
            if selections:
                return {
                    "tool_name": selections[0].tool_name,
                    "reasoning": selections[0].reasoning
                }
        return None


def mixin_tool_integration(
    agent_class,
    tool_service: Any,
    llm_client: Any,
    tool_selector: Optional[Any] = None
):
    """Mixin helper to add tool integration to an existing agent class.
    
    Args:
        agent_class: Agent class to extend
        tool_service: AgentToolService instance
        llm_client: LLM client
        tool_selector: Optional tool selector
    
    Returns:
        New class with tool integration
    """
    class MixedClass(ToolIntegrationMixin, agent_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._init_tool_integration(tool_service, llm_client, tool_selector)
    
    return MixedClass
