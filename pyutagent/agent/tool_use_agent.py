"""Tool-Enabled Agent - Integration of tools with ReAct Agent.

This module provides:
- ToolUseAgent: Agent that uses tools for task completion
- Integration with AgentToolService
- Feedback loop with LLM
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .tool_service import AgentToolService
from .intelligent_selector import IntelligentToolSelector, create_intelligent_selector
from .prompts import ToolUsagePromptBuilder

logger = logging.getLogger(__name__)


@dataclass
class ToolUseTurn:
    """One turn of tool use."""
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    result: Optional[Any] = None


@dataclass
class ToolUseResult:
    """Result of tool-enabled task execution."""
    success: bool
    final_output: str
    turns: List[ToolUseTurn] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    error: Optional[str] = None


class ToolUseAgent:
    """Agent that uses tools to complete tasks.
    
    Features:
    - Integrates with AgentToolService
    - Intelligent tool selection
    - ReAct-style reasoning loop
    - LLM feedback
    """
    
    def __init__(
        self,
        tool_service: AgentToolService,
        llm_client: Any,
        max_turns: int = 10,
        tool_selector: Optional[IntelligentToolSelector] = None
    ):
        """Initialize tool use agent.
        
        Args:
            tool_service: Tool service for execution
            llm_client: LLM client for reasoning
            max_turns: Maximum number of turns
            tool_selector: Optional intelligent selector
        """
        self.tool_service = tool_service
        self.llm_client = llm_client
        self.max_turns = max_turns
        
        self.tool_selector = tool_selector or create_intelligent_selector()
        self.tool_selector.set_tool_memory(None)
        
        self._turns: List[ToolUseTurn] = []
        
        logger.info("[ToolUseAgent] Initialized")
    
    async def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> ToolUseResult:
        """Run the agent on a task.
        
        Args:
            task: Task to complete
            context: Optional context
        
        Returns:
            ToolUseResult
        """
        context = context or {}
        self._turns = []
        tools_used = []
        
        system_prompt = ToolUsagePromptBuilder.SYSTEM_PROMPT
        schemas = self.tool_service.get_schemas_json()
        
        full_prompt = f"""{system_prompt}

## Tools Available
{schemas}

## Task
{task}

## Instructions
1. Think about what to do
2. Use a tool if needed
3. Observe the result
4. Continue until task is complete

Respond in the following format:
Thought: <your reasoning>
Action: <tool_name> (or "complete" if done)
Action Input: <parameters as JSON>
Observation: <result of action>

Let's begin:
"""
        
        try:
            for turn_num in range(self.max_turns):
                logger.info(f"[ToolUseAgent] Turn {turn_num + 1}/{self.max_turns}")
                
                response = await self.llm_client.chat(
                    messages=[{"role": "user", "content": full_prompt}],
                    temperature=0.7
                )
                
                content = response.content if hasattr(response, 'content') else str(response)
                
                turn = self._parse_turn(content)
                
                if not turn.action or turn.action == "complete":
                    return ToolUseResult(
                        success=True,
                        final_output=turn.thought,
                        turns=self._turns,
                        tools_used=tools_used
                    )
                
                result = await self._execute_tool(turn.action, turn.action_input or {})
                turn.result = result
                turn.observation = result.output if result.success else result.error
                
                self._turns.append(turn)
                
                if turn.action not in tools_used:
                    tools_used.append(turn.action)
                
                full_prompt += f"\n\nThought: {turn.thought}\n"
                full_prompt += f"Action: {turn.action}\n"
                full_prompt += f"Action Input: {json.dumps(turn.action_input or {})}\n"
                full_prompt += f"Observation: {turn.observation[:500]}\n"
                
                if result.success and self._check_completion(task, result):
                    return ToolUseResult(
                        success=True,
                        final_output=f"Task completed. {turn.observation[:200]}",
                        turns=self._turns,
                        tools_used=tools_used
                    )
            
            return ToolUseResult(
                success=False,
                final_output="Max turns reached",
                turns=self._turns,
                tools_used=tools_used,
                error="Max turns reached"
            )
            
        except Exception as e:
            logger.exception(f"[ToolUseAgent] Error: {e}")
            return ToolUseResult(
                success=False,
                final_output="Error occurred",
                turns=self._turns,
                tools_used=tools_used,
                error=str(e)
            )
    
    def _parse_turn(self, content: str) -> ToolUseTurn:
        """Parse LLM response into a turn.
        
        Args:
            content: LLM response content
        
        Returns:
            ToolUseTurn
        """
        turn = ToolUseTurn(thought="")
        
        lines = content.split("\n")
        current_field = None
        value_lines = []
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("Thought:"):
                if current_field and value_lines:
                    self._set_field(turn, current_field, "\n".join(value_lines))
                current_field = "thought"
                value_lines = [line[8:].strip()]
            
            elif line.startswith("Action:"):
                if current_field and value_lines:
                    self._set_field(turn, current_field, "\n".join(value_lines))
                current_field = "action"
                value_lines = [line[7:].strip()]
            
            elif line.startswith("Action Input:") or line.startswith("Action Input :"):
                if current_field and value_lines:
                    self._set_field(turn, current_field, "\n".join(value_lines))
                current_field = "action_input"
                value_lines = [line.split(":", 1)[1].strip()]
            
            elif line.startswith("Observation:"):
                if current_field and value_lines:
                    self._set_field(turn, current_field, "\n".join(value_lines))
                current_field = "observation"
                value_lines = [line[12:].strip()]
            
            elif current_field and line:
                value_lines.append(line)
        
        if current_field and value_lines:
            self._set_field(turn, current_field, "\n".join(value_lines))
        
        return turn
    
    def _set_field(self, turn: ToolUseTurn, field: str, value: str):
        """Set field on turn.
        
        Args:
            turn: Turn to update
            field: Field name
            value: Field value
        """
        if field == "thought":
            turn.thought = value
        elif field == "action":
            turn.action = value.strip()
        elif field == "action_input":
            try:
                turn.action_input = json.loads(value)
            except json.JSONDecodeError:
                turn.action_input = {"raw": value}
        elif field == "observation":
            turn.observation = value
    
    async def _execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Execute a tool.
        
        Args:
            tool_name: Tool to execute
            params: Parameters
        
        Returns:
            Tool result
        """
        logger.info(f"[ToolUseAgent] Executing tool: {tool_name}")
        
        try:
            result = await self.tool_service.execute_tool(tool_name, params)
            return result
        except Exception as e:
            logger.error(f"[ToolUseAgent] Tool execution failed: {e}")
            from ..tools.tool import ToolResult
            return ToolResult(success=False, error=str(e))
    
    def _check_completion(self, task: str, result: Any) -> bool:
        """Check if task is complete.
        
        Args:
            task: Original task
            result: Tool result
        
        Returns:
            True if complete
        """
        if hasattr(result, 'success') and result.success:
            output = result.output if result.output else ""
            
            if len(output) > 10:
                return True
        
        return False
    
    def get_used_tools(self) -> List[str]:
        """Get list of tools used so far.
        
        Returns:
            List of tool names
        """
        return [turn.action for turn in self._turns if turn.action]
    
    def get_turns(self) -> List[Dict[str, Any]]:
        """Get all turns as dictionaries.
        
        Returns:
            List of turn dictionaries
        """
        return [
            {
                "thought": t.thought,
                "action": t.action,
                "action_input": t.action_input,
                "observation": t.observation
            }
            for t in self._turns
        ]


def create_tool_use_agent(
    tool_service: AgentToolService,
    llm_client: Any,
    max_turns: int = 10
) -> ToolUseAgent:
    """Create a tool use agent.
    
    Args:
        tool_service: Tool service
        llm_client: LLM client
        max_turns: Maximum turns
    
    Returns:
        ToolUseAgent instance
    """
    return ToolUseAgent(
        tool_service=tool_service,
        llm_client=llm_client,
        max_turns=max_turns
    )
