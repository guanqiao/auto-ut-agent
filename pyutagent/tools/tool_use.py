"""LLM-driven tool use capability.

This module provides:
- ToolUseAgent: LLM-driven tool selection and execution
- Conversation management for tool calls
- Structured tool result handling
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Awaitable
from enum import Enum, auto
from datetime import datetime

from ..llm.client import LLMClient
from ..core.config import DEFAULT_MAX_ITERATIONS
from .tool import Tool, ToolResult, ToolExecutor
from .tool_registry import ToolRegistry, get_registry

logger = logging.getLogger(__name__)


class ToolUseState(Enum):
    """States in tool use loop."""
    IDLE = auto()
    THINKING = auto()
    EXECUTING_TOOL = auto()
    WAITING_USER = auto()
    DONE = auto()
    ERROR = auto()


@dataclass
class ToolCall:
    """Represents a tool call."""
    id: str
    name: str
    arguments: Dict[str, Any]
    result: Optional[ToolResult] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def duration_ms(self) -> Optional[float]:
        """Get call duration in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "result": self.result.to_dict() if self.result else None,
            "duration_ms": self.duration_ms
        }


@dataclass
class ToolUseTurn:
    """A single turn in tool use conversation."""
    role: str
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class ToolUseAgent:
    """Agent that uses LLM to select and execute tools.

    This is similar to how Claude Code, Cursor, and other top coding
    agents work - the LLM decides which tools to use based on the task.

    Features:
    - Automatic tool selection based on task
    - Structured tool result feedback to LLM
    - Conversation history tracking
    - Tool use loop with max iterations
    - Error handling and recovery
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: Optional[ToolRegistry] = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        max_tool_calls_per_turn: int = 5,
        timeout: int = 60
    ):
        """Initialize ToolUse agent.

        Args:
            llm_client: LLM client for decision making
            tool_registry: Registry of available tools
            max_iterations: Max tool use iterations
            max_tool_calls_per_turn: Max tool calls per LLM turn
            timeout: Tool execution timeout in seconds
        """
        self.llm_client = llm_client
        self.tool_registry = tool_registry or get_registry()
        self.max_iterations = max_iterations
        self.max_tool_calls_per_turn = max_tool_calls_per_turn
        self.timeout = timeout
        self.tool_executor = ToolExecutor()

        self.conversation: List[ToolUseTurn] = []
        self.tool_calls: List[ToolCall] = []
        self.state = ToolUseState.IDLE
        self._system_prompt = self._build_system_prompt()

        logger.info(f"[ToolUseAgent] Initialized with {len(self.tool_registry)} tools")

    def _build_system_prompt(self) -> str:
        """Build system prompt describing available tools."""
        tool_descriptions = []

        for tool in self.tool_registry:
            defn = tool.definition
            params_desc = []

            for param in defn.parameters:
                required = "required" if param.required else "optional"
                params_desc.append(f"  - {param.name} ({param.type}, {required}): {param.description}")

            tool_descriptions.append(f"""
### {defn.name}
{defn.description}

Parameters:
{chr(10).join(params_desc) if params_desc else "  (none)"}
""")

        return f"""You are an AI coding assistant with access to various tools to help complete coding tasks.

## Available Tools

{''.join(tool_descriptions)}

## Guidelines

1. Analyze the user's request and determine what tools to use
2. For each tool call, provide the tool name and arguments
3. After each tool execution, analyze the results and continue as needed
4. When you have completed the task, provide a summary of what was done
5. If something fails, analyze the error and try an alternative approach

## Tool Call Format

Use tools by returning JSON in this format:
```json
{{
  "tool_calls": [
    {{
      "name": "tool_name",
      "arguments": {{
        "param1": "value1",
        "param2": "value2"
      }}
    }}
  ]
}}
```

If you don't need to use any tools, respond directly with your answer.
"""

    def add_user_message(self, content: str):
        """Add a user message to the conversation.

        Args:
            content: User message content
        """
        turn = ToolUseTurn(role="user", content=content)
        self.conversation.append(turn)
        logger.debug(f"[ToolUseAgent] Added user message: {len(content)} chars")

    def add_assistant_message(self, content: str, tool_calls: Optional[List[ToolCall]] = None):
        """Add an assistant message to the conversation.

        Args:
            content: Assistant message content
            tool_calls: Optional list of tool calls made
        """
        turn = ToolUseTurn(
            role="assistant",
            content=content,
            tool_calls=tool_calls or []
        )
        self.conversation.append(turn)

    def add_tool_result_message(self, tool_call: ToolCall):
        """Add a tool result as a message.

        Args:
            tool_call: The tool call with results
        """
        result = tool_call.result
        if result:
            if result.success:
                output = result.output
                if isinstance(output, (list, dict)):
                    output_str = json.dumps(output, indent=2, ensure_ascii=False)
                else:
                    output_str = str(output)

                content = f"Tool '{tool_call.name}' result:\n\n{output_str}"
            else:
                content = f"Tool '{tool_call.name}' error:\n\n{result.error}"

            turn = ToolUseTurn(role="tool", content=content)
            self.conversation.append(turn)

    async def run(self, user_message: str) -> Dict[str, Any]:
        """Run the tool use loop.

        Args:
            user_message: Initial user message

        Returns:
            Dictionary with results and conversation
        """
        self.add_user_message(user_message)

        for iteration in range(self.max_iterations):
            logger.info(f"[ToolUseAgent] Iteration {iteration + 1}/{self.max_iterations}")

            self.state = ToolUseState.THINKING

            llm_response = await self._get_llm_response()

            if not llm_response:
                logger.warning("[ToolUseAgent] Empty LLM response")
                break

            tool_calls = self._parse_tool_calls(llm_response)

            if not tool_calls:
                self.add_assistant_message(llm_response)
                self.state = ToolUseState.DONE
                logger.info("[ToolUseAgent] No more tool calls, task complete")
                break

            self.add_assistant_message(llm_response, tool_calls)

            self.state = ToolUseState.EXECUTING_TOOL

            for tool_call in tool_calls[:self.max_tool_calls_per_turn]:
                await self._execute_tool_call(tool_call)
                self.add_tool_result_message(tool_call)

            if len(tool_calls) > self.max_tool_calls_per_turn:
                logger.warning(
                    f"[ToolUseAgent] Truncated tool calls: "
                    f"{len(tool_calls)} -> {self.max_tool_calls_per_turn}"
                )

        final_response = self._get_final_response()

        return {
            "success": self.state == ToolUseState.DONE,
            "response": final_response,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "iterations": len([t for t in self.conversation if t.role == "assistant"]),
            "state": self.state.name
        }

    async def _get_llm_response(self) -> str:
        """Get response from LLM with tool schemas."""
        schemas = self.tool_registry.get_schemas()

        messages = []

        messages.append({"role": "system", "content": self._system_prompt})

        for turn in self.conversation:
            if turn.role == "tool":
                for tc in turn.tool_calls:
                    if tc.result:
                        if tc.result.success:
                            messages.append({
                                "role": "tool",
                                "name": tc.name,
                                "content": str(tc.result.output) if tc.result.output else ""
                            })
                        else:
                            messages.append({
                                "role": "tool",
                                "name": tc.name,
                                "content": f"ERROR: {tc.result.error}"
                            })
            else:
                msg_content = turn.content
                if turn.tool_calls:
                    tc_list = []
                    for tc in turn.tool_calls:
                        tc_list.append({
                            "name": tc.name,
                            "arguments": tc.arguments
                        })
                    msg_content = json.dumps({"tool_calls": tc_list})

                messages.append({
                    "role": turn.role,
                    "content": msg_content
                })

        try:
            response = await self.llm_client.complete(
                messages=messages,
                tools=schemas if schemas else None
            )
            return response
        except TypeError:
            try:
                response = await self.llm_client.complete(messages=messages)
                return response
            except Exception as e:
                logger.exception(f"[ToolUseAgent] LLM call failed: {e}")
                return ""
        except Exception as e:
            logger.exception(f"[ToolUseAgent] LLM call failed: {e}")
            return ""

    def _parse_tool_calls(self, response: str) -> List[ToolCall]:
        """Parse tool calls from LLM response."""
        tool_calls = []

        try:
            import re
            json_match = re.search(r'\{[\s\S]*"tool_calls"[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                for tc_data in data.get("tool_calls", []):
                    tool_calls.append(ToolCall(
                        id=f"call_{len(self.tool_calls)}_{tc_data.get('name', 'unknown')}",
                        name=tc_data.get("name", ""),
                        arguments=tc_data.get("arguments", {})
                    ))

        except json.JSONDecodeError as e:
            logger.warning(f"[ToolUseAgent] Failed to parse tool calls: {e}")

        if not tool_calls:
            json_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)```', response)
            for block in json_blocks:
                try:
                    data = json.loads(block.strip())
                    if "tool_calls" in data:
                        for tc_data in data["tool_calls"]:
                            tool_calls.append(ToolCall(
                                id=f"call_{len(self.tool_calls)}_{tc_data.get('name', 'unknown')}",
                                name=tc_data.get("name", ""),
                                arguments=tc_data.get("arguments", {})
                            ))
                except json.JSONDecodeError:
                    continue

        logger.debug(f"[ToolUseAgent] Parsed {len(tool_calls)} tool calls")
        return tool_calls

    async def _execute_tool_call(self, tool_call: ToolCall):
        """Execute a single tool call."""
        tool_call.start_time = datetime.now()
        logger.info(f"[ToolUseAgent] Executing tool: {tool_call.name}")

        try:
            tool = self.tool_registry.get(tool_call.name)
            result = await self.tool_executor.execute_tool(tool, tool_call.arguments)
            tool_call.result = result

            if result.success:
                logger.info(f"[ToolUseAgent] Tool {tool_call.name} succeeded")
            else:
                logger.warning(f"[ToolUseAgent] Tool {tool_call.name} failed: {result.error}")

        except Exception as e:
            logger.exception(f"[ToolUseAgent] Tool execution error: {e}")
            tool_call.result = ToolResult(success=False, error=str(e))

        tool_call.end_time = datetime.now()
        self.tool_calls.append(tool_call)

    def _get_final_response(self) -> str:
        """Get the final assistant response."""
        for turn in reversed(self.conversation):
            if turn.role == "assistant" and turn.content:
                return turn.content
        return ""

    def reset(self):
        """Reset the agent state."""
        self.conversation.clear()
        self.tool_calls.clear()
        self.state = ToolUseState.IDLE
        logger.debug("[ToolUseAgent] Reset state")

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        total_calls = len(self.tool_calls)
        successful_calls = sum(1 for tc in self.tool_calls if tc.result and tc.result.success)

        return {
            "total_tool_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": total_calls - successful_calls,
            "success_rate": successful_calls / total_calls if total_calls > 0 else 0,
            "iterations": len([t for t in self.conversation if t.role == "assistant"]),
            "state": self.state.name
        }


def create_tool_use_agent(
    llm_client: LLMClient,
    project_path: Optional[str] = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS
) -> ToolUseAgent:
    """Create a configured ToolUse agent.

    Args:
        llm_client: LLM client
        project_path: Project path for tools
        max_iterations: Max iterations

    Returns:
        Configured ToolUseAgent
    """
    from .standard_tools import ReadTool, WriteTool, EditTool, GlobTool, GrepTool, BashTool

    registry = get_registry()
    registry.clear()

    if project_path:
        registry.register(ReadTool(base_path=project_path))
        registry.register(WriteTool(base_path=project_path))
        registry.register(EditTool(base_path=project_path))
        registry.register(GlobTool(base_path=project_path))
        registry.register(GrepTool(base_path=project_path))
        registry.register(BashTool(base_path=project_path))
    else:
        registry.register(ReadTool())
        registry.register(WriteTool())
        registry.register(EditTool())
        registry.register(GlobTool())
        registry.register(GrepTool())
        registry.register(BashTool())

    return ToolUseAgent(
        llm_client=llm_client,
        tool_registry=registry,
        max_iterations=max_iterations
    )
