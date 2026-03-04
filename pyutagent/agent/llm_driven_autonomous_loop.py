"""LLM-Driven Enhanced Autonomous Loop.

This module provides truly autonomous decision-making capabilities:
- LLM-powered action selection (not preset flow)
- Dynamic tool selection based on context
- Goal-driven decision making
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

from ..llm.client import LLMClient
from ..tools.tool_registry import ToolRegistry
from ..tools.tool import ToolResult

logger = logging.getLogger(__name__)


class DecisionStrategy(Enum):
    """Strategy for making decisions."""
    GOAL_ORIENTED = "goal_oriented"
    EXPLORATION = "exploration"
    CONSERVATIVE = "conservative"


@dataclass
class DecisionContext:
    """Context for decision making."""
    task_goal: str
    current_state: str
    recent_actions: List[Dict[str, Any]] = field(default_factory=list)
    available_tools: List[str] = field(default_factory=list)
    last_result: Optional[Dict[str, Any]] = None
    iteration: int = 0
    max_iterations: int = 50


@dataclass
class LLMDecision:
    """Decision made by LLM."""
    reasoning: str
    action: str
    parameters: Dict[str, Any]
    expected_outcome: str
    confidence: float
    alternatives: List[Dict[str, Any]] = field(default_factory=list)


class LLMActionDecider:
    """LLM-powered action decider for autonomous decision making."""

    SYSTEM_PROMPT = """You are an autonomous coding agent that decides what action to take next.

You have access to various tools. Your job is to analyze the current state and decide the next action.

Available tools: {available_tools}

Current state: {current_state}
Task goal: {task_goal}
Recent actions: {recent_actions}
Last result: {last_result}
Iteration: {iteration}/{max_iterations}

Decide what to do next. Consider:
1. What is the current state of the task?
2. What information do you still need?
3. Which tool would give you the most useful information?
4. Have you tried similar approaches before?

Respond in JSON format:
{{
    "reasoning": "Why you chose this action",
    "action": "tool_name",
    "parameters": {{"param1": "value1"}},
    "expected_outcome": "What you expect to happen",
    "confidence": 0.0-1.0,
    "alternatives": [{{"action": "alt_tool", "reasoning": "why alternative"}}]
}}
"""

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        strategy: DecisionStrategy = DecisionStrategy.GOAL_ORIENTED
    ):
        """Initialize LLM action decider.

        Args:
            llm_client: LLM client for decision making
            tool_registry: Tool registry for available tools
            strategy: Decision strategy
        """
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.strategy = strategy

    async def decide(self, context: DecisionContext) -> LLMDecision:
        """Decide the next action based on context.

        Args:
            context: Decision context

        Returns:
            LLMDecision with action and reasoning
        """
        available_tools = await self._get_available_tools()
        context.available_tools = available_tools

        prompt = self._build_prompt(context)

        try:
            response = await self.llm_client.agenerate(
                prompt=prompt,
                temperature=self._get_temperature(),
                max_tokens=1024
            )

            decision = self._parse_response(response, context)
            logger.info(f"[LLMActionDecider] Decided: {decision.action} (confidence: {decision.confidence})")
            return decision

        except Exception as e:
            logger.error(f"[LLMActionDecider] Decision failed: {e}")
            return self._fallback_decision(context)

    async def _get_available_tools(self) -> List[str]:
        """Get list of available tools."""
        tools = self.tool_registry.list_tools()
        return [tool.name for tool in tools]

    def _build_prompt(self, context: DecisionContext) -> str:
        """Build prompt for LLM."""
        available_tools_str = ", ".join(context.available_tools)

        recent_actions_str = "None"
        if context.recent_actions:
            actions = [f"{a.get('action', 'unknown')}: {a.get('result', 'N/A')[:100]}"
                      for a in context.recent_actions[-3:]]
            recent_actions_str = "\n".join(actions)

        last_result_str = "None"
        if context.last_result:
            last_result_str = str(context.last_result)[:200]

        return self.SYSTEM_PROMPT.format(
            available_tools=available_tools_str,
            current_state=context.current_state,
            task_goal=context.task_goal,
            recent_actions=recent_actions_str,
            last_result=last_result_str,
            iteration=context.iteration,
            max_iterations=context.max_iterations
        )

    def _parse_response(self, response: str, context: DecisionContext) -> LLMDecision:
        """Parse LLM response into decision."""
        import json

        try:
            data = json.loads(response.strip())
            return LLMDecision(
                reasoning=data.get("reasoning", "No reasoning provided"),
                action=data.get("action", "bash"),
                parameters=data.get("parameters", {}),
                expected_outcome=data.get("expected_outcome", ""),
                confidence=data.get("confidence", 0.5),
                alternatives=data.get("alternatives", [])
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"[LLMActionDecider] Failed to parse response: {e}")
            return self._fallback_decision(context)

    def _fallback_decision(self, context: DecisionContext) -> LLMDecision:
        """Fallback decision when LLM fails."""
        if context.recent_actions:
            return LLMDecision(
                reasoning="Fallback: continuing with task",
                action="bash",
                parameters={"command": f"echo 'Working on: {context.task_goal}'"},
                expected_outcome="Continue task",
                confidence=0.3,
                alternatives=[]
            )

        return LLMDecision(
            reasoning="Fallback: starting with exploration",
            action="git_status",
            parameters={},
            expected_outcome="Get repository status",
            confidence=0.3,
            alternatives=[]
        )

    def _get_temperature(self) -> float:
        """Get temperature based on strategy."""
        return {
            DecisionStrategy.GOAL_ORIENTED: 0.3,
            DecisionStrategy.EXPLORATION: 0.7,
            DecisionStrategy.CONSERVATIVE: 0.1
        }.get(self.strategy, 0.5)


class DynamicToolSelector:
    """Dynamic tool selector based on task context."""

    def __init__(self, tool_registry: ToolRegistry):
        """Initialize dynamic tool selector.

        Args:
            tool_registry: Tool registry
        """
        self.tool_registry = tool_registry

    async def select_tools(
        self,
        task_type: str,
        context: Dict[str, Any]
    ) -> List[str]:
        """Select appropriate tools for the task.

        Args:
            task_type: Type of task
            context: Task context

        Returns:
            List of tool names to use
        """
        tools = self.tool_registry.list_tools()
        tool_map = {tool.name: tool for tool in tools}

        if task_type == "ut_generation":
            return self._select_ut_tools(tool_map)
        elif task_type == "code_refactoring":
            return self._select_refactor_tools(tool_map)
        elif task_type == "bug_fix":
            return self._select_debug_tools(tool_map)
        else:
            return self._select_general_tools(tool_map)

    def _select_ut_tools(self, tool_map: Dict[str, Any]) -> List[str]:
        """Select tools for UT generation."""
        return ["read_file", "glob", "grep", "bash", "git_status"]

    def _select_refactor_tools(self, tool_map: Dict[str, Any]) -> List[str]:
        """Select tools for code refactoring."""
        return ["read_file", "grep", "edit_tool", "bash", "git_diff"]

    def _select_debug_tools(self, tool_map: Dict[str, Any]) -> List[str]:
        """Select tools for bug fixing."""
        return ["read_file", "grep", "bash", "git_log", "git_diff"]

    def _select_general_tools(self, tool_map: Dict[str, Any]) -> List[str]:
        """Select general tools."""
        return list(tool_map.keys())


class LLMDrivenAutonomousLoop:
    """LLM-driven autonomous loop with true decision making.

    Unlike the preset flow in AutonomousLoop, this loop uses LLM
    to decide what to do next based on the current state.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        max_iterations: int = 50,
        confidence_threshold: float = 0.85,
        strategy: DecisionStrategy = DecisionStrategy.GOAL_ORIENTED
    ):
        """Initialize LLM-driven autonomous loop.

        Args:
            llm_client: LLM client for decision making
            tool_registry: Tool registry for available tools
            max_iterations: Maximum iterations before stopping
            confidence_threshold: Confidence threshold to consider task complete
            strategy: Decision strategy
        """
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self.strategy = strategy

        self._decider = LLMActionDecider(llm_client, tool_registry, strategy)
        self._tool_selector = DynamicToolSelector(tool_registry)

        self._iteration = 0
        self._history: List[Dict[str, Any]] = []

    async def run(
        self,
        task_goal: str,
        initial_context: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """Run the autonomous loop.

        Args:
            task_goal: Goal to accomplish
            initial_context: Initial context
            progress_callback: Callback for progress updates

        Returns:
            Execution result with history
        """
        initial_context = initial_context or {}
        self._iteration = 0
        self._history = []

        context = DecisionContext(
            task_goal=task_goal,
            current_state="initial",
            iteration=0,
            max_iterations=self.max_iterations
        )

        logger.info(f"[LLMDrivenAutonomousLoop] Starting: {task_goal}")

        while self._iteration < self.max_iterations:
            self._iteration += 1
            context.iteration = self._iteration

            if progress_callback:
                progress_callback({
                    "iteration": self._iteration,
                    "max_iterations": self.max_iterations,
                    "task": task_goal
                })

            decision = await self._decider.decide(context)

            if decision.confidence >= self.confidence_threshold:
                logger.info(f"[LLMDrivenAutonomousLoop] Task complete with confidence {decision.confidence}")
                return {
                    "success": True,
                    "iterations": self._iteration,
                    "history": self._history,
                    "final_state": "completed"
                }

            result = await self._execute_action(decision.action, decision.parameters)

            self._history.append({
                "iteration": self._iteration,
                "decision": {
                    "action": decision.action,
                    "reasoning": decision.reasoning,
                    "confidence": decision.confidence
                },
                "result": result.to_dict() if hasattr(result, 'to_dict') else str(result)
            })

            context.recent_actions = self._history[-3:]
            context.last_result = result.to_dict() if hasattr(result, 'to_dict') else str(result)
            context.current_state = self._analyze_state(result)

            if self._is_task_complete(task_goal, result):
                logger.info(f"[LLMDrivenAutonomousLoop] Task completed at iteration {self._iteration}")
                return {
                    "success": True,
                    "iterations": self._iteration,
                    "history": self._history,
                    "final_state": "completed"
                }

        logger.warning(f"[LLMDrivenAutonomousLoop] Max iterations reached")
        return {
            "success": False,
            "iterations": self._iteration,
            "history": self._history,
            "final_state": "max_iterations"
        }

    async def _execute_action(self, action: str, parameters: Dict[str, Any]) -> ToolResult:
        """Execute an action.

        Args:
            action: Action name
            parameters: Action parameters

        Returns:
            Execution result
        """
        try:
            tool = self.tool_registry.get_tool(action)
            if tool:
                return await tool.execute(**parameters)
            else:
                return ToolResult(success=False, error=f"Tool not found: {action}")
        except Exception as e:
            logger.error(f"[LLMDrivenAutonomousLoop] Action failed: {e}")
            return ToolResult(success=False, error=str(e))

    def _analyze_state(self, result: ToolResult) -> str:
        """Analyze current state from result."""
        if hasattr(result, 'success'):
            if result.success:
                return "action_succeeded"
            else:
                return "action_failed"
        return "unknown"

    def _is_task_complete(self, task_goal: str, result: ToolResult) -> bool:
        """Check if task is complete."""
        if hasattr(result, 'success'):
            return result.success
        return False


def create_llm_driven_loop(
    llm_client: LLMClient,
    tool_registry: ToolRegistry,
    max_iterations: int = 50,
    strategy: DecisionStrategy = DecisionStrategy.GOAL_ORIENTED
) -> LLMDrivenAutonomousLoop:
    """Create LLM-driven autonomous loop.

    Args:
        llm_client: LLM client
        tool_registry: Tool registry
        max_iterations: Max iterations
        strategy: Decision strategy

    Returns:
        LLMDrivenAutonomousLoop instance
    """
    return LLMDrivenAutonomousLoop(
        llm_client=llm_client,
        tool_registry=tool_registry,
        max_iterations=max_iterations,
        strategy=strategy
    )
