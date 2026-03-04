"""Enhanced Tool Orchestrator with autonomous tool selection.

This module extends the base ToolOrchestrator with:
- LLM-based tool selection
- Dynamic tool chain planning
- Tool execution result reasoning
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

from .tool_orchestrator import (
    ToolOrchestrator,
    ExecutionPlan,
    ToolCall,
    ToolState,
    PlanState,
    OrchestrationResult,
)
from ..llm.client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class ToolSelection:
    """Represents a selected tool with reasoning."""
    tool_name: str
    parameters: Dict[str, Any]
    reasoning: str
    confidence: float = 1.0


class EnhancedToolOrchestrator(ToolOrchestrator):
    """Enhanced tool orchestrator with LLM-based tool selection.
    
    Features:
    - Uses LLM to analyze goals and select appropriate tools
    - Generates dynamic tool execution plans
    - Learns from execution results
    """
    
    def __init__(
        self,
        tools: Optional[Dict[str, Callable]] = None,
        tool_definitions: Optional[Dict[str, Any]] = None,
        max_parallel: int = 3,
        adaptation_enabled: bool = True,
        llm_client: Optional[LLMClient] = None
    ):
        """Initialize enhanced tool orchestrator.
        
        Args:
            tools: Dictionary of tool functions
            tool_definitions: Tool definitions
            max_parallel: Maximum parallel executions
            adaptation_enabled: Enable plan adaptation
            llm_client: LLM client for tool selection
        """
        super().__init__(tools, tool_definitions, max_parallel, adaptation_enabled)
        self._llm_client = llm_client
    
    def set_llm_client(self, llm_client: LLMClient):
        """Set the LLM client for tool selection.
        
        Args:
            llm_client: LLM client instance
        """
        self._llm_client = llm_client
    
    async def plan_from_goal(
        self,
        goal: str,
        context: Dict[str, Any],
        available_tools: Optional[List[str]] = None
    ) -> ExecutionPlan:
        """Create execution plan from natural language goal using LLM.
        
        Args:
            goal: The goal to achieve (natural language)
            context: Current context and available data
            available_tools: Optional list of tool names to consider
        
        Returns:
            ExecutionPlan with planned tool calls
        """
        import uuid
        
        if not self._llm_client:
            logger.warning("[EnhancedToolOrchestrator] No LLM client, using fallback planning")
            return self.plan_tool_sequence(goal, context)
        
        tools_info = self._get_tools_info(available_tools)
        
        prompt = self._build_tool_selection_prompt(goal, context, tools_info)
        
        try:
            response = await self._llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            selections = self._parse_tool_selections(response.content)
            
            plan_id = str(uuid.uuid4())[:8]
            steps = []
            
            for selection in selections:
                tool_call = ToolCall(
                    tool_name=selection.tool_name,
                    kwargs=selection.parameters,
                    state=ToolState.PENDING,
                    metadata={"reasoning": selection.reasoning, "confidence": selection.confidence}
                )
                steps.append(tool_call)
            
            plan = ExecutionPlan(
                id=plan_id,
                goal=goal,
                steps=steps,
                state=PlanState.CREATED,
                metadata={
                    "context": context,
                    "created_by": "llm_planner",
                    "original_goal": goal
                }
            )
            
            self._active_plans[plan_id] = plan
            logger.info(f"[EnhancedToolOrchestrator] Created LLM plan {plan_id} with {len(steps)} steps")
            
            return plan
            
        except Exception as e:
            logger.error(f"[EnhancedToolOrchestrator] LLM planning failed: {e}")
            return self.plan_tool_sequence(goal, context)
    
    def _get_tools_info(self, tool_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get tool information for LLM.
        
        Args:
            tool_names: Optional list of tool names
        
        Returns:
            List of tool info dictionaries
        """
        if tool_names is None:
            tool_names = list(self.tool_definitions.keys())
        
        tools_info = []
        for name in tool_names:
            if name in self.tool_definitions:
                tool_def = self.tool_definitions[name]
                tools_info.append({
                    "name": tool_def.name,
                    "description": tool_def.description,
                    "parameters": [p.name for p in tool_def.parameters] if hasattr(tool_def, 'parameters') else [],
                    "dependencies": tool_def.dependencies if hasattr(tool_def, 'dependencies') else [],
                })
        
        return tools_info
    
    def _build_tool_selection_prompt(
        self,
        goal: str,
        context: Dict[str, Any],
        tools_info: List[Dict[str, Any]]
    ) -> str:
        """Build prompt for LLM tool selection.
        
        Args:
            goal: The goal to achieve
            context: Current context
            tools_info: Available tools information
        
        Returns:
            Prompt string
        """
        context_str = json.dumps(context, indent=2, default=str)
        tools_str = json.dumps(tools_info, indent=2)
        
        return f"""You are a tool selection expert. Analyze the following goal and select the appropriate tools to achieve it.

## Goal
{goal}

## Current Context
{context_str}

## Available Tools
{tools_str}

## Task
Select the tools needed to achieve this goal. Consider:
1. The goal requirements
2. Tool dependencies (use outputs from one tool as inputs to another)
3. The correct order of execution

## Output Format
Return a JSON array of tool selections, each with:
- tool_name: The name of the tool
- parameters: The parameters to pass to the tool
- reasoning: Why this tool is needed
- confidence: How confident you are (0.0-1.0)

Example:
[
  {{"tool_name": "read_file", "parameters": {{"file_path": "src/App.java"}}, "reasoning": "Need to read source file first", "confidence": 0.95}},
  {{"tool_name": "generate_tests", "parameters": {{"class_info": "$context.class_info"}}, "reasoning": "Generate tests based on source", "confidence": 0.9}}
]

Only output the JSON array, no additional text:"""
    
    def _parse_tool_selections(self, response: str) -> List[ToolSelection]:
        """Parse LLM response to extract tool selections.
        
        Args:
            response: LLM response text
        
        Returns:
            List of ToolSelection objects
        """
        try:
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start == -1 or json_end == 0:
                logger.warning("[EnhancedToolOrchestrator] No JSON found in response")
                return []
            
            json_str = response[json_start:json_end]
            selections_data = json.loads(json_str)
            
            selections = []
            for item in selections_data:
                tool_name = item.get("tool_name", "")
                if tool_name and tool_name in self.tool_definitions:
                    selections.append(ToolSelection(
                        tool_name=tool_name,
                        parameters=item.get("parameters", {}),
                        reasoning=item.get("reasoning", ""),
                        confidence=item.get("confidence", 0.5)
                    ))
            
            return selections
            
        except json.JSONDecodeError as e:
            logger.error(f"[EnhancedToolOrchestrator] Failed to parse JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"[EnhancedToolOrchestrator] Failed to parse selections: {e}")
            return []
    
    async def execute_with_reasoning(
        self,
        goal: str,
        context: Dict[str, Any],
        on_progress: Optional[Callable[[float], None]] = None
    ) -> OrchestrationResult:
        """Execute goal with LLM-generated plan and reasoning.
        
        Args:
            goal: The goal to achieve
            context: Current context
            on_progress: Progress callback
        
        Returns:
            OrchestrationResult with execution results
        """
        plan = await self.plan_from_goal(goal, context)
        
        if not plan.steps:
            return OrchestrationResult(
                success=False,
                plan=plan,
                results={},
                message="No tools selected for goal"
            )
        
        return await self.execute_plan(plan, on_progress)
    
    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics.
        
        Returns:
            Statistics dictionary
        """
        stats = self.get_tool_stats()
        
        total_executions = sum(s["total_calls"] for s in stats.values())
        total_success = sum(s["successful"] for s in stats.values())
        
        return {
            "total_executions": total_executions,
            "total_successes": total_success,
            "overall_success_rate": total_success / total_executions if total_executions > 0 else 0,
            "by_tool": stats
        }


def create_enhanced_orchestrator(
    tools: Optional[Dict[str, Callable]] = None,
    llm_client: Optional[LLMClient] = None,
    max_parallel: int = 3
) -> EnhancedToolOrchestrator:
    """Create an EnhancedToolOrchestrator instance.
    
    Args:
        tools: Tool functions dictionary
        llm_client: LLM client for tool selection
        max_parallel: Maximum parallel executions
    
    Returns:
        EnhancedToolOrchestrator instance
    """
    return EnhancedToolOrchestrator(
        tools=tools,
        llm_client=llm_client,
        max_parallel=max_parallel
    )
