"""Tool Composer - Automatic tool composition for complex tasks.

This module provides:
- Goal analysis
- Tool sequence planning
- Automatic tool chain generation
- Parameter inference between tools
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from .tool_orchestrator import ExecutionPlan, ToolCall, ToolState

logger = logging.getLogger(__name__)


@dataclass
class ToolChain:
    """Represents a chain of tools to execute."""
    tools: List[Dict[str, Any]] = field(default_factory=list)
    description: str = ""
    estimated_duration: float = 0.0


@dataclass
class ToolInput:
    """Represents input requirements for a tool."""
    required: List[str] = field(default_factory=list)
    optional: List[str] = field(default_factory=list)
    from_context: List[str] = field(default_factory=list)


@dataclass
class ToolOutput:
    """Represents outputs from a tool."""
    provides: List[str] = field(default_factory=list)


class ToolComposer:
    """Composer for automatic tool chain generation.
    
    Features:
    - Analyzes complex goals
    - Generates tool execution sequences
    - Infers parameters between tools
    - Handles tool dependencies
    """
    
    TOOL_INPUTS = {
        "read_file": ToolInput(required=["file_path"], optional=["offset", "limit"]),
        "write_file": ToolInput(required=["file_path", "content"]),
        "edit_file": ToolInput(required=["file_path", "search", "replace"], optional=["global_replace"]),
        "grep": ToolInput(required=["pattern"], optional=["path", "glob", "regex"]),
        "glob": ToolInput(required=["pattern"], optional=["path", "recursive"]),
        "bash": ToolInput(required=["command"], optional=["timeout", "env"]),
        "git_status": ToolInput(required=[], optional=[]),
        "git_diff": ToolInput(required=[], optional=["file_path", "staged"]),
        "git_commit": ToolInput(required=["message"], optional=["add_all", "amend"]),
        "git_branch": ToolInput(required=[], optional=["action", "branch_name"]),
        "git_log": ToolInput(required=[], optional=["max_count", "file_path"]),
        "git_add": ToolInput(required=[], optional=["file_path", "all"]),
        "git_push": ToolInput(required=[], optional=["remote", "branch"]),
        "git_pull": ToolInput(required=[], optional=["remote", "branch"]),
    }
    
    TOOL_OUTPUTS = {
        "read_file": ToolOutput(provides=["content"]),
        "write_file": ToolOutput(provides=["file_path"]),
        "edit_file": ToolOutput(provides=["file_path"]),
        "grep": ToolOutput(provides=["matches", "results"]),
        "glob": ToolOutput(provides=["files"]),
        "bash": ToolOutput(provides=["stdout", "stderr", "exit_code"]),
        "git_status": ToolOutput(provides=["status", "changed_files"]),
        "git_diff": ToolOutput(provides=["diff"]),
        "git_commit": ToolOutput(provides=["commit_hash"]),
        "git_branch": ToolOutput(provides=["branches"]),
        "git_log": ToolOutput(provides=["commits"]),
    }
    
    def __init__(self, available_tools: Optional[List[str]] = None):
        """Initialize tool composer.
        
        Args:
            available_tools: List of available tool names
        """
        self.available_tools = available_tools or list(self.TOOL_INPUTS.keys())
        self._llm_client = None
    
    def set_llm_client(self, llm_client):
        """Set LLM client for intelligent composition.
        
        Args:
            llm_client: LLM client instance
        """
        self._llm_client = llm_client
    
    async def compose(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ToolChain:
        """Compose tool chain from goal.
        
        Args:
            goal: The goal to achieve
            context: Current context
        
        Returns:
            ToolChain with planned tools
        """
        context = context or {}
        
        if self._llm_client:
            return await self._compose_with_llm(goal, context)
        else:
            return self._compose_heuristic(goal, context)
    
    async def _compose_with_llm(
        self,
        goal: str,
        context: Dict[str, Any]
    ) -> ToolChain:
        """Compose using LLM for intelligent tool selection.
        
        Args:
            goal: The goal
            context: Current context
        
        Returns:
            ToolChain
        """
        import json
        
        tools_info = []
        for tool_name in self.available_tools:
            if tool_name in self.TOOL_INPUTS:
                tools_info.append({
                    "name": tool_name,
                    "inputs": self.TOOL_INPUTS[tool_name].required,
                    "outputs": self.TOOL_OUTPUTS.get(tool_name, ToolOutput()).provides
                })
        
        prompt = f"""Analyze the following goal and create a tool execution plan.

Goal: {goal}

Context:
{json.dumps(context, indent=2)}

Available Tools:
{json.dumps(tools_info, indent=2)}

Create a JSON array of tools to execute in order. Each tool should have:
- name: tool name
- parameters: parameters for the tool
- reasoning: why this tool is needed

Output only JSON:"""
        
        try:
            response = await self._llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            json_start = response.content.find('[')
            json_end = response.content.rfind(']') + 1
            
            if json_start == -1:
                logger.warning("[ToolComposer] No JSON in LLM response")
                return self._compose_heuristic(goal, context)
            
            tools_data = json.loads(response.content[json_start:json_end])
            
            tools = []
            for tool_data in tools_data:
                tools.append({
                    "tool_name": tool_data.get("name"),
                    "parameters": tool_data.get("parameters", {}),
                    "reasoning": tool_data.get("reasoning", "")
                })
            
            return ToolChain(
                tools=tools,
                description=f"LLM-composed chain for: {goal}",
                estimated_duration=len(tools) * 5.0
            )
            
        except Exception as e:
            logger.error(f"[ToolComposer] LLM composition failed: {e}")
            return self._compose_heuristic(goal, context)
    
    def _compose_heuristic(
        self,
        goal: str,
        context: Dict[str, Any]
    ) -> ToolChain:
        """Compose using heuristics when LLM is not available.
        
        Args:
            goal: The goal
            context: Current context
        
        Returns:
            ToolChain
        """
        goal_lower = goal.lower()
        tools = []
        
        if any(kw in goal_lower for kw in ["read", "view", "check", "show"]):
            if "diff" in goal_lower:
                tools.append({"tool_name": "git_diff", "parameters": {}, "reasoning": "View changes"})
            elif "status" in goal_lower:
                tools.append({"tool_name": "git_status", "parameters": {}, "reasoning": "Check status"})
            elif "log" in goal_lower:
                tools.append({"tool_name": "git_log", "parameters": {"max_count": 10}, "reasoning": "View history"})
            else:
                file_path = context.get("file_path")
                if file_path:
                    tools.append({"tool_name": "read_file", "parameters": {"file_path": file_path}, "reasoning": "Read file"})
        
        if any(kw in goal_lower for kw in ["write", "create", "save"]):
            file_path = context.get("file_path")
            content = context.get("content")
            if file_path and content:
                tools.append({"tool_name": "write_file", "parameters": {"file_path": file_path, "content": content}, "reasoning": "Write file"})
        
        if any(kw in goal_lower for kw in ["edit", "modify", "update", "change"]):
            file_path = context.get("file_path")
            search = context.get("search")
            replace = context.get("replace")
            if file_path and search and replace:
                tools.append({"tool_name": "edit_file", "parameters": {"file_path": file_path, "search": search, "replace": replace}, "reasoning": "Edit file"})
        
        if any(kw in goal_lower for kw in ["run", "execute", "build", "test"]):
            command = context.get("command", "echo 'Build complete'")
            tools.append({"tool_name": "bash", "parameters": {"command": command}, "reasoning": "Execute command"})
        
        if any(kw in goal_lower for kw in ["commit", "push", "pull"]):
            if "commit" in goal_lower:
                message = context.get("message", "Update")
                tools.append({"tool_name": "git_add", "parameters": {"file_path": "."}, "reasoning": "Stage files"})
                tools.append({"tool_name": "git_commit", "parameters": {"message": message}, "reasoning": "Commit changes"})
            elif "push" in goal_lower:
                tools.append({"tool_name": "git_push", "parameters": {}, "reasoning": "Push to remote"})
            elif "pull" in goal_lower:
                tools.append({"tool_name": "git_pull", "parameters": {}, "reasoning": "Pull from remote"})
        
        if any(kw in goal_lower for kw in ["search", "find", "grep"]):
            pattern = context.get("pattern", ".")
            path = context.get("path", ".")
            tools.append({"tool_name": "grep", "parameters": {"pattern": pattern, "path": path}, "reasoning": "Search code"})
        
        return ToolChain(
            tools=tools,
            description=f"Heuristic-composed chain for: {goal}",
            estimated_duration=len(tools) * 5.0
        )
    
    def infer_parameters(
        self,
        previous_outputs: Dict[str, Any],
        tool_name: str
    ) -> Dict[str, Any]:
        """Infer parameters for a tool from previous outputs.
        
        Args:
            previous_outputs: Outputs from previous tool executions
            tool_name: Name of the tool
        
        Returns:
            Inferred parameters
        """
        if tool_name not in self.TOOL_INPUTS:
            return {}
        
        inputs = self.TOOL_INPUTS[tool_name]
        inferred = {}
        
        for req in inputs.required:
            if req in previous_outputs:
                inferred[req] = previous_outputs[req]
            elif req in inputs.from_context:
                pass
        
        return inferred
    
    def get_tool_dependencies(self, tool_name: str) -> List[str]:
        """Get dependencies for a tool.
        
        Args:
            tool_name: Name of the tool
        
        Returns:
            List of tool names this tool depends on
        """
        output = self.TOOL_OUTPUTS.get(tool_name)
        if not output:
            return []
        
        dependencies = []
        
        for other_tool, other_output in self.TOOL_OUTPUTS.items():
            if other_tool == tool_name:
                continue
            
            for provide in output.provides:
                if provide in other_output.provides:
                    dependencies.append(other_tool)
                    break
        
        return dependencies
    
    def validate_chain(self, chain: ToolChain) -> tuple[bool, List[str]]:
        """Validate a tool chain for correctness.
        
        Args:
            chain: ToolChain to validate
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        provided_vars: Set[str] = set()
        
        for tool_def in chain.tools:
            tool_name = tool_def.get("tool_name")
            
            if tool_name not in self.available_tools:
                errors.append(f"Tool not available: {tool_name}")
                continue
            
            if tool_name in self.TOOL_INPUTS:
                inputs = self.TOOL_INPUTS[tool_name]
                
                for req in inputs.required:
                    if req not in tool_def.get("parameters", {}) and req not in provided_vars:
                        errors.append(f"Tool {tool_name} requires {req} which is not provided")
            
            if tool_name in self.TOOL_OUTPUTS:
                outputs = self.TOOL_OUTPUTS[tool_name]
                provided_vars.update(outputs.provides)
        
        return len(errors) == 0, errors


def create_tool_composer(available_tools: Optional[List[str]] = None) -> ToolComposer:
    """Create a ToolComposer instance.
    
    Args:
        available_tools: List of available tool names
    
    Returns:
        ToolComposer instance
    """
    return ToolComposer(available_tools)
