"""Intelligent Tool Selector - Smart tool selection based on history.

This module provides:
- Success rate based tool ranking
- Context-aware parameter suggestions
- Task-type based tool recommendations
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ToolScore:
    """Score for a tool recommendation."""
    tool_name: str
    score: float
    reasoning: str
    success_rate: float
    usage_count: int


class IntelligentToolSelector:
    """Intelligent tool selector with learning capabilities.
    
    Features:
    - Success rate based ranking
    - Context-aware recommendations
    - Task pattern matching
    """
    
    def __init__(self, tool_memory=None):
        """Initialize the selector.
        
        Args:
            tool_memory: Optional ToolMemory instance for learning
        """
        self._tool_memory = tool_memory
        self._task_patterns: Dict[str, List[str]] = {}
        self._initialize_default_patterns()
    
    def _initialize_default_patterns(self):
        """Initialize default task-tool patterns."""
        self._task_patterns = {
            "read_code": ["read_file", "grep"],
            "write_code": ["write_file", "edit_file"],
            "search": ["grep", "glob", "web_search"],
            "build": ["bash"],
            "test": ["bash"],
            "version_control": ["git_status", "git_diff", "git_commit", "git_branch"],
            "debug": ["bash", "grep", "read_file"],
            "analyze": ["read_file", "grep", "glob"],
            "search_docs": ["web_search", "web_fetch"],
        }
    
    def set_tool_memory(self, tool_memory):
        """Set the tool memory for learning.
        
        Args:
            tool_memory: ToolMemory instance
        """
        self._tool_memory = tool_memory
    
    def select_tools(
        self,
        task: str,
        available_tools: List[str],
        context: Optional[Dict[str, Any]] = None,
        limit: int = 3
    ) -> List[ToolScore]:
        """Select best tools for a task.
        
        Args:
            task: Task description
            available_tools: List of available tool names
            context: Optional context
            limit: Maximum number of tools to return
        
        Returns:
            List of ToolScore with rankings
        """
        context = context or {}
        scores = []
        
        task_lower = task.lower()
        
        for tool_name in available_tools:
            score = 0.0
            reasons = []
            
            pattern_score = self._match_task_pattern(task_lower, tool_name)
            if pattern_score > 0:
                score += pattern_score * 0.4
                reasons.append(f"Pattern match: {pattern_score:.2f}")
            
            if self._tool_memory:
                success_rate, usage_count = self._get_tool_stats(tool_name)
                if usage_count > 0:
                    score += success_rate * 0.4
                    reasons.append(f"Success rate: {success_rate:.2f} ({usage_count} calls)")
            
            context_score = self._match_context(task_lower, tool_name, context)
            if context_score > 0:
                score += context_score * 0.2
                reasons.append(f"Context match: {context_score:.2f}")
            
            if score > 0:
                success_rate, usage_count = self._get_tool_stats(tool_name)
                scores.append(ToolScore(
                    tool_name=tool_name,
                    score=score,
                    reasoning="; ".join(reasons) if reasons else "Default selection",
                    success_rate=success_rate,
                    usage_count=usage_count
                ))
        
        scores.sort(key=lambda x: x.score, reverse=True)
        return scores[:limit]
    
    def _match_task_pattern(self, task: str, tool_name: str) -> float:
        """Match tool against task patterns.
        
        Args:
            task: Task description (lowercase)
            tool_name: Tool name
        
        Returns:
            Match score 0-1
        """
        for pattern_name, tools in self._task_patterns.items():
            if pattern_name in task:
                if tool_name in tools:
                    return 1.0
                for tool in tools:
                    if tool in tool_name or tool_name in tool:
                        return 0.8
        
        keyword_mappings = {
            "read": ["read_file"],
            "write": ["write_file"],
            "edit": ["edit_file"],
            "search": ["grep", "glob", "web_search"],
            "find": ["grep", "glob"],
            "git": ["git_"],
            "build": ["bash", "maven", "gradle"],
            "test": ["bash"],
            "run": ["bash"],
            "compile": ["bash"],
            "error": ["grep", "bash"],
            "debug": ["bash", "grep"],
            "web": ["web_search", "web_fetch"],
            "internet": ["web_search"],
            "documentation": ["web_fetch", "read_file"],
        }
        
        for keyword, matching_tools in keyword_mappings.items():
            if keyword in task:
                for mt in matching_tools:
                    if mt in tool_name or tool_name in mt:
                        return 0.7
        
        return 0.0
    
    def _get_tool_stats(self, tool_name: str) -> Tuple[float, int]:
        """Get tool statistics from memory.
        
        Args:
            tool_name: Tool name
        
        Returns:
            Tuple of (success_rate, usage_count)
        """
        if not self._tool_memory:
            return 0.5, 0
        
        try:
            stats = self._tool_memory.get_tool_stats(tool_name)
            if stats:
                return stats.get("success_rate", 0.5), stats.get("total_calls", 0)
        except Exception:
            pass
        
        return 0.5, 0
    
    def _match_context(
        self,
        task: str,
        tool_name: str,
        context: Dict[str, Any]
    ) -> float:
        """Match tool against context.
        
        Args:
            task: Task description
            tool_name: Tool name
            context: Execution context
        
        Returns:
            Match score 0-1
        """
        score = 0.0
        
        if "file_path" in context:
            if tool_name in ["read_file", "write_file", "edit_file"]:
                score += 0.5
            if tool_name in ["grep", "glob"]:
                score += 0.3
        
        if "command" in context or "shell" in context:
            if tool_name == "bash":
                score += 0.5
        
        if "url" in context:
            if tool_name in ["web_fetch", "web_search"]:
                score += 0.8
        
        if "query" in context:
            if tool_name in ["web_search", "grep"]:
                score += 0.5
        
        return min(score, 1.0)
    
    def suggest_parameters(
        self,
        tool_name: str,
        task: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Suggest parameters for a tool based on task and context.
        
        Args:
            tool_name: Tool name
            task: Task description
            context: Execution context
        
        Returns:
            Suggested parameters
        """
        params = {}
        
        task_lower = task.lower()
        
        if tool_name == "read_file":
            if "file_path" in context:
                params["file_path"] = context["file_path"]
            elif "file" in context:
                params["file_path"] = context["file"]
        
        elif tool_name == "grep":
            if "pattern" in context:
                params["pattern"] = context["pattern"]
            elif "search" in task_lower:
                params["pattern"] = task_lower.replace("search for ", "").strip()
            if "file_path" in context:
                params["path"] = context["file_path"]
        
        elif tool_name == "bash":
            if "command" in context:
                params["command"] = context["command"]
            elif "build" in task_lower:
                params["command"] = "mvn build"
            elif "test" in task_lower:
                params["command"] = "mvn test"
        
        elif tool_name == "git_commit":
            if "message" in context:
                params["message"] = context["message"]
            elif "commit" in task_lower:
                words = task_lower.replace("commit", "").strip()
                if words:
                    params["message"] = words
                else:
                    params["message"] = "Update"
            params["add_all"] = True
        
        elif tool_name == "web_search":
            if "query" in context:
                params["query"] = context["query"]
            else:
                params["query"] = task
        
        elif tool_name == "web_fetch":
            if "url" in context:
                params["url"] = context["url"]
        
        return params
    
    def add_task_pattern(self, task_type: str, tool_names: List[str]):
        """Add a new task-tool pattern.
        
        Args:
            task_type: Type of task
            tool_names: List of tool names for this task
        """
        self._task_patterns[task_type] = tool_names
        logger.info(f"[IntelligentToolSelector] Added pattern: {task_type} -> {tool_names}")
    
    def learn_from_execution(
        self,
        tool_name: str,
        task: str,
        success: bool,
        params: Dict[str, Any]
    ):
        """Learn from tool execution for future recommendations.
        
        Args:
            tool_name: Tool that was executed
            task: Task description
            success: Whether execution was successful
            params: Parameters used
        """
        if not self._tool_memory:
            return
        
        task_type = self._classify_task(task)
        
        if success:
            if task_type not in self._task_patterns:
                self._task_patterns[task_type] = []
            
            if tool_name not in self._task_patterns[task_type]:
                self._task_patterns[task_type].append(tool_name)
    
    def _classify_task(self, task: str) -> str:
        """Classify task into a category.
        
        Args:
            task: Task description
        
        Returns:
            Task category
        """
        task_lower = task.lower()
        
        for pattern_name in self._task_patterns:
            if pattern_name in task_lower:
                return pattern_name
        
        return "general"


def create_intelligent_selector(tool_memory=None) -> IntelligentToolSelector:
    """Create an intelligent tool selector.
    
    Args:
        tool_memory: Optional tool memory
    
    Returns:
        IntelligentToolSelector instance
    """
    return IntelligentToolSelector(tool_memory)
