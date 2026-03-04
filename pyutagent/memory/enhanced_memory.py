"""Enhanced Learning System for tool usage.

This module provides:
- EnhancedToolMemory: Extended memory with learning capabilities
- ToolPatternLearner: Learn patterns from execution
- SuccessPredictor: Predict tool success
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExecutionRecord:
    """Record of a tool execution."""
    tool_name: str
    parameters: Dict[str, Any]
    context: Dict[str, Any]
    success: bool
    duration: float
    timestamp: datetime
    error: Optional[str] = None


@dataclass
class ToolPattern:
    """Learned pattern for tool usage."""
    task_type: str
    tool_sequence: List[str]
    success_rate: float
    usage_count: int
    avg_duration: float


class ToolPatternLearner:
    """Learn patterns from tool execution history."""
    
    def __init__(self):
        self._patterns: Dict[str, List[ToolPattern]] = defaultdict(list)
        self._task_keywords = {
            "read": ["read", "view", "check", "show"],
            "write": ["write", "create", "save", "new"],
            "edit": ["edit", "modify", "change", "update"],
            "search": ["search", "find", "grep"],
            "build": ["build", "compile", "make"],
            "test": ["test", "run", "execute"],
            "git": ["commit", "push", "pull", "branch"],
            "debug": ["debug", "error", "fix"],
        }
    
    def learn(self, records: List[ExecutionRecord]):
        """Learn patterns from execution records.
        
        Args:
            records: List of execution records
        """
        tool_sequences = defaultdict(list)
        
        for record in records:
            task_type = self._classify_from_context(record.context)
            key = (task_type, tuple(record.parameters.keys()))
            tool_sequences[key].append(record)
        
        for (task_type, param_keys), recs in tool_sequences.items():
            successes = sum(1 for r in recs if r.success)
            total = len(recs)
            avg_duration = sum(r.duration for r in recs) / total
            
            pattern = ToolPattern(
                task_type=task_type,
                tool_sequence=[recs[0].tool_name],
                success_rate=successes / total,
                usage_count=total,
                avg_duration=avg_duration
            )
            
            self._patterns[task_type].append(pattern)
    
    def _classify_from_context(self, context: Dict[str, Any]) -> str:
        """Classify task from context."""
        context_str = str(context).lower()
        
        for task_type, keywords in self._task_keywords.items():
            for kw in keywords:
                if kw in context_str:
                    return task_type
        
        return "general"
    
    def get_best_tool(self, task_type: str) -> Optional[str]:
        """Get best tool for task type.
        
        Args:
            task_type: Type of task
        
        Returns:
            Best tool name or None
        """
        patterns = self._patterns.get(task_type, [])
        
        if not patterns:
            return None
        
        best = max(patterns, key=lambda p: (p.success_rate * p.usage_count, -p.avg_duration))
        
        if best.success_rate > 0.3:
            return best.tool_sequence[0] if best.tool_sequence else None
        
        return None
    
    def get_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all learned patterns.
        
        Returns:
            Dictionary of patterns
        """
        output = {}
        
        for task_type, patterns in self._patterns.items():
            output[task_type] = [
                {
                    "tools": p.tool_sequence,
                    "success_rate": p.success_rate,
                    "usage_count": p.usage_count,
                    "avg_duration": p.avg_duration
                }
                for p in patterns
            ]
        
        return output


class SuccessPredictor:
    """Predict tool execution success."""
    
    def __init__(self):
        self._tool_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "successes": 0,
            "failures": 0,
            "total_duration": 0.0,
            "errors": []
        })
    
    def record(self, tool_name: str, success: bool, duration: float, error: Optional[str] = None):
        """Record execution result.
        
        Args:
            tool_name: Tool name
            success: Whether execution succeeded
            duration: Execution duration
            error: Error message if failed
        """
        stats = self._tool_stats[tool_name]
        
        if success:
            stats["successes"] += 1
        else:
            stats["failures"] += 1
            if error:
                stats["errors"].append(error[:100])
        
        stats["total_duration"] += duration
    
    def predict(self, tool_name: str) -> Dict[str, Any]:
        """Predict success probability.
        
        Args:
            tool_name: Tool name
        
        Returns:
            Prediction dictionary
        """
        stats = self._tool_stats[tool_name]
        
        total = stats["successes"] + stats["failures"]
        
        if total == 0:
            return {
                "tool": tool_name,
                "success_probability": 0.5,
                "confidence": "low",
                "total_executions": 0
            }
        
        success_rate = stats["successes"] / total
        avg_duration = stats["total_duration"] / total
        
        confidence = "low"
        if total >= 10:
            confidence = "medium"
        if total >= 30:
            confidence = "high"
        
        return {
            "tool": tool_name,
            "success_probability": success_rate,
            "confidence": confidence,
            "total_executions": total,
            "avg_duration": avg_duration,
            "recent_errors": list(set(stats["errors"]))[:3]
        }
    
    def get_all_predictions(self) -> List[Dict[str, Any]]:
        """Get predictions for all tools.
        
        Returns:
            List of predictions
        """
        return [self.predict(tool) for tool in self._tool_stats.keys()]


class EnhancedToolMemory:
    """Enhanced tool memory with learning capabilities."""
    
    def __init__(self, storage_path: Optional[str] = None):
        from ..memory.tool_memory import ToolMemory
        self._base = ToolMemory(storage_path)
        self._learner = ToolPatternLearner()
        self._predictor = SuccessPredictor()
    
    async def record(self, tool_name: str, params: Dict, context: Dict, result: Any, duration: float = 0.0):
        """Record execution and learn.
        
        Args:
            tool_name: Tool name
            params: Parameters
            context: Context
            result: Result
            duration: Execution duration
        """
        success = getattr(result, 'success', False) if result else False
        error = getattr(result, 'error', None)
        
        if success:
            await self._base.record_success(tool_name, params, context, result)
        else:
            await self._base.record_failure(tool_name, params, context, error or "Unknown")
        
        record = ExecutionRecord(
            tool_name=tool_name,
            parameters=params,
            context=context,
            success=success,
            duration=duration,
            timestamp=datetime.now(),
            error=error
        )
        
        self._learner.learn([record])
        self._predictor.record(tool_name, success, duration, error)
    
    def suggest_tool(self, task: str, context: Dict) -> Optional[Dict[str, Any]]:
        """Suggest best tool for task.
        
        Args:
            task: Task description
            context: Context
        
        Returns:
            Suggestion dictionary
        """
        task_type = self._classify_task(task)
        
        tool_name = self._learner.get_best_tool(task_type)
        
        if tool_name:
            prediction = self._predictor.predict(tool_name)
            return {
                "tool": tool_name,
                "task_type": task_type,
                "success_probability": prediction["success_probability"],
                "confidence": prediction["confidence"]
            }
        
        return None
    
    def _classify_task(self, task: str) -> str:
        """Classify task type."""
        task_lower = task.lower()
        
        keywords = {
            "read": ["read", "view", "check", "show", "get"],
            "write": ["write", "create", "save", "new", "add"],
            "edit": ["edit", "modify", "change", "update", "fix"],
            "search": ["search", "find", "grep", "look"],
            "build": ["build", "compile", "make"],
            "test": ["test", "run", "execute"],
            "git": ["commit", "push", "pull", "branch", "merge"],
            "debug": ["debug", "error", "fix", "problem"],
        }
        
        for task_type, kws in keywords.items():
            for kw in kws:
                if kw in task_lower:
                    return task_type
        
        return "general"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive stats.
        
        Returns:
            Stats dictionary
        """
        base_stats = self._base.get_all_stats()
        predictions = self._predictor.get_all_predictions()
        patterns = self._learner.get_patterns()
        
        return {
            "base_stats": base_stats,
            "predictions": predictions,
            "patterns": patterns
        }


def create_enhanced_memory(storage_path: Optional[str] = None) -> EnhancedToolMemory:
    """Create enhanced tool memory.
    
    Args:
        storage_path: Optional storage path
    
    Returns:
        EnhancedToolMemory instance
    """
    return EnhancedToolMemory(storage_path)
