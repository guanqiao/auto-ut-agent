"""Tool Memory - Learning from tool usage patterns.

This module provides:
- Recording successful and failed tool calls
- Retrieving recommended tools for task types
- Learning tool usage patterns
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolCallRecord:
    """Record of a tool call."""
    tool_name: str
    parameters: Dict[str, Any]
    context: Dict[str, Any]
    result: Any
    success: bool
    timestamp: datetime
    task_type: str = ""
    error: Optional[str] = None


@dataclass
class ToolRecommendation:
    """Recommendation for tool usage."""
    tool_name: str
    reason: str
    success_rate: float
    usage_count: int
    avg_duration: float


class ToolMemory:
    """Tool usage memory - learns from success and failure patterns.
    
    Features:
    - Records tool call history
    - Tracks success/failure rates
    - Provides recommendations for task types
    - Persists to disk
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """Initialize tool memory.
        
        Args:
            storage_path: Optional path for persistence
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self._records: List[ToolCallRecord] = []
        self._task_type_tools: Dict[str, Dict[str, int]] = {}
        self._tool_stats: Dict[str, Dict[str, Any]] = {}
        
        if self.storage_path and self.storage_path.exists():
            self._load()
        
        logger.info(f"[ToolMemory] Initialized with {len(self._records)} records")
    
    async def record_success(
        self,
        tool_name: str,
        params: Dict[str, Any],
        context: Dict[str, Any],
        result: Any,
        task_type: str = ""
    ):
        """Record a successful tool call.
        
        Args:
            tool_name: Name of the tool
            params: Parameters used
            context: Execution context
            result: Tool result
            task_type: Type of task
        """
        record = ToolCallRecord(
            tool_name=tool_name,
            parameters=params,
            context=context,
            result=str(result)[:500] if result else "",
            success=True,
            timestamp=datetime.now(),
            task_type=task_type
        )
        
        self._records.append(record)
        self._update_stats(record)
        
        if task_type:
            if task_type not in self._task_type_tools:
                self._task_type_tools[task_type] = {}
            if tool_name not in self._task_type_tools[task_type]:
                self._task_type_tools[task_type][tool_name] = 0
            self._task_type_tools[task_type][tool_name] += 1
        
        logger.debug(f"[ToolMemory] Recorded success: {tool_name}")
        
        if self.storage_path:
            self._save()
    
    async def record_failure(
        self,
        tool_name: str,
        params: Dict[str, Any],
        context: Dict[str, Any],
        error: str,
        task_type: str = ""
    ):
        """Record a failed tool call.
        
        Args:
            tool_name: Name of the tool
            params: Parameters used
            context: Execution context
            error: Error message
            task_type: Type of task
        """
        record = ToolCallRecord(
            tool_name=tool_name,
            parameters=params,
            context=context,
            result="",
            success=False,
            timestamp=datetime.now(),
            task_type=task_type,
            error=error[:500]
        )
        
        self._records.append(record)
        self._update_stats(record)
        
        if self.storage_path:
            self._save()
    
    def _update_stats(self, record: ToolCallRecord):
        """Update statistics for a tool.
        
        Args:
            record: Tool call record
        """
        tool_name = record.tool_name
        
        if tool_name not in self._tool_stats:
            self._tool_stats[tool_name] = {
                "total_calls": 0,
                "successes": 0,
                "failures": 0,
                "success_rate": 0.0,
                "avg_duration": 0.0,
                "last_used": None
            }
        
        stats = self._tool_stats[tool_name]
        stats["total_calls"] += 1
        
        if record.success:
            stats["successes"] += 1
        else:
            stats["failures"] += 1
        
        stats["success_rate"] = stats["successes"] / stats["total_calls"]
        stats["last_used"] = record.timestamp.isoformat()
    
    async def get_recommended_tools(
        self,
        task_type: str,
        limit: int = 5
    ) -> List[ToolRecommendation]:
        """Get recommended tools for a task type.
        
        Args:
            task_type: Type of task
            limit: Maximum number of recommendations
        
        Returns:
            List of ToolRecommendation
        """
        if task_type not in self._task_type_tools:
            return []
        
        tool_counts = self._task_type_tools[task_type]
        
        recommendations = []
        for tool_name, count in tool_counts.items():
            if tool_name in self._tool_stats:
                stats = self._tool_stats[tool_name]
                recommendations.append(ToolRecommendation(
                    tool_name=tool_name,
                    reason=f"Used {count} times for {task_type} tasks",
                    success_rate=stats["success_rate"],
                    usage_count=count,
                    avg_duration=stats.get("avg_duration", 0.0)
                ))
        
        recommendations.sort(key=lambda x: (x.success_rate * x.usage_count), reverse=True)
        return recommendations[:limit]
    
    def get_tool_stats(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific tool.
        
        Args:
            tool_name: Name of the tool
        
        Returns:
            Statistics dictionary or None
        """
        return self._tool_stats.get(tool_name)
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all tools.
        
        Returns:
            Dictionary of tool statistics
        """
        return self._tool_stats.copy()
    
    def get_most_successful_tools(self, min_calls: int = 3) -> List[Dict[str, Any]]:
        """Get most successful tools.
        
        Args:
            min_calls: Minimum number of calls to consider
        
        Returns:
            List of successful tools
        """
        results = []
        
        for tool_name, stats in self._tool_stats.items():
            if stats["total_calls"] >= min_calls:
                results.append({
                    "tool_name": tool_name,
                    "success_rate": stats["success_rate"],
                    "total_calls": stats["total_calls"],
                    "last_used": stats.get("last_used")
                })
        
        results.sort(key=lambda x: x["success_rate"], reverse=True)
        return results
    
    def get_least_reliable_tools(self, min_calls: int = 3) -> List[Dict[str, Any]]:
        """Get least reliable tools.
        
        Args:
            min_calls: Minimum number of calls to consider
        
        Returns:
            List of unreliable tools
        """
        results = []
        
        for tool_name, stats in self._tool_stats.items():
            if stats["total_calls"] >= min_calls:
                results.append({
                    "tool_name": tool_name,
                    "success_rate": stats["success_rate"],
                    "total_calls": stats["total_calls"],
                    "failures": stats["failures"]
                })
        
        results.sort(key=lambda x: x["success_rate"])
        return results
    
    def get_recent_records(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent tool call records.
        
        Args:
            limit: Maximum number of records
        
        Returns:
            List of recent records
        """
        recent = sorted(self._records, key=lambda x: x.timestamp, reverse=True)[:limit]
        
        return [
            {
                "tool_name": r.tool_name,
                "success": r.success,
                "timestamp": r.timestamp.isoformat(),
                "task_type": r.task_type,
                "error": r.error
            }
            for r in recent
        ]
    
    def clear_history(self):
        """Clear all history."""
        self._records.clear()
        self._task_type_tools.clear()
        self._tool_stats.clear()
        
        if self.storage_path and self.storage_path.exists():
            self.storage_path.unlink()
        
        logger.info("[ToolMemory] History cleared")
    
    def _save(self):
        """Save memory to disk."""
        if not self.storage_path:
            return
        
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "records": [
                    {
                        "tool_name": r.tool_name,
                        "parameters": r.parameters,
                        "context": r.context,
                        "result": r.result,
                        "success": r.success,
                        "timestamp": r.timestamp.isoformat(),
                        "task_type": r.task_type,
                        "error": r.error
                    }
                    for r in self._records[-1000:]
                ],
                "task_type_tools": self._task_type_tools,
                "tool_stats": self._tool_stats
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"[ToolMemory] Saved to {self.storage_path}")
            
        except Exception as e:
            logger.error(f"[ToolMemory] Failed to save: {e}")
    
    def _load(self):
        """Load memory from disk."""
        if not self.storage_path or not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            self._records = [
                ToolCallRecord(
                    tool_name=r["tool_name"],
                    parameters=r["parameters"],
                    context=r["context"],
                    result=r["result"],
                    success=r["success"],
                    timestamp=datetime.fromisoformat(r["timestamp"]),
                    task_type=r.get("task_type", ""),
                    error=r.get("error")
                )
                for r in data.get("records", [])
            ]
            
            self._task_type_tools = data.get("task_type_tools", {})
            self._tool_stats = data.get("tool_stats", {})
            
            logger.info(f"[ToolMemory] Loaded {len(self._records)} records from {self.storage_path}")
            
        except Exception as e:
            logger.error(f"[ToolMemory] Failed to load: {e}")


def create_tool_memory(storage_path: Optional[str] = None) -> ToolMemory:
    """Create a ToolMemory instance.
    
    Args:
        storage_path: Optional path for persistence
    
    Returns:
        ToolMemory instance
    """
    return ToolMemory(storage_path)
