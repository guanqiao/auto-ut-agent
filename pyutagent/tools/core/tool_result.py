"""Tool Result Definition.

This module provides standard result types for tool execution.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Type
import json


class ResultStatus(Enum):
    """Status of tool execution."""
    SUCCESS = auto()
    FAILURE = auto()
    TIMEOUT = auto()
    CANCELLED = auto()
    PENDING = auto()


@dataclass
class ToolError:
    """Error information from tool execution."""
    code: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolError":
        """Create from dictionary."""
        return cls(
            code=data.get("code", "UNKNOWN"),
            message=data.get("message", ""),
            details=data.get("details", {}),
        )


@dataclass
class ToolResult:
    """Standard result from tool execution.
    
    Provides:
    - Status tracking
    - Output data
    - Error handling
    - Metadata
    - Artifacts (files created/modified)
    """
    
    status: ResultStatus = ResultStatus.SUCCESS
    output: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[ToolError] = None
    
    duration_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def success(self) -> bool:
        """Check if execution was successful."""
        return self.status == ResultStatus.SUCCESS
    
    @property
    def failed(self) -> bool:
        """Check if execution failed."""
        return self.status == ResultStatus.FAILURE
    
    @property
    def timed_out(self) -> bool:
        """Check if execution timed out."""
        return self.status == ResultStatus.TIMEOUT
    
    @property
    def cancelled(self) -> bool:
        """Check if execution was cancelled."""
        return self.status == ResultStatus.CANCELLED
    
    @classmethod
    def ok(
        cls,
        output: str = "",
        data: Optional[Dict[str, Any]] = None,
        artifacts: Optional[List[str]] = None,
        **kwargs,
    ) -> "ToolResult":
        """Create a successful result.
        
        Args:
            output: Output string
            data: Result data
            artifacts: List of artifacts
            **kwargs: Additional metadata
            
        Returns:
            Successful ToolResult
        """
        return cls(
            status=ResultStatus.SUCCESS,
            output=output,
            data=data or {},
            artifacts=artifacts or [],
            metadata=kwargs,
        )
    
    @classmethod
    def fail(
        cls,
        error: str,
        code: str = "EXECUTION_ERROR",
        details: Optional[Dict[str, Any]] = None,
        output: str = "",
    ) -> "ToolResult":
        """Create a failure result.
        
        Args:
            error: Error message
            code: Error code
            details: Error details
            output: Partial output if any
            
        Returns:
            Failed ToolResult
        """
        return cls(
            status=ResultStatus.FAILURE,
            output=output,
            error=ToolError(
                code=code,
                message=error,
                details=details or {},
            ),
        )
    
    @classmethod
    def timeout(
        cls,
        timeout_seconds: float,
        output: str = "",
    ) -> "ToolResult":
        """Create a timeout result.
        
        Args:
            timeout_seconds: Timeout duration
            output: Partial output if any
            
        Returns:
            Timeout ToolResult
        """
        return cls(
            status=ResultStatus.TIMEOUT,
            output=output,
            error=ToolError(
                code="TIMEOUT",
                message=f"Execution timed out after {timeout_seconds}s",
            ),
        )
    
    @classmethod
    def cancel(cls, reason: str = "") -> "ToolResult":
        """Create a cancelled result.
        
        Args:
            reason: Cancellation reason
            
        Returns:
            Cancelled ToolResult
        """
        return cls(
            status=ResultStatus.CANCELLED,
            error=ToolError(
                code="CANCELLED",
                message=reason or "Execution was cancelled",
            ),
        )
    
    def with_duration(self, duration_ms: int) -> "ToolResult":
        """Add duration to result.
        
        Args:
            duration_ms: Duration in milliseconds
            
        Returns:
            Self for chaining
        """
        self.duration_ms = duration_ms
        return self
    
    def with_metadata(self, **kwargs) -> "ToolResult":
        """Add metadata to result.
        
        Args:
            **kwargs: Metadata key-value pairs
            
        Returns:
            Self for chaining
        """
        self.metadata.update(kwargs)
        return self
    
    def with_artifacts(self, *paths: str) -> "ToolResult":
        """Add artifacts to result.
        
        Args:
            *paths: Artifact paths
            
        Returns:
            Self for chaining
        """
        self.artifacts.extend(paths)
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "status": self.status.name,
            "output": self.output,
            "data": self.data,
            "error": self.error.to_dict() if self.error else None,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "artifacts": self.artifacts,
            "timestamp": self.timestamp.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolResult":
        """Create from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            ToolResult instance
        """
        error = None
        if data.get("error"):
            error = ToolError.from_dict(data["error"])
        
        return cls(
            status=ResultStatus[data.get("status", "SUCCESS")],
            output=data.get("output", ""),
            data=data.get("data", {}),
            error=error,
            duration_ms=data.get("duration_ms", 0),
            metadata=data.get("metadata", {}),
            artifacts=data.get("artifacts", []),
        )
    
    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.success
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ToolResult(status={self.status.name}, duration={self.duration_ms}ms)"
