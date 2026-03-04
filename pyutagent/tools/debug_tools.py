"""Debug tools for error analysis and troubleshooting.

This module provides:
- ErrorAnalysisTool: Analyze error messages
- StackTraceTool: Parse and analyze stack traces
- LogAnalyzerTool: Analyze log files
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .tool import (
    Tool,
    ToolCategory,
    ToolDefinition,
    ToolResult,
    create_tool_parameter,
)

logger = logging.getLogger(__name__)


class ErrorAnalysisTool(Tool):
    """Tool for analyzing error messages."""
    
    def __init__(self):
        super().__init__()
        self._definition = ToolDefinition(
            name="error_analysis",
            description="Analyze error messages and provide possible solutions. "
                       "Use this to understand and fix errors.",
            category=ToolCategory.DEBUG,
            parameters=[
                create_tool_parameter(
                    name="error",
                    param_type="string",
                    description="Error message to analyze",
                    required=True
                ),
                create_tool_parameter(
                    name="language",
                    param_type="string",
                    description="Programming language",
                    required=False,
                    default="auto"
                )
            ],
            examples=[
                {
                    "params": {"error": "java.lang.NullPointerException"},
                    "description": "Analyze NullPointerException"
                },
                {
                    "params": {"error": "pytest failed: assertEqual"},
                    "description": "Analyze pytest failure"
                }
            ],
            tags=["debug", "error", "analysis", "fix"]
        )
    
    @property
    def definition(self) -> ToolDefinition:
        return self._definition
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute error analysis."""
        error = kwargs.get("error", "")
        language = kwargs.get("language", "auto")
        
        if not error:
            return ToolResult(success=False, error="error is required")
        
        analysis = self._analyze_error(error, language)
        
        return ToolResult(
            success=True,
            output=self._format_analysis(analysis)
        )
    
    def _analyze_error(self, error: str, language: str) -> Dict[str, Any]:
        """Analyze error and provide solutions."""
        error_lower = error.lower()
        
        analysis = {
            "error_type": self._extract_error_type(error),
            "severity": "high",
            "possible_causes": [],
            "suggestions": []
        }
        
        if "nullpointerexception" in error_lower or "null" in error_lower:
            analysis["possible_causes"].append("Attempting to access null object")
            analysis["suggestions"].append("Add null check before using the object")
            analysis["suggestions"].append("Use Optional<T> for nullable values")
        
        if "filenotfoundexception" in error_lower or "no such file" in error_lower:
            analysis["possible_causes"].append("File path is incorrect")
            analysis["possible_causes"].append("File does not exist")
            analysis["suggestions"].append("Verify file path is correct")
            analysis["suggestions"].append("Check if file exists before accessing")
        
        if "classnotfoundexception" in error_lower or "importerror" in error_lower or "modulenotfounderror" in error_lower:
            analysis["possible_causes"].append("Missing dependency")
            analysis["possible_causes"].append("Incorrect import path")
            analysis["suggestions"].append("Install missing dependency")
            analysis["suggestions"].append("Check import statement")
        
        if "outofmemory" in error_lower or "oom" in error_lower:
            analysis["severity"] = "critical"
            analysis["possible_causes"].append("Memory leak")
            analysis["possible_causes"].append("Insufficient heap size")
            analysis["suggestions"].append("Increase heap size (-Xmx)")
            analysis["suggestions"].append("Check for memory leaks")
        
        if "timeout" in error_lower:
            analysis["severity"] = "medium"
            analysis["possible_causes"].append("Operation took too long")
            analysis["possible_causes"].append("Network latency")
            analysis["suggestions"].append("Increase timeout value")
            analysis["suggestions"].append("Check network connection")
        
        if "assertionerror" in error_lower or "assert" in error_lower:
            analysis["possible_causes"].append("Test assertion failed")
            analysis["possible_causes"].append("Unexpected value")
            analysis["suggestions"].append("Check expected vs actual value")
            analysis["suggestions"].append("Review test logic")
        
        if not analysis["possible_causes"]:
            analysis["possible_causes"].append("Unknown error type")
            analysis["suggestions"].append("Search for error message online")
            analysis["suggestions"].append("Check logs for more details")
        
        return analysis
    
    def _extract_error_type(self, error: str) -> str:
        """Extract error type from message."""
        patterns = [
            r"(\w+Exception)",
            r"(\w+Error)",
            r"(\w+Failed)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, error)
            if match:
                return match.group(1)
        
        return "Unknown"
    
    def _format_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format analysis as readable text."""
        lines = [
            f"Error Type: {analysis['error_type']}",
            f"Severity: {analysis['severity'].upper()}",
            "",
            "Possible Causes:",
        ]
        
        for cause in analysis["possible_causes"]:
            lines.append(f"  - {cause}")
        
        lines.append("")
        lines.append("Suggestions:")
        
        for suggestion in analysis["suggestions"]:
            lines.append(f"  - {suggestion}")
        
        return "\n".join(lines)


class StackTraceTool(Tool):
    """Tool for parsing and analyzing stack traces."""
    
    def __init__(self):
        super().__init__()
        self._definition = ToolDefinition(
            name="stack_trace",
            description="Parse and analyze stack traces to identify root cause. "
                       "Use this to find where errors occurred.",
            category=ToolCategory.DEBUG,
            parameters=[
                create_tool_parameter(
                    name="stack_trace",
                    param_type="string",
                    description="Stack trace text",
                    required=True
                ),
                create_tool_parameter(
                    name="language",
                    param_type="string",
                    description="Programming language",
                    required=False,
                    default="java"
                )
            ],
            examples=[
                {
                    "params": {"stack_trace": "at com.app.Main.main(Main.java:10)"},
                    "description": "Analyze Java stack trace"
                }
            ],
            tags=["debug", "stack", "trace", "analysis"]
        )
    
    @property
    def definition(self) -> ToolDefinition:
        return self._definition
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute stack trace analysis."""
        stack_trace = kwargs.get("stack_trace", "")
        
        if not stack_trace:
            return ToolResult(success=False, error="stack_trace is required")
        
        analysis = self._parse_stack_trace(stack_trace)
        
        return ToolResult(
            success=True,
            output=self._format_analysis(analysis)
        )
    
    def _parse_stack_trace(self, trace: str) -> Dict[str, Any]:
        """Parse stack trace."""
        lines = trace.strip().split("\n")
        
        frames = []
        
        java_pattern = r"at\s+([\w.$]+)\(([^:]+):(\d+)\)"
        python_pattern = r'File\s+"([^"]+)",\s+line\s+(\d+)'
        
        for line in lines:
            match = re.search(java_pattern, line)
            if match:
                frames.append({
                    "method": match.group(1),
                    "file": match.group(2),
                    "line": match.group(3),
                    "source": line.strip()
                })
                continue
            
            match = re.search(python_pattern, line)
            if match:
                frames.append({
                    "file": match.group(1),
                    "line": match.group(2),
                    "source": line.strip()
                })
        
        root_cause = frames[-1] if frames else None
        
        return {
            "total_frames": len(frames),
            "frames": frames[:10],
            "root_cause": root_cause,
            "error_location": frames[0] if frames else None
        }
    
    def _format_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format analysis."""
        lines = [
            f"Total Frames: {analysis['total_frames']}",
            ""
        ]
        
        if analysis.get("error_location"):
            loc = analysis["error_location"]
            lines.append(f"Error Location:")
            lines.append(f"  File: {loc.get('file', 'unknown')}")
            lines.append(f"  Line: {loc.get('line', 'unknown')}")
            if "method" in loc:
                lines.append(f"  Method: {loc['method']}")
            lines.append("")
        
        if analysis.get("root_cause"):
            loc = analysis["root_cause"]
            lines.append(f"Root Cause:")
            lines.append(f"  File: {loc.get('file', 'unknown')}")
            lines.append(f"  Line: {loc.get('line', 'unknown')}")
            lines.append("")
        
        if analysis.get("frames"):
            lines.append("Stack Frames (first 10):")
            for i, frame in enumerate(analysis["frames"], 1):
                lines.append(f"  {i}. {frame.get('source', '')}")
        
        return "\n".join(lines)


class LogAnalyzerTool(Tool):
    """Tool for analyzing log files."""
    
    def __init__(self, base_path: Optional[str] = None):
        super().__init__()
        self._base_path = Path(base_path) if base_path else None
        self._definition = ToolDefinition(
            name="log_analyze",
            description="Analyze log files to find patterns, errors, and anomalies. "
                       "Use this to debug issues using logs.",
            category=ToolCategory.DEBUG,
            parameters=[
                create_tool_parameter(
                    name="file_path",
                    param_type="string",
                    description="Path to log file",
                    required=False
                ),
                create_tool_parameter(
                    name="pattern",
                    param_type="string",
                    description="Pattern to search for",
                    required=False
                ),
                create_tool_parameter(
                    name="level",
                    param_type="string",
                    description="Log level filter (ERROR, WARN, INFO, DEBUG)",
                    required=False
                ),
                create_tool_parameter(
                    name="lines",
                    param_type="integer",
                    description="Number of lines to analyze",
                    required=False,
                    default=100
                )
            ],
            examples=[
                {
                    "params": {"file_path": "app.log", "level": "ERROR"},
                    "description": "Find ERROR level logs"
                },
                {
                    "params": {"file_path": "app.log", "pattern": "Exception"},
                    "description": "Search for exceptions"
                }
            ],
            tags=["debug", "log", "analyze", "filter"]
        )
    
    @property
    def definition(self) -> ToolDefinition:
        return self._definition
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute log analysis."""
        file_path = kwargs.get("file_path")
        pattern = kwargs.get("pattern")
        level = kwargs.get("level")
        lines = kwargs.get("lines", 100)
        
        if file_path:
            path = self._base_path / file_path if self._base_path else Path(file_path)
            if not path.exists():
                return ToolResult(success=False, error=f"File not found: {file_path}")
            
            try:
                content = path.read_text(encoding="utf-8")
                log_lines = content.split("\n")[-lines:]
            except Exception as e:
                return ToolResult(success=False, error=f"Failed to read file: {e}")
        else:
            return ToolResult(success=False, error="file_path is required")
        
        analysis = self._analyze_logs(log_lines, pattern, level)
        
        return ToolResult(
            success=True,
            output=self._format_analysis(analysis)
        )
    
    def _analyze_logs(
        self,
        lines: List[str],
        pattern: Optional[str],
        level: Optional[str]
    ) -> Dict[str, Any]:
        """Analyze log lines."""
        filtered = []
        
        level_pattern = None
        if level:
            level_upper = level.upper()
            level_pattern = re.compile(rf"\b{level_upper}\b", re.IGNORECASE)
        
        for line in lines:
            if level_pattern and not level_pattern.search(line):
                continue
            
            if pattern and pattern.lower() not in line.lower():
                continue
            
            filtered.append(line)
        
        stats = {
            "error_count": sum(1 for l in lines if "ERROR" in l.upper()),
            "warn_count": sum(1 for l in lines if "WARN" in l.upper()),
            "info_count": sum(1 for l in lines if "INFO" in l.upper()),
            "debug_count": sum(1 for l in lines if "DEBUG" in l.upper()),
        }
        
        return {
            "total_lines": len(lines),
            "filtered_count": len(filtered),
            "filtered_lines": filtered[-20:],
            "stats": stats
        }
    
    def _format_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format analysis."""
        lines = [
            f"Total Lines: {analysis['total_lines']}",
            f"Filtered Lines: {analysis['filtered_count']}",
            "",
            "Statistics:",
            f"  ERROR: {analysis['stats']['error_count']}",
            f"  WARN:  {analysis['stats']['warn_count']}",
            f"  INFO:  {analysis['stats']['info_count']}",
            f"  DEBUG: {analysis['stats']['debug_count']}",
            "",
            "Filtered Lines (last 20):"
        ]
        
        for line in analysis["filtered_lines"]:
            lines.append(f"  {line[:120]}")
        
        return "\n".join(lines)


def get_all_debug_tools(base_path: Optional[str] = None):
    """Get all debug tools."""
    return [
        ErrorAnalysisTool(),
        StackTraceTool(),
        LogAnalyzerTool(base_path),
    ]
