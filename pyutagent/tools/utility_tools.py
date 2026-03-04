"""Additional utility tools.

This module provides:
- FileWatcherTool: Watch files for changes
- DiffTool: Compare files/directories
- ProcessTool: Manage processes
"""

import asyncio
import logging
import subprocess
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


class FileWatcherTool(Tool):
    """Tool for watching file changes."""
    
    def __init__(self, base_path: Optional[str] = None):
        super().__init__()
        self._base_path = Path(base_path) if base_path else None
        self._watchers: Dict[str, Any] = {}
        self._definition = ToolDefinition(
            name="file_watcher",
            description="Watch files or directories for changes. "
                       "Use this to monitor file modifications.",
            category=ToolCategory.FILE,
            parameters=[
                create_tool_parameter(
                    name="path",
                    param_type="string",
                    description="Path to watch",
                    required=True
                ),
                create_tool_parameter(
                    name="pattern",
                    param_type="string",
                    description="File pattern to watch",
                    required=False,
                    default="*"
                ),
                create_tool_parameter(
                    name="timeout",
                    param_type="integer",
                    description="Watch timeout in seconds",
                    required=False,
                    default=5
                )
            ],
            examples=[
                {
                    "params": {"path": "src/", "timeout": 10},
                    "description": "Watch src directory for 10 seconds"
                }
            ],
            tags=["file", "watch", "monitor", "changes"]
        )
    
    @property
    def definition(self) -> ToolDefinition:
        return self._definition
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute file watching."""
        path = kwargs.get("path")
        pattern = kwargs.get("pattern", "*")
        timeout = kwargs.get("timeout", 5)
        
        if not path:
            return ToolResult(success=False, error="path is required")
        
        watch_path = self._base_path / path if self._base_path else Path(path)
        
        if not watch_path.exists():
            return ToolResult(success=False, error=f"Path not found: {path}")
        
        try:
            import time
            changes = []
            start_time = time.time()
            
            if watch_path.is_file():
                mtime_before = watch_path.stat().st_mtime
                await asyncio.sleep(timeout)
                mtime_after = watch_path.stat().st_mtime
                
                if mtime_after > mtime_before:
                    changes.append({
                        "type": "modified",
                        "path": str(watch_path)
                    })
            else:
                files_before = set(watch_path.rglob(pattern))
                await asyncio.sleep(timeout)
                files_after = set(watch_path.rglob(pattern))
                
                for f in files_after - files_before:
                    changes.append({"type": "added", "path": str(f)})
                
                for f in files_before - files_after:
                    changes.append({"type": "deleted", "path": str(f)})
                
                for f in files_before & files_after:
                    if f.stat().st_mtime > start_time:
                        changes.append({"type": "modified", "path": str(f)})
            
            output = f"Watched {watch_path} for {timeout}s\n"
            output += f"Changes detected: {len(changes)}\n"
            for c in changes[:10]:
                output += f"  - {c['type']}: {c['path']}\n"
            
            return ToolResult(success=True, output=output)
            
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class DiffTool(Tool):
    """Tool for comparing files."""
    
    def __init__(self, base_path: Optional[str] = None):
        super().__init__()
        self._base_path = Path(base_path) if base_path else None
        self._definition = ToolDefinition(
            name="diff",
            description="Compare two files or directories. "
                       "Use this to see differences between versions.",
            category=ToolCategory.FILE,
            parameters=[
                create_tool_parameter(
                    name="file1",
                    param_type="string",
                    description="First file path",
                    required=True
                ),
                create_tool_parameter(
                    name="file2",
                    param_type="string",
                    description="Second file path",
                    required=True
                ),
                create_tool_parameter(
                    name="context",
                    param_type="integer",
                    description="Lines of context",
                    required=False,
                    default=3
                )
            ],
            examples=[
                {
                    "params": {"file1": "old.java", "file2": "new.java"},
                    "description": "Compare two Java files"
                }
            ],
            tags=["diff", "compare", "file", "difference"]
        )
    
    @property
    def definition(self) -> ToolDefinition:
        return self._definition
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute diff."""
        file1 = kwargs.get("file1")
        file2 = kwargs.get("file2")
        context = kwargs.get("context", 3)
        
        if not file1 or not file2:
            return ToolResult(success=False, error="file1 and file2 are required")
        
        path1 = self._base_path / file1 if self._base_path else Path(file1)
        path2 = self._base_path / file2 if self._base_path else Path(file2)
        
        if not path1.exists():
            return ToolResult(success=False, error=f"File not found: {file1}")
        
        if not path2.exists():
            return ToolResult(success=False, error=f"File not found: {file2}")
        
        try:
            import difflib
            
            with open(path1, 'r', encoding='utf-8') as f1:
                lines1 = f1.readlines()
            
            with open(path2, 'r', encoding='utf-8') as f2:
                lines2 = f2.readlines()
            
            diff = list(difflib.unified_diff(
                lines1, lines2,
                fromfile=str(path1),
                tofile=str(path2),
                lineterm='',
                n=context
            ))
            
            if diff:
                output = '\n'.join(diff[:100])
                return ToolResult(success=True, output=output)
            else:
                return ToolResult(success=True, output="Files are identical")
                
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class ProcessTool(Tool):
    """Tool for managing processes."""
    
    def __init__(self):
        super().__init__()
        self._definition = ToolDefinition(
            name="process",
            description="Manage processes - list, kill, or get info. "
                       "Use this to monitor or control running processes.",
            category=ToolCategory.SYSTEM,
            parameters=[
                create_tool_parameter(
                    name="action",
                    param_type="string",
                    description="Action: list, kill, info",
                    required=True
                ),
                create_tool_parameter(
                    name="pid",
                    param_type="integer",
                    description="Process ID (for kill/info)",
                    required=False
                ),
                create_tool_parameter(
                    name="name",
                    param_type="string",
                    description="Process name pattern",
                    required=False
                )
            ],
            examples=[
                {
                    "params": {"action": "list"},
                    "description": "List running processes"
                },
                {
                    "params": {"action": "kill", "name": "java"},
                    "description": "Kill Java processes"
                }
            ],
            tags=["process", "system", "kill", "monitor"]
        )
    
    @property
    def definition(self) -> ToolDefinition:
        return self._definition
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute process management."""
        action = kwargs.get("action", "list")
        pid = kwargs.get("pid")
        name = kwargs.get("name")
        
        try:
            if action == "list":
                return await self._list_processes(name)
            elif action == "kill":
                return await self._kill_process(pid, name)
            elif action == "info":
                return await self._process_info(pid)
            else:
                return ToolResult(success=False, error=f"Unknown action: {action}")
                
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    async def _list_processes(self, name: Optional[str]) -> ToolResult:
        """List processes."""
        try:
            import psutil
            
            processes = []
            for p in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    pinfo = p.info
                    if name is None or name.lower() in pinfo['name'].lower():
                        processes.append(pinfo)
                except:
                    pass
            
            output = f"Found {len(processes)} processes:\n"
            for p in processes[:20]:
                output += f"  PID: {p['pid']}, Name: {p['name']}, CPU: {p['cpu_percent']:.1f}%, Mem: {p['memory_percent']:.1f}%\n"
            
            return ToolResult(success=True, output=output)
            
        except ImportError:
            return ToolResult(success=False, error="psutil not installed")
    
    async def _kill_process(self, pid: Optional[int], name: Optional[str]) -> ToolResult:
        """Kill process."""
        try:
            import psutil
            
            killed = 0
            
            if pid:
                try:
                    p = psutil.Process(pid)
                    p.terminate()
                    killed = 1
                except:
                    pass
            
            if name:
                for p in psutil.process_iter(['pid', 'name']):
                    try:
                        if name.lower() in p.info['name'].lower():
                            p.terminate()
                            killed += 1
                    except:
                        pass
            
            return ToolResult(success=True, output=f"Killed {killed} process(es)")
            
        except ImportError:
            return ToolResult(success=False, error="psutil not installed")
    
    async def _process_info(self, pid: int) -> ToolResult:
        """Get process info."""
        try:
            import psutil
            
            p = psutil.Process(pid)
            info = {
                "pid": p.pid,
                "name": p.name(),
                "status": p.status(),
                "cpu_percent": p.cpu_percent(),
                "memory_percent": p.memory_percent(),
                "cmdline": ' '.join(p.cmdline())
            }
            
            output = f"Process {pid}:\n"
            for k, v in info.items():
                output += f"  {k}: {v}\n"
            
            return ToolResult(success=True, output=output)
            
        except ImportError:
            return ToolResult(success=False, error="psutil not installed")
        except psutil.NoSuchProcess:
            return ToolResult(success=False, error=f"Process {pid} not found")


class EnvTool(Tool):
    """Tool for environment variables."""
    
    def __init__(self):
        super().__init__()
        self._definition = ToolDefinition(
            name="env",
            description="Get or set environment variables. "
                       "Use this to check system configuration.",
            category=ToolCategory.SYSTEM,
            parameters=[
                create_tool_parameter(
                    name="action",
                    param_type="string",
                    description="Action: get, set, list",
                    required=True
                ),
                create_tool_parameter(
                    name="key",
                    param_type="string",
                    description="Variable name",
                    required=False
                ),
                create_tool_parameter(
                    name="value",
                    param_type="string",
                    description="Variable value (for set)",
                    required=False
                )
            ],
            examples=[
                {
                    "params": {"action": "get", "key": "JAVA_HOME"},
                    "description": "Get JAVA_HOME"
                },
                {
                    "params": {"action": "list"},
                    "description": "List all env vars"
                }
            ],
            tags=["env", "environment", "system", "config"]
        )
    
    @property
    def definition(self) -> ToolDefinition:
        return self._definition
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute environment operation."""
        action = kwargs.get("action")
        key = kwargs.get("key")
        value = kwargs.get("value")
        
        import os
        
        try:
            if action == "get":
                val = os.environ.get(key, "")
                return ToolResult(success=True, output=f"{key}={val}")
            
            elif action == "set":
                if key and value:
                    os.environ[key] = value
                    return ToolResult(success=True, output=f"Set {key}={value}")
                return ToolResult(success=False, error="key and value required")
            
            elif action == "list":
                output = "Environment variables:\n"
                for k, v in sorted(os.environ.items()):
                    if 'PATH' in k.upper() or 'HOME' in k.upper() or 'JAVA' in k.upper():
                        output += f"  {k}={v[:80]}...\n"
                    else:
                        output += f"  {k}={v[:50]}\n"
                return ToolResult(success=True, output=output)
            
            else:
                return ToolResult(success=False, error=f"Unknown action: {action}")
                
        except Exception as e:
            return ToolResult(success=False, error=str(e))


def get_all_utility_tools(base_path: Optional[str] = None):
    """Get all utility tools."""
    return [
        FileWatcherTool(base_path),
        DiffTool(base_path),
        ProcessTool(),
        EnvTool(),
    ]
