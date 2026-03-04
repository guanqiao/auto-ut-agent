"""Code analysis tools for structure understanding.

This module provides:
- CodeStructureTool: Analyze code structure
- DependencyGraphTool: Build dependency graphs
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .tool import (
    Tool,
    ToolCategory,
    ToolDefinition,
    ToolResult,
    create_tool_parameter,
)

logger = logging.getLogger(__name__)


class CodeStructureTool(Tool):
    """Tool for analyzing code structure."""
    
    def __init__(self, base_path: Optional[str] = None):
        """Initialize code structure tool.
        
        Args:
            base_path: Base path for file operations
        """
        super().__init__()
        self._base_path = Path(base_path) if base_path else Path.cwd()
        self._definition = ToolDefinition(
            name="code_structure",
            description="Analyze code structure - classes, methods, functions, imports. "
                       "Use this to understand codebase organization.",
            category=ToolCategory.ANALYSIS,
            parameters=[
                create_tool_parameter(
                    name="file_path",
                    param_type="string",
                    description="File or directory path to analyze",
                    required=True
                ),
                create_tool_parameter(
                    name="language",
                    param_type="string",
                    description="Programming language (python, java, js, etc.)",
                    required=False,
                    default="auto"
                )
            ],
            examples=[
                {
                    "params": {"file_path": "src/main/java/App.java"},
                    "description": "Analyze Java file structure"
                },
                {
                    "params": {"file_path": "pyutagent/agent/", "language": "python"},
                    "description": "Analyze Python directory"
                }
            ],
            tags=["analyze", "structure", "code", "classes", "methods"]
        )
    
    @property
    def definition(self) -> ToolDefinition:
        return self._definition
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute code structure analysis."""
        file_path = kwargs.get("file_path")
        language = kwargs.get("language", "auto")
        
        if not file_path:
            return ToolResult(success=False, error="file_path is required")
        
        path = self._base_path / file_path
        
        if not path.exists():
            return ToolResult(success=False, error=f"Path not found: {file_path}")
        
        if path.is_file():
            if language == "auto":
                language = self._detect_language(file_path)
            return await self._analyze_file(path, language)
        else:
            return await self._analyze_directory(path, language)
    
    def _detect_language(self, file_path: str) -> str:
        """Detect language from file extension."""
        ext = Path(file_path).suffix.lower()
        mapping = {
            ".py": "python",
            ".java": "java",
            ".js": "javascript",
            ".ts": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".cpp": "cpp",
            ".c": "c",
            ".cs": "csharp",
        }
        return mapping.get(ext, "unknown")
    
    async def _analyze_file(self, path: Path, language: str) -> ToolResult:
        """Analyze a single file."""
        try:
            content = path.read_text(encoding="utf-8")
            
            if language == "python":
                structure = self._analyze_python(content)
            elif language == "java":
                structure = self._analyze_java(content)
            elif language in ("javascript", "typescript"):
                structure = self._analyze_js(content)
            else:
                structure = {"message": f"Language {language} not fully supported"}
            
            result = {
                "file": str(path),
                "language": language,
                "lines": len(content.splitlines()),
                "structure": structure
            }
            
            return ToolResult(
                success=True,
                output=json.dumps(result, indent=2, ensure_ascii=False)
            )
            
        except Exception as e:
            logger.exception(f"[CodeStructureTool] Analysis failed: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _analyze_directory(self, path: Path, language: str) -> ToolResult:
        """Analyze a directory."""
        try:
            files = []
            
            for ext in [".py", ".java", ".js", ".ts", ".go"]:
                files.extend(path.rglob(f"*{ext}"))
            
            results = []
            for f in files[:50]:
                rel_path = f.relative_to(path)
                results.append({
                    "file": str(rel_path),
                    "type": f.suffix[1:]
                })
            
            result = {
                "directory": str(path),
                "files_found": len(results),
                "files": results[:20]
            }
            
            return ToolResult(
                success=True,
                output=json.dumps(result, indent=2)
            )
            
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    def _analyze_python(self, content: str) -> Dict[str, Any]:
        """Analyze Python code."""
        classes = re.findall(r'^class (\w+)', content, re.MULTILINE)
        functions = re.findall(r'^def (\w+)', content, re.MULTILINE)
        imports = re.findall(r'^import (\w+)', content, re.MULTILINE)
        from_imports = re.findall(r'^from (\w+) import', content, re.MULTILINE)
        
        return {
            "classes": classes[:20],
            "functions": functions[:30],
            "imports": list(set(imports + from_imports))[:20],
            "stats": {
                "class_count": len(classes),
                "function_count": len(functions),
                "import_count": len(imports) + len(from_imports)
            }
        }
    
    def _analyze_java(self, content: str) -> Dict[str, Any]:
        """Analyze Java code."""
        classes = re.findall(r'(?:public |private |protected )?class (\w+)', content)
        methods = re.findall(r'(?:public |private |protected )?(?:static )?\w+ \w+\((\w*)\)', content)
        packages = re.findall(r'^package ([^;]+);', content, re.MULTILINE)
        imports = re.findall(r'^import ([^;]+);', content, re.MULTILINE)
        
        return {
            "classes": classes[:20],
            "methods": list(set(methods))[:30],
            "package": packages[0] if packages else None,
            "imports": imports[:20],
            "stats": {
                "class_count": len(classes),
                "method_count": len(methods),
                "import_count": len(imports)
            }
        }
    
    def _analyze_js(self, content: str) -> Dict[str, Any]:
        """Analyze JavaScript/TypeScript code."""
        classes = re.findall(r'class (\w+)', content)
        functions = re.findall(r'(?:function (\w+)|const (\w+) = \((?:[^)]*)\) =>)', content)
        imports = re.findall(r"import .+ from ['\"]([^'\"]+)['\"]", content)
        
        funcs = [f[0] or f[1] for f in functions if f[0] or f[1]]
        
        return {
            "classes": classes[:20],
            "functions": funcs[:30],
            "imports": imports[:20],
            "stats": {
                "class_count": len(classes),
                "function_count": len(funcs),
                "import_count": len(imports)
            }
        }


class DependencyGraphTool(Tool):
    """Tool for building dependency graphs."""
    
    def __init__(self, base_path: Optional[str] = None):
        """Initialize dependency graph tool.
        
        Args:
            base_path: Base path for file operations
        """
        super().__init__()
        self._base_path = Path(base_path) if base_path else Path.cwd()
        self._definition = ToolDefinition(
            name="dependency_graph",
            description="Build dependency graph between files/modules. "
                       "Use this to understand code relationships.",
            category=ToolCategory.ANALYSIS,
            parameters=[
                create_tool_parameter(
                    name="root_path",
                    param_type="string",
                    description="Root path to analyze",
                    required=True
                ),
                create_tool_parameter(
                    name="depth",
                    param_type="integer",
                    description="Maximum depth to traverse",
                    required=False,
                    default=3
                )
            ],
            examples=[
                {
                    "params": {"root_path": "pyutagent/", "depth": 2},
                    "description": "Build dependency graph"
                }
            ],
            tags=["dependencies", "graph", "imports", "modules"]
        )
    
    @property
    def definition(self) -> ToolDefinition:
        return self._definition
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute dependency graph building."""
        root_path = kwargs.get("root_path")
        depth = kwargs.get("depth", 3)
        
        if not root_path:
            return ToolResult(success=False, error="root_path is required")
        
        path = self._base_path / root_path
        
        if not path.exists():
            return ToolResult(success=False, error=f"Path not found: {root_path}")
        
        try:
            dependencies = await self._build_graph(path, depth)
            
            return ToolResult(
                success=True,
                output=json.dumps(dependencies, indent=2)
            )
            
        except Exception as e:
            logger.exception(f"[DependencyGraphTool] Failed: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _build_graph(self, path: Path, depth: int) -> Dict[str, Any]:
        """Build dependency graph."""
        graph = {
            "root": str(path),
            "nodes": [],
            "edges": []
        }
        
        node_id = 0
        node_map = {}
        
        if path.is_file():
            files = [path]
        else:
            files = list(path.rglob("*.py"))[:100]
        
        for f in files:
            rel_path = str(f.relative_to(path))
            file_id = f"node_{node_id}"
            node_id += 1
            
            node_map[rel_path] = file_id
            
            graph["nodes"].append({
                "id": file_id,
                "label": rel_path,
                "type": f.suffix[1:]
            })
            
            try:
                content = f.read_text(encoding="utf-8")
                imports = re.findall(r'^import ([^;]+)|^from ([^ ]+)', content, re.MULTILINE)
                
                for imp in imports:
                    mod = imp[0] or imp[1]
                    target = mod.split('.')[0]
                    
                    graph["edges"].append({
                        "from": file_id,
                        "to": target,
                        "label": mod
                    })
                    
            except Exception:
                pass
        
        return graph


def get_all_analysis_tools(base_path: Optional[str] = None) -> List[Tool]:
    """Get all analysis tools.
    
    Args:
        base_path: Base path for operations
    
    Returns:
        List of tool instances
    """
    return [
        CodeStructureTool(base_path),
        DependencyGraphTool(base_path),
    ]
