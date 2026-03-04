"""Agent Tool Service - 统一管理Agent工具注册和执行。

This module provides:
- AgentToolService: 统一的工具服务层
- 整合标准工具、专用工具和MCP工具
- 提供LLM可用的工具schema
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..tools.tool import Tool, ToolExecutor, ToolResult
from ..tools.tool_registry import ToolRegistry
from ..tools.standard_tools import (
    ReadTool,
    WriteTool,
    EditTool,
    GlobTool,
    GrepTool,
    BashTool,
)
from ..tools.git_tools import get_all_git_tools
from ..tools.search_tools import get_all_search_tools
from ..llm.client import LLMClient

logger = logging.getLogger(__name__)


class AgentToolService:
    """Agent工具服务 - 统一管理工具注册和执行。
    
    Features:
    - 统一注册标准工具和专用工具
    - 提供LLM可用的工具schema
    - 执行工具并返回结果
    - 支持工具缓存和状态管理
    """
    
    def __init__(
        self,
        project_path: str,
        base_path: Optional[str] = None,
        allowed_commands: Optional[List[str]] = None,
        timeout: int = 60
    ):
        """初始化Agent工具服务。
        
        Args:
            project_path: 项目路径
            base_path: 基础路径（用于相对路径解析）
            allowed_commands: 允许的shell命令前缀列表
            timeout: 命令超时时间（秒）
        """
        self.project_path = Path(project_path)
        self.base_path = Path(base_path) if base_path else self.project_path
        self.timeout = timeout
        
        self.registry = ToolRegistry()
        self.executor = ToolExecutor()
        
        self._llm_client: Optional[LLMClient] = None
        self._execution_history: List[Dict[str, Any]] = []
        
        self._register_standard_tools(allowed_commands)
        self._register_project_tools()
        
        logger.info(f"[AgentToolService] Initialized for project: {project_path}")
    
    def _register_standard_tools(self, allowed_commands: Optional[List[str]] = None):
        """注册标准工具。
        
        Args:
            allowed_commands: 允许的shell命令前缀
        """
        read_tool = ReadTool(str(self.base_path))
        self.registry.register(read_tool, tags=["file", "read", "view"])
        
        write_tool = WriteTool(str(self.base_path))
        self.registry.register(write_tool, tags=["file", "write", "create"])
        
        edit_tool = EditTool(str(self.base_path))
        self.registry.register(edit_tool, tags=["file", "edit", "modify"])
        
        glob_tool = GlobTool(str(self.base_path))
        self.registry.register(glob_tool, tags=["search", "find", "glob"])
        
        grep_tool = GrepTool(str(self.base_path))
        self.registry.register(grep_tool, tags=["search", "grep", "find"])
        
        bash_tool = BashTool(
            str(self.project_path),
            allowed_commands=allowed_commands,
            timeout=self.timeout
        )
        self.registry.register(bash_tool, tags=["command", "bash", "shell", "execute"])
        
        git_tools = get_all_git_tools(str(self.base_path))
        for tool in git_tools:
            self.registry.register(tool)
        
        search_tools = get_all_search_tools()
        for tool in search_tools:
            self.registry.register(tool)
        
        logger.info(f"[AgentToolService] Registered {len(git_tools)} git tools")
        logger.info(f"[AgentToolService] Registered {len(search_tools)} search tools")
        logger.info("[AgentToolService] Registered standard tools")
    
    def _register_project_tools(self):
        """注册项目专用工具（如Java解析、Maven等）。"""
        pass
    
    def set_llm_client(self, llm_client: LLMClient):
        """设置LLM客户端。
        
        Args:
            llm_client: LLM客户端实例
        """
        self._llm_client = llm_client
    
    def get_schemas_for_llm(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """获取LLM可用的工具schema。
        
        Args:
            force_refresh: 强制刷新缓存
        
        Returns:
            工具schema列表（OpenAI Function Calling格式）
        """
        return self.registry.get_schemas(force_refresh)
    
    def get_schemas_json(self, force_refresh: bool = False) -> str:
        """获取工具schema的JSON字符串。
        
        Args:
            force_refresh: 强制刷新缓存
        
        Returns:
            JSON格式的工具schema
        """
        return self.registry.get_schema_json(force_refresh)
    
    async def execute_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """执行工具并返回结果。
        
        Args:
            tool_name: 工具名称
            params: 工具参数
            context: 执行上下文
        
        Returns:
            ToolResult: 工具执行结果
        """
        tool = self.registry.get_or_none(tool_name)
        
        if tool is None:
            logger.error(f"[AgentToolService] Tool not found: {tool_name}")
            return ToolResult(
                success=False,
                error=f"Tool not found: {tool_name}"
            )
        
        result = await self.executor.execute_tool(tool, params, context)
        
        self._execution_history.append({
            "tool": tool_name,
            "params": params,
            "success": result.success,
            "output_type": type(result.output).__name__ if result.output else None,
            "error": result.error
        })
        
        return result
    
    async def execute_tool_sequence(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> List[ToolResult]:
        """顺序执行多个工具调用。
        
        Args:
            tool_calls: 工具调用列表 [{"tool": "xxx", "params": {...}}, ...]
        
        Returns:
            ToolResult列表
        """
        results = []
        
        for call in tool_calls:
            tool_name = call.get("tool")
            params = call.get("params", {})
            
            result = await self.execute_tool(tool_name, params)
            results.append(result)
            
            if not result.success:
                logger.warning(
                    f"[AgentToolService] Tool {tool_name} failed, stopping sequence"
                )
                break
        
        return results
    
    def list_available_tools(self, category: Optional[str] = None) -> List[str]:
        """列出可用的工具。
        
        Args:
            category: 可选的分类过滤
        
        Returns:
            工具名称列表
        """
        if category:
            from ..tools.tool import ToolCategory
            try:
                cat = ToolCategory[category.upper()]
                return self.registry.list_tools(cat)
            except KeyError:
                logger.warning(f"[AgentToolService] Unknown category: {category}")
        
        return self.registry.list_tools()
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """获取工具信息。
        
        Args:
            tool_name: 工具名称
        
        Returns:
            工具定义信息
        """
        tool = self.registry.get_or_none(tool_name)
        if tool:
            return tool.definition.to_dict()
        return None
    
    def search_tools(self, query: str) -> List[str]:
        """搜索工具。
        
        Args:
            query: 搜索关键词
        
        Returns:
            匹配的工具名称列表
        """
        return self.registry.search(query)
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """获取工具执行历史。
        
        Returns:
            执行历史列表
        """
        return self._execution_history.copy()
    
    def clear_history(self):
        """清除执行历史。"""
        self._execution_history.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取工具服务统计信息。
        
        Returns:
            统计信息字典
        """
        return {
            "total_tools": len(self.registry),
            "available_tools": self.list_available_tools(),
            "execution_count": len(self._execution_history),
            "success_count": sum(1 for h in self._execution_history if h.get("success")),
            "failure_count": sum(1 for h in self._execution_history if not h.get("success")),
            "registry_stats": self.registry.get_stats()
        }
    
    def add_tool(self, tool: Tool, tags: Optional[List[str]] = None):
        """添加自定义工具。
        
        Args:
            tool: 工具实例
            tags: 可选的标签列表
        """
        self.registry.register(tool, tags=tags)
        logger.info(f"[AgentToolService] Added custom tool: {tool.definition.name}")
    
    def remove_tool(self, tool_name: str) -> bool:
        """移除工具。
        
        Args:
            tool_name: 工具名称
        
        Returns:
            是否成功移除
        """
        result = self.registry.unregister(tool_name)
        if result:
            logger.info(f"[AgentToolService] Removed tool: {tool_name}")
        return result


def create_agent_tool_service(
    project_path: str,
    base_path: Optional[str] = None,
    allowed_commands: Optional[List[str]] = None,
    timeout: int = 60
) -> AgentToolService:
    """创建AgentToolService实例的便捷函数。
    
    Args:
        project_path: 项目路径
        base_path: 基础路径
        allowed_commands: 允许的shell命令前缀
        timeout: 命令超时时间
    
    Returns:
        AgentToolService实例
    """
    return AgentToolService(
        project_path=project_path,
        base_path=base_path,
        allowed_commands=allowed_commands,
        timeout=timeout
    )
