"""Specialized Subagents - 专业化子代理

参考 Claude Code 的 Subagents 设计：
- BashAgent: 专注命令行任务
- PlanAgent: 专注方案设计  
- ExploreAgent: 专注代码库探索
- TestGenAgent: 专注测试生成
"""

from .specialized import (
    SpecializedSubagent,
    SubagentResult,
    BashSubagent,
    PlanSubagent,
    ExploreSubagent,
    TestGenSubagent,
    SubagentRouter,
    create_default_router
)

__all__ = [
    'SpecializedSubagent',
    'SubagentResult', 
    'BashSubagent',
    'PlanSubagent',
    'ExploreSubagent',
    'TestGenSubagent',
    'SubagentRouter',
    'create_default_router'
]
