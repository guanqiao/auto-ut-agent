"""Specialized Subagents - 专业化子代理

参考 Claude Code 的 Subagents 设计：
- BashAgent: 专注命令行任务
- PlanAgent: 专注方案设计  
- ExploreAgent: 专注代码库探索
- TestGenAgent: 专注测试生成

测试生成专用Agent：
- TestDesignAgent: 测试设计专家
- TestImplementAgent: 测试实现专家
- TestReviewAgent: 测试审查专家
- TestFixAgent: 测试修复专家
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

from .test_design_agent import TestDesignAgent
from .test_implement_agent import TestImplementAgent
from .test_review_agent import TestReviewAgent
from .test_fix_agent import TestFixAgent

__all__ = [
    'SpecializedSubagent',
    'SubagentResult', 
    'BashSubagent',
    'PlanSubagent',
    'ExploreSubagent',
    'TestGenSubagent',
    'SubagentRouter',
    'create_default_router',
    'TestDesignAgent',
    'TestImplementAgent',
    'TestReviewAgent',
    'TestFixAgent',
]
