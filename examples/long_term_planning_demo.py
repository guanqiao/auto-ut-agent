"""长期规划能力演示脚本

演示上下文压缩、自动压缩管理器和检查点机制。
参考 OpenCode 的 Auto Compact 机制。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
from typing import Dict, Any, List
from pathlib import Path

from pyutagent.core import (
    ContextCompactor,
    AutoCompactManager,
    CompactedContext,
    CompactionStrategy,
    CompactionEvent,
    Checkpoint,
    CheckpointMetadata,
    CheckpointManager,
)


class MockLLMClient:
    """模拟 LLM 客户端"""
    
    async def generate(self, prompt: str) -> str:
        """根据提示生成响应"""
        if "摘要" in prompt or "summarize" in prompt.lower():
            return json.dumps({
                "summary": "正在实现用户管理模块，已完成用户创建和查询功能，待办：添加邮箱验证和手机号支持。",
                "completed_tasks": [
                    "实现 User 实体类",
                    "实现 UserRepository 数据访问层",
                    "实现 UserService 业务逻辑层",
                    "编写单元测试"
                ],
                "current_focus": "添加手机号字段到 User 类",
                "pending_tasks": [
                    "更新数据库表结构",
                    "修改 UserService.createUser 方法",
                    "添加手机号格式验证",
                    "更新单元测试"
                ],
                "key_decisions": [
                    {
                        "decision": "使用 String 类型存储手机号",
                        "rationale": "支持国际号码格式",
                        "timestamp": "2024-01-15T10:30:00"
                    },
                    {
                        "decision": "在应用层进行格式验证",
                        "rationale": "保持数据库层的简单性",
                        "timestamp": "2024-01-15T11:00:00"
                    }
                ],
                "active_files": [
                    "src/main/java/com/example/model/User.java",
                    "src/main/java/com/example/service/UserService.java",
                    "src/test/java/com/example/service/UserServiceTest.java"
                ],
                "code_snippets": [
                    {
                        "file": "User.java",
                        "snippet": "private String phone;",
                        "purpose": "新增手机号字段"
                    }
                ],
                "error_history": [
                    {
                        "error": "NullPointerException in UserService.createUser",
                        "resolution": "添加空值检查",
                        "file": "UserService.java"
                    }
                ]
            })
        return json.dumps({"result": "success"})


def demo_compaction_strategy():
    """演示压缩策略"""
    print("\n" + "="*60)
    print("演示 1: 压缩策略 (Compaction Strategy)")
    print("="*60)
    
    print(f"\n可用策略:")
    for strategy in CompactionStrategy:
        print(f"  • {strategy.value}")
    
    print(f"\n策略说明:")
    print(f"  SUMMARIZE: 使用 LLM 生成摘要")
    print(f"  EXTRACT_KEY: 提取关键信息（任务、决策、错误）")
    print(f"  HYBRID: 混合策略（早期用提取，近期用摘要）")


def demo_compacted_context():
    """演示压缩后的上下文"""
    print("\n" + "="*60)
    print("演示 2: 压缩后的上下文 (Compacted Context)")
    print("="*60)
    
    context = CompactedContext(
        summary="正在实现用户管理模块，已完成用户创建和查询功能",
        completed_tasks=[
            "实现 User 实体类",
            "实现 UserRepository",
            "实现 UserService"
        ],
        current_focus="添加手机号字段",
        pending_tasks=[
            "更新数据库表结构",
            "修改 createUser 方法",
            "添加格式验证"
        ],
        key_decisions=[
            {
                "decision": "使用 String 类型存储手机号",
                "rationale": "支持国际号码格式"
            }
        ],
        active_files=[
            "User.java",
            "UserService.java",
            "UserServiceTest.java"
        ],
        code_snippets=[
            {
                "file": "User.java",
                "snippet": "private String phone;",
                "purpose": "新增手机号字段"
            }
        ],
        error_history=[
            {
                "error": "NullPointerException",
                "resolution": "添加空值检查",
                "file": "UserService.java"
            }
        ],
        original_token_count=50000,
        compacted_token_count=2000,
        strategy_used=CompactionStrategy.HYBRID
    )
    
    print(f"\n压缩上下文内容:")
    print(f"  摘要: {context.summary}")
    print(f"  已完成任务: {len(context.completed_tasks)} 项")
    print(f"  待办任务: {len(context.pending_tasks)} 项")
    print(f"  关键决策: {len(context.key_decisions)} 项")
    print(f"  活跃文件: {len(context.active_files)} 个")
    print(f"  代码片段: {len(context.code_snippets)} 个")
    print(f"  错误记录: {len(context.error_history)} 个")
    
    print(f"\n压缩统计:")
    print(f"  原始 Token: {context.original_token_count}")
    print(f"  压缩后 Token: {context.compacted_token_count}")
    print(f"  压缩率: {context.get_compression_ratio():.1%}")
    print(f"  使用策略: {context.strategy_used.value}")


def demo_context_compactor():
    """演示上下文压缩器"""
    print("\n" + "="*60)
    print("演示 3: 上下文压缩器 (Context Compactor)")
    print("="*60)
    
    llm = MockLLMClient()
    compactor = ContextCompactor(
        llm_client=llm,
        threshold=0.85,
        target_ratio=0.3,
        min_tokens_to_compact=10000
    )
    
    # 测试是否需要压缩
    test_cases = [
        (5000, 128000, "Token 数低于最小阈值"),
        (100000, 128000, "Token 使用率 78%"),
        (110000, 128000, "Token 使用率 86%"),
    ]
    
    print(f"\n压缩判断测试:")
    for current, max_tokens, description in test_cases:
        should_compact, reason = compactor.should_compact(current, max_tokens)
        status = "✅ 需要压缩" if should_compact else "⏸️  不需要"
        print(f"  {status} - {description}")
        print(f"      原因: {reason}")


def demo_token_estimation():
    """演示 Token 估算"""
    print("\n" + "="*60)
    print("演示 4: Token 估算")
    print("="*60)
    
    from pyutagent.core.context_compactor import estimate_tokens
    
    test_texts = [
        ("短文本", "Hello world"),
        ("中等文本", "public class UserService { private UserRepository repository; }"),
        ("长文本", "public class UserService {\n" + "    private UserRepository repository;\n" * 50 + "}"),
    ]
    
    print(f"\nToken 估算示例:")
    for name, text in test_texts:
        tokens = estimate_tokens(text)
        chars = len(text)
        print(f"  {name}:")
        print(f"    字符数: {chars}")
        print(f"    估算 Token: {tokens}")
        print(f"    比例: {tokens/chars:.2f} (约 1:4)")


def demo_auto_compact_manager():
    """演示自动压缩管理器"""
    print("\n" + "="*60)
    print("演示 5: 自动压缩管理器 (Auto Compact Manager)")
    print("="*60)
    
    llm = MockLLMClient()
    manager = AutoCompactManager(
        llm_client=llm,
        max_tokens=128000,
        threshold=0.85,
        enable_auto_compact=True
    )
    
    print(f"\n管理器配置:")
    print(f"  最大 Token: {manager.max_tokens}")
    print(f"  压缩阈值: {manager.compactor.threshold}")
    print(f"  自动压缩: {'启用' if manager.enable_auto_compact else '禁用'}")
    
    # 模拟 token 使用情况
    print(f"\nToken 使用监控:")
    usage_scenarios = [
        (50000, "初始阶段"),
        (80000, "开发进行中"),
        (100000, "接近阈值"),
        (110000, "超过阈值"),
    ]
    
    for tokens, description in usage_scenarios:
        ratio = tokens / manager.max_tokens
        status = "🟢 正常" if ratio < 0.85 else "🔴 需要压缩"
        print(f"  {status} {description}: {tokens} tokens ({ratio:.1%})")
    
    print(f"\n压缩统计:")
    stats = manager.get_compaction_stats()
    print(f"  最大 Token: {stats['max_tokens']}")
    print(f"  阈值: {stats['threshold']}")
    print(f"  自动压缩启用: {stats['auto_compact_enabled']}")


def demo_checkpoint():
    """演示检查点"""
    print("\n" + "="*60)
    print("演示 6: 检查点 (Checkpoint)")
    print("="*60)
    
    checkpoint = Checkpoint(
        id="abc12345",
        created_at="2024-01-15T14:30:00",
        step="generate_tests",
        iteration=3,
        state={
            "target_class": "UserService",
            "generated_tests": ["testCreateUser", "testFindById"],
            "coverage": 0.75,
            "pending_fixes": ["添加空值检查"]
        },
        metadata={
            "agent_version": "1.0.0",
            "llm_model": "gpt-4"
        }
    )
    
    print(f"\n检查点信息:")
    print(f"  ID: {checkpoint.id}")
    print(f"  创建时间: {checkpoint.created_at}")
    print(f"  当前步骤: {checkpoint.step}")
    print(f"  迭代次数: {checkpoint.iteration}")
    
    print(f"\n状态数据:")
    for key, value in checkpoint.state.items():
        print(f"  {key}: {value}")
    
    print(f"\n元数据:")
    for key, value in checkpoint.metadata.items():
        print(f"  {key}: {value}")
    
    # 转换为字典
    checkpoint_dict = checkpoint.to_dict()
    print(f"\n  字典表示:")
    print(f"    ID: {checkpoint_dict['id']}")
    print(f"    步骤: {checkpoint_dict['step']}")


def demo_checkpoint_manager():
    """演示检查点管理器"""
    print("\n" + "="*60)
    print("演示 7: 检查点管理器 (Checkpoint Manager)")
    print("="*60)
    
    # 使用临时目录
    temp_dir = Path("/tmp/checkpoints_demo")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        manager = CheckpointManager(
            persist_dir=str(temp_dir),
            max_checkpoints=5,
            auto_cleanup=True
        )
        
        print(f"\n检查点目录: {manager.persist_dir}")
        print(f"  最大检查点数: {manager.max_checkpoints}")
        print(f"  自动清理: {manager.auto_cleanup}")
        
        # 创建检查点
        print(f"\n创建检查点:")
        for i in range(3):
            cp_id = manager.save_checkpoint(
                step=f"step_{i+1}",
                iteration=i+1,
                state={
                    "progress": (i+1) * 33,
                    "completed": [f"task_{j+1}" for j in range(i+1)]
                },
                metadata={"version": "1.0.0"}
            )
            print(f"  ✓ 创建检查点: {cp_id}")
        
        # 获取最新检查点
        latest = manager.get_latest()
        if latest:
            print(f"\n最新检查点:")
            print(f"  ID: {latest.id}")
            print(f"  步骤: {latest.step}")
            print(f"  迭代: {latest.iteration}")
        
        # 获取统计
        metadata = manager.get_metadata()
        print(f"\n检查点统计:")
        print(f"  总数: {metadata.total_checkpoints}")
        print(f"  最新: {metadata.latest_checkpoint_id}")
        print(f"  最早: {metadata.earliest_checkpoint_id}")
        
        # 清理
        for cp_id in list(manager._checkpoints.keys()):
            manager._delete_checkpoint(cp_id)
        print(f"\n已清理所有检查点")
        
    finally:
        # 删除临时目录
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def demo_integration_with_agent():
    """演示与 Agent 集成"""
    print("\n" + "="*60)
    print("演示 8: 与 Agent 集成")
    print("="*60)
    
    print("""
长期规划能力与 Agent 集成:

1. 在 Agent 中启用上下文压缩
   ```python
   from pyutagent.core import AutoCompactManager
   
   class UniversalCodingAgent:
       def __init__(self):
           self.compact_manager = AutoCompactManager(
               llm_client=self.llm,
               max_tokens=128000,
               threshold=0.85
           )
       
       async def execute(self, task):
           # 检查是否需要压缩
           compacted = await self.compact_manager.check_and_compact(
               self.conversation_history,
               current_task=task
           )
           
           if compacted:
               # 使用压缩后的上下文
               context = compacted.format_for_prompt()
           else:
               # 使用原始上下文
               context = self.get_full_context()
           
           # 继续执行...
   ```

2. 检查点机制
   ```python
   from pyutagent.core import CheckpointManager
   
   class ResumableAgent:
       def __init__(self):
           self.checkpoint_manager = CheckpointManager()
       
       async def execute_with_checkpoints(self, task):
           # 创建检查点
           checkpoint_id = self.checkpoint_manager.save_checkpoint(
               step="analyze_target",
               iteration=1,
               state={"target_file": task.file_path}
           )
           
           try:
               # 执行可能失败的操作
               result = await self.risky_operation()
           except Exception:
               # 从检查点恢复
               checkpoint = self.checkpoint_manager.get_checkpoint(checkpoint_id)
               await self.restore_from_checkpoint(checkpoint)
               raise
   ```

3. 长任务处理
   ```python
   # 处理需要数小时的复杂任务
   async def long_running_task(self):
       for iteration in range(100):
           # 每 10 次迭代创建检查点
           if iteration % 10 == 0:
               self.checkpoint_manager.save_checkpoint(
                   step=f"iteration_{iteration}",
                   iteration=iteration,
                   state=self.get_current_state()
               )
           
           # 检查上下文大小
           compacted = await self.compact_manager.check_and_compact(
               self.history
           )
           
           if compacted:
               self.history = [compacted.format_for_prompt()]
           
           # 继续执行...
   ```

优势:
- 支持长周期任务（数小时）
- 自动管理上下文窗口
- 断点续传能力
- 降低 token 成本
""")


async def main():
    """主函数"""
    print("\n" + "="*60)
    print("长期规划能力演示")
    print("上下文压缩 + 检查点机制")
    print("="*60)
    
    demo_compaction_strategy()
    demo_compacted_context()
    demo_context_compactor()
    demo_token_estimation()
    demo_auto_compact_manager()
    demo_checkpoint()
    demo_checkpoint_manager()
    demo_integration_with_agent()
    
    print("\n" + "="*60)
    print("长期规划能力演示完成!")
    print("="*60)
    print("""
核心组件:
1. ContextCompactor - 上下文压缩器
   - SUMMARIZE: 摘要压缩
   - EXTRACT_KEY: 关键信息提取
   - HYBRID: 混合策略

2. AutoCompactManager - 自动压缩管理器
   - 自动监控 token 使用
   - 阈值触发压缩
   - 统计和监控

3. CheckpointManager - 检查点管理器
   - 状态持久化
   - 断点续传
   - 自动清理

使用场景:
- 长周期任务（数小时）
- 复杂重构操作
- 大型代码库分析
- 持续集成环境

参考: OpenCode Auto Compact 机制
""")


if __name__ == "__main__":
    asyncio.run(main())
