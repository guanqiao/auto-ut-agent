"""Context Compaction - 智能上下文压缩

参考 OpenCode 的 Auto Compact 机制：
当 Token 使用达到阈值时，自动压缩历史对话
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import json
import logging
import time

logger = logging.getLogger(__name__)


class CompactionStrategy(Enum):
    """压缩策略"""
    SUMMARIZE = "summarize"          # 摘要压缩
    EXTRACT_KEY = "extract_key"      # 提取关键信息
    HYBRID = "hybrid"                # 混合策略


@dataclass
class CompactedContext:
    """压缩后的上下文"""
    # 核心摘要
    summary: str = ""
    
    # 任务状态
    completed_tasks: List[str] = field(default_factory=list)
    current_focus: str = ""
    pending_tasks: List[str] = field(default_factory=list)
    
    # 关键决策
    key_decisions: List[Dict[str, str]] = field(default_factory=list)
    
    # 活跃文件
    active_files: List[str] = field(default_factory=list)
    
    # 重要代码片段
    code_snippets: List[Dict[str, str]] = field(default_factory=list)
    
    # 错误和修复记录
    error_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # 元数据
    original_token_count: int = 0
    compacted_token_count: int = 0
    compaction_ratio: float = 0.0
    timestamp: float = field(default_factory=time.time)
    strategy_used: CompactionStrategy = CompactionStrategy.SUMMARIZE
    
    def get_compression_ratio(self) -> float:
        """获取压缩率"""
        if self.original_token_count == 0:
            return 0.0
        return 1 - (self.compacted_token_count / self.original_token_count)


@dataclass
class CompactionEvent:
    """压缩事件记录"""
    timestamp: float
    trigger_reason: str
    original_tokens: int
    compacted_tokens: int
    strategy: CompactionStrategy


class ContextCompactor:
    """上下文压缩器
    
    参考 OpenCode 的 Auto Compact 机制：
    当 Token 使用达到阈值时，自动压缩历史对话
    """
    
    def __init__(
        self,
        llm_client: Any,
        threshold: float = 0.85,
        target_ratio: float = 0.3,
        min_tokens_to_compact: int = 10000
    ):
        self.llm = llm_client
        self.threshold = threshold  # 触发压缩的阈值（相对于最大 token 数）
        self.target_ratio = target_ratio  # 压缩后的目标比例
        self.min_tokens_to_compact = min_tokens_to_compact  # 最小压缩 token 数
        self._compaction_history: List[CompactionEvent] = []
    
    def should_compact(
        self,
        current_tokens: int,
        max_tokens: int
    ) -> Tuple[bool, str]:
        """判断是否需要压缩
        
        Returns:
            Tuple[bool, str]: (是否需要压缩, 原因)
        """
        if current_tokens < self.min_tokens_to_compact:
            return False, f"Token count {current_tokens} below minimum {self.min_tokens_to_compact}"
        
        ratio = current_tokens / max_tokens
        if ratio >= self.threshold:
            return True, f"Token usage {ratio:.1%} exceeds threshold {self.threshold:.1%}"
        
        return False, f"Token usage {ratio:.1%} below threshold {self.threshold:.1%}"
    
    async def compact(
        self,
        conversation_history: List[Dict[str, Any]],
        current_task: Optional[str] = None,
        strategy: CompactionStrategy = CompactionStrategy.HYBRID
    ) -> CompactedContext:
        """压缩对话历史
        
        Args:
            conversation_history: 对话历史
            current_task: 当前任务
            strategy: 压缩策略
            
        Returns:
            CompactedContext: 压缩后的上下文
        """
        original_tokens = self._estimate_tokens(conversation_history)
        
        logger.info(f"Compacting context: {len(conversation_history)} messages, "
                   f"~{original_tokens} tokens, strategy={strategy.value}")
        
        try:
            if strategy == CompactionStrategy.SUMMARIZE:
                compacted = await self._summarize_compact(
                    conversation_history, current_task
                )
            elif strategy == CompactionStrategy.EXTRACT_KEY:
                compacted = await self._extract_key_compact(
                    conversation_history, current_task
                )
            else:  # HYBRID
                compacted = await self._hybrid_compact(
                    conversation_history, current_task
                )
            
            # 计算压缩率
            compacted.original_token_count = original_tokens
            compacted.compacted_token_count = self._estimate_tokens(compacted)
            compacted.strategy_used = strategy
            
            # 记录事件
            self._compaction_history.append(CompactionEvent(
                timestamp=time.time(),
                trigger_reason="threshold_exceeded",
                original_tokens=original_tokens,
                compacted_tokens=compacted.compacted_token_count,
                strategy=strategy
            ))
            
            logger.info(f"Context compacted: {original_tokens} -> "
                       f"{compacted.compacted_token_count} tokens "
                       f"({compacted.get_compression_ratio():.1%} reduction)")
            
            return compacted
            
        except Exception as e:
            logger.error(f"Context compaction failed: {e}")
            # 返回最小化的上下文
            return CompactedContext(
                summary="Context compaction failed, using minimal context",
                current_focus=current_task or "",
                original_token_count=original_tokens,
                compacted_token_count=100  # 估算
            )
    
    async def _summarize_compact(
        self,
        conversation_history: List[Dict[str, Any]],
        current_task: Optional[str]
    ) -> CompactedContext:
        """使用摘要策略压缩"""
        # 截取最近的历史（避免超出 LLM 上下文）
        recent_history = conversation_history[-20:] if len(conversation_history) > 20 else conversation_history
        
        prompt = f"""请将以下对话历史压缩为结构化摘要：

当前任务：{current_task or 'Unknown'}

对话历史：
{json.dumps(recent_history, indent=2, ensure_ascii=False)[:6000]}

请提取并返回 JSON 格式：
{{
    "summary": "整体任务摘要（200字以内）",
    "completed_tasks": ["已完成的任务1", "任务2"],
    "current_focus": "当前正在进行的工作",
    "pending_tasks": ["待办事项1", "事项2"],
    "key_decisions": [
        {{"decision": "决策内容", "rationale": "决策理由", "timestamp": "时间"}}
    ],
    "active_files": ["活跃文件路径1", "路径2"],
    "code_snippets": [
        {{"file": "文件路径", "snippet": "代码片段", "purpose": "用途"}}
    ],
    "error_history": [
        {{"error": "错误描述", "resolution": "解决方案", "file": "相关文件"}}
    ]
}}

要求：
1. 保留关键决策和重要上下文
2. 记录已完成的任务和待办事项
3. 保留活跃文件和关键代码片段
4. 记录错误和修复历史"""
        
        response = await self.llm.generate(prompt)
        data = json.loads(response)
        
        return CompactedContext(
            summary=data.get('summary', ''),
            completed_tasks=data.get('completed_tasks', []),
            current_focus=data.get('current_focus', ''),
            pending_tasks=data.get('pending_tasks', []),
            key_decisions=data.get('key_decisions', []),
            active_files=data.get('active_files', []),
            code_snippets=data.get('code_snippets', []),
            error_history=data.get('error_history', [])
        )
    
    async def _extract_key_compact(
        self,
        conversation_history: List[Dict[str, Any]],
        current_task: Optional[str]
    ) -> CompactedContext:
        """使用关键信息提取策略压缩"""
        # 分析对话历史，提取关键信息
        completed = []
        pending = []
        decisions = []
        errors = []
        files = set()
        
        for msg in conversation_history:
            content = msg.get('content', '')
            
            # 提取完成的任务
            if 'completed' in content.lower() or 'finished' in content.lower():
                completed.append(content[:100])
            
            # 提取待办
            if 'todo' in content.lower() or 'pending' in content.lower():
                pending.append(content[:100])
            
            # 提取决策
            if 'decided' in content.lower() or 'decision' in content.lower():
                decisions.append({
                    'decision': content[:200],
                    'rationale': '',
                    'timestamp': msg.get('timestamp', '')
                })
            
            # 提取错误
            if 'error' in content.lower() or 'exception' in content.lower():
                errors.append({
                    'error': content[:200],
                    'resolution': '',
                    'file': ''
                })
            
            # 提取文件
            if 'file' in content.lower() or '.java' in content:
                import re
                file_matches = re.findall(r'[\w/]+\.\w+', content)
                files.update(file_matches[:5])
        
        # 生成摘要
        summary_prompt = f"""基于以下信息生成任务摘要：

当前任务：{current_task or 'Unknown'}
已完成：{len(completed)} 项
待办：{len(pending)} 项
关键决策：{len(decisions)} 项
错误记录：{len(errors)} 项

请生成简洁的摘要（100字以内）。"""
        
        summary = await self.llm.generate(summary_prompt)
        
        return CompactedContext(
            summary=summary.strip(),
            completed_tasks=completed[-10:],  # 保留最近10个
            current_focus=current_task or '',
            pending_tasks=pending[-10:],
            key_decisions=decisions[-5:],
            active_files=list(files)[:10],
            error_history=errors[-5:]
        )
    
    async def _hybrid_compact(
        self,
        conversation_history: List[Dict[str, Any]],
        current_task: Optional[str]
    ) -> CompactedContext:
        """使用混合策略压缩"""
        # 分割历史：最近的使用摘要，早期的使用关键提取
        split_point = len(conversation_history) // 2
        
        # 早期历史使用关键提取
        early_compact = await self._extract_key_compact(
            conversation_history[:split_point], 
            None
        )
        
        # 近期历史使用摘要
        recent_compact = await self._summarize_compact(
            conversation_history[split_point:],
            current_task
        )
        
        # 合并结果
        return CompactedContext(
            summary=f"{early_compact.summary}\n\nRecent: {recent_compact.summary}",
            completed_tasks=early_compact.completed_tasks + recent_compact.completed_tasks,
            current_focus=recent_compact.current_focus,
            pending_tasks=recent_compact.pending_tasks,
            key_decisions=early_compact.key_decisions + recent_compact.key_decisions,
            active_files=list(set(early_compact.active_files + recent_compact.active_files)),
            code_snippets=recent_compact.code_snippets,
            error_history=early_compact.error_history + recent_compact.error_history
        )
    
    def _estimate_tokens(self, obj: Any) -> int:
        """估算 Token 数量
        
        粗略估算：1 token ≈ 4 字符（英文）
        """
        if isinstance(obj, CompactedContext):
            # 估算 CompactedContext 的 token 数
            text = f"{obj.summary} {' '.join(obj.completed_tasks)} "
            text += f"{obj.current_focus} {' '.join(obj.pending_tasks)} "
            text += json.dumps(obj.key_decisions, ensure_ascii=False)
            text += ' '.join(obj.active_files)
            return len(text) // 4
        
        text = json.dumps(obj, ensure_ascii=False)
        return len(text) // 4
    
    def format_for_prompt(self, compacted: CompactedContext) -> str:
        """格式化为 Prompt 可用的上下文"""
        sections = []
        
        # 摘要
        if compacted.summary:
            sections.append(f"[任务摘要]\n{compacted.summary}")
        
        # 已完成
        if compacted.completed_tasks:
            tasks_text = '\n'.join(f"  ✓ {t}" for t in compacted.completed_tasks[-5:])
            sections.append(f"[已完成]\n{tasks_text}")
        
        # 当前焦点
        if compacted.current_focus:
            sections.append(f"[当前焦点]\n  → {compacted.current_focus}")
        
        # 待办事项
        if compacted.pending_tasks:
            tasks_text = '\n'.join(f"  □ {t}" for t in compacted.pending_tasks[:5])
            sections.append(f"[待办事项]\n{tasks_text}")
        
        # 关键决策
        if compacted.key_decisions:
            decisions_text = '\n'.join(
                f"  • {d.get('decision', '')[:80]}"
                for d in compacted.key_decisions[-3:]
            )
            sections.append(f"[关键决策]\n{decisions_text}")
        
        # 活跃文件
        if compacted.active_files:
            files_text = '\n'.join(f"  📄 {f}" for f in compacted.active_files[:5])
            sections.append(f"[活跃文件]\n{files_text}")
        
        # 错误历史
        if compacted.error_history:
            errors_text = '\n'.join(
                f"  ⚠ {e.get('error', '')[:60]}..."
                for e in compacted.error_history[-3:]
            )
            sections.append(f"[错误记录]\n{errors_text}")
        
        return '\n\n'.join(sections)
    
    def get_compaction_history(self) -> List[CompactionEvent]:
        """获取压缩历史"""
        return self._compaction_history.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self._compaction_history:
            return {
                'total_compactions': 0,
                'average_compression_ratio': 0.0,
                'total_tokens_saved': 0
            }
        
        total_original = sum(e.original_tokens for e in self._compaction_history)
        total_compacted = sum(e.compacted_tokens for e in self._compaction_history)
        total_saved = total_original - total_compacted
        
        return {
            'total_compactions': len(self._compaction_history),
            'average_compression_ratio': total_saved / total_original if total_original > 0 else 0,
            'total_tokens_saved': total_saved,
            'average_original_tokens': total_original / len(self._compaction_history),
            'average_compacted_tokens': total_compacted / len(self._compaction_history)
        }


class AutoCompactManager:
    """自动压缩管理器
    
    自动监控 token 使用情况，在达到阈值时触发压缩
    """
    
    def __init__(
        self,
        llm_client: Any,
        max_tokens: int = 128000,  # 默认 128k 上下文
        threshold: float = 0.85,
        enable_auto_compact: bool = True
    ):
        self.compactor = ContextCompactor(llm_client, threshold)
        self.max_tokens = max_tokens
        self.enable_auto_compact = enable_auto_compact
        self._compacted_contexts: List[CompactedContext] = []
        self._current_token_count = 0
    
    async def check_and_compact(
        self,
        conversation_history: List[Dict[str, Any]],
        current_tokens: Optional[int] = None,
        current_task: Optional[str] = None
    ) -> Optional[CompactedContext]:
        """检查并执行压缩
        
        Args:
            conversation_history: 对话历史
            current_tokens: 当前 token 数（如果为 None 则自动估算）
            current_task: 当前任务
            
        Returns:
            Optional[CompactedContext]: 压缩后的上下文，如果不需要压缩则返回 None
        """
        if not self.enable_auto_compact:
            return None
        
        # 估算当前 token 数
        if current_tokens is None:
            current_tokens = self.compactor._estimate_tokens(conversation_history)
        
        self._current_token_count = current_tokens
        
        # 检查是否需要压缩
        should_compact, reason = self.compactor.should_compact(
            current_tokens, self.max_tokens
        )
        
        if not should_compact:
            logger.debug(f"No compaction needed: {reason}")
            return None
        
        logger.info(f"Triggering auto-compaction: {reason}")
        
        # 执行压缩
        compacted = await self.compactor.compact(
            conversation_history, current_task
        )
        
        self._compacted_contexts.append(compacted)
        
        return compacted
    
    def get_current_token_count(self) -> int:
        """获取当前 token 数"""
        return self._current_token_count
    
    def get_token_usage_ratio(self) -> float:
        """获取 token 使用率"""
        return self._current_token_count / self.max_tokens
    
    def get_all_compacted_contexts(self) -> List[CompactedContext]:
        """获取所有压缩后的上下文"""
        return self._compacted_contexts.copy()
    
    def get_compaction_stats(self) -> Dict[str, Any]:
        """获取压缩统计"""
        base_stats = self.compactor.get_stats()
        
        return {
            **base_stats,
            'max_tokens': self.max_tokens,
            'threshold': self.compactor.threshold,
            'current_token_count': self._current_token_count,
            'current_usage_ratio': self.get_token_usage_ratio(),
            'auto_compact_enabled': self.enable_auto_compact
        }
    
    def enable(self) -> None:
        """启用自动压缩"""
        self.enable_auto_compact = True
        logger.info("Auto-compaction enabled")
    
    def disable(self) -> None:
        """禁用自动压缩"""
        self.enable_auto_compact = False
        logger.info("Auto-compaction disabled")


# 便捷函数
def estimate_tokens(text: str) -> int:
    """估算文本的 token 数"""
    # 粗略估算：1 token ≈ 4 字符
    return len(text) // 4


def should_compact_simple(
    current_tokens: int,
    max_tokens: int = 128000,
    threshold: float = 0.85
) -> bool:
    """简单判断是否需要压缩"""
    return current_tokens / max_tokens >= threshold
