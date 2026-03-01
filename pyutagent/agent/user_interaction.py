"""
P3.4 用户交互处理器 (User Interaction Handler)

功能：
1. 修复建议展示 - 以清晰易懂的方式展示修复方案
2. 交互式确认 - 让用户参与关键决策
3. 用户偏好学习 - 记录用户选择模式，优化后续交互
"""

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
import hashlib


class InteractionType(Enum):
    """交互类型"""
    REPAIR_CONFIRMATION = auto()      # 修复确认
    STRATEGY_SELECTION = auto()       # 策略选择
    PARAMETER_ADJUSTMENT = auto()     # 参数调整
    MULTI_CHOICE = auto()             # 多选
    INFORMATION_DISPLAY = auto()      # 信息展示


class UserChoice(Enum):
    """用户选择"""
    ACCEPT = "accept"
    REJECT = "reject"
    MODIFY = "modify"
    SKIP = "skip"
    ASK_FOR_HELP = "ask_for_help"


@dataclass
class RepairSuggestion:
    """修复建议"""
    suggestion_id: str
    title: str
    description: str
    code_before: Optional[str]
    code_after: Optional[str]
    explanation: str
    confidence: float  # 0-1
    estimated_impact: str  # "high", "medium", "low"
    affected_files: List[str]
    side_effects: List[str]
    alternatives: List[Dict[str, str]]  # 替代方案
    metadata: Dict[str, Any]


@dataclass
class UserPreference:
    """用户偏好"""
    preference_id: str
    category: str  # "repair_style", "interaction_frequency", "detail_level"
    value: Any
    confidence: float  # 0-1，基于历史数据的可信度
    sample_count: int  # 用于推断此偏好的样本数
    last_updated: datetime


@dataclass
class InteractionRecord:
    """交互记录"""
    record_id: str
    interaction_type: InteractionType
    context_hash: str  # 用于识别相似场景
    suggestion: Optional[RepairSuggestion]
    user_choice: UserChoice
    user_feedback: Optional[str]
    timestamp: datetime
    session_id: str
    metadata: Dict[str, Any]


@dataclass
class DisplayConfig:
    """显示配置"""
    show_line_numbers: bool = True
    show_diff_highlight: bool = True
    max_code_lines: int = 50
    show_confidence_score: bool = True
    show_alternatives: bool = True
    detail_level: str = "normal"  # "minimal", "normal", "verbose"
    color_scheme: str = "default"  # "default", "high_contrast"


class SuggestionFormatter:
    """建议格式化器"""

    def __init__(self, config: DisplayConfig = None):
        self.config = config or DisplayConfig()

    def format_suggestion(self, suggestion: RepairSuggestion) -> str:
        """格式化修复建议为可读文本"""
        lines = []

        # 标题和置信度
        confidence_emoji = self._get_confidence_emoji(suggestion.confidence)
        lines.append(f"{confidence_emoji} {suggestion.title}")
        lines.append("=" * 60)

        # 描述
        lines.append(f"\n📋 描述:")
        lines.append(f"   {suggestion.description}")

        # 影响评估
        impact_emoji = self._get_impact_emoji(suggestion.estimated_impact)
        lines.append(f"\n{impact_emoji} 影响评估: {suggestion.estimated_impact.upper()}")

        # 置信度
        if self.config.show_confidence_score:
            lines.append(f"\n🎯 置信度: {suggestion.confidence:.1%}")

        # 代码变更
        if suggestion.code_before or suggestion.code_after:
            lines.append(f"\n📝 代码变更:")
            lines.append(self._format_code_diff(
                suggestion.code_before,
                suggestion.code_after
            ))

        # 解释
        lines.append(f"\n💡 解释:")
        lines.append(f"   {suggestion.explanation}")

        # 受影响的文件
        if suggestion.affected_files:
            lines.append(f"\n📁 受影响的文件:")
            for f in suggestion.affected_files:
                lines.append(f"   • {f}")

        # 副作用
        if suggestion.side_effects:
            lines.append(f"\n⚠️  可能的副作用:")
            for se in suggestion.side_effects:
                lines.append(f"   • {se}")

        # 替代方案
        if self.config.show_alternatives and suggestion.alternatives:
            lines.append(f"\n🔄 替代方案:")
            for i, alt in enumerate(suggestion.alternatives, 1):
                lines.append(f"   {i}. {alt.get('title', '未命名')}")
                if 'description' in alt:
                    lines.append(f"      {alt['description']}")

        return "\n".join(lines)

    def _get_confidence_emoji(self, confidence: float) -> str:
        if confidence >= 0.9:
            return "✅"
        elif confidence >= 0.7:
            return "🟢"
        elif confidence >= 0.5:
            return "🟡"
        else:
            return "🔴"

    def _get_impact_emoji(self, impact: str) -> str:
        return {
            "high": "🔴",
            "medium": "🟡",
            "low": "🟢"
        }.get(impact.lower(), "⚪")

    def _format_code_diff(self, before: Optional[str], after: Optional[str]) -> str:
        """格式化代码差异"""
        if not before and not after:
            return "   (无代码变更)"

        lines = []
        lines.append("   ```")

        if before:
            before_lines = before.split("\n")[:self.config.max_code_lines]
            lines.append("   --- 修改前 ---")
            for i, line in enumerate(before_lines, 1):
                prefix = f"   {i:3d}│- " if self.config.show_line_numbers else "   - "
                lines.append(f"{prefix}{line}")

        if after:
            after_lines = after.split("\n")[:self.config.max_code_lines]
            lines.append("   --- 修改后 ---")
            for i, line in enumerate(after_lines, 1):
                prefix = f"   {i:3d}│+ " if self.config.show_line_numbers else "   + "
                lines.append(f"{prefix}{line}")

        lines.append("   ```")
        return "\n".join(lines)

    def format_multiple_suggestions(
        self,
        suggestions: List[RepairSuggestion],
        show_index: bool = True
    ) -> str:
        """格式化多个建议"""
        lines = []
        lines.append(f"发现 {len(suggestions)} 个修复建议:\n")

        for i, suggestion in enumerate(suggestions, 1):
            if show_index:
                lines.append(f"[{i}] {suggestion.title}")
            else:
                lines.append(suggestion.title)
            lines.append(f"    置信度: {suggestion.confidence:.1%} | 影响: {suggestion.estimated_impact}")
            lines.append("")

        return "\n".join(lines)


class UserInteractionHandler:
    """用户交互处理器"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(
            Path.home() / ".pyutagent" / "user_interactions.db"
        )
        self.formatter = SuggestionFormatter()
        self._preference_cache: Dict[str, UserPreference] = {}
        self._interaction_callbacks: Dict[InteractionType, List[Callable]] = {}
        self._init_db()
        self._load_preferences()

    def _init_db(self):
        """初始化数据库"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 交互记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interaction_records (
                record_id TEXT PRIMARY KEY,
                interaction_type TEXT NOT NULL,
                context_hash TEXT NOT NULL,
                suggestion_json TEXT,
                user_choice TEXT NOT NULL,
                user_feedback TEXT,
                timestamp TEXT NOT NULL,
                session_id TEXT NOT NULL,
                metadata_json TEXT
            )
        """)

        # 用户偏好表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                preference_id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                value TEXT NOT NULL,
                confidence REAL NOT NULL,
                sample_count INTEGER NOT NULL,
                last_updated TEXT NOT NULL
            )
        """)

        conn.commit()
        conn.close()

    def _load_preferences(self):
        """从数据库加载偏好"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM user_preferences")

        for row in cursor.fetchall():
            pref = UserPreference(
                preference_id=row[0],
                category=row[1],
                value=json.loads(row[2]),
                confidence=row[3],
                sample_count=row[4],
                last_updated=datetime.fromisoformat(row[5])
            )
            self._preference_cache[pref.category] = pref

        conn.close()

    def display_suggestion(
        self,
        suggestion: RepairSuggestion,
        config: DisplayConfig = None
    ) -> str:
        """展示单个修复建议"""
        formatter = SuggestionFormatter(config or self.formatter.config)
        formatted = formatter.format_suggestion(suggestion)
        print(formatted)
        return formatted

    def display_suggestions(
        self,
        suggestions: List[RepairSuggestion],
        config: DisplayConfig = None
    ) -> str:
        """展示多个修复建议"""
        formatter = SuggestionFormatter(config or self.formatter.config)

        # 按置信度排序
        sorted_suggestions = sorted(
            suggestions,
            key=lambda s: s.confidence,
            reverse=True
        )

        formatted = formatter.format_multiple_suggestions(sorted_suggestions)
        print(formatted)

        # 根据detail_level决定是否展示详细信息
        detail_level = (config or self.formatter.config).detail_level
        if detail_level == "verbose":
            for suggestion in sorted_suggestions:
                print("\n" + "=" * 60)
                self.display_suggestion(suggestion, config)

        return formatted

    async def request_confirmation(
        self,
        suggestion: RepairSuggestion,
        context: Dict[str, Any],
        auto_decide: bool = False
    ) -> Tuple[UserChoice, Optional[str]]:
        """请求用户确认"""
        context_hash = self._compute_context_hash(context)

        # 检查是否可以自动决策
        if auto_decide:
            auto_choice = self._try_auto_decide(suggestion, context_hash)
            if auto_choice:
                return auto_choice, None

        # 展示建议
        self.display_suggestion(suggestion)

        # 获取用户选择
        print("\n请选择操作:")
        print("  [a] 接受 (Accept)")
        print("  [r] 拒绝 (Reject)")
        print("  [m] 修改 (Modify)")
        print("  [s] 跳过 (Skip)")
        print("  [h] 请求帮助 (Ask for Help)")

        # 在实际应用中，这里应该等待用户输入
        # 这里使用模拟输入
        choice = await self._get_user_input("输入选择 (a/r/m/s/h): ")

        choice_map = {
            'a': UserChoice.ACCEPT,
            'r': UserChoice.REJECT,
            'm': UserChoice.MODIFY,
            's': UserChoice.SKIP,
            'h': UserChoice.ASK_FOR_HELP
        }

        user_choice = choice_map.get(choice.lower(), UserChoice.SKIP)

        # 请求反馈
        feedback = None
        if user_choice in [UserChoice.REJECT, UserChoice.MODIFY]:
            feedback = await self._get_user_input("请提供反馈 (可选): ")

        # 记录交互
        await self._record_interaction(
            InteractionType.REPAIR_CONFIRMATION,
