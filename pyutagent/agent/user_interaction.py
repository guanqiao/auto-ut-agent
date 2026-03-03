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
            context_hash,
            suggestion,
            user_choice,
            feedback
        )

        # 更新偏好
        self._update_preferences(suggestion, user_choice)

        return user_choice, feedback

    async def request_strategy_selection(
        self,
        strategies: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Tuple[int, Optional[str]]:
        """请求用户选择策略"""
        print("\n可用修复策略:")
        for i, strategy in enumerate(strategies, 1):
            print(f"  [{i}] {strategy.get('name', 'Unknown')}")
            print(f"       {strategy.get('description', '')}")

        choice_str = await self._get_user_input(f"选择策略 (1-{len(strategies)}): ")

        try:
            choice_idx = int(choice_str) - 1
            if 0 <= choice_idx < len(strategies):
                await self._record_interaction(
                    InteractionType.STRATEGY_SELECTION,
                    self._compute_context_hash(context),
                    None,
                    UserChoice.ACCEPT,
                    f"Selected strategy: {strategies[choice_idx].get('name')}"
                )
                return choice_idx, None
        except ValueError:
            pass

        return 0, "Invalid selection, using default"

    async def request_parameter_adjustment(
        self,
        parameters: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """请求用户调整参数"""
        print("\n当前参数:")
        for key, value in parameters.items():
            print(f"  {key}: {value}")

        adjusted = parameters.copy()

        while True:
            param_name = await self._get_user_input(
                "输入要修改的参数名 (或 'done' 完成): "
            )

            if param_name.lower() == 'done':
                break

            if param_name in adjusted:
                new_value = await self._get_user_input(f"输入 {param_name} 的新值: ")
                # 尝试保持原类型
                try:
                    if isinstance(adjusted[param_name], int):
                        adjusted[param_name] = int(new_value)
                    elif isinstance(adjusted[param_name], float):
                        adjusted[param_name] = float(new_value)
                    elif isinstance(adjusted[param_name], bool):
                        adjusted[param_name] = new_value.lower() in ('true', 'yes', '1')
                    else:
                        adjusted[param_name] = new_value
                except ValueError:
                    adjusted[param_name] = new_value

        await self._record_interaction(
            InteractionType.PARAMETER_ADJUSTMENT,
            self._compute_context_hash(context),
            None,
            UserChoice.ACCEPT,
            json.dumps(adjusted)
        )

        return adjusted

    def _try_auto_decide(
        self,
        suggestion: RepairSuggestion,
        context_hash: str
    ) -> Optional[UserChoice]:
        """尝试自动决策"""
        # 高置信度自动接受
        if suggestion.confidence >= 0.95 and suggestion.estimated_impact == "low":
            # 检查历史记录
            similar_choices = self._get_similar_interactions(context_hash)
            if similar_choices:
                accept_ratio = sum(
                    1 for c in similar_choices if c == UserChoice.ACCEPT
                ) / len(similar_choices)
                if accept_ratio > 0.8:
                    return UserChoice.ACCEPT

        # 低置信度自动请求帮助
        if suggestion.confidence < 0.3:
            return UserChoice.ASK_FOR_HELP

        return None

    def _get_similar_interactions(self, context_hash: str) -> List[UserChoice]:
        """获取相似场景的交互历史"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 使用前缀匹配找相似场景
        prefix = context_hash[:16]
        cursor.execute(
            """SELECT user_choice FROM interaction_records
               WHERE context_hash LIKE ?""",
            (f"{prefix}%",)
        )

        choices = []
        for row in cursor.fetchall():
            try:
                choices.append(UserChoice(row[0]))
            except ValueError:
                pass

        conn.close()
        return choices

    async def _get_user_input(self, prompt: str) -> str:
        """获取用户输入（异步包装）"""
        # 在实际应用中，这里应该使用异步输入
        # 这里使用模拟输入
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: input(prompt))

    async def _record_interaction(
        self,
        interaction_type: InteractionType,
        context_hash: str,
        suggestion: Optional[RepairSuggestion],
        user_choice: UserChoice,
        user_feedback: Optional[str]
    ):
        """记录交互"""
        record = InteractionRecord(
            record_id=self._generate_id(),
            interaction_type=interaction_type,
            context_hash=context_hash,
            suggestion=suggestion,
            user_choice=user_choice,
            user_feedback=user_feedback,
            timestamp=datetime.now(),
            session_id=self._get_session_id(),
            metadata={}
        )

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """INSERT INTO interaction_records
               (record_id, interaction_type, context_hash, suggestion_json,
                user_choice, user_feedback, timestamp, session_id, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.record_id,
                interaction_type.name,
                context_hash,
                json.dumps(asdict(suggestion)) if suggestion else None,
                user_choice.value,
                user_feedback,
                record.timestamp.isoformat(),
                record.session_id,
                json.dumps(record.metadata, default=str)
            )
        )

        conn.commit()
        conn.close()

        # 触发回调
        self._trigger_callbacks(interaction_type, record)

    def _update_preferences(self, suggestion: RepairSuggestion, choice: UserChoice):
        """更新用户偏好"""
        # 基于用户选择更新偏好
        if choice == UserChoice.ACCEPT:
            # 用户倾向于接受高置信度建议
            if suggestion.confidence > 0.8:
                self._increment_preference("auto_accept_high_confidence", 1)
        elif choice == UserChoice.MODIFY:
            # 用户倾向于修改建议
            self._increment_preference("prefers_modification", 1)
        elif choice == UserChoice.REJECT:
            # 记录拒绝模式
            self._increment_preference("rejection_count", 1)

    def _increment_preference(self, category: str, delta: int):
        """增加偏好计数"""
        if category in self._preference_cache:
            pref = self._preference_cache[category]
            pref.value = pref.value + delta if isinstance(pref.value, (int, float)) else delta
            pref.sample_count += 1
            pref.last_updated = datetime.now()
        else:
            self._preference_cache[category] = UserPreference(
                preference_id=self._generate_id(),
                category=category,
                value=delta,
                confidence=0.5,
                sample_count=1,
                last_updated=datetime.now()
            )

        self._save_preference(self._preference_cache[category])

    def _save_preference(self, preference: UserPreference):
        """保存偏好到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """INSERT OR REPLACE INTO user_preferences
               (preference_id, category, value, confidence, sample_count, last_updated)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                preference.preference_id,
                preference.category,
                json.dumps(preference.value),
                preference.confidence,
                preference.sample_count,
                preference.last_updated.isoformat()
            )
        )

        conn.commit()
        conn.close()

    def get_preference(self, category: str) -> Optional[UserPreference]:
        """获取用户偏好"""
        return self._preference_cache.get(category)

    def get_interaction_stats(self) -> Dict[str, Any]:
        """获取交互统计"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        # 总交互数
        cursor.execute("SELECT COUNT(*) FROM interaction_records")
        stats['total_interactions'] = cursor.fetchone()[0]

        # 各类型交互数
        cursor.execute(
            """SELECT interaction_type, COUNT(*) FROM interaction_records
               GROUP BY interaction_type"""
        )
        stats['by_type'] = dict(cursor.fetchall())

        # 用户选择分布
        cursor.execute(
            """SELECT user_choice, COUNT(*) FROM interaction_records
               GROUP BY user_choice"""
        )
        stats['choice_distribution'] = dict(cursor.fetchall())

        # 接受率趋势（最近30天）
        from datetime import timedelta
        thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
        cursor.execute(
            """SELECT COUNT(*) FROM interaction_records
               WHERE timestamp > ? AND user_choice = 'accept'""",
            (thirty_days_ago,)
        )
        recent_accepts = cursor.fetchone()[0]

        cursor.execute(
            """SELECT COUNT(*) FROM interaction_records WHERE timestamp > ?""",
            (thirty_days_ago,)
        )
        recent_total = cursor.fetchone()[0]

        stats['recent_accept_rate'] = (
            recent_accepts / recent_total if recent_total > 0 else 0
        )

        conn.close()
        return stats

    def register_callback(
        self,
        interaction_type: InteractionType,
        callback: Callable[[InteractionRecord], None]
    ):
        """注册交互回调"""
        if interaction_type not in self._interaction_callbacks:
            self._interaction_callbacks[interaction_type] = []
        self._interaction_callbacks[interaction_type].append(callback)

    def _trigger_callbacks(self, interaction_type: InteractionType, record: InteractionRecord):
        """触发回调"""
        callbacks = self._interaction_callbacks.get(interaction_type, [])
        for callback in callbacks:
            try:
                callback(record)
            except Exception:
                pass  # 忽略回调错误

    def _compute_context_hash(self, context: Dict[str, Any]) -> str:
        """计算上下文哈希"""
        context_str = json.dumps(context, sort_keys=True, default=str)
        return hashlib.sha256(context_str.encode()).hexdigest()

    def _generate_id(self) -> str:
        """生成唯一ID"""
        return hashlib.sha256(
            f"{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

    def _get_session_id(self) -> str:
        """获取当前会话ID"""
        # 在实际应用中，这应该是一个真正的会话ID
        return self._generate_id()


class InteractiveFixer:
    """交互式修复器"""

    def __init__(self, user_interaction: UserInteractionHandler):
        self.user_interaction = user_interaction

    async def request_fix_confirmation(
        self,
        suggestion: RepairSuggestion,
        context: Dict[str, Any]
    ) -> Tuple[UserChoice, Optional[str]]:
        """请求修复确认"""
        return await self.user_interaction.request_repair_confirmation(
            suggestion, context
        )

    async def request_strategy_selection(
        self,
        strategies: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Tuple[int, Optional[str]]:
        """请求策略选择"""
        return await self.user_interaction.request_strategy_selection(
            strategies, context
        )


def create_user_interaction_handler(timeout: int = 300) -> UserInteractionHandler:
    """创建用户交互处理器"""
    return UserInteractionHandler()


# 便捷函数
def create_repair_suggestion(
    title: str,
    description: str,
    code_before: Optional[str] = None,
    code_after: Optional[str] = None,
    explanation: str = "",
    confidence: float = 0.5,
    estimated_impact: str = "medium",
    affected_files: List[str] = None,
    side_effects: List[str] = None,
    alternatives: List[Dict[str, str]] = None,
    **kwargs
) -> RepairSuggestion:
    """创建修复建议的便捷函数"""
    return RepairSuggestion(
        suggestion_id=hashlib.sha256(title.encode()).hexdigest()[:16],
        title=title,
        description=description,
        code_before=code_before,
        code_after=code_after,
        explanation=explanation,
        confidence=confidence,
        estimated_impact=estimated_impact,
        affected_files=affected_files or [],
        side_effects=side_effects or [],
        alternatives=alternatives or [],
        metadata=kwargs
    )
