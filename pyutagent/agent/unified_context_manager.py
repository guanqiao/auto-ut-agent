"""Unified Context Manager - 统一的上下文管理。

整合所有上下文管理功能：
- 应用级上下文
- Agent间共享上下文
- 代码上下文压缩
- 上下文快照和恢复
"""

import copy
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TypeVar, Generic
from uuid import uuid4

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ContextScope(Enum):
    """上下文作用域。"""
    GLOBAL = "global"
    SESSION = "session"
    TASK = "task"
    AGENT = "agent"
    LOCAL = "local"


class ContextVisibility(Enum):
    """上下文可见性。"""
    PUBLIC = "public"
    PROTECTED = "protected"
    PRIVATE = "private"


class CompressionStrategy(Enum):
    """上下文压缩策略。"""
    NONE = auto()
    METHOD_ONLY = auto()
    SMART = auto()
    SUMMARY = auto()
    HYBRID = auto()


@dataclass
class ContextEntry(Generic[T]):
    """上下文条目。"""
    key: str
    value: T
    scope: ContextScope = ContextScope.SESSION
    visibility: ContextVisibility = ContextVisibility.PUBLIC
    owner_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value if not isinstance(self.value, (dict, list)) else str(self.value),
            "scope": self.scope.value,
            "visibility": self.visibility.value,
            "owner_id": self.owner_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
            "metadata": self.metadata
        }


@dataclass
class CodeSnippet:
    """代码片段。"""
    content: str
    start_line: int
    end_line: int
    snippet_type: str
    name: Optional[str] = None
    relevance_score: float = 0.0
    dependencies: Set[str] = field(default_factory=set)


@dataclass
class HierarchicalSummary:
    """层级摘要。"""
    class_name: str
    package: str = ""
    class_description: str = ""
    method_summaries: Dict[str, str] = field(default_factory=dict)
    field_summaries: Dict[str, str] = field(default_factory=dict)
    key_dependencies: List[str] = field(default_factory=list)


@dataclass
class ContextSnapshot:
    """上下文快照。"""
    snapshot_id: str
    timestamp: str
    entries: Dict[str, ContextEntry]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp,
            "entries": {k: v.to_dict() for k, v in self.entries.items()},
            "metadata": self.metadata
        }


@dataclass
class CompressionResult:
    """压缩结果。"""
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    strategy_used: CompressionStrategy
    snippets_included: List[CodeSnippet] = field(default_factory=list)
    summary: Optional[HierarchicalSummary] = None


    processed_code: str = ""


class TokenEstimator:
    """Token计数估算器。"""

    TOKENS_PER_CHAR = 0.25

    @classmethod
    def estimate(cls, text: str) -> int:
        if not text:
            return 0
        return int(len(text) * cls.TOKENS_PER_CHAR)

    @classmethod
    def estimate_snippets(cls, snippets: List[CodeSnippet]) -> int:
        return sum(cls.estimate(s.content) for s in snippets)


class UnifiedContextManager:
    """统一的上下文管理器。

    整合功能：
    - 应用级状态管理
    - Agent间上下文共享
    - 代码上下文压缩
    - 快照和恢复
    """

    _instance = None
    _lock = threading.RLock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        max_tokens: int = 8000,
        target_tokens: int = 6000,
        compression_strategy: CompressionStrategy = CompressionStrategy.HYBRID,
        auto_initialize: bool = True
    ):
        self.max_tokens = max_tokens
        self.target_tokens = target_tokens
        self.compression_strategy = compression_strategy
        self.token_estimator = TokenEstimator()

        self._lock = threading.RLock()
        self._initialized = False

        self._global_context: Dict[str, ContextEntry] = {}
        self._session_context: Dict[str, ContextEntry] = {}
        self._agent_contexts: Dict[str, Dict[str, ContextEntry]] = {}
        self._snapshots: Dict[str, ContextSnapshot] = {}

        self._settings: Optional[Any] = None
        self._container: Optional[Any] = None

        if auto_initialize:
            self._initialized = True
            logger.info("[UnifiedContextManager] Initialized")

    def initialize(
        self,
        settings: Optional[Any] = None,
        container: Optional[Any] = None
    ) -> None:
        with self._lock:
            if self._initialized:
                logger.warning("[UnifiedContextManager] Already initialized")
                return

            self._settings = settings
            self._container = container
            self._initialized = True
            logger.info("[UnifiedContextManager] Context initialized")

    def reset(self) -> None:
        with self._lock:
            self._global_context.clear()
            self._session_context.clear()
            self._agent_contexts.clear()
            self._snapshots.clear()
            self._settings = None
            self._container = None
            self._initialized = False
            logger.info("[UnifiedContextManager] Context reset")

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def set(
        self,
        key: str,
        value: Any,
        scope: ContextScope = ContextScope.SESSION,
        visibility: ContextVisibility = ContextVisibility.PUBLIC,
        owner_id: Optional[str] = None
    ) -> None:
        with self._lock:
            entry = ContextEntry(
                key=key,
                value=value,
                scope=scope,
                visibility=visibility,
                owner_id=owner_id
            )

            if scope == ContextScope.GLOBAL:
                self._global_context[key] = entry
            elif scope == ContextScope.SESSION:
                self._session_context[key] = entry
            elif scope == ContextScope.AGENT and owner_id:
                if owner_id not in self._agent_contexts:
                    self._agent_contexts[owner_id] = {}
                self._agent_contexts[owner_id][key] = entry

            logger.debug(f"[UnifiedContextManager] Set: {key} in {scope.value} scope")

    def get(
        self,
        key: str,
        scope: ContextScope = ContextScope.SESSION,
        owner_id: Optional[str] = None,
        default: Any = None
    ) -> Any:
        with self._lock:
            if scope == ContextScope.GLOBAL:
                entry = self._global_context.get(key)
            elif scope == ContextScope.SESSION:
                entry = self._session_context.get(key)
            elif scope == ContextScope.AGENT and owner_id:
                agent_ctx = self._agent_contexts.get(owner_id, {})
                entry = agent_ctx.get(key)
            else:
                entry = None

            if entry:
                return entry.value
            return default

    def delete(
        self,
        key: str,
        scope: ContextScope = ContextScope.SESSION,
        owner_id: Optional[str] = None
    ) -> bool:
        with self._lock:
            if scope == ContextScope.GLOBAL:
                if key in self._global_context:
                    del self._global_context[key]
                    return True
            elif scope == ContextScope.SESSION:
                if key in self._session_context:
                    del self._session_context[key]
                    return True
            elif scope == ContextScope.AGENT and owner_id:
                if owner_id in self._agent_contexts and key in self._agent_contexts[owner_id]:
                    del self._agent_contexts[owner_id][key]
                    return True
            return False

    def create_agent_context(self, agent_id: str, parent_id: Optional[str] = None) -> None:
        with self._lock:
            if agent_id not in self._agent_contexts:
                self._agent_contexts[agent_id] = {}

                if parent_id and parent_id in self._agent_contexts:
                    for key, entry in self._agent_contexts[parent_id].items():
                        if entry.visibility in [ContextVisibility.PUBLIC, ContextVisibility.PROTECTED]:
                            inherited_entry = ContextEntry(
                                key=entry.key,
                                value=entry.value,
                                scope=ContextScope.AGENT,
                                visibility=entry.visibility,
                                owner_id=agent_id
                            )
                            self._agent_contexts[agent_id][key] = inherited_entry

                logger.info(f"[UnifiedContextManager] Created context for agent: {agent_id}")

    def get_agent_context(self, agent_id: str) -> Dict[str, ContextEntry]:
        with self._lock:
            return self._agent_contexts.get(agent_id, {}).copy()

    def clear_agent_context(self, agent_id: str) -> None:
        with self._lock:
            if agent_id in self._agent_contexts:
                del self._agent_contexts[agent_id]
                logger.info(f"[UnifiedContextManager] Cleared context for agent: {agent_id}")

    def create_snapshot(self, name: Optional[str] = None) -> ContextSnapshot:
        snapshot_id = str(uuid4())[:8]
        timestamp = datetime.now().isoformat()

        with self._lock:
            entries = {}
            for key, entry in self._global_context.items():
                entries[key] = entry
            for key, entry in self._session_context.items():
                entries[key] = entry

            snapshot = ContextSnapshot(
                snapshot_id=snapshot_id,
                timestamp=timestamp,
                entries=entries
            )
            self._snapshots[snapshot_id] = snapshot

            logger.info(f"[UnifiedContextManager] Created snapshot: {snapshot_id}")
            return snapshot

    def restore_snapshot(self, snapshot_id: str) -> bool:
        with self._lock:
            snapshot = self._snapshots.get(snapshot_id)
            if not snapshot:
                logger.warning(f"[UnifiedContextManager] Snapshot not found: {snapshot_id}")
                return False

            for key, entry in snapshot.entries.items():
                if entry.scope == ContextScope.GLOBAL:
                    self._global_context[key] = entry
                elif entry.scope == ContextScope.SESSION:
                    self._session_context[key] = entry

            logger.info(f"[UnifiedContextManager] Restored snapshot: {snapshot_id}")
            return True

    def delete_snapshot(self, snapshot_id: str) -> bool:
        with self._lock:
            if snapshot_id in self._snapshots:
                del self._snapshots[snapshot_id]
                logger.info(f"[UnifiedContextManager] Deleted snapshot: {snapshot_id}")
                return True
            return False

    def list_snapshots(self) -> List[str]:
        with self._lock:
            return list(self._snapshots.keys())

    def compress_code(
        self,
        code: str,
        target_methods: Optional[List[str]] = None,
        strategy: Optional[CompressionStrategy] = None
    ) -> CompressionResult:
        strategy = strategy or self.compression_strategy
        original_tokens = self.token_estimator.estimate(code)

        logger.info(f"[UnifiedContextManager] Compressing code - Strategy: {strategy.name}")

        if original_tokens <= self.target_tokens:
            return CompressionResult(
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
                strategy_used=CompressionStrategy.NONE,
                processed_code=code
            )

        snippets = self._parse_code_snippets(code)

        if strategy == CompressionStrategy.METHOD_ONLY:
            result = self._apply_method_only(snippets, target_methods)
        elif strategy == CompressionStrategy.SMART:
            result = self._apply_smart_compression(snippets, target_methods)
        elif strategy == CompressionStrategy.HYBRID:
            result = self._apply_hybrid_compression(snippets, target_methods)
        else:
            result = CompressionResult(
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
                strategy_used=CompressionStrategy.NONE,
                processed_code=code
            )

        result.original_tokens = original_tokens
        result.strategy_used = strategy

        logger.info(f"[UnifiedContextManager] Compression complete - Ratio: {result.compression_ratio:.1%}")

        return result

    def _parse_code_snippets(self, code: str) -> List[CodeSnippet]:
        snippets = []
        lines = code.split('\n')

        current_type = None
        current_name = None
        current_start = 0
        brace_count = 0
        in_block = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            if not stripped:
                continue

            if stripped.startswith('import ') or stripped.startswith('package '):
                snippets.append(CodeSnippet(
                    content=line,
                    start_line=i + 1,
                    end_line=i + 1,
                    snippet_type='import',
                    name=stripped
                ))
                continue

            if re.match(r'^(public|private|protected)?\s*(class|interface|enum)\s+\w+', stripped):
                match = re.search(r'(class|interface|enum)\s+(\w+)', stripped)
                if match:
                    current_type = 'class_header'
                    current_name = match.group(2)
                    current_start = i + 1
                    in_block = True
                    brace_count = 0

            elif re.match(r'^(public|private|protected)\s+\w+.*\s+\w+\s*\(', stripped):
                match = re.search(r'\s+(\w+)\s*\(', stripped)
                if match:
                    current_type = 'method'
                    current_name = match.group(1)
                    current_start = i + 1
                    in_block = True
                    brace_count = 0

            if in_block:
                brace_count += line.count('{') - line.count('}')

                if brace_count == 0:
                    content = '\n'.join(lines[current_start - 1:i + 1])
                    snippets.append(CodeSnippet(
                        content=content,
                        start_line=current_start,
                        end_line=i + 1,
                        snippet_type=current_type or 'block',
                        name=current_name
                    ))
                    in_block = False

        return snippets

    def _apply_method_only(
        self,
        snippets: List[CodeSnippet],
        target_methods: Optional[List[str]]
    ) -> CompressionResult:
        target_methods = target_methods or []

        selected = [s for s in snippets if s.snippet_type == 'method' and s.name in target_methods]
        imports = [s for s in snippets if s.snippet_type == 'import']
        class_headers = [s for s in snippets if s.snippet_type == 'class_header']

        all_snippets = imports + class_headers + selected
        processed_code = '\n'.join(s.content for s in all_snippets)
        compressed_tokens = self.token_estimator.estimate(processed_code)

        return CompressionResult(
            original_tokens=0,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / max(1, self.token_estimator.estimate(processed_code)),
            strategy_used=CompressionStrategy.METHOD_ONLY,
            snippets_included=all_snippets,
            processed_code=processed_code
        )

    def _apply_smart_compression(
        self,
        snippets: List[CodeSnippet],
        target_methods: Optional[List[str]]
    ) -> CompressionResult:
        target_methods = target_methods or []

        for snippet in snippets:
            if snippet.snippet_type == 'method':
                if snippet.name in target_methods:
                    snippet.relevance_score = 1.0
                else:
                    snippet.relevance_score = 0.5
            elif snippet.snippet_type in ['import', 'class_header']:
                snippet.relevance_score = 0.8
            else:
                snippet.relevance_score = 0.3

        sorted_snippets = sorted(snippets, key=lambda s: s.relevance_score, reverse=True)

        selected = []
        total_tokens = 0

        for snippet in sorted_snippets:
            snippet_tokens = self.token_estimator.estimate(snippet.content)
            if total_tokens + snippet_tokens <= self.target_tokens:
                selected.append(snippet)
                total_tokens += snippet_tokens

        selected.sort(key=lambda s: s.start_line)
        processed_code = '\n'.join(s.content for s in selected)

        return CompressionResult(
            original_tokens=0,
            compressed_tokens=total_tokens,
            compression_ratio=total_tokens / max(1, self.token_estimator.estimate(processed_code)),
            strategy_used=CompressionStrategy.SMART,
            snippets_included=selected,
            processed_code=processed_code
        )

    def _apply_hybrid_compression(
        self,
        snippets: List[CodeSnippet],
        target_methods: Optional[List[str]]
    ) -> CompressionResult:
        imports = [s for s in snippets if s.snippet_type == 'import']
        class_headers = [s for s in snippets if s.snippet_type == 'class_header']

        target_methods = target_methods or []
        target_method_snippets = [s for s in snippets if s.snippet_type == 'method' and s.name in target_methods]

        other_methods = [s for s in snippets if s.snippet_type == 'method' and s.name not in target_methods]

        selected = imports + class_headers + target_method_snippets
        current_tokens = self.token_estimator.estimate_snippets(selected)

        for snippet in other_methods:
            snippet_tokens = self.token_estimator.estimate(snippet.content)
            if current_tokens + snippet_tokens <= self.target_tokens:
                selected.append(snippet)
                current_tokens += snippet_tokens

        selected.sort(key=lambda s: s.start_line)
        processed_code = '\n'.join(s.content for s in selected)

        original_tokens = self.token_estimator.estimate('\n'.join(s.content for s in snippets))

        return CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=current_tokens,
            compression_ratio=current_tokens / max(1, original_tokens),
            strategy_used=CompressionStrategy.HYBRID,
            snippets_included=selected,
            processed_code=processed_code
        )

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "initialized": self._initialized,
                "global_entries": len(self._global_context),
                "session_entries": len(self._session_context),
                "agent_contexts": len(self._agent_contexts),
                "snapshots": len(self._snapshots),
                "max_tokens": self.max_tokens,
                "target_tokens": self.target_tokens,
                "compression_strategy": self.compression_strategy.name
            }


def get_context_manager() -> UnifiedContextManager:
    """获取全局上下文管理器实例。"""
    return UnifiedContextManager()


def reset_context_manager() -> None:
    """重置全局上下文管理器。"""
    if UnifiedContextManager._instance is not None:
        UnifiedContextManager._instance.reset()
        UnifiedContextManager._instance = None
