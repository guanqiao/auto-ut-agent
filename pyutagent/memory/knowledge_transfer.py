"""Knowledge Transfer - Cross-project knowledge migration.

This module provides knowledge transfer capabilities for applying
learned patterns and strategies across different projects.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class TransferStrategy(Enum):
    """Strategies for knowledge transfer."""
    DIRECT = "direct"
    ADAPTIVE = "adaptive"
    CONSERVATIVE = "conservative"


class PatternType(Enum):
    """Types of transferable patterns."""
    CODE_PATTERN = "code_pattern"
    ERROR_SOLUTION = "error_solution"
    TEST_STRATEGY = "test_strategy"
    REFACTORING_PATTERN = "refactoring_pattern"
    CONFIGURATION = "configuration"


@dataclass
class Pattern:
    """A transferable pattern."""
    pattern_id: str
    pattern_type: PatternType
    name: str
    description: str
    source_project: str
    code_template: Optional[str] = None
    conditions: List[str] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    transfer_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type.value,
            "name": self.name,
            "description": self.description,
            "source_project": self.source_project,
            "code_template": self.code_template,
            "conditions": self.conditions,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "transfer_count": self.transfer_count,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pattern":
        return cls(
            pattern_id=data["pattern_id"],
            pattern_type=PatternType(data["pattern_type"]),
            name=data["name"],
            description=data["description"],
            source_project=data["source_project"],
            code_template=data.get("code_template"),
            conditions=data.get("conditions", []),
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            transfer_count=data.get("transfer_count", 0),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )


@dataclass
class TransferResult:
    """Result of knowledge transfer."""
    success: bool
    pattern_id: str
    target_project: str
    adaptations_made: List[str]
    message: str
    confidence: float


class PatternLibrary:
    """Library of transferable patterns."""
    
    def __init__(self, storage_path: str):
        """Initialize pattern library.
        
        Args:
            storage_path: Path to pattern storage
        """
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._patterns: Dict[str, Pattern] = {}
        self._load_patterns()
    
    def _load_patterns(self):
        """Load patterns from storage."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for pattern_data in data.get("patterns", []):
                pattern = Pattern.from_dict(pattern_data)
                self._patterns[pattern.pattern_id] = pattern
            
            logger.info(f"[PatternLibrary] Loaded {len(self._patterns)} patterns")
        except Exception as e:
            logger.warning(f"[PatternLibrary] Failed to load patterns: {e}")
    
    def _save_patterns(self):
        """Save patterns to storage."""
        data = {
            "patterns": [p.to_dict() for p in self._patterns.values()],
            "updated_at": datetime.now().isoformat(),
        }
        
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def add_pattern(self, pattern: Pattern):
        """Add a pattern to the library."""
        self._patterns[pattern.pattern_id] = pattern
        self._save_patterns()
        logger.info(f"[PatternLibrary] Added pattern: {pattern.name}")
    
    def get_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """Get a pattern by ID."""
        return self._patterns.get(pattern_id)
    
    def find_patterns(
        self,
        pattern_type: Optional[PatternType] = None,
        min_success_rate: float = 0.0,
        limit: int = 10
    ) -> List[Pattern]:
        """Find patterns matching criteria."""
        results = []
        
        for pattern in self._patterns.values():
            if pattern_type and pattern.pattern_type != pattern_type:
                continue
            
            if pattern.success_rate < min_success_rate:
                continue
            
            results.append(pattern)
        
        results.sort(key=lambda p: p.success_rate, reverse=True)
        return results[:limit]
    
    def search_patterns(self, query: str, limit: int = 10) -> List[Pattern]:
        """Search patterns by name or description."""
        query_lower = query.lower()
        results = []
        
        for pattern in self._patterns.values():
            if query_lower in pattern.name.lower():
                results.append(pattern)
            elif query_lower in pattern.description.lower():
                results.append(pattern)
        
        results.sort(key=lambda p: p.success_rate, reverse=True)
        return results[:limit]
    
    def record_success(self, pattern_id: str):
        """Record successful application of a pattern."""
        pattern = self._patterns.get(pattern_id)
        if pattern:
            pattern.success_count += 1
            pattern.updated_at = datetime.now()
            self._save_patterns()
    
    def record_failure(self, pattern_id: str):
        """Record failed application of a pattern."""
        pattern = self._patterns.get(pattern_id)
        if pattern:
            pattern.failure_count += 1
            pattern.updated_at = datetime.now()
            self._save_patterns()


class KnowledgeTransfer:
    """Knowledge transfer system for cross-project learning."""
    
    def __init__(
        self,
        pattern_library: PatternLibrary,
        strategy: TransferStrategy = TransferStrategy.ADAPTIVE
    ):
        """Initialize knowledge transfer.
        
        Args:
            pattern_library: Pattern library
            strategy: Transfer strategy
        """
        self.pattern_library = pattern_library
        self.strategy = strategy
        self._transfer_history: List[TransferResult] = []
    
    async def transfer_patterns(
        self,
        source_project: str,
        target_project: str,
        pattern_types: Optional[List[PatternType]] = None
    ) -> List[TransferResult]:
        """Transfer patterns from source to target project.
        
        Args:
            source_project: Source project name
            target_project: Target project name
            pattern_types: Types of patterns to transfer
            
        Returns:
            List of transfer results
        """
        results = []
        
        patterns = self.pattern_library.find_patterns()
        
        for pattern in patterns:
            if pattern.source_project != source_project:
                continue
            
            if pattern_types and pattern.pattern_type not in pattern_types:
                continue
            
            result = await self._transfer_pattern(pattern, target_project)
            results.append(result)
            self._transfer_history.append(result)
        
        logger.info(f"[KnowledgeTransfer] Transferred {len(results)} patterns to {target_project}")
        return results
    
    async def _transfer_pattern(
        self,
        pattern: Pattern,
        target_project: str
    ) -> TransferResult:
        """Transfer a single pattern."""
        adaptations = []
        confidence = pattern.success_rate
        
        if self.strategy == TransferStrategy.CONSERVATIVE:
            if confidence < 0.8:
                return TransferResult(
                    success=False,
                    pattern_id=pattern.pattern_id,
                    target_project=target_project,
                    adaptations_made=[],
                    message="Pattern success rate too low for conservative transfer",
                    confidence=confidence
                )
        
        if self.strategy == TransferStrategy.ADAPTIVE:
            adaptations = self._generate_adaptations(pattern, target_project)
            confidence *= 0.9
        
        pattern.transfer_count += 1
        self.pattern_library._save_patterns()
        
        return TransferResult(
            success=True,
            pattern_id=pattern.pattern_id,
            target_project=target_project,
            adaptations_made=adaptations,
            message=f"Pattern transferred successfully with {len(adaptations)} adaptations",
            confidence=confidence
        )
    
    def _generate_adaptations(
        self,
        pattern: Pattern,
        target_project: str
    ) -> List[str]:
        """Generate adaptations for target project."""
        adaptations = []
        
        if pattern.code_template:
            adaptations.append("Adjust package names to match target project")
            adaptations.append("Update import statements")
        
        if pattern.pattern_type == PatternType.TEST_STRATEGY:
            adaptations.append("Adapt test framework configuration")
        
        if pattern.pattern_type == PatternType.CONFIGURATION:
            adaptations.append("Adjust paths and directories")
        
        return adaptations
    
    async def suggest_patterns(
        self,
        task_description: str,
        project_context: Dict[str, Any]
    ) -> List[Tuple[Pattern, float]]:
        """Suggest relevant patterns for a task.
        
        Args:
            task_description: Task description
            project_context: Project context
            
        Returns:
            List of (pattern, relevance_score) tuples
        """
        suggestions = []
        
        keywords = task_description.lower().split()
        
        for pattern in self.pattern_library.find_patterns():
            relevance = self._calculate_relevance(pattern, keywords, project_context)
            
            if relevance > 0.3:
                suggestions.append((pattern, relevance))
        
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:5]
    
    def _calculate_relevance(
        self,
        pattern: Pattern,
        keywords: List[str],
        context: Dict[str, Any]
    ) -> float:
        """Calculate relevance score for a pattern."""
        score = 0.0
        
        pattern_keywords = pattern.name.lower().split() + pattern.description.lower().split()
        
        matching_keywords = sum(1 for kw in keywords if kw in pattern_keywords)
        keyword_score = matching_keywords / max(len(keywords), 1)
        score += keyword_score * 0.5
        
        score += pattern.success_rate * 0.3
        
        if context.get("project_type") == pattern.metadata.get("project_type"):
            score += 0.2
        
        return min(score, 1.0)
    
    def get_transfer_stats(self) -> Dict[str, Any]:
        """Get transfer statistics."""
        if not self._transfer_history:
            return {"total_transfers": 0}
        
        successful = sum(1 for r in self._transfer_history if r.success)
        
        return {
            "total_transfers": len(self._transfer_history),
            "successful_transfers": successful,
            "success_rate": successful / len(self._transfer_history),
            "average_confidence": sum(r.confidence for r in self._transfer_history) / len(self._transfer_history),
        }


def create_knowledge_transfer(
    storage_dir: str = ".pyutagent",
    strategy: TransferStrategy = TransferStrategy.ADAPTIVE
) -> KnowledgeTransfer:
    """Create knowledge transfer instance.
    
    Args:
        storage_dir: Storage directory
        strategy: Transfer strategy
        
    Returns:
        KnowledgeTransfer instance
    """
    storage_path = Path(storage_dir) / "pattern_library.json"
    pattern_library = PatternLibrary(str(storage_path))
    return KnowledgeTransfer(pattern_library, strategy)
