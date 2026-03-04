"""Test generation history management.

This module provides data models and storage for test generation history,
allowing users to track, review, and re-run previous test generations.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any
import uuid

logger = logging.getLogger(__name__)


class TestGenerationStatus(Enum):
    """Status of test generation."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"  # Some tests passed but coverage not met
    CANCELLED = "cancelled"


@dataclass
class TestGenerationRecord:
    """Record of a single test generation.
    
    Attributes:
        id: Unique identifier for this record
        timestamp: When the generation was started
        project_path: Path to the project
        source_file: Path to the source Java file
        test_file: Path to the generated test file (if successful)
        status: Generation status
        coverage: Final coverage percentage (0.0-1.0)
        target_coverage: Target coverage percentage
        iterations: Number of iterations performed
        duration_seconds: Total duration in seconds
        error_message: Error message if failed
        model_used: Name of the LLM model used
        prompt_tokens: Number of prompt tokens used
        completion_tokens: Number of completion tokens used
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    project_path: str = ""
    source_file: str = ""
    test_file: str = ""
    status: str = TestGenerationStatus.SUCCESS.value
    coverage: float = 0.0
    target_coverage: float = 0.8
    iterations: int = 0
    duration_seconds: float = 0.0
    error_message: str = ""
    model_used: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestGenerationRecord':
        """Create from dictionary."""
        return cls(**data)
    
    @property
    def source_file_name(self) -> str:
        """Get source file name."""
        return Path(self.source_file).name if self.source_file else "Unknown"
    
    @property
    def formatted_timestamp(self) -> str:
        """Get formatted timestamp."""
        try:
            dt = datetime.fromisoformat(self.timestamp)
            return dt.strftime("%Y-%m-%d %H:%M")
        except:
            return self.timestamp
    
    @property
    def formatted_duration(self) -> str:
        """Get formatted duration."""
        if self.duration_seconds < 60:
            return f"{self.duration_seconds:.1f}s"
        else:
            minutes = int(self.duration_seconds // 60)
            seconds = int(self.duration_seconds % 60)
            return f"{minutes}m {seconds}s"
    
    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        return self.prompt_tokens + self.completion_tokens


@dataclass
class TestHistory:
    """Collection of test generation records.
    
    Attributes:
        records: List of test generation records
        max_records: Maximum number of records to keep
    """
    records: List[TestGenerationRecord] = field(default_factory=list)
    max_records: int = 100
    
    def add_record(self, record: TestGenerationRecord) -> None:
        """Add a new record.
        
        Args:
            record: The record to add
        """
        self.records.insert(0, record)  # Add to beginning (newest first)
        
        # Keep only max_records
        if len(self.records) > self.max_records:
            self.records = self.records[:self.max_records]
        
        logger.info(f"[TestHistory] Added record: {record.id} for {record.source_file_name}")
    
    def get_record(self, record_id: str) -> Optional[TestGenerationRecord]:
        """Get a record by ID.
        
        Args:
            record_id: The record ID
            
        Returns:
            The record if found, None otherwise
        """
        for record in self.records:
            if record.id == record_id:
                return record
        return None
    
    def delete_record(self, record_id: str) -> bool:
        """Delete a record by ID.
        
        Args:
            record_id: The record ID
            
        Returns:
            True if deleted, False if not found
        """
        for i, record in enumerate(self.records):
            if record.id == record_id:
                self.records.pop(i)
                logger.info(f"[TestHistory] Deleted record: {record_id}")
                return True
        return False
    
    def clear(self) -> None:
        """Clear all records."""
        self.records.clear()
        logger.info("[TestHistory] Cleared all records")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the history.
        
        Returns:
            Dictionary with statistics
        """
        if not self.records:
            return {
                "total": 0,
                "success_rate": 0.0,
                "avg_coverage": 0.0,
                "avg_duration": 0.0,
            }
        
        total = len(self.records)
        successful = sum(1 for r in self.records if r.status == TestGenerationStatus.SUCCESS.value)
        avg_coverage = sum(r.coverage for r in self.records) / total
        avg_duration = sum(r.duration_seconds for r in self.records) / total
        
        return {
            "total": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": successful / total,
            "avg_coverage": avg_coverage,
            "avg_duration": avg_duration,
        }
    
    def filter_by_project(self, project_path: str) -> List[TestGenerationRecord]:
        """Filter records by project path.
        
        Args:
            project_path: The project path
            
        Returns:
            List of matching records
        """
        return [r for r in self.records if r.project_path == project_path]
    
    def filter_by_status(self, status: TestGenerationStatus) -> List[TestGenerationRecord]:
        """Filter records by status.
        
        Args:
            status: The status to filter by
            
        Returns:
            List of matching records
        """
        return [r for r in self.records if r.status == status.value]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "records": [r.to_dict() for r in self.records],
            "max_records": self.max_records
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestHistory':
        """Create from dictionary."""
        records = [
            TestGenerationRecord.from_dict(r)
            for r in data.get("records", [])
        ]
        return cls(
            records=records,
            max_records=data.get("max_records", 100)
        )


# Global test history instance
_test_history: Optional[TestHistory] = None


def get_test_history() -> TestHistory:
    """Get the global test history instance."""
    global _test_history
    if _test_history is None:
        _test_history = load_test_history()
    return _test_history


def get_history_file_path() -> Path:
    """Get the path to the history file."""
    from .config import get_data_dir
    return get_data_dir() / "test_history.json"


def load_test_history() -> TestHistory:
    """Load test history from file.
    
    Returns:
        The loaded test history
    """
    history_file = get_history_file_path()
    
    if not history_file.exists():
        logger.info(f"[TestHistory] No history file found at {history_file}")
        return TestHistory()
    
    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        history = TestHistory.from_dict(data)
        logger.info(f"[TestHistory] Loaded {len(history.records)} records from {history_file}")
        return history
    except Exception as e:
        logger.error(f"[TestHistory] Failed to load history: {e}")
        return TestHistory()


def save_test_history(history: Optional[TestHistory] = None) -> bool:
    """Save test history to file.
    
    Args:
        history: The history to save (defaults to global instance)
        
    Returns:
        True if saved successfully
    """
    if history is None:
        history = get_test_history()
    
    history_file = get_history_file_path()
    
    try:
        history_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"[TestHistory] Saved {len(history.records)} records to {history_file}")
        return True
    except Exception as e:
        logger.error(f"[TestHistory] Failed to save history: {e}")
        return False


def add_generation_record(
    project_path: str,
    source_file: str,
    test_file: str = "",
    status: str = TestGenerationStatus.SUCCESS.value,
    coverage: float = 0.0,
    target_coverage: float = 0.8,
    iterations: int = 0,
    duration_seconds: float = 0.0,
    error_message: str = "",
    model_used: str = "",
    prompt_tokens: int = 0,
    completion_tokens: int = 0
) -> TestGenerationRecord:
    """Add a new generation record.
    
    Args:
        project_path: Path to the project
        source_file: Path to the source file
        test_file: Path to the generated test file
        status: Generation status
        coverage: Final coverage
        target_coverage: Target coverage
        iterations: Number of iterations
        duration_seconds: Duration in seconds
        error_message: Error message if failed
        model_used: Model name
        prompt_tokens: Prompt tokens used
        completion_tokens: Completion tokens used
        
    Returns:
        The created record
    """
    record = TestGenerationRecord(
        project_path=project_path,
        source_file=source_file,
        test_file=test_file,
        status=status,
        coverage=coverage,
        target_coverage=target_coverage,
        iterations=iterations,
        duration_seconds=duration_seconds,
        error_message=error_message,
        model_used=model_used,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens
    )
    
    history = get_test_history()
    history.add_record(record)
    save_test_history(history)
    
    return record
