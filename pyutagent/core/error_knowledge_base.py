"""Error knowledge base for learning from historical errors and solutions.

This module provides persistent storage and retrieval of error patterns
and their solutions, enabling the agent to learn from past experiences
and suggest fixes more efficiently.
"""

import logging
import sqlite3
import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
import re

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Categories of errors."""
    COMPILATION = "compilation"
    TEST_FAILURE = "test_failure"
    RUNTIME = "runtime"
    SYNTAX = "syntax"
    DEPENDENCY = "dependency"
    CONFIGURATION = "configuration"
    TOOL_EXECUTION = "tool_execution"
    UNKNOWN = "unknown"


class SolutionStatus(Enum):
    """Status of a solution."""
    PROPOSED = "proposed"      # Initial proposal, not verified
    VERIFIED = "verified"      # Verified to work
    DEPRECATED = "deprecated"  # No longer recommended
    FAILED = "failed"          # Tried and failed


@dataclass
class ErrorContext:
    """Context information about an error."""
    error_message: str
    error_type: Optional[str] = None
    stack_trace: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    category: ErrorCategory = ErrorCategory.UNKNOWN
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "error_message": self.error_message,
            "error_type": self.error_type,
            "stack_trace": self.stack_trace,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "code_snippet": self.code_snippet,
            "category": self.category.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ErrorContext':
        """Create from dictionary."""
        return cls(
            error_message=data.get("error_message", ""),
            error_type=data.get("error_type"),
            stack_trace=data.get("stack_trace"),
            file_path=data.get("file_path"),
            line_number=data.get("line_number"),
            code_snippet=data.get("code_snippet"),
            category=ErrorCategory(data.get("category", "unknown"))
        )


@dataclass
class ErrorSolution:
    """A solution to an error."""
    solution_id: str
    error_pattern: str
    fix_description: str
    fix_code: Optional[str] = None
    category: ErrorCategory = ErrorCategory.UNKNOWN
    status: SolutionStatus = SolutionStatus.PROPOSED
    success_count: int = 0
    failure_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: Optional[str] = None
    context_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "solution_id": self.solution_id,
            "error_pattern": self.error_pattern,
            "fix_description": self.fix_description,
            "fix_code": self.fix_code,
            "category": self.category.value,
            "status": self.status.value,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "context_hash": self.context_hash,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ErrorSolution':
        """Create from dictionary."""
        return cls(
            solution_id=data["solution_id"],
            error_pattern=data["error_pattern"],
            fix_description=data["fix_description"],
            fix_code=data.get("fix_code"),
            category=ErrorCategory(data.get("category", "unknown")),
            status=SolutionStatus(data.get("status", "proposed")),
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            created_at=data.get("created_at", datetime.now().isoformat()),
            last_used=data.get("last_used"),
            context_hash=data.get("context_hash"),
            metadata=data.get("metadata", {})
        )


@dataclass
class SimilarErrorResult:
    """Result of similar error search."""
    solution: ErrorSolution
    similarity_score: float
    match_details: Dict[str, Any] = field(default_factory=dict)


class ErrorPatternMatcher:
    """Matches error patterns using various algorithms."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def compute_similarity(
        self,
        error1: ErrorContext,
        error2: ErrorContext
    ) -> float:
        """Compute similarity between two errors (0.0 - 1.0).
        
        Args:
            error1: First error
            error2: Second error
            
        Returns:
            Similarity score
        """
        scores = []
        
        # Message similarity (most important)
        msg_sim = self._text_similarity(error1.error_message, error2.error_message)
        scores.append((msg_sim, 0.5))  # Weight: 50%
        
        # Type similarity
        if error1.error_type and error2.error_type:
            type_sim = 1.0 if error1.error_type == error2.error_type else 0.0
            scores.append((type_sim, 0.2))  # Weight: 20%
        
        # Category similarity
        cat_sim = 1.0 if error1.category == error2.category else 0.0
        scores.append((cat_sim, 0.15))  # Weight: 15%
        
        # Stack trace similarity
        if error1.stack_trace and error2.stack_trace:
            trace_sim = self._stack_trace_similarity(error1.stack_trace, error2.stack_trace)
            scores.append((trace_sim, 0.15))  # Weight: 15%
        
        # Calculate weighted average
        total_weight = sum(w for _, w in scores)
        if total_weight == 0:
            return 0.0
        
        weighted_sum = sum(s * w for s, w in scores)
        return weighted_sum / total_weight
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Compute text similarity using multiple methods."""
        if not text1 or not text2:
            return 0.0
        
        # Normalize texts
        t1 = self._normalize_text(text1)
        t2 = self._normalize_text(text2)
        
        # Exact match
        if t1 == t2:
            return 1.0
        
        # Jaccard similarity on words
        words1 = set(t1.split())
        words2 = set(t2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        jaccard = len(intersection) / len(union)
        
        # Check for key error patterns
        key_patterns1 = self._extract_key_patterns(text1)
        key_patterns2 = self._extract_key_patterns(text2)
        
        if key_patterns1 and key_patterns2:
            pattern_match = len(key_patterns1 & key_patterns2) / max(len(key_patterns1), len(key_patterns2))
            return (jaccard + pattern_match) / 2
        
        return jaccard
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Convert to lowercase
        text = text.lower()
        # Remove file paths (they vary)
        text = re.sub(r'/[\w/]+/', ' ', text)
        text = re.sub(r'\\[\w\\]+\\', ' ', text)
        # Remove line numbers
        text = re.sub(r':\d+', '', text)
        # Remove specific identifiers (keep general patterns)
        text = re.sub(r'\b[a-z]+[A-Z]\w+\b', '<identifier>', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text
    
    def _extract_key_patterns(self, text: str) -> Set[str]:
        """Extract key error patterns from text."""
        patterns = set()
        
        # Common Java error patterns
        error_patterns = [
            r'nullpointerexception',
            r'classnotfoundexception',
            r'nosuchmethoderror',
            r'nosuchfielderror',
            r'classcastexception',
            r'arrayindexoutofboundsexception',
            r'stringindexoutofboundsexception',
            r'illegalargumentexception',
            r'illegalstateexception',
            r'numberformatexception',
            r'parseexception',
            r'ioexception',
            r'filenotfoundexception',
            r'cannot find symbol',
            r'package .* does not exist',
            r'cannot access',
            r'incompatible types',
            r'required:',
            r'found:',
            r'expected',
            r'actual',
            r'assertion failed',
            r'comparison failure',
        ]
        
        text_lower = text.lower()
        for pattern in error_patterns:
            if re.search(pattern, text_lower):
                patterns.add(pattern)
        
        return patterns
    
    def _stack_trace_similarity(self, trace1: str, trace2: str) -> float:
        """Compute similarity between stack traces."""
        # Extract method names from stack traces
        methods1 = self._extract_stack_methods(trace1)
        methods2 = self._extract_stack_methods(trace2)
        
        if not methods1 or not methods2:
            return 0.0
        
        # Jaccard similarity on method names
        intersection = methods1 & methods2
        union = methods1 | methods2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _extract_stack_methods(self, stack_trace: str) -> Set[str]:
        """Extract method names from stack trace."""
        methods = set()
        
        # Pattern: at package.Class.method(File.java:123)
        pattern = r'at\s+([\w.$]+)\.(\w+)\('
        for match in re.finditer(pattern, stack_trace):
            class_name = match.group(1)
            method_name = match.group(2)
            methods.add(f"{class_name}.{method_name}")
        
        return methods
    
    def generate_error_hash(self, error: ErrorContext) -> str:
        """Generate a hash for the error context."""
        # Create a normalized representation
        normalized = f"{error.error_type or ''}:{self._normalize_text(error.error_message)}"
        return hashlib.md5(normalized.encode()).hexdigest()[:16]


class ErrorKnowledgeBase:
    """Knowledge base for storing and retrieving error solutions."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize knowledge base.
        
        Args:
            db_path: Path to SQLite database (default: ~/.pyutagent/error_kb.db)
        """
        if db_path is None:
            home = Path.home()
            db_dir = home / ".pyutagent"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "error_kb.db"
        
        self.db_path = str(db_path)
        self.pattern_matcher = ErrorPatternMatcher()
        self._init_database()
        
        logger.info(f"[ErrorKnowledgeBase] Initialized with database: {self.db_path}")
    
    def _init_database(self):
        """Initialize SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Solutions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS solutions (
                    solution_id TEXT PRIMARY KEY,
                    error_pattern TEXT NOT NULL,
                    fix_description TEXT NOT NULL,
                    fix_code TEXT,
                    category TEXT DEFAULT 'unknown',
                    status TEXT DEFAULT 'proposed',
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_used TEXT,
                    context_hash TEXT,
                    metadata TEXT
                )
            ''')
            
            # Error history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS error_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    error_hash TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    error_type TEXT,
                    category TEXT,
                    solution_id TEXT,
                    success BOOLEAN,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    context TEXT,
                    FOREIGN KEY (solution_id) REFERENCES solutions(solution_id)
                )
            ''')
            
            # Create indexes
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_solutions_category 
                ON solutions(category)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_solutions_status 
                ON solutions(status)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_error_history_hash 
                ON error_history(error_hash)
            ''')
            
            conn.commit()
    
    def record_solution(
        self,
        error_context: ErrorContext,
        fix_description: str,
        fix_code: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Record a new error solution.
        
        Args:
            error_context: Error context
            fix_description: Description of the fix
            fix_code: Optional code for the fix
            metadata: Additional metadata
            
        Returns:
            Solution ID
        """
        solution_id = self._generate_solution_id()
        context_hash = self.pattern_matcher.generate_error_hash(error_context)
        
        solution = ErrorSolution(
            solution_id=solution_id,
            error_pattern=error_context.error_message[:500],  # Truncate long messages
            fix_description=fix_description,
            fix_code=fix_code,
            category=error_context.category,
            status=SolutionStatus.PROPOSED,
            context_hash=context_hash,
            metadata=metadata or {}
        )
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO solutions 
                (solution_id, error_pattern, fix_description, fix_code, category,
                 status, context_hash, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                solution.solution_id,
                solution.error_pattern,
                solution.fix_description,
                solution.fix_code,
                solution.category.value,
                solution.status.value,
                solution.context_hash,
                json.dumps(solution.metadata)
            ))
            conn.commit()
        
        logger.info(f"[ErrorKnowledgeBase] Recorded solution {solution_id} for {error_context.category.value} error")
        return solution_id
    
    def find_similar_errors(
        self,
        error_context: ErrorContext,
        min_similarity: float = 0.6,
        max_results: int = 5
    ) -> List[SimilarErrorResult]:
        """Find similar errors and their solutions.
        
        Args:
            error_context: Error to search for
            min_similarity: Minimum similarity threshold
            max_results: Maximum number of results
            
        Returns:
            List of similar error results
        """
        results = []
        
        # Load all verified solutions
        solutions = self._load_solutions_by_category(error_context.category)
        
        for solution in solutions:
            # Create error context from solution
            solution_error = ErrorContext(
                error_message=solution.error_pattern,
                category=solution.category
            )
            
            # Compute similarity
            similarity = self.pattern_matcher.compute_similarity(error_context, solution_error)
            
            if similarity >= min_similarity:
                results.append(SimilarErrorResult(
                    solution=solution,
                    similarity_score=similarity,
                    match_details={"category_match": error_context.category == solution.category}
                ))
        
        # Sort by similarity (descending)
        results.sort(key=lambda r: r.similarity_score, reverse=True)
        
        return results[:max_results]
    
    def get_suggested_fix(
        self,
        error_context: ErrorContext,
        min_confidence: float = 0.7
    ) -> Optional[ErrorSolution]:
        """Get the best suggested fix for an error.
        
        Args:
            error_context: Error context
            min_confidence: Minimum confidence threshold
            
        Returns:
            Best solution or None
        """
        similar = self.find_similar_errors(error_context, min_similarity=min_confidence)
        
        if not similar:
            return None
        
        # Filter for verified solutions with good success rates
        verified = [
            r for r in similar 
            if r.solution.status == SolutionStatus.VERIFIED 
            and r.solution.success_rate >= 0.5
        ]
        
        if verified:
            # Return the highest similarity verified solution
            return verified[0].solution
        
        # Fall back to best similar solution
        return similar[0].solution
    
    def record_outcome(
        self,
        error_context: ErrorContext,
        solution_id: str,
        success: bool,
        context: Optional[Dict[str, Any]] = None
    ):
        """Record the outcome of applying a solution.
        
        Args:
            error_context: Error context
            solution_id: Solution that was applied
            success: Whether the solution worked
            context: Additional context
        """
        error_hash = self.pattern_matcher.generate_error_hash(error_context)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Record in history
            cursor.execute('''
                INSERT INTO error_history 
                (error_hash, error_message, error_type, category, solution_id, success, context)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                error_hash,
                error_context.error_message[:500],
                error_context.error_type,
                error_context.category.value,
                solution_id,
                success,
                json.dumps(context, default=str) if context else None
            ))
            
            # Update solution stats
            if success:
                cursor.execute('''
                    UPDATE solutions 
                    SET success_count = success_count + 1,
                        last_used = CURRENT_TIMESTAMP,
                        status = CASE WHEN success_count + 1 >= 2 THEN 'verified' ELSE status END
                    WHERE solution_id = ?
                ''', (solution_id,))
            else:
                cursor.execute('''
                    UPDATE solutions 
                    SET failure_count = failure_count + 1,
                        last_used = CURRENT_TIMESTAMP
                    WHERE solution_id = ?
                ''', (solution_id,))
            
            conn.commit()
        
        logger.info(f"[ErrorKnowledgeBase] Recorded {'success' if success else 'failure'} for solution {solution_id}")
    
    def learn_from_success(
        self,
        original_error: ErrorContext,
        applied_fix: str,
        final_code: str
    ) -> str:
        """Learn from a successful fix.
        
        Args:
            original_error: Original error context
            applied_fix: Description of the fix applied
            final_code: Final working code
            
        Returns:
            Solution ID
        """
        # Check if similar solution already exists
        similar = self.find_similar_errors(original_error, min_similarity=0.9)
        
        if similar:
            # Update existing solution
            solution_id = similar[0].solution.solution_id
            self.record_outcome(original_error, solution_id, success=True)
            return solution_id
        
        # Record new solution
        return self.record_solution(
            error_context=original_error,
            fix_description=applied_fix,
            fix_code=final_code,
            metadata={"learned_from_success": True}
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics.
        
        Returns:
            Statistics dictionary
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total solutions
            cursor.execute("SELECT COUNT(*) FROM solutions")
            total_solutions = cursor.fetchone()[0]
            
            # Solutions by status
            cursor.execute('''
                SELECT status, COUNT(*) FROM solutions GROUP BY status
            ''')
            by_status = dict(cursor.fetchall())
            
            # Solutions by category
            cursor.execute('''
                SELECT category, COUNT(*) FROM solutions GROUP BY category
            ''')
            by_category = dict(cursor.fetchall())
            
            # Success rate
            cursor.execute('''
                SELECT SUM(success_count), SUM(failure_count) FROM solutions
            ''')
            total_success, total_failure = cursor.fetchone()
            
            # Recent errors
            cursor.execute('''
                SELECT COUNT(*) FROM error_history 
                WHERE timestamp > datetime('now', '-7 days')
            ''')
            recent_errors = cursor.fetchone()[0]
            
            return {
                "total_solutions": total_solutions,
                "by_status": by_status,
                "by_category": by_category,
                "total_success": total_success or 0,
                "total_failure": total_failure or 0,
                "overall_success_rate": (
                    total_success / (total_success + total_failure) 
                    if (total_success + total_failure) > 0 else 0
                ),
                "recent_errors_7d": recent_errors
            }
    
    def _generate_solution_id(self) -> str:
        """Generate unique solution ID."""
        import uuid
        return f"sol_{uuid.uuid4().hex[:12]}"
    
    def _load_solutions_by_category(
        self,
        category: ErrorCategory
    ) -> List[ErrorSolution]:
        """Load solutions filtered by category."""
        solutions = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Load from same category first
            cursor.execute('''
                SELECT * FROM solutions 
                WHERE category = ? AND status != 'deprecated'
                ORDER BY success_count DESC, created_at DESC
            ''', (category.value,))
            
            for row in cursor.fetchall():
                solutions.append(self._row_to_solution(row))
            
            # Also load from other categories (lower priority)
            cursor.execute('''
                SELECT * FROM solutions 
                WHERE category != ? AND status != 'deprecated'
                ORDER BY success_count DESC, created_at DESC
                LIMIT 50
            ''', (category.value,))
            
            for row in cursor.fetchall():
                solutions.append(self._row_to_solution(row))
        
        return solutions
    
    def _row_to_solution(self, row: Tuple) -> ErrorSolution:
        """Convert database row to ErrorSolution."""
        return ErrorSolution(
            solution_id=row[0],
            error_pattern=row[1],
            fix_description=row[2],
            fix_code=row[3],
            category=ErrorCategory(row[4]),
            status=SolutionStatus(row[5]),
            success_count=row[6],
            failure_count=row[7],
            created_at=row[8],
            last_used=row[9],
            context_hash=row[10],
            metadata=json.loads(row[11]) if row[11] else {}
        )


# Convenience functions

def quick_error_lookup(
    error_message: str,
    db_path: Optional[str] = None
) -> Optional[str]:
    """Quick lookup for error solution.
    
    Args:
        error_message: Error message
        db_path: Path to knowledge base
        
    Returns:
        Suggested fix or None
    """
    kb = ErrorKnowledgeBase(db_path)
    
    error_context = ErrorContext(
        error_message=error_message,
        category=ErrorCategory.UNKNOWN
    )
    
    solution = kb.get_suggested_fix(error_context)
    
    if solution:
        return solution.fix_description
    return None


def record_successful_fix(
    error_message: str,
    fix_description: str,
    db_path: Optional[str] = None
) -> str:
    """Record a successful fix for future use.
    
    Args:
        error_message: Original error message
        fix_description: Description of the fix
        db_path: Path to knowledge base
        
    Returns:
        Solution ID
    """
    kb = ErrorKnowledgeBase(db_path)
    
    error_context = ErrorContext(
        error_message=error_message,
        category=ErrorCategory.UNKNOWN
    )
    
    return kb.record_solution(error_context, fix_description)
