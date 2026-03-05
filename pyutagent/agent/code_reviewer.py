"""AI-powered code review system for real-time code analysis.

This module provides:
- CodeReviewer: Main code review engine
- ReviewRule: Configurable review rules
- Issue: Structured issue representation
- ReviewReport: Comprehensive review results
"""

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class IssueSeverity(Enum):
    """Severity levels for review issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IssueCategory(Enum):
    """Categories of code issues."""
    BUG = "bug"
    SECURITY = "security"
    PERFORMANCE = "performance"
    STYLE = "style"
    BEST_PRACTICE = "best_practice"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    DESIGN = "design"
    TODO = "todo"
    MISC = "misc"


@dataclass
class CodeIssue:
    """Represents a code issue found during review."""
    issue_id: str
    category: IssueCategory
    severity: IssueSeverity
    message: str
    file_path: str
    line_start: int
    line_end: int = 0
    column_start: int = 0
    column_end: int = 0
    code_snippet: str = ""
    suggestion: str = ""
    rule_id: str = ""
    auto_fixable: bool = False
    confidence: float = 1.0
    related_issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.issue_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "code_snippet": self.code_snippet,
            "suggestion": self.suggestion,
            "rule_id": self.rule_id,
            "auto_fixable": self.auto_fixable,
            "confidence": self.confidence
        }


@dataclass
class ReviewRule:
    """A rule for code review."""
    rule_id: str
    name: str
    description: str
    category: IssueCategory
    severity: IssueSeverity
    enabled: bool = True
    pattern: Optional[str] = None
    check_function: Optional[Callable] = None
    auto_fixable: bool = False
    fix_function: Optional[Callable] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "severity": self.severity.value,
            "enabled": self.enabled,
            "auto_fixable": self.auto_fixable
        }


@dataclass
class ReviewReport:
    """Complete code review report."""
    file_path: str
    review_time: datetime = field(default_factory=datetime.now)
    issues: List[CodeIssue] = field(default_factory=list)
    lines_of_code: int = 0
    issues_by_severity: Dict[str, int] = field(default_factory=dict)
    issues_by_category: Dict[str, int] = field(default_factory=dict)
    score: float = 100.0
    summary: str = ""
    auto_fix_count: int = 0

    def add_issue(self, issue: CodeIssue):
        """Add an issue to the report."""
        self.issues.append(issue)

        severity = issue.severity.value
        self.issues_by_severity[severity] = self.issues_by_severity.get(severity, 0) + 1

        category = issue.category.value
        self.issues_by_category[category] = self.issues_by_category.get(category, 0) + 1

        if issue.auto_fixable:
            self.auto_fix_count += 1

    def calculate_score(self):
        """Calculate overall score based on issues."""
        weights = {
            IssueSeverity.CRITICAL: 10,
            IssueSeverity.HIGH: 5,
            IssueSeverity.MEDIUM: 2,
            IssueSeverity.LOW: 1,
            IssueSeverity.INFO: 0.5
        }

        penalty = 0
        for issue in self.issues:
            penalty += weights.get(issue.severity, 1) * (1 - issue.confidence)

        self.score = max(0, 100 - penalty)

        if self.issues_by_severity:
            critical = self.issues_by_severity.get("critical", 0)
            high = self.issues_by_severity.get("high", 0)
            if critical > 0:
                self.summary = f"Found {critical} critical issue(s) requiring immediate attention"
            elif high > 0:
                self.summary = f"Found {high} high severity issue(s) that should be addressed"
            else:
                self.summary = f"Code review complete: {len(self.issues)} issue(s) found"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "review_time": self.review_time.isoformat(),
            "issues": [i.to_dict() for i in self.issues],
            "issues_by_severity": self.issues_by_severity,
            "issues_by_category": self.issues_by_category,
            "score": self.score,
            "summary": self.summary,
            "lines_of_code": self.lines_of_code,
            "auto_fix_count": self.auto_fix_count
        }


class ReviewRuleRegistry:
    """Registry of code review rules."""

    def __init__(self):
        self._rules: Dict[str, ReviewRule] = {}
        self._category_rules: Dict[IssueCategory, List[str]] = {cat: [] for cat in IssueCategory}

    def register(self, rule: ReviewRule):
        """Register a review rule."""
        self._rules[rule.rule_id] = rule
        self._category_rules[rule.category].append(rule.rule_id)

    def get(self, rule_id: str) -> Optional[ReviewRule]:
        """Get a rule by ID."""
        return self._rules.get(rule_id)

    def get_enabled_rules(self, category: Optional[IssueCategory] = None) -> List[ReviewRule]:
        """Get all enabled rules."""
        if category:
            return [self._rules[rid] for rid in self._category_rules.get(category, [])
                   if self._rules[rid].enabled]
        return [r for r in self._rules.values() if r.enabled]

    def enable(self, rule_id: str):
        """Enable a rule."""
        if rule_id in self._rules:
            self._rules[rule_id].enabled = True

    def disable(self, rule_id: str):
        """Disable a rule."""
        if rule_id in self._rules:
            self._rules[rule_id].enabled = False


class CodeReviewer:
    """Main code review engine."""

    def __init__(self, registry: Optional[ReviewRuleRegistry] = None):
        self._registry = registry or self._create_default_registry()
        self._issue_counter = 0

    def _create_default_registry(self) -> ReviewRuleRegistry:
        """Create default rule registry."""
        registry = ReviewRuleRegistry()

        rules = [
            ReviewRule(
                rule_id="S001",
                name="Empty catch block",
                description="Empty catch blocks hide errors",
                category=IssueCategory.BUG,
                severity=IssueSeverity.HIGH,
                pattern=r"catch\s*\([^)]+\)\s*\{\s*\}"
            ),
            ReviewRule(
                rule_id="S002",
                name="Hardcoded credentials",
                description="Hardcoded passwords or API keys detected",
                category=IssueCategory.SECURITY,
                severity=IssueSeverity.CRITICAL,
                pattern=r"(password|api_key|apikey|secret|token)\s*=\s*['\"][^'\"]+['\"]"
            ),
            ReviewRule(
                rule_id="S003",
                name="TODO comment",
                description="TODO comments should be addressed",
                category=IssueCategory.TODO,
                severity=IssueSeverity.LOW,
                pattern=r"//\s*TODO|/\*\s*TODO"
            ),
            ReviewRule(
                rule_id="S004",
                name="Long line",
                description="Line exceeds recommended length",
                category=IssueCategory.STYLE,
                severity=IssueSeverity.LOW
            ),
            ReviewRule(
                rule_id="S005",
                name="Unused variable",
                description="Variable is declared but never used",
                category=IssueCategory.BEST_PRACTICE,
                severity=IssueSeverity.MEDIUM,
                pattern=r"(int|String|bool|var|let|const)\s+\w+\s*;"
            ),
            ReviewRule(
                rule_id="S006",
                name="System.exit usage",
                description="System.exit should not be called in library code",
                category=IssueCategory.BEST_PRACTICE,
                severity=IssueSeverity.MEDIUM,
                pattern=r"System\.exit\("
            ),
            ReviewRule(
                rule_id="S007",
                name="Null pointer check missing",
                description="Potential NullPointerException",
                category=IssueCategory.BUG,
                severity=IssueSeverity.HIGH,
                pattern=r"\w+\.\w+\(\)\s*==\s*null"
            ),
            ReviewRule(
                rule_id="S008",
                name="Print statement in code",
                description="System.out.println found - use logging instead",
                category=IssueCategory.BEST_PRACTICE,
                severity=IssueSeverity.LOW,
                pattern=r"System\.(out|err)\.(print|println)"
            ),
            ReviewRule(
                rule_id="S009",
                name="Magic number",
                description="Magic number should be a named constant",
                category=IssueCategory.BEST_PRACTICE,
                severity=IssueSeverity.LOW,
                pattern=r"[!=<>]=\s*\d{2,}"
            ),
            ReviewRule(
                rule_id="S010",
                name="Missing @Override",
                description="Method overrides superclass but missing @Override",
                category=IssueCategory.BEST_PRACTICE,
                severity=IssueSeverity.MEDIUM
            ),
            ReviewRule(
                rule_id="S011",
                name="Complex method",
                description="Method has high cyclomatic complexity",
                category=IssueCategory.DESIGN,
                severity=IssueSeverity.MEDIUM
            ),
            ReviewRule(
                rule_id="S012",
                name="Large class",
                description="Class has too many lines of code",
                category=IssueCategory.DESIGN,
                severity=IssueSeverity.MEDIUM
            ),
        ]

        for rule in rules:
            registry.register(rule)

        return registry

    def review_file(self, file_path: str, content: str, language: str = "java") -> ReviewReport:
        """Review a file and generate a report.

        Args:
            file_path: Path to the file
            content: File content
            language: Programming language

        Returns:
            Review report
        """
        report = ReviewReport(file_path=file_path)
        report.lines_of_code = len(content.splitlines())

        enabled_rules = self._registry.get_enabled_rules()

        for rule in enabled_rules:
            self._check_rule(rule, content, report)

        report.calculate_score()
        return report

    def _check_rule(self, rule: ReviewRule, content: str, report: ReviewReport):
        """Check content against a rule."""
        if rule.pattern:
            self._check_pattern(rule, content, report)
        elif rule.check_function:
            try:
                issues = rule.check_function(content, report.file_path)
                for issue in issues:
                    report.add_issue(issue)
            except Exception as e:
                logger.debug(f"Rule {rule.rule_id} check failed: {e}")

    def _check_pattern(self, rule: ReviewRule, content: str, report: ReviewReport):
        """Check content using a regex pattern."""
        try:
            lines = content.splitlines()
            pattern = re.compile(rule.pattern, re.IGNORECASE)

            for line_num, line in enumerate(lines, 1):
                if pattern.search(line):
                    issue = CodeIssue(
                        issue_id=f"{rule.rule_id}_{self._issue_counter}",
                        category=rule.category,
                        severity=rule.severity,
                        message=rule.description,
                        file_path=report.file_path,
                        line_start=line_num,
                        line_end=line_num,
                        code_snippet=line.strip()[:100],
                        suggestion=self._get_suggestion(rule),
                        rule_id=rule.rule_id,
                        auto_fixable=rule.auto_fixable
                    )
                    report.add_issue(issue)
                    self._issue_counter += 1

        except re.error as e:
            logger.warning(f"Invalid pattern in rule {rule.rule_id}: {e}")

    def _get_suggestion(self, rule: ReviewRule) -> str:
        """Get suggestion for a rule."""
        suggestions = {
            "S001": "Add error handling or logging to the catch block",
            "S002": "Use environment variables or secure configuration",
            "S003": "Address the TODO or create a tracking issue",
            "S004": "Consider breaking the line into multiple lines",
            "S005": "Remove unused variable or use it",
            "S006": "Use exception handling instead of System.exit",
            "S007": "Add null check before using the object",
            "S008": "Use a logging framework instead",
            "S009": "Extract to a named constant",
            "S010": "Add @Override annotation",
            "S011": "Consider refactoring to reduce complexity",
            "S012": "Consider splitting into smaller classes"
        }
        return suggestions.get(rule.rule_id, "Review and fix this issue")

    def review_multiple_files(self, files: Dict[str, str]) -> List[ReviewReport]:
        """Review multiple files.

        Args:
            files: Dictionary of file_path -> content

        Returns:
            List of review reports
        """
        reports = []
        for file_path, content in files.items():
            report = self.review_file(file_path, content)
            reports.append(report)
        return reports


class AIEnhancedReviewer(CodeReviewer):
    """AI-enhanced code reviewer with LLM capabilities."""

    def __init__(self, registry: Optional[ReviewRuleRegistry] = None, llm_client=None):
        super().__init__(registry)
        self._llm_client = llm_client

    async def review_with_ai(self, file_path: str, content: str, language: str = "java") -> ReviewReport:
        """Review file with AI enhancement.

        Args:
            file_path: Path to the file
            content: File content
            language: Programming language

        Returns:
            Enhanced review report
        """
        report = self.review_file(file_path, content, language)

        if self._llm_client:
            try:
                ai_issues = await self._get_ai_analysis(content, language)
                for issue in ai_issues:
                    report.add_issue(issue)
                report.calculate_score()
            except Exception as e:
                logger.warning(f"AI review failed: {e}")

        return report

    async def _get_ai_analysis(self, content: str, language: str) -> List[CodeIssue]:
        """Get AI-powered analysis of the code.

        This is a placeholder - in production, this would call the LLM.
        """
        return []

    def set_llm_client(self, client):
        """Set LLM client for AI enhancement."""
        self._llm_client = client


def get_default_reviewer() -> CodeReviewer:
    """Get default code reviewer instance."""
    global _default_reviewer
    if _default_reviewer is None:
        _default_reviewer = CodeReviewer()
    return _default_reviewer


_default_reviewer: Optional[CodeReviewer] = None
