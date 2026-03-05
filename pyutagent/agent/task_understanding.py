"""Task understanding module for universal coding agent.

This module provides task classification and understanding capabilities,
allowing the agent to handle various types of programming tasks beyond UT generation.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class Intent(Enum):
    """User intents."""
    GENERATE = "generate"
    MODIFY = "modify"
    FIX = "fix"
    ANALYZE = "analyze"
    EXPLAIN = "explain"
    OPTIMIZE = "optimize"
    QUERY = "query"
    UNKNOWN = "unknown"


@dataclass
class ExtractedEntity:
    """An entity extracted from user request."""
    type: str
    value: str
    confidence: float
    position: Tuple[int, int]


@dataclass
class EntityExtractionResult:
    """Result of entity extraction."""
    files: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    packages: List[str] = field(default_factory=list)
    commands: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)


class TaskType(Enum):
    """Types of programming tasks the agent can handle."""
    UT_GENERATION = "ut_generation"
    CODE_REFACTORING = "code_refactoring"
    BUG_FIX = "bug_fix"
    FEATURE_ADD = "feature_add"
    CODE_REVIEW = "code_review"
    DOCUMENTATION = "documentation"
    CODE_EXPLANATION = "code_explanation"
    TEST_DEBUG = "test_debug"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    SECURITY_AUDIT = "security_audit"
    UNKNOWN = "unknown"


class TaskPriority(Enum):
    """Priority levels for tasks."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskComplexity(Enum):
    """Complexity levels for tasks."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


@dataclass
class TargetScope:
    """Target scope for a task."""
    files: List[str] = field(default_factory=list)
    directories: List[str] = field(default_factory=list)
    packages: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    
    def is_empty(self) -> bool:
        return not any([
            self.files, self.directories, 
            self.packages, self.classes, self.methods
        ])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "files": self.files,
            "directories": self.directories,
            "packages": self.packages,
            "classes": self.classes,
            "methods": self.methods,
        }


@dataclass
class Constraint:
    """Constraint for task execution."""
    name: str
    value: Any
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "description": self.description,
        }


@dataclass
class SuccessCriterion:
    """Success criterion for task completion."""
    description: str
    measurable: bool = True
    threshold: Optional[float] = None
    validator: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "measurable": self.measurable,
            "threshold": self.threshold,
            "validator": self.validator,
        }


@dataclass
class TaskUnderstanding:
    """Complete understanding of a user task.
    
    This represents the agent's understanding of what the user wants to accomplish.
    """
    task_id: str = ""
    name: str = ""
    description: str = ""
    task_type: TaskType = TaskType.UNKNOWN
    original_request: str = ""
    requirements: str = ""
    target_scope: TargetScope = field(default_factory=TargetScope)
    constraints: List[Constraint] = field(default_factory=list)
    success_criteria: List[SuccessCriterion] = field(default_factory=list)
    priority: TaskPriority = TaskPriority.MEDIUM
    complexity: int = 5
    estimated_steps: int = 5
    required_tools: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    context_needed: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    confidence: float = 0.0
    type: str = ""
    sub_tasks: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type.value,
            "original_request": self.original_request,
            "requirements": self.requirements,
            "target_scope": self.target_scope.to_dict(),
            "constraints": [c.to_dict() for c in self.constraints],
            "success_criteria": [s.to_dict() for s in self.success_criteria],
            "priority": self.priority.value,
            "complexity": self.complexity.value,
            "estimated_steps": self.estimated_steps,
            "required_tools": self.required_tools,
            "dependencies": self.dependencies,
            "context_needed": self.context_needed,
            "risks": self.risks,
            "confidence": self.confidence,
        }


TASK_CLASSIFICATION_PROMPT = """You are a task classifier for a coding agent. Analyze the user's request and classify it.

## Task Types
1. ut_generation: Generate unit tests for code
2. code_refactoring: Refactor or restructure existing code
3. bug_fix: Fix a bug or error in code
4. feature_add: Add new functionality to existing code
5. code_review: Review code for quality, security, or best practices
6. documentation: Generate or update documentation
7. code_explanation: Explain how code works
8. test_debug: Debug failing tests
9. performance_optimization: Optimize code performance
10. security_audit: Audit code for security vulnerabilities
11. unknown: Cannot determine task type

## User Request
{user_request}

## Project Context
{project_context}

## Output Format (JSON)
{{
    "task_type": "<task_type>",
    "requirements": "<detailed requirements extracted from request>",
    "target_files": ["<file1>", "<file2>"],
    "target_directories": ["<dir1>"],
    "target_packages": ["<package1>"],
    "target_classes": ["<class1>"],
    "target_methods": ["<method1>"],
    "constraints": [
        {{"name": "<constraint_name>", "value": "<value>", "description": "<desc>"}}
    ],
    "success_criteria": [
        {{"description": "<criterion>", "measurable": true, "threshold": <value>}}
    ],
    "priority": "<low|medium|high|critical>",
    "complexity": "<simple|moderate|complex|very_complex>",
    "estimated_steps": <number>,
    "required_tools": ["<tool1>", "<tool2>"],
    "dependencies": ["<dependency1>"],
    "context_needed": ["<context1>"],
    "risks": ["<risk1>"],
    "confidence": <0.0-1.0>
}}

Analyze the request and respond with only valid JSON.
"""


class TaskClassifier:
    """Classifies user tasks into task types with detailed understanding."""
    
    TASK_TYPE_KEYWORDS: Dict[TaskType, List[str]] = {
        TaskType.UT_GENERATION: [
            "generate test", "unit test", "test case", "junit", "test coverage",
            "write test", "create test", "add test", "testing", "ut", "测试",
        ],
        TaskType.CODE_REFACTORING: [
            "refactor", "restructure", "reorganize", "clean up", "improve code",
            "optimize structure", "rename", "extract method", "move class",
            "重构", "优化结构",
        ],
        TaskType.BUG_FIX: [
            "fix bug", "fix error", "resolve issue", "debug", "error", "exception",
            "not working", "broken", "crash", "修复", "bug", "错误",
        ],
        TaskType.FEATURE_ADD: [
            "add feature", "implement", "new functionality", "extend", "enhance",
            "add support", "新增", "实现", "功能",
        ],
        TaskType.CODE_REVIEW: [
            "review", "check code", "analyze code", "code quality", "audit",
            "best practice", "代码审查", "代码质量",
        ],
        TaskType.DOCUMENTATION: [
            "document", "documentation", "comment", "javadoc", "readme", "doc",
            "文档", "注释",
        ],
        TaskType.CODE_EXPLANATION: [
            "explain", "how does", "what does", "understand", "describe",
            "解释", "说明", "理解",
        ],
        TaskType.TEST_DEBUG: [
            "test failing", "test error", "fix test", "debug test", "test not passing",
            "测试失败", "调试测试",
        ],
        TaskType.PERFORMANCE_OPTIMIZATION: [
            "optimize performance", "speed up", "slow", "performance", "efficient",
            "性能优化", "加速",
        ],
        TaskType.SECURITY_AUDIT: [
            "security", "vulnerability", "secure", "exploit", "injection",
            "安全", "漏洞",
        ],
    }
    
    PRIORITY_KEYWORDS: Dict[TaskPriority, List[str]] = {
        TaskPriority.CRITICAL: ["urgent", "critical", "asap", "immediately", "紧急", "严重"],
        TaskPriority.HIGH: ["important", "high priority", "soon", "重要", "优先"],
        TaskPriority.LOW: ["low priority", "when possible", "minor", "低优先级", "次要"],
    }
    
    COMPLEXITY_INDICATORS: Dict[TaskComplexity, List[str]] = {
        TaskComplexity.SIMPLE: ["single file", "simple", "quick", "minor", "单个文件", "简单"],
        TaskComplexity.COMPLEX: ["multiple files", "complex", "significant", "多个文件", "复杂"],
        TaskComplexity.VERY_COMPLEX: ["architecture", "redesign", "major refactor", "架构", "重大"],
    }
    
    def __init__(self, llm_client=None):
        """Initialize task classifier.
        
        Args:
            llm_client: Optional LLM client for advanced classification
        """
        self.llm_client = llm_client
    
    def classify_by_keywords(self, request: str) -> TaskType:
        """Classify task type using keyword matching.
        
        Args:
            request: User's request string
            
        Returns:
            Most likely TaskType based on keywords
        """
        request_lower = request.lower()
        scores: Dict[TaskType, int] = {}
        
        for task_type, keywords in self.TASK_TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in request_lower)
            if score > 0:
                scores[task_type] = score
        
        if not scores:
            return TaskType.UNKNOWN
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def determine_priority(self, request: str) -> TaskPriority:
        """Determine task priority from request.
        
        Args:
            request: User's request string
            
        Returns:
            TaskPriority level
        """
        request_lower = request.lower()
        
        for priority, keywords in self.PRIORITY_KEYWORDS.items():
            if any(kw in request_lower for kw in keywords):
                return priority
        
        return TaskPriority.MEDIUM
    
    def determine_complexity(self, request: str) -> TaskComplexity:
        """Determine task complexity from request.
        
        Args:
            request: User's request string
            
        Returns:
            TaskComplexity level
        """
        request_lower = request.lower()
        
        for complexity, indicators in self.COMPLEXITY_INDICATORS.items():
            if any(ind in request_lower for ind in indicators):
                return complexity
        
        return TaskComplexity.MODERATE
    
    def extract_target_files(self, request: str, project_path: Optional[Path] = None) -> List[str]:
        """Extract target file paths from request.
        
        Args:
            request: User's request string
            project_path: Optional project path for validation
            
        Returns:
            List of target file paths
        """
        import re
        
        patterns = [
            r'[\w/\\-]+\.java',
            r'[\w/\\-]+\.py',
            r'[\w/\\-]+\.ts',
            r'[\w/\\-]+\.js',
            r'[\w/\\-]+\.go',
            r'[\w/\\-]+\.rs',
            r'`([^`]+)`',
            r'"([^"]+)"',
            r"'([^']+)'",
        ]
        
        files = []
        for pattern in patterns:
            matches = re.findall(pattern, request)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match else ""
                if match and not match.startswith('http'):
                    files.append(match)
        
        if project_path:
            files = [
                f for f in files 
                if (project_path / f).exists() or f.endswith(('.java', '.py', '.ts', '.js', '.go', '.rs'))
            ]
        
        return list(set(files))
    
    async def classify_with_llm(
        self, 
        request: str, 
        project_context: str = ""
    ) -> TaskUnderstanding:
        """Classify task using LLM for advanced understanding.
        
        Args:
            request: User's request string
            project_context: Context about the project
            
        Returns:
            TaskUnderstanding with detailed analysis
        """
        if not self.llm_client:
            return self.create_basic_understanding(request)
        
        try:
            prompt = TASK_CLASSIFICATION_PROMPT.format(
                user_request=request,
                project_context=project_context or "No project context available"
            )
            
            response = await self.llm_client.generate(prompt)
            
            import json
            result = json.loads(self._extract_json(response))
            
            return self._parse_llm_result(request, result)
            
        except Exception as e:
            logger.warning(f"[TaskClassifier] LLM classification failed: {e}, falling back to basic")
            return self.create_basic_understanding(request)
    
    def create_basic_understanding(self, request: str) -> TaskUnderstanding:
        """Create basic task understanding without LLM.
        
        Args:
            request: User's request string
            
        Returns:
            TaskUnderstanding with basic analysis
        """
        task_type = self.classify_by_keywords(request)
        priority = self.determine_priority(request)
        complexity = self.determine_complexity(request)
        target_files = self.extract_target_files(request)
        
        return TaskUnderstanding(
            task_type=task_type,
            original_request=request,
            requirements=request,
            target_scope=TargetScope(files=target_files),
            priority=priority,
            complexity=complexity,
            confidence=0.6 if task_type != TaskType.UNKNOWN else 0.3,
        )
    
    def _extract_json(self, response: str) -> str:
        """Extract JSON from LLM response."""
        import re
        
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            return json_match.group()
        return response
    
    def _parse_llm_result(self, request: str, result: Dict[str, Any]) -> TaskUnderstanding:
        """Parse LLM classification result into TaskUnderstanding."""
        task_type_str = result.get("task_type", "unknown")
        try:
            task_type = TaskType(task_type_str)
        except ValueError:
            task_type = TaskType.UNKNOWN
        
        priority_str = result.get("priority", "medium")
        try:
            priority = TaskPriority(priority_str)
        except ValueError:
            priority = TaskPriority.MEDIUM
        
        complexity_str = result.get("complexity", "moderate")
        try:
            complexity = TaskComplexity(complexity_str)
        except ValueError:
            complexity = TaskComplexity.MODERATE
        
        constraints = [
            Constraint(
                name=c.get("name", ""),
                value=c.get("value"),
                description=c.get("description", ""),
            )
            for c in result.get("constraints", [])
        ]
        
        success_criteria = [
            SuccessCriterion(
                description=s.get("description", ""),
                measurable=s.get("measurable", True),
                threshold=s.get("threshold"),
            )
            for s in result.get("success_criteria", [])
        ]
        
        return TaskUnderstanding(
            task_type=task_type,
            original_request=request,
            requirements=result.get("requirements", request),
            target_scope=TargetScope(
                files=result.get("target_files", []),
                directories=result.get("target_directories", []),
                packages=result.get("target_packages", []),
                classes=result.get("target_classes", []),
                methods=result.get("target_methods", []),
            ),
            constraints=constraints,
            success_criteria=success_criteria,
            priority=priority,
            complexity=complexity,
            estimated_steps=result.get("estimated_steps", 5),
            required_tools=result.get("required_tools", []),
            dependencies=result.get("dependencies", []),
            context_needed=result.get("context_needed", []),
            risks=result.get("risks", []),
            confidence=result.get("confidence", 0.7),
        )

    def extract_entities(self, request: str) -> EntityExtractionResult:
        """Extract entities from user request.

        Args:
            request: User's request string

        Returns:
            EntityExtractionResult with extracted entities
        """
        result = EntityExtractionResult()

        file_patterns = [
            r'([\w/\\\-]+\.(?:java|py|ts|js|go|rs|cpp|c|h))',
            r'`([^`]+)`',
            r'"([^"]+\.[^"]+)"',
        ]
        for pattern in file_patterns:
            matches = re.findall(pattern, request)
            result.files.extend([m for m in matches if '.' in m])

        class_pattern = r'(?:class|interface|enum)\s+(\w+)'
        result.classes.extend(re.findall(class_pattern, request))

        method_pattern = r'(?:method|function)\s+(\w+)|(\w+)\s*\('
        for match in re.finditer(method_pattern, request):
            if match.group(1):
                result.methods.append(match.group(1))
            elif match.group(2):
                result.methods.append(match.group(2))

        package_pattern = r'(?:package|包)\s+([\w.]+)'
        result.packages.extend(re.findall(package_pattern, request))

        command_pattern = r'(?:command|命令|run|执行)\s+["\']?([\w\-]+)'
        result.commands.extend(re.findall(command_pattern, request, re.IGNORECASE))

        return result

    def detect_intent(self, request: str) -> Intent:
        """Detect user intent from request.

        Args:
            request: User's request string

        Returns:
            Detected Intent
        """
        request_lower = request.lower()

        intent_keywords = {
            Intent.GENERATE: ['generate', 'create', 'add', 'write', '新建', '生成', '创建'],
            Intent.MODIFY: ['modify', 'change', 'update', 'edit', '修改', '更新', '编辑'],
            Intent.FIX: ['fix', 'repair', 'debug', 'resolve', '修复', '调试', '解决'],
            Intent.ANALYZE: ['analyze', 'check', 'review', 'audit', '分析', '审查', '检查'],
            Intent.EXPLAIN: ['explain', 'describe', 'understand', '解释', '说明', '理解'],
            Intent.OPTIMIZE: ['optimize', 'improve', 'refactor', '优化', '改进', '重构'],
            Intent.QUERY: ['what', 'how', 'why', 'where', '什么', '如何', '为什么'],
        }

        for intent, keywords in intent_keywords.items():
            if any(kw in request_lower for kw in keywords):
                return intent

        return Intent.UNKNOWN


class EnhancedTaskClassifier(TaskClassifier):
    """Enhanced task classifier with entity extraction and intent detection."""

    def __init__(self, llm_client=None, project_path: Optional[str] = None):
        """Initialize enhanced task classifier.

        Args:
            llm_client: Optional LLM client
            project_path: Optional project path for context
        """
        super().__init__(llm_client)
        self.project_path = Path(project_path) if project_path else None

    async def classify_with_context(
        self,
        request: str,
        project_context: str,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> TaskUnderstanding:
        """Classify task with full context including history.

        Args:
            request: User's request
            project_context: Project context string
            history: Optional list of previous tasks

        Returns:
            TaskUnderstanding with classification
        """
        understanding = await self.classify_with_llm(request, project_context)

        entities = self.extract_entities(request)
        intent = self.detect_intent(request)

        if entities.files:
            understanding.target_scope.files = list(set(
                understanding.target_scope.files + entities.files
            ))
        if entities.classes:
            understanding.target_scope.classes = list(set(
                understanding.target_scope.classes + entities.classes
            ))

        understanding.context_needed.extend(entities.packages)

        logger.info(
            f"[EnhancedTaskClassifier] Classified: {understanding.task_type.value}, "
            f"Intent: {intent.value}, Entities: {len(entities.files)} files"
        )

        return understanding
