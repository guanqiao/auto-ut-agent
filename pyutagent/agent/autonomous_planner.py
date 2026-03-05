"""Autonomous Planner - Task understanding and decomposition.

This module provides:
- TaskUnderstanding: Analysis of user intent and task type
- Subtask: Decomposed executable unit
- AutonomousPlanner: Main planner with LLM-based reasoning
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of programming tasks."""
    UNIT_TEST_GENERATION = auto()
    CODE_REFACTORING = auto()
    FEATURE_IMPLEMENTATION = auto()
    BUG_FIXING = auto()
    CODE_REVIEW = auto()
    DOCUMENTATION = auto()
    DEPENDENCY_UPDATE = auto()
    CONFIGURATION = auto()
    EXPLORATION = auto()
    UNKNOWN = auto()


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    OPTIONAL = 5


@dataclass
class TaskUnderstanding:
    """Understanding of user task."""
    original_request: str
    task_type: TaskType
    intent: str
    target_files: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    context_hints: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Subtask:
    """A decomposed subtask."""
    id: str
    name: str
    description: str
    task_type: TaskType
    priority: TaskPriority
    dependencies: List[str] = field(default_factory=list)
    estimated_effort: int = 1  # in abstract units
    required_tools: List[str] = field(default_factory=list)
    expected_output: str = ""
    validation_criteria: List[str] = field(default_factory=list)
    completed: bool = False
    result: Optional[Any] = None


@dataclass
class ExecutionPlan:
    """Complete execution plan."""
    task_id: str
    understanding: TaskUnderstanding
    subtasks: List[Subtask]
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionFeedback:
    """Feedback from execution."""
    subtask_id: str
    success: bool
    message: str
    output: Any = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class AutonomousPlanner:
    """Autonomous planner for programming tasks.
    
    Features:
    - LLM-based task understanding
    - Intelligent task decomposition
    - Dynamic plan refinement
    - Fallback strategies
    """
    
    def __init__(
        self,
        llm_client: Any,
        max_subtasks: int = 50,
        refinement_threshold: float = 0.7
    ):
        """Initialize autonomous planner.
        
        Args:
            llm_client: LLM client for reasoning
            max_subtasks: Maximum number of subtasks (default: 50)
            refinement_threshold: Confidence threshold for refinement
        """
        self.llm_client = llm_client
        self.max_subtasks = max_subtasks
        self.refinement_threshold = refinement_threshold
        
        # Task type patterns for quick classification
        self._task_patterns = {
            TaskType.UNIT_TEST_GENERATION: [
                "test", "testing", "unit test", "generate test",
                "test case", "coverage", "jacoco", "junit"
            ],
            TaskType.CODE_REFACTORING: [
                "refactor", "refactoring", "restructure", "clean up",
                "simplify", "optimize", "improve"
            ],
            TaskType.FEATURE_IMPLEMENTATION: [
                "implement", "add feature", "create", "build",
                "develop", "new feature", "functionality"
            ],
            TaskType.BUG_FIXING: [
                "fix", "bug", "error", "issue", "problem",
                "debug", "resolve", "repair"
            ],
            TaskType.CODE_REVIEW: [
                "review", "check", "analyze", "inspect",
                "evaluate", "assess"
            ],
            TaskType.DOCUMENTATION: [
                "document", "doc", "comment", "readme",
                "javadoc", "documentation"
            ],
            TaskType.DEPENDENCY_UPDATE: [
                "update", "upgrade", "dependency", "version",
                "pom.xml", "build.gradle"
            ],
            TaskType.CONFIGURATION: [
                "config", "configuration", "setting", "property",
                "yaml", "json", "xml"
            ]
        }
        
        logger.info("[AutonomousPlanner] Initialized")
    
    async def understand_task(
        self,
        user_request: str,
        project_context: Optional[Dict[str, Any]] = None
    ) -> TaskUnderstanding:
        """Understand user task using LLM.
        
        Args:
            user_request: Original user request
            project_context: Optional project context
            
        Returns:
            TaskUnderstanding with analyzed intent
        """
        logger.info(f"[AutonomousPlanner] Understanding task: {user_request[:100]}...")
        
        # Quick pattern-based classification
        initial_type = self._classify_by_pattern(user_request)
        
        # Build prompt for LLM analysis
        context_str = json.dumps(project_context, indent=2) if project_context else "{}"
        
        prompt = f"""Analyze the following programming task and provide structured understanding.

User Request: {user_request}

Project Context: {context_str}

Initial Classification: {initial_type.name if initial_type else "UNKNOWN"}

Please analyze and return JSON with:
{{
    "task_type": "one of: UNIT_TEST_GENERATION, CODE_REFACTORING, FEATURE_IMPLEMENTATION, BUG_FIXING, CODE_REVIEW, DOCUMENTATION, DEPENDENCY_UPDATE, CONFIGURATION, EXPLORATION, UNKNOWN",
    "intent": "clear description of what the user wants to achieve",
    "target_files": ["list of files likely involved"],
    "constraints": ["any constraints mentioned or implied"],
    "requirements": ["specific requirements"],
    "context_hints": {{"additional context": "values"}},
    "confidence": 0.0-1.0
}}

Analysis:"""
        
        try:
            response = await self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON from response
            analysis = self._extract_json(content)
            
            if analysis:
                task_type = self._parse_task_type(analysis.get("task_type", "UNKNOWN"))
                
                understanding = TaskUnderstanding(
                    original_request=user_request,
                    task_type=task_type,
                    intent=analysis.get("intent", user_request),
                    target_files=analysis.get("target_files", []),
                    constraints=analysis.get("constraints", []),
                    requirements=analysis.get("requirements", []),
                    context_hints=analysis.get("context_hints", {}),
                    confidence=analysis.get("confidence", 0.5)
                )
                
                logger.info(f"[AutonomousPlanner] Task understood: {task_type.name}, confidence: {understanding.confidence}")
                return understanding
            
        except Exception as e:
            logger.error(f"[AutonomousPlanner] LLM analysis failed: {e}")
        
        # Fallback to pattern-based understanding
        return TaskUnderstanding(
            original_request=user_request,
            task_type=initial_type or TaskType.UNKNOWN,
            intent=user_request,
            confidence=0.3
        )
    
    async def decompose_task(
        self,
        understanding: TaskUnderstanding
    ) -> List[Subtask]:
        """Decompose task into subtasks.
        
        Args:
            understanding: Task understanding
            
        Returns:
            List of subtasks
        """
        logger.info(f"[AutonomousPlanner] Decomposing task: {understanding.task_type.name}")
        
        # Use task-type-specific decomposition strategies
        if understanding.task_type == TaskType.UNIT_TEST_GENERATION:
            return self._decompose_test_generation(understanding)
        elif understanding.task_type == TaskType.CODE_REFACTORING:
            return self._decompose_refactoring(understanding)
        elif understanding.task_type == TaskType.FEATURE_IMPLEMENTATION:
            return await self._decompose_feature_implementation(understanding)
        elif understanding.task_type == TaskType.BUG_FIXING:
            return await self._decompose_bug_fixing(understanding)
        else:
            # Generic decomposition using LLM
            return await self._decompose_generic(understanding)
    
    async def create_plan(
        self,
        user_request: str,
        project_context: Optional[Dict[str, Any]] = None
    ) -> ExecutionPlan:
        """Create complete execution plan.
        
        Args:
            user_request: User request
            project_context: Project context
            
        Returns:
            ExecutionPlan
        """
        # Step 1: Understand the task
        understanding = await self.understand_task(user_request, project_context)
        
        # Step 2: Decompose into subtasks
        subtasks = await self.decompose_task(understanding)
        
        # Step 3: Validate and optimize plan
        subtasks = self._validate_and_optimize(subtasks)
        
        plan = ExecutionPlan(
            task_id=f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            understanding=understanding,
            subtasks=subtasks
        )
        
        logger.info(f"[AutonomousPlanner] Plan created with {len(subtasks)} subtasks")
        return plan
    
    async def refine_plan(
        self,
        current_plan: ExecutionPlan,
        execution_feedback: ExecutionFeedback
    ) -> ExecutionPlan:
        """Refine plan based on execution feedback.
        
        Args:
            current_plan: Current execution plan
            execution_feedback: Feedback from execution
            
        Returns:
            Refined ExecutionPlan
        """
        logger.info(f"[AutonomousPlanner] Refining plan based on feedback: {execution_feedback.subtask_id}")
        
        # Find the subtask that produced feedback
        target_subtask = None
        for subtask in current_plan.subtasks:
            if subtask.id == execution_feedback.subtask_id:
                target_subtask = subtask
                break
        
        if not target_subtask:
            logger.warning(f"[AutonomousPlanner] Subtask not found: {execution_feedback.subtask_id}")
            return current_plan
        
        # Update subtask with feedback
        target_subtask.completed = execution_feedback.success
        target_subtask.result = execution_feedback.output
        
        # If failed, consider adding recovery subtasks
        if not execution_feedback.success:
            recovery_subtasks = await self._generate_recovery_subtasks(
                target_subtask, execution_feedback
            )
            
            # Insert recovery subtasks after failed subtask
            target_idx = current_plan.subtasks.index(target_subtask)
            for i, recovery in enumerate(recovery_subtasks):
                current_plan.subtasks.insert(target_idx + 1 + i, recovery)
            
            logger.info(f"[AutonomousPlanner] Added {len(recovery_subtasks)} recovery subtasks")
        
        current_plan.updated_at = datetime.now()
        return current_plan
    
    def _classify_by_pattern(self, request: str) -> Optional[TaskType]:
        """Classify task by pattern matching."""
        request_lower = request.lower()
        
        scores = {}
        for task_type, patterns in self._task_patterns.items():
            score = sum(1 for pattern in patterns if pattern in request_lower)
            if score > 0:
                scores[task_type] = score
        
        if scores:
            return max(scores, key=scores.get)
        return None
    
    def _parse_task_type(self, type_str: str) -> TaskType:
        """Parse task type string to enum."""
        try:
            return TaskType[type_str.upper()]
        except KeyError:
            return TaskType.UNKNOWN
    
    def _extract_json(self, content: str) -> Optional[Dict]:
        """Extract JSON from LLM response."""
        try:
            # Try to find JSON block
            start = content.find('{')
            end = content.rfind('}')
            
            if start != -1 and end != -1:
                json_str = content[start:end+1]
                return json.loads(json_str)
            
            # Try parsing entire content
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning("[AutonomousPlanner] Failed to extract JSON from response")
            return None
    
    def _decompose_test_generation(
        self,
        understanding: TaskUnderstanding
    ) -> List[Subtask]:
        """Decompose unit test generation task."""
        subtasks = []
        base_id = f"test_gen_{datetime.now().strftime('%H%M%S')}"
        
        # Identify target files
        target_files = understanding.target_files or ["target.java"]
        
        for i, target_file in enumerate(target_files):
            file_id = f"{base_id}_{i}"
            
            # Subtask 1: Parse and analyze
            subtasks.append(Subtask(
                id=f"{file_id}_parse",
                name=f"Parse {target_file}",
                description=f"Parse and analyze the target class {target_file}",
                task_type=TaskType.UNIT_TEST_GENERATION,
                priority=TaskPriority.HIGH,
                dependencies=[],
                required_tools=["read_file", "parse_java"],
                expected_output=f"ClassInfo for {target_file}",
                validation_criteria=["Class structure extracted", "Methods identified"]
            ))
            
            # Subtask 2: Generate tests
            subtasks.append(Subtask(
                id=f"{file_id}_generate",
                name=f"Generate tests for {target_file}",
                description=f"Generate unit tests for {target_file}",
                task_type=TaskType.UNIT_TEST_GENERATION,
                priority=TaskPriority.HIGH,
                dependencies=[f"{file_id}_parse"],
                required_tools=["generate_test"],
                expected_output=f"Test file for {target_file}",
                validation_criteria=["Test file created", "Test methods present"]
            ))
            
            # Subtask 3: Compile
            subtasks.append(Subtask(
                id=f"{file_id}_compile",
                name=f"Compile tests for {target_file}",
                description=f"Compile generated tests",
                task_type=TaskType.UNIT_TEST_GENERATION,
                priority=TaskPriority.HIGH,
                dependencies=[f"{file_id}_generate"],
                required_tools=["compile_test"],
                expected_output="Compilation successful",
                validation_criteria=["No compilation errors"]
            ))
            
            # Subtask 4: Run tests
            subtasks.append(Subtask(
                id=f"{file_id}_run",
                name=f"Run tests for {target_file}",
                description=f"Execute generated tests",
                task_type=TaskType.UNIT_TEST_GENERATION,
                priority=TaskPriority.HIGH,
                dependencies=[f"{file_id}_compile"],
                required_tools=["run_test"],
                expected_output="Test execution results",
                validation_criteria=["Tests executed", "Results available"]
            ))
            
            # Subtask 5: Analyze coverage
            subtasks.append(Subtask(
                id=f"{file_id}_coverage",
                name=f"Analyze coverage for {target_file}",
                description=f"Analyze test coverage",
                task_type=TaskType.UNIT_TEST_GENERATION,
                priority=TaskPriority.MEDIUM,
                dependencies=[f"{file_id}_run"],
                required_tools=["analyze_coverage"],
                expected_output="Coverage report",
                validation_criteria=["Coverage data available"]
            ))
        
        return subtasks
    
    def _decompose_refactoring(
        self,
        understanding: TaskUnderstanding
    ) -> List[Subtask]:
        """Decompose refactoring task."""
        subtasks = []
        base_id = f"refactor_{datetime.now().strftime('%H%M%S')}"
        
        # Subtask 1: Analyze current code
        subtasks.append(Subtask(
            id=f"{base_id}_analyze",
            name="Analyze code structure",
            description="Analyze current code structure and identify refactoring targets",
            task_type=TaskType.CODE_REFACTORING,
            priority=TaskPriority.HIGH,
            dependencies=[],
            required_tools=["read_file", "grep"],
            expected_output="Code analysis report",
            validation_criteria=["Target areas identified"]
        ))
        
        # Subtask 2: Create backup
        subtasks.append(Subtask(
            id=f"{base_id}_backup",
            name="Create backup",
            description="Create backup of files to be modified",
            task_type=TaskType.CODE_REFACTORING,
            priority=TaskPriority.HIGH,
            dependencies=[],
            required_tools=["git_status", "bash"],
            expected_output="Backup created",
            validation_criteria=["Backup verified"]
        ))
        
        # Subtask 3: Apply refactoring
        subtasks.append(Subtask(
            id=f"{base_id}_apply",
            name="Apply refactoring",
            description="Apply refactoring changes",
            task_type=TaskType.CODE_REFACTORING,
            priority=TaskPriority.HIGH,
            dependencies=[f"{base_id}_analyze", f"{base_id}_backup"],
            required_tools=["edit_file", "write_file"],
            expected_output="Refactored code",
            validation_criteria=["Changes applied successfully"]
        ))
        
        # Subtask 4: Verify changes
        subtasks.append(Subtask(
            id=f"{base_id}_verify",
            name="Verify refactoring",
            description="Verify refactoring didn't break functionality",
            task_type=TaskType.CODE_REFACTORING,
            priority=TaskPriority.HIGH,
            dependencies=[f"{base_id}_apply"],
            required_tools=["compile_test", "run_test"],
            expected_output="Verification results",
            validation_criteria=["Tests pass", "No regressions"]
        ))
        
        return subtasks
    
    async def _decompose_feature_implementation(
        self,
        understanding: TaskUnderstanding
    ) -> List[Subtask]:
        """Decompose feature implementation task using LLM."""
        prompt = f"""Decompose the following feature implementation task into subtasks.

Task: {understanding.intent}
Target Files: {understanding.target_files}
Requirements: {understanding.requirements}
Constraints: {understanding.constraints}

Return JSON array of subtasks:
[
    {{
        "name": "subtask name",
        "description": "detailed description",
        "priority": "CRITICAL|HIGH|MEDIUM|LOW",
        "dependencies": ["dependency names"],
        "required_tools": ["tool names"],
        "expected_output": "what should be produced",
        "validation_criteria": ["criteria"]
    }}
]

Subtasks:"""
        
        try:
            response = await self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            content = response.content if hasattr(response, 'content') else str(response)
            subtasks_data = self._extract_json(content)
            
            if subtasks_data and isinstance(subtasks_data, list):
                subtasks = []
                base_id = f"feature_{datetime.now().strftime('%H%M%S')}"
                
                for i, data in enumerate(subtasks_data):
                    subtasks.append(Subtask(
                        id=f"{base_id}_{i}",
                        name=data.get("name", f"Subtask {i}"),
                        description=data.get("description", ""),
                        task_type=TaskType.FEATURE_IMPLEMENTATION,
                        priority=TaskPriority[data.get("priority", "MEDIUM")],
                        dependencies=data.get("dependencies", []),
                        required_tools=data.get("required_tools", []),
                        expected_output=data.get("expected_output", ""),
                        validation_criteria=data.get("validation_criteria", [])
                    ))
                
                return subtasks[:self.max_subtasks]
        
        except Exception as e:
            logger.error(f"[AutonomousPlanner] Feature decomposition failed: {e}")
        
        # Fallback to generic decomposition
        return await self._decompose_generic(understanding)
    
    async def _decompose_bug_fixing(
        self,
        understanding: TaskUnderstanding
    ) -> List[Subtask]:
        """Decompose bug fixing task."""
        subtasks = []
        base_id = f"bugfix_{datetime.now().strftime('%H%M%S')}"
        
        # Subtask 1: Reproduce bug
        subtasks.append(Subtask(
            id=f"{base_id}_reproduce",
            name="Reproduce bug",
            description="Reproduce the reported bug to understand the issue",
            task_type=TaskType.BUG_FIXING,
            priority=TaskPriority.CRITICAL,
            dependencies=[],
            required_tools=["run_test", "bash"],
            expected_output="Bug reproduction confirmation",
            validation_criteria=["Bug reproduced successfully"]
        ))
        
        # Subtask 2: Analyze root cause
        subtasks.append(Subtask(
            id=f"{base_id}_analyze",
            name="Analyze root cause",
            description="Analyze code to identify root cause of the bug",
            task_type=TaskType.BUG_FIXING,
            priority=TaskPriority.CRITICAL,
            dependencies=[f"{base_id}_reproduce"],
            required_tools=["read_file", "grep"],
            expected_output="Root cause analysis",
            validation_criteria=["Root cause identified"]
        ))
        
        # Subtask 3: Implement fix
        subtasks.append(Subtask(
            id=f"{base_id}_fix",
            name="Implement fix",
            description="Implement the bug fix",
            task_type=TaskType.BUG_FIXING,
            priority=TaskPriority.CRITICAL,
            dependencies=[f"{base_id}_analyze"],
            required_tools=["edit_file"],
            expected_output="Fixed code",
            validation_criteria=["Fix applied"]
        ))
        
        # Subtask 4: Verify fix
        subtasks.append(Subtask(
            id=f"{base_id}_verify",
            name="Verify fix",
            description="Verify the bug is fixed and no regressions introduced",
            task_type=TaskType.BUG_FIXING,
            priority=TaskPriority.HIGH,
            dependencies=[f"{base_id}_fix"],
            required_tools=["run_test", "compile_test"],
            expected_output="Verification results",
            validation_criteria=["Bug fixed", "Tests pass"]
        ))
        
        return subtasks
    
    async def _decompose_generic(
        self,
        understanding: TaskUnderstanding
    ) -> List[Subtask]:
        """Generic task decomposition using LLM."""
        prompt = f"""Decompose the following task into executable subtasks.

Task: {understanding.intent}
Type: {understanding.task_type.name}
Requirements: {understanding.requirements}

Return JSON array of subtasks (max 5):
[
    {{
        "name": "subtask name",
        "description": "detailed description",
        "priority": "CRITICAL|HIGH|MEDIUM|LOW",
        "dependencies": [],
        "required_tools": ["tool names"],
        "expected_output": "expected result",
        "validation_criteria": ["criteria"]
    }}
]

Subtasks:"""
        
        try:
            response = await self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            content = response.content if hasattr(response, 'content') else str(response)
            subtasks_data = self._extract_json(content)
            
            if subtasks_data and isinstance(subtasks_data, list):
                subtasks = []
                base_id = f"generic_{datetime.now().strftime('%H%M%S')}"
                
                for i, data in enumerate(subtasks_data[:self.max_subtasks]):
                    subtasks.append(Subtask(
                        id=f"{base_id}_{i}",
                        name=data.get("name", f"Subtask {i}"),
                        description=data.get("description", ""),
                        task_type=understanding.task_type,
                        priority=TaskPriority[data.get("priority", "MEDIUM")],
                        dependencies=[f"{base_id}_{j}" for j in range(i) if j < i],
                        required_tools=data.get("required_tools", []),
                        expected_output=data.get("expected_output", ""),
                        validation_criteria=data.get("validation_criteria", [])
                    ))
                
                return subtasks
        
        except Exception as e:
            logger.error(f"[AutonomousPlanner] Generic decomposition failed: {e}")
        
        # Ultimate fallback: single subtask
        return [Subtask(
            id=f"generic_{datetime.now().strftime('%H%M%S')}_0",
            name="Execute task",
            description=understanding.intent,
            task_type=understanding.task_type,
            priority=TaskPriority.HIGH,
            required_tools=["read_file", "write_file"],
            expected_output="Task completed",
            validation_criteria=["Task executed"]
        )]
    
    def _validate_and_optimize(self, subtasks: List[Subtask]) -> List[Subtask]:
        """Validate and optimize subtask list."""
        if not subtasks:
            return subtasks
        
        # Check for circular dependencies
        task_ids = {st.id for st in subtasks}
        for subtask in subtasks:
            for dep in subtask.dependencies:
                if dep not in task_ids:
                    logger.warning(f"[AutonomousPlanner] Dependency not found: {dep}")
        
        # Limit number of subtasks
        if len(subtasks) > self.max_subtasks:
            logger.warning(f"[AutonomousPlanner] Truncating subtasks from {len(subtasks)} to {self.max_subtasks}")
            subtasks = subtasks[:self.max_subtasks]
        
        return subtasks
    
    async def _generate_recovery_subtasks(
        self,
        failed_subtask: Subtask,
        feedback: ExecutionFeedback
    ) -> List[Subtask]:
        """Generate recovery subtasks for failed subtask."""
        recovery_subtasks = []
        base_id = f"recovery_{datetime.now().strftime('%H%M%S')}"
        
        # Analyze failure and create appropriate recovery
        error_message = feedback.message.lower()
        
        if "not found" in error_message or "missing" in error_message:
            # Recovery: Search for file
            recovery_subtasks.append(Subtask(
                id=f"{base_id}_search",
                name=f"Search for missing resource",
                description=f"Search for resource needed by {failed_subtask.name}",
                task_type=TaskType.EXPLORATION,
                priority=TaskPriority.HIGH,
                dependencies=[],
                required_tools=["glob", "grep"],
                expected_output="Located resource",
                validation_criteria=["Resource found"]
            ))
        
        elif "compile" in error_message or "syntax" in error_message:
            # Recovery: Fix compilation error
            recovery_subtasks.append(Subtask(
                id=f"{base_id}_fix_compile",
                name=f"Fix compilation error",
                description=f"Fix compilation error in {failed_subtask.name}",
                task_type=TaskType.BUG_FIXING,
                priority=TaskPriority.CRITICAL,
                dependencies=[],
                required_tools=["read_file", "edit_file"],
                expected_output="Compilation fixed",
                validation_criteria=["Code compiles successfully"]
            ))
        
        elif "test" in error_message and "fail" in error_message:
            # Recovery: Fix test failure
            recovery_subtasks.append(Subtask(
                id=f"{base_id}_fix_test",
                name=f"Fix test failure",
                description=f"Fix failing test in {failed_subtask.name}",
                task_type=TaskType.BUG_FIXING,
                priority=TaskPriority.HIGH,
                dependencies=[],
                required_tools=["read_file", "edit_file", "run_test"],
                expected_output="Tests pass",
                validation_criteria=["All tests pass"]
            ))
        
        else:
            # Generic recovery: Retry with different approach
            recovery_subtasks.append(Subtask(
                id=f"{base_id}_retry",
                name=f"Retry with alternative approach",
                description=f"Retry {failed_subtask.name} with alternative approach",
                task_type=failed_subtask.task_type,
                priority=TaskPriority.HIGH,
                dependencies=[],
                required_tools=failed_subtask.required_tools,
                expected_output="Task completed",
                validation_criteria=failed_subtask.validation_criteria
            ))
        
        return recovery_subtasks


def create_autonomous_planner(
    llm_client: Any,
    max_subtasks: int = 10
) -> AutonomousPlanner:
    """Factory function to create AutonomousPlanner.
    
    Args:
        llm_client: LLM client
        max_subtasks: Maximum subtasks
        
    Returns:
        AutonomousPlanner instance
    """
    return AutonomousPlanner(
        llm_client=llm_client,
        max_subtasks=max_subtasks
    )
