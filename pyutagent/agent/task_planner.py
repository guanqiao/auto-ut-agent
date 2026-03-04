"""Task planning module for universal coding agent.

This module provides task decomposition and planning capabilities,
breaking down complex tasks into executable subtasks.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import uuid

from .task_understanding import (
    TaskUnderstanding,
    TaskType,
    TaskPriority,
    TaskComplexity,
)

logger = logging.getLogger(__name__)


class SubTaskStatus(Enum):
    """Status of a subtask."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class SubTaskType(Enum):
    """Types of subtasks."""
    ANALYZE = "analyze"
    READ = "read"
    WRITE = "write"
    EDIT = "edit"
    SEARCH = "search"
    EXECUTE = "execute"
    TEST = "test"
    COMPILE = "compile"
    GENERATE = "generate"
    VALIDATE = "validate"
    REVIEW = "review"


@dataclass
class SubTask:
    """A subtask in the execution plan."""
    id: str
    description: str
    task_type: SubTaskType
    dependencies: List[str] = field(default_factory=list)
    status: SubTaskStatus = SubTaskStatus.PENDING
    priority: int = 5
    estimated_tokens: int = 1000
    required_tools: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "task_type": self.task_type.value,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "priority": self.priority,
            "estimated_tokens": self.estimated_tokens,
            "required_tools": self.required_tools,
            "parameters": self.parameters,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
        }
    
    def mark_started(self):
        """Mark subtask as started."""
        self.status = SubTaskStatus.IN_PROGRESS
        self.started_at = datetime.now()
    
    def mark_completed(self, result: Optional[Dict[str, Any]] = None):
        """Mark subtask as completed."""
        self.status = SubTaskStatus.COMPLETED
        self.completed_at = datetime.now()
        self.result = result
    
    def mark_failed(self, error: str):
        """Mark subtask as failed."""
        self.status = SubTaskStatus.FAILED
        self.completed_at = datetime.now()
        self.error = error
    
    def can_retry(self) -> bool:
        """Check if subtask can be retried."""
        return self.retry_count < self.max_retries
    
    def increment_retry(self):
        """Increment retry count and reset status."""
        self.retry_count += 1
        self.status = SubTaskStatus.PENDING
        self.error = None


@dataclass
class ExecutionPlan:
    """Complete execution plan for a task."""
    id: str
    task_understanding: TaskUnderstanding
    subtasks: List[SubTask] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    current_subtask_id: Optional[str] = None
    total_tokens_used: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "task_understanding": self.task_understanding.to_dict(),
            "subtasks": [st.to_dict() for st in self.subtasks],
            "created_at": self.created_at.isoformat(),
            "current_subtask_id": self.current_subtask_id,
            "total_tokens_used": self.total_tokens_used,
            "metadata": self.metadata,
        }
    
    def get_subtask(self, subtask_id: str) -> Optional[SubTask]:
        """Get subtask by ID."""
        for st in self.subtasks:
            if st.id == subtask_id:
                return st
        return None
    
    def get_pending_subtasks(self) -> List[SubTask]:
        """Get all pending subtasks."""
        return [st for st in self.subtasks if st.status == SubTaskStatus.PENDING]
    
    def get_ready_subtasks(self) -> List[SubTask]:
        """Get subtasks that are ready to execute (dependencies met)."""
        ready = []
        for st in self.subtasks:
            if st.status != SubTaskStatus.PENDING:
                continue
            
            deps_met = all(
                self.get_subtask(dep_id) is not None and 
                self.get_subtask(dep_id).status == SubTaskStatus.COMPLETED
                for dep_id in st.dependencies
            )
            
            if deps_met:
                ready.append(st)
        
        return sorted(ready, key=lambda x: x.priority, reverse=True)
    
    def get_progress(self) -> Dict[str, Any]:
        """Get execution progress."""
        total = len(self.subtasks)
        completed = sum(1 for st in self.subtasks if st.status == SubTaskStatus.COMPLETED)
        failed = sum(1 for st in self.subtasks if st.status == SubTaskStatus.FAILED)
        in_progress = sum(1 for st in self.subtasks if st.status == SubTaskStatus.IN_PROGRESS)
        
        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "in_progress": in_progress,
            "pending": total - completed - failed - in_progress,
            "percentage": (completed / total * 100) if total > 0 else 0,
        }
    
    def is_complete(self) -> bool:
        """Check if plan execution is complete."""
        return all(
            st.status in (SubTaskStatus.COMPLETED, SubTaskStatus.SKIPPED, SubTaskStatus.FAILED)
            for st in self.subtasks
        )
    
    def is_successful(self) -> bool:
        """Check if plan execution was successful."""
        return self.is_complete() and all(
            st.status in (SubTaskStatus.COMPLETED, SubTaskStatus.SKIPPED)
            for st in self.subtasks
        )


TASK_DECOMPOSITION_PROMPT = """You are a task planner for a coding agent. Break down the task into executable subtasks.

## Task Understanding
{task_understanding}

## Available Tools
- read: Read file contents
- write: Write new file
- edit: Edit existing file
- search: Search for code patterns
- execute: Run shell commands
- test: Run tests
- compile: Compile code
- generate: Generate code using LLM
- validate: Validate code quality
- review: Review code changes

## Output Format (JSON)
{{
    "subtasks": [
        {{
            "description": "<description>",
            "task_type": "<analyze|read|write|edit|search|execute|test|compile|generate|validate|review>",
            "dependencies": ["<subtask_id>"],
            "priority": <1-10>,
            "estimated_tokens": <number>,
            "required_tools": ["<tool>"],
            "parameters": {{
                "<param>": "<value>"
            }}
        }}
    ]
}}

Create a detailed execution plan. Each subtask should have a unique ID like "st_1", "st_2", etc.
Respond with only valid JSON.
"""


class TaskPlanner:
    """Plans and decomposes tasks into executable subtasks."""
    
    DEFAULT_TEMPLATES: Dict[TaskType, List[Dict[str, Any]]] = {
        TaskType.UT_GENERATION: [
            {"description": "Analyze target file structure", "task_type": SubTaskType.ANALYZE, "priority": 10},
            {"description": "Parse Java class for test generation", "task_type": SubTaskType.READ, "priority": 9},
            {"description": "Generate unit tests", "task_type": SubTaskType.GENERATE, "priority": 8},
            {"description": "Compile generated tests", "task_type": SubTaskType.COMPILE, "priority": 7},
            {"description": "Run tests and verify", "task_type": SubTaskType.TEST, "priority": 6},
            {"description": "Analyze coverage", "task_type": SubTaskType.VALIDATE, "priority": 5},
        ],
        TaskType.BUG_FIX: [
            {"description": "Analyze error or issue description", "task_type": SubTaskType.ANALYZE, "priority": 10},
            {"description": "Search for related code", "task_type": SubTaskType.SEARCH, "priority": 9},
            {"description": "Read affected files", "task_type": SubTaskType.READ, "priority": 8},
            {"description": "Identify root cause", "task_type": SubTaskType.ANALYZE, "priority": 7},
            {"description": "Implement fix", "task_type": SubTaskType.EDIT, "priority": 6},
            {"description": "Run tests to verify fix", "task_type": SubTaskType.TEST, "priority": 5},
        ],
        TaskType.CODE_REFACTORING: [
            {"description": "Analyze current code structure", "task_type": SubTaskType.ANALYZE, "priority": 10},
            {"description": "Read files to refactor", "task_type": SubTaskType.READ, "priority": 9},
            {"description": "Plan refactoring changes", "task_type": SubTaskType.ANALYZE, "priority": 8},
            {"description": "Apply refactoring", "task_type": SubTaskType.EDIT, "priority": 7},
            {"description": "Run tests to verify", "task_type": SubTaskType.TEST, "priority": 6},
            {"description": "Review changes", "task_type": SubTaskType.REVIEW, "priority": 5},
        ],
        TaskType.FEATURE_ADD: [
            {"description": "Analyze requirements", "task_type": SubTaskType.ANALYZE, "priority": 10},
            {"description": "Search for related code", "task_type": SubTaskType.SEARCH, "priority": 9},
            {"description": "Read relevant files", "task_type": SubTaskType.READ, "priority": 8},
            {"description": "Design implementation", "task_type": SubTaskType.ANALYZE, "priority": 7},
            {"description": "Implement feature", "task_type": SubTaskType.WRITE, "priority": 6},
            {"description": "Write tests", "task_type": SubTaskType.GENERATE, "priority": 5},
            {"description": "Run tests", "task_type": SubTaskType.TEST, "priority": 4},
        ],
        TaskType.CODE_REVIEW: [
            {"description": "Read code to review", "task_type": SubTaskType.READ, "priority": 10},
            {"description": "Analyze code quality", "task_type": SubTaskType.ANALYZE, "priority": 9},
            {"description": "Check for issues", "task_type": SubTaskType.VALIDATE, "priority": 8},
            {"description": "Generate review report", "task_type": SubTaskType.GENERATE, "priority": 7},
        ],
        TaskType.DOCUMENTATION: [
            {"description": "Read code to document", "task_type": SubTaskType.READ, "priority": 10},
            {"description": "Analyze code structure", "task_type": SubTaskType.ANALYZE, "priority": 9},
            {"description": "Generate documentation", "task_type": SubTaskType.GENERATE, "priority": 8},
            {"description": "Write documentation file", "task_type": SubTaskType.WRITE, "priority": 7},
        ],
        TaskType.CODE_EXPLANATION: [
            {"description": "Read code to explain", "task_type": SubTaskType.READ, "priority": 10},
            {"description": "Analyze code logic", "task_type": SubTaskType.ANALYZE, "priority": 9},
            {"description": "Generate explanation", "task_type": SubTaskType.GENERATE, "priority": 8},
        ],
        TaskType.TEST_DEBUG: [
            {"description": "Read failing test", "task_type": SubTaskType.READ, "priority": 10},
            {"description": "Run test to see error", "task_type": SubTaskType.TEST, "priority": 9},
            {"description": "Analyze failure", "task_type": SubTaskType.ANALYZE, "priority": 8},
            {"description": "Fix test or code", "task_type": SubTaskType.EDIT, "priority": 7},
            {"description": "Verify fix", "task_type": SubTaskType.TEST, "priority": 6},
        ],
    }
    
    def __init__(self, llm_client=None):
        """Initialize task planner.
        
        Args:
            llm_client: Optional LLM client for advanced planning
        """
        self.llm_client = llm_client
    
    def create_plan(
        self, 
        understanding: TaskUnderstanding,
        use_llm: bool = True
    ) -> ExecutionPlan:
        """Create execution plan from task understanding.
        
        Args:
            understanding: Task understanding
            use_llm: Whether to use LLM for planning
            
        Returns:
            ExecutionPlan with subtasks
        """
        plan_id = self._generate_plan_id()
        
        if use_llm and self.llm_client:
            subtasks = self._plan_with_llm(understanding)
        else:
            subtasks = self._plan_from_template(understanding)
        
        return ExecutionPlan(
            id=plan_id,
            task_understanding=understanding,
            subtasks=subtasks,
        )
    
    def refine_plan(
        self,
        plan: ExecutionPlan,
        feedback: Dict[str, Any]
    ) -> ExecutionPlan:
        """Refine plan based on execution feedback.
        
        Args:
            plan: Current execution plan
            feedback: Feedback from execution
            
        Returns:
            Refined execution plan
        """
        failed_subtasks = [
            st for st in plan.subtasks 
            if st.status == SubTaskStatus.FAILED
        ]
        
        for subtask in failed_subtasks:
            if subtask.can_retry():
                subtask.increment_retry()
            else:
                alternative = self._create_alternative_subtask(subtask, feedback)
                if alternative:
                    plan.subtasks.append(alternative)
        
        return plan
    
    def _generate_plan_id(self) -> str:
        """Generate unique plan ID."""
        return f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    def _generate_subtask_id(self, index: int) -> str:
        """Generate subtask ID."""
        return f"st_{index + 1}"
    
    def _plan_from_template(self, understanding: TaskUnderstanding) -> List[SubTask]:
        """Create plan from predefined template."""
        template = self.DEFAULT_TEMPLATES.get(
            understanding.task_type,
            self.DEFAULT_TEMPLATES[TaskType.CODE_REVIEW]
        )
        
        subtasks = []
        prev_id = None
        
        for i, step in enumerate(template):
            subtask_id = self._generate_subtask_id(i)
            dependencies = [prev_id] if prev_id else []
            
            subtask = SubTask(
                id=subtask_id,
                description=step["description"],
                task_type=step["task_type"],
                dependencies=dependencies,
                priority=step.get("priority", 5),
                estimated_tokens=step.get("estimated_tokens", 1000),
                required_tools=step.get("required_tools", []),
                parameters=step.get("parameters", {}),
            )
            
            subtasks.append(subtask)
            prev_id = subtask_id
        
        self._customize_subtasks(subtasks, understanding)
        
        return subtasks
    
    def _customize_subtasks(
        self, 
        subtasks: List[SubTask], 
        understanding: TaskUnderstanding
    ) -> None:
        """Customize subtasks based on task understanding."""
        target_files = understanding.target_scope.files
        
        for subtask in subtasks:
            if subtask.task_type == SubTaskType.READ and target_files:
                subtask.parameters["files"] = target_files
            elif subtask.task_type == SubTaskType.ANALYZE:
                subtask.parameters["scope"] = understanding.target_scope.to_dict()
            elif subtask.task_type == SubTaskType.GENERATE:
                subtask.parameters["requirements"] = understanding.requirements
    
    async def _plan_with_llm(self, understanding: TaskUnderstanding) -> List[SubTask]:
        """Create plan using LLM."""
        try:
            prompt = TASK_DECOMPOSITION_PROMPT.format(
                task_understanding=understanding.to_dict()
            )
            
            response = await self.llm_client.generate(prompt)
            
            import json
            result = json.loads(self._extract_json(response))
            
            return self._parse_llm_subtasks(result)
            
        except Exception as e:
            logger.warning(f"[TaskPlanner] LLM planning failed: {e}, using template")
            return self._plan_from_template(understanding)
    
    def _extract_json(self, response: str) -> str:
        """Extract JSON from LLM response."""
        import re
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            return json_match.group()
        return response
    
    def _parse_llm_subtasks(self, result: Dict[str, Any]) -> List[SubTask]:
        """Parse LLM result into subtasks."""
        subtasks = []
        
        for i, st_data in enumerate(result.get("subtasks", [])):
            task_type_str = st_data.get("task_type", "analyze")
            try:
                task_type = SubTaskType(task_type_str)
            except ValueError:
                task_type = SubTaskType.ANALYZE
            
            subtask = SubTask(
                id=self._generate_subtask_id(i),
                description=st_data.get("description", ""),
                task_type=task_type,
                dependencies=st_data.get("dependencies", []),
                priority=st_data.get("priority", 5),
                estimated_tokens=st_data.get("estimated_tokens", 1000),
                required_tools=st_data.get("required_tools", []),
                parameters=st_data.get("parameters", {}),
            )
            
            subtasks.append(subtask)
        
        return subtasks
    
    def _create_alternative_subtask(
        self, 
        failed_subtask: SubTask, 
        feedback: Dict[str, Any]
    ) -> Optional[SubTask]:
        """Create alternative subtask for failed one."""
        error_type = feedback.get("error_type", "unknown")
        
        if error_type == "compilation_error":
            return SubTask(
                id=f"{failed_subtask.id}_alt",
                description=f"Fix compilation errors from {failed_subtask.id}",
                task_type=SubTaskType.EDIT,
                dependencies=[],
                priority=failed_subtask.priority,
                parameters={"fix_type": "compilation"},
            )
        elif error_type == "test_failure":
            return SubTask(
                id=f"{failed_subtask.id}_alt",
                description=f"Fix test failures from {failed_subtask.id}",
                task_type=SubTaskType.EDIT,
                dependencies=[],
                priority=failed_subtask.priority,
                parameters={"fix_type": "test"},
            )
        
        return None


class PlanExecutor:
    """Executes execution plans."""
    
    def __init__(
        self,
        tool_registry: Optional[Dict[str, Callable]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """Initialize plan executor.
        
        Args:
            tool_registry: Registry of available tools
            progress_callback: Optional callback for progress updates
        """
        self.tool_registry = tool_registry or {}
        self.progress_callback = progress_callback
    
    async def execute_plan(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """Execute an execution plan.
        
        Args:
            plan: Execution plan to execute
            
        Returns:
            Execution results
        """
        results = {
            "plan_id": plan.id,
            "success": False,
            "subtask_results": {},
            "errors": [],
        }
        
        while not plan.is_complete():
            ready_subtasks = plan.get_ready_subtasks()
            
            if not ready_subtasks:
                blocked = [st for st in plan.subtasks if st.status == SubTaskStatus.PENDING]
                if blocked:
                    results["errors"].append(f"Deadlock detected: {len(blocked)} subtasks blocked")
                break
            
            for subtask in ready_subtasks:
                result = await self._execute_subtask(subtask)
                results["subtask_results"][subtask.id] = result
                
                if self.progress_callback:
                    self.progress_callback({
                        "plan_id": plan.id,
                        "subtask_id": subtask.id,
                        "status": subtask.status.value,
                        "progress": plan.get_progress(),
                    })
        
        results["success"] = plan.is_successful()
        results["progress"] = plan.get_progress()
        
        return results
    
    async def _execute_subtask(self, subtask: SubTask) -> Dict[str, Any]:
        """Execute a single subtask."""
        subtask.mark_started()
        
        try:
            tool_name = subtask.task_type.value
            tool = self.tool_registry.get(tool_name)
            
            if not tool:
                raise ValueError(f"Tool not found: {tool_name}")
            
            result = await tool(**subtask.parameters)
            subtask.mark_completed(result)
            
            return {"success": True, "result": result}
            
        except Exception as e:
            subtask.mark_failed(str(e))
            return {"success": False, "error": str(e)}
