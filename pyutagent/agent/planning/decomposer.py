"""Task Decomposer Module.

Provides intelligent task decomposition strategies:
- Atomic: Single-step tasks
- Sequential: Linear multi-step tasks
- Composite: Complex tasks with sub-tasks
- Parallel: Independent tasks that can run concurrently
- Template-based: Using predefined templates

This is part of Phase 3 Week 17-18: Task Planning Enhancement.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

from pyutagent.agent.execution.execution_plan import Step, StepType, SubTask, SubTaskType
from pyutagent.agent.task_understanding import TaskUnderstanding

if TYPE_CHECKING:
    from pyutagent.llm.llm_client import LLMClient

logger = logging.getLogger(__name__)


class DecompositionStrategy(Enum):
    """Strategy for decomposing a task."""
    ATOMIC = "atomic"
    SEQUENTIAL = "sequential"
    COMPOSITE = "composite"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    TEMPLATE_BASED = "template"


@dataclass
class DecompositionContext:
    """Context for task decomposition."""
    task_description: str
    complexity: int = 5
    task_type: str = "default"
    available_tools: List[str] = field(default_factory=list)
    available_skills: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    parent_task_id: Optional[str] = None
    depth: int = 0
    max_depth: int = 3


class TaskDecomposer(ABC):
    """Abstract base class for task decomposers."""

    @abstractmethod
    def get_strategy(self, complexity: int) -> DecompositionStrategy:
        """Determine the decomposition strategy for a task."""
        pass

    @abstractmethod
    def decompose(self, context: DecompositionContext) -> List[Step]:
        """Decompose a task into steps."""
        pass


class SimpleTaskDecomposer(TaskDecomposer):
    """Simple task decomposer based on complexity."""

    def __init__(self):
        self._complexity_thresholds = {
            DecompositionStrategy.ATOMIC: 2,
            DecompositionStrategy.SEQUENTIAL: 5,
            DecompositionStrategy.COMPOSITE: 8,
        }

    def get_strategy(self, complexity: int) -> DecompositionStrategy:
        """Determine strategy based on task complexity."""
        if complexity <= self._complexity_thresholds[DecompositionStrategy.ATOMIC]:
            return DecompositionStrategy.ATOMIC
        elif complexity <= self._complexity_thresholds[DecompositionStrategy.SEQUENTIAL]:
            return DecompositionStrategy.SEQUENTIAL
        else:
            return DecompositionStrategy.COMPOSITE

    def decompose(self, context: DecompositionContext) -> List[Step]:
        """Decompose task based on strategy."""
        strategy = self.get_strategy(context.complexity)
        
        if strategy == DecompositionStrategy.ATOMIC:
            return self._decompose_atomic(context)
        elif strategy == DecompositionStrategy.SEQUENTIAL:
            return self._decompose_sequential(context)
        else:
            return self._decompose_composite(context)

    def _decompose_atomic(self, context: DecompositionContext) -> List[Step]:
        """Create a single atomic step."""
        return [
            Step(
                id=f"step_atomic",
                name=context.task_description[:50],
                description=context.task_description,
                step_type=StepType.ACTION,
            )
        ]

    def _decompose_sequential(self, context: DecompositionContext) -> List[Step]:
        """Decompose into sequential steps."""
        steps_text = context.task_description.split("\n")
        steps = [s.strip() for s in steps_text if s.strip()]
        
        if len(steps) <= 1:
            return self._decompose_atomic(context)
        
        result = []
        prev_id = None
        
        for i, step_desc in enumerate(steps):
            step = Step(
                id=f"step_{i}",
                name=f"Step {i + 1}",
                description=step_desc,
                step_type=StepType.ACTION,
                dependencies=[prev_id] if prev_id else [],
            )
            result.append(step)
            prev_id = step.id
        
        return result

    def _decompose_composite(self, context: DecompositionContext) -> List[Step]:
        """Decompose into composite structure."""
        steps = self._decompose_sequential(context)
        
        if context.depth < context.max_depth:
            for step in steps:
                if len(step.description) > 100:
                    nested_context = DecompositionContext(
                        task_description=step.description,
                        complexity=context.complexity // 2,
                        available_tools=context.available_tools,
                        available_skills=context.available_skills,
                        depth=context.depth + 1,
                        max_depth=context.max_depth,
                    )
                    step.substeps = self.decompose(nested_context)
        
        return steps


class TemplateTaskDecomposer(TaskDecomposer):
    """Template-based task decomposer."""

    DEFAULT_TEMPLATES = {
        "test_generation": [
            {"name": "Analyze target class", "type": "analyze", "action": "analyze_class"},
            {"name": "Identify test cases", "type": "plan", "action": "identify_tests"},
            {"name": "Generate tests", "type": "execute", "action": "generate_tests"},
            {"name": "Run tests", "type": "verify", "action": "run_tests"},
        ],
        "code_refactoring": [
            {"name": "Analyze code", "type": "analyze", "action": "analyze_code"},
            {"name": "Identify issues", "type": "plan", "action": "identify_issues"},
            {"name": "Apply refactoring", "type": "execute", "action": "refactor"},
            {"name": "Verify changes", "type": "verify", "action": "verify"},
        ],
        "bug_fix": [
            {"name": "Reproduce bug", "type": "analyze", "action": "reproduce"},
            {"name": "Identify cause", "type": "analyze", "action": "identify_cause"},
            {"name": "Implement fix", "type": "execute", "action": "fix"},
            {"name": "Verify fix", "type": "verify", "action": "verify"},
        ],
    }

    def __init__(self, templates: Optional[Dict[str, List[Dict]]] = None):
        self._templates = templates or self.DEFAULT_TEMPLATES

    def get_strategy(self, task: Optional[TaskUnderstanding] = None, complexity: int = 5) -> DecompositionStrategy:
        """Return template-based strategy."""
        return DecompositionStrategy.TEMPLATE_BASED

    def decompose(
        self,
        task: Optional[TaskUnderstanding] = None,
        context: Optional[DecompositionContext] = None,
    ) -> List[SubTask]:
        """Decompose using templates."""
        if context is None:
            context = DecompositionContext()
        
        if task is None and context.task_understanding:
            task = context.task_understanding
        elif task is None:
            task = TaskUnderstanding(
                task_id="default",
                name="Default Task",
                description=context.task_description,
                complexity=context.complexity,
                type=context.task_type,
            )
        
        task_type = getattr(task, "type", context.task_type) or "default"
        template = self._templates.get(task_type, self._get_default_template())
        
        task_id = getattr(task, "task_id", "default") or "default"
        
        subtasks = []
        prev_id = None
        
        for i, step in enumerate(template):
            st = SubTask(
                id=f"{task_id}_{step['type']}_{i}",
                name=step["name"],
                description=step.get("description", step["name"]),
                task_type=self._map_step_type(step["type"]),
                action=step["action"],
                priority=getattr(task, "priority", 5) if task else 5,
                dependencies=[prev_id] if prev_id else [],
            )
            subtasks.append(st)
            prev_id = st.id
        
        return subtasks

    def _get_default_template(self) -> List[Dict]:
        """Get default template."""
        return [
            {"name": "Analyze", "type": StepType.ANALYZE, "description": "Analyze the task"},
            {"name": "Execute", "type": StepType.ACTION, "description": "Execute the task"},
            {"name": "Verify", "type": StepType.TEST, "description": "Verify results"},
        ]


class LLMTaskDecomposer(TaskDecomposer):
    """LLM-powered task decomposer."""

    DECOMPOSITION_PROMPT = """You are a task decomposition expert. Break down the following task into steps.

Task: {task_description}
Complexity: {complexity}

Available tools: {tools}
Available skills: {skills}

Provide a JSON array of steps with the following structure:
[
  {{
    "name": "Step name",
    "description": "Detailed description",
    "type": "analyze|plan|action|test",
    "dependencies": ["dependency_step_name"]
  }}
]

Only output the JSON array, no other text."""

    def __init__(self, llm_client: "LLMClient"):
        self._llm_client = llm_client

    def get_strategy(self, complexity: int) -> DecompositionStrategy:
        """Return LLM-based strategy."""
        return DecompositionStrategy.COMPOSITE

    def decompose(self, context: DecompositionContext) -> List[Step]:
        """Decompose using LLM."""
        import json
        
        prompt = self.DECOMPOSITION_PROMPT.format(
            task_description=context.task_description,
            complexity=context.complexity,
            tools=", ".join(context.available_tools) or "none",
            skills=", ".join(context.available_skills) or "none",
        )
        
        try:
            response = asyncio.get_event_loop().run_until_complete(
                self._llm_client.generate(prompt)
            )
            step_data = json.loads(response)
            
            steps = []
            name_to_id = {}
            
            for i, data in enumerate(step_data):
                step = Step(
                    id=f"step_llm_{i}",
                    name=data.get("name", f"Step {i}"),
                    description=data.get("description", ""),
                    step_type=self._map_type(data.get("type", "action")),
                )
                steps.append(step)
                name_to_id[data.get("name", "")] = step.id
            
            for i, data in enumerate(step_data):
                deps = data.get("dependencies", [])
                steps[i].dependencies = [
                    name_to_id[dep] for dep in deps if dep in name_to_id
                ]
            
            return steps
            
        except Exception as e:
            logger.error(f"LLM decomposition failed: {e}")
            return SimpleTaskDecomposer().decompose(context)

    def _map_type(self, type_str: str) -> StepType:
        """Map string type to StepType."""
        mapping = {
            "analyze": StepType.ANALYZE,
            "plan": StepType.PLAN,
            "action": StepType.ACTION,
            "test": StepType.TEST,
        }
        return mapping.get(type_str, StepType.ACTION)


def get_task_decomposer(
    use_templates: bool = False,
    llm_client: Optional["LLMClient"] = None,
) -> TaskDecomposer:
    """Get appropriate task decomposer.
    
    Args:
        use_templates: Whether to use template-based decomposition
        llm_client: LLM client for LLM-based decomposition
        
    Returns:
        TaskDecomposer instance
    """
    if llm_client is not None:
        return LLMTaskDecomposer(llm_client)
    elif use_templates:
        return TemplateTaskDecomposer()
    else:
        return SimpleTaskDecomposer()
