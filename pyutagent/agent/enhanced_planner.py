"""Enhanced planning system with clarification questions.

This module provides:
- ClarificationQuestion: Structured clarification question
- PlanMode: Enhanced plan mode with iterative refinement
- PlanClarifier: Generate clarification questions based on task
- InteractivePlanner: Planner that asks questions before executing
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class QuestionType(Enum):
    """Types of clarification questions."""
    SCOPE = "scope"  # What is the scope of the task?
    LANGUAGE = "language"  # What programming language?
    FRAMEWORK = "framework"  # What framework should be used?
    STYLE = "style"  # Code style preferences?
    TESTING = "testing"  # Testing requirements?
    DEPENDENCIES = "dependencies"  # Any dependencies to consider?
    OUTPUT = "output"  # Where should output go?
    CONSTRAINTS = "constraints"  # Any constraints?
    PRIORITY = "priority"  # What's the priority?
    TIMELINE = "timeline"  # Any timeline constraints?


@dataclass
class ClarificationQuestion:
    """A clarification question to ask the user."""
    question_id: str
    question_type: QuestionType
    question_text: str
    options: List[str] = field(default_factory=list)
    requires_input: bool = True
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.question_id,
            "type": self.question_type.value,
            "text": self.question_text,
            "options": self.options,
            "requires_input": self.requires_input,
            "context": self.context,
            "priority": self.priority
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClarificationQuestion':
        return cls(
            question_id=data.get("id", ""),
            question_type=QuestionType(data.get("type", "scope")),
            question_text=data.get("text", ""),
            options=data.get("options", []),
            requires_input=data.get("requires_input", True),
            context=data.get("context", {}),
            priority=data.get("priority", 5)
        )


@dataclass
class UserResponse:
    """User's response to a clarification question."""
    question_id: str
    response: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "response": self.response,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class PlanRefinement:
    """Refined plan based on user responses."""
    original_task: str
    refined_task: str
    questions_asked: List[ClarificationQuestion] = field(default_factory=list)
    user_responses: List[UserResponse] = field(default_factory=list)
    refined_steps: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_task": self.original_task,
            "refined_task": self.refined_task,
            "questions_asked": [q.to_dict() for q in self.questions_asked],
            "user_responses": [r.to_dict() for r in self.user_responses],
            "refined_steps": self.refined_steps,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat()
        }


class QuestionGenerator:
    """Generate clarification questions based on task analysis."""

    def __init__(self):
        self._templates = self._load_templates()

    def _load_templates(self) -> Dict[QuestionType, List[str]]:
        return {
            QuestionType.SCOPE: [
                "What specific files or components should be included?",
                "Should this cover the entire project or just specific modules?",
                "Are there any files that should be excluded?"
            ],
            QuestionType.LANGUAGE: [
                "What programming language(s) are you working with?",
                "Should the code be compatible with a specific language version?"
            ],
            QuestionType.FRAMEWORK: [
                "Are you using any specific frameworks (e.g., Spring, Django, React)?",
                "Should the code follow any specific framework conventions?"
            ],
            QuestionType.STYLE: [
                "Do you have any specific code style preferences?",
                "Should this follow any linting or formatting rules?"
            ],
            QuestionType.TESTING: [
                "What testing framework should be used?",
                "Should unit tests be included?",
                "What's the desired test coverage level?"
            ],
            QuestionType.DEPENDENCIES: [
                "Are there any existing dependencies to consider?",
                "Should new dependencies be added if needed?"
            ],
            QuestionType.OUTPUT: [
                "Where should the generated code be placed?",
                "Should existing files be modified or new ones created?"
            ],
            QuestionType.CONSTRAINTS: [
                "Are there any specific constraints or requirements?",
                "Any performance or security considerations?"
            ],
            QuestionType.PRIORITY: [
                "What's the priority of this task?",
                "Should we focus on functionality or code quality first?"
            ],
            QuestionType.TIMELINE: [
                "Is there a specific deadline or timeline?",
                "Should we prioritize speed over completeness?"
            ]
        }

    def generate_questions(self, task_description: str, context: Dict[str, Any] = None) -> List[ClarificationQuestion]:
        """Generate clarification questions based on task.

        Args:
            task_description: The user's task description
            context: Additional context about the project

        Returns:
            List of clarification questions
        """
        questions = []
        context = context or {}

        task_lower = task_description.lower()
        question_id = 0

        if any(word in task_lower for word in ["generate", "create", "write", "add"]):
            if "test" in task_lower or "spec" in task_lower:
                questions.append(ClarificationQuestion(
                    question_id=f"q_{question_id}",
                    question_type=QuestionType.TESTING,
                    question_text="What testing framework should be used?",
                    options=self._get_testing_options(context),
                    priority=10
                ))
                question_id += 1

                questions.append(ClarificationQuestion(
                    question_id=f"q_{question_id}",
                    question_type=QuestionType.SCOPE,
                    question_text="Which classes or methods should have tests?",
                    priority=9
                ))
                question_id += 1
            else:
                questions.append(ClarificationQuestion(
                    question_id=f"q_{question_id}",
                    question_type=QuestionType.OUTPUT,
                    question_text="Should this create new files or modify existing ones?",
                    options=["Create new files", "Modify existing files", "Both"],
                    priority=8
                ))
                question_id += 1

        if any(word in task_lower for word in ["refactor", "improve", "clean"]):
            questions.append(ClarificationQuestion(
                question_id=f"q_{question_id}",
                question_type=QuestionType.STYLE,
                question_text="What code style should be followed?",
                options=self._get_style_options(context),
                priority=7
            ))
            question_id += 1

            questions.append(ClarificationQuestion(
                question_id=f"q_{question_id}",
                question_type=QuestionType.CONSTRAINTS,
                question_text="Are there any specific constraints to consider?",
                priority=6
            ))
            question_id += 1

        if any(word in task_lower for word in ["fix", "bug", "error", "debug"]):
            questions.append(ClarificationQuestion(
                question_id=f"q_{question_id}",
                question_type=QuestionType.SCOPE,
                question_text="What is the expected behavior?",
                priority=9
            ))
            question_id += 1

            questions.append(ClarificationQuestion(
                question_id=f"q_{question_id}",
                question_type=QuestionType.TESTING,
                question_text="Should we add tests to prevent regression?",
                options=["Yes, add tests", "No tests needed", "Only if easy to add"],
                priority=5
            ))
            question_id += 1

        if "migrate" in task_lower or "convert" in task_lower:
            questions.append(ClarificationQuestion(
                question_id=f"q_{question_id}",
                question_type=QuestionType.FRAMEWORK,
                question_text="What is the target framework/version?",
                priority=10
            ))
            question_id += 1

            questions.append(ClarificationQuestion(
                question_id=f"q_{question_id}",
                question_type=QuestionType.DEPENDENCIES,
                question_text="Should old dependencies be removed?",
                options=["Yes, clean up", "Keep both for now"],
                priority=7
            ))
            question_id += 1

        questions.extend(self._generate_generic_questions(task_lower, question_id))

        questions.sort(key=lambda q: q.priority, reverse=True)

        return questions[:5]

    def _generate_generic_questions(self, task_lower: str, start_id: int) -> List[ClarificationQuestion]:
        """Generate generic questions if needed."""
        questions = []

        if len(task_lower) < 50:
            questions.append(ClarificationQuestion(
                question_id=f"q_{start_id}",
                question_type=QuestionType.SCOPE,
                question_text="Could you provide more details about what you want to accomplish?",
                priority=10,
                requires_input=True
            ))

        return questions

    def _get_testing_options(self, context: Dict[str, Any]) -> List[str]:
        """Get testing framework options based on context."""
        language = context.get("language", "").lower()

        if "java" in language:
            return ["JUnit 4", "JUnit 5", "TestNG", "Spock"]
        elif "python" in language:
            return ["pytest", "unittest", "doctest"]
        elif "javascript" in language or "typescript" in language:
            return ["Jest", "Mocha", "Vitest", "JUnit"]
        else:
            return ["Default framework", "No preference"]

    def _get_style_options(self, context: Dict[str, Any]) -> List[str]:
        """Get code style options."""
        return [
            "Follow existing project style",
            "Google style guide",
            "Airbnb style guide",
            "PEP 8 (Python)",
            "No preference"
        ]


class PlanClarifier:
    """Handle plan clarification with user."""

    def __init__(self):
        self._question_generator = QuestionGenerator()
        self._current_plan: Optional[PlanRefinement] = None

    def analyze_task(self, task_description: str, context: Dict[str, Any] = None) -> PlanRefinement:
        """Analyze task and prepare for clarification.

        Args:
            task_description: The user's task description
            context: Additional context

        Returns:
            Plan refinement with questions to ask
        """
        questions = self._question_generator.generate_questions(task_description, context)

        self._current_plan = PlanRefinement(
            original_task=task_description,
            refined_task=task_description,
            questions_asked=questions
        )

        logger.info(f"[PlanClarifier] Generated {len(questions)} clarification questions")

        return self._current_plan

    def get_next_question(self) -> Optional[ClarificationQuestion]:
        """Get the next question to ask the user.

        Returns:
            Next question or None if no more questions
        """
        if not self._current_plan:
            return None

        asked_ids = {q.question_id for q in self._current_plan.user_responses}

        for question in self._current_plan.questions_asked:
            if question.question_id not in asked_ids:
                return question

        return None

    def add_response(self, question_id: str, response: str) -> bool:
        """Add a user response to the current plan.

        Args:
            question_id: ID of the question
            response: User's response

        Returns:
            True if response was added successfully
        """
        if not self._current_plan:
            return False

        question = next(
            (q for q in self._current_plan.questions_asked if q.question_id == question_id),
            None
        )

        if not question:
            return False

        user_response = UserResponse(
            question_id=question_id,
            response=response
        )

        self._current_plan.user_responses.append(user_response)

        self._refine_task()

        logger.info(f"[PlanClarifier] Added response for question: {question_id}")
        return True

    def _refine_task(self):
        """Refine the task based on user responses."""
        if not self._current_plan:
            return

        refined_parts = [self._current_plan.original_task]

        for response in self._current_plan.user_responses:
            question = next(
                (q for q in self._current_plan.questions_asked if q.question_id == response.question_id),
                None
            )

            if question and response.response:
                refined_parts.append(f"[{question.question_type.value}: {response.response}]")

        self._current_plan.refined_task = " ".join(refined_parts)

        self._generate_refined_steps()

    def _generate_refined_steps(self):
        """Generate refined execution steps based on responses."""
        if not self._current_plan:
            return

        steps = []

        responses_dict = {r.question_id: r.response for r in self._current_plan.user_responses}

        for question in self._current_plan.questions_asked:
            if question.question_id in responses_dict:
                response = responses_dict[question.question_id]

                if question.question_type == QuestionType.TESTING:
                    if "yes" in response.lower() or "add" in response.lower():
                        steps.append({
                            "action": "generate_tests",
                            "framework": response,
                            "priority": 10
                        })

                elif question.question_type == QuestionType.SCOPE:
                    steps.append({
                        "action": "analyze_scope",
                        "details": response,
                        "priority": 9
                    })

                elif question.question_type == QuestionType.STYLE:
                    steps.append({
                        "action": "apply_style",
                        "style": response,
                        "priority": 7
                    })

        self._current_plan.refined_steps = steps

        confidence = min(1.0, 0.5 + len(self._current_plan.user_responses) * 0.1)
        self._current_plan.confidence = confidence

    def is_complete(self) -> bool:
        """Check if clarification is complete.

        Returns:
            True if all important questions are answered
        """
        if not self._current_plan:
            return True

        important_questions = [q for q in self._current_plan.questions_asked if q.priority >= 7]
        answered_ids = {r.question_id for r in self._current_plan.user_responses}

        return all(q.question_id in answered_ids for q in important_questions)

    def get_refined_plan(self) -> Optional[PlanRefinement]:
        """Get the refined plan.

        Returns:
            Refined plan or None
        """
        return self._current_plan

    def reset(self):
        """Reset the clarifier."""
        self._current_plan = None


class InteractivePlanner:
    """Planner that can ask clarification questions before executing.

    Features:
    - Pre-execution clarification
    - Iterative refinement
    - User-guided planning
    """

    def __init__(self):
        self._clarifier = PlanClarifier()
        self._execution_callback: Optional[Callable] = None

    def set_execution_callback(self, callback: Callable):
        """Set callback for plan execution.

        Args:
            callback: Function to call with refined plan
        """
        self._execution_callback = callback

    async def plan_with_clarification(self, task_description: str,
                                       context: Dict[str, Any] = None) -> PlanRefinement:
        """Plan a task with user clarification.

        Args:
            task_description: Initial task description
            context: Additional context

        Returns:
            Refined plan
        """
        plan = self._clarifier.analyze_task(task_description, context)

        while not self._clarifier.is_complete():
            question = self._clarifier.get_next_question()

            if not question:
                break

            logger.info(f"[InteractivePlanner] Asking question: {question.question_text}")

            return plan

        if self._execution_callback and self._clarifier.is_complete():
            refined_plan = self._clarifier.get_refined_plan()
            if refined_plan:
                await self._execution_callback(refined_plan)

        return self._clarifier.get_refined_plan()

    def add_user_response(self, question_id: str, response: str) -> bool:
        """Add a user response and check if planning is complete.

        Args:
            question_id: ID of the question
            response: User's response

        Returns:
            True if planning is now complete
        """
        result = self._clarifier.add_response(question_id, response)
        if result:
            return self._clarifier.is_complete()
        return False

    def get_current_plan(self) -> Optional[PlanRefinement]:
        """Get current plan state.

        Returns:
            Current plan or None
        """
        return self._clarifier.get_refined_plan()

    def get_pending_questions(self) -> List[ClarificationQuestion]:
        """Get list of pending questions.

        Returns:
            List of questions not yet answered
        """
        if not self._clarifier.get_refined_plan():
            return []

        answered_ids = {r.question_id for r in self._clarifier.get_refined_plan().user_responses}

        return [
            q for q in self._clarifier.get_refined_plan().questions_asked
            if q.question_id not in answered_ids
        ]

    def reset(self):
        """Reset the planner."""
        self._clarifier.reset()


def get_global_planner() -> InteractivePlanner:
    """Get global interactive planner instance."""
    global _global_planner
    if _global_planner is None:
        _global_planner = InteractivePlanner()
    return _global_planner


_global_planner: Optional[InteractivePlanner] = None
