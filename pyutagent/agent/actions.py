"""Action registry for agent tools."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Callable
from enum import Enum, auto


class ActionType(Enum):
    """Types of actions the agent can perform."""
    PARSE_CODE = auto()
    GENERATE_TESTS = auto()
    COMPILE = auto()
    RUN_TESTS = auto()
    ANALYZE_COVERAGE = auto()
    FIX_ERRORS = auto()
    STORE_MEMORY = auto()
    RETRIEVE_MEMORY = auto()


@dataclass
class ActionResult:
    """Result of an action execution."""
    success: bool
    message: str
    data: Dict[str, Any]
    action_type: ActionType


class Action(ABC):
    """Base class for agent actions."""
    
    def __init__(self, name: str, action_type: ActionType, description: str = ""):
        """Initialize action.
        
        Args:
            name: Action name
            action_type: Type of action
            description: Action description
        """
        self.name = name
        self.action_type = action_type
        self.description = description
    
    @abstractmethod
    async def execute(self, **kwargs) -> ActionResult:
        """Execute the action.
        
        Args:
            **kwargs: Action-specific parameters
            
        Returns:
            ActionResult with execution results
        """
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the action schema for LLM function calling."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._get_parameters()
        }
    
    @abstractmethod
    def _get_parameters(self) -> Dict[str, Any]:
        """Get parameter schema for this action."""
        pass


class ActionRegistry:
    """Registry for agent actions."""
    
    def __init__(self):
        """Initialize action registry."""
        self._actions: Dict[ActionType, Action] = {}
        self._action_by_name: Dict[str, Action] = {}
    
    def register(self, action: Action):
        """Register an action.
        
        Args:
            action: Action to register
        """
        self._actions[action.action_type] = action
        self._action_by_name[action.name] = action
    
    def get(self, action_type: ActionType) -> Optional[Action]:
        """Get action by type.
        
        Args:
            action_type: Type of action
            
        Returns:
            Action if found, None otherwise
        """
        return self._actions.get(action_type)
    
    def get_by_name(self, name: str) -> Optional[Action]:
        """Get action by name.
        
        Args:
            name: Action name
            
        Returns:
            Action if found, None otherwise
        """
        return self._action_by_name.get(name)
    
    def list_actions(self) -> List[Action]:
        """List all registered actions.
        
        Returns:
            List of actions
        """
        return list(self._actions.values())
    
    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all registered actions.
        
        Returns:
            List of action schemas
        """
        return [action.get_schema() for action in self._actions.values()]
    
    async def execute(self, action_type: ActionType, **kwargs) -> ActionResult:
        """Execute an action by type.
        
        Args:
            action_type: Type of action to execute
            **kwargs: Action parameters
            
        Returns:
            ActionResult
        """
        action = self.get(action_type)
        if not action:
            return ActionResult(
                success=False,
                message=f"Action {action_type} not found",
                data={},
                action_type=action_type
            )
        
        return await action.execute(**kwargs)


# Concrete action implementations

class ParseCodeAction(Action):
    """Action to parse Java source code."""
    
    def __init__(self, java_parser):
        """Initialize parse action.
        
        Args:
            java_parser: Java code parser instance
        """
        super().__init__(
            name="parse_code",
            action_type=ActionType.PARSE_CODE,
            description="Parse Java source code to extract class information"
        )
        self.java_parser = java_parser
    
    async def execute(self, **kwargs) -> ActionResult:
        """Execute parse action."""
        source_code = kwargs.get("source_code", "")
        file_path = kwargs.get("file_path", "")
        
        try:
            class_info = self.java_parser.parse_class(source_code)
            return ActionResult(
                success=True,
                message=f"Successfully parsed {class_info.get('name', 'unknown')}",
                data={"class_info": class_info, "file_path": file_path},
                action_type=self.action_type
            )
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Failed to parse code: {str(e)}",
                data={"error": str(e), "file_path": file_path},
                action_type=self.action_type
            )
    
    def _get_parameters(self) -> Dict[str, Any]:
        """Get parameter schema."""
        return {
            "type": "object",
            "properties": {
                "source_code": {
                    "type": "string",
                    "description": "Java source code to parse"
                },
                "file_path": {
                    "type": "string",
                    "description": "Path to the source file"
                }
            },
            "required": ["source_code"]
        }


class GenerateTestsAction(Action):
    """Action to generate unit tests."""
    
    def __init__(self, llm_client, prompt_builder):
        """Initialize generate action.
        
        Args:
            llm_client: LLM client
            prompt_builder: Prompt builder
        """
        super().__init__(
            name="generate_tests",
            action_type=ActionType.GENERATE_TESTS,
            description="Generate unit tests for a Java class"
        )
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder
    
    async def execute(self, **kwargs) -> ActionResult:
        """Execute generate action."""
        class_info = kwargs.get("class_info", {})
        source_code = kwargs.get("source_code", "")
        existing_tests = kwargs.get("existing_tests", "")
        uncovered_info = kwargs.get("uncovered_info", {})
        
        try:
            if existing_tests and uncovered_info:
                # Generate additional tests
                prompt = self.prompt_builder.build_additional_tests_prompt(
                    class_info=class_info,
                    existing_tests=existing_tests,
                    uncovered_info=uncovered_info,
                    current_coverage=kwargs.get("current_coverage", 0.0)
                )
            else:
                # Generate initial tests
                prompt = self.prompt_builder.build_initial_test_prompt(
                    class_info=class_info,
                    source_code=source_code
                )
            
            response = await self.llm_client.generate(prompt)
            
            return ActionResult(
                success=True,
                message="Tests generated successfully",
                data={"test_code": response, "class_info": class_info},
                action_type=self.action_type
            )
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Failed to generate tests: {str(e)}",
                data={"error": str(e)},
                action_type=self.action_type
            )
    
    def _get_parameters(self) -> Dict[str, Any]:
        """Get parameter schema."""
        return {
            "type": "object",
            "properties": {
                "class_info": {
                    "type": "object",
                    "description": "Parsed class information"
                },
                "source_code": {
                    "type": "string",
                    "description": "Source code of the class"
                },
                "existing_tests": {
                    "type": "string",
                    "description": "Existing test code (for additional tests)"
                },
                "uncovered_info": {
                    "type": "object",
                    "description": "Information about uncovered code"
                }
            },
            "required": ["class_info"]
        }


class CompileAction(Action):
    """Action to compile tests."""
    
    def __init__(self, maven_runner):
        """Initialize compile action.
        
        Args:
            maven_runner: Maven runner instance
        """
        super().__init__(
            name="compile",
            action_type=ActionType.COMPILE,
            description="Compile the test code"
        )
        self.maven_runner = maven_runner
    
    async def execute(self, **kwargs) -> ActionResult:
        """Execute compile action."""
        try:
            success = self.maven_runner.compile_project()
            
            return ActionResult(
                success=success,
                message="Compilation successful" if success else "Compilation failed",
                data={},
                action_type=self.action_type
            )
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Compilation error: {str(e)}",
                data={"error": str(e)},
                action_type=self.action_type
            )
    
    def _get_parameters(self) -> Dict[str, Any]:
        """Get parameter schema."""
        return {
            "type": "object",
            "properties": {},
            "required": []
        }


class RunTestsAction(Action):
    """Action to run tests."""
    
    def __init__(self, maven_runner):
        """Initialize run tests action.
        
        Args:
            maven_runner: Maven runner instance
        """
        super().__init__(
            name="run_tests",
            action_type=ActionType.RUN_TESTS,
            description="Run the unit tests"
        )
        self.maven_runner = maven_runner
    
    async def execute(self, **kwargs) -> ActionResult:
        """Execute run tests action."""
        try:
            success = self.maven_runner.run_tests()
            
            return ActionResult(
                success=success,
                message="All tests passed" if success else "Some tests failed",
                data={},
                action_type=self.action_type
            )
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Test execution error: {str(e)}",
                data={"error": str(e)},
                action_type=self.action_type
            )
    
    def _get_parameters(self) -> Dict[str, Any]:
        """Get parameter schema."""
        return {
            "type": "object",
            "properties": {},
            "required": []
        }


class AnalyzeCoverageAction(Action):
    """Action to analyze test coverage."""
    
    def __init__(self, coverage_analyzer):
        """Initialize analyze coverage action.
        
        Args:
            coverage_analyzer: Coverage analyzer instance
        """
        super().__init__(
            name="analyze_coverage",
            action_type=ActionType.ANALYZE_COVERAGE,
            description="Analyze test coverage using JaCoCo"
        )
        self.coverage_analyzer = coverage_analyzer
    
    async def execute(self, **kwargs) -> ActionResult:
        """Execute analyze coverage action."""
        try:
            report = self.coverage_analyzer.parse_report()
            
            if report:
                return ActionResult(
                    success=True,
                    message=f"Coverage: {report.line_coverage:.1%}",
                    data={
                        "line_coverage": report.line_coverage,
                        "branch_coverage": report.branch_coverage,
                        "method_coverage": report.method_coverage,
                        "report": report
                    },
                    action_type=self.action_type
                )
            else:
                return ActionResult(
                    success=False,
                    message="Failed to parse coverage report",
                    data={},
                    action_type=self.action_type
                )
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Coverage analysis error: {str(e)}",
                data={"error": str(e)},
                action_type=self.action_type
            )
    
    def _get_parameters(self) -> Dict[str, Any]:
        """Get parameter schema."""
        return {
            "type": "object",
            "properties": {},
            "required": []
        }


class FixErrorsAction(Action):
    """Action to fix compilation or test errors."""
    
    def __init__(self, llm_client, prompt_builder):
        """Initialize fix errors action.
        
        Args:
            llm_client: LLM client
            prompt_builder: Prompt builder
        """
        super().__init__(
            name="fix_errors",
            action_type=ActionType.FIX_ERRORS,
            description="Fix compilation or test errors"
        )
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder
    
    async def execute(self, **kwargs) -> ActionResult:
        """Execute fix errors action."""
        error_type = kwargs.get("error_type", "compilation")
        test_code = kwargs.get("test_code", "")
        errors = kwargs.get("errors", [])
        class_info = kwargs.get("class_info", {})
        
        try:
            if error_type == "compilation":
                prompt = self.prompt_builder.build_fix_compilation_prompt(
                    test_code=test_code,
                    compilation_errors="\n".join(errors),
                    class_info=class_info
                )
            else:
                prompt = self.prompt_builder.build_fix_test_failure_prompt(
                    test_code=test_code,
                    failures=errors,
                    class_info=class_info
                )
            
            response = await self.llm_client.generate(prompt)
            
            return ActionResult(
                success=True,
                message=f"Fixed {error_type} errors",
                data={"fixed_code": response},
                action_type=self.action_type
            )
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Failed to fix errors: {str(e)}",
                data={"error": str(e)},
                action_type=self.action_type
            )
    
    def _get_parameters(self) -> Dict[str, Any]:
        """Get parameter schema."""
        return {
            "type": "object",
            "properties": {
                "error_type": {
                    "type": "string",
                    "enum": ["compilation", "test"],
                    "description": "Type of errors to fix"
                },
                "test_code": {
                    "type": "string",
                    "description": "Current test code"
                },
                "errors": {
                    "type": "array",
                    "description": "List of errors"
                }
            },
            "required": ["error_type", "test_code", "errors"]
        }