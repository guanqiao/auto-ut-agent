"""Action Definitions - Centralized action type definitions for LLM recommendations.

This module provides a single source of truth for all action types that can be
recommended by LLM and executed by the ActionExecutor. It ensures consistency
between prompts and execution logic.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class ActionCategory(Enum):
    """Categories of actions based on error type."""
    COMPILATION = "compilation"
    TEST_FAILURE = "test_failure"
    DEPENDENCY = "dependency"
    GENERAL = "general"


@dataclass
class ActionParameter:
    """Definition of an action parameter."""
    name: str
    param_type: str
    required: bool = True
    description: str = ""
    default: Optional[Any] = None
    example: Optional[str] = None


@dataclass
class ActionDefinition:
    """Complete definition of an executable action.
    
    Attributes:
        action_type: Primary action type identifier (e.g., "fix_imports")
        aliases: Alternative names that map to this action
        description: Human-readable description of what the action does
        category: Category this action belongs to
        parameters: List of parameters this action accepts
        example_yaml: Example YAML format for LLM output
    """
    action_type: str
    aliases: List[str] = field(default_factory=list)
    description: str = ""
    category: ActionCategory = ActionCategory.GENERAL
    parameters: List[ActionParameter] = field(default_factory=list)
    example_yaml: str = ""
    
    def get_all_names(self) -> List[str]:
        """Get all valid names for this action (type + aliases)."""
        return [self.action_type] + self.aliases
    
    def get_required_parameters(self) -> List[ActionParameter]:
        """Get list of required parameters."""
        return [p for p in self.parameters if p.required]
    
    def to_prompt_description(self) -> str:
        """Generate description for use in LLM prompts."""
        params_str = ""
        if self.parameters:
            param_names = [f"{p.name}" + ("" if p.required else "?") for p in self.parameters]
            params_str = f" [{', '.join(param_names)}]"
        return f"**{self.action_type}**: {self.description}{params_str}"
    
    def to_prompt_example(self) -> str:
        """Generate example for use in LLM prompts."""
        if self.example_yaml:
            return self.example_yaml
        
        example = f"- action: {self.action_type}\n"
        for param in self.parameters[:3]:
            if param.example:
                example += f"  {param.name}: {param.example}\n"
            elif param.param_type == "list":
                example += f"  {param.name}: [\"value1\", \"value2\"]\n"
            elif param.param_type == "string":
                example += f"  {param.name}: \"value\"\n"
        return example


ACTION_DEFINITIONS: Dict[str, ActionDefinition] = {
    "fix_imports": ActionDefinition(
        action_type="fix_imports",
        aliases=["add_import", "add_imports"],
        description="Add missing import statements to the test file",
        category=ActionCategory.COMPILATION,
        parameters=[
            ActionParameter(
                name="imports",
                param_type="list",
                required=True,
                description="List of import statements to add",
                example="[\"import java.util.List;\", \"import org.example.MyClass;\"]"
            )
        ],
        example_yaml="""- action: fix_imports
  imports: ["import java.util.List;", "import org.example.MyClass;"]"""
    ),
    
    "add_dependency": ActionDefinition(
        action_type="add_dependency",
        aliases=["add_dependencies"],
        description="Add Maven dependency to pom.xml",
        category=ActionCategory.DEPENDENCY,
        parameters=[
            ActionParameter(
                name="group_id",
                param_type="string",
                required=True,
                description="Maven group ID",
                example="org.example"
            ),
            ActionParameter(
                name="artifact_id",
                param_type="string",
                required=True,
                description="Maven artifact ID",
                example="example-lib"
            ),
            ActionParameter(
                name="version",
                param_type="string",
                required=True,
                description="Version number",
                example="1.0.0"
            ),
            ActionParameter(
                name="scope",
                param_type="string",
                required=False,
                description="Dependency scope (compile, test, provided, etc.)",
                default="test",
                example="test"
            )
        ],
        example_yaml="""- action: add_dependency
  group_id: org.example
  artifact_id: example-lib
  version: 1.0.0
  scope: test"""
    ),
    
    "fix_syntax": ActionDefinition(
        action_type="fix_syntax",
        aliases=["fix_syntax_error"],
        description="Fix syntax errors in the code",
        category=ActionCategory.COMPILATION,
        parameters=[
            ActionParameter(
                name="fixed_code",
                param_type="string",
                required=True,
                description="Complete fixed code block",
                example="```java\\n// fixed code\\n```"
            )
        ],
        example_yaml="""- action: fix_syntax
  fixed_code: ```java
  // fixed code here
  ```"""
    ),
    
    "fix_type_error": ActionDefinition(
        action_type="fix_type_error",
        aliases=["fix_type"],
        description="Fix type mismatch errors",
        category=ActionCategory.COMPILATION,
        parameters=[
            ActionParameter(
                name="fixed_code",
                param_type="string",
                required=True,
                description="Code with corrected types",
                example="```java\\n// code with fixed types\\n```"
            )
        ],
        example_yaml="""- action: fix_type_error
  fixed_code: ```java
  // code with corrected types
  ```"""
    ),
    
    "modify_code": ActionDefinition(
        action_type="modify_code",
        aliases=["apply_fix"],
        description="Apply a complete code fix to the test file",
        category=ActionCategory.GENERAL,
        parameters=[
            ActionParameter(
                name="fixed_code",
                param_type="string",
                required=True,
                description="Complete fixed test code",
                example="```java\\n// complete fixed test class\\n```"
            )
        ],
        example_yaml="""- action: modify_code
  fixed_code: ```java
  // complete fixed test code here
  ```"""
    ),
    
    "regenerate_test": ActionDefinition(
        action_type="regenerate_test",
        aliases=["regenerate"],
        description="Request complete test regeneration (use when fixes are too complex)",
        category=ActionCategory.GENERAL,
        parameters=[],
        example_yaml="- action: regenerate_test"
    ),
    
    "fix_test_logic": ActionDefinition(
        action_type="fix_test_logic",
        aliases=["fix_logic"],
        description="Fix the test logic to match expected behavior",
        category=ActionCategory.TEST_FAILURE,
        parameters=[
            ActionParameter(
                name="test_method",
                param_type="string",
                required=False,
                description="Name of the test method to fix",
                example="testSomething"
            ),
            ActionParameter(
                name="fixed_code",
                param_type="string",
                required=True,
                description="Fixed test method code",
                example="```java\\n@Test\\nvoid testSomething() { ... }\\n```"
            )
        ],
        example_yaml="""- action: fix_test_logic
  test_method: testSomething
  fixed_code: ```java
  // fixed test method
  ```"""
    ),
    
    "fix_assertion": ActionDefinition(
        action_type="fix_assertion",
        aliases=["fix_assertions"],
        description="Fix incorrect assertions in tests",
        category=ActionCategory.TEST_FAILURE,
        parameters=[
            ActionParameter(
                name="test_method",
                param_type="string",
                required=False,
                description="Name of the test method",
                example="testSomething"
            ),
            ActionParameter(
                name="fixed_code",
                param_type="string",
                required=True,
                description="Fixed assertion code",
                example="```java\\nassertEquals(expected, actual);\\n```"
            )
        ],
        example_yaml="""- action: fix_assertion
  test_method: testSomething
  fixed_code: ```java
  // fixed assertion
  ```"""
    ),
    
    "add_mock": ActionDefinition(
        action_type="add_mock",
        aliases=["add_mocks", "fix_mock"],
        description="Add missing mock configuration",
        category=ActionCategory.TEST_FAILURE,
        parameters=[
            ActionParameter(
                name="mock_setup",
                param_type="string",
                required=True,
                description="Mock setup code",
                example="when(mockService.getData()).thenReturn(\"test\");"
            )
        ],
        example_yaml="""- action: add_mock
  mock_setup: when(mockService.getData()).thenReturn("test");"""
    ),
    
    "skip_test": ActionDefinition(
        action_type="skip_test",
        aliases=["skip_tests"],
        description="Skip a test that cannot be fixed (use sparingly, only as last resort)",
        category=ActionCategory.GENERAL,
        parameters=[
            ActionParameter(
                name="reason",
                param_type="string",
                required=False,
                description="Reason for skipping",
                example="External dependency unavailable"
            )
        ],
        example_yaml="""- action: skip_test
  reason: "External dependency unavailable for testing\""""
    ),
    
    "install_dependency": ActionDefinition(
        action_type="install_dependency",
        aliases=["install_dependencies"],
        description="Install a dependency (e.g., run mvn install)",
        category=ActionCategory.DEPENDENCY,
        parameters=[
            ActionParameter(
                name="dependency",
                param_type="string",
                required=True,
                description="Dependency to install",
                example="org.example:example-lib:1.0.0"
            )
        ],
        example_yaml="""- action: install_dependency
  dependency: "org.example:example-lib:1.0.0\""""
    ),
    
    "resolve_dependency": ActionDefinition(
        action_type="resolve_dependency",
        aliases=["resolve_dependencies"],
        description="Resolve dependency conflicts or issues",
        category=ActionCategory.DEPENDENCY,
        parameters=[
            ActionParameter(
                name="dependency",
                param_type="string",
                required=True,
                description="Dependency to resolve",
                example="org.example:example-lib"
            )
        ],
        example_yaml="""- action: resolve_dependency
  dependency: "org.example:example-lib\""""
    ),
}


def get_all_action_names() -> List[str]:
    """Get all valid action names (including aliases)."""
    names = []
    for action_def in ACTION_DEFINITIONS.values():
        names.extend(action_def.get_all_names())
    return names


def get_action_definition(action_name: str) -> Optional[ActionDefinition]:
    """Get action definition by name or alias.
    
    Args:
        action_name: Action name or alias
        
    Returns:
        ActionDefinition if found, None otherwise
    """
    action_lower = action_name.lower().strip()
    
    for action_def in ACTION_DEFINITIONS.values():
        if action_lower in [n.lower() for n in action_def.get_all_names()]:
            return action_def
    
    return None


def is_valid_action_name(action_name: str) -> bool:
    """Check if action name is valid (exists in definitions or aliases).
    
    Args:
        action_name: Action name to validate
        
    Returns:
        True if valid action name
    """
    return get_action_definition(action_name) is not None


def get_actions_by_category(category: ActionCategory) -> List[ActionDefinition]:
    """Get all actions in a specific category.
    
    Args:
        category: Category to filter by
        
    Returns:
        List of action definitions in the category
    """
    return [a for a in ACTION_DEFINITIONS.values() if a.category == category]


def generate_prompt_action_list(categories: Optional[List[ActionCategory]] = None) -> str:
    """Generate formatted action list for LLM prompts.
    
    Args:
        categories: Optional filter by categories. If None, includes all actions.
        
    Returns:
        Formatted string for use in prompts
    """
    if categories:
        actions = []
        for cat in categories:
            actions.extend(get_actions_by_category(cat))
    else:
        actions = list(ACTION_DEFINITIONS.values())
    
    lines = ["You MUST ONLY use one of the following predefined actions. Do NOT invent new action types.", ""]
    lines.append("| Action | Description | Required Parameters |")
    lines.append("|--------|-------------|---------------------|")
    
    for action in actions:
        params = action.get_required_parameters()
        params_str = ", ".join([p.name for p in params]) if params else "(none)"
        lines.append(f"| {action.action_type} | {action.description} | {params_str} |")
    
    return "\n".join(lines)


def generate_prompt_examples(categories: Optional[List[ActionCategory]] = None) -> str:
    """Generate example actions for LLM prompts.
    
    Args:
        categories: Optional filter by categories
        
    Returns:
        Formatted examples string
    """
    if categories:
        actions = []
        for cat in categories:
            actions.extend(get_actions_by_category(cat))
    else:
        actions = list(ACTION_DEFINITIONS.values())
    
    examples = ["Example actions:"]
    for action in actions[:6]:
        examples.append(action.example_yaml)
    
    return "\n".join(examples)


def build_action_type_map() -> Dict[str, str]:
    """Build mapping from all action names/aliases to canonical action type.
    
    Returns:
        Dict mapping lowercase name/alias to canonical action type
    """
    mapping = {}
    for action_def in ACTION_DEFINITIONS.values():
        for name in action_def.get_all_names():
            mapping[name.lower()] = action_def.action_type
    return mapping


ACTION_TYPE_MAP = build_action_type_map()
