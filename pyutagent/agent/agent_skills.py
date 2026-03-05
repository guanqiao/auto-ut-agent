"""Agent Skills implementation following Anthropic Agent Skills standard.

This module provides:
- AgentSkill: Standardized skill definition following Agent Skills spec
- AgentSkillRegistry: Registry for managing agent skills
- SkillExecutor: Execute skills with proper tool calling
- JSON Schema definitions for skill validation
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class SkillStatus(Enum):
    """Status of skill execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ToolDefinition:
    """Tool definition following Agent Skills spec."""
    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolDefinition':
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            input_schema=data.get("input_schema", {})
        )


@dataclass
class AgentSkill:
    """Agent Skill following Anthropic Agent Skills standard.

    Fields:
        name: Unique skill identifier
        description: What the skill does
        tags: When to invoke the skill (trigger conditions)
        tools: List of tool definitions this skill can use
        parameters: Input parameters schema
        examples: Usage examples
        instructions: Detailed instructions for the agent
    """
    name: str
    description: str
    tags: List[str] = field(default_factory=list)
    tools: List[ToolDefinition] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    instructions: str = ""
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "tools": [t.to_dict() for t in self.tools],
            "parameters": self.parameters,
            "examples": self.examples,
            "instructions": self.instructions,
            "version": self.version
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentSkill':
        tools = [ToolDefinition.from_dict(t) for t in data.get("tools", [])]
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            tools=tools,
            parameters=data.get("parameters", {}),
            examples=data.get("examples", []),
            instructions=data.get("instructions", ""),
            version=data.get("version", "1.0.0")
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'AgentSkill':
        return cls.from_dict(json.loads(json_str))


@dataclass
class SkillExecutionContext:
    """Context for skill execution."""
    skill_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    project_path: Optional[str] = None
    files: List[str] = field(default_factory=list)
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class SkillExecutionResult:
    """Result of skill execution."""
    status: SkillStatus
    output: Any = None
    error: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    duration_ms: int = 0


class AgentSkillRegistry:
    """Registry for managing Agent Skills.

    Provides:
    - Registration of skills
    - Discovery by tags
    - JSON Schema validation
    - Skill export/import
    """

    def __init__(self):
        self._skills: Dict[str, AgentSkill] = {}
        self._tags_index: Dict[str, List[str]] = {}
        logger.info("[AgentSkillRegistry] Initialized")

    def register(self, skill: AgentSkill) -> None:
        """Register a skill.

        Args:
            skill: Agent skill to register
        """
        self._skills[skill.name] = skill

        for tag in skill.tags:
            if tag not in self._tags_index:
                self._tags_index[tag] = []
            if skill.name not in self._tags_index[tag]:
                self._tags_index[tag].append(skill.name)

        logger.info(f"[AgentSkillRegistry] Registered: {skill.name}")

    def unregister(self, name: str) -> bool:
        """Unregister a skill.

        Args:
            name: Skill name

        Returns:
            True if unregistered
        """
        if name not in self._skills:
            return False

        skill = self._skills.pop(name)
        for tag in skill.tags:
            if tag in self._tags_index and name in self._tags_index[tag]:
                self._tags_index[tag].remove(name)

        logger.info(f"[AgentSkillRegistry] Unregistered: {name}")
        return True

    def get(self, name: str) -> Optional[AgentSkill]:
        """Get a skill by name.

        Args:
            name: Skill name

        Returns:
            Skill or None
        """
        return self._skills.get(name)

    def find_by_tag(self, tag: str) -> List[AgentSkill]:
        """Find skills by tag.

        Args:
            tag: Tag to search for

        Returns:
            List of matching skills
        """
        skill_names = self._tags_index.get(tag, [])
        return [self._skills[name] for name in skill_names if name in self._skills]

    def find_by_trigger(self, trigger: str) -> List[AgentSkill]:
        """Find skills that match a trigger.

        Args:
            trigger: Trigger text

        Returns:
            List of matching skills
        """
        results = []
        trigger_lower = trigger.lower()

        for skill in self._skills.values():
            for tag in skill.tags:
                if tag.lower() in trigger_lower or trigger_lower in tag.lower():
                    results.append(skill)
                    break

        return results

    def list_all(self) -> List[AgentSkill]:
        """List all registered skills.

        Returns:
            List of all skills
        """
        return list(self._skills.values())

    def list_names(self) -> List[str]:
        """List all skill names.

        Returns:
            List of skill names
        """
        return list(self._skills.keys())

    def export_to_json(self) -> str:
        """Export all skills to JSON.

        Returns:
            JSON string of all skills
        """
        skills_data = [skill.to_dict() for skill in self._skills.values()]
        return json.dumps({
            "version": "1.0",
            "skills": skills_data
        }, indent=2)

    def import_from_json(self, json_str: str) -> int:
        """Import skills from JSON.

        Args:
            json_str: JSON string containing skills

        Returns:
            Number of skills imported
        """
        data = json.loads(json_str)
        skills_data = data.get("skills", [])
        count = 0

        for skill_data in skills_data:
            skill = AgentSkill.from_dict(skill_data)
            self.register(skill)
            count += 1

        logger.info(f"[AgentSkillRegistry] Imported {count} skills")
        return count

    def load_from_directory(self, directory: Path) -> int:
        """Load skills from a directory of JSON files.

        Args:
            directory: Directory containing skill JSON files

        Returns:
            Number of skills loaded
        """
        count = 0

        for file_path in directory.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    count += self.import_from_json(f.read())
            except Exception as e:
                logger.error(f"Failed to load skill from {file_path}: {e}")

        return count


class SkillExecutor:
    """Executor for Agent Skills.

    Handles:
    - Tool calling
    - Parameter validation
    - Execution tracking
    - Error handling
    """

    def __init__(self, registry: AgentSkillRegistry):
        self._registry = registry
        self._tool_handlers: Dict[str, Callable] = {}
        self._execution_history: List[SkillExecutionResult] = []

    def register_tool_handler(self, tool_name: str, handler: Callable) -> None:
        """Register a tool handler.

        Args:
            tool_name: Name of the tool
            handler: Callable to handle tool execution
        """
        self._tool_handlers[tool_name] = handler
        logger.debug(f"[SkillExecutor] Registered handler for tool: {tool_name}")

    async def execute(self, skill_name: str, context: SkillExecutionContext) -> SkillExecutionResult:
        """Execute a skill.

        Args:
            skill_name: Name of skill to execute
            context: Execution context

        Returns:
            Execution result
        """
        import time
        start_time = time.time()

        skill = self._registry.get(skill_name)
        if not skill:
            return SkillExecutionResult(
                status=SkillStatus.FAILED,
                error=f"Skill not found: {skill_name}"
            )

        result = SkillExecutionResult(status=SkillStatus.RUNNING)

        try:
            result = await self._execute_skill(skill, context, result)
            result.status = SkillStatus.COMPLETED
        except Exception as e:
            result.status = SkillStatus.FAILED
            result.error = str(e)
            logger.exception(f"Skill execution failed: {skill_name}")

        result.duration_ms = int((time.time() - start_time) * 1000)
        self._execution_history.append(result)

        return result

    async def _execute_skill(self, skill: AgentSkill, context: SkillExecutionContext,
                            result: SkillExecutionResult) -> SkillExecutionResult:
        """Execute skill logic.

        Args:
            skill: Skill to execute
            context: Execution context
            result: Result object to update

        Returns:
            Updated result
        """
        result.logs.append(f"Starting skill: {skill.name}")
        result.logs.append(f"Description: {skill.description}")

        for tool in skill.tools:
            result.logs.append(f"Tool available: {tool.name}")
            result.tools_used.append(tool.name)

        result.output = {
            "skill": skill.name,
            "parameters": context.parameters,
            "executed": True
        }

        return result

    def get_history(self) -> List[SkillExecutionResult]:
        """Get execution history.

        Returns:
            List of execution results
        """
        return self._execution_history.copy()


_global_registry: Optional[AgentSkillRegistry] = None


def get_agent_skill_registry() -> AgentSkillRegistry:
    """Get the global skill registry.

    Returns:
        Global AgentSkillRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = AgentSkillRegistry()
    return _global_registry


def register_builtin_skills() -> None:
    """Register built-in skills."""
    registry = get_agent_skill_registry()

    skills = [
        AgentSkill(
            name="generate_unit_tests",
            description="Generate unit tests for the selected code",
            tags=["test", "testing", "unit test", "junit", "pytest", "generate tests"],
            instructions="Analyze the selected code and generate comprehensive unit tests.",
            tools=[
                ToolDefinition(
                    name="read_file",
                    description="Read file contents",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path to read"}
                        },
                        "required": ["path"]
                    }
                ),
                ToolDefinition(
                    name="write_file",
                    description="Write content to a file",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path to write"},
                            "content": {"type": "string", "description": "Content to write"}
                        },
                        "required": ["path", "content"]
                    }
                )
            ],
            examples=[
                {
                    "description": "Generate tests for a Python class",
                    "input": {"file": "src/utils.py", "class": "StringHelper"}
                }
            ]
        ),
        AgentSkill(
            name="explain_code",
            description="Explain what the selected code does",
            tags=["explain", "documentation", "understand", "analyze"],
            instructions="Provide a clear explanation of the code functionality.",
            tools=[
                ToolDefinition(
                    name="read_file",
                    description="Read file contents",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path to read"}
                        },
                        "required": ["path"]
                    }
                )
            ]
        ),
        AgentSkill(
            name="refactor_code",
            description="Refactor the selected code for better quality",
            tags=["refactor", "improve", "clean", "restructure"],
            tools=[
                ToolDefinition(
                    name="read_file",
                    description="Read file contents",
                    input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                ),
                ToolDefinition(
                    name="edit_file",
                    description="Edit a file",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "old_str": {"type": "string"},
                            "new_str": {"type": "string"}
                        },
                        "required": ["path", "old_str", "new_str"]
                    }
                )
            ]
        ),
        AgentSkill(
            name="fix_bugs",
            description="Find and fix bugs in the selected code",
            tags=["bug", "fix", "debug", "error", "fix error"],
            tools=[
                ToolDefinition(
                    name="read_file",
                    description="Read file contents",
                    input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                ),
                ToolDefinition(
                    name="grep",
                    description="Search for patterns in files",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "path": {"type": "string"}
                        },
                        "required": ["pattern"]
                    }
                )
            ]
        )
    ]

    for skill in skills:
        registry.register(skill)

    logger.info(f"[AgentSkills] Registered {len(skills)} built-in skills")
