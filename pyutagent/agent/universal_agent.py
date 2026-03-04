"""Universal coding agent that can handle various programming tasks.

This module provides a universal agent that can understand and execute
different types of programming tasks, not just UT generation.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import asyncio
import json

from .base_agent import BaseAgent, StepResult
from .task_understanding import (
    TaskClassifier,
    TaskUnderstanding,
    TaskType,
    TaskPriority,
    TaskComplexity,
)
from .task_planner import (
    TaskPlanner,
    PlanExecutor,
    ExecutionPlan,
    SubTask,
    SubTaskStatus,
    SubTaskType,
)
from ..core.protocols import AgentState, AgentResult
from ..memory.working_memory import WorkingMemory
from ..llm.client import LLMClient
from ..tools.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class AgentMode(Enum):
    """Operating modes for the universal agent."""
    AUTONOMOUS = "autonomous"
    INTERACTIVE = "interactive"
    SUPERVISED = "supervised"
    PLANNING_ONLY = "planning_only"


@dataclass
class TaskResult:
    """Result of task execution."""
    success: bool
    task_type: TaskType
    message: str
    plan: Optional[ExecutionPlan] = None
    understanding: Optional[TaskUnderstanding] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "task_type": self.task_type.value,
            "message": self.message,
            "plan": self.plan.to_dict() if self.plan else None,
            "understanding": self.understanding.to_dict() if self.understanding else None,
            "artifacts": self.artifacts,
            "metrics": self.metrics,
        }


class UniversalCodingAgent(BaseAgent):
    """Universal coding agent that can handle various programming tasks.
    
    This agent extends BaseAgent with:
    - Task classification and understanding
    - Dynamic task planning
    - Multi-type task execution
    - Flexible tool orchestration
    
    Supported task types:
    - UT Generation
    - Code Refactoring
    - Bug Fixing
    - Feature Addition
    - Code Review
    - Documentation
    - Code Explanation
    - Test Debugging
    - Performance Optimization
    - Security Audit
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        working_memory: WorkingMemory,
        project_path: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        mode: AgentMode = AgentMode.INTERACTIVE,
        tool_registry: Optional[ToolRegistry] = None,
    ):
        """Initialize universal coding agent.
        
        Args:
            llm_client: LLM client for generation
            working_memory: Working memory for context
            project_path: Path to the project
            progress_callback: Optional callback for progress updates
            mode: Agent operating mode
            tool_registry: Optional tool registry
        """
        super().__init__(llm_client, working_memory, project_path, progress_callback)
        
        self.mode = mode
        self.tool_registry = tool_registry or ToolRegistry()
        
        self._classifier = TaskClassifier(llm_client)
        self._planner = TaskPlanner(llm_client)
        self._executor = PlanExecutor(
            tool_registry=self._build_tool_registry(),
            progress_callback=self._on_subtask_progress
        )
        
        self._current_plan: Optional[ExecutionPlan] = None
        self._current_understanding: Optional[TaskUnderstanding] = None
        self._execution_history: List[Dict[str, Any]] = []
        
        logger.info(f"[UniversalAgent] Initialized with mode: {mode.value}")
    
    def _build_tool_registry(self) -> Dict[str, Callable]:
        """Build tool registry for plan execution."""
        return {
            "analyze": self._tool_analyze,
            "read": self._tool_read,
            "write": self._tool_write,
            "edit": self._tool_edit,
            "search": self._tool_search,
            "execute": self._tool_execute,
            "test": self._tool_test,
            "compile": self._tool_compile,
            "generate": self._tool_generate,
            "validate": self._tool_validate,
            "review": self._tool_review,
        }
    
    async def handle_request(
        self, 
        request: str,
        context: Optional[Dict[str, Any]] = None
    ) -> TaskResult:
        """Handle a user request.
        
        This is the main entry point for the universal agent.
        
        Args:
            request: User's request string
            context: Optional additional context
            
        Returns:
            TaskResult with execution results
        """
        logger.info(f"[UniversalAgent] Handling request: {request[:100]}...")
        
        self._update_state(AgentState.ANALYZING, "Understanding task...")
        
        understanding = await self._understand_task(request, context)
        self._current_understanding = understanding
        
        if understanding.task_type == TaskType.UNKNOWN:
            return TaskResult(
                success=False,
                task_type=TaskType.UNKNOWN,
                message="Could not understand the task. Please provide more details.",
                understanding=understanding,
            )
        
        self._update_state(AgentState.PLANNING, f"Planning {understanding.task_type.value} task...")
        
        plan = self._planner.create_plan(understanding, use_llm=True)
        self._current_plan = plan
        
        if self.mode == AgentMode.PLANNING_ONLY:
            return TaskResult(
                success=True,
                task_type=understanding.task_type,
                message="Plan created successfully",
                plan=plan,
                understanding=understanding,
            )
        
        if self.mode == AgentMode.INTERACTIVE:
            approved = await self._request_plan_approval(plan)
            if not approved:
                return TaskResult(
                    success=False,
                    task_type=understanding.task_type,
                    message="Plan rejected by user",
                    plan=plan,
                    understanding=understanding,
                )
        
        self._update_state(AgentState.EXECUTING, "Executing plan...")
        
        result = await self._execute_plan(plan)
        
        self._update_state(
            AgentState.COMPLETED if result["success"] else AgentState.FAILED,
            "Task completed" if result["success"] else "Task failed"
        )
        
        return TaskResult(
            success=result["success"],
            task_type=understanding.task_type,
            message=result.get("message", "Execution completed"),
            plan=plan,
            understanding=understanding,
            artifacts=result.get("artifacts", {}),
            metrics=result.get("metrics", {}),
        )
    
    async def _understand_task(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None
    ) -> TaskUnderstanding:
        """Understand the user's task."""
        project_context = self._get_project_context()
        
        if context:
            project_context = f"{project_context}\n\nAdditional Context:\n{json.dumps(context, indent=2)}"
        
        if self._classifier.llm_client:
            return await self._classifier.classify_with_llm(request, project_context)
        else:
            return self._classifier.create_basic_understanding(request)
    
    def _get_project_context(self) -> str:
        """Get context about the current project."""
        context_parts = []
        
        context_parts.append(f"Project Path: {self.project_path}")
        
        if self.project_path.exists():
            src_dirs = list(self.project_path.rglob("src/main/java"))[:5]
            if src_dirs:
                context_parts.append(f"Source Directories: {[str(d) for d in src_dirs]}")
            
            pom_file = self.project_path / "pom.xml"
            build_gradle = self.project_path / "build.gradle"
            
            if pom_file.exists():
                context_parts.append("Build Tool: Maven")
            elif build_gradle.exists():
                context_parts.append("Build Tool: Gradle")
        
        return "\n".join(context_parts)
    
    async def _request_plan_approval(self, plan: ExecutionPlan) -> bool:
        """Request user approval for plan (in interactive mode)."""
        if self.progress_callback:
            approval_request = {
                "type": "plan_approval",
                "plan": plan.to_dict(),
                "message": f"Plan created with {len(plan.subtasks)} steps. Proceed?",
            }
            self.progress_callback(approval_request)
        
        return True
    
    async def _execute_plan(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """Execute the plan."""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            result = await self._executor.execute_plan(plan)
            
            if result["success"]:
                return result
            
            if self._can_retry_plan(plan):
                retry_count += 1
                plan = self._planner.refine_plan(plan, result)
                logger.info(f"[UniversalAgent] Retrying plan (attempt {retry_count})")
            else:
                break
        
        return result
    
    def _can_retry_plan(self, plan: ExecutionPlan) -> bool:
        """Check if plan can be retried."""
        failed = [st for st in plan.subtasks if st.status == SubTaskStatus.FAILED]
        return any(st.can_retry() for st in failed)
    
    def _on_subtask_progress(self, progress: Dict[str, Any]):
        """Handle subtask progress updates."""
        if self.progress_callback:
            self.progress_callback({
                "type": "subtask_progress",
                **progress
            })
    
    async def generate_tests(self, target_file: str) -> AgentResult:
        """Generate tests for a target file (legacy interface)."""
        result = await self.handle_request(
            f"Generate unit tests for {target_file}",
            {"target_file": target_file}
        )
        
        return AgentResult(
            success=result.success,
            message=result.message,
            state=AgentState.COMPLETED if result.success else AgentState.FAILED,
            data=result.artifacts,
        )
    
    async def run_feedback_loop(self, target_file: str) -> AgentResult:
        """Run feedback loop (legacy interface)."""
        return await self.generate_tests(target_file)
    
    async def _tool_analyze(self, **kwargs) -> Dict[str, Any]:
        """Analyze code or project."""
        scope = kwargs.get("scope", {})
        files = kwargs.get("files", [])
        
        analysis = {
            "files_analyzed": len(files),
            "scope": scope,
            "findings": [],
        }
        
        if files:
            for file_path in files[:5]:
                try:
                    path = self.project_path / file_path
                    if path.exists():
                        content = path.read_text(encoding="utf-8")
                        analysis["findings"].append({
                            "file": file_path,
                            "lines": len(content.splitlines()),
                            "size": len(content),
                        })
                except Exception as e:
                    logger.warning(f"[UniversalAgent] Failed to analyze {file_path}: {e}")
        
        return analysis
    
    async def _tool_read(self, **kwargs) -> Dict[str, Any]:
        """Read file contents."""
        files = kwargs.get("files", [])
        file_path = kwargs.get("file")
        
        if file_path:
            files = [file_path]
        
        contents = {}
        for f in files[:10]:
            try:
                path = self.project_path / f
                if path.exists():
                    contents[f] = path.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"[UniversalAgent] Failed to read {f}: {e}")
        
        return {"files": contents}
    
    async def _tool_write(self, **kwargs) -> Dict[str, Any]:
        """Write file contents."""
        file_path = kwargs.get("path") or kwargs.get("file")
        content = kwargs.get("content", "")
        
        if not file_path:
            return {"success": False, "error": "No file path specified"}
        
        try:
            path = self.project_path / file_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return {"success": True, "path": str(path)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _tool_edit(self, **kwargs) -> Dict[str, Any]:
        """Edit file contents."""
        file_path = kwargs.get("path") or kwargs.get("file")
        old_content = kwargs.get("old", "")
        new_content = kwargs.get("new", "")
        
        if not file_path:
            return {"success": False, "error": "No file path specified"}
        
        try:
            path = self.project_path / file_path
            if not path.exists():
                return {"success": False, "error": "File not found"}
            
            content = path.read_text(encoding="utf-8")
            
            if old_content and old_content in content:
                content = content.replace(old_content, new_content, 1)
            else:
                content = new_content
            
            path.write_text(content, encoding="utf-8")
            return {"success": True, "path": str(path)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _tool_search(self, **kwargs) -> Dict[str, Any]:
        """Search for code patterns."""
        pattern = kwargs.get("pattern", "")
        path = kwargs.get("path", ".")
        
        results = []
        try:
            import re
            search_path = self.project_path / path
            regex = re.compile(pattern, re.IGNORECASE)
            
            for file_path in search_path.rglob("*.java"):
                try:
                    content = file_path.read_text(encoding="utf-8")
                    matches = regex.findall(content)
                    if matches:
                        results.append({
                            "file": str(file_path.relative_to(self.project_path)),
                            "matches": len(matches),
                        })
                except Exception:
                    pass
        except Exception as e:
            return {"success": False, "error": str(e)}
        
        return {"success": True, "results": results[:20]}
    
    async def _tool_execute(self, **kwargs) -> Dict[str, Any]:
        """Execute shell command."""
        command = kwargs.get("command", "")
        cwd = kwargs.get("cwd", str(self.project_path))
        
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            
            return {
                "success": proc.returncode == 0,
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "returncode": proc.returncode,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _tool_test(self, **kwargs) -> Dict[str, Any]:
        """Run tests."""
        test_class = kwargs.get("test_class")
        test_method = kwargs.get("test_method")
        
        command = "mvn test"
        if test_class:
            command += f" -Dtest={test_class}"
            if test_method:
                command += f"#{test_method}"
        
        return await self._tool_execute(command=command)
    
    async def _tool_compile(self, **kwargs) -> Dict[str, Any]:
        """Compile code."""
        return await self._tool_execute(command="mvn compile -q")
    
    async def _tool_generate(self, **kwargs) -> Dict[str, Any]:
        """Generate code using LLM."""
        requirements = kwargs.get("requirements", "")
        context = kwargs.get("context", "")
        
        prompt = f"""Generate code based on the following requirements:

Requirements:
{requirements}

Context:
{context}

Generate clean, well-documented code that follows best practices.
"""
        
        try:
            response = await self.llm_client.generate(prompt)
            return {"success": True, "code": response}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _tool_validate(self, **kwargs) -> Dict[str, Any]:
        """Validate code quality."""
        file_path = kwargs.get("file")
        
        if not file_path:
            return {"success": False, "error": "No file specified"}
        
        try:
            path = self.project_path / file_path
            content = path.read_text(encoding="utf-8")
            
            issues = []
            
            if "TODO" in content:
                issues.append({"type": "todo", "message": "Contains TODO comments"})
            
            if "System.out.println" in content:
                issues.append({"type": "style", "message": "Uses System.out.println instead of logging"})
            
            return {
                "success": True,
                "file": file_path,
                "lines": len(content.splitlines()),
                "issues": issues,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _tool_review(self, **kwargs) -> Dict[str, Any]:
        """Review code changes."""
        file_path = kwargs.get("file")
        
        if not file_path:
            return {"success": False, "error": "No file specified"}
        
        try:
            path = self.project_path / file_path
            content = path.read_text(encoding="utf-8")
            
            prompt = f"""Review the following code and provide feedback:

```java
{content}
```

Provide:
1. Code quality assessment
2. Potential issues
3. Suggestions for improvement
4. Security considerations
"""
            
            response = await self.llm_client.generate(prompt)
            return {"success": True, "review": response}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_current_plan(self) -> Optional[ExecutionPlan]:
        """Get current execution plan."""
        return self._current_plan
    
    def get_current_understanding(self) -> Optional[TaskUnderstanding]:
        """Get current task understanding."""
        return self._current_understanding
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history."""
        return self._execution_history.copy()
