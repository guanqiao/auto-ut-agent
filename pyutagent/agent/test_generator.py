"""Test generator agent with pause/resume functionality and Aider integration."""

import asyncio
import logging
from typing import Optional, Callable, AsyncIterator, Dict, Any
from pathlib import Path
from enum import Enum, auto

from .conversation import ConversationManager, MessageRole
from ..memory.working_memory import WorkingMemory
from ..llm.config import LLMConfig
from ..tools.java_parser import JavaCodeParser
from ..tools.maven_tools import MavenRunner, CoverageAnalyzer, ProjectScanner
from ..tools.error_analyzer import CompilationErrorAnalyzer, ErrorAnalysis
from ..tools.failure_analyzer import TestFailureAnalyzer, FailureAnalysis
from ..tools.aider_integration import (
    AiderCodeFixer, AiderTestGenerator, 
    FixResult, apply_diff_edit
)

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status enumeration."""
    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()


class TestGeneratorAgent:
    """Agent for generating Java unit tests.
    
    Features:
    - Parse Java source files
    - Generate tests using LLM
    - Run Maven tests and analyze coverage
    - Iterative optimization
    - Pause/Resume functionality
    """
    
    def __init__(
        self,
        project_path: str,
        llm_config: LLMConfig,
        conversation: ConversationManager,
        working_memory: WorkingMemory,
    ):
        """Initialize test generator agent.
        
        Args:
            project_path: Path to Maven project
            llm_config: LLM configuration
            conversation: Conversation manager
            working_memory: Working memory for task state
        """
        self.project_path = Path(project_path)
        self.llm_config = llm_config
        self.conversation = conversation
        self.working_memory = working_memory
        
        # Tools
        self.java_parser = JavaCodeParser()
        self.maven_runner = MavenRunner(project_path)
        self.coverage_analyzer = CoverageAnalyzer(project_path)
        self.project_scanner = ProjectScanner(project_path)
        self.error_analyzer = CompilationErrorAnalyzer()
        self.failure_analyzer = TestFailureAnalyzer(project_path)
        
        # Aider integration
        self._aider_fixer: Optional[AiderCodeFixer] = None
        self._aider_generator: Optional[AiderTestGenerator] = None
        
        # State
        self.status = TaskStatus.IDLE
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Not paused by default
        self._current_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self.on_progress: Optional[Callable[[int, str], None]] = None
        self.on_log: Optional[Callable[[str], None]] = None
        
        # LLM client (lazy initialization)
        self._llm_client = None
    
    def _get_llm_client(self):
        """Get or create LLM client."""
        if self._llm_client is None:
            from ..llm.client import LLMClient
            self._llm_client = LLMClient.from_config(self.llm_config)
        return self._llm_client
    
    def _get_aider_fixer(self) -> AiderCodeFixer:
        """Get or create Aider code fixer."""
        if self._aider_fixer is None:
            self._aider_fixer = AiderCodeFixer(
                self._get_llm_client(),
                max_attempts=3
            )
        return self._aider_fixer
    
    def _get_aider_generator(self) -> AiderTestGenerator:
        """Get or create Aider test generator."""
        if self._aider_generator is None:
            self._aider_generator = AiderTestGenerator(
                self._get_llm_client()
            )
        return self._aider_generator
    
    def _log(self, message: str):
        """Log message."""
        logger.info(message)
        if self.on_log:
            self.on_log(message)
    
    def _update_progress(self, value: int, status: str):
        """Update progress."""
        if self.on_progress:
            self.on_progress(value, status)
    
    async def _check_pause(self):
        """Check if paused and wait if necessary."""
        if not self._pause_event.is_set():
            self.status = TaskStatus.PAUSED
            self._log("任务已暂停，等待恢复...")
            await self._pause_event.wait()
            self.status = TaskStatus.RUNNING
            self._log("任务已恢复")
    
    def pause(self):
        """Pause the current task."""
        if self.status == TaskStatus.RUNNING:
            self._pause_event.clear()
            self.status = TaskStatus.PAUSED
            self.working_memory.pause()
            self._log("暂停请求已发送")
    
    def resume(self):
        """Resume the paused task."""
        if self.status == TaskStatus.PAUSED:
            self._pause_event.set()
            self.status = TaskStatus.RUNNING
            self.working_memory.resume()
            self._log("恢复请求已发送")
    
    def is_paused(self) -> bool:
        """Check if task is paused."""
        return self.status == TaskStatus.PAUSED
    
    def is_running(self) -> bool:
        """Check if task is running."""
        return self.status == TaskStatus.RUNNING
    
    def get_status(self) -> TaskStatus:
        """Get current status."""
        return self.status
    
    async def generate_tests(
        self,
        target_file: str,
        target_coverage: float = 0.8,
        max_iterations: int = 10,
    ) -> Dict[str, Any]:
        """Generate tests for a Java file.
        
        Args:
            target_file: Path to Java file
            target_coverage: Target coverage ratio
            max_iterations: Maximum optimization iterations
            
        Returns:
            Result dictionary
        """
        self.status = TaskStatus.RUNNING
        self.working_memory.target_coverage = target_coverage
        self.working_memory.max_iterations = max_iterations
        self.working_memory.current_file = target_file
        
        try:
            # Step 1: Parse Java file
            await self._check_pause()
            self._update_progress(10, "解析 Java 文件...")
            self._log(f"正在解析: {target_file}")
            
            java_class = self.java_parser.parse_file(target_file)
            self._log(f"找到类: {java_class.name}, 方法数: {len(java_class.methods)}")
            
            # Step 2: Generate initial tests
            await self._check_pause()
            self._update_progress(30, "生成初始测试...")
            
            test_code = await self._generate_test_code(java_class)
            
            # Step 3: Save and run tests
            await self._check_pause()
            self._update_progress(50, "运行测试...")
            
            test_file_path = self._save_test_file(target_file, test_code)
            success = self.maven_runner.run_tests()
            
            if not success:
                self.status = TaskStatus.FAILED
                return {
                    "success": False,
                    "error": "Tests failed to run",
                    "test_file": test_file_path,
                }
            
            # Step 4: Analyze coverage
            await self._check_pause()
            self._update_progress(70, "分析覆盖率...")
            
            coverage_report = self.coverage_analyzer.parse_report()
            if coverage_report:
                self.working_memory.update_coverage(coverage_report.line_coverage)
                self._log(f"当前覆盖率: {coverage_report.line_coverage:.1%}")
            
            # Step 5: Iterative optimization
            iteration = 0
            while (
                self.working_memory.current_coverage < target_coverage
                and iteration < max_iterations
            ):
                await self._check_pause()
                
                iteration += 1
                self.working_memory.increment_iteration()
                self._update_progress(
                    70 + (20 * iteration // max_iterations),
                    f"优化迭代 {iteration}/{max_iterations}..."
                )
                
                self._log(f"迭代 {iteration}: 覆盖率 {self.working_memory.current_coverage:.1%}")
                
                # Get uncovered lines
                uncovered = self.coverage_analyzer.get_uncovered_lines(
                    Path(target_file).name
                )
                
                if not uncovered:
                    self._log("所有行已覆盖")
                    break
                
                # Generate additional tests
                additional_tests = await self._generate_additional_tests(
                    java_class, uncovered
                )
                
                # Append to test file
                self._append_test_code(test_file_path, additional_tests)
                
                # Re-run tests
                success = self.maven_runner.run_tests()
                if not success:
                    self._log("测试运行失败，停止优化")
                    break
                
                # Update coverage
                coverage_report = self.coverage_analyzer.parse_report()
                if coverage_report:
                    self.working_memory.update_coverage(coverage_report.line_coverage)
            
            # Complete
            self.status = TaskStatus.COMPLETED
            self._update_progress(100, "完成!")
            
            return {
                "success": True,
                "test_file": test_file_path,
                "iterations": iteration,
                "final_coverage": self.working_memory.current_coverage,
                "target_coverage": target_coverage,
            }
            
        except Exception as e:
            self.status = TaskStatus.FAILED
            logger.exception("Test generation failed")
            return {
                "success": False,
                "error": str(e),
            }
    
    async def _fix_compilation_errors_with_aider(
        self,
        test_code: str,
        compiler_output: str,
        test_file_path: str
    ) -> Optional[str]:
        """Fix compilation errors using Aider-style editing.
        
        Args:
            test_code: Current test code with errors
            compiler_output: Compiler error output
            test_file_path: Path to test file
            
        Returns:
            Fixed code or None if fix failed
        """
        self._log("使用 Aider 风格编辑修复编译错误...")
        
        # Analyze errors
        error_analysis = self.error_analyzer.analyze(compiler_output, test_file_path)
        
        if not error_analysis.errors:
            self._log("未检测到编译错误")
            return test_code
        
        self._log(f"检测到 {len(error_analysis.errors)} 个编译错误")
        
        # Try to fix using Aider
        try:
            fix_result = await self._get_aider_fixer().fix_compilation_errors(
                test_code=test_code,
                error_analysis=error_analysis
            )
            
            if fix_result.success:
                self._log(f"编译错误修复成功 (尝试 {fix_result.attempts} 次)")
                return fix_result.fixed_code
            else:
                self._log(f"编译错误修复失败: {fix_result.error_message}")
                return None
                
        except Exception as e:
            logger.exception("Aider fix failed")
            self._log(f"Aider 修复失败: {e}")
            return None
    
    async def _fix_test_failures_with_aider(
        self,
        test_code: str,
        test_file_path: str
    ) -> Optional[str]:
        """Fix test failures using Aider-style editing.
        
        Args:
            test_code: Current test code
            test_file_path: Path to test file
            
        Returns:
            Fixed code or None if fix failed
        """
        self._log("使用 Aider 风格编辑修复测试失败...")
        
        # Analyze failures
        failure_analysis = self.failure_analyzer.analyze()
        
        if not failure_analysis.failures:
            self._log("未检测到测试失败")
            return test_code
        
        self._log(f"检测到 {len(failure_analysis.failures)} 个测试失败")
        
        # Try to fix using Aider
        try:
            fix_result = await self._get_aider_fixer().fix_test_failures(
                test_code=test_code,
                failure_analysis=failure_analysis
            )
            
            if fix_result.success:
                self._log(f"测试失败修复成功 (尝试 {fix_result.attempts} 次)")
                return fix_result.fixed_code
            else:
                self._log(f"测试失败修复失败: {fix_result.error_message}")
                return None
                
        except Exception as e:
            logger.exception("Aider fix failed")
            self._log(f"Aider 修复失败: {e}")
            return None
    
    async def generate_tests_with_aider(
        self,
        target_file: str,
        target_coverage: float = 0.8,
        max_iterations: int = 10,
    ) -> Dict[str, Any]:
        """Generate tests using Aider-style iterative improvement.
        
        This method uses Aider's Search/Replace editing for more precise
        code modifications during the fix iteration process.
        
        Args:
            target_file: Path to Java file
            target_coverage: Target coverage ratio
            max_iterations: Maximum optimization iterations
            
        Returns:
            Result dictionary
        """
        self.status = TaskStatus.RUNNING
        self.working_memory.target_coverage = target_coverage
        self.working_memory.max_iterations = max_iterations
        self.working_memory.current_file = target_file
        
        try:
            # Step 1: Parse Java file
            await self._check_pause()
            self._update_progress(10, "解析 Java 文件...")
            self._log(f"正在解析: {target_file}")
            
            java_class = self.java_parser.parse_file(target_file)
            self._log(f"找到类: {java_class.name}, 方法数: {len(java_class.methods)}")
            
            # Step 2: Generate initial tests using Aider generator
            await self._check_pause()
            self._update_progress(30, "生成初始测试...")
            
            class_info = {
                'name': java_class.name,
                'package': java_class.package,
                'methods': [
                    {
                        'name': m.name,
                        'return_type': m.return_type,
                        'parameters': m.parameters
                    }
                    for m in java_class.methods
                ]
            }
            
            test_code = await self._get_aider_generator().generate_initial_test(class_info)
            
            # Step 3: Save and compile
            await self._check_pause()
            self._update_progress(50, "编译测试...")
            
            test_file_path = self._save_test_file(target_file, test_code)
            
            # Try to compile and fix errors
            compile_success = False
            for attempt in range(3):
                compile_output = self.maven_runner.compile_tests()
                
                if compile_output is True or (isinstance(compile_output, str) and 'BUILD SUCCESS' in compile_output):
                    compile_success = True
                    break
                
                self._log(f"编译失败，尝试使用 Aider 修复 (尝试 {attempt + 1}/3)...")
                
                fixed_code = await self._fix_compilation_errors_with_aider(
                    test_code, str(compile_output), test_file_path
                )
                
                if fixed_code:
                    test_code = fixed_code
                    self._save_test_file(target_file, test_code)
                else:
                    break
            
            if not compile_success:
                self.status = TaskStatus.FAILED
                return {
                    "success": False,
                    "error": "Failed to compile test after multiple attempts",
                    "test_file": test_file_path,
                }
            
            # Step 4: Run tests and fix failures
            await self._check_pause()
            self._update_progress(60, "运行测试...")
            
            test_success = self.maven_runner.run_tests()
            
            if not test_success:
                self._log("测试失败，尝试使用 Aider 修复...")
                
                fixed_code = await self._fix_test_failures_with_aider(
                    test_code, test_file_path
                )
                
                if fixed_code:
                    test_code = fixed_code
                    self._save_test_file(target_file, test_code)
                    
                    # Re-run tests
                    test_success = self.maven_runner.run_tests()
            
            # Step 5: Analyze coverage
            await self._check_pause()
            self._update_progress(70, "分析覆盖率...")
            
            coverage_report = self.coverage_analyzer.parse_report()
            if coverage_report:
                self.working_memory.update_coverage(coverage_report.line_coverage)
                self._log(f"当前覆盖率: {coverage_report.line_coverage:.1%}")
            
            # Step 6: Iterative optimization with Aider
            iteration = 0
            while (
                self.working_memory.current_coverage < target_coverage
                and iteration < max_iterations
            ):
                await self._check_pause()
                
                iteration += 1
                self.working_memory.increment_iteration()
                self._update_progress(
                    70 + (20 * iteration // max_iterations),
                    f"优化迭代 {iteration}/{max_iterations}..."
                )
                
                self._log(f"迭代 {iteration}: 覆盖率 {self.working_memory.current_coverage:.1%}")
                
                # Get uncovered lines
                uncovered = self.coverage_analyzer.get_uncovered_lines(
                    Path(target_file).name
                )
                
                if not uncovered:
                    self._log("所有行已覆盖")
                    break
                
                # Generate additional tests using Aider
                try:
                    fix_result = await self._get_aider_fixer().improve_coverage(
                        test_code=test_code,
                        uncovered_lines=uncovered,
                        class_info=class_info
                    )
                    
                    if fix_result.success:
                        test_code = fix_result.fixed_code
                        self._save_test_file(target_file, test_code)
                        self._log("已添加额外测试代码")
                    else:
                        self._log(f"覆盖率优化失败: {fix_result.error_message}")
                        break
                        
                except Exception as e:
                    self._log(f"覆盖率优化异常: {e}")
                    break
                
                # Re-run tests
                success = self.maven_runner.run_tests()
                if not success:
                    self._log("测试运行失败，停止优化")
                    break
                
                # Update coverage
                coverage_report = self.coverage_analyzer.parse_report()
                if coverage_report:
                    self.working_memory.update_coverage(coverage_report.line_coverage)
            
            # Complete
            self.status = TaskStatus.COMPLETED
            self._update_progress(100, "完成!")
            
            return {
                "success": True,
                "test_file": test_file_path,
                "iterations": iteration,
                "final_coverage": self.working_memory.current_coverage,
                "target_coverage": target_coverage,
                "used_aider": True,
            }
            
        except Exception as e:
            self.status = TaskStatus.FAILED
            logger.exception("Test generation with Aider failed")
            return {
                "success": False,
                "error": str(e),
            }
    
    async def _generate_test_code(self, java_class) -> str:
        """Generate test code for a Java class.
        
        Args:
            java_class: Parsed Java class
            
        Returns:
            Generated test code
        """
        # Build prompt
        prompt = self._build_test_generation_prompt(java_class)
        
        # Generate using LLM
        client = self._get_llm_client()
        
        system_prompt = """You are a Java unit test expert. Generate JUnit 5 tests following best practices:
- Use @Test annotation
- Use meaningful test method names
- Include assertions
- Mock external dependencies
- Cover edge cases

Return only the test code without explanations."""
        
        try:
            response = await client.agenerate(prompt, system_prompt)
            return response
        except Exception as e:
            self._log(f"LLM generation failed: {e}")
            # Return basic test template as fallback
            return self._generate_basic_test_template(java_class)
    
    async def _generate_additional_tests(
        self,
        java_class,
        uncovered_lines: list
    ) -> str:
        """Generate additional tests for uncovered lines.
        
        Args:
            java_class: Parsed Java class
            uncovered_lines: List of uncovered line numbers
            
        Returns:
            Additional test code
        """
        prompt = f"""Generate additional JUnit 5 tests for the following uncovered lines:
Class: {java_class.name}
Uncovered lines: {uncovered_lines}

Focus on covering these specific lines. Return only the test methods."""
        
        client = self._get_llm_client()
        
        try:
            response = await client.agenerate(prompt)
            return response
        except Exception as e:
            self._log(f"Additional test generation failed: {e}")
            return ""
    
    def _build_test_generation_prompt(self, java_class) -> str:
        """Build prompt for test generation.
        
        Args:
            java_class: Parsed Java class
            
        Returns:
            Prompt string
        """
        methods_str = "\n".join([
            f"- {m.name}({', '.join(f'{t} {n}' for t, n in m.parameters)}): {m.return_type or 'void'}"
            for m in java_class.methods
        ])
        
        return f"""Generate JUnit 5 unit tests for the following Java class:

Package: {java_class.package}
Class: {java_class.name}

Methods:
{methods_str}

Generate comprehensive tests covering:
1. Normal cases
2. Edge cases
3. Null/empty inputs
4. Exception handling

Use Mockito for mocking and AssertJ for assertions."""
    
    def _generate_basic_test_template(self, java_class) -> str:
        """Generate basic test template as fallback.
        
        Args:
            java_class: Parsed Java class
            
        Returns:
            Basic test code
        """
        package = java_class.package
        class_name = java_class.name
        test_class_name = f"{class_name}Test"
        
        return f"""package {package};

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

public class {test_class_name} {{
    
    private {class_name} target;
    
    @BeforeEach
    void setUp() {{
        target = new {class_name}();
    }}
    
    @Test
    void testBasic() {{
        // TODO: Add test implementation
        assertNotNull(target);
    }}
}}"""
    
    def _save_test_file(self, source_file: str, test_code: str) -> str:
        """Save test code to file.
        
        Args:
            source_file: Original source file path
            test_code: Generated test code
            
        Returns:
            Test file path
        """
        source_path = Path(source_file)
        
        # Determine test file path
        relative_path = source_path.relative_to(self.project_path / "src" / "main" / "java")
        test_file_name = source_path.stem + "Test.java"
        test_path = (
            self.project_path / "src" / "test" / "java" /
            relative_path.parent / test_file_name
        )
        
        # Create directory if needed
        test_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write test file
        test_path.write_text(test_code, encoding='utf-8')
        
        self._log(f"测试文件已保存: {test_path}")
        return str(test_path)
    
    def _append_test_code(self, test_file_path: str, additional_code: str):
        """Append additional test code to existing file.
        
        Args:
            test_file_path: Path to test file
            additional_code: Additional test code
        """
        test_path = Path(test_file_path)
        
        if not test_path.exists():
            return
        
        content = test_path.read_text(encoding='utf-8')
        
        # Find position to insert (before last closing brace)
        insert_pos = content.rfind('}')
        if insert_pos > 0:
            new_content = content[:insert_pos] + "\n" + additional_code + "\n" + content[insert_pos:]
            test_path.write_text(new_content, encoding='utf-8')
            self._log(f"已追加测试代码到: {test_file_path}")
    
    def stop(self):
        """Stop the current task."""
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            self.status = TaskStatus.IDLE
            self._log("任务已停止")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state for persistence.
        
        Returns:
            State dictionary
        """
        return {
            "status": self.status.name,
            "working_memory": self.working_memory.to_dict(),
            "project_path": str(self.project_path),
        }
    
    @classmethod
    def from_state(
        cls,
        state: Dict[str, Any],
        llm_config: LLMConfig,
        conversation: ConversationManager,
    ) -> "TestGeneratorAgent":
        """Restore agent from state.
        
        Args:
            state: State dictionary
            llm_config: LLM configuration
            conversation: Conversation manager
            
        Returns:
            Restored agent
        """
        working_memory = WorkingMemory.from_dict(state["working_memory"])
        
        agent = cls(
            project_path=state["project_path"],
            llm_config=llm_config,
            conversation=conversation,
            working_memory=working_memory,
        )
        
        # Restore status
        agent.status = TaskStatus[state.get("status", "IDLE")]
        
        return agent
