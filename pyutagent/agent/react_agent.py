"""ReAct Agent for UT generation with self-feedback loop."""

import re
from pathlib import Path
from typing import Dict, List, Optional, Any
import asyncio

from .base_agent import BaseAgent, AgentState, AgentResult, StepResult
from .prompts import PromptBuilder
from .actions import ActionRegistry
from ..tools.java_parser import JavaCodeParser
from ..tools.maven_tools import MavenRunner, CoverageAnalyzer, ProjectScanner
from ..memory.working_memory import WorkingMemory
from ..llm.client import LLMClient


class ReActAgent(BaseAgent):
    """ReAct agent for iterative UT generation with feedback loop."""
    
    def __init__(
        self,
        llm_client: LLMClient,
        working_memory: WorkingMemory,
        project_path: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """Initialize ReAct agent."""
        super().__init__(llm_client, working_memory, project_path, progress_callback)
        
        self.prompt_builder = PromptBuilder()
        self.action_registry = ActionRegistry()
        self.java_parser = JavaCodeParser()
        self.maven_runner = MavenRunner(project_path)
        self.coverage_analyzer = CoverageAnalyzer(project_path)
        self.project_scanner = ProjectScanner(project_path)
        
        self.current_test_file: Optional[str] = None
        self.target_class_info: Optional[Dict[str, Any]] = None
        
    async def generate_tests(self, target_file: str) -> AgentResult:
        """Generate tests for a target file with feedback loop."""
        return await self.run_feedback_loop(target_file)
    
    async def run_feedback_loop(self, target_file: str) -> AgentResult:
        """Run the complete feedback loop for UT generation.
        
        The loop follows this pattern:
        1. Parse target Java file
        2. Generate initial tests
        3. Compile tests
        4. If compilation fails -> analyze errors -> fix -> back to step 3
        5. Run tests
        6. If tests fail -> analyze failures -> fix -> back to step 3
        7. Check coverage
        8. If coverage < target -> generate additional tests -> back to step 3
        9. Complete
        """
        self._update_state(AgentState.PARSING, f"Parsing {target_file}")
        
        # Step 1: Parse target file
        parse_result = await self._parse_target_file(target_file)
        if not parse_result.success:
            return AgentResult(
                success=False,
                message=f"Failed to parse target file: {parse_result.message}",
                errors=[parse_result.message]
            )
        
        self.target_class_info = parse_result.data.get("class_info")
        self.working_memory.current_file = target_file
        
        # Step 2: Generate initial tests
        self._update_state(AgentState.GENERATING, "Generating initial tests")
        generate_result = await self._generate_initial_tests()
        if not generate_result.success:
            return AgentResult(
                success=False,
                message=f"Failed to generate tests: {generate_result.message}",
                errors=[generate_result.message]
            )
        
        self.current_test_file = generate_result.data.get("test_file")
        
        # Main feedback loop
        while self._should_continue():
            self.current_iteration += 1
            self.working_memory.increment_iteration()
            
            # Step 3: Compile tests
            self._update_state(AgentState.COMPILING, f"Iteration {self.current_iteration}: Compiling tests")
            compile_result = await self._compile_tests()
            
            if not compile_result.success:
                # Compilation failed - analyze and fix
                self._update_state(AgentState.FIXING, "Fixing compilation errors")
                fix_result = await self._fix_compilation_errors(compile_result.data.get("errors", []))
                
                if not fix_result.success:
                    return AgentResult(
                        success=False,
                        message=f"Failed to fix compilation errors: {fix_result.message}",
                        test_file=self.current_test_file,
                        iterations=self.current_iteration,
                        errors=[fix_result.message]
                    )
                continue  # Retry compilation
            
            # Step 4: Run tests
            self._update_state(AgentState.TESTING, "Running tests")
            test_result = await self._run_tests()
            
            if not test_result.success:
                # Tests failed - analyze and fix
                self._update_state(AgentState.FIXING, "Fixing test failures")
                fix_result = await self._fix_test_failures(test_result.data.get("failures", []))
                
                if not fix_result.success:
                    return AgentResult(
                        success=False,
                        message=f"Failed to fix test failures: {fix_result.message}",
                        test_file=self.current_test_file,
                        iterations=self.current_iteration,
                        errors=[fix_result.message]
                    )
                continue  # Retry tests
            
            # Step 5: Analyze coverage
            self._update_state(AgentState.ANALYZING, "Analyzing coverage")
            coverage_result = await self._analyze_coverage()
            
            if not coverage_result.success:
                return AgentResult(
                    success=False,
                    message=f"Failed to analyze coverage: {coverage_result.message}",
                    test_file=self.current_test_file,
                    iterations=self.current_iteration,
                    errors=[coverage_result.message]
                )
            
            current_coverage = coverage_result.data.get("line_coverage", 0.0)
            self.working_memory.update_coverage(current_coverage)
            
            # Check if target reached
            if current_coverage >= self.target_coverage:
                self._update_state(AgentState.COMPLETED, f"Target coverage reached: {current_coverage:.1%}")
                return AgentResult(
                    success=True,
                    message=f"Successfully generated tests with {current_coverage:.1%} coverage",
                    test_file=self.current_test_file,
                    coverage=current_coverage,
                    iterations=self.current_iteration
                )
            
            # Step 6: Generate additional tests for uncovered code
            self._update_state(AgentState.OPTIMIZING, f"Coverage {current_coverage:.1%} < target {self.target_coverage:.1%}, generating additional tests")
            additional_result = await self._generate_additional_tests(coverage_result.data)
            
            if not additional_result.success:
                return AgentResult(
                    success=False,
                    message=f"Failed to generate additional tests: {additional_result.message}",
                    test_file=self.current_test_file,
                    coverage=current_coverage,
                    iterations=self.current_iteration,
                    errors=[additional_result.message]
                )
        
        # Loop ended without reaching target
        final_coverage = self.working_memory.current_coverage
        self._update_state(AgentState.COMPLETED, f"Max iterations reached. Final coverage: {final_coverage:.1%}")
        
        return AgentResult(
            success=final_coverage > 0,
            message=f"Completed after {self.current_iteration} iterations with {final_coverage:.1%} coverage",
            test_file=self.current_test_file,
            coverage=final_coverage,
            iterations=self.current_iteration
        )
    
    async def _parse_target_file(self, target_file: str) -> StepResult:
        """Parse the target Java file."""
        try:
            file_path = self.project_path / target_file
            if not file_path.exists():
                return StepResult(
                    success=False,
                    state=AgentState.FAILED,
                    message=f"File not found: {target_file}"
                )
            
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            class_info = self.java_parser.parse_class(source_code)
            
            return StepResult(
                success=True,
                state=AgentState.PARSING,
                message=f"Successfully parsed {class_info.get('name', 'unknown')}",
                data={"class_info": class_info, "source_code": source_code}
            )
        except Exception as e:
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Error parsing file: {str(e)}"
            )
    
    async def _generate_initial_tests(self) -> StepResult:
        """Generate initial test cases."""
        try:
            prompt = self.prompt_builder.build_initial_test_prompt(
                class_info=self.target_class_info,
                source_code=self.target_class_info.get("source", "")
            )
            
            response = await self.llm_client.generate(prompt)
            test_code = self._extract_java_code(response)
            
            # Determine test file path
            class_name = self.target_class_info.get("name", "Unknown")
            test_file_name = f"{class_name}Test.java"
            
            # Find appropriate test directory
            test_dir = self.project_path / "src" / "test" / "java"
            package_path = self.target_class_info.get("package", "").replace(".", "/")
            if package_path:
                test_dir = test_dir / package_path
            
            test_dir.mkdir(parents=True, exist_ok=True)
            test_file_path = test_dir / test_file_name
            
            # Write test file
            with open(test_file_path, 'w', encoding='utf-8') as f:
                f.write(test_code)
            
            self.current_test_file = str(test_file_path.relative_to(self.project_path))
            self.working_memory.add_generated_test(
                file=self.current_test_file,
                method="initial",
                code=test_code
            )
            
            return StepResult(
                success=True,
                state=AgentState.GENERATING,
                message=f"Generated initial tests: {self.current_test_file}",
                data={"test_file": self.current_test_file, "test_code": test_code}
            )
        except Exception as e:
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Error generating tests: {str(e)}"
            )
    
    async def _compile_tests(self) -> StepResult:
        """Compile the generated tests."""
        try:
            # First compile the test file specifically
            import subprocess
            
            # Get classpath
            classpath_result = subprocess.run(
                ["mvn", "dependency:build-classpath", "-Dmdep.outputFile=cp.txt", "-q"],
                cwd=self.project_path,
                capture_output=True,
                text=True
            )
            
            classpath = ""
            cp_file = self.project_path / "cp.txt"
            if cp_file.exists():
                classpath = cp_file.read_text().strip()
            
            # Add target/classes and target/test-classes
            classpath = f"{self.project_path}/target/classes;{self.project_path}/target/test-classes;{classpath}"
            
            # Compile test file
            test_file_path = self.project_path / self.current_test_file
            compile_cmd = [
                "javac", "-cp", classpath,
                "-d", str(self.project_path / "target" / "test-classes"),
                str(test_file_path)
            ]
            
            result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return StepResult(
                    success=True,
                    state=AgentState.COMPILING,
                    message="Tests compiled successfully"
                )
            else:
                return StepResult(
                    success=False,
                    state=AgentState.FIXING,
                    message="Compilation failed",
                    data={"errors": [result.stderr], "stdout": result.stdout}
                )
        except Exception as e:
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Error compiling tests: {str(e)}"
            )
    
    async def _fix_compilation_errors(self, errors: List[str]) -> StepResult:
        """Fix compilation errors using LLM."""
        try:
            # Read current test code
            test_file_path = self.project_path / self.current_test_file
            with open(test_file_path, 'r', encoding='utf-8') as f:
                current_test_code = f.read()
            
            prompt = self.prompt_builder.build_fix_compilation_prompt(
                test_code=current_test_code,
                compilation_errors="\n".join(errors),
                class_info=self.target_class_info
            )
            
            response = await self.llm_client.generate(prompt)
            fixed_code = self._extract_java_code(response)
            
            # Write fixed code
            with open(test_file_path, 'w