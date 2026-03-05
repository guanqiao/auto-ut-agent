"""Test Generation Agent for multi-agent collaboration.

Specialized agent for generating unit tests for Java code.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
from pathlib import Path

from .specialized_agent import SpecializedAgent, AgentCapability, AgentTask
from .message_bus import MessageBus
from .shared_knowledge import SharedKnowledgeBase, ExperienceReplay

logger = logging.getLogger(__name__)


class TestGenerationAgent(SpecializedAgent):
    """Agent specialized in generating unit tests.
    
    Capabilities:
    - Generate JUnit test cases
    - Create test fixtures and setup
    - Generate mocks and stubs
    - Apply test patterns and best practices
    """
    
    def __init__(
        self,
        agent_id: str,
        message_bus: MessageBus,
        knowledge_base: SharedKnowledgeBase,
        experience_replay: Optional[ExperienceReplay] = None,
        llm_client=None,
        prompt_builder=None
    ):
        """Initialize test generation agent.
        
        Args:
            agent_id: Unique agent identifier
            message_bus: Message bus for communication
            knowledge_base: Shared knowledge base
            experience_replay: Optional experience replay buffer
            llm_client: LLM client for test generation
            prompt_builder: Prompt builder for creating prompts
        """
        super().__init__(
            agent_id=agent_id,
            capabilities={
                AgentCapability.TEST_IMPLEMENTATION,
                AgentCapability.MOCK_GENERATION,
                AgentCapability.TEST_DESIGN
            },
            message_bus=message_bus,
            knowledge_base=knowledge_base,
            experience_replay=experience_replay
        )
        
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder
        self._generation_cache: Dict[str, Any] = {}
        
        logger.info(f"[TestGenerationAgent:{agent_id}] Initialized")
    
    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute test generation task.
        
        Args:
            task: Task containing generation parameters
            
        Returns:
            Generation results
        """
        task_type = task.task_type
        payload = task.payload
        
        logger.info(f"[TestGenerationAgent:{self.agent_id}] Executing task: {task_type}")
        
        try:
            if task_type == "generate_tests":
                return await self._generate_tests(payload)
            elif task_type == "generate_test_for_method":
                return await self._generate_test_for_method(payload)
            elif task_type == "generate_mocks":
                return await self._generate_mocks(payload)
            elif task_type == "create_test_fixture":
                return await self._create_test_fixture(payload)
            else:
                return {
                    "success": False,
                    "error": f"Unknown task type: {task_type}"
                }
        except Exception as e:
            logger.exception(f"[TestGenerationAgent:{self.agent_id}] Task execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _generate_tests(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate tests for a Java class.
        
        Args:
            payload: Contains file_path, class_info, and generation options
            
        Returns:
            Generated test code
        """
        file_path = payload.get("file_path")
        class_info = payload.get("class_info", {})
        methods = payload.get("methods", [])
        options = payload.get("options", {})
        
        if not file_path:
            return {"success": False, "error": "No file_path provided"}
        
        # Check cache
        cache_key = f"{file_path}:{hash(str(methods))}"
        if cache_key in self._generation_cache:
            logger.debug(f"[TestGenerationAgent:{self.agent_id}] Using cached generation for {file_path}")
            return {
                "success": True,
                "output": self._generation_cache[cache_key]
            }
        
        try:
            # Read source code
            path = Path(file_path)
            if not path.exists():
                return {"success": False, "error": f"File not found: {file_path}"}
            
            source_code = path.read_text(encoding='utf-8')
            class_name = class_info.get("class_name") or self._extract_class_name(source_code)
            
            if not class_name:
                return {"success": False, "error": "Could not determine class name"}
            
            # Generate test class name
            test_class_name = f"{class_name}Test"
            
            # Build prompt for test generation
            prompt = self._build_test_generation_prompt(
                source_code=source_code,
                class_name=class_name,
                methods=methods,
                options=options
            )
            
            # Generate tests using LLM if available
            if self.llm_client:
                test_code = await self._generate_with_llm(prompt, options)
            else:
                # Fallback: generate basic test template
                test_code = self._generate_test_template(
                    class_name=class_name,
                    methods=methods,
                    package=class_info.get("package")
                )
            
            result = {
                "file_path": file_path,
                "class_name": class_name,
                "test_class_name": test_class_name,
                "test_code": test_code,
                "methods_covered": len(methods),
                "generation_options": options
            }
            
            # Cache result
            self._generation_cache[cache_key] = result
            
            # Share knowledge
            self.share_knowledge(
                item_type="test_generation",
                content={
                    "file_path": file_path,
                    "class_name": class_name,
                    "methods_covered": len(methods),
                    "test_class_name": test_class_name
                },
                confidence=0.85,
                tags=["test_generation", "java", "junit"]
            )
            
            return {
                "success": True,
                "output": result
            }
            
        except Exception as e:
            logger.exception(f"Test generation failed for {file_path}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _generate_test_for_method(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test for a specific method.
        
        Args:
            payload: Contains method_info and context
            
        Returns:
            Generated test method code
        """
        method_info = payload.get("method_info", {})
        class_context = payload.get("class_context", {})
        options = payload.get("options", {})
        
        method_name = method_info.get("name")
        if not method_name:
            return {"success": False, "error": "No method name provided"}
        
        try:
            # Build method-specific prompt
            prompt = self._build_method_test_prompt(
                method_info=method_info,
                class_context=class_context,
                options=options
            )
            
            # Generate test
            if self.llm_client:
                test_code = await self._generate_with_llm(prompt, options)
            else:
                test_code = self._generate_method_test_template(
                    method_info=method_info,
                    class_name=class_context.get("class_name", "TargetClass")
                )
            
            return {
                "success": True,
                "output": {
                    "method_name": method_name,
                    "test_code": test_code,
                    "test_method_name": f"test{method_name.capitalize()}"
                }
            }
            
        except Exception as e:
            logger.exception(f"Method test generation failed for {method_name}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _generate_mocks(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock objects for dependencies.
        
        Args:
            payload: Contains dependencies list
            
        Returns:
            Mock configuration code
        """
        dependencies = payload.get("dependencies", [])
        class_name = payload.get("class_name", "TargetClass")
        
        if not dependencies:
            return {
                "success": True,
                "output": {
                    "mocks": [],
                    "message": "No dependencies to mock"
                }
            }
        
        try:
            mocks = []
            for dep in dependencies:
                mock = self._generate_mock_for_dependency(dep, class_name)
                mocks.append(mock)
            
            return {
                "success": True,
                "output": {
                    "mocks": mocks,
                    "mock_count": len(mocks),
                    "framework": "mockito"
                }
            }
            
        except Exception as e:
            logger.exception(f"Mock generation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _create_test_fixture(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Create test fixture/setup code.
        
        Args:
            payload: Contains class_info and setup requirements
            
        Returns:
            Fixture code
        """
        class_info = payload.get("class_info", {})
        dependencies = payload.get("dependencies", [])
        
        class_name = class_info.get("class_name", "TargetClass")
        package = class_info.get("package")
        
        try:
            fixture = self._generate_test_fixture(class_name, package, dependencies)
            
            return {
                "success": True,
                "output": {
                    "fixture_code": fixture,
                    "class_name": class_name,
                    "setup_method": "setUp"
                }
            }
            
        except Exception as e:
            logger.exception(f"Fixture creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _build_test_generation_prompt(
        self,
        source_code: str,
        class_name: str,
        methods: List[Dict],
        options: Dict[str, Any]
    ) -> str:
        """Build prompt for test generation.
        
        Args:
            source_code: Java source code
            class_name: Class name
            methods: Methods to test
            options: Generation options
            
        Returns:
            Generation prompt
        """
        framework = options.get("framework", "JUnit5")
        mock_framework = options.get("mock_framework", "Mockito")
        include_edge_cases = options.get("include_edge_cases", True)
        
        prompt = f"""Generate comprehensive unit tests for the following Java class using {framework} and {mock_framework}.

Class Name: {class_name}

Source Code:
```java
{source_code}
```

Methods to test:
"""
        
        for method in methods:
            prompt += f"- {method.get('signature', method.get('name', 'unknown'))}\n"
        
        prompt += f"""
Requirements:
1. Generate a complete test class with proper imports
2. Include test methods for each method listed above
3. Use {framework} annotations (@Test, @BeforeEach, etc.)
4. Use {mock_framework} for mocking dependencies
5. Include setup and teardown methods if needed
"""
        
        if include_edge_cases:
            prompt += """6. Include edge case tests (null inputs, empty collections, boundary values)
7. Include error case tests (exceptions, invalid inputs)
"""
        
        return prompt
    
    def _build_method_test_prompt(
        self,
        method_info: Dict[str, Any],
        class_context: Dict[str, Any],
        options: Dict[str, Any]
    ) -> str:
        """Build prompt for method test generation.
        
        Args:
            method_info: Method information
            class_context: Class context
            options: Generation options
            
        Returns:
            Generation prompt
        """
        method_name = method_info.get("name")
        signature = method_info.get("signature", method_name)
        return_type = method_info.get("return_type", "void")
        
        return f"""Generate unit test for the following method:

Method: {signature}
Return Type: {return_type}
Class: {class_context.get('class_name', 'Unknown')}

Generate a complete test method with:
1. Descriptive test name
2. Arrange-Act-Assert structure
3. Appropriate assertions
4. Comments explaining the test
"""
    
    async def _generate_with_llm(self, prompt: str, options: Dict[str, Any]) -> str:
        """Generate code using LLM.
        
        Args:
            prompt: Generation prompt
            options: Generation options
            
        Returns:
            Generated code
        """
        if not self.llm_client:
            raise ValueError("LLM client not available")
        
        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                temperature=options.get("temperature", 0.2),
                max_tokens=options.get("max_tokens", 2000)
            )
            
            # Extract code from response
            code = response.get("content", "")
            
            # Clean up code (remove markdown code blocks if present)
            code = self._extract_code_from_markdown(code)
            
            return code
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
    
    def _extract_code_from_markdown(self, text: str) -> str:
        """Extract code from markdown code blocks.
        
        Args:
            text: Text that may contain markdown code blocks
            
        Returns:
            Extracted code
        """
        import re
        
        # Try to find code block
        code_block_pattern = r'```(?:java)?\s*\n(.*?)\n```'
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        return text.strip()
    
    def _generate_test_template(
        self,
        class_name: str,
        methods: List[Dict],
        package: Optional[str] = None
    ) -> str:
        """Generate basic test template.
        
        Args:
            class_name: Class name
            methods: Methods to test
            package: Package name
            
        Returns:
            Test template code
        """
        lines = []
        
        # Package
        if package:
            lines.append(f"package {package};")
            lines.append("")
        
        # Imports
        lines.extend([
            "import org.junit.jupiter.api.Test;",
            "import org.junit.jupiter.api.BeforeEach;",
            "import static org.junit.jupiter.api.Assertions.*;",
            "import static org.mockito.Mockito.*;",
            ""
        ])
        
        # Class declaration
        lines.append(f"public class {class_name}Test {{")
        lines.append("")
        lines.append(f"    private {class_name} target;")
        lines.append("")
        
        # Setup method
        lines.append("    @BeforeEach")
        lines.append("    void setUp() {")
        lines.append(f"        target = new {class_name}();")
        lines.append("    }")
        lines.append("")
        
        # Test methods
        for method in methods:
            method_name = method.get("name", "unknown")
            lines.append("    @Test")
            lines.append(f"    void test{method_name.capitalize()}() {{")
            lines.append("        // TODO: Implement test")
            lines.append("        // Arrange")
            lines.append("        ")
            lines.append("        // Act")
            lines.append(f"        // target.{method_name}();")
            lines.append("        ")
            lines.append("        // Assert")
            lines.append("        // assertEquals(expected, actual);")
            lines.append("    }")
            lines.append("")
        
        lines.append("}")
        
        return "\n".join(lines)
    
    def _generate_method_test_template(
        self,
        method_info: Dict[str, Any],
        class_name: str
    ) -> str:
        """Generate method test template.
        
        Args:
            method_info: Method information
            class_name: Class name
            
        Returns:
            Test method code
        """
        method_name = method_info.get("name", "unknown")
        
        lines = [
            "    @Test",
            f"    void test{method_name.capitalize()}() {{",
            "        // Arrange",
            f"        {class_name} target = new {class_name}();",
            "        ",
            "        // Act",
            f"        // var result = target.{method_name}();",
            "        ",
            "        // Assert",
            "        // assertNotNull(result);",
            "    }}"
        ]
        
        return "\n".join(lines)
    
    def _generate_mock_for_dependency(
        self,
        dependency: str,
        class_name: str
    ) -> Dict[str, Any]:
        """Generate mock for a dependency.
        
        Args:
            dependency: Dependency class name
            class_name: Target class name
            
        Returns:
            Mock configuration
        """
        # Extract simple name from full class name
        simple_name = dependency.split('.')[-1]
        var_name = simple_name[0].lower() + simple_name[1:]
        
        return {
            "dependency": dependency,
            "simple_name": simple_name,
            "variable_name": var_name,
            "mock_declaration": f"@Mock\n    private {simple_name} {var_name};",
            "import_statement": f"import {dependency};",
            "mockito_import": "import static org.mockito.Mockito.*;"
        }
    
    def _generate_test_fixture(
        self,
        class_name: str,
        package: Optional[str],
        dependencies: List[str]
    ) -> str:
        """Generate test fixture code.
        
        Args:
            class_name: Class name
            package: Package name
            dependencies: Dependencies
            
        Returns:
            Fixture code
        """
        lines = []
        
        if package:
            lines.append(f"package {package};")
            lines.append("")
        
        lines.extend([
            "import org.junit.jupiter.api.BeforeEach;",
            "import org.junit.jupiter.api.AfterEach;",
            "import org.mockito.Mock;",
            "import org.mockito.MockitoAnnotations;",
            ""
        ])
        
        lines.append(f"public class {class_name}Test {{")
        lines.append("")
        
        # Mock declarations
        for dep in dependencies:
            simple_name = dep.split('.')[-1]
            var_name = simple_name[0].lower() + simple_name[1:]
            lines.append(f"    @Mock")
            lines.append(f"    private {simple_name} {var_name};")
        
        lines.append("")
        lines.append(f"    private {class_name} target;")
        lines.append("")
        
        # Setup
        lines.append("    @BeforeEach")
        lines.append("    void setUp() {")
        lines.append("        MockitoAnnotations.openMocks(this);")
        
        if dependencies:
            lines.append(f"        target = new {class_name}(")
            for i, dep in enumerate(dependencies):
                simple_name = dep.split('.')[-1]
                var_name = simple_name[0].lower() + simple_name[1:]
                suffix = "," if i < len(dependencies) - 1 else ""
                lines.append(f"            {var_name}{suffix}")
            lines.append("        );")
        else:
            lines.append(f"        target = new {class_name}();")
        
        lines.append("    }")
        lines.append("")
        
        # Teardown
        lines.append("    @AfterEach")
        lines.append("    void tearDown() {")
        lines.append("        // Cleanup if needed")
        lines.append("    }")
        lines.append("")
        lines.append("}")
        
        return "\n".join(lines)
    
    def _extract_class_name(self, source_code: str) -> Optional[str]:
        """Extract class name from source code.
        
        Args:
            source_code: Java source code
            
        Returns:
            Class name or None
        """
        import re
        
        match = re.search(r'(?:public\s+)?(?:class|interface|enum)\s+(\w+)', source_code)
        if match:
            return match.group(1)
        return None
