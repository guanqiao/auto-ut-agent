"""TestNG test generator."""

import logging
from typing import Dict, Any, Optional, List

from .base_generator import BaseTestGenerator
from ...core.project_config import TestFramework, MockFramework

logger = logging.getLogger(__name__)


class TestNGTestGenerator(BaseTestGenerator):
    """Generates tests using TestNG framework."""
    
    def __init__(
        self,
        project_path: str,
        mock_framework: MockFramework = MockFramework.MOCKITO
    ):
        """Initialize TestNG test generator.
        
        Args:
            project_path: Path to the project
            mock_framework: Mock framework to use (default: Mockito)
        """
        super().__init__(project_path)
        self.mock_framework = mock_framework
        self.annotations = self._load_testng_annotations()
    
    def _load_testng_annotations(self) -> Dict[str, str]:
        """Load TestNG annotation templates.
        
        Returns:
            Dictionary of annotation templates
        """
        return {
            'test': '@Test',
            'before_method': '@BeforeMethod',
            'after_method': '@AfterMethod',
            'before_class': '@BeforeClass',
            'after_class': '@AfterClass',
            'before_test': '@BeforeTest',
            'after_test': '@AfterTest',
            'before_suite': '@BeforeSuite',
            'after_suite': '@AfterSuite',
            'data_provider': '@DataProvider(name = "{name}")',
            'parameters': '@Parameters({params})',
        }
    
    async def generate_initial_test(self, class_info: Dict[str, Any]) -> str:
        """Generate initial TestNG test code for a class.
        
        Args:
            class_info: Class information dictionary
            
        Returns:
            Generated TestNG test code
        """
        try:
            test_code = self._generate_testng_template(class_info)
            return test_code
        except Exception as e:
            logger.exception(f"TestNG generation failed: {e}")
            return self._generate_basic_testng_template(class_info)
    
    async def generate_additional_tests(
        self,
        class_info: Dict[str, Any],
        uncovered_lines: list
    ) -> str:
        """Generate additional TestNG tests for uncovered lines.
        
        Args:
            class_info: Class information dictionary
            uncovered_lines: List of uncovered line numbers
            
        Returns:
            Additional TestNG test code
        """
        try:
            test_code = self._generate_coverage_tests(class_info, uncovered_lines)
            return test_code
        except Exception as e:
            logger.exception(f"Additional TestNG tests generation failed: {e}")
            return ""
    
    def _generate_testng_template(self, class_info: Dict[str, Any]) -> str:
        """Generate TestNG test template.
        
        Args:
            class_info: Class information dictionary
            
        Returns:
            Complete TestNG test class
        """
        class_name = class_info.get('name', 'Unknown')
        package = class_info.get('package', '')
        methods = class_info.get('methods', [])
        dependencies = class_info.get('dependencies', [])
        fields = class_info.get('fields', [])
        
        # Build imports
        imports = self._build_imports()
        
        # Build package declaration
        package_line = f"package {package};\n\n" if package else ""
        
        # Build class header
        test_class_name = f"{class_name}Test"
        class_header = f"{package_line}{imports}public class {test_class_name} {{\n"
        
        # Build fields
        fields_section = self._build_fields(class_name, dependencies, fields)
        
        # Build setup method
        setup_method = self._build_setup_method(class_name, dependencies)
        
        # Build test methods (use default test if no methods)
        if methods:
            test_methods = self._build_test_methods(methods, class_info)
        else:
            test_methods = self._generate_default_test()
        
        # Build teardown method
        teardown_method = self._build_teardown_method()
        
        # Build class footer
        class_footer = "}\n"
        
        return (
            class_header +
            fields_section +
            setup_method +
            test_methods +
            teardown_method +
            class_footer
        )
    
    def _build_imports(self) -> str:
        """Build TestNG import statements.
        
        Returns:
            Import statements string
        """
        imports = [
            "import org.testng.annotations.*;",
            "import static org.testng.Assert.*;",
            "",
        ]
        
        # Add Mockito imports if using Mockito
        if self.mock_framework == MockFramework.MOCKITO:
            imports.extend([
                "import org.mockito.Mockito;",
                "import org.mockito.Mock;",
                "import org.mockito.testng.MockitoTestNGListener;",
                "import static org.mockito.Mockito.*;",
                "",
            ])
        
        # Add common utility imports
        imports.extend([
            "import java.util.*;",
            "import java.io.*;",
            "",
        ])
        
        return "\n".join(imports)
    
    def _build_fields(
        self,
        class_name: str,
        dependencies: List[Dict[str, Any]],
        fields: List[Dict[str, Any]]
    ) -> str:
        """Build test class fields.
        
        Args:
            class_name: Target class name
            dependencies: List of dependency information
            fields: List of field information
            
        Returns:
            Fields declaration string
        """
        field_lines = []
        
        # Add target instance
        field_lines.append(f"    private {class_name} target;")
        field_lines.append("")
        
        # Add mock dependencies
        for dep in dependencies:
            dep_name = dep.get('name', 'mock')
            dep_type = dep.get('type', 'Object')
            field_lines.append(f"    @Mock")
            field_lines.append(f"    private {dep_type} {dep_name};")
        
        if dependencies:
            field_lines.append("")
        
        return "\n".join(field_lines)
    
    def _build_setup_method(
        self,
        class_name: str,
        dependencies: List[Dict[str, Any]]
    ) -> str:
        """Build @BeforeMethod setup method.
        
        Args:
            class_name: Target class name
            dependencies: List of dependency information
            
        Returns:
            Setup method string
        """
        setup = "    @BeforeMethod\n"
        setup += "    public void setUp() {\n"
        
        # Initialize mocks
        if dependencies and self.mock_framework == MockFramework.MOCKITO:
            setup += "        // Initialize mocks\n"
            for dep in dependencies:
                dep_name = dep.get('name', 'mock')
                dep_type = dep.get('type', 'Object')
                setup += f"        {dep_name} = mock({dep_type}.class);\n"
            setup += "\n"
        
        # Initialize target
        setup += f"        // Initialize target instance\n"
        setup += f"        target = new {class_name}();\n"
        setup += "    }\n\n"
        
        return setup
    
    def _build_test_methods(
        self,
        methods: List[Dict[str, Any]],
        class_info: Dict[str, Any]
    ) -> str:
        """Build test methods for class methods.
        
        Args:
            methods: List of method information
            class_info: Class information dictionary
            
        Returns:
            Test methods string
        """
        test_methods = []
        
        if not methods:
            test_methods.append(self._generate_default_test())
        else:
            for method in methods:
                basic_test = self._generate_basic_test(method, class_info)
                test_methods.append(basic_test)
                
                if method.get('parameters') and len(method['parameters']) > 0:
                    param_test = self._generate_parameterized_test(method, class_info)
                    if param_test:
                        test_methods.append(param_test)
                
                if method.get('throws_exceptions'):
                    exception_tests = self._generate_exception_tests(method, class_info)
                    test_methods.extend(exception_tests)
        
        return "\n".join(test_methods)
    
    def _generate_default_test(self) -> str:
        """Generate default test method when no methods exist.
        
        Returns:
            Default test method string
        """
        return """    @Test
    public void testInstance() {
        // Basic instance test
        assertNotNull(target);
    }
"""
    
    def _generate_basic_test(
        self,
        method: Dict[str, Any],
        class_info: Dict[str, Any]
    ) -> str:
        """Generate basic test method.
        
        Args:
            method: Method information
            class_info: Class information dictionary
            
        Returns:
            Basic test method string
        """
        method_name = method.get('name', 'unknown')
        return_type = method.get('return_type', 'void')
        parameters = method.get('parameters', [])
        
        # Generate test method name
        test_name = self._generate_test_method_name(method_name)
        
        # Build test method
        test = f"    @Test\n"
        test += f"    public void {test_name}() {{\n"
        test += f"        // Given\n"
        
        # Setup parameters
        if parameters:
            for param in parameters:
                param_type = param.get('type', 'Object')
                param_name = param.get('name', 'param')
                default_value = self._get_default_value(param_type)
                test += f"        {param_type} {param_name} = {default_value};\n"
        
        test += f"\n"
        test += f"        // When\n"
        
        # Build method call
        if return_type and return_type != 'void':
            test += f"        {return_type} result = "
        else:
            test += f"        "
        
        test += f"target.{method_name}("
        if parameters:
            test += ", ".join([p.get('name', 'param') for p in parameters])
        test += ");\n"
        
        test += f"\n"
        test += f"        // Then\n"
        
        # Add assertions
        if return_type and return_type != 'void':
            if return_type in ['boolean', 'Boolean']:
                test += f"        assertTrue(result);\n"
            elif return_type in ['int', 'Integer', 'long', 'Long', 'double', 'Double', 'float', 'Float']:
                test += f"        assertNotNull(result);\n"
            else:
                test += f"        assertNotNull(result);\n"
        else:
            test += f"        // Verify method execution\n"
        
        test += f"    }}\n\n"
        
        return test
    
    def _generate_parameterized_test(
        self,
        method: Dict[str, Any],
        class_info: Dict[str, Any]
    ) -> Optional[str]:
        """Generate parameterized test using DataProvider.
        
        Args:
            method: Method information
            class_info: Class information dictionary
            
        Returns:
            Parameterized test method string or None
        """
        method_name = method.get('name', 'unknown')
        parameters = method.get('parameters', [])
        
        if not parameters:
            return None
        
        # Generate data provider name
        data_provider_name = f"provide{method_name[0].upper()}{method_name[1:]}Data"
        
        # Build data provider
        test = f"    @DataProvider(name = \"{data_provider_name}\")\n"
        test += f"    public Object[][] {data_provider_name}() {{\n"
        test += f"        return new Object[][] {{\n"
        
        # Generate sample test data
        test_data = self._generate_sample_test_data(parameters)
        for i, data_row in enumerate(test_data):
            test += f"            {{ {', '.join(data_row)} }}{',' if i < len(test_data) - 1 else ''}\n"
        
        test += f"        }};\n"
        test += f"    }}\n\n"
        
        # Build parameterized test method
        test_name = f"test{method_name[0].upper()}{method_name[1:]}WithParams"
        
        # Build parameter types and names
        param_types = ", ".join([f"Object {p.get('name', 'param')}" for p in parameters])
        
        test += f"    @Test(dataProvider = \"{data_provider_name}\")\n"
        test += f"    public void {test_name}({param_types}) {{\n"
        test += f"        // Given\n"
        
        # Cast parameters to appropriate types
        for param in parameters:
            param_type = param.get('type', 'Object')
            param_name = param.get('name', 'param')
            test += f"        {param_type} {param_name}Cast = ({param_type}) {param_name};\n"
        
        test += f"\n"
        test += f"        // When\n"
        test += f"        // TODO: Add method call and assertions\n"
        test += f"\n"
        test += f"        // Then\n"
        test += f"        // TODO: Add assertions\n"
        test += f"    }}\n\n"
        
        return test
    
    def _generate_exception_tests(
        self,
        method: Dict[str, Any],
        class_info: Dict[str, Any]
    ) -> List[str]:
        """Generate exception test methods.
        
        Args:
            method: Method information
            class_info: Class information dictionary
            
        Returns:
            List of exception test method strings
        """
        exception_tests = []
        exceptions = method.get('throws_exceptions', [])
        method_name = method.get('name', 'unknown')
        parameters = method.get('parameters', [])
        
        for exception in exceptions:
            exception_name = exception.get('name', 'Exception')
            exception_full = exception.get('full_name', exception_name)
            
            test_name = f"test{method_name[0].upper()}{method_name[1:]}Throws{exception_name}"
            
            test = f"    @Test(expectedExceptions = {{{exception_full}.class}})\n"
            test += f"    public void {test_name}() {{\n"
            test += f"        // Given\n"
            
            # Setup parameters that might cause exception
            if parameters:
                for param in parameters:
                    param_type = param.get('type', 'Object')
                    param_name = param.get('name', 'param')
                    # Provide values that might trigger exception
                    if param_type in ['int', 'Integer']:
                        test += f"        {param_type} {param_name} = -1; // Invalid value\n"
                    elif param_type in ['String', 'string']:
                        test += f"        String {param_name} = null; // Null value\n"
                    else:
                        test += f"        {param_type} {param_name} = null;\n"
            
            test += f"\n"
            test += f"        // When\n"
            test += f"        target.{method_name}("
            if parameters:
                test += ", ".join([p.get('name', 'param') for p in parameters])
            test += ");\n"
            test += f"\n"
            test += f"        // Then\n"
            test += f"        // Exception expected - test will fail if not thrown\n"
            test += f"    }}\n\n"
            
            exception_tests.append(test)
        
        return exception_tests
    
    def _generate_coverage_tests(
        self,
        class_info: Dict[str, Any],
        uncovered_lines: List[int]
    ) -> str:
        """Generate tests to improve coverage.
        
        Args:
            class_info: Class information dictionary
            uncovered_lines: List of uncovered line numbers
            
        Returns:
            Test methods to improve coverage
        """
        if not uncovered_lines:
            return ""
        
        methods = class_info.get('methods', [])
        
        # Try to map uncovered lines to methods
        tests = []
        for method in methods:
            method_line = method.get('line_number', 0)
            method_end = method.get('end_line', method_line + 10)
            
            # Check if any uncovered line falls within this method
            method_uncovered = [
                line for line in uncovered_lines
                if method_line <= line <= method_end
            ]
            
            if method_uncovered:
                test = self._generate_targeted_test(method, method_uncovered, class_info)
                tests.append(test)
        
        return "\n".join(tests)
    
    def _generate_targeted_test(
        self,
        method: Dict[str, Any],
        uncovered_lines: List[int],
        class_info: Dict[str, Any]
    ) -> str:
        """Generate targeted test for specific uncovered lines.
        
        Args:
            method: Method information
            uncovered_lines: List of uncovered lines in this method
            class_info: Class information dictionary
            
        Returns:
            Targeted test method string
        """
        method_name = method.get('name', 'unknown')
        test_name = f"test{method_name[0].upper()}{method_name[1:]}Coverage"
        
        test = f"    @Test\n"
        test += f"    public void {test_name}() {{\n"
        test += f"        // Test to improve coverage for lines: {uncovered_lines}\n"
        test += f"        // TODO: Add test logic to cover specific code paths\n"
        test += f"\n"
        test += f"        // Given\n"
        test += f"        // Setup test data\n"
        test += f"\n"
        test += f"        // When\n"
        test += f"        // Execute method\n"
        test += f"\n"
        test += f"        // Then\n"
        test += f"        // Verify coverage\n"
        test += f"    }}\n\n"
        
        return test
    
    def _build_teardown_method(self) -> str:
        """Build @AfterMethod teardown method.
        
        Returns:
            Teardown method string
        """
        teardown = "    @AfterMethod\n"
        teardown += "    public void tearDown() {\n"
        teardown += "        // Clean up resources\n"
        teardown += "        target = null;\n"
        teardown += "    }\n\n"
        
        return teardown
    
    def _generate_test_method_name(self, method_name: str) -> str:
        """Generate test method name from source method name.
        
        Args:
            method_name: Source method name
            
        Returns:
            Test method name
        """
        # Convert to test naming convention
        # e.g., "calculateTotal" -> "testCalculateTotal"
        if not method_name:
            return "testBasic"
        
        return f"test{method_name[0].upper()}{method_name[1:]}"
    
    def _get_default_value(self, param_type: str) -> str:
        """Get default value for a parameter type.
        
        Args:
            param_type: Parameter type
            
        Returns:
            Default value string
        """
        type_defaults = {
            'int': '0',
            'Integer': '0',
            'long': '0L',
            'Long': '0L',
            'double': '0.0',
            'Double': '0.0',
            'float': '0.0f',
            'Float': '0.0f',
            'boolean': 'false',
            'Boolean': 'Boolean.FALSE',
            'String': '""',
            'char': "'\\0'",
            'Character': "'\\0'",
            'byte': '(byte) 0',
            'Byte': '(byte) 0',
            'short': '(short) 0',
            'Short': '(short) 0',
        }
        
        return type_defaults.get(param_type, 'null')
    
    def _generate_sample_test_data(
        self,
        parameters: List[Dict[str, Any]]
    ) -> List[List[str]]:
        """Generate sample test data for parameterized tests.
        
        Args:
            parameters: List of parameter information
            
        Returns:
            List of test data rows
        """
        # Generate 3 sample data rows
        test_data = []
        
        for i in range(3):
            row = []
            for param in parameters:
                param_type = param.get('type', 'Object')
                default = self._get_default_value(param_type)
                
                # Vary the values for different test cases
                if param_type in ['int', 'Integer']:
                    row.append(str(i + 1))
                elif param_type in ['String', 'string']:
                    row.append(f'"test{i+1}"')
                else:
                    row.append(default)
            
            test_data.append(row)
        
        return test_data
    
    def _generate_basic_testng_template(self, class_info: Dict[str, Any]) -> str:
        """Generate basic TestNG template as fallback.
        
        Args:
            class_info: Class information dictionary
            
        Returns:
            Basic TestNG test code
        """
        package = class_info.get('package', '')
        class_name = class_info.get('name', 'Unknown')
        test_class_name = f"{class_name}Test"
        
        package_line = f"package {package};\n\n" if package else ""
        
        return f"""{package_line}import org.testng.annotations.*;
import static org.testng.Assert.*;
import org.mockito.Mockito;
import static org.mockito.Mockito.*;

public class {test_class_name} {{
    
    private {class_name} target;
    
    @BeforeMethod
    public void setUp() {{
        target = new {class_name}();
    }}
    
    @Test
    public void testBasic() {{
        // TODO: Add test implementation
        assertNotNull(target);
    }}
    
    @AfterMethod
    public void tearDown() {{
        target = null;
    }}
}}"""
