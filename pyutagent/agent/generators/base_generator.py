"""Base class for test generators."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class BaseTestGenerator(ABC):
    """Abstract base class for test generators."""
    
    def __init__(self, project_path: str):
        """Initialize test generator.
        
        Args:
            project_path: Path to the project
        """
        self.project_path = project_path
    
    @abstractmethod
    async def generate_initial_test(self, class_info: Dict[str, Any]) -> str:
        """Generate initial test code for a class.
        
        Args:
            class_info: Class information dictionary
            
        Returns:
            Generated test code
        """
        pass
    
    @abstractmethod
    async def generate_additional_tests(
        self,
        class_info: Dict[str, Any],
        uncovered_lines: list
    ) -> str:
        """Generate additional tests for uncovered lines.
        
        Args:
            class_info: Class information dictionary
            uncovered_lines: List of uncovered line numbers
            
        Returns:
            Additional test code
        """
        pass
    
    def _generate_basic_test_template(self, class_info: Dict[str, Any]) -> str:
        """Generate basic test template as fallback.
        
        Args:
            class_info: Class information dictionary
            
        Returns:
            Basic test code
        """
        package = class_info.get('package', '')
        class_name = class_info.get('name', 'Unknown')
        test_class_name = f"{class_name}Test"
        
        package_line = f"package {package};\n\n" if package else ""
        
        return f"""{package_line}import org.junit.jupiter.api.Test;
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
