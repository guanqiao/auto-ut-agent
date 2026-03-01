"""Manager for test file operations."""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class TestFileManager:
    """Manages test file creation and modification."""
    
    def __init__(self, project_path: str):
        """Initialize test file manager.
        
        Args:
            project_path: Path to the project
        """
        self.project_path = Path(project_path)
    
    def save_test_file(self, source_file: str, test_code: str) -> str:
        """Save test code to file.
        
        Args:
            source_file: Original source file path
            test_code: Generated test code
            
        Returns:
            Test file path
        """
        source_path = Path(source_file)
        
        # Determine test file path
        try:
            relative_path = source_path.relative_to(
                self.project_path / "src" / "main" / "java"
            )
        except ValueError:
            # If not in src/main/java, use the file name directly
            relative_path = source_path.relative_to(self.project_path)
        
        test_file_name = source_path.stem + "Test.java"
        test_path = (
            self.project_path / "src" / "test" / "java" /
            relative_path.parent / test_file_name
        )
        
        # Create directory if needed
        test_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write test file with proper encoding
        try:
            test_path.write_text(test_code, encoding='utf-8')
            logger.info(f"Test file saved: {test_path}")
        except Exception as e:
            logger.exception(f"Failed to save test file: {e}")
            raise
        
        return str(test_path)
    
    def append_test_code(self, test_file_path: str, additional_code: str) -> bool:
        """Append additional test code to existing file.
        
        Args:
            test_file_path: Path to test file
            additional_code: Additional test code
            
        Returns:
            True if successful
        """
        test_path = Path(test_file_path)
        
        if not test_path.exists():
            logger.warning(f"Test file does not exist: {test_file_path}")
            return False
        
        try:
            content = test_path.read_text(encoding='utf-8')
            
            # Find position to insert (before last closing brace)
            insert_pos = content.rfind('}')
            if insert_pos > 0:
                new_content = (
                    content[:insert_pos] + "\n" +
                    additional_code + "\n" +
                    content[insert_pos:]
                )
                test_path.write_text(new_content, encoding='utf-8')
                logger.info(f"Appended test code to: {test_file_path}")
                return True
            else:
                logger.warning(f"Could not find insertion point in: {test_file_path}")
                return False
        except Exception as e:
            logger.exception(f"Failed to append test code: {e}")
            return False
    
    def read_test_file(self, test_file_path: str) -> Optional[str]:
        """Read test file content.
        
        Args:
            test_file_path: Path to test file
            
        Returns:
            File content or None if failed
        """
        test_path = Path(test_file_path)
        
        if not test_path.exists():
            return None
        
        try:
            return test_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.exception(f"Failed to read test file: {e}")
            return None
    
    def get_test_file_path(self, source_file: str) -> str:
        """Get the expected test file path for a source file.
        
        Args:
            source_file: Original source file path
            
        Returns:
            Expected test file path
        """
        source_path = Path(source_file)
        
        try:
            relative_path = source_path.relative_to(
                self.project_path / "src" / "main" / "java"
            )
        except ValueError:
            relative_path = source_path.relative_to(self.project_path)
        
        test_file_name = source_path.stem + "Test.java"
        test_path = (
            self.project_path / "src" / "test" / "java" /
            relative_path.parent / test_file_name
        )
        
        return str(test_path)
