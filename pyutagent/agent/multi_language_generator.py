"""Multi-language test generator supporting Python, TypeScript, and more.

This module provides:
- MultiLanguageTestGenerator: Generate tests for multiple languages
- LanguageHandler: Language-specific test generation logic
- TestFrameworkDetector: Detect test frameworks automatically
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SupportedLanguage(Enum):
    """Supported programming languages."""
    JAVA = "java"
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    GO = "go"
    RUST = "rust"
    UNKNOWN = "unknown"


class TestFramework(Enum):
    """Supported test frameworks."""
    # Java
    JUNIT4 = "junit4"
    JUNIT5 = "junit5"
    TESTNG = "testng"
    
    # Python
    PYTEST = "pytest"
    UNITTEST = "unittest"
    
    # TypeScript/JavaScript
    JEST = "jest"
    MOCHA = "mocha"
    VITEST = "vitest"
    
    # Go
    TESTING = "testing"
    
    # Rust
    CARGO_TEST = "cargo_test"
    
    UNKNOWN = "unknown"


@dataclass
class LanguageConfig:
    """Configuration for a language."""
    language: SupportedLanguage
    file_extensions: List[str]
    test_frameworks: List[TestFramework]
    default_framework: TestFramework
    test_file_suffix: str
    test_directory: str
    import_patterns: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language.value,
            "file_extensions": self.file_extensions,
            "test_frameworks": [f.value for f in self.test_frameworks],
            "default_framework": self.default_framework.value,
            "test_file_suffix": self.test_file_suffix,
            "test_directory": self.test_directory,
            "import_patterns": self.import_patterns,
        }


LANGUAGE_CONFIGS: Dict[SupportedLanguage, LanguageConfig] = {
    SupportedLanguage.JAVA: LanguageConfig(
        language=SupportedLanguage.JAVA,
        file_extensions=[".java"],
        test_frameworks=[TestFramework.JUNIT5, TestFramework.JUNIT4, TestFramework.TESTNG],
        default_framework=TestFramework.JUNIT5,
        test_file_suffix="Test",
        test_directory="src/test/java",
        import_patterns=[
            "import org.junit.jupiter.api.*;",
            "import static org.junit.jupiter.api.Assertions.*;",
        ],
    ),
    SupportedLanguage.PYTHON: LanguageConfig(
        language=SupportedLanguage.PYTHON,
        file_extensions=[".py"],
        test_frameworks=[TestFramework.PYTEST, TestFramework.UNITTEST],
        default_framework=TestFramework.PYTEST,
        test_file_suffix="_test",
        test_directory="tests",
        import_patterns=[
            "import pytest",
            "from unittest import TestCase",
        ],
    ),
    SupportedLanguage.TYPESCRIPT: LanguageConfig(
        language=SupportedLanguage.TYPESCRIPT,
        file_extensions=[".ts", ".tsx"],
        test_frameworks=[TestFramework.JEST, TestFramework.VITEST],
        default_framework=TestFramework.JEST,
        test_file_suffix=".test",
        test_directory="tests",
        import_patterns=[
            "import { describe, it, expect } from '@jest/globals';",
        ],
    ),
    SupportedLanguage.JAVASCRIPT: LanguageConfig(
        language=SupportedLanguage.JAVASCRIPT,
        file_extensions=[".js", ".jsx"],
        test_frameworks=[TestFramework.JEST, TestFramework.MOCHA],
        default_framework=TestFramework.JEST,
        test_file_suffix=".test",
        test_directory="tests",
        import_patterns=[
            "const { describe, it, expect } = require('@jest/globals');",
        ],
    ),
    SupportedLanguage.GO: LanguageConfig(
        language=SupportedLanguage.GO,
        file_extensions=[".go"],
        test_frameworks=[TestFramework.TESTING],
        default_framework=TestFramework.TESTING,
        test_file_suffix="_test",
        test_directory=".",
        import_patterns=[
            'import "testing"',
        ],
    ),
    SupportedLanguage.RUST: LanguageConfig(
        language=SupportedLanguage.RUST,
        file_extensions=[".rs"],
        test_frameworks=[TestFramework.CARGO_TEST],
        default_framework=TestFramework.CARGO_TEST,
        test_file_suffix="",
        test_directory="tests",
        import_patterns=[
            "#[cfg(test)]",
            "#[test]",
        ],
    ),
}


@dataclass
class ParsedClass:
    """Parsed class/function information."""
    name: str
    type: str  # "class", "function", "method"
    parameters: List[Dict[str, str]] = field(default_factory=list)
    return_type: Optional[str] = None
    docstring: Optional[str] = None
    body: Optional[str] = None
    line_start: int = 0
    line_end: int = 0
    methods: List["ParsedClass"] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "parameters": self.parameters,
            "return_type": self.return_type,
            "docstring": self.docstring,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "methods": [m.to_dict() for m in self.methods],
        }


class LanguageHandler(ABC):
    """Abstract base class for language handlers."""
    
    @property
    @abstractmethod
    def language(self) -> SupportedLanguage:
        """Return the language this handler supports."""
        pass
    
    @abstractmethod
    def parse_file(self, content: str) -> List[ParsedClass]:
        """Parse source file and extract testable units."""
        pass
    
    @abstractmethod
    def generate_test_imports(self, framework: TestFramework) -> str:
        """Generate import statements for tests."""
        pass
    
    @abstractmethod
    def generate_test_class(
        self,
        parsed: ParsedClass,
        framework: TestFramework
    ) -> str:
        """Generate test class/function."""
        pass
    
    @abstractmethod
    def get_test_file_path(
        self,
        source_path: str,
        test_dir: str
    ) -> str:
        """Get the test file path for a source file."""
        pass


class JavaHandler(LanguageHandler):
    """Handler for Java language."""
    
    @property
    def language(self) -> SupportedLanguage:
        return SupportedLanguage.JAVA
    
    def parse_file(self, content: str) -> List[ParsedClass]:
        classes = []
        
        class_pattern = r'(?:public\s+)?(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w,\s]+)?\s*\{'
        method_pattern = r'(?:public|private|protected)\s+(?:static\s+)?(?:\w+(?:<[\w\s,<>]+>)?)\s+(\w+)\s*\(([^)]*)\)'
        
        lines = content.split('\n')
        
        for match in re.finditer(class_pattern, content):
            class_name = match.group(1)
            start_pos = match.start()
            start_line = content[:start_pos].count('\n') + 1
            
            brace_count = 0
            end_pos = start_pos
            for i in range(start_pos, len(content)):
                if content[i] == '{':
                    brace_count += 1
                elif content[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i
                        break
            
            end_line = content[:end_pos].count('\n') + 1
            class_body = content[start_pos:end_pos + 1]
            
            methods = []
            for method_match in re.finditer(method_pattern, class_body):
                method_name = method_match.group(1)
                params_str = method_match.group(2)
                
                if method_name in ('equals', 'hashCode', 'toString', 'clone'):
                    continue
                
                params = []
                if params_str.strip():
                    for param in params_str.split(','):
                        param = param.strip()
                        if param:
                            parts = param.rsplit(' ', 1)
                            if len(parts) == 2:
                                params.append({"type": parts[0], "name": parts[1]})
                
                methods.append(ParsedClass(
                    name=method_name,
                    type="method",
                    parameters=params,
                ))
            
            classes.append(ParsedClass(
                name=class_name,
                type="class",
                methods=methods,
                line_start=start_line,
                line_end=end_line,
            ))
        
        return classes
    
    def generate_test_imports(self, framework: TestFramework) -> str:
        if framework == TestFramework.JUNIT5:
            return """import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;
import org.mockito.Mockito;
import static org.mockito.Mockito.*;"""
        elif framework == TestFramework.JUNIT4:
            return """import org.junit.*;
import static org.junit.Assert.*;
import org.mockito.Mockito;
import static org.mockito.Mockito.*;"""
        else:
            return ""
    
    def generate_test_class(
        self,
        parsed: ParsedClass,
        framework: TestFramework
    ) -> str:
        test_class_name = f"{parsed.name}Test"
        
        lines = [f"class {test_class_name} {{"]
        
        if framework == TestFramework.JUNIT5:
            lines.append("    @BeforeEach")
            lines.append("    void setUp() {")
            lines.append("    }")
            lines.append("")
        
        for method in parsed.methods:
            if framework == TestFramework.JUNIT5:
                lines.append(f"    @Test")
                lines.append(f"    void test{method.name.capitalize()}() {{")
                lines.append(f"        // TODO: Implement test for {method.name}")
                lines.append("    }")
                lines.append("")
        
        lines.append("}")
        
        return '\n'.join(lines)
    
    def get_test_file_path(self, source_path: str, test_dir: str) -> str:
        path = Path(source_path)
        return str(Path(test_dir) / f"{path.stem}Test{path.suffix}")


class PythonHandler(LanguageHandler):
    """Handler for Python language."""
    
    @property
    def language(self) -> SupportedLanguage:
        return SupportedLanguage.PYTHON
    
    def parse_file(self, content: str) -> List[ParsedClass]:
        classes = []
        
        class_pattern = r'class\s+(\w+)(?:\([^)]*\))?:'
        def_pattern = r'def\s+(\w+)\s*\(([^)]*)\)'
        
        lines = content.split('\n')
        
        for match in re.finditer(class_pattern, content):
            class_name = match.group(1)
            start_line = content[:match.start()].count('\n') + 1
            
            methods = []
            class_indent = len(lines[start_line - 1]) - len(lines[start_line - 1].lstrip())
            
            for i in range(start_line, len(lines)):
                line = lines[i]
                if not line.strip():
                    continue
                
                current_indent = len(line) - len(line.lstrip())
                
                if current_indent <= class_indent and line.strip() and not line.strip().startswith('#'):
                    break
                
                def_match = re.match(r'(\s*)def\s+(\w+)\s*\(([^)]*)\)', line)
                if def_match:
                    method_name = def_match.group(2)
                    params_str = def_match.group(3)
                    
                    if method_name.startswith('_') and not method_name.startswith('__'):
                        continue
                    
                    params = []
                    if params_str.strip():
                        for param in params_str.split(','):
                            param = param.strip()
                            if param and param != 'self' and param != 'cls':
                                if ':' in param:
                                    name, type_hint = param.split(':', 1)
                                    params.append({"name": name.strip(), "type": type_hint.strip()})
                                else:
                                    if '=' in param:
                                        param = param.split('=')[0]
                                    params.append({"name": param.strip(), "type": "Any"})
                    
                    methods.append(ParsedClass(
                        name=method_name,
                        type="method",
                        parameters=params,
                    ))
            
            classes.append(ParsedClass(
                name=class_name,
                type="class",
                methods=methods,
                line_start=start_line,
            ))
        
        for match in re.finditer(def_pattern, content):
            def_name = match.group(1)
            
            if def_name.startswith('_'):
                continue
            
            if not any(def_name in [m.name for m in c.methods] for c in classes):
                start_line = content[:match.start()].count('\n') + 1
                
                params = []
                params_str = match.group(2)
                if params_str.strip():
                    for param in params_str.split(','):
                        param = param.strip()
                        if param and param != 'self' and param != 'cls':
                            if '=' in param:
                                param = param.split('=')[0]
                            params.append({"name": param.strip(), "type": "Any"})
                
                classes.append(ParsedClass(
                    name=def_name,
                    type="function",
                    parameters=params,
                    line_start=start_line,
                ))
        
        return classes
    
    def generate_test_imports(self, framework: TestFramework) -> str:
        if framework == TestFramework.PYTEST:
            return """import pytest
from unittest.mock import Mock, patch, MagicMock
"""
        elif framework == TestFramework.UNITTEST:
            return """import unittest
from unittest.mock import Mock, patch, MagicMock
"""
        return ""
    
    def generate_test_class(
        self,
        parsed: ParsedClass,
        framework: TestFramework
    ) -> str:
        lines = []
        
        if framework == TestFramework.PYTEST:
            if parsed.type == "class":
                lines.append(f"class Test{parsed.name}:")
                lines.append("")
                
                for method in parsed.methods:
                    lines.append(f"    def test_{method.name}(self):")
                    lines.append(f"        # TODO: Implement test for {method.name}")
                    lines.append("        pass")
                    lines.append("")
            else:
                lines.append(f"def test_{parsed.name}():")
                lines.append(f"    # TODO: Implement test for {parsed.name}")
                lines.append("    pass")
        
        elif framework == TestFramework.UNITTEST:
            lines.append(f"class Test{parsed.name}(unittest.TestCase):")
            lines.append("")
            lines.append("    def setUp(self):")
            lines.append("        pass")
            lines.append("")
            
            for method in parsed.methods:
                lines.append(f"    def test_{method.name}(self):")
                lines.append(f"        # TODO: Implement test for {method.name}")
                lines.append("        pass")
                lines.append("")
        
        return '\n'.join(lines)
    
    def get_test_file_path(self, source_path: str, test_dir: str) -> str:
        path = Path(source_path)
        return str(Path(test_dir) / f"test_{path.stem}.py")


class TypeScriptHandler(LanguageHandler):
    """Handler for TypeScript language."""
    
    @property
    def language(self) -> SupportedLanguage:
        return SupportedLanguage.TYPESCRIPT
    
    def parse_file(self, content: str) -> List[ParsedClass]:
        classes = []
        
        class_pattern = r'(?:export\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w,\s]+)?\s*\{'
        interface_pattern = r'(?:export\s+)?interface\s+(\w+)\s*\{'
        function_pattern = r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)'
        method_pattern = r'(?:public|private|protected|async)\s+(\w+)\s*\(([^)]*)\s*\)(?:\s*:\s*([\w<>\[\],\s]+))?'
        
        for match in re.finditer(class_pattern, content):
            class_name = match.group(1)
            start_line = content[:match.start()].count('\n') + 1
            
            brace_count = 0
            end_pos = match.start()
            for i in range(match.start(), len(content)):
                if content[i] == '{':
                    brace_count += 1
                elif content[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i
                        break
            
            class_body = content[match.start():end_pos + 1]
            
            methods = []
            for method_match in re.finditer(method_pattern, class_body):
                method_name = method_match.group(1)
                
                if method_name == 'constructor':
                    continue
                
                params = []
                params_str = method_match.group(2)
                if params_str.strip():
                    for param in params_str.split(','):
                        param = param.strip()
                        if param:
                            if ':' in param:
                                name, type_hint = param.split(':', 1)
                                params.append({"name": name.strip(), "type": type_hint.strip()})
                            else:
                                params.append({"name": param.strip(), "type": "any"})
                
                methods.append(ParsedClass(
                    name=method_name,
                    type="method",
                    parameters=params,
                    return_type=method_match.group(3) if method_match.group(3) else "void",
                ))
            
            classes.append(ParsedClass(
                name=class_name,
                type="class",
                methods=methods,
                line_start=start_line,
            ))
        
        for match in re.finditer(function_pattern, content):
            func_name = match.group(1)
            start_line = content[:match.start()].count('\n') + 1
            
            params = []
            params_str = match.group(2)
            if params_str.strip():
                for param in params_str.split(','):
                    param = param.strip()
                    if param:
                        if ':' in param:
                            name, type_hint = param.split(':', 1)
                            params.append({"name": name.strip(), "type": type_hint.strip()})
                        else:
                            params.append({"name": param.strip(), "type": "any"})
            
            classes.append(ParsedClass(
                name=func_name,
                type="function",
                parameters=params,
                line_start=start_line,
            ))
        
        return classes
    
    def generate_test_imports(self, framework: TestFramework) -> str:
        if framework == TestFramework.JEST:
            return """import { describe, it, expect, beforeEach, jest } from '@jest/globals';
import { Mock } from 'jest-mock';
"""
        elif framework == TestFramework.VITEST:
            return """import { describe, it, expect, beforeEach, vi } from 'vitest';
"""
        return ""
    
    def generate_test_class(
        self,
        parsed: ParsedClass,
        framework: TestFramework
    ) -> str:
        lines = []
        
        if framework in (TestFramework.JEST, TestFramework.VITEST):
            lines.append(f"describe('{parsed.name}', () => {{")
            lines.append("  beforeEach(() => {")
            lines.append("  });")
            lines.append("")
            
            if parsed.type == "class":
                for method in parsed.methods:
                    lines.append(f"  it('should test {method.name}', () => {{")
                    lines.append(f"    // TODO: Implement test for {method.name}")
                    lines.append("  });")
                    lines.append("")
            else:
                lines.append(f"  it('should test {parsed.name}', () => {{")
                lines.append(f"    // TODO: Implement test for {parsed.name}")
                lines.append("  });")
            
            lines.append("});")
        
        return '\n'.join(lines)
    
    def get_test_file_path(self, source_path: str, test_dir: str) -> str:
        path = Path(source_path)
        return str(Path(test_dir) / f"{path.stem}.test{path.suffix}")


class MultiLanguageTestGenerator:
    """Generate tests for multiple programming languages."""
    
    HANDLERS: Dict[SupportedLanguage, LanguageHandler] = {
        SupportedLanguage.JAVA: JavaHandler(),
        SupportedLanguage.PYTHON: PythonHandler(),
        SupportedLanguage.TYPESCRIPT: TypeScriptHandler(),
    }
    
    def __init__(self, llm_client=None):
        """Initialize multi-language test generator.
        
        Args:
            llm_client: Optional LLM client for enhanced generation
        """
        self.llm_client = llm_client
    
    def detect_language(self, file_path: str) -> SupportedLanguage:
        """Detect language from file extension."""
        ext = Path(file_path).suffix.lower()
        
        for lang, config in LANGUAGE_CONFIGS.items():
            if ext in config.file_extensions:
                return lang
        
        return SupportedLanguage.UNKNOWN
    
    def detect_framework(
        self,
        project_path: str,
        language: SupportedLanguage
    ) -> TestFramework:
        """Detect test framework from project configuration."""
        project = Path(project_path)
        config = LANGUAGE_CONFIGS.get(language)
        
        if not config:
            return TestFramework.UNKNOWN
        
        if language == SupportedLanguage.JAVA:
            pom = project / "pom.xml"
            if pom.exists():
                content = pom.read_text(encoding="utf-8")
                if "junit-jupiter" in content or "junit5" in content:
                    return TestFramework.JUNIT5
                if "junit" in content:
                    return TestFramework.JUNIT4
                if "testng" in content:
                    return TestFramework.TESTNG
        
        elif language == SupportedLanguage.PYTHON:
            pyproject = project / "pyproject.toml"
            setup_cfg = project / "setup.cfg"
            requirements = project / "requirements.txt"
            
            for config_file in [pyproject, setup_cfg, requirements]:
                if config_file.exists():
                    content = config_file.read_text(encoding="utf-8")
                    if "pytest" in content:
                        return TestFramework.PYTEST
                    if "unittest" in content:
                        return TestFramework.UNITTEST
        
        elif language in (SupportedLanguage.TYPESCRIPT, SupportedLanguage.JAVASCRIPT):
            package_json = project / "package.json"
            if package_json.exists():
                content = package_json.read_text(encoding="utf-8")
                if "vitest" in content:
                    return TestFramework.VITEST
                if "jest" in content:
                    return TestFramework.JEST
                if "mocha" in content:
                    return TestFramework.MOCHA
        
        return config.default_framework
    
    def get_handler(self, language: SupportedLanguage) -> Optional[LanguageHandler]:
        """Get language handler."""
        return self.HANDLERS.get(language)
    
    def parse_file(
        self,
        content: str,
        language: SupportedLanguage
    ) -> List[ParsedClass]:
        """Parse source file."""
        handler = self.get_handler(language)
        if handler:
            return handler.parse_file(content)
        return []
    
    def generate_tests(
        self,
        source_path: str,
        content: str,
        framework: Optional[TestFramework] = None,
        project_path: Optional[str] = None
    ) -> Tuple[str, str]:
        """Generate test file for source file.
        
        Args:
            source_path: Path to source file
            content: Source file content
            framework: Optional test framework
            project_path: Optional project path for framework detection
            
        Returns:
            Tuple of (test_file_path, test_content)
        """
        language = self.detect_language(source_path)
        
        if language == SupportedLanguage.UNKNOWN:
            raise ValueError(f"Unsupported file type: {source_path}")
        
        handler = self.get_handler(language)
        config = LANGUAGE_CONFIGS[language]
        
        if framework is None and project_path:
            framework = self.detect_framework(project_path, language)
        elif framework is None:
            framework = config.default_framework
        
        parsed = handler.parse_file(content)
        
        imports = handler.generate_test_imports(framework)
        
        test_classes = []
        for p in parsed:
            test_class = handler.generate_test_class(p, framework)
            test_classes.append(test_class)
        
        test_content = imports + "\n\n" + "\n\n".join(test_classes)
        
        test_dir = config.test_directory
        test_path = handler.get_test_file_path(source_path, test_dir)
        
        return test_path, test_content
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return [lang.value for lang in LANGUAGE_CONFIGS.keys()]
    
    def get_supported_frameworks(self, language: SupportedLanguage) -> List[str]:
        """Get supported frameworks for a language."""
        config = LANGUAGE_CONFIGS.get(language)
        if config:
            return [f.value for f in config.test_frameworks]
        return []
