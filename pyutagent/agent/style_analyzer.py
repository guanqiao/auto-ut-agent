"""Code Style Analyzer and Learner for project-specific style learning.

This module provides:
- StyleAnalyzer: Analyze code style from existing codebase
- StyleProfile: Learned style profile for code generation
- StyleApplier: Apply learned style to generated code
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from collections import Counter
import json

logger = logging.getLogger(__name__)


class NamingConvention(Enum):
    """Naming convention types."""
    CAMEL_CASE = "camelCase"
    PASCAL_CASE = "PascalCase"
    SNAKE_CASE = "snake_case"
    SCREAMING_SNAKE = "SCREAMING_SNAKE"
    KEBAB_CASE = "kebab-case"
    UNKNOWN = "unknown"


class IndentStyle(Enum):
    """Indentation styles."""
    SPACES = "spaces"
    TABS = "tabs"
    UNKNOWN = "unknown"


@dataclass
class NamingStats:
    """Statistics for naming conventions."""
    convention: NamingConvention
    count: int = 0
    examples: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "convention": self.convention.value,
            "count": self.count,
            "examples": self.examples[:10],
        }


@dataclass
class StyleProfile:
    """Learned code style profile."""
    project_name: str
    language: str = "java"
    
    # Naming conventions
    class_naming: Optional[NamingConvention] = None
    method_naming: Optional[NamingConvention] = None
    variable_naming: Optional[NamingConvention] = None
    constant_naming: Optional[NamingConvention] = None
    package_naming: Optional[NamingConvention] = None
    
    # Formatting
    indent_style: IndentStyle = IndentStyle.SPACES
    indent_size: int = 4
    max_line_length: int = 120
    
    # Braces
    opening_brace_same_line: bool = True
    closing_brace_newline: bool = True
    
    # Spacing
    space_after_keyword: bool = True
    space_before_brace: bool = True
    space_around_operators: bool = True
    
    # Imports
    import_order: List[str] = field(default_factory=list)
    wildcard_imports: bool = False
    
    # Comments
    javadoc_style: bool = True
    inline_comment_style: str = "//"
    
    # Annotations
    annotation_on_same_line: bool = False
    annotation_parameter_alignment: str = "wrap"
    
    # Statistics
    files_analyzed: int = 0
    confidence_score: float = 0.0
    
    # Examples
    class_examples: List[str] = field(default_factory=list)
    method_examples: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_name": self.project_name,
            "language": self.language,
            "class_naming": self.class_naming.value if self.class_naming else None,
            "method_naming": self.method_naming.value if self.method_naming else None,
            "variable_naming": self.variable_naming.value if self.variable_naming else None,
            "constant_naming": self.constant_naming.value if self.constant_naming else None,
            "package_naming": self.package_naming.value if self.package_naming else None,
            "indent_style": self.indent_style.value,
            "indent_size": self.indent_size,
            "max_line_length": self.max_line_length,
            "opening_brace_same_line": self.opening_brace_same_line,
            "closing_brace_newline": self.closing_brace_newline,
            "space_after_keyword": self.space_after_keyword,
            "space_before_brace": self.space_before_brace,
            "space_around_operators": self.space_around_operators,
            "import_order": self.import_order,
            "wildcard_imports": self.wildcard_imports,
            "javadoc_style": self.javadoc_style,
            "inline_comment_style": self.inline_comment_style,
            "annotation_on_same_line": self.annotation_on_same_line,
            "annotation_parameter_alignment": self.annotation_parameter_alignment,
            "files_analyzed": self.files_analyzed,
            "confidence_score": self.confidence_score,
            "class_examples": self.class_examples[:5],
            "method_examples": self.method_examples[:5],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StyleProfile":
        return cls(
            project_name=data.get("project_name", "unknown"),
            language=data.get("language", "java"),
            class_naming=NamingConvention(data["class_naming"]) if data.get("class_naming") else None,
            method_naming=NamingConvention(data["method_naming"]) if data.get("method_naming") else None,
            variable_naming=NamingConvention(data["variable_naming"]) if data.get("variable_naming") else None,
            constant_naming=NamingConvention(data["constant_naming"]) if data.get("constant_naming") else None,
            package_naming=NamingConvention(data["package_naming"]) if data.get("package_naming") else None,
            indent_style=IndentStyle(data.get("indent_style", "spaces")),
            indent_size=data.get("indent_size", 4),
            max_line_length=data.get("max_line_length", 120),
            opening_brace_same_line=data.get("opening_brace_same_line", True),
            closing_brace_newline=data.get("closing_brace_newline", True),
            space_after_keyword=data.get("space_after_keyword", True),
            space_before_brace=data.get("space_before_brace", True),
            space_around_operators=data.get("space_around_operators", True),
            import_order=data.get("import_order", []),
            wildcard_imports=data.get("wildcard_imports", False),
            javadoc_style=data.get("javadoc_style", True),
            inline_comment_style=data.get("inline_comment_style", "//"),
            annotation_on_same_line=data.get("annotation_on_same_line", False),
            annotation_parameter_alignment=data.get("annotation_parameter_alignment", "wrap"),
            files_analyzed=data.get("files_analyzed", 0),
            confidence_score=data.get("confidence_score", 0.0),
            class_examples=data.get("class_examples", []),
            method_examples=data.get("method_examples", []),
        )


class StyleAnalyzer:
    """Analyzes code style from existing codebase."""
    
    PATTERNS = {
        "java": {
            "class": r'(?:public|private|protected)?\s*(?:abstract\s+)?class\s+(\w+)',
            "interface": r'interface\s+(\w+)',
            "method": r'(?:public|private|protected|static)\s+[\w<>\[\],\s]+\s+(\w+)\s*\(',
            "variable": r'(?:private|protected|public)\s+[\w<>\[\],\s]+\s+(\w+)\s*[;=]',
            "constant": r'(?:public\s+)?(?:static\s+)?final\s+[\w<>\[\],\s]+\s+(\w+)\s*=',
            "package": r'package\s+([\w.]+)',
            "import": r'import\s+([\w.]+)',
        },
        "python": {
            "class": r'class\s+(\w+)',
            "function": r'def\s+(\w+)',
            "variable": r'(\w+)\s*=\s*',
            "constant": r'^([A-Z][A-Z0-9_]*)\s*=',
        },
    }
    
    def __init__(self, project_path: str):
        """Initialize style analyzer.
        
        Args:
            project_path: Path to the project
        """
        self.project_path = Path(project_path)
    
    def analyze_project(
        self,
        file_patterns: Optional[List[str]] = None,
        max_files: int = 100
    ) -> StyleProfile:
        """Analyze project code style.
        
        Args:
            file_patterns: File patterns to analyze
            max_files: Maximum files to analyze
            
        Returns:
            Learned StyleProfile
        """
        if file_patterns is None:
            file_patterns = ["**/*.java", "**/*.py"]
        
        profile = StyleProfile(
            project_name=self.project_path.name,
            language="java"
        )
        
        files_analyzed = 0
        class_names: List[str] = []
        method_names: List[str] = []
        variable_names: List[str] = []
        constant_names: List[str] = []
        package_names: List[str] = []
        imports: List[str] = []
        
        indent_sizes: Counter = Counter()
        line_lengths: List[int] = []
        brace_styles: Counter = Counter()
        
        for pattern in file_patterns:
            for file_path in self.project_path.glob(pattern):
                if files_analyzed >= max_files:
                    break
                
                try:
                    content = file_path.read_text(encoding="utf-8")
                    language = "java" if file_path.suffix == ".java" else "python"
                    
                    if language == "java":
                        self._analyze_java_file(
                            content, class_names, method_names,
                            variable_names, constant_names,
                            package_names, imports,
                            indent_sizes, line_lengths, brace_styles
                        )
                    
                    files_analyzed += 1
                    
                except Exception as e:
                    logger.warning(f"[StyleAnalyzer] Failed to analyze {file_path}: {e}")
        
        # Determine naming conventions
        profile.class_naming = self._detect_naming_convention(class_names)
        profile.method_naming = self._detect_naming_convention(method_names)
        profile.variable_naming = self._detect_naming_convention(variable_names)
        profile.constant_naming = self._detect_naming_convention(constant_names)
        profile.package_naming = self._detect_naming_convention(package_names)
        
        # Determine formatting
        if indent_sizes:
            most_common = indent_sizes.most_common(1)[0]
            profile.indent_size = most_common[0]
            profile.indent_style = IndentStyle.SPACES if most_common[0] > 0 else IndentStyle.TABS
        
        if line_lengths:
            profile.max_line_length = int(sum(line_lengths) / len(line_lengths) * 1.5)
        
        # Determine brace style
        if brace_styles:
            profile.opening_brace_same_line = brace_styles.get("same_line", 0) > brace_styles.get("new_line", 0)
        
        # Import order
        profile.import_order = self._determine_import_order(imports)
        
        # Examples
        profile.class_examples = class_names[:5]
        profile.method_examples = method_names[:5]
        
        # Statistics
        profile.files_analyzed = files_analyzed
        profile.confidence_score = self._calculate_confidence(
            len(class_names), len(method_names), files_analyzed
        )
        
        logger.info(f"[StyleAnalyzer] Analyzed {files_analyzed} files, confidence: {profile.confidence_score:.2f}")
        
        return profile
    
    def _analyze_java_file(
        self,
        content: str,
        class_names: List[str],
        method_names: List[str],
        variable_names: List[str],
        constant_names: List[str],
        package_names: List[str],
        imports: List[str],
        indent_sizes: Counter,
        line_lengths: List[int],
        brace_styles: Counter
    ):
        """Analyze a Java file."""
        patterns = self.PATTERNS["java"]
        
        # Extract names
        class_names.extend(re.findall(patterns["class"], content))
        method_names.extend(re.findall(patterns["method"], content))
        variable_names.extend(re.findall(patterns["variable"], content))
        constant_names.extend(re.findall(patterns["constant"], content))
        package_names.extend(re.findall(patterns["package"], content))
        imports.extend(re.findall(patterns["import"], content))
        
        # Analyze indentation
        lines = content.split('\n')
        for line in lines:
            line_lengths.append(len(line))
            
            if line.startswith(' '):
                spaces = len(line) - len(line.lstrip(' '))
                if spaces > 0:
                    indent_sizes[spaces] += 1
            elif line.startswith('\t'):
                indent_sizes[0] += 1
        
        # Analyze brace style
        for match in re.finditer(r'(class|interface|method|if|for|while)\s*\w*\s*\{', content):
            brace_styles["same_line"] += 1
        
        for match in re.finditer(r'\n\s*\{', content):
            brace_styles["new_line"] += 1
    
    def _detect_naming_convention(self, names: List[str]) -> NamingConvention:
        """Detect naming convention from examples."""
        if not names:
            return NamingConvention.UNKNOWN
        
        conventions: Counter = Counter()
        
        for name in names:
            if re.match(r'^[a-z][a-z0-9]*([A-Z][a-z0-9]*)*$', name):
                conventions[NamingConvention.CAMEL_CASE] += 1
            elif re.match(r'^[A-Z][a-z0-9]*([A-Z][a-z0-9]*)*$', name):
                conventions[NamingConvention.PASCAL_CASE] += 1
            elif re.match(r'^[a-z][a-z0-9]*(_[a-z0-9]+)*$', name):
                conventions[NamingConvention.SNAKE_CASE] += 1
            elif re.match(r'^[A-Z][A-Z0-9]*(_[A-Z0-9]+)*$', name):
                conventions[NamingConvention.SCREAMING_SNAKE] += 1
        
        if not conventions:
            return NamingConvention.UNKNOWN
        
        return conventions.most_common(1)[0][0]
    
    def _determine_import_order(self, imports: List[str]) -> List[str]:
        """Determine import ordering pattern."""
        if not imports:
            return []
        
        order = []
        has_java = any(i.startswith("java.") for i in imports)
        has_javax = any(i.startswith("javax.") for i in imports)
        has_org = any(i.startswith("org.") for i in imports)
        has_com = any(i.startswith("com.") for i in imports)
        
        if has_java:
            order.append("java.*")
        if has_javax:
            order.append("javax.*")
        if has_org:
            order.append("org.*")
        if has_com:
            order.append("com.*")
        
        return order
    
    def _calculate_confidence(
        self,
        class_count: int,
        method_count: int,
        file_count: int
    ) -> float:
        """Calculate confidence score for the analysis."""
        if file_count == 0:
            return 0.0
        
        sample_size_score = min(file_count / 50, 1.0) * 0.4
        diversity_score = min((class_count + method_count) / 100, 1.0) * 0.4
        consistency_score = 0.2
        
        return sample_size_score + diversity_score + consistency_score


class StyleApplier:
    """Applies learned style to generated code."""
    
    def __init__(self, profile: StyleProfile):
        """Initialize style applier.
        
        Args:
            profile: Style profile to apply
        """
        self.profile = profile
    
    def apply_to_class(self, class_name: str, code: str) -> str:
        """Apply style to class code."""
        if self.profile.class_naming == NamingConvention.PASCAL_CASE:
            class_name = self._to_pascal_case(class_name)
        elif self.profile.class_naming == NamingConvention.CAMEL_CASE:
            class_name = self._to_camel_case(class_name)
        
        code = self._apply_indentation(code)
        code = self._apply_brace_style(code)
        
        return code
    
    def apply_to_method(self, method_name: str, code: str) -> str:
        """Apply style to method code."""
        if self.profile.method_naming == NamingConvention.CAMEL_CASE:
            method_name = self._to_camel_case(method_name)
        elif self.profile.method_naming == NamingConvention.SNAKE_CASE:
            method_name = self._to_snake_case(method_name)
        
        code = self._apply_indentation(code)
        
        return code
    
    def _apply_indentation(self, code: str) -> str:
        """Apply indentation style."""
        lines = code.split('\n')
        result = []
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            
            if stripped.startswith('}'):
                indent_level = max(0, indent_level - 1)
            
            if stripped:
                indent = ' ' * (indent_level * self.profile.indent_size)
                result.append(indent + stripped)
            else:
                result.append('')
            
            if stripped.endswith('{'):
                indent_level += 1
        
        return '\n'.join(result)
    
    def _apply_brace_style(self, code: str) -> str:
        """Apply brace style."""
        if self.profile.opening_brace_same_line:
            code = re.sub(r'\n\s*\{', ' {', code)
        else:
            code = re.sub(r'\s*\{', '\n{', code)
        
        return code
    
    def _to_pascal_case(self, name: str) -> str:
        """Convert to PascalCase."""
        parts = re.split(r'[_\s]', name)
        return ''.join(p.capitalize() for p in parts)
    
    def _to_camel_case(self, name: str) -> str:
        """Convert to camelCase."""
        pascal = self._to_pascal_case(name)
        return pascal[0].lower() + pascal[1:] if pascal else pascal
    
    def _to_snake_case(self, name: str) -> str:
        """Convert to snake_case."""
        s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
    def get_style_prompt(self) -> str:
        """Get style instructions for LLM prompt."""
        instructions = []
        
        if self.profile.class_naming:
            instructions.append(f"Use {self.profile.class_naming.value} for class names")
        
        if self.profile.method_naming:
            instructions.append(f"Use {self.profile.method_naming.value} for method names")
        
        instructions.append(f"Use {self.profile.indent_size} spaces for indentation")
        instructions.append(f"Max line length: {self.profile.max_line_length}")
        
        if self.profile.opening_brace_same_line:
            instructions.append("Place opening brace on same line")
        else:
            instructions.append("Place opening brace on new line")
        
        if self.profile.import_order:
            instructions.append(f"Import order: {', '.join(self.profile.import_order)}")
        
        return "\n".join(f"- {i}" for i in instructions)


def create_style_profile(project_path: str) -> StyleProfile:
    """Create style profile from project.
    
    Args:
        project_path: Path to the project
        
    Returns:
        Learned StyleProfile
    """
    analyzer = StyleAnalyzer(project_path)
    return analyzer.analyze_project()
