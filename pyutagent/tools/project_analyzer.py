"""Project analyzer for multi-file coordination.

This module provides project-level analysis capabilities:
- Project structure analysis
- Dependency graph building
- Multi-file coordination
- Cross-file refactoring support
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

logger = logging.getLogger(__name__)


@dataclass
class ClassInfo:
    """Information about a Java class."""
    name: str
    package: str
    file_path: str
    imports: List[str] = field(default_factory=list)
    fields: List[Dict[str, Any]] = field(default_factory=list)
    methods: List[Dict[str, Any]] = field(default_factory=list)
    annotations: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    is_interface: bool = False
    is_enum: bool = False
    modifiers: List[str] = field(default_factory=list)
    extends: Optional[str] = None
    implements: List[str] = field(default_factory=list)


@dataclass
class Dependency:
    """Represents a dependency between files."""
    source_file: str
    target_file: str
    dependency_type: str
    strength: float = 1.0
    symbols: List[str] = field(default_factory=list)


@dataclass
class ProjectStructure:
    """Structure of a Java project."""
    root_path: str
    source_dirs: List[str] = field(default_factory=list)
    test_dirs: List[str] = field(default_factory=list)
    classes: Dict[str, ClassInfo] = field(default_factory=dict)
    dependencies: List[Dependency] = field(default_factory=list)
    file_tree: Dict[str, Any] = field(default_factory=dict)
    package_structure: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class FileEdit:
    """Represents an edit to a file."""
    file_path: str
    edit_type: str
    content: str
    location: Optional[Tuple[int, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiEditResult:
    """Result of multi-file editing."""
    success: bool
    edited_files: List[str]
    failed_files: List[str]
    errors: List[Exception] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProjectAnalyzer:
    """Analyzes Java project structure and dependencies.
    
    Features:
    - Project structure discovery
    - Dependency graph building
    - Cross-file reference tracking
    - Package structure analysis
    """
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self._structure: Optional[ProjectStructure] = None
        self._class_index: Dict[str, ClassInfo] = {}
        self._file_index: Dict[str, str] = {}
    
    async def analyze_project(self) -> ProjectStructure:
        """Analyze the project structure.
        
        Returns:
            ProjectStructure with complete analysis
        """
        logger.info(f"[ProjectAnalyzer] Analyzing project: {self.project_path}")
        
        self._structure = ProjectStructure(root_path=str(self.project_path))
        
        self._structure.source_dirs = self._find_source_dirs()
        self._structure.test_dirs = self._find_test_dirs()
        
        for source_dir in self._structure.source_dirs:
            await self._analyze_source_dir(source_dir, is_test=False)
        
        for test_dir in self._structure.test_dirs:
            await self._analyze_source_dir(test_dir, is_test=True)
        
        self._build_dependency_graph()
        
        self._build_package_structure()
        
        logger.info(
            f"[ProjectAnalyzer] Analysis complete - "
            f"Classes: {len(self._structure.classes)}, "
            f"Dependencies: {len(self._structure.dependencies)}"
        )
        
        return self._structure
    
    def _find_source_dirs(self) -> List[str]:
        """Find source directories in the project."""
        source_dirs = []
        
        standard_paths = [
            "src/main/java",
            "src/java",
            "src",
            "source",
        ]
        
        for path in standard_paths:
            full_path = self.project_path / path
            if full_path.exists() and full_path.is_dir():
                source_dirs.append(str(full_path))
        
        if not source_dirs:
            for item in self.project_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    if self._has_java_files(item):
                        source_dirs.append(str(item))
        
        return source_dirs
    
    def _find_test_dirs(self) -> List[str]:
        """Find test directories in the project."""
        test_dirs = []
        
        standard_paths = [
            "src/test/java",
            "src/test",
            "test",
            "tests",
        ]
        
        for path in standard_paths:
            full_path = self.project_path / path
            if full_path.exists() and full_path.is_dir():
                test_dirs.append(str(full_path))
        
        return test_dirs
    
    def _has_java_files(self, directory: Path) -> bool:
        """Check if directory contains Java files."""
        for item in directory.rglob("*.java"):
            return True
        return False
    
    async def _analyze_source_dir(
        self,
        source_dir: str,
        is_test: bool = False
    ):
        """Analyze a source directory."""
        source_path = Path(source_dir)
        
        for java_file in source_path.rglob("*.java"):
            try:
                class_info = await self._analyze_java_file(java_file)
                if class_info:
                    self._structure.classes[class_info.name] = class_info
                    self._class_index[class_info.name] = class_info
                    self._file_index[str(java_file)] = class_info.name
            except Exception as e:
                logger.warning(f"[ProjectAnalyzer] Failed to analyze {java_file}: {e}")
    
    async def _analyze_java_file(self, file_path: Path) -> Optional[ClassInfo]:
        """Analyze a single Java file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"[ProjectAnalyzer] Cannot read {file_path}: {e}")
            return None
        
        package = self._extract_package(content)
        imports = self._extract_imports(content)
        
        class_match = re.search(
            r'(?:public\s+)?(?:abstract\s+)?(?:class|interface|enum)\s+(\w+)',
            content
        )
        
        if not class_match:
            return None
        
        class_name = class_match.group(1)
        
        is_interface = 'interface ' in content[class_match.start():class_match.end()+20]
        is_enum = 'enum ' in content[class_match.start():class_match.end()+20]
        
        extends = self._extract_extends(content)
        implements = self._extract_implements(content)
        
        fields = self._extract_fields(content)
        methods = self._extract_methods(content)
        annotations = self._extract_class_annotations(content)
        modifiers = self._extract_modifiers(content, class_match.start())
        
        dependencies = self._extract_dependencies(imports, content)
        
        return ClassInfo(
            name=class_name,
            package=package,
            file_path=str(file_path.relative_to(self.project_path)),
            imports=imports,
            fields=fields,
            methods=methods,
            annotations=annotations,
            dependencies=dependencies,
            is_interface=is_interface,
            is_enum=is_enum,
            modifiers=modifiers,
            extends=extends,
            implements=implements
        )
    
    def _extract_package(self, content: str) -> str:
        """Extract package name from content."""
        match = re.search(r'package\s+([\w.]+)\s*;', content)
        return match.group(1) if match else ""
    
    def _extract_imports(self, content: str) -> List[str]:
        """Extract import statements from content."""
        imports = []
        for match in re.finditer(r'import\s+(?:static\s+)?([\w.]+)\s*;', content):
            imports.append(match.group(1))
        return imports
    
    def _extract_extends(self, content: str) -> Optional[str]:
        """Extract extends clause."""
        match = re.search(r'extends\s+(\w+)', content)
        return match.group(1) if match else None
    
    def _extract_implements(self, content: str) -> List[str]:
        """Extract implements clause."""
        match = re.search(r'implements\s+([\w,\s]+?)(?:\{|$)', content)
        if match:
            return [i.strip() for i in match.group(1).split(',')]
        return []
    
    def _extract_fields(self, content: str) -> List[Dict[str, Any]]:
        """Extract field definitions."""
        fields = []
        pattern = r'(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?(\w+(?:<[\w,\s<>]+>)?)\s+(\w+)\s*(?:=|;)'
        
        for match in re.finditer(pattern, content):
            fields.append({
                "type": match.group(1),
                "name": match.group(2)
            })
        
        return fields
    
    def _extract_methods(self, content: str) -> List[Dict[str, Any]]:
        """Extract method definitions."""
        methods = []
        pattern = r'(?:@[\w]+(?:\([^)]*\))?\s*)*(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?(\w+(?:<[\w,\s<>]+>)?)\s+(\w+)\s*\(([^)]*)\)'
        
        for match in re.finditer(pattern, content):
            methods.append({
                "return_type": match.group(1),
                "name": match.group(2),
                "parameters": match.group(3)
            })
        
        return methods
    
    def _extract_class_annotations(self, content: str) -> List[str]:
        """Extract class-level annotations."""
        annotations = []
        pattern = r'@(\w+)(?:\([^)]*\))?'
        
        class_pos = re.search(r'(?:public\s+)?(?:abstract\s+)?(?:class|interface|enum)\s+\w+', content)
        if class_pos:
            before_class = content[:class_pos.start()]
            for match in re.finditer(pattern, before_class):
                annotations.append(match.group(1))
        
        return annotations
    
    def _extract_modifiers(self, content: str, position: int) -> List[str]:
        """Extract modifiers for a declaration."""
        modifiers = []
        before = content[max(0, position-50):position]
        
        modifier_keywords = ['public', 'private', 'protected', 'static', 'final', 'abstract']
        for keyword in modifier_keywords:
            if re.search(rf'\b{keyword}\b', before):
                modifiers.append(keyword)
        
        return modifiers
    
    def _extract_dependencies(self, imports: List[str], content: str) -> List[str]:
        """Extract class dependencies."""
        dependencies = set()
        
        for imp in imports:
            class_name = imp.split('.')[-1]
            if class_name != '*':
                dependencies.add(class_name)
        
        type_pattern = r'\b([A-Z][a-zA-Z0-9]*)\b'
        for match in re.finditer(type_pattern, content):
            dependencies.add(match.group(1))
        
        return list(dependencies)
    
    def _build_dependency_graph(self):
        """Build dependency graph between files."""
        for class_name, class_info in self._structure.classes.items():
            for dep_name in class_info.dependencies:
                if dep_name in self._class_index:
                    dep_class = self._class_index[dep_name]
                    dependency = Dependency(
                        source_file=class_info.file_path,
                        target_file=dep_class.file_path,
                        dependency_type="uses",
                        strength=1.0,
                        symbols=[dep_name]
                    )
                    self._structure.dependencies.append(dependency)
    
    def _build_package_structure(self):
        """Build package structure mapping."""
        for class_name, class_info in self._structure.classes.items():
            package = class_info.package or "default"
            if package not in self._structure.package_structure:
                self._structure.package_structure[package] = []
            self._structure.package_structure[package].append(class_name)
    
    async def analyze_dependencies(
        self,
        target_file: str
    ) -> List[Dependency]:
        """Analyze dependencies for a specific file.
        
        Args:
            target_file: Path to the target file
            
        Returns:
            List of dependencies
        """
        if not self._structure:
            await self.analyze_project()
        
        dependencies = []
        
        for dep in self._structure.dependencies:
            if dep.source_file == target_file or dep.target_file == target_file:
                dependencies.append(dep)
        
        return dependencies
    
    async def get_related_files(
        self,
        target_file: str,
        max_depth: int = 2
    ) -> List[str]:
        """Get files related to the target file.
        
        Args:
            target_file: Path to the target file
            max_depth: Maximum dependency depth
            
        Returns:
            List of related file paths
        """
        if not self._structure:
            await self.analyze_project()
        
        related = set()
        to_visit = [(target_file, 0)]
        visited = set()
        
        while to_visit:
            current_file, depth = to_visit.pop(0)
            
            if current_file in visited or depth > max_depth:
                continue
            
            visited.add(current_file)
            
            for dep in self._structure.dependencies:
                if dep.source_file == current_file:
                    related.add(dep.target_file)
                    if depth < max_depth:
                        to_visit.append((dep.target_file, depth + 1))
                elif dep.target_file == current_file:
                    related.add(dep.source_file)
                    if depth < max_depth:
                        to_visit.append((dep.source_file, depth + 1))
        
        return list(related)
    
    def get_class_info(self, class_name: str) -> Optional[ClassInfo]:
        """Get class information by name."""
        return self._class_index.get(class_name)
    
    def get_file_class(self, file_path: str) -> Optional[str]:
        """Get class name for a file."""
        return self._file_index.get(file_path)
    
    def get_structure(self) -> Optional[ProjectStructure]:
        """Get the analyzed project structure."""
        return self._structure


class MultiFileCoordinator:
    """Coordinates operations across multiple files.
    
    Features:
    - Multi-file editing
    - Cross-file refactoring
    - Dependency-aware updates
    """
    
    def __init__(self, project_analyzer: ProjectAnalyzer):
        self.analyzer = project_analyzer
    
    async def multi_file_edit(
        self,
        edits: List[FileEdit]
    ) -> MultiEditResult:
        """Apply edits to multiple files.
        
        Args:
            edits: List of file edits to apply
            
        Returns:
            MultiEditResult with edit outcomes
        """
        edited_files = []
        failed_files = []
        errors = []
        
        for edit in edits:
            try:
                file_path = self.analyzer.project_path / edit.file_path
                
                if edit.edit_type == "create":
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(edit.content)
                    edited_files.append(edit.file_path)
                
                elif edit.edit_type == "modify":
                    if not file_path.exists():
                        raise FileNotFoundError(f"File not found: {edit.file_path}")
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        original_content = f.read()
                    
                    modified_content = self._apply_edit(
                        original_content,
                        edit
                    )
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(modified_content)
                    
                    edited_files.append(edit.file_path)
                
                elif edit.edit_type == "delete":
                    if file_path.exists():
                        file_path.unlink()
                    edited_files.append(edit.file_path)
                
            except Exception as e:
                failed_files.append(edit.file_path)
                errors.append(e)
                logger.warning(f"[MultiFileCoordinator] Failed to edit {edit.file_path}: {e}")
        
        return MultiEditResult(
            success=len(failed_files) == 0,
            edited_files=edited_files,
            failed_files=failed_files,
            errors=errors
        )
    
    def _apply_edit(
        self,
        content: str,
        edit: FileEdit
    ) -> str:
        """Apply an edit to content."""
        if edit.location:
            lines = content.split('\n')
            start, end = edit.location
            
            if 0 <= start < len(lines):
                if end >= len(lines):
                    lines = lines[:start] + edit.content.split('\n')
                else:
                    lines = lines[:start] + edit.content.split('\n') + lines[end+1:]
                
                return '\n'.join(lines)
        
        return content
    
    async def propagate_changes(
        self,
        source_file: str,
        change_type: str,
        old_name: Optional[str] = None,
        new_name: Optional[str] = None
    ) -> MultiEditResult:
        """Propagate changes to dependent files.
        
        Args:
            source_file: File that was changed
            change_type: Type of change (rename, delete, etc.)
            old_name: Old name (for rename)
            new_name: New name (for rename)
            
        Returns:
            MultiEditResult with propagation outcomes
        """
        related_files = await self.analyzer.get_related_files(source_file)
        
        edits = []
        
        if change_type == "rename" and old_name and new_name:
            for related_file in related_files:
                edits.append(FileEdit(
                    file_path=related_file,
                    edit_type="modify",
                    content="",
                    metadata={
                        "operation": "rename",
                        "old_name": old_name,
                        "new_name": new_name
                    }
                ))
        
        return await self.multi_file_edit(edits)
    
    async def get_test_files_for_class(
        self,
        class_name: str
    ) -> List[str]:
        """Get test files for a class.
        
        Args:
            class_name: Name of the class
            
        Returns:
            List of test file paths
        """
        test_files = []
        
        test_name_patterns = [
            f"{class_name}Test.java",
            f"{class_name}Tests.java",
            f"Test{class_name}.java",
        ]
        
        for test_dir in self.analyzer._structure.test_dirs if self.analyzer._structure else []:
            test_path = Path(test_dir)
            for pattern in test_name_patterns:
                for test_file in test_path.rglob(pattern):
                    test_files.append(str(test_file.relative_to(self.analyzer.project_path)))
        
        return test_files


def create_project_analyzer(project_path: str) -> ProjectAnalyzer:
    """Create a ProjectAnalyzer instance.
    
    Args:
        project_path: Path to the project root
        
    Returns:
        Configured ProjectAnalyzer
    """
    return ProjectAnalyzer(project_path)
