"""Code editor with Aider-style Search/Replace diff format.

This module provides precise code editing capabilities using the Search/Replace
diff format popularized by Aider. It allows LLMs to make targeted edits to
code files without regenerating entire files.
"""

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

from tree_sitter import Parser, Language
import tree_sitter_java as tsjava


class EditStatus(Enum):
    """Status of an edit operation."""
    PENDING = auto()
    APPLIED = auto()
    FAILED = auto()
    VALIDATED = auto()


class EditError(Exception):
    """Exception raised when an edit operation fails."""
    pass


@dataclass
class EditOperation:
    """Represents a single Search/Replace edit operation."""
    search_text: str
    replace_text: str
    file_path: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    status: EditStatus = EditStatus.PENDING
    error_message: str = ""
    original_match: str = ""  # The actual text that was matched
    
    def __post_init__(self):
        """Normalize search/replace text."""
        # Remove leading/trailing whitespace for matching
        self.search_text = self._normalize_text(self.search_text)
        self.replace_text = self._normalize_text(self.replace_text)
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for consistent matching."""
        # Remove trailing whitespace from each line
        lines = [line.rstrip() for line in text.split('\n')]
        # Remove common leading indentation
        return dedent_text('\n'.join(lines))


@dataclass
class EditResult:
    """Result of applying edit operations."""
    success: bool
    original_content: str
    modified_content: str
    edits_applied: List[EditOperation] = field(default_factory=list)
    edits_failed: List[EditOperation] = field(default_factory=list)
    error_message: str = ""


def dedent_text(text: str) -> str:
    """Remove common leading indentation from text.
    
    Args:
        text: Input text with potential indentation
        
    Returns:
        Text with common leading indentation removed
    """
    lines = text.split('\n')
    
    # Find minimum indentation (excluding empty lines)
    min_indent = float('inf')
    for line in lines:
        if line.strip():  # Non-empty line
            indent = len(line) - len(line.lstrip())
            min_indent = min(min_indent, indent)
    
    if min_indent == float('inf'):
        return text
    
    # Remove common indentation
    dedented_lines = []
    for line in lines:
        if line.strip():
            dedented_lines.append(line[min_indent:])
        else:
            dedented_lines.append('')
    
    return '\n'.join(dedented_lines)


class DiffParser:
    """Parser for Search/Replace diff format."""
    
    # Pattern for matching SEARCH/REPLACE blocks
    DIFF_PATTERN = re.compile(
        r'<<<<<<< SEARCH\n'
        r'(.*?)'
        r'=======\n'
        r'(.*?)'
        r'>>>>>>> REPLACE',
        re.DOTALL
    )
    
    # Alternative pattern with file path
    DIFF_WITH_FILE_PATTERN = re.compile(
        r'###\s*(.+?)\n'
        r'<<<<<<< SEARCH\n'
        r'(.*?)'
        r'=======\n'
        r'(.*?)'
        r'>>>>>>> REPLACE',
        re.DOTALL
    )
    
    @classmethod
    def parse(cls, diff_text: str, default_file: Optional[str] = None) -> List[EditOperation]:
        """Parse diff text into edit operations.
        
        Args:
            diff_text: Text containing SEARCH/REPLACE blocks
            default_file: Default file path if not specified in diff
            
        Returns:
            List of EditOperation objects
        """
        edits = []
        
        # Try pattern with file path first
        for match in cls.DIFF_WITH_FILE_PATTERN.finditer(diff_text):
            file_path = match.group(1).strip()
            search_text = match.group(2)
            replace_text = match.group(3)
            
            edits.append(EditOperation(
                search_text=search_text,
                replace_text=replace_text,
                file_path=file_path
            ))
        
        # If no file-specific edits found, try generic pattern
        if not edits:
            for match in cls.DIFF_PATTERN.finditer(diff_text):
                search_text = match.group(1)
                replace_text = match.group(2)
                
                edits.append(EditOperation(
                    search_text=search_text,
                    replace_text=replace_text,
                    file_path=default_file
                ))
        
        return edits
    
    @classmethod
    def create_diff(
        cls,
        search_text: str,
        replace_text: str,
        file_path: Optional[str] = None
    ) -> str:
        """Create a diff string from search and replace text.
        
        Args:
            search_text: Text to search for
            replace_text: Text to replace with
            file_path: Optional file path
            
        Returns:
            Formatted diff string
        """
        lines = []
        
        if file_path:
            lines.append(f"### {file_path}")
        
        lines.extend([
            "<<<<<<< SEARCH",
            search_text.rstrip(),
            "=======",
            replace_text.rstrip(),
            ">>>>>>> REPLACE"
        ])
        
        return '\n'.join(lines)


class CodeEditor:
    """Editor for applying Search/Replace edits to code files.
    
    This class implements the Aider-style editing approach where LLMs
    generate SEARCH/REPLACE blocks that precisely identify and modify
    specific sections of code.
    """
    
    def __init__(self):
        """Initialize the code editor."""
        self.parser = Parser(Language(tsjava.language()))
        self.edit_history: List[EditResult] = []
    
    def apply_edits(
        self,
        content: str,
        edits: List[EditOperation],
        validate: bool = True
    ) -> EditResult:
        """Apply multiple edit operations to content.
        
        Args:
            content: Original file content
            edits: List of edit operations to apply
            validate: Whether to validate edits before applying
            
        Returns:
            EditResult with success status and modified content
        """
        if not edits:
            return EditResult(
                success=True,
                original_content=content,
                modified_content=content
            )
        
        modified_content = content
        applied_edits = []
        failed_edits = []
        
        # Sort edits by line number (descending) to avoid position shifts
        sorted_edits = sorted(
            edits,
            key=lambda e: (e.line_start or 0, e.line_end or 0),
            reverse=True
        )
        
        for edit in sorted_edits:
            try:
                result = self._apply_single_edit(modified_content, edit)
                if result.success:
                    modified_content = result.modified_content
                    edit.status = EditStatus.APPLIED
                    applied_edits.append(edit)
                else:
                    edit.status = EditStatus.FAILED
                    edit.error_message = result.error_message
                    failed_edits.append(edit)
            except Exception as e:
                edit.status = EditStatus.FAILED
                edit.error_message = str(e)
                failed_edits.append(edit)
        
        # Validate final result if requested
        if validate and modified_content != content:
            is_valid = self._validate_syntax(modified_content)
            if not is_valid:
                return EditResult(
                    success=False,
                    original_content=content,
                    modified_content=modified_content,
                    edits_applied=applied_edits,
                    edits_failed=failed_edits,
                    error_message="Syntax validation failed after edits"
                )
        
        success = len(failed_edits) == 0
        result = EditResult(
            success=success,
            original_content=content,
            modified_content=modified_content,
            edits_applied=applied_edits,
            edits_failed=failed_edits,
            error_message="; ".join(e.error_message for e in failed_edits) if failed_edits else ""
        )
        
        self.edit_history.append(result)
        return result
    
    def _apply_single_edit(
        self,
        content: str,
        edit: EditOperation
    ) -> EditResult:
        """Apply a single edit operation.
        
        Args:
            content: Current content
            edit: Edit operation to apply
            
        Returns:
            EditResult with success status
        """
        search_text = edit.search_text
        replace_text = edit.replace_text
        
        # Try exact match first
        if search_text in content:
            new_content = content.replace(search_text, replace_text, 1)
            edit.original_match = search_text
            return EditResult(
                success=True,
                original_content=content,
                modified_content=new_content
            )
        
        # Try with normalized whitespace
        normalized_content = self._normalize_for_matching(content)
        normalized_search = self._normalize_for_matching(search_text)
        
        if normalized_search in normalized_content:
            # Find position in original content
            pos = self._find_fuzzy_match(content, search_text)
            if pos >= 0:
                before = content[:pos]
                after = content[pos + len(search_text):]
                new_content = before + replace_text + after
                edit.original_match = search_text
                return EditResult(
                    success=True,
                    original_content=content,
                    modified_content=new_content
                )
        
        # Try line-by-line matching for partial matches
        match_result = self._find_line_match(content, search_text)
        if match_result:
            start_pos, end_pos, matched_text = match_result
            new_content = content[:start_pos] + replace_text + content[end_pos:]
            edit.original_match = matched_text
            return EditResult(
                success=True,
                original_content=content,
                modified_content=new_content
            )
        
        return EditResult(
            success=False,
            original_content=content,
            modified_content=content,
            error_message=f"Search text not found: {search_text[:50]}..."
        )
    
    def _normalize_for_matching(self, text: str) -> str:
        """Normalize text for fuzzy matching.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Remove extra whitespace and normalize line endings
        lines = text.split('\n')
        normalized_lines = []
        
        for line in lines:
            # Remove leading/trailing whitespace
            line = line.strip()
            # Collapse multiple spaces
            line = ' '.join(line.split())
            if line:
                normalized_lines.append(line)
        
        return '\n'.join(normalized_lines)
    
    def _find_fuzzy_match(self, content: str, search_text: str) -> int:
        """Find fuzzy match position in content.
        
        Args:
            content: Content to search in
            search_text: Text to search for
            
        Returns:
            Position of match or -1 if not found
        """
        content_lines = content.split('\n')
        search_lines = search_text.strip().split('\n')
        
        if not search_lines:
            return -1
        
        first_line = search_lines[0].strip()
        
        for i, line in enumerate(content_lines):
            if line.strip() == first_line:
                # Check if subsequent lines match
                match = True
                for j, search_line in enumerate(search_lines[1:], 1):
                    if i + j >= len(content_lines):
                        match = False
                        break
                    if content_lines[i + j].strip() != search_line.strip():
                        match = False
                        break
                
                if match:
                    # Calculate position
                    pos = 0
                    for k in range(i):
                        pos += len(content_lines[k]) + 1  # +1 for newline
                    return pos
        
        return -1
    
    def _find_line_match(
        self,
        content: str,
        search_text: str
    ) -> Optional[Tuple[int, int, str]]:
        """Find match based on line content ignoring whitespace.
        
        Args:
            content: Content to search in
            search_text: Text to search for
            
        Returns:
            Tuple of (start_pos, end_pos, matched_text) or None
        """
        content_lines = content.split('\n')
        search_lines = [line.strip() for line in search_text.strip().split('\n') if line.strip()]
        
        if not search_lines:
            return None
        
        for i in range(len(content_lines) - len(search_lines) + 1):
            match = True
            matched_lines = []
            
            for j, search_line in enumerate(search_lines):
                content_line = content_lines[i + j].strip()
                matched_lines.append(content_lines[i + j])
                
                if content_line != search_line:
                    match = False
                    break
            
            if match:
                # Calculate positions
                start_pos = sum(len(content_lines[k]) + 1 for k in range(i))
                end_pos = start_pos + sum(len(line) + 1 for line in matched_lines)
                matched_text = '\n'.join(matched_lines)
                return (start_pos, end_pos, matched_text)
        
        return None
    
    def _validate_syntax(self, content: str) -> bool:
        """Validate Java syntax using tree-sitter.
        
        Args:
            content: Java code to validate
            
        Returns:
            True if syntax is valid
        """
        try:
            tree = self.parser.parse(content.encode())
            root = tree.root_node
            
            # Check for syntax errors
            def has_errors(node) -> bool:
                if node.type == 'ERROR':
                    return True
                if node.is_missing:
                    return True
                return any(has_errors(child) for child in node.children)
            
            return not has_errors(root)
        except Exception:
            return False
    
    def apply_diff_to_file(
        self,
        file_path: str,
        diff_text: str,
        backup: bool = True
    ) -> EditResult:
        """Apply diff text directly to a file.
        
        Args:
            file_path: Path to the file
            diff_text: Diff text with SEARCH/REPLACE blocks
            backup: Whether to create a backup
            
        Returns:
            EditResult with success status
        """
        path = Path(file_path)
        
        if not path.exists():
            return EditResult(
                success=False,
                original_content="",
                modified_content="",
                error_message=f"File not found: {file_path}"
            )
        
        content = path.read_text(encoding='utf-8')
        
        # Create backup if requested
        if backup:
            backup_path = path.with_suffix(path.suffix + '.backup')
            backup_path.write_text(content, encoding='utf-8')
        
        # Parse and apply edits
        edits = DiffParser.parse(diff_text, file_path)
        result = self.apply_edits(content, edits)
        
        # Write result if successful
        if result.success:
            path.write_text(result.modified_content, encoding='utf-8')
        
        return result
    
    def get_edit_history(self) -> List[EditResult]:
        """Get history of all edit operations."""
        return self.edit_history.copy()
    
    def undo_last_edit(self, file_path: str) -> bool:
        """Undo the last edit to a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if undo was successful
        """
        path = Path(file_path)
        backup_path = path.with_suffix(path.suffix + '.backup')
        
        if backup_path.exists():
            content = backup_path.read_text(encoding='utf-8')
            path.write_text(content, encoding='utf-8')
            backup_path.unlink()
            return True
        
        return False


class TestCodeEditor(CodeEditor):
    """Specialized editor for test code with Java-specific validation."""
    
    def __init__(self):
        """Initialize test code editor."""
        super().__init__()
    
    def apply_test_fixes(
        self,
        test_code: str,
        failure_analysis: Dict[str, Any],
        diff_text: str
    ) -> EditResult:
        """Apply fixes to test code based on failure analysis.
        
        Args:
            test_code: Current test code
            failure_analysis: Failure analysis results
            diff_text: Diff text with fixes
            
        Returns:
            EditResult with success status
        """
        # Parse edits from diff
        edits = DiffParser.parse(diff_text)
        
        if not edits:
            return EditResult(
                success=False,
                original_content=test_code,
                modified_content=test_code,
                error_message="No valid edits found in diff text"
            )
        
        # Apply edits with validation
        result = self.apply_edits(test_code, edits, validate=True)
        
        # Additional validation for test code
        if result.success:
            is_valid_test = self._validate_test_structure(result.modified_content)
            if not is_valid_test:
                return EditResult(
                    success=False,
                    original_content=test_code,
                    modified_content=result.modified_content,
                    edits_applied=result.edits_applied,
                    edits_failed=result.edits_failed,
                    error_message="Modified code is not a valid test class"
                )
        
        return result
    
    def _validate_test_structure(self, code: str) -> bool:
        """Validate that code is a valid test class.
        
        Args:
            code: Java code to validate
            
        Returns:
            True if code appears to be a valid test class
        """
        # Check for basic test class indicators
        has_class = 'class' in code
        has_test_annotation = '@Test' in code
        has_imports = 'import' in code
        
        # Must have class declaration
        if not has_class:
            return False
        
        # Should have test annotation or imports
        if not (has_test_annotation or has_imports):
            return False
        
        # Check for balanced braces
        open_braces = code.count('{')
        close_braces = code.count('}')
        if open_braces != close_braces:
            return False
        
        # Check for balanced parentheses
        open_parens = code.count('(')
        close_parens = code.count(')')
        if open_parens != close_parens:
            return False
        
        return True
    
    def generate_import_edit(self, import_statement: str, test_code: str) -> Optional[EditOperation]:
        """Generate an edit to add an import statement.
        
        Args:
            import_statement: Import statement to add
            test_code: Current test code
            
        Returns:
            EditOperation or None if import already exists
        """
        # Check if import already exists
        if import_statement in test_code:
            return None
        
        # Find position to add import (after package or at top)
        lines = test_code.split('\n')
        insert_line = 0
        
        for i, line in enumerate(lines):
            if line.strip().startswith('package '):
                insert_line = i + 1
            elif line.strip().startswith('import '):
                insert_line = i + 1
        
        # Generate search/replace
        if insert_line < len(lines):
            search_text = lines[insert_line] if insert_line > 0 else ''
            replace_text = f"{import_statement}\n{search_text}"
            
            return EditOperation(
                search_text=search_text,
                replace_text=replace_text,
                line_start=insert_line,
                line_end=insert_line
            )
        
        return None


def create_edit_prompt(
    original_code: str,
    error_message: str,
    fix_instructions: str
) -> str:
    """Create a prompt for LLM to generate edit diff.
    
    Args:
        original_code: Original code that needs fixing
        error_message: Error or failure message
        fix_instructions: Instructions on how to fix
        
    Returns:
        Prompt string for LLM
    """
    return f"""You are an expert Java developer. Fix the following test code based on the error message.

## Error Message
{error_message}

## Fix Instructions
{fix_instructions}

## Current Test Code
```java
{original_code}
```

## Task
Provide the fix using the following Search/Replace format. Only modify the necessary parts:

<<<<<<< SEARCH
[exact code to find]
=======
[replacement code]
>>>>>>> REPLACE

If multiple changes are needed, provide multiple SEARCH/REPLACE blocks.
Be precise - the search text must match exactly (including whitespace).
"""