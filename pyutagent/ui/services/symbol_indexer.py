"""Symbol indexing service for @symbol references and smart autocomplete.

This module provides:
- SymbolIndexer: Parse and index code symbols (classes/methods/functions)
- Support for Python/Java/TypeScript/JavaScript/Go/Rust languages
- Incremental updates for changed files
- Fuzzy matching for symbol search
- Recent usage tracking for priority sorting

参考 Cursor 的 @-symbol 引用设计:
- https://www.cursor.com/blog/llm-chat-code-context
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from pyutagent.indexing.codebase_indexer import SymbolType, CodeSymbol

logger = logging.getLogger(__name__)


class Language(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVA = "java"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    GO = "go"
    RUST = "rust"
    UNKNOWN = "unknown"


@dataclass
class SymbolIndexEntry:
    """Entry in the symbol index."""
    id: str
    name: str
    full_name: str
    symbol_type: SymbolType
    language: Language
    file_path: str
    start_line: int
    end_line: int
    signature: str = ""
    docstring: str = ""
    modifiers: List[str] = field(default_factory=list)
    parameters: List[Tuple[str, str]] = field(default_factory=list)
    return_type: Optional[str] = None
    parent_name: Optional[str] = None
    content_hash: str = ""
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "full_name": self.full_name,
            "symbol_type": self.symbol_type.value,
            "language": self.language.value,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "signature": self.signature,
            "docstring": self.docstring,
            "modifiers": self.modifiers,
            "parameters": self.parameters,
            "return_type": self.return_type,
            "parent_name": self.parent_name,
            "content_hash": self.content_hash,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SymbolIndexEntry:
        return cls(
            id=data["id"],
            name=data["name"],
            full_name=data["full_name"],
            symbol_type=SymbolType(data["symbol_type"]),
            language=Language(data.get("language", "unknown")),
            file_path=data["file_path"],
            start_line=data["start_line"],
            end_line=data["end_line"],
            signature=data.get("signature", ""),
            docstring=data.get("docstring", ""),
            modifiers=data.get("modifiers", []),
            parameters=[tuple(p) for p in data.get("parameters", [])],
            return_type=data.get("return_type"),
            parent_name=data.get("parent_name"),
            content_hash=data.get("content_hash", ""),
            last_accessed=data.get("last_accessed", time.time()),
            access_count=data.get("access_count", 0),
        )

    def mark_accessed(self):
        """Mark this symbol as accessed."""
        self.last_accessed = time.time()
        self.access_count += 1

    @property
    def priority_score(self) -> float:
        """Calculate priority score based on recency and frequency."""
        time_factor = 1.0 / (1.0 + (time.time() - self.last_accessed) / 3600)
        frequency_factor = min(self.access_count / 10.0, 1.0)
        return time_factor * 0.6 + frequency_factor * 0.4


@dataclass
class SymbolIndexStats:
    """Statistics for the symbol index."""
    total_symbols: int = 0
    total_files: int = 0
    languages: Set[str] = field(default_factory=set)
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    index_size_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_symbols": self.total_symbols,
            "total_files": self.total_files,
            "languages": list(self.languages),
            "last_updated": self.last_updated,
            "index_size_mb": self.index_size_mb,
        }


class PythonSymbolParser:
    """Parse Python source code for symbols."""

    def parse(self, file_path: str, content: str) -> List[SymbolIndexEntry]:
        """Parse Python file and extract symbols."""
        symbols = []
        content_hash = hashlib.md5(content.encode()).hexdigest()

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.warning(f"[PythonSymbolParser] Syntax error in {file_path}: {e}")
            return symbols

        module_name = Path(file_path).stem

        # Track method IDs to avoid duplicates
        method_ids = set()

        # Process only top-level nodes
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                symbol = self._parse_class(node, file_path, content, module_name, content_hash)
                symbols.append(symbol)

                # Parse methods within class
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method = self._parse_method(
                            item, file_path, content, node.name, symbol.id, content_hash
                        )
                        method_ids.add(method.id)
                        symbols.append(method)

            elif isinstance(node, ast.FunctionDef):
                # Module-level function
                symbol = self._parse_function(node, file_path, content, module_name, content_hash)
                # Skip if this function was already added as a method
                if symbol.id not in method_ids:
                    symbols.append(symbol)

        return symbols

    def _parse_class(self, node: ast.ClassDef, file_path: str, content: str,
                     module_name: str, content_hash: str) -> SymbolIndexEntry:
        """Parse a class definition."""
        start_line = node.lineno
        end_line = node.end_lineno or start_line

        # Get docstring
        docstring = ast.get_docstring(node) or ""

        # Build signature
        bases = [self._get_name(base) for base in node.bases]
        signature = f"class {node.name}"
        if bases:
            signature += f"({', '.join(bases)})"

        return SymbolIndexEntry(
            id=f"{file_path}#class#{node.name}",
            name=node.name,
            full_name=f"{module_name}.{node.name}",
            symbol_type=SymbolType.CLASS,
            language=Language.PYTHON,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            signature=signature,
            docstring=docstring,
            content_hash=content_hash,
        )

    def _parse_method(self, node: ast.FunctionDef, file_path: str, content: str,
                      class_name: str, parent_id: str, content_hash: str) -> SymbolIndexEntry:
        """Parse a method definition."""
        start_line = node.lineno
        end_line = node.end_lineno or start_line

        # Determine if it's a constructor
        is_constructor = node.name == "__init__"
        symbol_type = SymbolType.CONSTRUCTOR if is_constructor else SymbolType.METHOD

        # Get docstring
        docstring = ast.get_docstring(node) or ""

        # Parse parameters
        parameters = []
        for arg in node.args.args:
            if arg.arg != "self":
                arg_type = self._get_annotation(arg.annotation)
                parameters.append((arg_type or "Any", arg.arg))

        # Get return type
        return_type = self._get_annotation(node.returns)

        # Build signature
        params_str = ", ".join(f"{t} {n}" for t, n in parameters)
        signature = f"def {node.name}({params_str})"
        if return_type:
            signature += f" -> {return_type}"

        return SymbolIndexEntry(
            id=f"{file_path}#method#{class_name}.{node.name}",
            name=node.name,
            full_name=f"{class_name}.{node.name}",
            symbol_type=symbol_type,
            language=Language.PYTHON,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            signature=signature,
            docstring=docstring,
            parameters=parameters,
            return_type=return_type,
            parent_name=class_name,
            content_hash=content_hash,
        )

    def _parse_function(self, node: ast.FunctionDef, file_path: str, content: str,
                        module_name: str, content_hash: str) -> SymbolIndexEntry:
        """Parse a module-level function."""
        start_line = node.lineno
        end_line = node.end_lineno or start_line

        # Get docstring
        docstring = ast.get_docstring(node) or ""

        # Parse parameters
        parameters = []
        for arg in node.args.args:
            arg_type = self._get_annotation(arg.annotation)
            parameters.append((arg_type or "Any", arg.arg))

        # Get return type
        return_type = self._get_annotation(node.returns)

        # Build signature
        params_str = ", ".join(f"{t} {n}" for t, n in parameters)
        signature = f"def {node.name}({params_str})"
        if return_type:
            signature += f" -> {return_type}"

        return SymbolIndexEntry(
            id=f"{file_path}#function#{node.name}",
            name=node.name,
            full_name=f"{module_name}.{node.name}",
            symbol_type=SymbolType.METHOD,
            language=Language.PYTHON,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            signature=signature,
            docstring=docstring,
            parameters=parameters,
            return_type=return_type,
            content_hash=content_hash,
        )

    def _get_name(self, node: ast.AST) -> str:
        """Get name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return str(node)

    def _get_annotation(self, node: Optional[ast.AST]) -> Optional[str]:
        """Get type annotation as string."""
        if node is None:
            return None
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Subscript):
            return self._get_name(node.value)
        return str(node)


class JavaScriptSymbolParser:
    """Parse JavaScript/TypeScript source code for symbols."""

    # Regex patterns for JS/TS
    CLASS_PATTERN = re.compile(
        r'(?:export\s+)?(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([^{]+))?',
        re.MULTILINE
    )
    FUNCTION_PATTERN = re.compile(
        r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)(?:\s*:\s*(\w+))?',
        re.MULTILINE
    )
    METHOD_PATTERN = re.compile(
        r'(?:async\s+)?(\w+)\s*\(([^)]*)\)(?:\s*:\s*(\w+))?\s*\{',
        re.MULTILINE
    )
    ARROW_FUNCTION_PATTERN = re.compile(
        r'(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(([^)]*)\)(?:\s*:\s*(\w+))?\s*=>',
        re.MULTILINE
    )
    INTERFACE_PATTERN = re.compile(
        r'(?:export\s+)?interface\s+(\w+)(?:\s+extends\s+([^{]+))?',
        re.MULTILINE
    )

    def parse(self, file_path: str, content: str, is_typescript: bool = False) -> List[SymbolIndexEntry]:
        """Parse JS/TS file and extract symbols."""
        symbols = []
        content_hash = hashlib.md5(content.encode()).hexdigest()
        language = Language.TYPESCRIPT if is_typescript else Language.JAVASCRIPT
        lines = content.split('\n')

        # Parse classes
        for match in self.CLASS_PATTERN.finditer(content):
            symbol = self._parse_class(match, file_path, lines, content_hash, language)
            symbols.append(symbol)

        # Parse interfaces (TypeScript only)
        if is_typescript:
            for match in self.INTERFACE_PATTERN.finditer(content):
                symbol = self._parse_interface(match, file_path, lines, content_hash)
                symbols.append(symbol)

        # Parse functions
        for match in self.FUNCTION_PATTERN.finditer(content):
            symbol = self._parse_function(match, file_path, lines, content_hash, language)
            symbols.append(symbol)

        # Parse arrow functions
        for match in self.ARROW_FUNCTION_PATTERN.finditer(content):
            symbol = self._parse_arrow_function(match, file_path, lines, content_hash, language)
            symbols.append(symbol)

        return symbols

    def _get_line_number(self, content: str, pos: int) -> int:
        """Get line number for a position in content."""
        return content[:pos].count('\n') + 1

    def _parse_class(self, match: re.Match, file_path: str, lines: List[str],
                     content_hash: str, language: Language) -> SymbolIndexEntry:
        """Parse a class definition."""
        class_name = match.group(1)
        parent_class = match.group(2)
        start_line = self._get_line_number('\n'.join(lines), match.start())

        # Find end line (simplified - find closing brace)
        end_line = start_line
        brace_count = 0
        for i in range(start_line - 1, len(lines)):
            brace_count += lines[i].count('{') - lines[i].count('}')
            if brace_count > 0:
                end_line = i + 1
            if brace_count == 0 and i > start_line - 1:
                break

        # Build signature
        signature = f"class {class_name}"
        if parent_class:
            signature += f" extends {parent_class}"

        return SymbolIndexEntry(
            id=f"{file_path}#class#{class_name}",
            name=class_name,
            full_name=class_name,
            symbol_type=SymbolType.CLASS,
            language=language,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            signature=signature,
            content_hash=content_hash,
        )

    def _parse_interface(self, match: re.Match, file_path: str, lines: List[str],
                         content_hash: str) -> SymbolIndexEntry:
        """Parse a TypeScript interface."""
        interface_name = match.group(1)
        start_line = self._get_line_number('\n'.join(lines), match.start())

        # Find end line
        end_line = start_line
        brace_count = 0
        for i in range(start_line - 1, len(lines)):
            brace_count += lines[i].count('{') - lines[i].count('}')
            if brace_count > 0:
                end_line = i + 1
            if brace_count == 0 and i > start_line - 1:
                break

        return SymbolIndexEntry(
            id=f"{file_path}#interface#{interface_name}",
            name=interface_name,
            full_name=interface_name,
            symbol_type=SymbolType.INTERFACE,
            language=Language.TYPESCRIPT,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            signature=f"interface {interface_name}",
            content_hash=content_hash,
        )

    def _parse_function(self, match: re.Match, file_path: str, lines: List[str],
                        content_hash: str, language: Language) -> SymbolIndexEntry:
        """Parse a function definition."""
        func_name = match.group(1)
        params = match.group(2)
        return_type = match.group(3)
        start_line = self._get_line_number('\n'.join(lines), match.start())

        # Parse parameters
        parameters = []
        if params:
            for param in params.split(','):
                param = param.strip()
                if ':' in param:
                    parts = param.split(':', 1)
                    parameters.append((parts[1].strip(), parts[0].strip()))
                else:
                    parameters.append(("any", param))

        # Build signature
        params_str = ", ".join(f"{n}: {t}" for t, n in parameters)
        signature = f"function {func_name}({params_str})"
        if return_type:
            signature += f": {return_type}"

        return SymbolIndexEntry(
            id=f"{file_path}#function#{func_name}",
            name=func_name,
            full_name=func_name,
            symbol_type=SymbolType.METHOD,
            language=language,
            file_path=file_path,
            start_line=start_line,
            end_line=start_line,
            signature=signature,
            parameters=parameters,
            return_type=return_type,
            content_hash=content_hash,
        )

    def _parse_arrow_function(self, match: re.Match, file_path: str, lines: List[str],
                              content_hash: str, language: Language) -> SymbolIndexEntry:
        """Parse an arrow function."""
        func_name = match.group(1)
        params = match.group(2)
        return_type = match.group(3)
        start_line = self._get_line_number('\n'.join(lines), match.start())

        # Parse parameters
        parameters = []
        if params:
            for param in params.split(','):
                param = param.strip()
                if ':' in param:
                    parts = param.split(':', 1)
                    parameters.append((parts[1].strip(), parts[0].strip()))
                else:
                    parameters.append(("any", param))

        # Build signature
        params_str = ", ".join(f"{n}: {t}" for t, n in parameters)
        signature = f"const {func_name} = ({params_str})"
        if return_type:
            signature += f": {return_type}"
        signature += " => {...}"

        return SymbolIndexEntry(
            id=f"{file_path}#function#{func_name}",
            name=func_name,
            full_name=func_name,
            symbol_type=SymbolType.METHOD,
            language=language,
            file_path=file_path,
            start_line=start_line,
            end_line=start_line,
            signature=signature,
            parameters=parameters,
            return_type=return_type,
            content_hash=content_hash,
        )


class GoSymbolParser:
    """Parse Go source code for symbols."""

    # Regex patterns for Go
    TYPE_PATTERN = re.compile(r'type\s+(\w+)\s+(struct|interface)\s*\{', re.MULTILINE)
    FUNC_PATTERN = re.compile(r'func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(([^)]*)\)(?:\s*\([^)]*\))?(?:\s*(\w+))?\s*\{', re.MULTILINE)

    def parse(self, file_path: str, content: str) -> List[SymbolIndexEntry]:
        """Parse Go file and extract symbols."""
        symbols = []
        content_hash = hashlib.md5(content.encode()).hexdigest()
        lines = content.split('\n')

        # Parse types (structs and interfaces)
        for match in self.TYPE_PATTERN.finditer(content):
            symbol = self._parse_type(match, file_path, lines, content_hash)
            symbols.append(symbol)

        # Parse functions
        for match in self.FUNC_PATTERN.finditer(content):
            symbol = self._parse_function(match, file_path, lines, content_hash)
            symbols.append(symbol)

        return symbols

    def _get_line_number(self, content: str, pos: int) -> int:
        """Get line number for a position in content."""
        return content[:pos].count('\n') + 1

    def _parse_type(self, match: re.Match, file_path: str, lines: List[str],
                    content_hash: str) -> SymbolIndexEntry:
        """Parse a type definition."""
        type_name = match.group(1)
        type_kind = match.group(2)
        start_line = self._get_line_number('\n'.join(lines), match.start())

        # Find end line
        end_line = start_line
        brace_count = 0
        for i in range(start_line - 1, len(lines)):
            brace_count += lines[i].count('{') - lines[i].count('}')
            if brace_count > 0:
                end_line = i + 1
            if brace_count == 0 and i > start_line - 1:
                break

        symbol_type = SymbolType.INTERFACE if type_kind == "interface" else SymbolType.CLASS

        return SymbolIndexEntry(
            id=f"{file_path}#type#{type_name}",
            name=type_name,
            full_name=type_name,
            symbol_type=symbol_type,
            language=Language.GO,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            signature=f"type {type_name} {type_kind}",
            content_hash=content_hash,
        )

    def _parse_function(self, match: re.Match, file_path: str, lines: List[str],
                        content_hash: str) -> SymbolIndexEntry:
        """Parse a function definition."""
        func_name = match.group(1)
        params = match.group(2)
        return_type = match.group(3)
        start_line = self._get_line_number('\n'.join(lines), match.start())

        # Parse parameters
        parameters = []
        if params:
            for param in params.split(','):
                param = param.strip()
                parts = param.split()
                if len(parts) >= 2:
                    param_name = parts[0]
                    param_type = ' '.join(parts[1:])
                    parameters.append((param_type, param_name))

        # Build signature
        params_str = ", ".join(f"{n} {t}" for t, n in parameters)
        signature = f"func {func_name}({params_str})"
        if return_type:
            signature += f" {return_type}"

        return SymbolIndexEntry(
            id=f"{file_path}#func#{func_name}",
            name=func_name,
            full_name=func_name,
            symbol_type=SymbolType.METHOD,
            language=Language.GO,
            file_path=file_path,
            start_line=start_line,
            end_line=start_line,
            signature=signature,
            parameters=parameters,
            return_type=return_type,
            content_hash=content_hash,
        )


class RustSymbolParser:
    """Parse Rust source code for symbols."""

    # Regex patterns for Rust
    STRUCT_PATTERN = re.compile(r'(?:pub\s+)?struct\s+(\w+)(?:<[^>]+>)?(?:\s*\([^)]*\)|\s*\{)?', re.MULTILINE)
    ENUM_PATTERN = re.compile(r'(?:pub\s+)?enum\s+(\w+)(?:<[^>]+>)?', re.MULTILINE)
    TRAIT_PATTERN = re.compile(r'(?:pub\s+)?trait\s+(\w+)(?:<[^>]+>)?', re.MULTILINE)
    IMPL_PATTERN = re.compile(r'impl(?:<[^>]+>)?\s+(?:\w+\s+for\s+)?(\w+)', re.MULTILINE)
    FN_PATTERN = re.compile(r'(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*([^\{]+))?', re.MULTILINE)

    def parse(self, file_path: str, content: str) -> List[SymbolIndexEntry]:
        """Parse Rust file and extract symbols."""
        symbols = []
        content_hash = hashlib.md5(content.encode()).hexdigest()
        lines = content.split('\n')

        # Parse structs
        for match in self.STRUCT_PATTERN.finditer(content):
            symbol = self._parse_struct(match, file_path, lines, content_hash)
            symbols.append(symbol)

        # Parse enums
        for match in self.ENUM_PATTERN.finditer(content):
            symbol = self._parse_enum(match, file_path, lines, content_hash)
            symbols.append(symbol)

        # Parse traits
        for match in self.TRAIT_PATTERN.finditer(content):
            symbol = self._parse_trait(match, file_path, lines, content_hash)
            symbols.append(symbol)

        # Parse functions
        for match in self.FN_PATTERN.finditer(content):
            symbol = self._parse_function(match, file_path, lines, content_hash)
            symbols.append(symbol)

        return symbols

    def _get_line_number(self, content: str, pos: int) -> int:
        """Get line number for a position in content."""
        return content[:pos].count('\n') + 1

    def _parse_struct(self, match: re.Match, file_path: str, lines: List[str],
                      content_hash: str) -> SymbolIndexEntry:
        """Parse a struct definition."""
        struct_name = match.group(1)
        start_line = self._get_line_number('\n'.join(lines), match.start())

        return SymbolIndexEntry(
            id=f"{file_path}#struct#{struct_name}",
            name=struct_name,
            full_name=struct_name,
            symbol_type=SymbolType.CLASS,
            language=Language.RUST,
            file_path=file_path,
            start_line=start_line,
            end_line=start_line,
            signature=f"struct {struct_name}",
            content_hash=content_hash,
        )

    def _parse_enum(self, match: re.Match, file_path: str, lines: List[str],
                    content_hash: str) -> SymbolIndexEntry:
        """Parse an enum definition."""
        enum_name = match.group(1)
        start_line = self._get_line_number('\n'.join(lines), match.start())

        return SymbolIndexEntry(
            id=f"{file_path}#enum#{enum_name}",
            name=enum_name,
            full_name=enum_name,
            symbol_type=SymbolType.ENUM,
            language=Language.RUST,
            file_path=file_path,
            start_line=start_line,
            end_line=start_line,
            signature=f"enum {enum_name}",
            content_hash=content_hash,
        )

    def _parse_trait(self, match: re.Match, file_path: str, lines: List[str],
                     content_hash: str) -> SymbolIndexEntry:
        """Parse a trait definition."""
        trait_name = match.group(1)
        start_line = self._get_line_number('\n'.join(lines), match.start())

        return SymbolIndexEntry(
            id=f"{file_path}#trait#{trait_name}",
            name=trait_name,
            full_name=trait_name,
            symbol_type=SymbolType.INTERFACE,
            language=Language.RUST,
            file_path=file_path,
            start_line=start_line,
            end_line=start_line,
            signature=f"trait {trait_name}",
            content_hash=content_hash,
        )

    def _parse_function(self, match: re.Match, file_path: str, lines: List[str],
                        content_hash: str) -> SymbolIndexEntry:
        """Parse a function definition."""
        func_name = match.group(1)
        params = match.group(2)
        return_type = match.group(3)
        start_line = self._get_line_number('\n'.join(lines), match.start())

        # Parse parameters
        parameters = []
        if params:
            for param in params.split(','):
                param = param.strip()
                if ':' in param:
                    parts = param.rsplit(':', 1)
                    parameters.append((parts[1].strip(), parts[0].strip()))

        # Build signature
        params_str = ", ".join(f"{n}: {t}" for t, n in parameters)
        signature = f"fn {func_name}({params_str})"
        if return_type:
            signature += f" -> {return_type.strip()}"

        return SymbolIndexEntry(
            id=f"{file_path}#fn#{func_name}",
            name=func_name,
            full_name=func_name,
            symbol_type=SymbolType.METHOD,
            language=Language.RUST,
            file_path=file_path,
            start_line=start_line,
            end_line=start_line,
            signature=signature,
            parameters=parameters,
            return_type=return_type.strip() if return_type else None,
            content_hash=content_hash,
        )


class SymbolIndexer:
    """Service for indexing code symbols with support for multiple languages.

    Features:
    - Parse and index symbols from Python/Java/TypeScript/Go/Rust
    - Incremental updates for changed files
    - Fuzzy matching for symbol search
    - Recent usage tracking for priority sorting
    - Thread-safe operations
    """

    # File extensions to language mapping
    EXTENSION_MAP = {
        '.py': Language.PYTHON,
        '.java': Language.JAVA,
        '.ts': Language.TYPESCRIPT,
        '.tsx': Language.TYPESCRIPT,
        '.js': Language.JAVASCRIPT,
        '.jsx': Language.JAVASCRIPT,
        '.go': Language.GO,
        '.rs': Language.RUST,
    }

    def __init__(self, project_path: str, index_dir: str = ".pyutagent/symbol_index"):
        """Initialize the symbol indexer.

        Args:
            project_path: Path to the project root
            index_dir: Directory for storing index data
        """
        self.project_path = Path(project_path).resolve()
        self.index_dir = self.project_path / index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Initialize parsers
        self._parsers: Dict[Language, Any] = {
            Language.PYTHON: PythonSymbolParser(),
            Language.TYPESCRIPT: JavaScriptSymbolParser(),
            Language.JAVASCRIPT: JavaScriptSymbolParser(),
            Language.GO: GoSymbolParser(),
            Language.RUST: RustSymbolParser(),
        }

        # Index storage
        self._symbols: Dict[str, SymbolIndexEntry] = {}
        self._file_symbols: Dict[str, Set[str]] = {}
        self._name_index: Dict[str, Set[str]] = {}
        self._type_index: Dict[SymbolType, Set[str]] = {}
        self._file_hashes: Dict[str, str] = {}

        # Recent usage tracking
        self._recent_symbols: List[str] = []
        self._max_recent = 50

        # Thread safety
        self._lock = threading.RLock()

        # Load existing index
        self._load_index()

        logger.info(f"[SymbolIndexer] Initialized for {project_path}")

    def _load_index(self) -> None:
        """Load index from disk."""
        index_file = self.index_dir / "symbol_index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Load symbols
                for symbol_data in data.get("symbols", []):
                    symbol = SymbolIndexEntry.from_dict(symbol_data)
                    self._symbols[symbol.id] = symbol
                    self._index_symbol(symbol)

                # Load file hashes
                self._file_hashes = data.get("file_hashes", {})

                # Load recent symbols
                self._recent_symbols = data.get("recent_symbols", [])

                logger.info(f"[SymbolIndexer] Loaded {len(self._symbols)} symbols")
            except Exception as e:
                logger.warning(f"[SymbolIndexer] Failed to load index: {e}")

    def _save_index(self) -> None:
        """Save index to disk."""
        try:
            data = {
                "symbols": [s.to_dict() for s in self._symbols.values()],
                "file_hashes": self._file_hashes,
                "recent_symbols": self._recent_symbols,
                "saved_at": datetime.now().isoformat(),
            }

            index_file = self.index_dir / "symbol_index.json"
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"[SymbolIndexer] Failed to save index: {e}")

    def _index_symbol(self, symbol: SymbolIndexEntry) -> None:
        """Add symbol to indices."""
        # File index
        if symbol.file_path not in self._file_symbols:
            self._file_symbols[symbol.file_path] = set()
        self._file_symbols[symbol.file_path].add(symbol.id)

        # Name index
        if symbol.name not in self._name_index:
            self._name_index[symbol.name] = set()
        self._name_index[symbol.name].add(symbol.id)

        # Type index
        if symbol.symbol_type not in self._type_index:
            self._type_index[symbol.symbol_type] = set()
        self._type_index[symbol.symbol_type].add(symbol.id)

    def _unindex_symbol(self, symbol_id: str) -> None:
        """Remove symbol from indices."""
        symbol = self._symbols.get(symbol_id)
        if not symbol:
            return

        # Remove from file index
        if symbol.file_path in self._file_symbols:
            self._file_symbols[symbol.file_path].discard(symbol_id)

        # Remove from name index
        if symbol.name in self._name_index:
            self._name_index[symbol.name].discard(symbol_id)

        # Remove from type index
        if symbol.symbol_type in self._type_index:
            self._type_index[symbol.symbol_type].discard(symbol_id)

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute MD5 hash of file content."""
        try:
            content = file_path.read_bytes()
            return hashlib.md5(content).hexdigest()
        except Exception:
            return ""

    def _get_language(self, file_path: Path) -> Language:
        """Determine language from file extension."""
        ext = file_path.suffix.lower()
        return self.EXTENSION_MAP.get(ext, Language.UNKNOWN)

    def _parse_file(self, file_path: Path) -> List[SymbolIndexEntry]:
        """Parse a file and extract symbols."""
        language = self._get_language(file_path)

        if language == Language.UNKNOWN:
            return []

        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            content_hash = hashlib.md5(content.encode()).hexdigest()

            parser = self._parsers.get(language)
            if not parser:
                return []

            if language == Language.TYPESCRIPT:
                symbols = parser.parse(str(file_path), content, is_typescript=True)
            elif language == Language.JAVASCRIPT:
                symbols = parser.parse(str(file_path), content, is_typescript=False)
            else:
                symbols = parser.parse(str(file_path), content)

            # Update content hash
            for symbol in symbols:
                symbol.content_hash = content_hash

            return symbols
        except Exception as e:
            logger.warning(f"[SymbolIndexer] Failed to parse {file_path}: {e}")
            return []

    def index_file(self, file_path: Union[str, Path]) -> bool:
        """Index a single file.

        Args:
            file_path: Path to the file

        Returns:
            True if indexed successfully
        """
        path = Path(file_path)
        if not path.exists():
            return False

        # Check if file has changed
        current_hash = self._compute_file_hash(path)
        str_path = str(path)

        with self._lock:
            if str_path in self._file_hashes and self._file_hashes[str_path] == current_hash:
                return True  # No change

            # Remove old symbols for this file
            if str_path in self._file_symbols:
                for symbol_id in list(self._file_symbols[str_path]):
                    self._unindex_symbol(symbol_id)
                    if symbol_id in self._symbols:
                        del self._symbols[symbol_id]

            # Parse and index new symbols
            symbols = self._parse_file(path)
            for symbol in symbols:
                self._symbols[symbol.id] = symbol
                self._index_symbol(symbol)

            # Update file hash
            self._file_hashes[str_path] = current_hash

        logger.debug(f"[SymbolIndexer] Indexed {path}: {len(symbols)} symbols")
        return True

    def index_project(
        self,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> SymbolIndexStats:
        """Index the entire project.

        Args:
            include_patterns: Glob patterns for files to include
            exclude_patterns: Patterns for files to exclude
            progress_callback: Callback(current, total)

        Returns:
            Index statistics
        """
        if include_patterns is None:
            include_patterns = ["**/*.py", "**/*.java", "**/*.ts", "**/*.js", "**/*.go", "**/*.rs"]

        if exclude_patterns is None:
            exclude_patterns = [
                "**/node_modules/**", "**/__pycache__/**", "**/target/**",
                "**/build/**", "**/dist/**", "**/.git/**", "**/venv/**",
                "**/.venv/**", "**/env/**", "**/.env/**"
            ]

        # Collect files to index
        files_to_index = []
        for pattern in include_patterns:
            for file_path in self.project_path.glob(pattern):
                # Check exclude patterns
                if any(file_path.match(ex) for ex in exclude_patterns):
                    continue
                if any(part.startswith(".") for part in file_path.parts):
                    continue
                files_to_index.append(file_path)

        # Index files
        total = len(files_to_index)
        indexed = 0
        languages: Set[str] = set()

        for i, file_path in enumerate(files_to_index):
            if self.index_file(file_path):
                indexed += 1
                lang = self._get_language(file_path)
                if lang != Language.UNKNOWN:
                    languages.add(lang.value)

            if progress_callback:
                progress_callback(i + 1, total)

        # Save index
        self._save_index()

        # Calculate stats
        index_file = self.index_dir / "symbol_index.json"
        index_size_mb = index_file.stat().st_size / (1024 * 1024) if index_file.exists() else 0

        stats = SymbolIndexStats(
            total_symbols=len(self._symbols),
            total_files=indexed,
            languages=languages,
            last_updated=datetime.now().isoformat(),
            index_size_mb=index_size_mb,
        )

        logger.info(f"[SymbolIndexer] Project indexed: {stats.total_symbols} symbols in {stats.total_files} files")
        return stats

    def search(
        self,
        query: str,
        symbol_type: Optional[SymbolType] = None,
        language: Optional[Language] = None,
        limit: int = 20,
        fuzzy: bool = True,
    ) -> List[SymbolIndexEntry]:
        """Search for symbols.

        Args:
            query: Search query
            symbol_type: Filter by symbol type
            language: Filter by language
            limit: Maximum results
            fuzzy: Enable fuzzy matching

        Returns:
            List of matching symbols sorted by relevance
        """
        with self._lock:
            results = []
            query_lower = query.lower()

            # Get candidate symbols
            if symbol_type:
                candidate_ids = self._type_index.get(symbol_type, set())
            else:
                candidate_ids = set(self._symbols.keys())

            for symbol_id in candidate_ids:
                symbol = self._symbols.get(symbol_id)
                if not symbol:
                    continue

                # Language filter
                if language and symbol.language != language:
                    continue

                # Match score
                score = self._calculate_match_score(symbol, query_lower, fuzzy)
                if score > 0:
                    results.append((symbol, score))

            # Sort by score (descending) and priority
            results.sort(key=lambda x: (x[1] + x[0].priority_score), reverse=True)

            return [r[0] for r in results[:limit]]

    def _calculate_match_score(self, symbol: SymbolIndexEntry, query: str, fuzzy: bool) -> float:
        """Calculate match score for a symbol."""
        score = 0.0

        # Exact name match
        if query == symbol.name.lower():
            score += 100.0
        elif query in symbol.name.lower():
            score += 50.0

        # Full name match
        if query in symbol.full_name.lower():
            score += 30.0

        # Fuzzy matching
        if fuzzy and len(query) >= 2:
            # Check if all characters in query appear in order in name
            if self._fuzzy_match(query, symbol.name.lower()):
                score += 20.0

        # Signature match
        if symbol.signature and query in symbol.signature.lower():
            score += 10.0

        return score

    def _fuzzy_match(self, query: str, target: str) -> bool:
        """Check if query fuzzy matches target."""
        query_idx = 0
        target_idx = 0

        while query_idx < len(query) and target_idx < len(target):
            if query[query_idx] == target[target_idx]:
                query_idx += 1
            target_idx += 1

        return query_idx == len(query)

    def get_symbol(self, symbol_id: str) -> Optional[SymbolIndexEntry]:
        """Get a symbol by ID."""
        with self._lock:
            symbol = self._symbols.get(symbol_id)
            if symbol:
                symbol.mark_accessed()
                self._update_recent_symbols(symbol_id)
            return symbol

    def get_symbol_by_name(self, name: str) -> Optional[SymbolIndexEntry]:
        """Get a symbol by name."""
        with self._lock:
            symbol_ids = self._name_index.get(name, set())
            if symbol_ids:
                # Return most recently accessed
                symbol_id = max(symbol_ids, key=lambda sid: self._symbols.get(sid, SymbolIndexEntry("", "", "", SymbolType.CLASS, Language.UNKNOWN, "", 0, 0)).last_accessed)
                return self.get_symbol(symbol_id)
            return None

    def _update_recent_symbols(self, symbol_id: str) -> None:
        """Update recent symbols list."""
        if symbol_id in self._recent_symbols:
            self._recent_symbols.remove(symbol_id)
        self._recent_symbols.insert(0, symbol_id)
        self._recent_symbols = self._recent_symbols[:self._max_recent]

    def get_recent_symbols(self, limit: int = 10) -> List[SymbolIndexEntry]:
        """Get recently accessed symbols."""
        with self._lock:
            result = []
            for symbol_id in self._recent_symbols[:limit]:
                symbol = self._symbols.get(symbol_id)
                if symbol:
                    result.append(symbol)
            return result

    def get_symbols_by_type(self, symbol_type: SymbolType, limit: int = 50) -> List[SymbolIndexEntry]:
        """Get symbols by type."""
        with self._lock:
            symbol_ids = list(self._type_index.get(symbol_type, set()))[:limit]
            return [self._symbols[sid] for sid in symbol_ids if sid in self._symbols]

    def get_file_symbols(self, file_path: str) -> List[SymbolIndexEntry]:
        """Get all symbols in a file."""
        with self._lock:
            symbol_ids = self._file_symbols.get(file_path, set())
            return [self._symbols[sid] for sid in symbol_ids if sid in self._symbols]

    def remove_file(self, file_path: str) -> None:
        """Remove all symbols for a file."""
        with self._lock:
            if file_path in self._file_symbols:
                for symbol_id in list(self._file_symbols[file_path]):
                    self._unindex_symbol(symbol_id)
                    if symbol_id in self._symbols:
                        del self._symbols[symbol_id]
                del self._file_symbols[file_path]

            if file_path in self._file_hashes:
                del self._file_hashes[file_path]

    def clear(self) -> None:
        """Clear the entire index."""
        with self._lock:
            self._symbols.clear()
            self._file_symbols.clear()
            self._name_index.clear()
            self._type_index.clear()
            self._file_hashes.clear()
            self._recent_symbols.clear()

        # Remove index file
        index_file = self.index_dir / "symbol_index.json"
        if index_file.exists():
            index_file.unlink()

        logger.info("[SymbolIndexer] Index cleared")

    def get_stats(self) -> SymbolIndexStats:
        """Get index statistics."""
        with self._lock:
            languages = set(s.language.value for s in self._symbols.values())

            index_file = self.index_dir / "symbol_index.json"
            index_size_mb = index_file.stat().st_size / (1024 * 1024) if index_file.exists() else 0

            return SymbolIndexStats(
                total_symbols=len(self._symbols),
                total_files=len(self._file_symbols),
                languages=languages,
                last_updated=datetime.now().isoformat(),
                index_size_mb=index_size_mb,
            )

    def refresh(self) -> SymbolIndexStats:
        """Refresh the index by checking for changed files."""
        with self._lock:
            changed_files = []

            for file_path_str in list(self._file_hashes.keys()):
                file_path = Path(file_path_str)
                if file_path.exists():
                    current_hash = self._compute_file_hash(file_path)
                    if current_hash != self._file_hashes.get(file_path_str):
                        changed_files.append(file_path)
                else:
                    # File was deleted
                    self.remove_file(file_path_str)

        # Re-index changed files
        for file_path in changed_files:
            self.index_file(file_path)

        self._save_index()
        return self.get_stats()
