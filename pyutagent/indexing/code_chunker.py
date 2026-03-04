"""Code chunking module for intelligent code splitting.

This module provides code chunking capabilities for indexing,
splitting code into meaningful chunks (functions, classes, files)
for efficient retrieval.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


class ChunkType(Enum):
    """Types of code chunks."""
    FILE = "file"
    CLASS = "class"
    METHOD = "method"
    FUNCTION = "function"
    INTERFACE = "interface"
    ENUM = "enum"
    IMPORT = "import"
    COMMENT = "comment"
    UNKNOWN = "unknown"


class ChunkStrategy(Enum):
    """Strategies for chunking code."""
    BY_FILE = "by_file"
    BY_CLASS = "by_class"
    BY_METHOD = "by_method"
    BY_LINES = "by_lines"
    BY_TOKENS = "by_tokens"
    HYBRID = "hybrid"


@dataclass
class CodeChunk:
    """A chunk of code for indexing."""
    id: str
    content: str
    chunk_type: ChunkType
    file_path: str
    start_line: int
    end_line: int
    name: Optional[str] = None
    parent: Optional[str] = None
    language: str = "java"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "chunk_type": self.chunk_type.value,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "name": self.name,
            "parent": self.parent,
            "language": self.language,
            "metadata": self.metadata,
        }
    
    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1
    
    @property
    def token_estimate(self) -> int:
        return len(self.content.split())


@dataclass
class ChunkingConfig:
    """Configuration for code chunking."""
    strategy: ChunkStrategy = ChunkStrategy.HYBRID
    max_chunk_lines: int = 100
    max_chunk_tokens: int = 2000
    min_chunk_lines: int = 5
    overlap_lines: int = 5
    include_imports: bool = True
    include_comments: bool = False
    split_nested_classes: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "max_chunk_lines": self.max_chunk_lines,
            "max_chunk_tokens": self.max_chunk_tokens,
            "min_chunk_lines": self.min_chunk_lines,
            "overlap_lines": self.overlap_lines,
            "include_imports": self.include_imports,
            "include_comments": self.include_comments,
            "split_nested_classes": self.split_nested_classes,
        }


class CodeChunker:
    """Intelligent code chunker for various programming languages."""
    
    LANGUAGE_PATTERNS: Dict[str, Dict[str, Any]] = {
        "java": {
            "class_pattern": r'(?:public|private|protected)?\s*(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w,\s]+)?\s*\{',
            "interface_pattern": r'(?:public\s+)?interface\s+(\w+)(?:\s+extends\s+[\w,\s]+)?\s*\{',
            "enum_pattern": r'(?:public\s+)?enum\s+(\w+)(?:\s+implements\s+[\w,\s]+)?\s*\{',
            "method_pattern": r'(?:public|private|protected|static|\s)+[\w<>\[\],\s]+\s+(\w+)\s*\([^)]*\)(?:\s*throws\s+[\w,\s]+)?\s*\{',
            "import_pattern": r'import\s+[\w.]+;',
            "comment_pattern": r'(?:/\*[\s\S]*?\*/|//[^\n]*)',
            "extensions": [".java"],
        },
        "python": {
            "class_pattern": r'class\s+(\w+)(?:\([^)]*\))?\s*:',
            "function_pattern": r'def\s+(\w+)\s*\([^)]*\)\s*(?:->\s*[\w\[\],\s]+)?:',
            "import_pattern": r'(?:from\s+[\w.]+\s+)?import\s+[\w.,\s]+',
            "comment_pattern": r'(?:#[^\n]*|"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')',
            "extensions": [".py"],
        },
        "typescript": {
            "class_pattern": r'(?:export\s+)?(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w,\s]+)?\s*\{',
            "interface_pattern": r'(?:export\s+)?interface\s+(\w+)(?:\s+extends\s+[\w,\s]+)?\s*\{',
            "function_pattern": r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\([^)]*\)(?:\s*:\s*[\w\[\],\s]+)?\s*\{',
            "method_pattern": r'(?:public|private|protected|static|async|\s)+[\w<>\[\],\s]+\s*\((\w+)\s*\([^)]*\)\)(?:\s*:\s*[\w\[\],\s]+)?\s*\{',
            "import_pattern": r'import\s+[\w{},\s]+\s+from\s+[\'"][\w./]+[\'"];',
            "comment_pattern": r'(?:/\*[\s\S]*?\*/|//[^\n]*)',
            "extensions": [".ts", ".tsx"],
        },
        "javascript": {
            "class_pattern": r'class\s+(\w+)(?:\s+extends\s+\w+)?\s*\{',
            "function_pattern": r'(?:async\s+)?function\s+(\w+)\s*\([^)]*\)\s*\{',
            "arrow_function_pattern": r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[\w]+)\s*=>',
            "import_pattern": r'import\s+[\w{},\s]+\s+from\s+[\'"][\w./]+[\'"];',
            "comment_pattern": r'(?:/\*[\s\S]*?\*/|//[^\n]*)',
            "extensions": [".js", ".jsx"],
        },
    }
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize code chunker.
        
        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkingConfig()
        self._chunk_counter = 0
    
    def detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Language identifier
        """
        ext = Path(file_path).suffix.lower()
        
        for lang, patterns in self.LANGUAGE_PATTERNS.items():
            if ext in patterns.get("extensions", []):
                return lang
        
        return "unknown"
    
    def chunk_file(
        self, 
        file_path: str, 
        content: Optional[str] = None
    ) -> List[CodeChunk]:
        """Chunk a file into code chunks.
        
        Args:
            file_path: Path to the file
            content: Optional file content (will read if not provided)
            
        Returns:
            List of CodeChunks
        """
        if content is None:
            try:
                content = Path(file_path).read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"[CodeChunker] Failed to read {file_path}: {e}")
                return []
        
        language = self.detect_language(file_path)
        
        if self.config.strategy == ChunkStrategy.BY_FILE:
            return self._chunk_by_file(file_path, content, language)
        elif self.config.strategy == ChunkStrategy.BY_CLASS:
            return self._chunk_by_class(file_path, content, language)
        elif self.config.strategy == ChunkStrategy.BY_METHOD:
            return self._chunk_by_method(file_path, content, language)
        elif self.config.strategy == ChunkStrategy.BY_LINES:
            return self._chunk_by_lines(file_path, content, language)
        elif self.config.strategy == ChunkStrategy.BY_TOKENS:
            return self._chunk_by_tokens(file_path, content, language)
        else:
            return self._chunk_hybrid(file_path, content, language)
    
    def _generate_chunk_id(self) -> str:
        """Generate unique chunk ID."""
        self._chunk_counter += 1
        return f"chunk_{self._chunk_counter:06d}"
    
    def _chunk_by_file(
        self, 
        file_path: str, 
        content: str, 
        language: str
    ) -> List[CodeChunk]:
        """Create a single chunk for the entire file."""
        lines = content.splitlines()
        
        chunk = CodeChunk(
            id=self._generate_chunk_id(),
            content=content,
            chunk_type=ChunkType.FILE,
            file_path=file_path,
            start_line=1,
            end_line=len(lines),
            name=Path(file_path).stem,
            language=language,
        )
        
        return [chunk]
    
    def _chunk_by_class(
        self, 
        file_path: str, 
        content: str, 
        language: str
    ) -> List[CodeChunk]:
        """Chunk by class definitions."""
        chunks = []
        lines = content.splitlines()
        patterns = self.LANGUAGE_PATTERNS.get(language, {})
        
        class_pattern = patterns.get("class_pattern")
        if not class_pattern:
            return self._chunk_by_file(file_path, content, language)
        
        class_matches = list(re.finditer(class_pattern, content))
        
        if not class_matches:
            return self._chunk_by_file(file_path, content, language)
        
        for i, match in enumerate(class_matches):
            start_pos = match.start()
            end_pos = self._find_matching_brace(content, match.end() - 1)
            
            if end_pos == -1:
                end_pos = len(content)
            
            start_line = content[:start_pos].count('\n') + 1
            end_line = content[:end_pos].count('\n') + 1
            
            class_content = content[start_pos:end_pos + 1]
            
            chunk = CodeChunk(
                id=self._generate_chunk_id(),
                content=class_content,
                chunk_type=ChunkType.CLASS,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                name=match.group(1) if match.groups() else f"class_{i}",
                language=language,
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_method(
        self, 
        file_path: str, 
        content: str, 
        language: str
    ) -> List[CodeChunk]:
        """Chunk by method/function definitions."""
        chunks = []
        patterns = self.LANGUAGE_PATTERNS.get(language, {})
        
        method_pattern = patterns.get("method_pattern") or patterns.get("function_pattern")
        if not method_pattern:
            return self._chunk_by_class(file_path, content, language)
        
        class_chunks = self._chunk_by_class(file_path, content, language)
        
        for class_chunk in class_chunks:
            if class_chunk.chunk_type != ChunkType.CLASS:
                chunks.append(class_chunk)
                continue
            
            class_content = class_chunk.content
            method_matches = list(re.finditer(method_pattern, class_content))
            
            if not method_matches:
                chunks.append(class_chunk)
                continue
            
            for i, match in enumerate(method_matches):
                start_pos = match.start()
                end_pos = self._find_matching_brace(class_content, match.end() - 1)
                
                if end_pos == -1:
                    end_pos = len(class_content)
                
                method_content = class_content[start_pos:end_pos + 1]
                
                start_line = class_chunk.start_line + class_content[:start_pos].count('\n')
                end_line = class_chunk.start_line + class_content[:end_pos].count('\n')
                
                chunk = CodeChunk(
                    id=self._generate_chunk_id(),
                    content=method_content,
                    chunk_type=ChunkType.METHOD,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    name=match.group(1) if match.groups() else f"method_{i}",
                    parent=class_chunk.name,
                    language=language,
                )
                
                chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_lines(
        self, 
        file_path: str, 
        content: str, 
        language: str
    ) -> List[CodeChunk]:
        """Chunk by fixed line count."""
        chunks = []
        lines = content.splitlines()
        max_lines = self.config.max_chunk_lines
        overlap = self.config.overlap_lines
        
        i = 0
        while i < len(lines):
            end_idx = min(i + max_lines, len(lines))
            chunk_lines = lines[i:end_idx]
            
            chunk = CodeChunk(
                id=self._generate_chunk_id(),
                content='\n'.join(chunk_lines),
                chunk_type=ChunkType.FILE,
                file_path=file_path,
                start_line=i + 1,
                end_line=end_idx,
                name=f"lines_{i+1}_{end_idx}",
                language=language,
                metadata={"chunk_index": len(chunks)},
            )
            
            chunks.append(chunk)
            i += max_lines - overlap
        
        return chunks
    
    def _chunk_by_tokens(
        self, 
        file_path: str, 
        content: str, 
        language: str
    ) -> List[CodeChunk]:
        """Chunk by estimated token count."""
        chunks = []
        lines = content.splitlines()
        max_tokens = self.config.max_chunk_tokens
        
        current_chunk_lines = []
        current_tokens = 0
        start_line = 1
        
        for i, line in enumerate(lines):
            line_tokens = len(line.split())
            
            if current_tokens + line_tokens > max_tokens and current_chunk_lines:
                chunk = CodeChunk(
                    id=self._generate_chunk_id(),
                    content='\n'.join(current_chunk_lines),
                    chunk_type=ChunkType.FILE,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=i,
                    name=f"tokens_{start_line}_{i}",
                    language=language,
                    metadata={"estimated_tokens": current_tokens},
                )
                chunks.append(chunk)
                
                current_chunk_lines = [line]
                current_tokens = line_tokens
                start_line = i + 1
            else:
                current_chunk_lines.append(line)
                current_tokens += line_tokens
        
        if current_chunk_lines:
            chunk = CodeChunk(
                id=self._generate_chunk_id(),
                content='\n'.join(current_chunk_lines),
                chunk_type=ChunkType.FILE,
                file_path=file_path,
                start_line=start_line,
                end_line=len(lines),
                name=f"tokens_{start_line}_{len(lines)}",
                language=language,
                metadata={"estimated_tokens": current_tokens},
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_hybrid(
        self, 
        file_path: str, 
        content: str, 
        language: str
    ) -> List[CodeChunk]:
        """Hybrid chunking strategy combining class and method chunking."""
        chunks = []
        patterns = self.LANGUAGE_PATTERNS.get(language, {})
        
        class_chunks = self._chunk_by_class(file_path, content, language)
        
        for class_chunk in class_chunks:
            if class_chunk.chunk_type != ChunkType.CLASS:
                chunks.append(class_chunk)
                continue
            
            if class_chunk.line_count <= self.config.max_chunk_lines:
                chunks.append(class_chunk)
                continue
            
            method_chunks = self._extract_methods_from_chunk(class_chunk, language, patterns)
            
            if method_chunks:
                chunks.extend(method_chunks)
            else:
                line_chunks = self._chunk_by_lines(
                    file_path, 
                    class_chunk.content, 
                    language
                )
                for lc in line_chunks:
                    lc.start_line += class_chunk.start_line - 1
                    lc.end_line += class_chunk.start_line - 1
                    lc.parent = class_chunk.name
                chunks.extend(line_chunks)
        
        return chunks
    
    def _extract_methods_from_chunk(
        self, 
        class_chunk: CodeChunk, 
        language: str,
        patterns: Dict[str, Any]
    ) -> List[CodeChunk]:
        """Extract methods from a class chunk."""
        chunks = []
        content = class_chunk.content
        
        method_pattern = patterns.get("method_pattern") or patterns.get("function_pattern")
        if not method_pattern:
            return []
        
        method_matches = list(re.finditer(method_pattern, content))
        
        for i, match in enumerate(method_matches):
            start_pos = match.start()
            end_pos = self._find_matching_brace(content, match.end() - 1)
            
            if end_pos == -1:
                end_pos = len(content)
            
            method_content = content[start_pos:end_pos + 1]
            
            if len(method_content.splitlines()) < self.config.min_chunk_lines:
                continue
            
            start_line = class_chunk.start_line + content[:start_pos].count('\n')
            end_line = class_chunk.start_line + content[:end_pos].count('\n')
            
            chunk = CodeChunk(
                id=self._generate_chunk_id(),
                content=method_content,
                chunk_type=ChunkType.METHOD,
                file_path=class_chunk.file_path,
                start_line=start_line,
                end_line=end_line,
                name=match.group(1) if match.groups() else f"method_{i}",
                parent=class_chunk.name,
                language=language,
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _find_matching_brace(self, content: str, start_pos: int) -> int:
        """Find the position of the matching closing brace.
        
        Args:
            content: The content to search
            start_pos: Starting position (should be at or before opening brace)
            
        Returns:
            Position of matching closing brace, or -1 if not found
        """
        brace_count = 0
        found_open = False
        
        for i in range(start_pos, len(content)):
            char = content[i]
            
            if char == '{':
                brace_count += 1
                found_open = True
            elif char == '}':
                brace_count -= 1
                
                if found_open and brace_count == 0:
                    return i
        
        return -1
    
    def chunk_project(
        self, 
        project_path: str,
        file_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> List[CodeChunk]:
        """Chunk all files in a project.
        
        Args:
            project_path: Path to the project
            file_patterns: File patterns to include (e.g., ["**/*.java"])
            exclude_patterns: Patterns to exclude
            
        Returns:
            List of all CodeChunks in the project
        """
        all_chunks = []
        project_path = Path(project_path)
        
        if file_patterns is None:
            file_patterns = ["**/*.java", "**/*.py", "**/*.ts", "**/*.js"]
        
        if exclude_patterns is None:
            exclude_patterns = ["**/test/**", "**/tests/**", "**/__pycache__/**", "**/node_modules/**"]
        
        for pattern in file_patterns:
            for file_path in project_path.glob(pattern):
                if any(file_path.match(ex) for ex in exclude_patterns):
                    continue
                
                chunks = self.chunk_file(str(file_path))
                all_chunks.extend(chunks)
        
        logger.info(f"[CodeChunker] Chunked {len(all_chunks)} chunks from project")
        return all_chunks
