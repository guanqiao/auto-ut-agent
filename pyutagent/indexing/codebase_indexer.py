"""Codebase indexer for comprehensive code understanding and navigation.

This module provides a Cursor-like codebase indexing system with:
- Full project semantic understanding
- Dependency graph construction
- Natural language code search
- Precise @-symbol referencing

参考 Cursor Codebase Indexing 的设计:
- https://www.cursor.com/blog/cursorless-codebase-indexing
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from pyutagent.tools.java_parser import JavaCodeParser, JavaClass, JavaMethod
from pyutagent.indexing.code_chunker import CodeChunk, CodeChunker, ChunkingConfig
from pyutagent.memory.vector_store import SQLiteVecStore

logger = logging.getLogger(__name__)

T = TypeVar("T")


class SymbolType(Enum):
    """Types of code symbols."""
    CLASS = "class"
    INTERFACE = "interface"
    ENUM = "enum"
    METHOD = "method"
    CONSTRUCTOR = "constructor"
    FIELD = "field"
    ANNOTATION = "annotation"
    PACKAGE = "package"
    MODULE = "module"


class RelationType(Enum):
    """Types of relationships between symbols."""
    EXTENDS = "extends"
    IMPLEMENTS = "implements"
    CALLS = "calls"
    IMPORTS = "imports"
    CONTAINS = "contains"
    REFERENCES = "references"
    OVERRIDES = "overrides"
    DEPENDS_ON = "depends_on"


@dataclass
class CodeSymbol:
    """Represents a code symbol (class, method, field, etc.)."""
    id: str
    name: str
    symbol_type: SymbolType
    file_path: str
    start_line: int
    end_line: int
    signature: str = ""
    docstring: str = ""
    modifiers: List[str] = field(default_factory=list)
    annotations: List[str] = field(default_factory=list)
    parameters: List[Tuple[str, str]] = field(default_factory=list)  # [(type, name), ...]
    return_type: Optional[str] = None
    parent_id: Optional[str] = None
    language: str = "java"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "symbol_type": self.symbol_type.value,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "signature": self.signature,
            "docstring": self.docstring,
            "modifiers": self.modifiers,
            "annotations": self.annotations,
            "parameters": self.parameters,
            "return_type": self.return_type,
            "parent_id": self.parent_id,
            "language": self.language,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CodeSymbol:
        return cls(
            id=data["id"],
            name=data["name"],
            symbol_type=SymbolType(data["symbol_type"]),
            file_path=data["file_path"],
            start_line=data["start_line"],
            end_line=data["end_line"],
            signature=data.get("signature", ""),
            docstring=data.get("docstring", ""),
            modifiers=data.get("modifiers", []),
            annotations=data.get("annotations", []),
            parameters=[tuple(p) for p in data.get("parameters", [])],
            return_type=data.get("return_type"),
            parent_id=data.get("parent_id"),
            language=data.get("language", "java"),
            metadata=data.get("metadata", {}),
        )

    @property
    def full_name(self) -> str:
        """Get fully qualified name."""
        if self.parent_id:
            return f"{self.parent_id}.{self.name}"
        return self.name

    @property
    def location(self) -> str:
        """Get human-readable location."""
        return f"{self.file_path}:{self.start_line}"


@dataclass
class SymbolRelation:
    """Represents a relationship between two symbols."""
    source_id: str
    target_id: str
    relation_type: RelationType
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SymbolRelation:
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=RelationType(data["relation_type"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class FileIndex:
    """Index data for a single file."""
    file_path: str
    content_hash: str
    last_modified: str
    symbols: List[CodeSymbol] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    language: str = "java"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "content_hash": self.content_hash,
            "last_modified": self.last_modified,
            "symbols": [s.to_dict() for s in self.symbols],
            "imports": self.imports,
            "language": self.language,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FileIndex:
        return cls(
            file_path=data["file_path"],
            content_hash=data["content_hash"],
            last_modified=data["last_modified"],
            symbols=[CodeSymbol.from_dict(s) for s in data.get("symbols", [])],
            imports=data.get("imports", []),
            language=data.get("language", "java"),
        )


@dataclass
class CodebaseIndexState:
    """Overall state of the codebase index."""
    project_path: str
    version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    total_files: int = 0
    total_symbols: int = 0
    file_hashes: Dict[str, str] = field(default_factory=dict)
    indexed_languages: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_path": self.project_path,
            "version": self.version,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "total_files": self.total_files,
            "total_symbols": self.total_symbols,
            "file_hashes": self.file_hashes,
            "indexed_languages": list(self.indexed_languages),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CodebaseIndexState:
        return cls(
            project_path=data["project_path"],
            version=data.get("version", "1.0"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            last_updated=data.get("last_updated", datetime.now().isoformat()),
            total_files=data.get("total_files", 0),
            total_symbols=data.get("total_symbols", 0),
            file_hashes=data.get("file_hashes", {}),
            indexed_languages=set(data.get("indexed_languages", [])),
        )


@dataclass
class IndexerConfig:
    """Configuration for the codebase indexer."""
    # File patterns
    include_patterns: List[str] = field(default_factory=lambda: [
        "**/*.java", "**/*.py", "**/*.ts", "**/*.js", "**/*.go", "**/*.rs"
    ])
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "**/test/**", "**/tests/**", "**/__pycache__/**",
        "**/node_modules/**", "**/target/**", "**/build/**",
        "**/dist/**", "**/.git/**", "**/.idea/**", "**/.vscode/**"
    ])

    # Indexing options
    enable_incremental: bool = True
    enable_semantic_search: bool = True
    enable_dependency_graph: bool = True
    index_docstrings: bool = True
    index_comments: bool = False

    # Storage
    index_dir: str = ".pyutagent/codebase_index"
    embedding_dimension: int = 384

    # Chunking
    chunking_config: Optional[ChunkingConfig] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "include_patterns": self.include_patterns,
            "exclude_patterns": self.exclude_patterns,
            "enable_incremental": self.enable_incremental,
            "enable_semantic_search": self.enable_semantic_search,
            "enable_dependency_graph": self.enable_dependency_graph,
            "index_docstrings": self.index_docstrings,
            "index_comments": self.index_comments,
            "index_dir": self.index_dir,
            "embedding_dimension": self.embedding_dimension,
            "chunking_config": self.chunking_config.to_dict() if self.chunking_config else None,
        }


class DependencyGraph:
    """Call graph and dependency graph for the codebase."""

    def __init__(self):
        self._nodes: Dict[str, CodeSymbol] = {}
        self._edges: Dict[str, List[SymbolRelation]] = {}  # source_id -> relations
        self._reverse_edges: Dict[str, List[SymbolRelation]] = {}  # target_id -> relations

    def add_symbol(self, symbol: CodeSymbol) -> None:
        """Add a symbol to the graph."""
        self._nodes[symbol.id] = symbol
        if symbol.id not in self._edges:
            self._edges[symbol.id] = []
        if symbol.id not in self._reverse_edges:
            self._reverse_edges[symbol.id] = []

    def add_relation(self, relation: SymbolRelation) -> None:
        """Add a relationship between symbols."""
        if relation.source_id not in self._edges:
            self._edges[relation.source_id] = []
        if relation.target_id not in self._reverse_edges:
            self._reverse_edges[relation.target_id] = []

        self._edges[relation.source_id].append(relation)
        self._reverse_edges[relation.target_id].append(relation)

    def get_symbol(self, symbol_id: str) -> Optional[CodeSymbol]:
        """Get a symbol by ID."""
        return self._nodes.get(symbol_id)

    def get_relations(self, symbol_id: str, relation_type: Optional[RelationType] = None) -> List[SymbolRelation]:
        """Get all relations from a symbol."""
        relations = self._edges.get(symbol_id, [])
        if relation_type:
            relations = [r for r in relations if r.relation_type == relation_type]
        return relations

    def get_reverse_relations(self, symbol_id: str, relation_type: Optional[RelationType] = None) -> List[SymbolRelation]:
        """Get all relations to a symbol."""
        relations = self._reverse_edges.get(symbol_id, [])
        if relation_type:
            relations = [r for r in relations if r.relation_type == relation_type]
        return relations

    def get_callers(self, method_id: str) -> List[CodeSymbol]:
        """Get all methods that call this method."""
        relations = self.get_reverse_relations(method_id, RelationType.CALLS)
        return [self._nodes[r.source_id] for r in relations if r.source_id in self._nodes]

    def get_callees(self, method_id: str) -> List[CodeSymbol]:
        """Get all methods called by this method."""
        relations = self.get_relations(method_id, RelationType.CALLS)
        return [self._nodes[r.target_id] for r in relations if r.target_id in self._nodes]

    def get_dependencies(self, symbol_id: str) -> List[CodeSymbol]:
        """Get all symbols this symbol depends on."""
        relations = self.get_relations(symbol_id, RelationType.DEPENDS_ON)
        return [self._nodes[r.target_id] for r in relations if r.target_id in self._nodes]

    def get_dependents(self, symbol_id: str) -> List[CodeSymbol]:
        """Get all symbols that depend on this symbol."""
        relations = self.get_reverse_relations(symbol_id, RelationType.DEPENDS_ON)
        return [self._nodes[r.source_id] for r in relations if r.source_id in self._nodes]

    def find_path(self, start_id: str, end_id: str, max_depth: int = 10) -> Optional[List[str]]:
        """Find a path between two symbols using BFS."""
        if start_id not in self._nodes or end_id not in self._nodes:
            return None

        visited = {start_id}
        queue = [(start_id, [start_id])]

        while queue:
            current, path = queue.pop(0)
            if current == end_id:
                return path

            if len(path) >= max_depth:
                continue

            for relation in self._edges.get(current, []):
                if relation.target_id not in visited:
                    visited.add(relation.target_id)
                    queue.append((relation.target_id, path + [relation.target_id]))

        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dict."""
        return {
            "nodes": {k: v.to_dict() for k, v in self._nodes.items()},
            "edges": {
                k: [r.to_dict() for r in v]
                for k, v in self._edges.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DependencyGraph:
        """Deserialize graph from dict."""
        graph = cls()
        for node_data in data.get("nodes", {}).values():
            graph.add_symbol(CodeSymbol.from_dict(node_data))
        for source_id, relations in data.get("edges", {}).items():
            for relation_data in relations:
                graph.add_relation(SymbolRelation.from_dict(relation_data))
        return graph


class JavaSymbolExtractor:
    """Extract symbols from Java source code."""

    def __init__(self):
        self._parser = JavaCodeParser()
        self._symbol_counter = 0

    def _generate_symbol_id(self) -> str:
        """Generate unique symbol ID."""
        self._symbol_counter += 1
        return f"sym_{self._symbol_counter:08d}"

    def extract_from_file(self, file_path: str, content: Optional[str] = None) -> FileIndex:
        """Extract all symbols from a Java file."""
        path = Path(file_path)

        if content is None:
            content = path.read_text(encoding="utf-8")

        content_hash = hashlib.md5(content.encode()).hexdigest()

        try:
            java_class = self._parser.parse(content.encode())
        except Exception as e:
            logger.warning(f"[JavaSymbolExtractor] Failed to parse {file_path}: {e}")
            return FileIndex(
                file_path=str(file_path),
                content_hash=content_hash,
                last_modified=datetime.now().isoformat(),
                symbols=[],
                imports=[],
                language="java",
            )

        symbols = []
        class_id = None

        # Create class symbol
        if java_class.name:
            class_id = self._generate_symbol_id()
            symbol_type = SymbolType.CLASS
            if "interface" in content[:500]:  # Simple heuristic
                symbol_type = SymbolType.INTERFACE
            elif "enum" in content[:500]:
                symbol_type = SymbolType.ENUM

            class_symbol = CodeSymbol(
                id=class_id,
                name=java_class.name,
                symbol_type=symbol_type,
                file_path=str(file_path),
                start_line=1,
                end_line=len(content.splitlines()),
                signature=f"{java_class.package}.{java_class.name}" if java_class.package else java_class.name,
                modifiers=[],
                annotations=java_class.annotations,
                language="java",
                metadata={
                    "package": java_class.package,
                    "imports": java_class.imports,
                },
            )
            symbols.append(class_symbol)

        # Create method symbols
        for method in java_class.methods:
            method_id = self._generate_symbol_id()
            method_symbol = CodeSymbol(
                id=method_id,
                name=method.name,
                symbol_type=SymbolType.CONSTRUCTOR if method.return_type is None else SymbolType.METHOD,
                file_path=str(file_path),
                start_line=method.start_line + 1,
                end_line=method.end_line + 1,
                signature=self._build_method_signature(method),
                modifiers=method.modifiers,
                annotations=method.annotations,
                parameters=method.parameters,
                return_type=method.return_type,
                parent_id=class_id,
                language="java",
            )
            symbols.append(method_symbol)

        # Create field symbols
        for field_type, field_name, modifiers in java_class.fields:
            field_id = self._generate_symbol_id()
            field_symbol = CodeSymbol(
                id=field_id,
                name=field_name,
                symbol_type=SymbolType.FIELD,
                file_path=str(file_path),
                start_line=1,  # Approximate
                end_line=1,
                signature=f"{field_type} {field_name}",
                modifiers=modifiers.split() if modifiers else [],
                return_type=field_type,
                parent_id=class_id,
                language="java",
            )
            symbols.append(field_symbol)

        return FileIndex(
            file_path=str(file_path),
            content_hash=content_hash,
            last_modified=datetime.now().isoformat(),
            symbols=symbols,
            imports=java_class.imports,
            language="java",
        )

    def _build_method_signature(self, method: JavaMethod) -> str:
        """Build a method signature string."""
        modifiers_str = " ".join(method.modifiers) + " " if method.modifiers else ""
        return_type_str = f"{method.return_type} " if method.return_type else ""
        params_str = ", ".join(f"{t} {n}" for t, n in method.parameters)
        return f"{modifiers_str}{return_type_str}{method.name}({params_str})"


class CodebaseIndexer:
    """Main indexer for comprehensive codebase understanding.

    Features:
    - Full project file scanning
    - Java code parsing (method signatures, class dependencies)
    - Call graph and dependency graph construction
    - Incremental index updates
    - Natural language semantic search
    - @-symbol reference resolution
    """

    def __init__(
        self,
        project_path: str,
        config: Optional[IndexerConfig] = None,
        embedding_model: Optional[Any] = None,
    ):
        """Initialize the codebase indexer.

        Args:
            project_path: Path to the project root
            config: Indexer configuration
            embedding_model: Optional embedding model for semantic search
        """
        self.project_path = Path(project_path).resolve()
        self.config = config or IndexerConfig()
        self.embedding_model = embedding_model

        # Initialize components
        self._java_extractor = JavaSymbolExtractor()
        self._chunker = CodeChunker(self.config.chunking_config)
        self._dependency_graph = DependencyGraph()

        # Storage
        self._index_dir = self.project_path / self.config.index_dir
        self._index_dir.mkdir(parents=True, exist_ok=True)

        # State
        self._state = CodebaseIndexState(project_path=str(self.project_path))
        self._file_indices: Dict[str, FileIndex] = {}
        self._symbol_index: Dict[str, CodeSymbol] = {}  # symbol_id -> symbol
        self._name_index: Dict[str, List[str]] = {}  # name -> symbol_ids

        # Vector store for semantic search
        self._vector_store: Optional[SQLiteVecStore] = None
        if self.config.enable_semantic_search:
            db_path = str(self._index_dir / "semantic_index.db")
            self._vector_store = SQLiteVecStore(
                db_path=db_path,
                dimension=self.config.embedding_dimension,
            )

        # Load existing state
        self._load_state()

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, '_vector_store') and self._vector_store:
            try:
                self._vector_store.close()
            except Exception:
                pass

    def _load_state(self) -> None:
        """Load index state from disk."""
        state_file = self._index_dir / "codebase_state.json"
        if state_file.exists():
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._state = CodebaseIndexState.from_dict(data)
                logger.info(f"[CodebaseIndexer] Loaded state: {self._state.total_symbols} symbols")
            except Exception as e:
                logger.warning(f"[CodebaseIndexer] Failed to load state: {e}")

        # Load file indices
        indices_dir = self._index_dir / "file_indices"
        if indices_dir.exists():
            for index_file in indices_dir.glob("*.json"):
                try:
                    with open(index_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    file_index = FileIndex.from_dict(data)
                    self._file_indices[file_index.file_path] = file_index

                    # Index symbols
                    for symbol in file_index.symbols:
                        self._symbol_index[symbol.id] = symbol
                        if symbol.name not in self._name_index:
                            self._name_index[symbol.name] = []
                        self._name_index[symbol.name].append(symbol.id)
                except Exception as e:
                    logger.warning(f"[CodebaseIndexer] Failed to load index {index_file}: {e}")

        # Load dependency graph
        graph_file = self._index_dir / "dependency_graph.json"
        if graph_file.exists():
            try:
                with open(graph_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._dependency_graph = DependencyGraph.from_dict(data)
            except Exception as e:
                logger.warning(f"[CodebaseIndexer] Failed to load dependency graph: {e}")

    def _save_state(self) -> None:
        """Save index state to disk."""
        # Save state
        state_file = self._index_dir / "codebase_state.json"
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(self._state.to_dict(), f, indent=2)

        # Save file indices
        indices_dir = self._index_dir / "file_indices"
        indices_dir.mkdir(parents=True, exist_ok=True)

        for file_path, file_index in self._file_indices.items():
            # Use hash of file path as filename
            file_hash = hashlib.md5(file_path.encode()).hexdigest()
            index_file = indices_dir / f"{file_hash}.json"
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(file_index.to_dict(), f, indent=2)

        # Save dependency graph
        graph_file = self._index_dir / "dependency_graph.json"
        with open(graph_file, "w", encoding="utf-8") as f:
            json.dump(self._dependency_graph.to_dict(), f, indent=2)

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute MD5 hash of file content."""
        try:
            content = file_path.read_bytes()
            return hashlib.md5(content).hexdigest()
        except Exception:
            return ""

    def _has_file_changed(self, file_path: Path) -> bool:
        """Check if file has changed since last indexing."""
        if not self.config.enable_incremental:
            return True

        current_hash = self._compute_file_hash(file_path)
        stored_hash = self._state.file_hashes.get(str(file_path))
        return current_hash != stored_hash

    def _get_files_to_index(self) -> List[Path]:
        """Get list of files that need indexing."""
        files_to_index = []

        for pattern in self.config.include_patterns:
            for file_path in self.project_path.glob(pattern):
                # Check exclude patterns
                if any(file_path.match(ex) for ex in self.config.exclude_patterns):
                    continue

                # Skip hidden files and directories
                if any(part.startswith(".") for part in file_path.parts):
                    continue

                # Check if file has changed
                if self._has_file_changed(file_path):
                    files_to_index.append(file_path)

        return files_to_index

    def _index_file(self, file_path: Path) -> Optional[FileIndex]:
        """Index a single file."""
        try:
            content = file_path.read_text(encoding="utf-8")
            suffix = file_path.suffix.lower()

            if suffix == ".java":
                file_index = self._java_extractor.extract_from_file(str(file_path), content)
            else:
                # For other languages, create basic file index
                content_hash = hashlib.md5(content.encode()).hexdigest()
                file_index = FileIndex(
                    file_path=str(file_path),
                    content_hash=content_hash,
                    last_modified=datetime.now().isoformat(),
                    symbols=[],
                    imports=[],
                    language=suffix.lstrip("."),
                )

            # Update symbol indices
            for symbol in file_index.symbols:
                self._symbol_index[symbol.id] = symbol
                if symbol.name not in self._name_index:
                    self._name_index[symbol.name] = []
                if symbol.id not in self._name_index[symbol.name]:
                    self._name_index[symbol.name].append(symbol.id)

                # Add to dependency graph
                self._dependency_graph.add_symbol(symbol)

            # Build semantic index
            if self._vector_store and self.embedding_model:
                self._index_semantic(file_index, content)

            return file_index

        except Exception as e:
            logger.warning(f"[CodebaseIndexer] Failed to index {file_path}: {e}")
            return None

    def _index_semantic(self, file_index: FileIndex, content: str) -> None:
        """Index file content for semantic search."""
        if not self._vector_store or not self.embedding_model:
            return

        # Create chunks for semantic indexing
        chunks = self._chunker.chunk_file(file_index.file_path, content)

        texts = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "file_path": chunk.file_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "chunk_type": chunk.chunk_type.value,
                "name": chunk.name,
            }
            for chunk in chunks
        ]

        try:
            # Generate embeddings
            if hasattr(self.embedding_model, "encode"):
                embeddings = self.embedding_model.encode(texts)
                if hasattr(embeddings, "tolist"):
                    embeddings = embeddings.tolist()

                # Add to vector store
                self._vector_store.add(texts, embeddings, metadatas)
        except Exception as e:
            logger.warning(f"[CodebaseIndexer] Failed to index semantically: {e}")

    def _build_dependency_graph(self) -> None:
        """Build call and dependency relationships."""
        if not self.config.enable_dependency_graph:
            return

        logger.info("[CodebaseIndexer] Building dependency graph...")

        # Build inheritance relationships
        for symbol in self._symbol_index.values():
            if symbol.symbol_type == SymbolType.CLASS and symbol.metadata.get("package"):
                # Check for extends/implements
                for other in self._symbol_index.values():
                    if other.symbol_type == SymbolType.CLASS and other.id != symbol.id:
                        # This is a simplified check - in reality would need full type resolution
                        pass

        # Build call relationships from method bodies
        for symbol in self._symbol_index.values():
            if symbol.symbol_type == SymbolType.METHOD:
                # Find method calls in the file
                file_path = Path(symbol.file_path)
                if file_path.exists():
                    try:
                        content = file_path.read_text(encoding="utf-8")
                        lines = content.splitlines()
                        method_body = "\n".join(
                            lines[symbol.start_line - 1:symbol.end_line]
                        )

                        # Find method calls (simplified regex-based approach)
                        call_pattern = r'\b(\w+)\s*\('
                        calls = re.findall(call_pattern, method_body)

                        for call_name in calls:
                            if call_name in self._name_index:
                                for target_id in self._name_index[call_name]:
                                    target = self._symbol_index.get(target_id)
                                    if target and target.symbol_type == SymbolType.METHOD:
                                        relation = SymbolRelation(
                                            source_id=symbol.id,
                                            target_id=target_id,
                                            relation_type=RelationType.CALLS,
                                        )
                                        self._dependency_graph.add_relation(relation)
                    except Exception as e:
                        logger.debug(f"Failed to analyze calls in {file_path}: {e}")

    def index_project(
        self,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Index the entire project.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            Indexing results summary
        """
        logger.info(f"[CodebaseIndexer] Starting project indexing: {self.project_path}")

        files_to_index = self._get_files_to_index()
        total_files = len(files_to_index)

        logger.info(f"[CodebaseIndexer] Found {total_files} files to index")

        indexed_count = 0
        error_count = 0
        symbol_count = 0

        for i, file_path in enumerate(files_to_index):
            file_index = self._index_file(file_path)

            if file_index:
                self._file_indices[str(file_path)] = file_index
                self._state.file_hashes[str(file_path)] = file_index.content_hash
                indexed_count += 1
                symbol_count += len(file_index.symbols)

                # Track languages
                self._state.indexed_languages.add(file_index.language)
            else:
                error_count += 1

            if progress_callback:
                progress_callback({
                    "current": i + 1,
                    "total": total_files,
                    "file": str(file_path),
                    "symbols": len(file_index.symbols) if file_index else 0,
                })

        # Build dependency graph
        self._build_dependency_graph()

        # Update state
        self._state.total_files = len(self._file_indices)
        self._state.total_symbols = len(self._symbol_index)
        self._state.last_updated = datetime.now().isoformat()

        # Save state
        self._save_state()

        result = {
            "success": True,
            "total_files": total_files,
            "indexed": indexed_count,
            "errors": error_count,
            "total_symbols": symbol_count,
            "languages": list(self._state.indexed_languages),
        }

        logger.info(f"[CodebaseIndexer] Indexing complete: {result}")
        return result

    def search_symbols(
        self,
        query: str,
        symbol_type: Optional[SymbolType] = None,
        limit: int = 20,
    ) -> List[CodeSymbol]:
        """Search for symbols by name.

        Args:
            query: Search query (supports wildcards)
            symbol_type: Optional symbol type filter
            limit: Maximum results

        Returns:
            List of matching symbols
        """
        results = []
        pattern = query.replace("*", ".*").replace("?", ".")
        regex = re.compile(pattern, re.IGNORECASE)

        for name, symbol_ids in self._name_index.items():
            if regex.search(name):
                for symbol_id in symbol_ids:
                    symbol = self._symbol_index.get(symbol_id)
                    if symbol:
                        if symbol_type is None or symbol.symbol_type == symbol_type:
                            results.append(symbol)

            if len(results) >= limit:
                break

        return results[:limit]

    def search_semantic(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search code using natural language.

        Args:
            query: Natural language query
            limit: Maximum results

        Returns:
            List of search results with content and metadata
        """
        if not self._vector_store or not self.embedding_model:
            logger.warning("[CodebaseIndexer] Semantic search not available")
            return []

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            if hasattr(query_embedding, "tolist"):
                query_embedding = query_embedding.tolist()[0]
            else:
                query_embedding = query_embedding[0]

            # Search vector store
            results = self._vector_store.search(query_embedding, k=limit)

            return [
                {
                    "content": content,
                    "score": 1.0 - distance,
                    "distance": distance,
                    **metadata,
                }
                for content, distance, metadata in results
            ]
        except Exception as e:
            logger.warning(f"[CodebaseIndexer] Semantic search failed: {e}")
            return []

    def get_symbol(self, symbol_id: str) -> Optional[CodeSymbol]:
        """Get a symbol by ID."""
        return self._symbol_index.get(symbol_id)

    def get_symbol_by_name(
        self,
        name: str,
        file_path: Optional[str] = None,
    ) -> Optional[CodeSymbol]:
        """Get a symbol by name, optionally filtered by file.

        Args:
            name: Symbol name
            file_path: Optional file path to narrow search

        Returns:
            Matching symbol or None
        """
        symbol_ids = self._name_index.get(name, [])

        if not symbol_ids:
            return None

        if file_path:
            for symbol_id in symbol_ids:
                symbol = self._symbol_index.get(symbol_id)
                if symbol and symbol.file_path == file_path:
                    return symbol

        return self._symbol_index.get(symbol_ids[0])

    def get_file_symbols(self, file_path: str) -> List[CodeSymbol]:
        """Get all symbols in a file."""
        file_index = self._file_indices.get(file_path)
        if file_index:
            return file_index.symbols
        return []

    def get_related_symbols(self, symbol_id: str) -> List[CodeSymbol]:
        """Get symbols related to the given symbol."""
        related = []

        # Get symbols in the same class
        symbol = self._symbol_index.get(symbol_id)
        if symbol and symbol.parent_id:
            parent = self._symbol_index.get(symbol.parent_id)
            if parent:
                for sym in self._symbol_index.values():
                    if sym.parent_id == symbol.parent_id and sym.id != symbol_id:
                        related.append(sym)

        # Get call relationships
        for relation in self._dependency_graph.get_relations(symbol_id):
            target = self._symbol_index.get(relation.target_id)
            if target:
                related.append(target)

        for relation in self._dependency_graph.get_reverse_relations(symbol_id):
            source = self._symbol_index.get(relation.source_id)
            if source:
                related.append(source)

        return related

    def get_call_graph(self, method_id: str) -> Dict[str, List[CodeSymbol]]:
        """Get call graph for a method.

        Returns:
            Dict with 'callers' and 'callees' lists
        """
        return {
            "callers": self._dependency_graph.get_callers(method_id),
            "callees": self._dependency_graph.get_callees(method_id),
        }

    def resolve_reference(self, ref: str) -> Optional[Dict[str, Any]]:
        """Resolve an @-reference to a symbol or file.

        Supports:
        - @file:path/to/File.java
        - @folder:path/to/folder
        - @code:ClassName.methodName
        - @symbol:symbol_name

        Args:
            ref: Reference string (without @ prefix)

        Returns:
            Resolved reference info or None
        """
        if ":" not in ref:
            # Try to guess type
            if "." in ref and not ref.endswith((".java", ".py", ".ts")):
                ref = f"code:{ref}"
            elif "/" in ref or "\\" in ref:
                if Path(ref).suffix:
                    ref = f"file:{ref}"
                else:
                    ref = f"folder:{ref}"
            else:
                ref = f"symbol:{ref}"

        ref_type, ref_value = ref.split(":", 1)

        if ref_type == "file":
            file_path = self.project_path / ref_value
            if file_path.exists():
                return {
                    "type": "file",
                    "path": str(file_path),
                    "relative_path": ref_value,
                }
            # Try to find in indexed files
            for indexed_path in self._file_indices.keys():
                if indexed_path.endswith(ref_value) or ref_value in indexed_path:
                    return {
                        "type": "file",
                        "path": indexed_path,
                        "relative_path": ref_value,
                    }

        elif ref_type == "folder":
            folder_path = self.project_path / ref_value
            if folder_path.exists() and folder_path.is_dir():
                files = list(folder_path.rglob("*"))
                return {
                    "type": "folder",
                    "path": str(folder_path),
                    "relative_path": ref_value,
                    "file_count": len([f for f in files if f.is_file()]),
                }

        elif ref_type in ("code", "symbol"):
            # Try to find symbol
            symbol = self.get_symbol_by_name(ref_value)
            if symbol:
                return {
                    "type": "symbol",
                    "symbol": symbol,
                    "related": self.get_related_symbols(symbol.id),
                }

            # Try partial match
            parts = ref_value.split(".")
            if len(parts) > 1:
                class_name = parts[0]
                method_name = parts[1] if len(parts) > 1 else None

                class_symbol = self.get_symbol_by_name(class_name)
                if class_symbol and method_name:
                    # Find method in class
                    for sym in self._symbol_index.values():
                        if (sym.parent_id == class_symbol.id and
                            sym.name == method_name):
                            return {
                                "type": "symbol",
                                "symbol": sym,
                                "related": self.get_related_symbols(sym.id),
                            }

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "project_path": str(self.project_path),
            "total_files": self._state.total_files,
            "total_symbols": self._state.total_symbols,
            "languages": list(self._state.indexed_languages),
            "last_updated": self._state.last_updated,
            "index_version": self._state.version,
        }

    def clear_index(self) -> None:
        """Clear the entire index."""
        self._state = CodebaseIndexState(project_path=str(self.project_path))
        self._file_indices.clear()
        self._symbol_index.clear()
        self._name_index.clear()
        self._dependency_graph = DependencyGraph()

        # Close vector store connection before clearing
        if self._vector_store:
            try:
                self._vector_store.close()
            except Exception:
                pass
            self._vector_store = None

        # Remove index files
        if self._index_dir.exists():
            import shutil
            try:
                shutil.rmtree(self._index_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"[CodebaseIndexer] Failed to remove index dir: {e}")

        # Re-initialize vector store if needed
        if self.config.enable_semantic_search:
            db_path = str(self._index_dir / "semantic_index.db")
            self._vector_store = SQLiteVecStore(
                db_path=db_path,
                dimension=self.config.embedding_dimension,
            )

        logger.info("[CodebaseIndexer] Index cleared")

    def refresh_file(self, file_path: str) -> Dict[str, Any]:
        """Refresh index for a specific file.

        Args:
            file_path: Path to the file

        Returns:
            Refresh results
        """
        path = Path(file_path)
        if not path.exists():
            return {"success": False, "error": "File not found"}

        # Remove old index
        if str(file_path) in self._file_indices:
            old_index = self._file_indices[str(file_path)]
            for symbol in old_index.symbols:
                if symbol.id in self._symbol_index:
                    del self._symbol_index[symbol.id]
                if symbol.name in self._name_index:
                    self._name_index[symbol.name] = [
                        sid for sid in self._name_index[symbol.name]
                        if sid != symbol.id
                    ]

        # Re-index
        file_index = self._index_file(path)

        if file_index:
            self._file_indices[str(file_path)] = file_index
            self._state.file_hashes[str(file_path)] = file_index.content_hash
            self._state.total_files = len(self._file_indices)
            self._state.total_symbols = len(self._symbol_index)
            self._save_state()

            return {
                "success": True,
                "file": str(file_path),
                "symbols": len(file_index.symbols),
            }

        return {"success": False, "error": "Failed to index file"}
