"""Project Knowledge Graph for semantic code understanding.

This module provides a knowledge graph that captures project structure,
code relationships, and semantic information for intelligent test generation.
"""

import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import defaultdict
import uuid

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in the knowledge graph."""
    CLASS = "class"
    INTERFACE = "interface"
    METHOD = "method"
    FIELD = "field"
    PARAMETER = "parameter"
    PACKAGE = "package"
    ANNOTATION = "annotation"
    EXCEPTION = "exception"
    TEST_CLASS = "test_class"
    TEST_METHOD = "test_method"
    DEPENDENCY = "dependency"


class RelationType(Enum):
    """Types of relationships between nodes."""
    CONTAINS = "contains"
    EXTENDS = "extends"
    IMPLEMENTS = "implements"
    CALLS = "calls"
    USES = "uses"
    DEPENDS_ON = "depends_on"
    TESTS = "tests"
    MOCKS = "mocks"
    THROWS = "throws"
    RETURNS = "returns"
    HAS_PARAMETER = "has_parameter"
    HAS_FIELD = "has_field"
    ANNOTATED_WITH = "annotated_with"


@dataclass
class GraphNode:
    """A node in the knowledge graph."""
    node_id: str
    node_type: NodeType
    name: str
    full_name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    source_file: Optional[str] = None
    line_number: Optional[int] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "name": self.name,
            "full_name": self.full_name,
            "properties": self.properties,
            "source_file": self.source_file,
            "line_number": self.line_number,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphNode':
        """Create from dictionary."""
        return cls(
            node_id=data["node_id"],
            node_type=NodeType(data["node_type"]),
            name=data["name"],
            full_name=data["full_name"],
            properties=data.get("properties", {}),
            source_file=data.get("source_file"),
            line_number=data.get("line_number"),
            created_at=data.get("created_at", datetime.now().isoformat())
        )


@dataclass
class GraphEdge:
    """An edge in the knowledge graph."""
    edge_id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "properties": self.properties,
            "weight": self.weight,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphEdge':
        """Create from dictionary."""
        return cls(
            edge_id=data["edge_id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=RelationType(data["relation_type"]),
            properties=data.get("properties", {}),
            weight=data.get("weight", 1.0),
            created_at=data.get("created_at", datetime.now().isoformat())
        )


@dataclass
class CodePattern:
    """A code pattern identified in the project."""
    pattern_id: str
    pattern_type: str
    description: str
    examples: List[str]
    frequency: int = 1
    confidence: float = 0.8
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProjectKnowledgeGraph:
    """Knowledge graph for project code understanding.
    
    Features:
    - Code structure representation
    - Relationship tracking
    - Pattern recognition
    - Test coverage analysis
    - Dependency analysis
    - Semantic search
    """
    
    def __init__(self, db_path: Optional[str] = None, project_root: Optional[str] = None):
        """Initialize project knowledge graph.
        
        Args:
            db_path: Path to SQLite database
            project_root: Root directory of the project
        """
        if db_path is None:
            home = Path.home()
            db_dir = home / ".pyutagent"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "project_knowledge.db"
        
        self.db_path = str(db_path)
        self.project_root = project_root
        
        self._init_database()
        
        self._node_cache: Dict[str, GraphNode] = {}
        self._edge_cache: Dict[str, GraphEdge] = {}
        self._name_index: Dict[str, Set[str]] = defaultdict(set)
        self._type_index: Dict[NodeType, Set[str]] = defaultdict(set)
        
        logger.info(f"[ProjectKnowledgeGraph] Initialized with database: {self.db_path}")
    
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id TEXT PRIMARY KEY,
                    node_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    full_name TEXT NOT NULL,
                    properties TEXT,
                    source_file TEXT,
                    line_number INTEGER,
                    created_at TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS edges (
                    edge_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    properties TEXT,
                    weight REAL DEFAULT 1.0,
                    created_at TEXT,
                    FOREIGN KEY (source_id) REFERENCES nodes(node_id),
                    FOREIGN KEY (target_id) REFERENCES nodes(node_id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS patterns (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    description TEXT,
                    examples TEXT,
                    frequency INTEGER DEFAULT 1,
                    confidence REAL DEFAULT 0.8,
                    tags TEXT
                )
            ''')
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_node_type ON nodes(node_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_node_name ON nodes(name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_edge_source ON edges(source_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_edge_target ON edges(target_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_edge_type ON edges(relation_type)')
            
            conn.commit()
    
    def add_node(
        self,
        node_type: NodeType,
        name: str,
        full_name: str,
        properties: Optional[Dict[str, Any]] = None,
        source_file: Optional[str] = None,
        line_number: Optional[int] = None
    ) -> str:
        """Add a node to the graph.
        
        Args:
            node_type: Type of the node
            name: Short name
            full_name: Fully qualified name
            properties: Additional properties
            source_file: Source file path
            line_number: Line number in source
            
        Returns:
            Node ID
        """
        node_id = self._generate_id(f"{node_type.value}:{full_name}")
        
        node = GraphNode(
            node_id=node_id,
            node_type=node_type,
            name=name,
            full_name=full_name,
            properties=properties or {},
            source_file=source_file,
            line_number=line_number
        )
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO nodes
                (node_id, node_type, name, full_name, properties, source_file, line_number, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                node_id,
                node_type.value,
                name,
                full_name,
                json.dumps(properties or {}),
                source_file,
                line_number,
                node.created_at
            ))
            conn.commit()
        
        self._node_cache[node_id] = node
        self._name_index[name].add(node_id)
        self._name_index[full_name].add(node_id)
        self._type_index[node_type].add(node_id)
        
        logger.debug(f"[ProjectKnowledgeGraph] Added node: {node_id[:8]} ({node_type.value})")
        return node_id
    
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        properties: Optional[Dict[str, Any]] = None,
        weight: float = 1.0
    ) -> str:
        """Add an edge to the graph.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            relation_type: Type of relationship
            properties: Additional properties
            weight: Edge weight
            
        Returns:
            Edge ID
        """
        edge_id = self._generate_id(f"{source_id}:{relation_type.value}:{target_id}")
        
        edge = GraphEdge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            properties=properties or {},
            weight=weight
        )
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO edges
                (edge_id, source_id, target_id, relation_type, properties, weight, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                edge_id,
                source_id,
                target_id,
                relation_type.value,
                json.dumps(properties or {}),
                weight,
                edge.created_at
            ))
            conn.commit()
        
        self._edge_cache[edge_id] = edge
        
        logger.debug(f"[ProjectKnowledgeGraph] Added edge: {source_id[:8]} -> {target_id[:8]} ({relation_type.value})")
        return edge_id
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID."""
        if node_id in self._node_cache:
            return self._node_cache[node_id]
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM nodes WHERE node_id = ?', (node_id,))
            row = cursor.fetchone()
            
            if row:
                node = self._row_to_node(row)
                self._node_cache[node_id] = node
                return node
        
        return None
    
    def find_nodes_by_name(self, name: str, node_type: Optional[NodeType] = None) -> List[GraphNode]:
        """Find nodes by name.
        
        Args:
            name: Node name (short or full)
            node_type: Optional type filter
            
        Returns:
            List of matching nodes
        """
        node_ids = self._name_index.get(name, set())
        
        if not node_ids:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT * FROM nodes WHERE name = ? OR full_name = ?',
                    (name, name)
                )
                rows = cursor.fetchall()
                node_ids = {row[0] for row in rows}
        
        nodes = []
        for nid in node_ids:
            node = self.get_node(nid)
            if node and (node_type is None or node.node_type == node_type):
                nodes.append(node)
        
        return nodes
    
    def find_nodes_by_type(self, node_type: NodeType) -> List[GraphNode]:
        """Find all nodes of a specific type."""
        node_ids = self._type_index.get(node_type, set())
        
        if not node_ids:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM nodes WHERE node_type = ?', (node_type.value,))
                rows = cursor.fetchall()
                return [self._row_to_node(row) for row in rows]
        
        return [self.get_node(nid) for nid in node_ids if self.get_node(nid)]
    
    def get_outgoing_edges(
        self,
        node_id: str,
        relation_type: Optional[RelationType] = None
    ) -> List[GraphEdge]:
        """Get outgoing edges from a node."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if relation_type:
                cursor.execute(
                    'SELECT * FROM edges WHERE source_id = ? AND relation_type = ?',
                    (node_id, relation_type.value)
                )
            else:
                cursor.execute('SELECT * FROM edges WHERE source_id = ?', (node_id,))
            
            return [self._row_to_edge(row) for row in cursor.fetchall()]
    
    def get_incoming_edges(
        self,
        node_id: str,
        relation_type: Optional[RelationType] = None
    ) -> List[GraphEdge]:
        """Get incoming edges to a node."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if relation_type:
                cursor.execute(
                    'SELECT * FROM edges WHERE target_id = ? AND relation_type = ?',
                    (node_id, relation_type.value)
                )
            else:
                cursor.execute('SELECT * FROM edges WHERE target_id = ?', (node_id,))
            
            return [self._row_to_edge(row) for row in cursor.fetchall()]
    
    def get_neighbors(
        self,
        node_id: str,
        relation_type: Optional[RelationType] = None,
        direction: str = "both"
    ) -> List[GraphNode]:
        """Get neighboring nodes.
        
        Args:
            node_id: Node ID
            relation_type: Optional relation type filter
            direction: "incoming", "outgoing", or "both"
            
        Returns:
            List of neighboring nodes
        """
        neighbor_ids = set()
        
        if direction in ("outgoing", "both"):
            for edge in self.get_outgoing_edges(node_id, relation_type):
                neighbor_ids.add(edge.target_id)
        
        if direction in ("incoming", "both"):
            for edge in self.get_incoming_edges(node_id, relation_type):
                neighbor_ids.add(edge.source_id)
        
        return [self.get_node(nid) for nid in neighbor_ids if self.get_node(nid)]
    
    def find_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5
    ) -> Optional[List[str]]:
        """Find a path between two nodes using BFS.
        
        Args:
            start_id: Starting node ID
            end_id: Target node ID
            max_depth: Maximum search depth
            
        Returns:
            List of node IDs forming the path, or None
        """
        if start_id == end_id:
            return [start_id]
        
        visited = {start_id}
        queue = [(start_id, [start_id])]
        
        while queue:
            current_id, path = queue.pop(0)
            
            if len(path) > max_depth:
                continue
            
            for neighbor in self.get_neighbors(current_id):
                if neighbor.node_id == end_id:
                    return path + [end_id]
                
                if neighbor.node_id not in visited:
                    visited.add(neighbor.node_id)
                    queue.append((neighbor.node_id, path + [neighbor.node_id]))
        
        return None
    
    def get_class_dependencies(self, class_id: str) -> Dict[str, List[GraphNode]]:
        """Get all dependencies of a class.
        
        Args:
            class_id: Class node ID
            
        Returns:
            Dictionary of dependency categories
        """
        dependencies = {
            "extends": [],
            "implements": [],
            "uses": [],
            "depends_on": [],
            "calls": []
        }
        
        for edge in self.get_outgoing_edges(class_id):
            target = self.get_node(edge.target_id)
            if target and edge.relation_type.value in dependencies:
                dependencies[edge.relation_type.value].append(target)
        
        for method in self.get_neighbors(class_id, RelationType.CONTAINS):
            if method.node_type == NodeType.METHOD:
                for edge in self.get_outgoing_edges(method.node_id):
                    if edge.relation_type == RelationType.CALLS:
                        target = self.get_node(edge.target_id)
                        if target and target not in dependencies["calls"]:
                            dependencies["calls"].append(target)
        
        return dependencies
    
    def get_test_coverage_info(self, class_id: str) -> Dict[str, Any]:
        """Get test coverage information for a class.
        
        Args:
            class_id: Class node ID
            
        Returns:
            Coverage information dictionary
        """
        class_node = self.get_node(class_id)
        if not class_node or class_node.node_type not in (NodeType.CLASS, NodeType.INTERFACE):
            return {"error": "Invalid class node"}
        
        methods = self.get_neighbors(class_id, RelationType.CONTAINS)
        methods = [m for m in methods if m.node_type == NodeType.METHOD]
        
        test_edges = self.get_incoming_edges(class_id, RelationType.TESTS)
        test_classes = [self.get_node(e.source_id) for e in test_edges]
        test_classes = [t for t in test_classes if t]
        
        tested_methods = set()
        for test_class in test_classes:
            test_methods = self.get_neighbors(test_class.node_id, RelationType.CONTAINS)
            test_methods = [m for m in test_methods if m.node_type == NodeType.TEST_METHOD]
            
            for tm in test_methods:
                for edge in self.get_outgoing_edges(tm.node_id, RelationType.TESTS):
                    tested_methods.add(edge.target_id)
        
        coverage_ratio = len(tested_methods) / len(methods) if methods else 0.0
        
        return {
            "class_name": class_node.name,
            "total_methods": len(methods),
            "tested_methods": len(tested_methods),
            "coverage_ratio": coverage_ratio,
            "test_classes": [t.name for t in test_classes],
            "untested_methods": [
                m.name for m in methods 
                if m.node_id not in tested_methods
            ]
        }
    
    def add_pattern(
        self,
        pattern_type: str,
        description: str,
        examples: List[str],
        tags: Optional[List[str]] = None,
        confidence: float = 0.8
    ) -> str:
        """Add a code pattern to the knowledge base.
        
        Args:
            pattern_type: Type of pattern
            description: Pattern description
            examples: Example code snippets
            tags: Optional tags
            confidence: Confidence level
            
        Returns:
            Pattern ID
        """
        pattern_id = self._generate_id(f"pattern:{pattern_type}:{description[:50]}")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO patterns
                (pattern_id, pattern_type, description, examples, frequency, confidence, tags)
                VALUES (?, ?, ?, ?, 1, ?, ?)
            ''', (
                pattern_id,
                pattern_type,
                description,
                json.dumps(examples),
                confidence,
                json.dumps(tags or [])
            ))
            conn.commit()
        
        return pattern_id
    
    def find_patterns(
        self,
        pattern_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_confidence: float = 0.0
    ) -> List[CodePattern]:
        """Find patterns matching criteria."""
        query = 'SELECT * FROM patterns WHERE confidence >= ?'
        params = [min_confidence]
        
        if pattern_type:
            query += ' AND pattern_type = ?'
            params.append(pattern_type)
        
        query += ' ORDER BY frequency DESC, confidence DESC'
        
        patterns = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            for row in cursor.fetchall():
                pattern = CodePattern(
                    pattern_id=row[0],
                    pattern_type=row[1],
                    description=row[2],
                    examples=json.loads(row[3]) if row[3] else [],
                    frequency=row[4],
                    confidence=row[5],
                    tags=json.loads(row[6]) if row[6] else []
                )
                
                if tags:
                    if not all(t in pattern.tags for t in tags):
                        continue
                
                patterns.append(pattern)
        
        return patterns
    
    def analyze_code_structure(self, source_code: str, file_path: str) -> Dict[str, Any]:
        """Analyze source code and extract structure.
        
        Args:
            source_code: Java source code
            file_path: File path
            
        Returns:
            Analysis results with extracted nodes and edges
        """
        results = {
            "classes": [],
            "methods": [],
            "fields": [],
            "dependencies": [],
            "node_ids": []
        }
        
        package_match = re.search(r'package\s+([\w.]+)\s*;', source_code)
        package_name = package_match.group(1) if package_match else ""
        
        if package_name:
            package_id = self.add_node(
                NodeType.PACKAGE,
                package_name,
                package_name,
                {"source_file": file_path}
            )
            results["node_ids"].append(package_id)
        
        class_pattern = r'(?:public|private|protected)?\s*(?:abstract|final)?\s*class\s+(\w+)(?:\s+extends\s+([\w<>,\s]+))?(?:\s+implements\s+([\w<>,\s]+))?'
        
        for match in re.finditer(class_pattern, source_code):
            class_name = match.group(1)
            full_name = f"{package_name}.{class_name}" if package_name else class_name
            
            class_id = self.add_node(
                NodeType.CLASS,
                class_name,
                full_name,
                {
                    "extends": match.group(2).strip() if match.group(2) else None,
                    "implements": match.group(3).strip() if match.group(3) else None
                },
                file_path
            )
            results["classes"].append({"name": class_name, "id": class_id})
            results["node_ids"].append(class_id)
            
            if package_name:
                self.add_edge(package_id, class_id, RelationType.CONTAINS)
            
            if match.group(2):
                extends_name = match.group(2).strip()
                extends_id = self.add_node(NodeType.CLASS, extends_name, extends_name)
                self.add_edge(class_id, extends_id, RelationType.EXTENDS)
                results["dependencies"].append({"from": class_name, "to": extends_name, "type": "extends"})
            
            if match.group(3):
                implements_list = [i.strip() for i in match.group(3).split(',')]
                for impl in implements_list:
                    impl_id = self.add_node(NodeType.INTERFACE, impl, impl)
                    self.add_edge(class_id, impl_id, RelationType.IMPLEMENTS)
                    results["dependencies"].append({"from": class_name, "to": impl, "type": "implements"})
        
        method_pattern = r'(?:public|private|protected)?\s*(?:static)?\s*(\w+(?:<[\w<>,\s]+>)?)\s+(\w+)\s*\(([^)]*)\)(?:\s+throws\s+([\w,\s]+))?'
        
        for match in re.finditer(method_pattern, source_code):
            return_type = match.group(1)
            method_name = match.group(2)
            params = match.group(3)
            throws = match.group(4)
            
            if method_name in (class_name for c in results["classes"]):
                continue
            
            if results["classes"]:
                parent_class_id = results["classes"][-1]["id"]
                
                method_id = self.add_node(
                    NodeType.METHOD,
                    method_name,
                    f"{results['classes'][-1]['name']}.{method_name}",
                    {
                        "return_type": return_type,
                        "parameters": params,
                        "throws": throws
                    },
                    file_path
                )
                results["methods"].append({"name": method_name, "id": method_id})
                results["node_ids"].append(method_id)
                
                self.add_edge(parent_class_id, method_id, RelationType.CONTAINS)
                
                if params:
                    for param in params.split(','):
                        param = param.strip()
                        if param:
                            param_parts = param.split()
                            if len(param_parts) >= 2:
                                param_type = param_parts[0]
                                param_name = param_parts[-1]
                                param_id = self.add_node(
                                    NodeType.PARAMETER,
                                    param_name,
                                    param_name,
                                    {"type": param_type}
                                )
                                self.add_edge(method_id, param_id, RelationType.HAS_PARAMETER)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM nodes')
            total_nodes = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM edges')
            total_edges = cursor.fetchone()[0]
            
            cursor.execute('SELECT node_type, COUNT(*) FROM nodes GROUP BY node_type')
            node_types = dict(cursor.fetchall())
            
            cursor.execute('SELECT relation_type, COUNT(*) FROM edges GROUP BY relation_type')
            edge_types = dict(cursor.fetchall())
            
            cursor.execute('SELECT COUNT(*) FROM patterns')
            total_patterns = cursor.fetchone()[0]
            
            return {
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "total_patterns": total_patterns,
                "node_type_distribution": node_types,
                "edge_type_distribution": edge_types,
                "cached_nodes": len(self._node_cache),
                "cached_edges": len(self._edge_cache)
            }
    
    def clear(self):
        """Clear all data from the graph."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM edges')
            cursor.execute('DELETE FROM nodes')
            cursor.execute('DELETE FROM patterns')
            conn.commit()
        
        self._node_cache.clear()
        self._edge_cache.clear()
        self._name_index.clear()
        self._type_index.clear()
        
        logger.info("[ProjectKnowledgeGraph] Cleared all data")
    
    def _generate_id(self, seed: str) -> str:
        """Generate a unique ID."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))
    
    def _row_to_node(self, row) -> GraphNode:
        """Convert database row to GraphNode."""
        return GraphNode(
            node_id=row[0],
            node_type=NodeType(row[1]),
            name=row[2],
            full_name=row[3],
            properties=json.loads(row[4]) if row[4] else {},
            source_file=row[5],
            line_number=row[6],
            created_at=row[7]
        )
    
    def _row_to_edge(self, row) -> GraphEdge:
        """Convert database row to GraphEdge."""
        return GraphEdge(
            edge_id=row[0],
            source_id=row[1],
            target_id=row[2],
            relation_type=RelationType(row[3]),
            properties=json.loads(row[4]) if row[4] else {},
            weight=row[5],
            created_at=row[6]
        )
