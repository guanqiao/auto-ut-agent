"""Knowledge Graph - Cross-project knowledge association and reasoning.

This module provides a comprehensive knowledge graph system that:
- Builds project-specific knowledge graphs
- Supports cross-project knowledge association
- Enables knowledge reasoning and inference
- Integrates with episodic, semantic, and procedural memory
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from collections import defaultdict
import uuid

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of entities in the knowledge graph."""
    PROJECT = "project"
    MODULE = "module"
    CLASS = "class"
    INTERFACE = "interface"
    METHOD = "method"
    FUNCTION = "function"
    FIELD = "field"
    VARIABLE = "variable"
    PARAMETER = "parameter"
    PACKAGE = "package"
    LIBRARY = "library"
    FRAMEWORK = "framework"
    PATTERN = "pattern"
    CONCEPT = "concept"
    TECHNOLOGY = "technology"
    BEST_PRACTICE = "best_practice"
    ERROR_PATTERN = "error_pattern"
    TEST_STRATEGY = "test_strategy"


class RelationType(Enum):
    """Types of relationships between entities."""
    # Structural relations
    CONTAINS = "contains"
    EXTENDS = "extends"
    IMPLEMENTS = "implements"
    DEPENDS_ON = "depends_on"
    IMPORTS = "imports"
    
    # Semantic relations
    IS_A = "is_a"
    HAS_A = "has_a"
    USES = "uses"
    CALLS = "calls"
    REFERENCES = "references"
    
    # Test relations
    TESTS = "tests"
    MOCKS = "mocks"
    COVERS = "covers"
    VERIFIES = "verifies"
    
    # Knowledge relations
    RELATED_TO = "related_to"
    SIMILAR_TO = "similar_to"
    INSTANCE_OF = "instance_of"
    APPLIES_TO = "applies_to"
    SOLVES = "solves"
    CAUSES = "causes"
    
    # Cross-project relations
    SHARED_WITH = "shared_with"
    ADAPTED_FROM = "adapted_from"
    INSPIRED_BY = "inspired_by"


@dataclass
class KnowledgeEntity:
    """An entity in the knowledge graph."""
    entity_id: str
    entity_type: EntityType
    name: str
    full_name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    source_project: Optional[str] = None
    source_file: Optional[str] = None
    line_number: Optional[int] = None
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type.value,
            "name": self.name,
            "full_name": self.full_name,
            "properties": self.properties,
            "source_project": self.source_project,
            "source_file": self.source_file,
            "line_number": self.line_number,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeEntity':
        """Create from dictionary."""
        return cls(
            entity_id=data["entity_id"],
            entity_type=EntityType(data["entity_type"]),
            name=data["name"],
            full_name=data["full_name"],
            properties=data.get("properties", {}),
            source_project=data.get("source_project"),
            source_file=data.get("source_file"),
            line_number=data.get("line_number"),
            confidence=data.get("confidence", 1.0),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        )


@dataclass
class KnowledgeRelation:
    """A relation between entities in the knowledge graph."""
    relation_id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "relation_id": self.relation_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "properties": self.properties,
            "weight": self.weight,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeRelation':
        """Create from dictionary."""
        return cls(
            relation_id=data["relation_id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=RelationType(data["relation_type"]),
            properties=data.get("properties", {}),
            weight=data.get("weight", 1.0),
            confidence=data.get("confidence", 1.0),
            created_at=datetime.fromisoformat(data["created_at"])
        )


@dataclass
class KnowledgePath:
    """A path in the knowledge graph representing a reasoning chain."""
    path_id: str
    entities: List[KnowledgeEntity]
    relations: List[KnowledgeRelation]
    confidence: float
    reasoning: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path_id": self.path_id,
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
            "confidence": self.confidence,
            "reasoning": self.reasoning
        }


class KnowledgeGraph:
    """Knowledge Graph for cross-project knowledge management and reasoning.
    
    This class provides:
    - Multi-project knowledge representation
    - Cross-project knowledge association
    - Knowledge reasoning and inference
    - Pattern recognition across projects
    - Knowledge evolution tracking
    
    Features:
    - Entity and relation management
    - Graph traversal and path finding
    - Similarity-based knowledge retrieval
    - Rule-based inference
    - Knowledge consolidation
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize knowledge graph.
        
        Args:
            db_path: Path to SQLite database
        """
        if db_path is None:
            home = Path.home()
            db_dir = home / ".pyutagent"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "knowledge_graph.db"
        
        self.db_path = str(db_path)
        self._init_database()
        
        # Caches for performance
        self._entity_cache: Dict[str, KnowledgeEntity] = {}
        self._relation_cache: Dict[str, KnowledgeRelation] = {}
        self._name_index: Dict[str, Set[str]] = defaultdict(set)
        self._type_index: Dict[EntityType, Set[str]] = defaultdict(set)
        self._project_index: Dict[str, Set[str]] = defaultdict(set)
        
        logger.info(f"[KnowledgeGraph] Initialized with database: {self.db_path}")
    
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Entities table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS entities (
                    entity_id TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    full_name TEXT NOT NULL,
                    properties TEXT,
                    source_project TEXT,
                    source_file TEXT,
                    line_number INTEGER,
                    confidence REAL DEFAULT 1.0,
                    created_at TEXT,
                    updated_at TEXT
                )
            ''')
            
            # Relations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS relations (
                    relation_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    properties TEXT,
                    weight REAL DEFAULT 1.0,
                    confidence REAL DEFAULT 1.0,
                    created_at TEXT,
                    FOREIGN KEY (source_id) REFERENCES entities(entity_id),
                    FOREIGN KEY (target_id) REFERENCES entities(entity_id)
                )
            ''')
            
            # Inference rules table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS inference_rules (
                    rule_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    premise TEXT NOT NULL,
                    conclusion TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    enabled INTEGER DEFAULT 1,
                    created_at TEXT
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(entity_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_project ON entities(source_project)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_relation_source ON relations(source_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_relation_target ON relations(target_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_relation_type ON relations(relation_type)')
            
            conn.commit()
    
    def add_entity(
        self,
        entity_type: EntityType,
        name: str,
        full_name: str,
        properties: Optional[Dict[str, Any]] = None,
        source_project: Optional[str] = None,
        source_file: Optional[str] = None,
        line_number: Optional[int] = None,
        confidence: float = 1.0
    ) -> str:
        """Add an entity to the knowledge graph.
        
        Args:
            entity_type: Type of entity
            name: Short name
            full_name: Fully qualified name
            properties: Additional properties
            source_project: Source project
            source_file: Source file path
            line_number: Line number in source
            confidence: Confidence level
            
        Returns:
            Entity ID
        """
        # Include source_project in ID generation to distinguish entities from different projects
        id_seed = f"{entity_type.value}:{full_name}"
        if source_project:
            id_seed = f"{source_project}:{id_seed}"
        entity_id = self._generate_id(id_seed)
        
        entity = KnowledgeEntity(
            entity_id=entity_id,
            entity_type=entity_type,
            name=name,
            full_name=full_name,
            properties=properties or {},
            source_project=source_project,
            source_file=source_file,
            line_number=line_number,
            confidence=confidence
        )
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO entities
                (entity_id, entity_type, name, full_name, properties, source_project,
                 source_file, line_number, confidence, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entity_id,
                entity_type.value,
                name,
                full_name,
                json.dumps(properties or {}),
                source_project,
                source_file,
                line_number,
                confidence,
                entity.created_at.isoformat(),
                entity.updated_at.isoformat()
            ))
            conn.commit()
        
        # Update caches
        self._entity_cache[entity_id] = entity
        self._name_index[name].add(entity_id)
        self._name_index[full_name].add(entity_id)
        self._type_index[entity_type].add(entity_id)
        if source_project:
            self._project_index[source_project].add(entity_id)
        
        logger.debug(f"[KnowledgeGraph] Added entity: {entity_id[:8]} ({entity_type.value})")
        return entity_id
    
    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        properties: Optional[Dict[str, Any]] = None,
        weight: float = 1.0,
        confidence: float = 1.0
    ) -> str:
        """Add a relation between entities.
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            relation_type: Type of relation
            properties: Additional properties
            weight: Relation weight
            confidence: Confidence level
            
        Returns:
            Relation ID
        """
        relation_id = self._generate_id(f"{source_id}:{relation_type.value}:{target_id}")
        
        relation = KnowledgeRelation(
            relation_id=relation_id,
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            properties=properties or {},
            weight=weight,
            confidence=confidence
        )
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO relations
                (relation_id, source_id, target_id, relation_type, properties, weight, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                relation_id,
                source_id,
                target_id,
                relation_type.value,
                json.dumps(properties or {}),
                weight,
                confidence,
                relation.created_at.isoformat()
            ))
            conn.commit()
        
        self._relation_cache[relation_id] = relation
        
        logger.debug(f"[KnowledgeGraph] Added relation: {source_id[:8]} -> {target_id[:8]} ({relation_type.value})")
        return relation_id
    
    def get_entity(self, entity_id: str) -> Optional[KnowledgeEntity]:
        """Get an entity by ID.
        
        Args:
            entity_id: Entity ID
            
        Returns:
            Entity or None
        """
        if entity_id in self._entity_cache:
            return self._entity_cache[entity_id]
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM entities WHERE entity_id = ?', (entity_id,))
            row = cursor.fetchone()
            
            if row:
                entity = self._row_to_entity(row)
                self._entity_cache[entity_id] = entity
                return entity
        
        return None
    
    def find_entities_by_name(
        self,
        name: str,
        entity_type: Optional[EntityType] = None,
        source_project: Optional[str] = None
    ) -> List[KnowledgeEntity]:
        """Find entities by name.
        
        Args:
            name: Entity name
            entity_type: Optional type filter
            source_project: Optional project filter
            
        Returns:
            List of matching entities
        """
        entity_ids = self._name_index.get(name, set())
        
        if not entity_ids:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT * FROM entities WHERE name = ? OR full_name = ?',
                    (name, name)
                )
                rows = cursor.fetchall()
                entity_ids = {row[0] for row in rows}
        
        entities = []
        for eid in entity_ids:
            entity = self.get_entity(eid)
            if entity:
                if entity_type and entity.entity_type != entity_type:
                    continue
                if source_project and entity.source_project != source_project:
                    continue
                entities.append(entity)
        
        return entities
    
    def find_entities_by_type(
        self,
        entity_type: EntityType,
        source_project: Optional[str] = None
    ) -> List[KnowledgeEntity]:
        """Find entities by type.
        
        Args:
            entity_type: Entity type
            source_project: Optional project filter
            
        Returns:
            List of matching entities
        """
        entity_ids = self._type_index.get(entity_type, set())
        
        if not entity_ids:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM entities WHERE entity_type = ?', (entity_type.value,))
                rows = cursor.fetchall()
                return [self._row_to_entity(row) for row in rows]
        
        entities = []
        for eid in entity_ids:
            entity = self.get_entity(eid)
            if entity:
                if source_project and entity.source_project != source_project:
                    continue
                entities.append(entity)
        
        return entities
    
    def get_outgoing_relations(
        self,
        entity_id: str,
        relation_type: Optional[RelationType] = None
    ) -> List[KnowledgeRelation]:
        """Get outgoing relations from an entity.
        
        Args:
            entity_id: Entity ID
            relation_type: Optional relation type filter
            
        Returns:
            List of outgoing relations
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if relation_type:
                cursor.execute(
                    'SELECT * FROM relations WHERE source_id = ? AND relation_type = ?',
                    (entity_id, relation_type.value)
                )
            else:
                cursor.execute('SELECT * FROM relations WHERE source_id = ?', (entity_id,))
            
            return [self._row_to_relation(row) for row in cursor.fetchall()]
    
    def get_incoming_relations(
        self,
        entity_id: str,
        relation_type: Optional[RelationType] = None
    ) -> List[KnowledgeRelation]:
        """Get incoming relations to an entity.
        
        Args:
            entity_id: Entity ID
            relation_type: Optional relation type filter
            
        Returns:
            List of incoming relations
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if relation_type:
                cursor.execute(
                    'SELECT * FROM relations WHERE target_id = ? AND relation_type = ?',
                    (entity_id, relation_type.value)
                )
            else:
                cursor.execute('SELECT * FROM relations WHERE target_id = ?', (entity_id,))
            
            return [self._row_to_relation(row) for row in cursor.fetchall()]
    
    def get_neighbors(
        self,
        entity_id: str,
        relation_type: Optional[RelationType] = None,
        direction: str = "both"
    ) -> List[KnowledgeEntity]:
        """Get neighboring entities.
        
        Args:
            entity_id: Entity ID
            relation_type: Optional relation type filter
            direction: "incoming", "outgoing", or "both"
            
        Returns:
            List of neighboring entities
        """
        neighbor_ids = set()
        
        if direction in ("outgoing", "both"):
            for relation in self.get_outgoing_relations(entity_id, relation_type):
                neighbor_ids.add(relation.target_id)
        
        if direction in ("incoming", "both"):
            for relation in self.get_incoming_relations(entity_id, relation_type):
                neighbor_ids.add(relation.source_id)
        
        return [self.get_entity(nid) for nid in neighbor_ids if self.get_entity(nid)]
    
    def find_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5,
        relation_types: Optional[List[RelationType]] = None
    ) -> Optional[KnowledgePath]:
        """Find a path between two entities using BFS.
        
        Args:
            start_id: Starting entity ID
            end_id: Target entity ID
            max_depth: Maximum search depth
            relation_types: Optional list of allowed relation types
            
        Returns:
            KnowledgePath or None
        """
        if start_id == end_id:
            entity = self.get_entity(start_id)
            if entity:
                return KnowledgePath(
                    path_id=self._generate_id(f"path:{start_id}:{end_id}"),
                    entities=[entity],
                    relations=[],
                    confidence=1.0,
                    reasoning="Same entity"
                )
            return None
        
        visited = {start_id}
        queue = [(start_id, [start_id], [])]
        
        while queue:
            current_id, entity_path, relation_path = queue.pop(0)
            
            if len(entity_path) > max_depth:
                continue
            
            # Get outgoing relations
            for relation in self.get_outgoing_relations(current_id):
                if relation_types and relation.relation_type not in relation_types:
                    continue
                
                neighbor_id = relation.target_id
                if neighbor_id == end_id:
                    entities = [self.get_entity(eid) for eid in entity_path + [end_id]]
                    entities = [e for e in entities if e]
                    return KnowledgePath(
                        path_id=self._generate_id(f"path:{start_id}:{end_id}"),
                        entities=entities,
                        relations=relation_path + [relation],
                        confidence=relation.confidence,
                        reasoning=f"Path found via {relation.relation_type.value}"
                    )
                
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, entity_path + [neighbor_id], relation_path + [relation]))
        
        return None
    
    def find_similar_entities(
        self,
        entity_id: str,
        max_results: int = 5
    ) -> List[Tuple[KnowledgeEntity, float]]:
        """Find entities similar to a given entity.
        
        Similarity is based on:
        - Shared neighbors
        - Common relations
        - Same type
        
        Args:
            entity_id: Reference entity ID
            max_results: Maximum number of results
            
        Returns:
            List of (entity, similarity_score) tuples
        """
        reference = self.get_entity(entity_id)
        if not reference:
            return []
        
        # Get reference neighbors
        ref_neighbors = set()
        for r in self.get_outgoing_relations(entity_id):
            ref_neighbors.add((r.relation_type, r.target_id))
        for r in self.get_incoming_relations(entity_id):
            ref_neighbors.add((r.relation_type, r.source_id))
        
        # Find candidates of same type
        candidates = self.find_entities_by_type(reference.entity_type)
        candidates = [c for c in candidates if c.entity_id != entity_id]
        
        similarities = []
        for candidate in candidates:
            # Calculate similarity based on shared neighbors
            cand_neighbors = set()
            for r in self.get_outgoing_relations(candidate.entity_id):
                cand_neighbors.add((r.relation_type, r.target_id))
            for r in self.get_incoming_relations(candidate.entity_id):
                cand_neighbors.add((r.relation_type, r.source_id))
            
            shared = ref_neighbors & cand_neighbors
            total = ref_neighbors | cand_neighbors
            
            if total:
                similarity = len(shared) / len(total)
                if similarity > 0:
                    similarities.append((candidate, similarity))
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:max_results]
    
    def add_inference_rule(
        self,
        name: str,
        description: str,
        premise: Dict[str, Any],
        conclusion: Dict[str, Any],
        confidence: float = 1.0
    ) -> str:
        """Add an inference rule.
        
        Args:
            name: Rule name
            description: Rule description
            premise: Premise pattern
            conclusion: Conclusion pattern
            confidence: Rule confidence
            
        Returns:
            Rule ID
        """
        rule_id = self._generate_id(f"rule:{name}")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO inference_rules
                (rule_id, name, description, premise, conclusion, confidence, enabled, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 1, ?)
            ''', (
                rule_id,
                name,
                description,
                json.dumps(premise),
                json.dumps(conclusion),
                confidence,
                datetime.now().isoformat()
            ))
            conn.commit()
        
        logger.info(f"[KnowledgeGraph] Added inference rule: {name}")
        return rule_id
    
    def apply_inference(self, entity_id: str) -> List[KnowledgeRelation]:
        """Apply inference rules to derive new relations.
        
        Args:
            entity_id: Entity ID to apply inference on
            
        Returns:
            List of inferred relations
        """
        inferred = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM inference_rules WHERE enabled = 1')
            rules = cursor.fetchall()
            
            for rule_row in rules:
                premise = json.loads(rule_row[3])
                conclusion = json.loads(rule_row[4])
                confidence = rule_row[5]
                
                # Check if premise matches
                if self._check_premise(entity_id, premise):
                    # Apply conclusion
                    new_relation = self._apply_conclusion(entity_id, conclusion, confidence)
                    if new_relation:
                        inferred.append(new_relation)
        
        return inferred
    
    def _check_premise(self, entity_id: str, premise: Dict[str, Any]) -> bool:
        """Check if an entity matches a premise pattern."""
        entity = self.get_entity(entity_id)
        if not entity:
            return False
        
        # Check entity type
        if "entity_type" in premise:
            if entity.entity_type.value != premise["entity_type"]:
                return False
        
        # Check for specific relations
        if "has_relation" in premise:
            rel_type = RelationType(premise["has_relation"]["type"])
            relations = self.get_outgoing_relations(entity_id, rel_type)
            if not relations:
                return False
        
        return True
    
    def _apply_conclusion(
        self,
        entity_id: str,
        conclusion: Dict[str, Any],
        confidence: float
    ) -> Optional[KnowledgeRelation]:
        """Apply a conclusion pattern."""
        if "create_relation" in conclusion:
            rel_def = conclusion["create_relation"]
            target_id = rel_def.get("target_id")
            
            if target_id:
                relation_type = RelationType(rel_def["type"])
                relation_id = self.add_relation(
                    entity_id,
                    target_id,
                    relation_type,
                    confidence=confidence
                )
                return self._relation_cache.get(relation_id)
        
        return None
    
    def get_project_knowledge(self, project: str) -> Dict[str, Any]:
        """Get all knowledge for a specific project.
        
        Args:
            project: Project name
            
        Returns:
            Project knowledge summary
        """
        entity_ids = self._project_index.get(project, set())
        
        if not entity_ids:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM entities WHERE source_project = ?', (project,))
                rows = cursor.fetchall()
                entity_ids = {row[0] for row in rows}
        
        entities = [self.get_entity(eid) for eid in entity_ids if self.get_entity(eid)]
        
        # Group by type
        by_type = defaultdict(list)
        for entity in entities:
            by_type[entity.entity_type.value].append(entity)
        
        return {
            "project": project,
            "total_entities": len(entities),
            "entity_types": {k: len(v) for k, v in by_type.items()},
            "entities": [e.to_dict() for e in entities]
        }
    
    def find_cross_project_patterns(self) -> List[Dict[str, Any]]:
        """Find patterns that appear across multiple projects.
        
        Returns:
            List of cross-project patterns
        """
        patterns = []
        
        # Find patterns that appear in multiple projects
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT name, entity_type, COUNT(DISTINCT source_project) as project_count
                FROM entities
                WHERE source_project IS NOT NULL
                GROUP BY name, entity_type
                HAVING project_count > 1
                ORDER BY project_count DESC
            ''')
            
            for row in cursor.fetchall():
                patterns.append({
                    "name": row[0],
                    "type": row[1],
                    "project_count": row[2]
                })
        
        return patterns
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics.
        
        Returns:
            Statistics dictionary
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM entities')
            total_entities = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM relations')
            total_relations = cursor.fetchone()[0]
            
            cursor.execute('SELECT entity_type, COUNT(*) FROM entities GROUP BY entity_type')
            entity_types = dict(cursor.fetchall())
            
            cursor.execute('SELECT relation_type, COUNT(*) FROM relations GROUP BY relation_type')
            relation_types = dict(cursor.fetchall())
            
            cursor.execute('SELECT COUNT(DISTINCT source_project) FROM entities WHERE source_project IS NOT NULL')
            project_count = cursor.fetchone()[0]
            
            return {
                "total_entities": total_entities,
                "total_relations": total_relations,
                "entity_type_distribution": entity_types,
                "relation_type_distribution": relation_types,
                "project_count": project_count,
                "cached_entities": len(self._entity_cache),
                "cached_relations": len(self._relation_cache)
            }
    
    def clear(self):
        """Clear all data from the knowledge graph."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM relations')
            cursor.execute('DELETE FROM entities')
            cursor.execute('DELETE FROM inference_rules')
            conn.commit()
        
        self._entity_cache.clear()
        self._relation_cache.clear()
        self._name_index.clear()
        self._type_index.clear()
        self._project_index.clear()
        
        logger.info("[KnowledgeGraph] Cleared all data")
    
    def _generate_id(self, seed: str) -> str:
        """Generate a unique ID."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))
    
    def _row_to_entity(self, row) -> KnowledgeEntity:
        """Convert database row to KnowledgeEntity."""
        return KnowledgeEntity(
            entity_id=row[0],
            entity_type=EntityType(row[1]),
            name=row[2],
            full_name=row[3],
            properties=json.loads(row[4]) if row[4] else {},
            source_project=row[5],
            source_file=row[6],
            line_number=row[7],
            confidence=row[8] if row[8] else 1.0,
            created_at=datetime.fromisoformat(row[9]) if row[9] else datetime.now(),
            updated_at=datetime.fromisoformat(row[10]) if row[10] else datetime.now()
        )
    
    def _row_to_relation(self, row) -> KnowledgeRelation:
        """Convert database row to KnowledgeRelation."""
        return KnowledgeRelation(
            relation_id=row[0],
            source_id=row[1],
            target_id=row[2],
            relation_type=RelationType(row[3]),
            properties=json.loads(row[4]) if row[4] else {},
            weight=row[5] if row[5] else 1.0,
            confidence=row[6] if row[6] else 1.0,
            created_at=datetime.fromisoformat(row[7]) if row[7] else datetime.now()
        )


def create_knowledge_graph(db_path: Optional[str] = None) -> KnowledgeGraph:
    """Create a knowledge graph instance.
    
    Args:
        db_path: Optional database path
        
    Returns:
        KnowledgeGraph instance
    """
    return KnowledgeGraph(db_path)
