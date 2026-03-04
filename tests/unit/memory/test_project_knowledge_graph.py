"""Unit tests for ProjectKnowledgeGraph module."""

import pytest
import tempfile
import os

from pyutagent.memory.project_knowledge_graph import (
    ProjectKnowledgeGraph,
    GraphNode,
    GraphEdge,
    NodeType,
    RelationType,
    CodePattern,
)


class TestProjectKnowledgeGraph:
    """Tests for ProjectKnowledgeGraph class."""

    def test_init_default_db(self):
        """Test initialization with default database."""
        graph = ProjectKnowledgeGraph()
        
        assert graph.db_path is not None
        assert graph._node_cache == {}
        assert graph._edge_cache == {}

    def test_init_custom_db(self):
        """Test initialization with custom database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_graph.db")
            graph = ProjectKnowledgeGraph(db_path=db_path)
            
            assert graph.db_path == db_path

    def test_add_node(self):
        """Test adding a node."""
        graph = ProjectKnowledgeGraph()
        
        node_id = graph.add_node(
            node_type=NodeType.CLASS,
            name="UserService",
            full_name="com.example.UserService",
            properties={"is_service": True}
        )
        
        assert node_id is not None
        node = graph.get_node(node_id)
        assert node is not None
        assert node.name == "UserService"

    def test_add_node_with_source_info(self):
        """Test adding node with source file info."""
        graph = ProjectKnowledgeGraph()
        
        node_id = graph.add_node(
            node_type=NodeType.METHOD,
            name="getUser",
            full_name="UserService.getUser",
            source_file="UserService.java",
            line_number=25
        )
        
        node = graph.get_node(node_id)
        assert node.source_file == "UserService.java"
        assert node.line_number == 25

    def test_add_edge(self):
        """Test adding an edge."""
        graph = ProjectKnowledgeGraph()
        
        class_id = graph.add_node(NodeType.CLASS, "Service", "Service")
        method_id = graph.add_node(NodeType.METHOD, "process", "Service.process")
        
        edge_id = graph.add_edge(
            source_id=class_id,
            target_id=method_id,
            relation_type=RelationType.CONTAINS
        )
        
        assert edge_id is not None

    def test_get_node_existing(self):
        """Test getting existing node."""
        graph = ProjectKnowledgeGraph()
        
        node_id = graph.add_node(NodeType.CLASS, "Test", "Test")
        node = graph.get_node(node_id)
        
        assert node is not None
        assert node.name == "Test"

    def test_get_node_non_existing(self):
        """Test getting non-existing node."""
        graph = ProjectKnowledgeGraph()
        
        node = graph.get_node("non_existing_id")
        
        assert node is None

    def test_find_nodes_by_name(self):
        """Test finding nodes by name."""
        graph = ProjectKnowledgeGraph()
        
        graph.add_node(NodeType.CLASS, "UserService", "com.example.UserService")
        graph.add_node(NodeType.CLASS, "UserService", "com.other.UserService")
        
        nodes = graph.find_nodes_by_name("UserService")
        
        assert len(nodes) >= 2

    def test_find_nodes_by_name_with_type_filter(self):
        """Test finding nodes by name with type filter."""
        graph = ProjectKnowledgeGraph()
        
        graph.add_node(NodeType.CLASS, "Test", "Test")
        graph.add_node(NodeType.METHOD, "Test", "someMethod.Test")
        
        nodes = graph.find_nodes_by_name("Test", node_type=NodeType.CLASS)
        
        assert all(n.node_type == NodeType.CLASS for n in nodes)

    def test_find_nodes_by_type(self):
        """Test finding nodes by type."""
        graph = ProjectKnowledgeGraph()
        
        graph.add_node(NodeType.CLASS, "Class1", "Class1")
        graph.add_node(NodeType.CLASS, "Class2", "Class2")
        graph.add_node(NodeType.METHOD, "method1", "method1")
        
        nodes = graph.find_nodes_by_type(NodeType.CLASS)
        
        assert len(nodes) >= 2
        assert all(n.node_type == NodeType.CLASS for n in nodes)

    def test_get_outgoing_edges(self):
        """Test getting outgoing edges."""
        graph = ProjectKnowledgeGraph()
        
        class_id = graph.add_node(NodeType.CLASS, "Service", "Service")
        method1_id = graph.add_node(NodeType.METHOD, "method1", "method1")
        method2_id = graph.add_node(NodeType.METHOD, "method2", "method2")
        
        graph.add_edge(class_id, method1_id, RelationType.CONTAINS)
        graph.add_edge(class_id, method2_id, RelationType.CONTAINS)
        
        edges = graph.get_outgoing_edges(class_id)
        
        assert len(edges) == 2

    def test_get_outgoing_edges_with_type_filter(self):
        """Test getting outgoing edges with type filter."""
        graph = ProjectKnowledgeGraph()
        
        class_id = graph.add_node(NodeType.CLASS, "Service", "Service")
        method_id = graph.add_node(NodeType.METHOD, "method", "method")
        dep_id = graph.add_node(NodeType.CLASS, "Dependency", "Dependency")
        
        graph.add_edge(class_id, method_id, RelationType.CONTAINS)
        graph.add_edge(class_id, dep_id, RelationType.DEPENDS_ON)
        
        edges = graph.get_outgoing_edges(class_id, RelationType.CONTAINS)
        
        assert len(edges) == 1
        assert edges[0].relation_type == RelationType.CONTAINS

    def test_get_incoming_edges(self):
        """Test getting incoming edges."""
        graph = ProjectKnowledgeGraph()
        
        class_id = graph.add_node(NodeType.CLASS, "Service", "Service")
        test_id = graph.add_node(NodeType.TEST_CLASS, "ServiceTest", "ServiceTest")
        
        graph.add_edge(test_id, class_id, RelationType.TESTS)
        
        edges = graph.get_incoming_edges(class_id)
        
        assert len(edges) == 1
        assert edges[0].relation_type == RelationType.TESTS

    def test_get_neighbors(self):
        """Test getting neighbors."""
        graph = ProjectKnowledgeGraph()
        
        class_id = graph.add_node(NodeType.CLASS, "Service", "Service")
        method_id = graph.add_node(NodeType.METHOD, "method", "method")
        
        graph.add_edge(class_id, method_id, RelationType.CONTAINS)
        
        neighbors = graph.get_neighbors(class_id)
        
        assert len(neighbors) == 1
        assert neighbors[0].node_id == method_id

    def test_find_path_direct(self):
        """Test finding direct path."""
        graph = ProjectKnowledgeGraph()
        
        node1_id = graph.add_node(NodeType.CLASS, "A", "A")
        node2_id = graph.add_node(NodeType.CLASS, "B", "B")
        
        graph.add_edge(node1_id, node2_id, RelationType.DEPENDS_ON)
        
        path = graph.find_path(node1_id, node2_id)
        
        assert path is not None
        assert len(path) == 2
        assert path[0] == node1_id
        assert path[1] == node2_id

    def test_find_path_indirect(self):
        """Test finding indirect path."""
        graph = ProjectKnowledgeGraph()
        
        node1_id = graph.add_node(NodeType.CLASS, "A", "A")
        node2_id = graph.add_node(NodeType.CLASS, "B", "B")
        node3_id = graph.add_node(NodeType.CLASS, "C", "C")
        
        graph.add_edge(node1_id, node2_id, RelationType.DEPENDS_ON)
        graph.add_edge(node2_id, node3_id, RelationType.DEPENDS_ON)
        
        path = graph.find_path(node1_id, node3_id)
        
        assert path is not None
        assert len(path) == 3

    def test_find_path_no_path(self):
        """Test finding path when none exists."""
        graph = ProjectKnowledgeGraph()
        
        node1_id = graph.add_node(NodeType.CLASS, "A", "A")
        node2_id = graph.add_node(NodeType.CLASS, "B", "B")
        
        path = graph.find_path(node1_id, node2_id)
        
        assert path is None

    def test_get_class_dependencies(self):
        """Test getting class dependencies."""
        graph = ProjectKnowledgeGraph()
        
        class_id = graph.add_node(NodeType.CLASS, "Service", "Service")
        dep_id = graph.add_node(NodeType.CLASS, "Repository", "Repository")
        
        graph.add_edge(class_id, dep_id, RelationType.DEPENDS_ON)
        
        deps = graph.get_class_dependencies(class_id)
        
        assert "depends_on" in deps
        assert len(deps["depends_on"]) == 1

    def test_get_test_coverage_info(self):
        """Test getting test coverage info."""
        graph = ProjectKnowledgeGraph()
        
        class_id = graph.add_node(
            NodeType.CLASS, "Service", "Service",
            properties={"methods": [{"name": "method1"}, {"name": "method2"}]}
        )
        test_id = graph.add_node(NodeType.TEST_CLASS, "ServiceTest", "ServiceTest")
        
        graph.add_edge(test_id, class_id, RelationType.TESTS)
        
        coverage = graph.get_test_coverage_info(class_id)
        
        assert "class_name" in coverage
        assert coverage["class_name"] == "Service"

    def test_add_pattern(self):
        """Test adding a code pattern."""
        graph = ProjectKnowledgeGraph()
        
        pattern_id = graph.add_pattern(
            pattern_type="test_pattern",
            description="Standard unit test pattern",
            examples=["@Test void test() {}"],
            tags=["unit", "junit5"]
        )
        
        assert pattern_id is not None

    def test_find_patterns(self):
        """Test finding patterns."""
        graph = ProjectKnowledgeGraph()
        
        graph.add_pattern("test_pattern", "Test pattern", ["example"], ["test"])
        graph.add_pattern("mock_pattern", "Mock pattern", ["example"], ["mock"])
        
        patterns = graph.find_patterns()
        
        assert len(patterns) >= 2

    def test_analyze_code_structure(self):
        """Test analyzing code structure."""
        graph = ProjectKnowledgeGraph()
        
        source_code = '''
package com.example;

public class UserService {
    private UserRepository repo;
    
    public User getUser(Long id) {
        return repo.findById(id);
    }
    
    public void saveUser(User user) {
        repo.save(user);
    }
}
'''
        
        result = graph.analyze_code_structure(source_code, "UserService.java")
        
        assert "classes" in result
        assert len(result["classes"]) >= 1
        assert "methods" in result

    def test_get_statistics(self):
        """Test getting statistics."""
        graph = ProjectKnowledgeGraph()
        
        graph.add_node(NodeType.CLASS, "Class1", "Class1")
        graph.add_node(NodeType.METHOD, "method1", "method1")
        
        stats = graph.get_statistics()
        
        assert "total_nodes" in stats
        assert stats["total_nodes"] >= 2

    def test_clear(self):
        """Test clearing the graph."""
        graph = ProjectKnowledgeGraph()
        
        graph.add_node(NodeType.CLASS, "Test", "Test")
        graph.clear()
        
        stats = graph.get_statistics()
        assert stats["total_nodes"] == 0


class TestGraphNode:
    """Tests for GraphNode dataclass."""

    def test_node_creation(self):
        """Test node creation."""
        node = GraphNode(
            node_id="test-id",
            node_type=NodeType.CLASS,
            name="UserService",
            full_name="com.example.UserService",
            properties={"is_service": True},
            source_file="UserService.java",
            line_number=10
        )
        
        assert node.node_id == "test-id"
        assert node.node_type == NodeType.CLASS
        assert node.name == "UserService"
        assert node.properties["is_service"] is True

    def test_node_to_dict(self):
        """Test node to dictionary conversion."""
        node = GraphNode(
            node_id="test-id",
            node_type=NodeType.METHOD,
            name="process",
            full_name="Service.process"
        )
        
        d = node.to_dict()
        
        assert d["node_id"] == "test-id"
        assert d["node_type"] == "method"
        assert d["name"] == "process"

    def test_node_from_dict(self):
        """Test node from dictionary."""
        d = {
            "node_id": "test-id",
            "node_type": "class",
            "name": "Test",
            "full_name": "Test",
            "properties": {},
            "source_file": None,
            "line_number": None,
            "created_at": "2024-01-01T00:00:00"
        }
        
        node = GraphNode.from_dict(d)
        
        assert node.node_id == "test-id"
        assert node.node_type == NodeType.CLASS


class TestGraphEdge:
    """Tests for GraphEdge dataclass."""

    def test_edge_creation(self):
        """Test edge creation."""
        edge = GraphEdge(
            edge_id="edge-id",
            source_id="source-id",
            target_id="target-id",
            relation_type=RelationType.CONTAINS,
            properties={"weight": 1.0},
            weight=1.0
        )
        
        assert edge.edge_id == "edge-id"
        assert edge.relation_type == RelationType.CONTAINS
        assert edge.weight == 1.0

    def test_edge_to_dict(self):
        """Test edge to dictionary conversion."""
        edge = GraphEdge(
            edge_id="edge-id",
            source_id="source",
            target_id="target",
            relation_type=RelationType.DEPENDS_ON
        )
        
        d = edge.to_dict()
        
        assert d["edge_id"] == "edge-id"
        assert d["relation_type"] == "depends_on"

    def test_edge_from_dict(self):
        """Test edge from dictionary."""
        d = {
            "edge_id": "edge-id",
            "source_id": "source",
            "target_id": "target",
            "relation_type": "calls",
            "properties": {},
            "weight": 1.0,
            "created_at": "2024-01-01T00:00:00"
        }
        
        edge = GraphEdge.from_dict(d)
        
        assert edge.edge_id == "edge-id"
        assert edge.relation_type == RelationType.CALLS


class TestNodeType:
    """Tests for NodeType enum."""

    def test_node_type_values(self):
        """Test node type enum values."""
        assert NodeType.CLASS.value == "class"
        assert NodeType.METHOD.value == "method"
        assert NodeType.FIELD.value == "field"
        assert NodeType.TEST_CLASS.value == "test_class"
        assert NodeType.TEST_METHOD.value == "test_method"


class TestRelationType:
    """Tests for RelationType enum."""

    def test_relation_type_values(self):
        """Test relation type enum values."""
        assert RelationType.CONTAINS.value == "contains"
        assert RelationType.DEPENDS_ON.value == "depends_on"
        assert RelationType.CALLS.value == "calls"
        assert RelationType.TESTS.value == "tests"
        assert RelationType.MOCKS.value == "mocks"


class TestCodePattern:
    """Tests for CodePattern dataclass."""

    def test_pattern_creation(self):
        """Test pattern creation."""
        pattern = CodePattern(
            pattern_id="pattern-123",
            pattern_type="test_pattern",
            description="Standard test pattern",
            examples=["@Test void test() {}"],
            frequency=5,
            confidence=0.9,
            tags=["unit", "junit5"]
        )
        
        assert pattern.pattern_id == "pattern-123"
        assert pattern.frequency == 5
        assert pattern.confidence == 0.9

    def test_pattern_to_dict(self):
        """Test pattern to dictionary conversion."""
        pattern = CodePattern(
            pattern_id="pattern-123",
            pattern_type="test",
            description="Test",
            examples=["example"],
            tags=["tag1"]
        )
        
        d = pattern.to_dict()
        
        assert d["pattern_id"] == "pattern-123"
        assert d["tags"] == ["tag1"]


class TestGraphIntegration:
    """Integration tests for knowledge graph."""

    def test_full_workflow(self):
        """Test full workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            graph = ProjectKnowledgeGraph(db_path=db_path)
            
            service_id = graph.add_node(NodeType.CLASS, "UserService", "com.example.UserService")
            repo_id = graph.add_node(NodeType.CLASS, "UserRepository", "com.example.UserRepository")
            method_id = graph.add_node(NodeType.METHOD, "getUser", "UserService.getUser")
            
            graph.add_edge(service_id, method_id, RelationType.CONTAINS)
            graph.add_edge(service_id, repo_id, RelationType.DEPENDS_ON)
            
            test_id = graph.add_node(NodeType.TEST_CLASS, "UserServiceTest", "UserServiceTest")
            graph.add_edge(test_id, service_id, RelationType.TESTS)
            
            deps = graph.get_class_dependencies(service_id)
            assert len(deps["depends_on"]) == 1
            
            neighbors = graph.get_neighbors(service_id)
            assert len(neighbors) == 2

    def test_code_analysis_workflow(self):
        """Test code analysis workflow."""
        graph = ProjectKnowledgeGraph()
        
        source = '''
package com.example;

@Service
public class OrderService {
    @Autowired
    private OrderRepository orderRepo;
    
    public Order createOrder(OrderRequest request) {
        Order order = new Order(request);
        return orderRepo.save(order);
    }
}
'''
        
        result = graph.analyze_code_structure(source, "OrderService.java")
        
        assert len(result["classes"]) >= 1
        assert len(result["node_ids"]) >= 1
