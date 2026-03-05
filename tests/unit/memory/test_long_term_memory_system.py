"""Tests for the long-term memory system."""

import pytest
import tempfile
import os
from datetime import datetime

from pyutagent.memory import (
    LongTermMemory,
    EpisodicMemory,
    SemanticMemory,
    ProceduralMemory,
    KnowledgeGraph,
    Episode,
    Concept,
    Skill,
    EntityType,
    create_long_term_memory,
    create_episodic_memory,
    create_semantic_memory,
    create_procedural_memory,
    create_knowledge_graph,
)
from pyutagent.memory.knowledge_graph import RelationType


class TestEpisodicMemory:
    """Test episodic memory functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        try:
            os.unlink(db_path)
        except:
            pass
    
    @pytest.mark.asyncio
    async def test_record_and_retrieve_episode(self, temp_db):
        """Test recording and retrieving an episode."""
        memory = EpisodicMemory(temp_db)
        
        episode = Episode(
            episode_id="test-1",
            project="test-project",
            task_type="test-generation",
            task_description="Generate tests for UserService",
            steps=[{"tool": "analyze", "result": "success"}],
            outcome="success",
            duration_seconds=120.0,
            lessons=["Use mocking for external dependencies"],
            timestamp=datetime.now()
        )
        
        await memory.record_episode(episode)
        
        # Retrieve by search
        results = await memory.search_similar("UserService")
        assert len(results) == 1
        assert results[0].task_description == "Generate tests for UserService"
        
        memory.close()
    
    @pytest.mark.asyncio
    async def test_project_summary(self, temp_db):
        """Test getting project summary."""
        memory = EpisodicMemory(temp_db)
        
        # Record multiple episodes
        for i in range(3):
            episode = Episode(
                episode_id=f"test-{i}",
                project="test-project",
                task_type="test-generation",
                task_description=f"Task {i}",
                steps=[],
                outcome="success" if i < 2 else "failed",
                duration_seconds=100.0,
                lessons=[],
                timestamp=datetime.now()
            )
            await memory.record_episode(episode)
        
        summary = await memory.get_project_summary("test-project")
        assert summary.total_episodes == 3
        assert summary.success_count == 2
        assert summary.failure_count == 1
        
        memory.close()


class TestSemanticMemory:
    """Test semantic memory functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        try:
            os.unlink(db_path)
        except:
            pass
    
    @pytest.mark.asyncio
    async def test_learn_and_query_concept(self, temp_db):
        """Test learning and querying a concept."""
        memory = SemanticMemory(temp_db)
        
        concept = Concept(
            concept_id="concept-1",
            name="Dependency Injection",
            category="design_pattern",
            description="A design pattern for loose coupling",
            examples=["@Autowired annotation"],
            related_concepts=["Inversion of Control"],
            source="documentation",
            confidence=0.95,
            usage_count=0,
            last_accessed=datetime.now(),
            created_at=datetime.now(),
            metadata={}
        )
        
        await memory.learn_concept(concept)
        
        # Query the concept
        results = await memory.query_concepts("Dependency Injection")
        assert len(results) == 1
        assert results[0].name == "Dependency Injection"
        
        memory.close()


class TestProceduralMemory:
    """Test procedural memory functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        try:
            os.unlink(db_path)
        except:
            pass
    
    @pytest.mark.asyncio
    async def test_learn_and_retrieve_skill(self, temp_db):
        """Test learning and retrieving a skill."""
        memory = ProceduralMemory(temp_db)
        
        strategy = {
            "name": "Mockito Setup Strategy",
            "steps": ["Add dependency", "Create mock", "Inject mock"]
        }
        
        skill = await memory.learn(
            task_type="mockito-setup",
            strategy=strategy,
            success=True,
            duration_seconds=60.0
        )
        
        assert skill.name == "Mockito Setup Strategy"
        assert skill.success_rate == 1.0
        
        # Retrieve skill
        skills = await memory.retrieve("mockito-setup")
        assert len(skills) == 1
        assert skills[0].name == "Mockito Setup Strategy"
        
        memory.close()


class TestKnowledgeGraph:
    """Test knowledge graph functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        try:
            os.unlink(db_path)
        except:
            pass
    
    def test_add_and_get_entity(self, temp_db):
        """Test adding and retrieving an entity."""
        kg = KnowledgeGraph(temp_db)
        
        entity_id = kg.add_entity(
            entity_type=EntityType.CLASS,
            name="UserService",
            full_name="com.example.UserService",
            source_project="test-project"
        )
        
        entity = kg.get_entity(entity_id)
        assert entity is not None
        assert entity.name == "UserService"
        assert entity.entity_type == EntityType.CLASS
        
    def test_add_and_find_relation(self, temp_db):
        """Test adding and finding relations."""
        kg = KnowledgeGraph(temp_db)
        
        # Add entities
        class_id = kg.add_entity(
            entity_type=EntityType.CLASS,
            name="UserService",
            full_name="com.example.UserService"
        )
        
        method_id = kg.add_entity(
            entity_type=EntityType.METHOD,
            name="getUser",
            full_name="com.example.UserService.getUser"
        )
        
        # Add relation
        relation_id = kg.add_relation(
            source_id=class_id,
            target_id=method_id,
            relation_type=RelationType.CONTAINS
        )
        
        # Find relations
        outgoing = kg.get_outgoing_relations(class_id)
        assert len(outgoing) == 1
        assert outgoing[0].relation_type == RelationType.CONTAINS
        
    def test_find_path(self, temp_db):
        """Test finding a path between entities."""
        kg = KnowledgeGraph(temp_db)
        
        # Create chain: A -> B -> C
        a_id = kg.add_entity(EntityType.CLASS, "A", "com.example.A")
        b_id = kg.add_entity(EntityType.CLASS, "B", "com.example.B")
        c_id = kg.add_entity(EntityType.CLASS, "C", "com.example.C")
        
        kg.add_relation(a_id, b_id, RelationType.DEPENDS_ON)
        kg.add_relation(b_id, c_id, RelationType.DEPENDS_ON)
        
        # Find path from A to C
        path = kg.find_path(a_id, c_id, max_depth=3)
        assert path is not None
        assert len(path.entities) == 3
        
    def test_find_similar_entities(self, temp_db):
        """Test finding similar entities."""
        kg = KnowledgeGraph(temp_db)
        
        # Create entities with shared relations
        service1 = kg.add_entity(EntityType.CLASS, "UserService", "com.example.UserService")
        service2 = kg.add_entity(EntityType.CLASS, "OrderService", "com.example.OrderService")
        common_dep = kg.add_entity(EntityType.CLASS, "Database", "com.example.Database")
        
        kg.add_relation(service1, common_dep, RelationType.DEPENDS_ON)
        kg.add_relation(service2, common_dep, RelationType.DEPENDS_ON)
        
        # Find similar to service1
        similar = kg.find_similar_entities(service1)
        assert len(similar) > 0
        assert similar[0][0].name == "OrderService"
        
    def test_cross_project_patterns(self, temp_db):
        """Test finding cross-project patterns."""
        kg = KnowledgeGraph(temp_db)
        
        # Add same pattern from different projects
        kg.add_entity(
            EntityType.PATTERN,
            "Singleton",
            "Singleton Pattern",
            source_project="project-a"
        )
        kg.add_entity(
            EntityType.PATTERN,
            "Singleton",
            "Singleton Pattern",
            source_project="project-b"
        )
        
        patterns = kg.find_cross_project_patterns()
        assert len(patterns) == 1
        assert patterns[0]["name"] == "Singleton"
        assert patterns[0]["project_count"] == 2


class TestLongTermMemoryIntegration:
    """Test long-term memory integration."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.mark.asyncio
    async def test_record_experience(self, temp_dir):
        """Test recording an experience."""
        ltm = create_long_term_memory(temp_dir)
        
        episode = await ltm.record_experience(
            project="test-project",
            task_type="test-generation",
            task_description="Generate tests for OrderService",
            steps=[{"tool": "analyze", "result": "success"}],
            outcome="success",
            duration_seconds=180.0,
            lessons=["Use @Mock annotation", "Verify interactions"]
        )
        
        assert episode.outcome == "success"
        assert len(episode.lessons) == 2
        
        ltm.close()
    
    @pytest.mark.asyncio
    async def test_retrieve_relevant_knowledge(self, temp_dir):
        """Test retrieving relevant knowledge."""
        ltm = create_long_term_memory(temp_dir)
        
        # Record experience
        await ltm.record_experience(
            project="test-project",
            task_type="test-generation",
            task_description="Generate tests for UserService with Mockito",
            steps=[{"tool": "mockito"}],
            outcome="success",
            duration_seconds=120.0,
            lessons=["Use @Mock"]
        )
        
        # Retrieve knowledge
        knowledge = await ltm.retrieve_relevant_knowledge(
            query="mockito",
            task_type="test-generation"
        )
        
        assert "episodes" in knowledge
        assert "concepts" in knowledge
        assert "skills" in knowledge
        
        ltm.close()
    
    @pytest.mark.asyncio
    async def test_get_best_practice(self, temp_dir):
        """Test getting best practice."""
        ltm = create_long_term_memory(temp_dir)
        
        # Record multiple experiences
        for i in range(3):
            await ltm.record_experience(
                project="test-project",
                task_type="test-generation",
                task_description=f"test-generation task {i}",  # Include task_type in description for search
                steps=[{"tool": "junit5"}],
                outcome="success",
                duration_seconds=100.0,
                lessons=["Use @DisplayName"]
            )
        
        best_practice = await ltm.get_best_practice("test-generation")
        assert best_practice is not None
        assert best_practice["task_type"] == "test-generation"
        assert best_practice["success_rate"] > 0
        
        ltm.close()
    
    @pytest.mark.asyncio
    async def test_suggest_approach(self, temp_dir):
        """Test suggesting an approach."""
        ltm = create_long_term_memory(temp_dir)
        
        # Record experience
        await ltm.record_experience(
            project="test-project",
            task_type="test-generation",
            task_description="Generate tests for complex service",
            steps=[{"tool": "analyze"}, {"tool": "generate"}],
            outcome="success",
            duration_seconds=200.0,
            lessons=["Analyze dependencies first"]
        )
        
        suggestion = await ltm.suggest_approach(
            task_description="Generate tests for another service",
            task_type="test-generation"
        )
        
        assert "suggested_steps" in suggestion
        assert "confidence" in suggestion
        
        ltm.close()


class TestMemoryFactoryFunctions:
    """Test factory functions."""
    
    def test_create_episodic_memory(self):
        """Test creating episodic memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = create_episodic_memory(tmpdir)
            assert isinstance(memory, EpisodicMemory)
            memory.close()
    
    def test_create_semantic_memory(self):
        """Test creating semantic memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = create_semantic_memory(tmpdir)
            assert isinstance(memory, SemanticMemory)
            memory.close()
    
    def test_create_procedural_memory(self):
        """Test creating procedural memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = create_procedural_memory(tmpdir)
            assert isinstance(memory, ProceduralMemory)
            memory.close()
    
    def test_create_knowledge_graph(self):
        """Test creating knowledge graph."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            kg = create_knowledge_graph(f.name)
            assert isinstance(kg, KnowledgeGraph)
            kg.clear()
            try:
                os.unlink(f.name)
            except:
                pass
    
    def test_create_long_term_memory(self):
        """Test creating long-term memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ltm = create_long_term_memory(tmpdir)
            assert isinstance(ltm, LongTermMemory)
            assert ltm.episodic is not None
            assert ltm.semantic is not None
            assert ltm.procedural is not None
            ltm.close()
