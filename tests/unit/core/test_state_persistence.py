"""状态持久化和快照测试"""
import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from pyutagent.core.state_store import AgentState, LifecycleState
from pyutagent.core.state_persistence import StatePersistence, SQLitePersistence
from pyutagent.core.state_snapshot import StateSnapshot, SnapshotManager


class TestStatePersistence:
    """状态持久化测试"""
    
    def test_create_sqlite_persistence(self):
        """测试创建 SQLite 状态持久化实例"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            persistence = SQLitePersistence(db_path)
            assert persistence is not None
    
    @pytest.mark.asyncio
    async def test_save_and_load_state(self):
        """测试保存和加载状态"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_state.db")
            persistence = SQLitePersistence(db_path)
            
            state = AgentState(
                lifecycle_state=LifecycleState.RUNNING,
                current_phase="TEST",
                current_iteration=5,
                target_coverage=0.9,
                current_coverage=0.75
            )
            
            await persistence.save_state("agent_1", state)
            loaded_state = await persistence.load_state("agent_1")
            
            assert loaded_state.lifecycle_state == LifecycleState.RUNNING
            assert loaded_state.current_phase == "TEST"
            assert loaded_state.current_iteration == 5
            assert loaded_state.target_coverage == 0.9
            assert loaded_state.current_coverage == 0.75
            
            await persistence.close()
    
    @pytest.mark.asyncio
    async def test_update_existing_state(self):
        """测试更新已存在的状态"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_state.db")
            persistence = SQLitePersistence(db_path)
            
            state1 = AgentState(
                lifecycle_state=LifecycleState.IDLE,
                current_iteration=1
            )
            
            await persistence.save_state("agent_1", state1)
            
            state2 = AgentState(
                lifecycle_state=LifecycleState.RUNNING,
                current_iteration=2
            )
            
            await persistence.save_state("agent_1", state2)
            loaded_state = await persistence.load_state("agent_1")
            
            assert loaded_state.lifecycle_state == LifecycleState.RUNNING
            assert loaded_state.current_iteration == 2
            
            await persistence.close()
    
    @pytest.mark.asyncio
    async def test_load_nonexistent_state(self):
        """测试加载不存在的状态"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_state.db")
            persistence = SQLitePersistence(db_path)
            
            loaded_state = await persistence.load_state("nonexistent")
            
            assert loaded_state is None
            
            await persistence.close()
    
    @pytest.mark.asyncio
    async def test_delete_state(self):
        """测试删除状态"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_state.db")
            persistence = SQLitePersistence(db_path)
            
            state = AgentState(lifecycle_state=LifecycleState.IDLE)
            await persistence.save_state("agent_1", state)
            
            await persistence.delete_state("agent_1")
            loaded_state = await persistence.load_state("agent_1")
            
            assert loaded_state is None
            
            await persistence.close()
    
    @pytest.mark.asyncio
    async def test_list_all_states(self):
        """测试列出所有状态"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_state.db")
            persistence = SQLitePersistence(db_path)
            
            for i in range(3):
                state = AgentState(
                    lifecycle_state=LifecycleState.IDLE,
                    current_iteration=i
                )
                await persistence.save_state(f"agent_{i}", state)
            
            all_ids = await persistence.list_all_state_ids()
            
            assert len(all_ids) == 3
            assert "agent_0" in all_ids
            assert "agent_1" in all_ids
            assert "agent_2" in all_ids
            
            await persistence.close()
    
    @pytest.mark.asyncio
    async def test_persist_complex_state(self):
        """测试持久化复杂状态"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_state.db")
            persistence = SQLitePersistence(db_path)
            
            state = AgentState(
                lifecycle_state=LifecycleState.PAUSED,
                current_phase="COMPLEX_TEST",
                working_memory={
                    "key1": "value1",
                    "key2": [1, 2, 3],
                    "key3": {"nested": "dict"}
                },
                error_state={
                    "last_error": "Test error",
                    "error_count": 5
                },
                metrics={
                    "execution_time": 123.45,
                    "memory_usage": 1024
                }
            )
            
            await persistence.save_state("agent_complex", state)
            loaded_state = await persistence.load_state("agent_complex")
            
            assert loaded_state.working_memory["key1"] == "value1"
            assert loaded_state.working_memory["key2"] == [1, 2, 3]
            assert loaded_state.working_memory["key3"]["nested"] == "dict"
            assert loaded_state.error_state["last_error"] == "Test error"
            assert loaded_state.metrics["execution_time"] == 123.45
            
            await persistence.close()


class TestStateSnapshot:
    """状态快照测试"""
    
    def test_create_snapshot(self):
        """测试创建快照"""
        state = AgentState(
            lifecycle_state=LifecycleState.RUNNING,
            current_iteration=10,
            current_coverage=0.85
        )
        
        snapshot = StateSnapshot.create(state)
        
        assert snapshot is not None
        assert snapshot.state_id is not None
        assert snapshot.timestamp is not None
        assert snapshot.state_data is not None
    
    def test_snapshot_preserves_state(self):
        """测试快照保留状态数据"""
        state = AgentState(
            lifecycle_state=LifecycleState.PAUSED,
            current_phase="SNAPSHOT_TEST",
            current_iteration=7,
            target_coverage=0.95,
            working_memory={"test": "data"}
        )
        
        snapshot = StateSnapshot.create(state)
        restored_state = snapshot.restore()
        
        assert restored_state.lifecycle_state == LifecycleState.PAUSED
        assert restored_state.current_phase == "SNAPSHOT_TEST"
        assert restored_state.current_iteration == 7
        assert restored_state.target_coverage == 0.95
        assert restored_state.working_memory["test"] == "data"
    
    def test_snapshot_metadata(self):
        """测试快照元数据"""
        state = AgentState(lifecycle_state=LifecycleState.IDLE)
        
        snapshot = StateSnapshot.create(
            state,
            label="test_label",
            description="Test snapshot description"
        )
        
        assert snapshot.label == "test_label"
        assert snapshot.description == "Test snapshot description"
        assert snapshot.size_bytes > 0
    
    def test_snapshot_comparison(self):
        """测试快照比较"""
        state1 = AgentState(
            lifecycle_state=LifecycleState.IDLE,
            current_iteration=5
        )
        
        state2 = AgentState(
            lifecycle_state=LifecycleState.RUNNING,
            current_iteration=10
        )
        
        snapshot1 = StateSnapshot.create(state1)
        snapshot2 = StateSnapshot.create(state2)
        
        diff = StateSnapshot.compare(snapshot1, snapshot2)
        
        assert "lifecycle_state" in diff
        assert "current_iteration" in diff
    
    @pytest.mark.asyncio
    async def test_snapshot_manager_create(self):
        """测试快照管理器创建快照"""
        manager = SnapshotManager()
        
        state = AgentState(lifecycle_state=LifecycleState.RUNNING)
        snapshot_id = await manager.create_snapshot("test_agent", state)
        
        assert snapshot_id is not None
        
        await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_snapshot_manager_restore(self):
        """测试快照管理器恢复快照"""
        manager = SnapshotManager()
        
        original_state = AgentState(
            lifecycle_state=LifecycleState.PAUSED,
            current_iteration=15
        )
        
        snapshot_id = await manager.create_snapshot("test_agent", original_state)
        restored_state = await manager.restore_snapshot(snapshot_id)
        
        assert restored_state.lifecycle_state == LifecycleState.PAUSED
        assert restored_state.current_iteration == 15
        
        await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_snapshot_manager_list(self):
        """测试快照管理器列出快照"""
        manager = SnapshotManager()
        
        for i in range(5):
            state = AgentState(current_iteration=i)
            await manager.create_snapshot(f"agent_{i}", state)
        
        snapshots = await manager.list_snapshots()
        
        assert len(snapshots) == 5
        
        await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_snapshot_manager_delete(self):
        """测试快照管理器删除快照"""
        manager = SnapshotManager()
        
        state = AgentState(lifecycle_state=LifecycleState.IDLE)
        snapshot_id = await manager.create_snapshot("test_agent", state)
        
        await manager.delete_snapshot(snapshot_id)
        snapshots = await manager.list_snapshots()
        
        assert len(snapshots) == 0
        
        await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_auto_snapshot_on_state_change(self):
        """测试状态变化时自动创建快照"""
        manager = SnapshotManager(auto_snapshot=True)
        
        state = AgentState(lifecycle_state=LifecycleState.IDLE)
        await manager.create_snapshot("agent_1", state)
        
        state.lifecycle_state = LifecycleState.RUNNING
        await manager.create_snapshot("agent_1", state)
        
        state.lifecycle_state = LifecycleState.PAUSED
        await manager.create_snapshot("agent_1", state)
        
        snapshots = await manager.list_snapshots()
        
        assert len(snapshots) >= 2
        
        await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_snapshot_restore_error_handling(self):
        """测试快照恢复错误处理"""
        manager = SnapshotManager()
        
        with pytest.raises(ValueError):
            await manager.restore_snapshot("nonexistent_snapshot_id")
        
        await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_snapshot_persistence_across_sessions(self):
        """测试快照在会话间的持久性"""
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_dir = os.path.join(temp_dir, "snapshots")
            
            manager1 = SnapshotManager(storage_path=snapshot_dir)
            state = AgentState(
                lifecycle_state=LifecycleState.RUNNING,
                current_iteration=100
            )
            snapshot_id = await manager1.create_snapshot("persistent_agent", state)
            await manager1.cleanup()
            
            manager2 = SnapshotManager(storage_path=snapshot_dir)
            restored_state = await manager2.restore_snapshot(snapshot_id)
            
            assert restored_state.lifecycle_state == LifecycleState.RUNNING
            assert restored_state.current_iteration == 100
            
            await manager2.cleanup()
