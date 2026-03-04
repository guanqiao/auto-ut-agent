"""状态快照模块 - 支持状态快照和恢复"""
import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

from .state_store import AgentState

logger = logging.getLogger(__name__)


@dataclass
class StateSnapshot:
    """状态快照"""
    snapshot_id: str
    state_id: str
    timestamp: datetime
    state_data: Dict[str, Any]
    label: Optional[str] = None
    description: Optional[str] = None
    size_bytes: int = 0
    
    @classmethod
    def create(
        cls,
        state: AgentState,
        state_id: Optional[str] = None,
        label: Optional[str] = None,
        description: Optional[str] = None
    ) -> 'StateSnapshot':
        """创建状态快照"""
        snapshot_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        state_data = {
            'lifecycle_state': state.lifecycle_state.value,
            'current_phase': state.current_phase,
            'current_iteration': state.current_iteration,
            'target_coverage': state.target_coverage,
            'current_coverage': state.current_coverage,
            'working_memory': state.working_memory,
            'error_state': state.error_state,
            'metrics': state.metrics
        }
        
        state_data_json = json.dumps(state_data)
        size_bytes = len(state_data_json.encode('utf-8'))
        
        return cls(
            snapshot_id=snapshot_id,
            state_id=state_id or str(uuid.uuid4()),
            timestamp=timestamp,
            state_data=state_data,
            label=label,
            description=description,
            size_bytes=size_bytes
        )
    
    def restore(self) -> AgentState:
        """从快照恢复状态"""
        from .state_store import LifecycleState
        
        return AgentState(
            lifecycle_state=LifecycleState(self.state_data['lifecycle_state']),
            current_phase=self.state_data['current_phase'],
            current_iteration=self.state_data['current_iteration'],
            target_coverage=self.state_data['target_coverage'],
            current_coverage=self.state_data['current_coverage'],
            working_memory=self.state_data['working_memory'].copy(),
            error_state=self.state_data['error_state'].copy(),
            metrics=self.state_data['metrics'].copy()
        )
    
    @classmethod
    def compare(cls, snapshot1: 'StateSnapshot', snapshot2: 'StateSnapshot') -> Dict[str, Any]:
        """比较两个快照的差异"""
        diff = {}
        
        for key in snapshot1.state_data.keys():
            val1 = snapshot1.state_data[key]
            val2 = snapshot2.state_data.get(key)
            
            if val1 != val2:
                diff[key] = {
                    'old': val1,
                    'new': val2
                }
        
        return diff
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'snapshot_id': self.snapshot_id,
            'state_id': self.state_id,
            'timestamp': self.timestamp.isoformat(),
            'state_data': self.state_data,
            'label': self.label,
            'description': self.description,
            'size_bytes': self.size_bytes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateSnapshot':
        """从字典创建"""
        return cls(
            snapshot_id=data['snapshot_id'],
            state_id=data['state_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            state_data=data['state_data'],
            label=data.get('label'),
            description=data.get('description'),
            size_bytes=data.get('size_bytes', 0)
        )


class SnapshotManager:
    """快照管理器"""
    
    def __init__(self, storage_path: Optional[str] = None, auto_snapshot: bool = False):
        self.storage_path = storage_path
        self.auto_snapshot = auto_snapshot
        self._snapshots: Dict[str, StateSnapshot] = {}
        self._lock = asyncio.Lock()
        
        if storage_path:
            os.makedirs(storage_path, exist_ok=True)
            self._load_snapshots_from_disk()
    
    def _get_storage_file(self) -> str:
        """获取存储文件路径"""
        if self.storage_path:
            return os.path.join(self.storage_path, "snapshots.json")
        return None
    
    def _load_snapshots_from_disk(self):
        """从磁盘加载快照"""
        storage_file = self._get_storage_file()
        if storage_file and os.path.exists(storage_file):
            try:
                with open(storage_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._snapshots = {
                        k: StateSnapshot.from_dict(v) for k, v in data.items()
                    }
                logger.debug(f"Loaded {len(self._snapshots)} snapshots from disk")
            except Exception as e:
                logger.error(f"Failed to load snapshots: {e}")
    
    def _save_snapshots_to_disk(self):
        """保存快照到磁盘"""
        storage_file = self._get_storage_file()
        if storage_file:
            try:
                with open(storage_file, 'w', encoding='utf-8') as f:
                    data = {k: v.to_dict() for k, v in self._snapshots.items()}
                    json.dump(data, f, indent=2)
                logger.debug("Saved snapshots to disk")
            except Exception as e:
                logger.error(f"Failed to save snapshots: {e}")
    
    async def create_snapshot(
        self,
        state_id: str,
        state: AgentState,
        label: Optional[str] = None,
        description: Optional[str] = None
    ) -> str:
        """创建快照"""
        async with self._lock:
            snapshot = StateSnapshot.create(
                state=state,
                state_id=state_id,
                label=label,
                description=description
            )
            
            self._snapshots[snapshot.snapshot_id] = snapshot
            self._save_snapshots_to_disk()
            
            logger.debug(f"Created snapshot {snapshot.snapshot_id} for {state_id}")
            return snapshot.snapshot_id
    
    async def restore_snapshot(self, snapshot_id: str) -> AgentState:
        """恢复快照"""
        async with self._lock:
            if snapshot_id not in self._snapshots:
                raise ValueError(f"Snapshot {snapshot_id} not found")
            
            snapshot = self._snapshots[snapshot_id]
            restored_state = snapshot.restore()
            
            logger.debug(f"Restored snapshot {snapshot_id}")
            return restored_state
    
    async def list_snapshots(self, state_id: Optional[str] = None) -> List[StateSnapshot]:
        """列出快照"""
        async with self._lock:
            if state_id:
                return [s for s in self._snapshots.values() if s.state_id == state_id]
            return list(self._snapshots.values())
    
    async def delete_snapshot(self, snapshot_id: str) -> bool:
        """删除快照"""
        async with self._lock:
            if snapshot_id in self._snapshots:
                del self._snapshots[snapshot_id]
                self._save_snapshots_to_disk()
                logger.debug(f"Deleted snapshot {snapshot_id}")
                return True
            return False
    
    async def get_snapshot(self, snapshot_id: str) -> Optional[StateSnapshot]:
        """获取快照"""
        async with self._lock:
            return self._snapshots.get(snapshot_id)
    
    async def cleanup(self):
        """清理资源"""
        self._save_snapshots_to_disk()
        logger.debug("SnapshotManager cleanup completed")
