"""状态持久化模块 - 支持 SQLite 存储"""
import asyncio
import json
import sqlite3
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from dataclasses import asdict
import logging

from .state_store import AgentState, LifecycleState

logger = logging.getLogger(__name__)


class StatePersistence(ABC):
    """状态持久化抽象基类"""
    
    @abstractmethod
    async def save_state(self, state_id: str, state: AgentState) -> None:
        """保存状态"""
        pass
    
    @abstractmethod
    async def load_state(self, state_id: str) -> Optional[AgentState]:
        """加载状态"""
        pass
    
    @abstractmethod
    async def delete_state(self, state_id: str) -> None:
        """删除状态"""
        pass
    
    @abstractmethod
    async def list_all_state_ids(self) -> List[str]:
        """列出所有状态 ID"""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """关闭连接"""
        pass


class SQLitePersistence(StatePersistence):
    """SQLite 状态持久化实现"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = asyncio.Lock()
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_states (
                state_id TEXT PRIMARY KEY,
                lifecycle_state TEXT NOT NULL,
                current_phase TEXT,
                current_iteration INTEGER,
                target_coverage REAL,
                current_coverage REAL,
                working_memory TEXT,
                error_state TEXT,
                metrics TEXT,
                state_history TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _serialize_state(self, state: AgentState) -> Dict[str, Any]:
        """序列化状态"""
        return {
            'lifecycle_state': state.lifecycle_state.value,
            'current_phase': state.current_phase,
            'current_iteration': state.current_iteration,
            'target_coverage': state.target_coverage,
            'current_coverage': state.current_coverage,
            'working_memory': json.dumps(state.working_memory),
            'error_state': json.dumps(state.error_state),
            'metrics': json.dumps(state.metrics),
            'state_history': json.dumps([
                {
                    'from': self._serialize_state(h['from']) if isinstance(h['from'], AgentState) else h['from'],
                    'to': self._serialize_state(h['to']) if isinstance(h['to'], AgentState) else h['to'],
                    'action': h['action']
                }
                for h in state.state_history
            ])
        }
    
    def _deserialize_state(self, data: Dict[str, Any]) -> AgentState:
        """反序列化状态"""
        if data is None:
            return None
        
        return AgentState(
            lifecycle_state=LifecycleState(data['lifecycle_state']),
            current_phase=data['current_phase'] or "IDLE",
            current_iteration=data['current_iteration'] or 0,
            target_coverage=data['target_coverage'] or 0.0,
            current_coverage=data['current_coverage'] or 0.0,
            working_memory=json.loads(data['working_memory']) if data['working_memory'] else {},
            error_state=json.loads(data['error_state']) if data['error_state'] else {},
            metrics=json.loads(data['metrics']) if data['metrics'] else {},
            state_history=json.loads(data['state_history']) if data['state_history'] else []
        )
    
    async def save_state(self, state_id: str, state: AgentState) -> None:
        """保存状态到数据库"""
        async with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            serialized = self._serialize_state(state)
            
            cursor.execute('''
                INSERT OR REPLACE INTO agent_states 
                (state_id, lifecycle_state, current_phase, current_iteration, 
                 target_coverage, current_coverage, working_memory, error_state, 
                 metrics, state_history, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                state_id,
                serialized['lifecycle_state'],
                serialized['current_phase'],
                serialized['current_iteration'],
                serialized['target_coverage'],
                serialized['current_coverage'],
                serialized['working_memory'],
                serialized['error_state'],
                serialized['metrics'],
                serialized['state_history']
            ))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Saved state for {state_id}")
    
    async def load_state(self, state_id: str) -> Optional[AgentState]:
        """从数据库加载状态"""
        async with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM agent_states WHERE state_id = ?
            ''', (state_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row is None:
                return None
            
            data = {
                'lifecycle_state': row[1],
                'current_phase': row[2],
                'current_iteration': row[3],
                'target_coverage': row[4],
                'current_coverage': row[5],
                'working_memory': row[6],
                'error_state': row[7],
                'metrics': row[8],
                'state_history': row[9]
            }
            
            state = self._deserialize_state(data)
            logger.debug(f"Loaded state for {state_id}")
            return state
    
    async def delete_state(self, state_id: str) -> None:
        """删除状态"""
        async with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM agent_states WHERE state_id = ?', (state_id,))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Deleted state for {state_id}")
    
    async def list_all_state_ids(self) -> List[str]:
        """列出所有状态 ID"""
        async with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT state_id FROM agent_states')
            rows = cursor.fetchall()
            conn.close()
            
            state_ids = [row[0] for row in rows]
            logger.debug(f"Listed {len(state_ids)} state IDs")
            return state_ids
    
    async def close(self) -> None:
        """关闭连接"""
        logger.debug("Closing SQLite persistence")
        pass
