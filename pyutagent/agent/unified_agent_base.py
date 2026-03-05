"""Unified Agent Base - 统一的Agent基类

整合所有Agent类的公共功能：
- 状态管理
- 生命周期控制
- 进度报告
- 错误处理
- 配置管理
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic
from uuid import uuid4

logger = logging.getLogger(__name__)

T = TypeVar('T')


class AgentState(Enum):
    """Agent状态枚举"""
    IDLE = auto()
    INITIALIZING = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPING = auto()
    STOPPED = auto()
    FAILED = auto()
    COMPLETED = auto()


class AgentCapability(Enum):
    """Agent能力枚举"""
    CODE_GENERATION = "code_generation"
    TEST_GENERATION = "test_generation"
    CODE_REVIEW = "code_review"
    REFACTORING = "refactoring"
    DEBUGGING = "debugging"
    ANALYSIS = "analysis"
    DOCUMENTATION = "documentation"
    PLANNING = "planning"
    EXECUTION = "execution"
    COORDINATION = "coordination"


@dataclass
class AgentConfig:
    """Agent配置基类"""
    name: str = "Agent"
    agent_type: str = "generic"
    description: str = ""
    capabilities: List[AgentCapability] = field(default_factory=list)
    max_iterations: int = 10
    timeout: int = 300
    max_retries: int = 3
    auto_restart: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "agent_type": self.agent_type,
            "description": self.description,
            "capabilities": [c.value for c in self.capabilities],
            "max_iterations": self.max_iterations,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "auto_restart": self.auto_restart,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        """从字典创建"""
        capabilities = [
            AgentCapability(c) for c in data.get("capabilities", [])
            if c in [e.value for e in AgentCapability]
        ]
        return cls(
            name=data.get("name", "Agent"),
            agent_type=data.get("agent_type", "generic"),
            description=data.get("description", ""),
            capabilities=capabilities,
            max_iterations=data.get("max_iterations", 10),
            timeout=data.get("timeout", 300),
            max_retries=data.get("max_retries", 3),
            auto_restart=data.get("auto_restart", True),
            metadata=data.get("metadata", {})
        )


@dataclass
class AgentResult(Generic[T]):
    """Agent执行结果"""
    success: bool
    output: Optional[T] = None
    error: Optional[str] = None
    state: AgentState = AgentState.COMPLETED
    iterations: int = 0
    execution_time_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "state": self.state.name,
            "iterations": self.iterations,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata
        }


@dataclass
class ProgressUpdate:
    """进度更新"""
    agent_id: str
    progress: float
    status: str
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedAgentBase(ABC):
    """统一的Agent基类
    
    提供所有Agent的公共功能：
    - 唯一标识
    - 状态管理
    - 生命周期控制（启动、停止、暂停、恢复）
    - 进度报告
    - 错误处理
    - 配置管理
    """
    
    def __init__(
        self,
        config: AgentConfig,
        progress_callback: Optional[Callable[[ProgressUpdate], None]] = None,
        result_callback: Optional[Callable[[AgentResult], None]] = None
    ):
        """初始化Agent
        
        Args:
            config: Agent配置
            progress_callback: 进度回调
            result_callback: 结果回调
        """
        self.id = str(uuid4())
        self.config = config
        self.name = config.name
        self.agent_type = config.agent_type
        
        self._state = AgentState.IDLE
        self._state_history: List[Dict[str, Any]] = []
        self._current_iteration = 0
        self._max_iterations = config.max_iterations
        
        self._stop_requested = False
        self._pause_requested = False
        self._terminated = False
        self._pause_event = asyncio.Event()
        self._pause_event.set()
        
        self._progress_callback = progress_callback
        self._result_callback = result_callback
        
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        
        self._execution_history: List[AgentResult] = []
        
        logger.info(f"[UnifiedAgentBase:{self.id}] Initialized: {self.name}")
    
    @property
    def state(self) -> AgentState:
        """获取当前状态"""
        return self._state
    
    @state.setter
    def state(self, value: AgentState) -> None:
        """设置状态"""
        self._state = value
        self._record_state(value)
    
    @property
    def is_running(self) -> bool:
        """是否正在运行"""
        return self._state == AgentState.RUNNING
    
    @property
    def is_paused(self) -> bool:
        """是否暂停"""
        return self._state == AgentState.PAUSED
    
    @property
    def is_stopped(self) -> bool:
        """是否已停止"""
        return self._state in [AgentState.STOPPED, AgentState.FAILED, AgentState.COMPLETED]
    
    @property
    def capabilities(self) -> List[AgentCapability]:
        """获取能力列表"""
        return self.config.capabilities
    
    def has_capability(self, capability: AgentCapability) -> bool:
        """检查是否具有某能力"""
        return capability in self.config.capabilities
    
    def _record_state(self, state: AgentState, message: str = "") -> None:
        """记录状态变化"""
        entry = {
            "state": state.name,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "iteration": self._current_iteration
        }
        self._state_history.append(entry)
        logger.debug(f"[UnifiedAgentBase:{self.id}] State: {state.name} - {message}")
    
    def _update_progress(self, progress: float, status: str, message: str) -> None:
        """更新进度"""
        if self._progress_callback:
            update = ProgressUpdate(
                agent_id=self.id,
                progress=progress,
                status=status,
                message=message
            )
            self._progress_callback(update)
    
    def _should_continue(self) -> bool:
        """检查是否应该继续执行"""
        if self._stop_requested or self._terminated:
            return False
        if self._current_iteration >= self._max_iterations:
            return False
        return True
    
    async def _check_pause(self) -> None:
        """检查暂停状态"""
        if self._pause_requested or not self._pause_event.is_set():
            self.state = AgentState.PAUSED
            self._update_progress(0, "paused", "Execution paused")
            await self._pause_event.wait()
            self.state = AgentState.RUNNING
    
    def start(self) -> bool:
        """启动Agent"""
        if self._state != AgentState.IDLE:
            logger.warning(f"[UnifiedAgentBase:{self.id}] Cannot start from state: {self._state}")
            return False
        
        self._stop_requested = False
        self._terminated = False
        self._pause_event.set()
        self._current_iteration = 0
        self.state = AgentState.INITIALIZING
        
        logger.info(f"[UnifiedAgentBase:{self.id}] Started")
        return True
    
    def stop(self) -> bool:
        """停止Agent"""
        self._stop_requested = True
        self.state = AgentState.STOPPING
        logger.info(f"[UnifiedAgentBase:{self.id}] Stop requested")
        return True
    
    def pause(self) -> bool:
        """暂停Agent"""
        if self._state != AgentState.RUNNING:
            return False
        
        self._pause_requested = True
        self._pause_event.clear()
        logger.info(f"[UnifiedAgentBase:{self.id}] Pause requested")
        return True
    
    def resume(self) -> bool:
        """恢复Agent"""
        if self._state != AgentState.PAUSED:
            return False
        
        self._pause_requested = False
        self._pause_event.set()
        self.state = AgentState.RUNNING
        logger.info(f"[UnifiedAgentBase:{self.id}] Resumed")
        return True
    
    def terminate(self) -> bool:
        """终止Agent"""
        self._terminated = True
        self._stop_requested = True
        self._pause_event.set()
        self.state = AgentState.STOPPED
        logger.info(f"[UnifiedAgentBase:{self.id}] Terminated")
        return True
    
    def reset(self) -> None:
        """重置Agent"""
        self._stop_requested = False
        self._terminated = False
        self._pause_requested = False
        self._pause_event.set()
        self._current_iteration = 0
        self._state = AgentState.IDLE
        self._state_history.clear()
        logger.info(f"[UnifiedAgentBase:{self.id}] Reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.agent_type,
            "state": self._state.name,
            "current_iteration": self._current_iteration,
            "max_iterations": self._max_iterations,
            "execution_count": len(self._execution_history),
            "capabilities": [c.value for c in self.capabilities]
        }
    
    def get_state_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取状态历史"""
        return self._state_history[-limit:]
    
    @abstractmethod
    async def execute(self, task: Any) -> AgentResult:
        """执行任务
        
        Args:
            task: 要执行的任务
            
        Returns:
            AgentResult: 执行结果
        """
        pass
    
    async def run(self, task: Any) -> AgentResult:
        """运行Agent（带生命周期管理）
        
        Args:
            task: 要执行的任务
            
        Returns:
            AgentResult: 执行结果
        """
        import time
        
        self._start_time = time.time()
        
        try:
            self.start()
            self.state = AgentState.RUNNING
            
            result = await self.execute(task)
            
            if self._stop_requested:
                self.state = AgentState.STOPPED
            elif result.success:
                self.state = AgentState.COMPLETED
            else:
                self.state = AgentState.FAILED
            
            self._execution_history.append(result)
            
            if self._result_callback:
                self._result_callback(result)
            
            return result
            
        except Exception as e:
            logger.exception(f"[UnifiedAgentBase:{self.id}] Execution failed: {e}")
            self.state = AgentState.FAILED
            
            result = AgentResult(
                success=False,
                error=str(e),
                state=AgentState.FAILED
            )
            self._execution_history.append(result)
            
            if self._result_callback:
                self._result_callback(result)
            
            return result
            
        finally:
            self._end_time = time.time()
            if self._start_time:
                execution_time_ms = int((self._end_time - self._start_time) * 1000)
                for r in self._execution_history:
                    if r.execution_time_ms == 0:
                        r.execution_time_ms = execution_time_ms


class AgentMixin:
    """Agent功能混入类
    
    提供可复用的功能模块：
    - 错误恢复
    - 重试逻辑
    - 超时处理
    - 日志记录
    """
    
    async def with_retry(
        self,
        func: Callable,
        max_retries: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0
    ) -> Any:
        """带重试的执行"""
        last_error = None
        current_delay = delay
        
        for attempt in range(max_retries):
            try:
                return await func()
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
        
        raise last_error
    
    async def with_timeout(
        self,
        func: Callable,
        timeout: float
    ) -> Any:
        """带超时的执行"""
        return await asyncio.wait_for(func(), timeout=timeout)


def create_agent_config(
    name: str,
    agent_type: str,
    description: str = "",
    capabilities: Optional[List[AgentCapability]] = None,
    **kwargs
) -> AgentConfig:
    """创建Agent配置"""
    return AgentConfig(
        name=name,
        agent_type=agent_type,
        description=description,
        capabilities=capabilities or [],
        **kwargs
    )
