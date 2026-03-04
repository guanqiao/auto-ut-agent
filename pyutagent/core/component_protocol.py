"""组件协议 - 标准化组件接口"""
from abc import ABC, abstractmethod
from typing import List, Any
from enum import Enum
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


class ComponentLifecycle(Enum):
    """组件生命周期状态"""
    CREATED = "created"
    INITIALIZED = "initialized"
    RUNNING = "running"
    STOPPED = "stopped"
    SHUTDOWN = "shutdown"


@dataclass
class ComponentCapability:
    """组件能力定义"""
    name: str
    description: str
    metadata: dict = field(default_factory=dict)


class IAgentComponent(ABC):
    """组件协议"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """组件名称"""
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """初始化组件"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """关闭组件"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[ComponentCapability]:
        """获取组件能力列表"""
        pass
    
    @property
    @abstractmethod
    def lifecycle_state(self) -> ComponentLifecycle:
        """当前生命周期状态"""
        pass
    
    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """是否已初始化"""
        pass


class ComponentBase(IAgentComponent, ABC):
    """组件基类，提供通用实现"""
    
    def __init__(self, name: str):
        self._name = name
        self._lifecycle_state = ComponentLifecycle.CREATED
        self._is_initialized = False
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def lifecycle_state(self) -> ComponentLifecycle:
        return self._lifecycle_state
    
    @property
    def is_initialized(self) -> bool:
        return self._is_initialized
    
    async def initialize(self) -> None:
        """初始化组件"""
        if self._is_initialized:
            logger.debug(f"Component {self._name} already initialized")
            return
        
        logger.info(f"Initializing component: {self._name}")
        await self._do_initialize()
        self._is_initialized = True
        self._lifecycle_state = ComponentLifecycle.INITIALIZED
        logger.info(f"Component {self._name} initialized successfully")
    
    async def shutdown(self) -> None:
        """关闭组件"""
        if not self._is_initialized:
            logger.debug(f"Component {self._name} not initialized, skipping shutdown")
            return
        
        logger.info(f"Shutting down component: {self._name}")
        await self._do_shutdown()
        self._is_initialized = False
        self._lifecycle_state = ComponentLifecycle.SHUTDOWN
        logger.info(f"Component {self._name} shut down successfully")
    
    async def start(self) -> None:
        """启动组件"""
        if not self._is_initialized:
            raise RuntimeError(f"Component {self._name} not initialized")
        
        logger.info(f"Starting component: {self._name}")
        await self._do_start()
        self._lifecycle_state = ComponentLifecycle.RUNNING
        logger.info(f"Component {self._name} started")
    
    async def stop(self) -> None:
        """停止组件"""
        logger.info(f"Stopping component: {self._name}")
        await self._do_stop()
        self._lifecycle_state = ComponentLifecycle.STOPPED
        logger.info(f"Component {self._name} stopped")
    
    @abstractmethod
    def get_capabilities(self) -> List[ComponentCapability]:
        """获取组件能力"""
        pass
    
    @abstractmethod
    async def _do_initialize(self) -> None:
        """执行初始化"""
        pass
    
    @abstractmethod
    async def _do_shutdown(self) -> None:
        """执行关闭"""
        pass
    
    @abstractmethod
    async def _do_start(self) -> None:
        """执行启动"""
        pass
    
    @abstractmethod
    async def _do_stop(self) -> None:
        """执行停止"""
        pass
