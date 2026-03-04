"""组件发现与注册 - 简化版"""
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Type
import logging

logger = logging.getLogger(__name__)


@dataclass
class ComponentInfo:
    """组件信息"""
    component_id: str
    component_class: Type
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SimpleComponent:
    """简单组件基类"""
    
    def __init__(self):
        self._initialized = False
    
    def initialize(self) -> bool:
        """初始化组件"""
        self._initialized = True
        return True
    
    def shutdown(self) -> bool:
        """关闭组件"""
        self._initialized = False
        return True


class ComponentRegistry:
    """组件注册表"""
    
    def __init__(self):
        self._components: Dict[str, SimpleComponent] = {}
        self._component_info: Dict[str, ComponentInfo] = {}
        self._initialized: Set[str] = set()
    
    def register(
        self,
        component_id: str,
        component_class: Type,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """注册组件"""
        logger.debug(f"Registering component: {component_id}")
        
        # 创建组件实例
        component = component_class()
        
        # 存储组件
        self._components[component_id] = component
        self._component_info[component_id] = ComponentInfo(
            component_id=component_id,
            component_class=component_class,
            dependencies=dependencies or [],
            metadata=metadata or {}
        )
        
        logger.info(f"Component registered: {component_id}")
    
    def unregister(self, component_id: str) -> None:
        """注销组件"""
        if component_id in self._components:
            logger.debug(f"Unregistering component: {component_id}")
            
            # 先关闭组件
            if component_id in self._initialized:
                self.shutdown_component(component_id)
            
            del self._components[component_id]
            del self._component_info[component_id]
            
            logger.info(f"Component unregistered: {component_id}")
    
    def has_component(self, component_id: str) -> bool:
        """检查组件是否存在"""
        return component_id in self._components
    
    def get_component(self, component_id: str) -> Optional[Any]:
        """获取组件实例"""
        if component_id not in self._components:
            logger.warning(f"Component not found: {component_id}")
            return None
        
        return self._components[component_id]
    
    def get_component_info(self, component_id: str) -> Optional[ComponentInfo]:
        """获取组件信息"""
        return self._component_info.get(component_id)
    
    def list_components(self) -> List[str]:
        """列出所有组件 ID"""
        return list(self._components.keys())
    
    def resolve_dependencies(self, component_id: str) -> List[str]:
        """解析组件依赖"""
        if component_id not in self._component_info:
            return []
        
        dependencies = []
        visited = set()
        
        def _resolve(comp_id: str):
            if comp_id in visited:
                raise ValueError(f"Circular dependency detected: {comp_id}")
            
            visited.add(comp_id)
            
            if comp_id not in self._component_info:
                return
            
            info = self._component_info[comp_id]
            for dep in info.dependencies:
                _resolve(dep)
                if dep not in dependencies:
                    dependencies.append(dep)
        
        try:
            _resolve(component_id)
        except ValueError as e:
            logger.error(str(e))
            raise
        
        return dependencies
    
    def initialize_component(self, component_id: str) -> bool:
        """初始化组件"""
        if component_id not in self._components:
            logger.error(f"Cannot initialize non-existent component: {component_id}")
            return False
        
        if component_id in self._initialized:
            logger.debug(f"Component already initialized: {component_id}")
            return True
        
        component = self._components[component_id]
        
        try:
            logger.debug(f"Initializing component: {component_id}")
            result = component.initialize()
            
            if result:
                self._initialized.add(component_id)
                logger.info(f"Component initialized: {component_id}")
            else:
                logger.warning(f"Component initialization failed: {component_id}")
            
            return result
        except Exception as e:
            logger.error(f"Component initialization error: {e}")
            return False
    
    def shutdown_component(self, component_id: str) -> bool:
        """关闭组件"""
        if component_id not in self._components:
            logger.error(f"Cannot shutdown non-existent component: {component_id}")
            return False
        
        if component_id not in self._initialized:
            logger.debug(f"Component not initialized: {component_id}")
            return True
        
        component = self._components[component_id]
        
        try:
            logger.debug(f"Shutting down component: {component_id}")
            result = component.shutdown()
            
            if result:
                self._initialized.discard(component_id)
                logger.info(f"Component shutdown: {component_id}")
            else:
                logger.warning(f"Component shutdown failed: {component_id}")
            
            return result
        except Exception as e:
            logger.error(f"Component shutdown error: {e}")
            return False
    
    def initialize_all(self) -> None:
        """初始化所有组件"""
        logger.info("Initializing all components")
        
        for component_id in self._components:
            self.initialize_component(component_id)
    
    def shutdown_all(self) -> None:
        """关闭所有组件"""
        logger.info("Shutting down all components")
        
        # 反向关闭（依赖顺序）
        for component_id in reversed(list(self._initialized)):
            self.shutdown_component(component_id)


def component(
    component_id: str,
    dependencies: Optional[List[str]] = None,
    **metadata
) -> Callable[[Type], Type]:
    """组件装饰器"""
    def decorator(cls: Type) -> Type:
        # 保存组件元数据到类属性
        cls._component_id = component_id
        cls._component_dependencies = dependencies or []
        cls._component_metadata = metadata
        
        logger.debug(f"Decorated component class: {cls.__name__} with ID: {component_id}")
        return cls
    
    return decorator


def discover_components(
    module: Any,
    registry: Optional[ComponentRegistry] = None
) -> ComponentRegistry:
    """在模块中发现并注册组件"""
    if registry is None:
        registry = ComponentRegistry()
    
    logger.debug(f"Discovering components in module: {module.__name__}")
    
    # 遍历模块的所有属性
    for name, obj in inspect.getmembers(module):
        # 检查是否是类且是 SimpleComponent 的子类
        if (inspect.isclass(obj) and 
            issubclass(obj, SimpleComponent) and 
            obj != SimpleComponent):
            
            # 检查是否有组件装饰器
            if hasattr(obj, '_component_id'):
                component_id = getattr(obj, '_component_id')
                dependencies = getattr(obj, '_component_dependencies', [])
                metadata = getattr(obj, '_component_metadata', {})
                
                registry.register(
                    component_id,
                    obj,
                    dependencies=dependencies,
                    metadata=metadata
                )
                
                logger.info(f"Discovered component: {component_id}")
    
    logger.info(f"Discovered {len(registry.list_components())} components in {module.__name__}")
    return registry
