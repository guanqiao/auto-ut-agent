"""测试组件协议和基类"""
import pytest
from typing import List
from pyutagent.core.component_protocol import (
    IAgentComponent,
    ComponentBase,
    ComponentLifecycle,
    ComponentCapability
)


class ConcreteComponent(ComponentBase):
    """具体组件实现（用于测试）"""
    
    def __init__(self, name: str):
        super().__init__(name)
    
    def get_capabilities(self) -> List[ComponentCapability]:
        return [
            ComponentCapability(name="test_capability", description="Test capability")
        ]
    
    async def _do_initialize(self):
        """执行初始化"""
        pass
    
    async def _do_shutdown(self):
        """执行关闭"""
        pass
    
    async def _do_start(self):
        """执行启动"""
        pass
    
    async def _do_stop(self):
        """执行停止"""
        pass


class TestComponentLifecycle:
    """测试组件生命周期"""
    
    def test_lifecycle_enum(self):
        """测试生命周期枚举"""
        assert ComponentLifecycle.CREATED.value == "created"
        assert ComponentLifecycle.INITIALIZED.value == "initialized"
        assert ComponentLifecycle.RUNNING.value == "running"
        assert ComponentLifecycle.STOPPED.value == "stopped"
        assert ComponentLifecycle.SHUTDOWN.value == "shutdown"


class TestComponentBase:
    """测试组件基类"""
    
    def test_create_component(self):
        """测试创建组件"""
        component = ConcreteComponent("test_component")
        
        assert component.name == "test_component"
        assert component.lifecycle_state == ComponentLifecycle.CREATED
        assert component.is_initialized is False
    
    @pytest.mark.asyncio
    async def test_component_initialize(self):
        """测试组件初始化"""
        component = ConcreteComponent("test_component")
        
        # 初始状态
        assert component.lifecycle_state == ComponentLifecycle.CREATED
        assert component.is_initialized is False
        
        # 初始化
        await component.initialize()
        
        # 验证状态
        assert component.lifecycle_state == ComponentLifecycle.INITIALIZED
        assert component.is_initialized is True
        
        # 重复初始化应该不会改变状态
        await component.initialize()
        assert component.is_initialized is True
    
    @pytest.mark.asyncio
    async def test_component_start(self):
        """测试组件启动"""
        component = ConcreteComponent("test_component")
        await component.initialize()
        
        # 启动
        await component.start()
        
        # 验证状态
        assert component.lifecycle_state == ComponentLifecycle.RUNNING
    
    @pytest.mark.asyncio
    async def test_component_start_without_initialize(self):
        """测试未初始化就启动应该抛出异常"""
        component = ConcreteComponent("test_component")
        
        with pytest.raises(RuntimeError, match="Component .* not initialized"):
            await component.start()
    
    @pytest.mark.asyncio
    async def test_component_stop(self):
        """测试组件停止"""
        component = ConcreteComponent("test_component")
        await component.initialize()
        await component.start()
        
        # 停止
        await component.stop()
        
        # 验证状态
        assert component.lifecycle_state == ComponentLifecycle.STOPPED
    
    @pytest.mark.asyncio
    async def test_component_shutdown(self):
        """测试组件关闭"""
        component = ConcreteComponent("test_component")
        await component.initialize()
        
        # 关闭
        await component.shutdown()
        
        # 验证状态
        assert component.lifecycle_state == ComponentLifecycle.SHUTDOWN
        assert component.is_initialized is False
        
        # 关闭后可以重新初始化
        await component.initialize()
        assert component.is_initialized is True
    
    def test_component_capabilities(self):
        """测试组件能力声明"""
        component = ConcreteComponent("test_component")
        capabilities = component.get_capabilities()
        
        assert isinstance(capabilities, list)
        assert len(capabilities) == 1
        assert capabilities[0].name == "test_capability"
    
    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        """测试完整的生命周期"""
        component = ConcreteComponent("test_component")
        
        # CREATED
        assert component.lifecycle_state == ComponentLifecycle.CREATED
        
        # 初始化 -> INITIALIZED
        await component.initialize()
        assert component.lifecycle_state == ComponentLifecycle.INITIALIZED
        
        # 启动 -> RUNNING
        await component.start()
        assert component.lifecycle_state == ComponentLifecycle.RUNNING
        
        # 停止 -> STOPPED
        await component.stop()
        assert component.lifecycle_state == ComponentLifecycle.STOPPED
        
        # 关闭 -> SHUTDOWN
        await component.shutdown()
        assert component.lifecycle_state == ComponentLifecycle.SHUTDOWN
