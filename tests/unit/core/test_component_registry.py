"""组件发现与注册测试"""
import pytest
from pyutagent.core.component_registry import (
    ComponentRegistry,
    ComponentInfo,
    component,
    discover_components,
    SimpleComponent
)


class TestComponentRegistry:
    """组件注册表测试"""
    
    def test_create_registry(self):
        """测试创建注册表"""
        registry = ComponentRegistry()
        assert registry is not None
    
    def test_register_component(self):
        """测试注册组件"""
        registry = ComponentRegistry()
        
        class TestComponent(SimpleComponent):
            pass
        
        registry.register("test_component", TestComponent)
        
        assert registry.has_component("test_component")
    
    def test_register_component_with_dependencies(self):
        """测试注册带依赖的组件"""
        registry = ComponentRegistry()
        
        class TestComponent(SimpleComponent):
            pass
        
        registry.register(
            "test_component",
            TestComponent,
            dependencies=["dep1", "dep2"]
        )
        
        info = registry.get_component_info("test_component")
        assert info.dependencies == ["dep1", "dep2"]
    
    def test_get_component(self):
        """测试获取组件"""
        registry = ComponentRegistry()
        
        class TestComponent(SimpleComponent):
            pass
        
        registry.register("test_component", TestComponent)
        component = registry.get_component("test_component")
        
        assert component is not None
        assert isinstance(component, TestComponent)
    
    def test_get_nonexistent_component(self):
        """测试获取不存在的组件"""
        registry = ComponentRegistry()
        
        component = registry.get_component("nonexistent")
        
        assert component is None
    
    def test_unregister_component(self):
        """测试注销组件"""
        registry = ComponentRegistry()
        
        class TestComponent(SimpleComponent):
            pass
        
        registry.register("test_component", TestComponent)
        registry.unregister("test_component")
        
        assert not registry.has_component("test_component")
    
    def test_list_components(self):
        """测试列出所有组件"""
        registry = ComponentRegistry()
        
        class Component1(SimpleComponent):
            pass
        
        class Component2(SimpleComponent):
            pass
        
        registry.register("comp1", Component1)
        registry.register("comp2", Component2)
        
        components = registry.list_components()
        
        assert len(components) == 2
        assert "comp1" in components
        assert "comp2" in components
    
    def test_resolve_dependencies(self):
        """测试解析依赖"""
        registry = ComponentRegistry()
        
        class DatabaseComponent(SimpleComponent):
            pass
        
        class CacheComponent(SimpleComponent):
            pass
        
        class ServiceComponent(SimpleComponent):
            pass
        
        registry.register("database", DatabaseComponent)
        registry.register("cache", CacheComponent)
        registry.register(
            "service",
            ServiceComponent,
            dependencies=["database", "cache"]
        )
        
        dependencies = registry.resolve_dependencies("service")
        
        assert "database" in dependencies
        assert "cache" in dependencies
    
    def test_resolve_circular_dependencies(self):
        """测试循环依赖检测"""
        registry = ComponentRegistry()
        
        class ComponentA(SimpleComponent):
            pass
        
        class ComponentB(SimpleComponent):
            pass
        
        registry.register("compA", ComponentA, dependencies=["compB"])
        registry.register("compB", ComponentB, dependencies=["compA"])
        
        # 应该能检测到循环依赖
        with pytest.raises(ValueError):
            registry.resolve_dependencies("compA")
    
    def test_initialize_component(self):
        """测试初始化组件"""
        registry = ComponentRegistry()
        
        class TestComponent(SimpleComponent):
            def initialize(self):
                self._custom_initialized = True
                return super().initialize()
        
        registry.register("test_component", TestComponent)
        component = registry.get_component("test_component")
        
        result = registry.initialize_component("test_component")
        
        assert result is True
        assert component._custom_initialized is True
    
    def test_shutdown_component(self):
        """测试关闭组件"""
        registry = ComponentRegistry()
        
        class TestComponent(SimpleComponent):
            def shutdown(self):
                self._custom_shutdown = True
                return super().shutdown()
        
        registry.register("test_component", TestComponent)
        
        # 先初始化再关闭
        registry.initialize_component("test_component")
        
        result = registry.shutdown_component("test_component")
        
        assert result is True
    
    def test_initialize_all_components(self):
        """测试初始化所有组件"""
        registry = ComponentRegistry()
        
        class Component1(SimpleComponent):
            def initialize(self):
                self._initialized = True
                return super().initialize()
        
        class Component2(SimpleComponent):
            def initialize(self):
                self._initialized = True
                return super().initialize()
        
        registry.register("comp1", Component1)
        registry.register("comp2", Component2)
        
        registry.initialize_all()
        
        comp1 = registry.get_component("comp1")
        comp2 = registry.get_component("comp2")
        
        assert comp1._initialized is True
        assert comp2._initialized is True
    
    def test_shutdown_all_components(self):
        """测试关闭所有组件"""
        registry = ComponentRegistry()
        
        class Component1(SimpleComponent):
            def shutdown(self):
                self._shutdown = True
                return super().shutdown()
        
        class Component2(SimpleComponent):
            def shutdown(self):
                self._shutdown = True
                return super().shutdown()
        
        registry.register("comp1", Component1)
        registry.register("comp2", Component2)
        
        # 先初始化所有
        registry.initialize_all()
        
        registry.shutdown_all()
        
        comp1 = registry.get_component("comp1")
        comp2 = registry.get_component("comp2")
        
        assert comp1._shutdown is True
        assert comp2._shutdown is True


class TestComponentDecorator:
    """组件装饰器测试"""
    
    def test_component_decorator(self):
        """测试组件装饰器"""
        
        @component("decorated_component")
        class DecoratedComponent(SimpleComponent):
            pass
        
        assert DecoratedComponent._component_id == "decorated_component"
    
    def test_component_decorator_with_metadata(self):
        """测试带元数据的组件装饰器"""
        
        @component("decorated_component", version="1.0", author="Test")
        class DecoratedComponent(SimpleComponent):
            pass
        
        assert DecoratedComponent._component_metadata["version"] == "1.0"
        assert DecoratedComponent._component_metadata["author"] == "Test"


class TestDiscoverComponents:
    """组件发现测试"""
    
    def test_discover_components_in_module(self):
        """测试在模块中发现组件"""
        
        # 创建一个测试模块
        import types
        test_module = types.ModuleType("test_module")
        
        class ModuleComponent1(SimpleComponent):
            pass
        
        class ModuleComponent2(SimpleComponent):
            pass
        
        test_module.Component1 = ModuleComponent1
        test_module.Component2 = ModuleComponent2
        
        # 添加装饰器信息
        ModuleComponent1._component_id = "component1"
        ModuleComponent2._component_id = "component2"
        
        registry = ComponentRegistry()
        discover_components(test_module, registry)
        
        assert registry.has_component("component1")
        assert registry.has_component("component2")
    
    def test_discover_components_with_dependencies(self):
        """测试发现带依赖的组件"""
        import types
        test_module = types.ModuleType("test_module")
        
        class DependentComponent(SimpleComponent):
            pass
        
        test_module.DependentComponent = DependentComponent
        DependentComponent._component_id = "dependent"
        DependentComponent._component_dependencies = ["dep1"]
        
        registry = ComponentRegistry()
        discover_components(test_module, registry)
        
        info = registry.get_component_info("dependent")
        assert "dep1" in info.dependencies
