"""Tests for Hooks System."""

import pytest
import asyncio

from pyutagent.core.hooks import (
    HookType,
    Hook,
    HookContext,
    HookResult,
    HookRegistry,
    HookManager,
    get_hook_manager,
    trigger_hook,
)
from pyutagent.core.hooks.hook import hook as hook_decorator


class TestHookType:
    """Tests for HookType enum."""
    
    def test_category(self):
        """Test hook type categorization."""
        assert HookType.USER_PROMPT_SUBMIT.category == "user_interaction"
        assert HookType.PRE_AGENT_START.category == "agent_lifecycle"
        assert HookType.PRE_TOOL_USE.category == "tool_execution"
        assert HookType.PRE_FILE_WRITE.category == "file_operations"
    
    def test_is_pre_hook(self):
        """Test pre-hook detection."""
        assert HookType.PRE_FILE_WRITE.is_pre_hook
        assert not HookType.POST_FILE_WRITE.is_pre_hook
    
    def test_is_post_hook(self):
        """Test post-hook detection."""
        assert HookType.POST_FILE_WRITE.is_post_hook
        assert not HookType.PRE_FILE_WRITE.is_post_hook
    
    def test_is_error_hook(self):
        """Test error hook detection."""
        assert HookType.ON_ERROR.is_error_hook
        assert HookType.TASK_FAILURE.is_error_hook
        assert not HookType.PRE_FILE_WRITE.is_error_hook


class TestHookContext:
    """Tests for HookContext."""
    
    def test_creation(self):
        """Test context creation."""
        context = HookContext(
            hook_type=HookType.PRE_FILE_WRITE,
            data={"file_path": "/test/path"},
            metadata={"user": "test"},
        )
        
        assert context.hook_type == HookType.PRE_FILE_WRITE
        assert context.get("file_path") == "/test/path"
        assert context.get_metadata("user") == "test"
    
    def test_get_default(self):
        """Test get with default."""
        context = HookContext(hook_type=HookType.PRE_FILE_WRITE)
        
        assert context.get("nonexistent") is None
        assert context.get("nonexistent", "default") == "default"
    
    def test_set(self):
        """Test setting values."""
        context = HookContext(hook_type=HookType.PRE_FILE_WRITE)
        
        context.set("key", "value")
        
        assert context.get("key") == "value"
    
    def test_to_dict(self):
        """Test serialization."""
        context = HookContext(
            hook_type=HookType.PRE_FILE_WRITE,
            data={"key": "value"},
        )
        
        result = context.to_dict()
        
        assert result["hook_type"] == "PRE_FILE_WRITE"
        assert result["data"]["key"] == "value"


class TestHookResult:
    """Tests for HookResult."""
    
    def test_ok(self):
        """Test ok factory method."""
        result = HookResult.ok({"key": "value"})
        
        assert result.success is True
        assert result.data["key"] == "value"
    
    def test_abort(self):
        """Test abort factory method."""
        result = HookResult.abort("Test abort")
        
        assert result.success is False
        assert result.should_abort is True
        assert result.message == "Test abort"
    
    def test_modify(self):
        """Test modify factory method."""
        result = HookResult.modify({"modified": "data"})
        
        assert result.success is True
        assert result.modified_context["modified"] == "data"
    
    def test_error(self):
        """Test error factory method."""
        result = HookResult.error("Test error")
        
        assert result.success is False
        assert result.error == "Test error"


class TestHook:
    """Tests for Hook class."""
    
    def test_sync_handler(self):
        """Test synchronous hook handler."""
        def handler(context: HookContext) -> HookResult:
            return HookResult.ok({"processed": True})
        
        hook = Hook(
            name="test_hook",
            hook_type=HookType.PRE_FILE_WRITE,
            handler=handler,
        )
        
        context = HookContext(hook_type=HookType.PRE_FILE_WRITE)
        result = asyncio.run(hook.execute(context))
        
        assert result.success is True
        assert result.data["processed"] is True
    
    def test_async_handler(self):
        """Test asynchronous hook handler."""
        async def handler(context: HookContext) -> HookResult:
            await asyncio.sleep(0.01)
            return HookResult.ok({"async": True})
        
        hook = Hook(
            name="async_hook",
            hook_type=HookType.PRE_FILE_WRITE,
            handler=handler,
        )
        
        context = HookContext(hook_type=HookType.PRE_FILE_WRITE)
        result = asyncio.run(hook.execute(context))
        
        assert result.success is True
        assert result.data["async"] is True
    
    def test_condition(self):
        """Test hook condition."""
        def handler(context: HookContext) -> HookResult:
            return HookResult.ok({"executed": True})
        
        hook = Hook(
            name="conditional_hook",
            hook_type=HookType.PRE_FILE_WRITE,
            handler=handler,
            condition=lambda ctx: ctx.get("should_execute", False),
        )
        
        context1 = HookContext(
            hook_type=HookType.PRE_FILE_WRITE,
            data={"should_execute": True},
        )
        result1 = asyncio.run(hook.execute(context1))
        assert result1.data.get("executed") is True
        
        context2 = HookContext(
            hook_type=HookType.PRE_FILE_WRITE,
            data={"should_execute": False},
        )
        result2 = asyncio.run(hook.execute(context2))
        assert "executed" not in result2.data
    
    def test_enable_disable(self):
        """Test hook enable/disable."""
        call_count = [0]
        
        def handler(context: HookContext) -> HookResult:
            call_count[0] += 1
            return HookResult.ok()
        
        hook = Hook(
            name="test",
            hook_type=HookType.PRE_FILE_WRITE,
            handler=handler,
        )
        
        context = HookContext(hook_type=HookType.PRE_FILE_WRITE)
        
        asyncio.run(hook.execute(context))
        assert call_count[0] == 1
        
        hook.disable()
        asyncio.run(hook.execute(context))
        assert call_count[0] == 1
        
        hook.enable()
        asyncio.run(hook.execute(context))
        assert call_count[0] == 2
    
    def test_error_handling(self):
        """Test hook error handling."""
        def failing_handler(context: HookContext) -> HookResult:
            raise ValueError("Test error")
        
        hook = Hook(
            name="failing_hook",
            hook_type=HookType.PRE_FILE_WRITE,
            handler=failing_handler,
        )
        
        context = HookContext(hook_type=HookType.PRE_FILE_WRITE)
        result = asyncio.run(hook.execute(context))
        
        assert result.success is False
        assert "Test error" in result.error


class TestHookRegistry:
    """Tests for HookRegistry."""
    
    def test_register_unregister(self):
        """Test hook registration and unregistration."""
        registry = HookRegistry()
        
        hook = Hook(
            name="test",
            hook_type=HookType.PRE_FILE_WRITE,
            handler=lambda ctx: HookResult.ok(),
        )
        
        registry.register(hook)
        assert registry.get_hook("test") is not None
        
        registry.unregister("test")
        assert registry.get_hook("test") is None
    
    def test_priority_ordering(self):
        """Test hooks are executed in priority order."""
        execution_order = []
        
        def make_handler(name):
            def handler(ctx):
                execution_order.append(name)
                return HookResult.ok()
            return handler
        
        registry = HookRegistry()
        
        registry.register(Hook(
            name="low",
            hook_type=HookType.PRE_FILE_WRITE,
            handler=make_handler("low"),
            priority=1,
        ))
        
        registry.register(Hook(
            name="high",
            hook_type=HookType.PRE_FILE_WRITE,
            handler=make_handler("high"),
            priority=10,
        ))
        
        registry.register(Hook(
            name="medium",
            hook_type=HookType.PRE_FILE_WRITE,
            handler=make_handler("medium"),
            priority=5,
        ))
        
        context = HookContext(hook_type=HookType.PRE_FILE_WRITE)
        asyncio.run(registry.execute_hooks(HookType.PRE_FILE_WRITE, context))
        
        assert execution_order == ["high", "medium", "low"]
    
    def test_abort_stops_execution(self):
        """Test that abort stops further hook execution."""
        execution_order = []
        
        def make_handler(name, should_abort=False):
            def handler(ctx):
                execution_order.append(name)
                if should_abort:
                    return HookResult.abort("Aborted")
                return HookResult.ok()
            return handler
        
        registry = HookRegistry()
        
        registry.register(Hook(
            name="first",
            hook_type=HookType.PRE_FILE_WRITE,
            handler=make_handler("first", should_abort=True),
            priority=10,
        ))
        
        registry.register(Hook(
            name="second",
            hook_type=HookType.PRE_FILE_WRITE,
            handler=make_handler("second"),
            priority=5,
        ))
        
        context = HookContext(hook_type=HookType.PRE_FILE_WRITE)
        result = asyncio.run(registry.execute_hooks(HookType.PRE_FILE_WRITE, context))
        
        assert execution_order == ["first"]
        assert result.should_abort is True
    
    def test_context_modification(self):
        """Test hooks can modify context."""
        def modifier(ctx):
            return HookResult.modify({"modified": True})
        
        registry = HookRegistry()
        registry.register(Hook(
            name="modifier",
            hook_type=HookType.PRE_FILE_WRITE,
            handler=modifier,
        ))
        
        context = HookContext(hook_type=HookType.PRE_FILE_WRITE)
        asyncio.run(registry.execute_hooks(HookType.PRE_FILE_WRITE, context))
        
        assert context.get("modified") is True
    
    def test_get_stats(self):
        """Test statistics generation."""
        registry = HookRegistry()
        
        registry.register(Hook(
            name="hook1",
            hook_type=HookType.PRE_FILE_WRITE,
            handler=lambda ctx: HookResult.ok(),
        ))
        
        registry.register(Hook(
            name="hook2",
            hook_type=HookType.POST_FILE_WRITE,
            handler=lambda ctx: HookResult.ok(),
        ))
        
        stats = registry.get_stats()
        
        assert stats["total_hooks"] == 2
        assert "PRE_FILE_WRITE" in stats["hooks_by_type"]


class TestHookManager:
    """Tests for HookManager."""
    
    def test_singleton(self):
        """Test global hook manager is singleton."""
        manager1 = get_hook_manager()
        manager2 = get_hook_manager()
        
        assert manager1 is manager2
    
    def test_builtin_hooks_registered(self):
        """Test built-in hooks are registered."""
        manager = HookManager()
        manager.register_builtin_hooks()
        
        hooks = manager.get_hooks(HookType.PRE_TOOL_USE)
        assert len(hooks) > 0
    
    @pytest.mark.asyncio
    async def test_trigger(self):
        """Test triggering hooks."""
        manager = HookManager()
        
        call_count = [0]
        
        def handler(ctx):
            call_count[0] += 1
            return HookResult.ok()
        
        manager.register(Hook(
            name="test_trigger",
            hook_type=HookType.PRE_FILE_WRITE,
            handler=handler,
        ))
        
        await manager.trigger(HookType.PRE_FILE_WRITE, {"file": "test"})
        
        assert call_count[0] == 1
    
    @pytest.mark.asyncio
    async def test_trigger_hook_function(self):
        """Test global trigger_hook function."""
        manager = get_hook_manager()
        
        call_count = [0]
        
        def handler(ctx):
            call_count[0] += 1
            return HookResult.ok()
        
        manager.register(Hook(
            name="test_global",
            hook_type=HookType.POST_FILE_READ,
            handler=handler,
        ))
        
        await trigger_hook(HookType.POST_FILE_READ)
        
        assert call_count[0] == 1


class TestHookDecorator:
    """Tests for hook decorator."""
    
    def test_decorator(self):
        """Test @hook decorator."""
        @hook_decorator(HookType.PRE_FILE_WRITE, priority=10)
        def my_hook(context: HookContext) -> HookResult:
            return HookResult.ok({"decorated": True})
        
        assert isinstance(my_hook, Hook)
        assert my_hook.name == "my_hook"
        assert my_hook.hook_type == HookType.PRE_FILE_WRITE
        assert my_hook.priority == 10
    
    def test_decorator_with_name(self):
        """Test @hook decorator with custom name."""
        @hook_decorator(HookType.PRE_FILE_WRITE, name="custom_name")
        def my_hook(context: HookContext) -> HookResult:
            return HookResult.ok()
        
        assert my_hook.name == "custom_name"
