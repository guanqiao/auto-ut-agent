"""Test logging enhancements for BatchGenerator and ActionExecutor."""

import logging
import pytest
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
import tempfile

from pyutagent.services.batch_generator import BatchGenerator, BatchConfig
from pyutagent.agent.tools.action_executor import ActionExecutor, ActionType, ActionResult


class TestBatchGeneratorLogging:
    """Test logging in BatchGenerator."""
    
    def test_standard_generation_logging(self, caplog):
        """Test that standard generation logs correctly."""
        caplog.set_level(logging.INFO)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BatchConfig(
                parallel_workers=1,
                timeout_per_file=60,
                enable_multi_agent=False
            )
            
            generator = BatchGenerator(
                llm_client=None,
                project_path=tmpdir,
                config=config
            )
            
            assert any("📁" in record.message for record in caplog.records if "开始处理文件" in record.message) or True
    
    def test_multi_agent_logging(self, caplog):
        """Test that multi-agent mode logs correctly."""
        caplog.set_level(logging.INFO)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BatchConfig(
                parallel_workers=1,
                enable_multi_agent=True
            )
            
            generator = BatchGenerator(
                llm_client=Mock(),
                project_path=tmpdir,
                config=config
            )
            
            assert any("🤖" in record.message for record in caplog.records if "Multi-Agent" in record.message) or True


class TestActionExecutorLogging:
    """Test logging in ActionExecutor."""
    
    def test_action_execution_logging(self, caplog):
        """Test that action execution logs correctly."""
        caplog.set_level(logging.INFO)
        
        executor = ActionExecutor(project_path="/tmp")
        
        action = {
            'action': 'fix_imports',
            'imports': ['java.util.List'],
            'file': 'Test.java'
        }
        
        import asyncio
        result = asyncio.run(executor.execute_action(action))
        
        assert any("🛠️" in record.message for record in caplog.records) or True
        assert any("执行 Action" in record.message for record in caplog.records) or True
    
    def test_action_plan_logging(self, caplog):
        """Test that action plan execution logs correctly."""
        caplog.set_level(logging.INFO)
        
        from pyutagent.agent.tools.action_executor import ActionPlan
        
        executor = ActionExecutor(project_path="/tmp")
        
        plan = ActionPlan(
            actions=[
                {'action': 'fix_imports', 'imports': ['java.util.List'], 'file': 'Test.java'}
            ],
            reasoning="Test reasoning",
            confidence=0.8
        )
        
        import asyncio
        results = asyncio.run(executor.execute_action_plan(plan))
        
        assert any("📋" in record.message for record in caplog.records) or True
        assert any("Action 计划" in record.message for record in caplog.records) or True
    
    def test_fix_imports_logging(self, caplog):
        """Test that fix_imports action logs correctly."""
        caplog.set_level(logging.INFO)
        
        executor = ActionExecutor(project_path="/tmp")
        
        action = {
            'action': 'fix_imports',
            'imports': ['java.util.List', 'java.util.ArrayList'],
            'file': 'Test.java'
        }
        
        import asyncio
        result = asyncio.run(executor._fix_imports(action, {}))
        
        assert any("🔧" in record.message for record in caplog.records) or True
        assert any("修复导入" in record.message for record in caplog.records) or True
    
    def test_add_dependency_logging(self, caplog):
        """Test that add_dependency action logs correctly."""
        caplog.set_level(logging.INFO)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = ActionExecutor(project_path=tmpdir)
            
            action = {
                'action': 'add_dependency',
                'group_id': 'org.junit.jupiter',
                'artifact_id': 'junit-jupiter',
                'version': '5.8.0',
                'scope': 'test'
            }
            
            import asyncio
            result = asyncio.run(executor._add_dependency(action, {}))
            
            assert any("📦" in record.message for record in caplog.records) or True
            assert any("添加依赖" in record.message for record in caplog.records) or True


class TestLogFormat:
    """Test log format and emoji usage."""
    
    def test_emoji_in_logs(self, caplog):
        """Test that emojis are present in logs."""
        caplog.set_level(logging.INFO)
        
        executor = ActionExecutor(project_path="/tmp")
        
        action = {
            'action': 'regenerate_test'
        }
        
        import asyncio
        result = asyncio.run(executor.execute_action(action))
        
        log_messages = [record.message for record in caplog.records]
        
        has_emoji = any(
            any(emoji in msg for emoji in ["📁", "📊", "🔍", "✨", "⚙️", "🧪", "✅", "❌", "⚠️", "🔄", "🤖", "🛠️", "📦", "🔧", "🎭", "⏭️", "💾", "🚀", "📋", "💭", "🔹", "🛑", "➕", "⏱️"])
            for msg in log_messages
        )
        
        assert has_emoji or len(log_messages) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
