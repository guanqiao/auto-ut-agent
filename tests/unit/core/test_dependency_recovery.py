"""Tests for dependency recovery handler."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pyutagent.core.error_recovery import (
    RecoveryStrategy,
    RecoveryResult,
    DependencyRecoveryHandler,
)


class TestDependencyRecoveryHandler:
    """Tests for DependencyRecoveryHandler class."""
    
    @pytest.fixture
    def handler(self, tmp_path):
        return DependencyRecoveryHandler(
            project_path=str(tmp_path),
            timeout=60
        )
    
    @pytest.fixture
    def mock_maven_runner(self):
        runner = MagicMock()
        runner.resolve_dependencies_async = AsyncMock(return_value=(True, ""))
        runner.resolve_test_dependencies_async = AsyncMock(return_value=(True, ""))
        return runner
    
    def test_init(self, handler, tmp_path):
        assert handler.project_path == str(tmp_path)
        assert handler.timeout == 60
        assert handler._resolution_attempts == 0
    
    @pytest.mark.asyncio
    async def test_resolve_dependencies_success(self, handler):
        with patch.object(handler, '_run_maven_resolve', return_value=(True, "")):
            result = await handler.resolve_dependencies()
            
            assert result.success is True
            assert result.strategy_used == RecoveryStrategy.RESOLVE_DEPENDENCIES
            assert result.action == "retry"
    
    @pytest.mark.asyncio
    async def test_resolve_dependencies_failure(self, handler):
        with patch.object(handler, '_run_maven_resolve', return_value=(False, "Error message")):
            result = await handler.resolve_dependencies()
            
            assert result.success is False
            assert result.strategy_used == RecoveryStrategy.RESOLVE_DEPENDENCIES
    
    @pytest.mark.asyncio
    async def test_resolve_test_dependencies_success(self, handler):
        with patch.object(handler, '_run_maven_test_compile', return_value=(True, "")):
            result = await handler.resolve_test_dependencies()
            
            assert result.success is True
            assert result.strategy_used == RecoveryStrategy.INSTALL_DEPENDENCIES
    
    @pytest.mark.asyncio
    async def test_install_missing_dependencies_test(self, handler):
        with patch.object(handler, 'resolve_test_dependencies') as mock_resolve:
            mock_resolve.return_value = RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.INSTALL_DEPENDENCIES,
                attempts_made=1
            )
            
            result = await handler.install_missing_dependencies(
                missing_packages=["org.junit.jupiter.api"],
                is_test_dependency=True
            )
            
            assert result.success is True
            mock_resolve.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_install_missing_dependencies_non_test(self, handler):
        with patch.object(handler, 'resolve_dependencies') as mock_resolve:
            mock_resolve.return_value = RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.RESOLVE_DEPENDENCIES,
                attempts_made=1
            )
            
            result = await handler.install_missing_dependencies(
                missing_packages=["com.example.lib"],
                is_test_dependency=False
            )
            
            assert result.success is True
            mock_resolve.assert_called_once()
    
    def test_check_pom_has_test_dependencies(self, handler, tmp_path):
        pom_content = '''
<project>
    <dependencies>
        <dependency>
            <groupId>org.junit.jupiter</groupId>
            <artifactId>junit-jupiter</artifactId>
        </dependency>
        <dependency>
            <groupId>org.mockito</groupId>
            <artifactId>mockito-core</artifactId>
        </dependency>
    </dependencies>
</project>
'''
        pom_path = tmp_path / "pom.xml"
        pom_path.write_text(pom_content)
        
        result = handler.check_pom_has_test_dependencies()
        
        assert result["junit_jupiter"] is True
        assert result["mockito"] is True
        assert result["assertj"] is False
    
    def test_check_pom_no_pom_file(self, handler):
        result = handler.check_pom_has_test_dependencies()
        assert result == {}
    
    def test_suggest_pom_additions_junit(self, handler):
        suggestions = handler.suggest_pom_additions(["org.junit.jupiter.api"])
        
        assert len(suggestions) == 1
        assert "junit-jupiter" in suggestions[0]
    
    def test_suggest_pom_additions_multiple(self, handler):
        suggestions = handler.suggest_pom_additions([
            "org.junit.jupiter.api",
            "org.mockito",
            "org.assertj.core.api"
        ])
        
        assert len(suggestions) == 3
        assert any("junit-jupiter" in s for s in suggestions)
        assert any("mockito-core" in s for s in suggestions)
        assert any("assertj-core" in s for s in suggestions)
    
    def test_suggest_pom_additions_unknown_package(self, handler):
        suggestions = handler.suggest_pom_additions(["com.unknown.package"])
        
        assert len(suggestions) == 0
    
    def test_reset_attempts(self, handler):
        handler._resolution_attempts = 5
        handler.reset_attempts()
        
        assert handler._resolution_attempts == 0


class TestDependencyRecoveryHandlerWithMavenRunner:
    """Tests for DependencyRecoveryHandler with MavenRunner integration."""
    
    @pytest.fixture
    def handler_with_runner(self, tmp_path, mock_maven_runner):
        return DependencyRecoveryHandler(
            project_path=str(tmp_path),
            maven_runner=mock_maven_runner,
            timeout=60
        )
    
    @pytest.fixture
    def mock_maven_runner(self):
        runner = MagicMock()
        runner.resolve_dependencies_async = AsyncMock(return_value=(True, ""))
        runner.resolve_test_dependencies_async = AsyncMock(return_value=(True, ""))
        return runner
    
    @pytest.mark.asyncio
    async def test_resolve_dependencies_uses_maven_runner(self, handler_with_runner, mock_maven_runner):
        result = await handler_with_runner.resolve_dependencies()
        
        mock_maven_runner.resolve_dependencies_async.assert_called_once()
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_resolve_test_dependencies_uses_maven_runner(self, handler_with_runner, mock_maven_runner):
        result = await handler_with_runner.resolve_test_dependencies()
        
        mock_maven_runner.resolve_test_dependencies_async.assert_called_once()
        assert result.success is True
