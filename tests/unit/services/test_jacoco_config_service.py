"""Unit tests for JaCoCo configuration service."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock

from pyutagent.services.jacoco_config_service import (
    JacocoConfigService,
    JacocoConfigResult,
    JacocoAnalysisResult,
)


class TestJacocoConfigService:
    """Test cases for JacocoConfigService."""
    
    @pytest.fixture
    def temp_project(self):
        """Create a temporary project directory with pom.xml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            pom_content = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>test-project</artifactId>
    <version>1.0.0</version>
    <packaging>jar</packaging>
    
    <dependencies>
        <dependency>
            <groupId>org.junit.jupiter</groupId>
            <artifactId>junit-jupiter</artifactId>
            <version>5.10.0</version>
            <scope>test</scope>
        </dependency>
    </dependencies>
</project>
"""
            (project_path / "pom.xml").write_text(pom_content, encoding='utf-8')
            yield project_path
    
    @pytest.fixture
    def temp_project_with_jacoco(self):
        """Create a temporary project directory with JaCoCo configured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            pom_content = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>test-project</artifactId>
    <version>1.0.0</version>
    <packaging>jar</packaging>
    
    <build>
        <plugins>
            <plugin>
                <groupId>org.jacoco</groupId>
                <artifactId>jacoco-maven-plugin</artifactId>
                <version>0.8.11</version>
                <executions>
                    <execution>
                        <id>prepare-agent</id>
                        <goals>
                            <goal>prepare-agent</goal>
                        </goals>
                    </execution>
                    <execution>
                        <id>report</id>
                        <phase>test</phase>
                        <goals>
                            <goal>report</goal>
                        </goals>
                    </execution>
                </executions>
            </plugin>
        </plugins>
    </build>
</project>
"""
            (project_path / "pom.xml").write_text(pom_content, encoding='utf-8')
            yield project_path
    
    def test_init(self, temp_project):
        """Test service initialization."""
        service = JacocoConfigService(str(temp_project))
        assert service.project_path == temp_project.resolve()
        assert service.llm_client is None
    
    def test_check_jacoco_configured_not_configured(self, temp_project):
        """Test checking JaCoCo configuration when not configured."""
        service = JacocoConfigService(str(temp_project))
        result = service.check_jacoco_configured()
        
        assert isinstance(result, JacocoAnalysisResult)
        assert result.is_configured is False
        assert result.plugin_version is None
        assert len(result.executions) == 0
    
    def test_check_jacoco_configured_already_configured(self, temp_project_with_jacoco):
        """Test checking JaCoCo configuration when already configured."""
        service = JacocoConfigService(str(temp_project_with_jacoco))
        result = service.check_jacoco_configured()
        
        assert isinstance(result, JacocoAnalysisResult)
        assert result.is_configured is True
        assert result.plugin_version == "0.8.11"
        assert "prepare-agent" in result.executions
        assert "report" in result.executions
    
    def test_check_jacoco_configured_no_pom(self):
        """Test checking JaCoCo configuration when pom.xml doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = JacocoConfigService(tmpdir)
            result = service.check_jacoco_configured()
            
            assert isinstance(result, JacocoAnalysisResult)
            assert result.is_configured is False
            assert "pom.xml not found" in result.issues
    
    def test_get_default_config(self, temp_project):
        """Test getting default configuration."""
        service = JacocoConfigService(str(temp_project))
        config = service._get_default_config()
        
        assert "build_plugins" in config
        assert len(config["build_plugins"]) == 1
        
        plugin = config["build_plugins"][0]
        assert plugin["group_id"] == "org.jacoco"
        assert plugin["artifact_id"] == "jacoco-maven-plugin"
        assert plugin["version"] == JacocoConfigService.DEFAULT_JACOCO_VERSION
        assert len(plugin["executions"]) == 2
    
    def test_extract_json_from_response_valid(self, temp_project):
        """Test extracting JSON from valid response."""
        service = JacocoConfigService(str(temp_project))
        
        response = '''```json
{
    "dependencies": [],
    "build_plugins": [
        {
            "group_id": "org.jacoco",
            "artifact_id": "jacoco-maven-plugin",
            "version": "0.8.11"
        }
    ]
}
```'''
        result = service._extract_json_from_response(response)
        
        assert result is not None
        assert result["build_plugins"][0]["group_id"] == "org.jacoco"
    
    def test_extract_json_from_response_no_code_block(self, temp_project):
        """Test extracting JSON from response without code block."""
        service = JacocoConfigService(str(temp_project))
        
        response = '{"dependencies": [], "build_plugins": []}'
        result = service._extract_json_from_response(response)
        
        assert result is not None
        assert "dependencies" in result
    
    def test_extract_json_from_response_invalid(self, temp_project):
        """Test extracting JSON from invalid response."""
        service = JacocoConfigService(str(temp_project))
        
        response = "This is not JSON"
        result = service._extract_json_from_response(response)
        
        assert result is None
    
    def test_apply_config(self, temp_project):
        """Test applying configuration."""
        service = JacocoConfigService(str(temp_project))
        
        config = {
            "build_plugins": [
                {
                    "group_id": "org.jacoco",
                    "artifact_id": "jacoco-maven-plugin",
                    "version": "0.8.11",
                    "executions": [
                        {
                            "id": "prepare-agent",
                            "goals": ["prepare-agent"],
                            "phase": "test-compile"
                        }
                    ]
                }
            ]
        }
        
        result = service.apply_config(config)
        
        assert isinstance(result, JacocoConfigResult)
        assert result.success is True
        assert result.applied is True
        assert result.backup_path is not None
        
        # Verify pom.xml was modified
        pom_content = (temp_project / "pom.xml").read_text(encoding='utf-8')
        assert "jacoco-maven-plugin" in pom_content
        assert "prepare-agent" in pom_content
    
    def test_apply_config_no_pom(self):
        """Test applying configuration when pom.xml doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = JacocoConfigService(tmpdir)
            result = service.apply_config({})
            
            assert isinstance(result, JacocoConfigResult)
            assert result.success is False
            assert "pom.xml not found" in result.message
    
    def test_generate_config_preview(self, temp_project):
        """Test generating configuration preview."""
        service = JacocoConfigService(str(temp_project))
        
        config = {
            "build_plugins": [
                {
                    "group_id": "org.jacoco",
                    "artifact_id": "jacoco-maven-plugin",
                    "version": "0.8.11",
                    "executions": [
                        {
                            "id": "prepare-agent",
                            "goals": ["prepare-agent"],
                            "phase": "test-compile"
                        },
                        {
                            "id": "report",
                            "goals": ["report"],
                            "phase": "test"
                        }
                    ]
                }
            ],
            "explanation": "Test configuration"
        }
        
        preview = service.generate_config_preview(config)
        
        assert "JaCoCo 配置预览" in preview
        assert "org.jacoco" in preview
        assert "jacoco-maven-plugin" in preview
        assert "prepare-agent" in preview
        assert "report" in preview
        assert "Test configuration" in preview
    
    @pytest.mark.asyncio
    async def test_generate_config_with_llm_no_client(self, temp_project):
        """Test generating config without LLM client."""
        service = JacocoConfigService(str(temp_project))
        config = await service.generate_config_with_llm()
        
        assert "build_plugins" in config
        assert len(config["build_plugins"]) == 1
    
    @pytest.mark.asyncio
    async def test_generate_config_with_llm(self, temp_project):
        """Test generating config with LLM client."""
        mock_llm = Mock()
        mock_llm.agenerate = AsyncMock(return_value='''```json
{
    "dependencies": [],
    "build_plugins": [
        {
            "group_id": "org.jacoco",
            "artifact_id": "jacoco-maven-plugin",
            "version": "0.8.11",
            "executions": [
                {"id": "prepare-agent", "goals": ["prepare-agent"], "phase": "test-compile"},
                {"id": "report", "goals": ["report"], "phase": "test"}
            ]
        }
    ],
    "explanation": "LLM generated config"
}
```''')
        
        service = JacocoConfigService(str(temp_project), mock_llm)
        config = await service.generate_config_with_llm()
        
        assert "build_plugins" in config
        assert config["explanation"] == "LLM generated config"
        mock_llm.agenerate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_auto_configure_already_configured(self, temp_project_with_jacoco):
        """Test auto-configure when already configured."""
        service = JacocoConfigService(str(temp_project_with_jacoco))
        result = await service.auto_configure(skip_if_exists=True)
        
        assert isinstance(result, JacocoConfigResult)
        assert result.success is True
        assert result.applied is False
        assert "already configured" in result.message
    
    @pytest.mark.asyncio
    async def test_auto_configure_new(self, temp_project):
        """Test auto-configure for new project."""
        service = JacocoConfigService(str(temp_project))
        
        # Mock maven runner to avoid actual maven calls
        service.maven_runner.resolve_dependencies_async = AsyncMock(return_value=(True, "OK"))
        
        result = await service.auto_configure(skip_if_exists=True)
        
        assert isinstance(result, JacocoConfigResult)
        assert result.success is True
        assert result.applied is True
        assert result.backup_path is not None
    
    def test_restore_backup(self, temp_project):
        """Test restoring from backup."""
        service = JacocoConfigService(str(temp_project))
        
        # First apply a config to create backup
        config = {
            "build_plugins": [
                {
                    "group_id": "org.jacoco",
                    "artifact_id": "jacoco-maven-plugin",
                    "version": "0.8.11",
                    "executions": []
                }
            ]
        }
        
        result = service.apply_config(config)
        backup_path = result.backup_path
        
        # Verify config was applied
        pom_content = (temp_project / "pom.xml").read_text(encoding='utf-8')
        assert "jacoco-maven-plugin" in pom_content
        
        # Restore backup
        restored = service.restore_backup(backup_path)
        assert restored is True
        
        # Verify config was restored
        pom_content = (temp_project / "pom.xml").read_text(encoding='utf-8')
        assert "jacoco-maven-plugin" not in pom_content
