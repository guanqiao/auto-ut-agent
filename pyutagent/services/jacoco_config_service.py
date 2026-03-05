"""JaCoCo configuration service for automatic setup.

This module provides automatic JaCoCo configuration generation and application
using LLM to analyze project structure and generate compatible configurations.
"""

import json
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..tools.pom_editor import PomEditor
from ..tools.maven_tools import MavenRunner
from ..agent.prompts.jacoco_config_prompts import (
    JACOCO_CONFIG_GENERATION_PROMPT,
    JACOCO_CONFIG_ANALYSIS_PROMPT,
)

logger = logging.getLogger(__name__)


@dataclass
class JacocoConfigResult:
    """Result of JaCoCo configuration operation."""
    success: bool
    message: str
    config: Optional[Dict[str, Any]] = None
    backup_path: Optional[str] = None
    applied: bool = False
    dependencies_installed: bool = False


@dataclass
class JacocoAnalysisResult:
    """Result of JaCoCo configuration analysis."""
    is_configured: bool
    plugin_version: Optional[str] = None
    executions: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class JacocoConfigService:
    """Service for automatic JaCoCo configuration.
    
    Features:
    - Detect existing JaCoCo configuration
    - Generate configuration using LLM
    - Apply configuration to pom.xml
    - Install dependencies
    - Create backups before modification
    
    Example:
        >>> service = JacocoConfigService("/path/to/project", llm_client)
        >>> result = await service.auto_configure()
        >>> if result.success:
        ...     print(f"Configuration applied: {result.backup_path}")
    """
    
    DEFAULT_JACOCO_VERSION = "0.8.11"
    
    def __init__(self, project_path: str, llm_client: Optional[Any] = None):
        """Initialize JaCoCo configuration service.
        
        Args:
            project_path: Path to Maven project root
            llm_client: Optional LLM client for configuration generation
        """
        self.project_path = Path(project_path).resolve()
        self.llm_client = llm_client
        self.pom_editor = PomEditor(project_path)
        self.maven_runner = MavenRunner(project_path)
        
        logger.debug(f"[JacocoConfigService] Initialized for project: {self.project_path}")
    
    def check_jacoco_configured(self) -> JacocoAnalysisResult:
        """Check if JaCoCo is already configured in pom.xml.
        
        Returns:
            JacocoAnalysisResult with configuration status
        """
        pom_path = self.project_path / "pom.xml"
        if not pom_path.exists():
            logger.warning(f"[JacocoConfigService] pom.xml not found at {pom_path}")
            return JacocoAnalysisResult(
                is_configured=False,
                issues=["pom.xml not found"]
            )
        
        try:
            content = pom_path.read_text(encoding='utf-8')
            
            # Check for jacoco-maven-plugin
            has_jacoco_plugin = 'jacoco-maven-plugin' in content
            has_jacoco_dep = 'org.jacoco' in content and 'jacoco' in content.lower()
            
            if not has_jacoco_plugin and not has_jacoco_dep:
                logger.info("[JacocoConfigService] JaCoCo not configured")
                return JacocoAnalysisResult(is_configured=False)
            
            # Parse XML for detailed analysis
            try:
                tree = ET.parse(pom_path)
                root = tree.getroot()
                
                ns = {'m': 'http://maven.apache.org/POM/4.0.0'}
                
                plugin_version = None
                executions = []
                issues = []
                
                # Find jacoco plugin in build/plugins
                build = root.find('m:build', ns)
                if build is not None:
                    plugins = build.find('m:plugins', ns)
                    if plugins is not None:
                        for plugin in plugins.findall('m:plugin', ns):
                            artifact_id = plugin.find('m:artifactId', ns)
                            if artifact_id is not None and artifact_id.text == 'jacoco-maven-plugin':
                                version_elem = plugin.find('m:version', ns)
                                if version_elem is not None:
                                    plugin_version = version_elem.text
                                
                                # Check executions
                                executions_elem = plugin.find('m:executions', ns)
                                if executions_elem is not None:
                                    for execution in executions_elem.findall('m:execution', ns):
                                        goals_elem = execution.find('m:goals', ns)
                                        if goals_elem is not None:
                                            for goal in goals_elem.findall('m:goal', ns):
                                                if goal.text:
                                                    executions.append(goal.text)
                                break
                
                # Also check pluginManagement
                plugin_management = root.find('.//m:pluginManagement', ns)
                if plugin_management is not None and plugin_version is None:
                    plugins = plugin_management.find('.//m:plugins', ns)
                    if plugins is not None:
                        for plugin in plugins.findall('m:plugin', ns):
                            artifact_id = plugin.find('m:artifactId', ns)
                            if artifact_id is not None and artifact_id.text == 'jacoco-maven-plugin':
                                version_elem = plugin.find('m:version', ns)
                                if version_elem is not None:
                                    plugin_version = version_elem.text
                                break
                
                # Check for common issues
                if not executions:
                    issues.append("JaCoCo plugin found but no executions configured")
                if 'prepare-agent' not in executions:
                    issues.append("Missing prepare-agent execution")
                if 'report' not in executions:
                    issues.append("Missing report execution")
                
                is_configured = has_jacoco_plugin and len(executions) >= 2
                
                logger.info(
                    f"[JacocoConfigService] JaCoCo analysis complete - "
                    f"configured: {is_configured}, version: {plugin_version}, "
                    f"executions: {executions}"
                )
                
                return JacocoAnalysisResult(
                    is_configured=is_configured,
                    plugin_version=plugin_version,
                    executions=executions,
                    issues=issues
                )
                
            except ET.ParseError as e:
                logger.error(f"[JacocoConfigService] XML parse error: {e}")
                return JacocoAnalysisResult(
                    is_configured=has_jacoco_plugin,
                    issues=[f"XML parse error: {e}"]
                )
                
        except Exception as e:
            logger.exception(f"[JacocoConfigService] Failed to check JaCoCo configuration: {e}")
            return JacocoAnalysisResult(
                is_configured=False,
                issues=[f"Error: {e}"]
            )
    
    async def generate_config_with_llm(self, pom_content: Optional[str] = None) -> Dict[str, Any]:
        """Generate JaCoCo configuration using LLM.
        
        Args:
            pom_content: Optional pom.xml content. If None, reads from file.
            
        Returns:
            Dictionary with generated configuration
        """
        if self.llm_client is None:
            logger.warning("[JacocoConfigService] No LLM client available, using default config")
            return self._get_default_config()
        
        if pom_content is None:
            try:
                pom_content = self.pom_editor.read_pom()
            except Exception as e:
                logger.error(f"[JacocoConfigService] Failed to read pom.xml: {e}")
                return self._get_default_config()
        
        try:
            prompt = JACOCO_CONFIG_GENERATION_PROMPT.format(pom_content=pom_content)
            
            logger.info("[JacocoConfigService] Generating JaCoCo config with LLM")
            
            response = await self.llm_client.agenerate(prompt)
            
            # Extract JSON from response
            config = self._extract_json_from_response(response)
            
            if config:
                logger.info("[JacocoConfigService] Successfully generated config with LLM")
                return config
            else:
                logger.warning("[JacocoConfigService] Failed to parse LLM response, using default")
                return self._get_default_config()
                
        except Exception as e:
            logger.exception(f"[JacocoConfigService] LLM generation failed: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default JaCoCo configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "dependencies": [],
            "build_plugins": [
                {
                    "group_id": "org.jacoco",
                    "artifact_id": "jacoco-maven-plugin",
                    "version": self.DEFAULT_JACOCO_VERSION,
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
                    ],
                    "configuration": {
                        "excludes": []
                    }
                }
            ],
            "explanation": "默认 JaCoCo 配置，包含 prepare-agent 和 report 两个 execution",
            "warnings": []
        }
    
    def _extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response.
        
        Args:
            response: LLM response text
            
        Returns:
            Parsed JSON dictionary or None
        """
        try:
            # Try to find JSON block
            import re
            
            # Look for JSON in code blocks
            json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON between curly braces
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = response
            
            # Clean up the JSON string
            json_str = json_str.strip()
            
            return json.loads(json_str)
            
        except json.JSONDecodeError as e:
            logger.error(f"[JacocoConfigService] JSON decode error: {e}")
            return None
        except Exception as e:
            logger.error(f"[JacocoConfigService] Failed to extract JSON: {e}")
            return None
    
    def apply_config(self, config: Dict[str, Any]) -> JacocoConfigResult:
        """Apply JaCoCo configuration to pom.xml.
        
        Args:
            config: Configuration dictionary from generate_config_with_llm
            
        Returns:
            JacocoConfigResult with operation result
        """
        pom_path = self.project_path / "pom.xml"
        if not pom_path.exists():
            return JacocoConfigResult(
                success=False,
                message="pom.xml not found"
            )
        
        try:
            # Create backup
            backup_path = self.pom_editor.backup_pom(label="before_jacoco_config")
            logger.info(f"[JacocoConfigService] Created backup: {backup_path}")
            
            # Parse current pom.xml
            tree = ET.parse(pom_path)
            root = tree.getroot()
            
            ns = {'m': 'http://maven.apache.org/POM/4.0.0'}
            
            # Get or create build section
            build = root.find('m:build', ns)
            if build is None:
                build = ET.SubElement(root, '{http://maven.apache.org/POM/4.0.0}build')
            
            # Get or create plugins section
            plugins = build.find('m:plugins', ns)
            if plugins is None:
                plugins = ET.SubElement(build, '{http://maven.apache.org/POM/4.0.0}plugins')
            
            # Add JaCoCo plugin
            build_plugins = config.get('build_plugins', [])
            for plugin_config in build_plugins:
                self._add_jacoco_plugin(plugins, plugin_config, ns)
            
            # Write back to file
            self.pom_editor._write_pom(tree)
            
            logger.info("[JacocoConfigService] Successfully applied JaCoCo configuration")
            
            return JacocoConfigResult(
                success=True,
                message="JaCoCo configuration applied successfully",
                config=config,
                backup_path=backup_path,
                applied=True
            )
            
        except Exception as e:
            logger.exception(f"[JacocoConfigService] Failed to apply configuration: {e}")
            return JacocoConfigResult(
                success=False,
                message=f"Failed to apply configuration: {e}"
            )
    
    def _add_jacoco_plugin(self, plugins_elem: ET.Element, plugin_config: Dict[str, Any], ns: Dict[str, str]):
        """Add JaCoCo plugin to plugins element.
        
        Args:
            plugins_elem: Plugins XML element
            plugin_config: Plugin configuration dictionary
            ns: Namespace dictionary
        """
        maven_ns = 'http://maven.apache.org/POM/4.0.0'
        
        # Check if plugin already exists
        existing = None
        for plugin in plugins_elem.findall('m:plugin', ns):
            artifact_id = plugin.find('m:artifactId', ns)
            if artifact_id is not None and artifact_id.text == 'jacoco-maven-plugin':
                existing = plugin
                break
        
        if existing is not None:
            # Remove existing plugin
            plugins_elem.remove(existing)
            logger.debug("[JacocoConfigService] Removed existing JaCoCo plugin")
        
        # Create new plugin element
        plugin = ET.SubElement(plugins_elem, f'{{{maven_ns}}}plugin')
        
        # Add groupId
        group_id = ET.SubElement(plugin, f'{{{maven_ns}}}groupId')
        group_id.text = plugin_config.get('group_id', 'org.jacoco')
        
        # Add artifactId
        artifact_id = ET.SubElement(plugin, f'{{{maven_ns}}}artifactId')
        artifact_id.text = plugin_config.get('artifact_id', 'jacoco-maven-plugin')
        
        # Add version
        version = ET.SubElement(plugin, f'{{{maven_ns}}}version')
        version.text = plugin_config.get('version', self.DEFAULT_JACOCO_VERSION)
        
        # Add executions
        executions_config = plugin_config.get('executions', [])
        if executions_config:
            executions = ET.SubElement(plugin, f'{{{maven_ns}}}executions')
            for exec_config in executions_config:
                execution = ET.SubElement(executions, f'{{{maven_ns}}}execution')
                
                # Add execution id
                exec_id = exec_config.get('id')
                if exec_id:
                    id_elem = ET.SubElement(execution, f'{{{maven_ns}}}id')
                    id_elem.text = exec_id
                
                # Add goals
                goals = exec_config.get('goals', [])
                if goals:
                    goals_elem = ET.SubElement(execution, f'{{{maven_ns}}}goals')
                    for goal in goals:
                        goal_elem = ET.SubElement(goals_elem, f'{{{maven_ns}}}goal')
                        goal_elem.text = goal
                
                # Add phase
                phase = exec_config.get('phase')
                if phase:
                    phase_elem = ET.SubElement(execution, f'{{{maven_ns}}}phase')
                    phase_elem.text = phase
        
        logger.debug("[JacocoConfigService] Added JaCoCo plugin configuration")
    
    async def install_dependencies(self) -> Tuple[bool, str]:
        """Install JaCoCo dependencies.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            logger.info("[JacocoConfigService] Installing JaCoCo dependencies")
            
            # Run maven to download dependencies
            success, output = await self.maven_runner.resolve_dependencies_async()
            
            if success:
                logger.info("[JacocoConfigService] Dependencies installed successfully")
                return True, "Dependencies installed successfully"
            else:
                logger.error(f"[JacocoConfigService] Failed to install dependencies: {output}")
                return False, f"Failed to install dependencies: {output}"
                
        except Exception as e:
            logger.exception(f"[JacocoConfigService] Error installing dependencies: {e}")
            return False, f"Error: {e}"
    
    async def auto_configure(self, skip_if_exists: bool = True) -> JacocoConfigResult:
        """Automatically configure JaCoCo for the project.
        
        This method performs the complete configuration workflow:
        1. Check if JaCoCo is already configured
        2. Generate configuration using LLM
        3. Apply configuration to pom.xml
        4. Install dependencies
        
        Args:
            skip_if_exists: If True, skip configuration if JaCoCo is already configured
            
        Returns:
            JacocoConfigResult with operation result
        """
        logger.info("[JacocoConfigService] Starting auto-configuration")
        
        # Step 1: Check if already configured
        analysis = self.check_jacoco_configured()
        if analysis.is_configured and skip_if_exists:
            logger.info("[JacocoConfigService] JaCoCo already configured, skipping")
            return JacocoConfigResult(
                success=True,
                message=f"JaCoCo is already configured (version: {analysis.plugin_version})",
                applied=False,
                dependencies_installed=True
            )
        
        # Step 2: Generate configuration
        config = await self.generate_config_with_llm()
        
        # Step 3: Apply configuration
        result = self.apply_config(config)
        if not result.success:
            return result
        
        # Step 4: Install dependencies
        deps_success, deps_message = await self.install_dependencies()
        
        if deps_success:
            result.dependencies_installed = True
            result.message = f"{result.message}. Dependencies installed successfully."
        else:
            result.message = f"{result.message}. Warning: {deps_message}"
        
        logger.info(f"[JacocoConfigService] Auto-configuration complete: {result.message}")
        return result
    
    def generate_config_preview(self, config: Dict[str, Any]) -> str:
        """Generate a user-friendly preview of the configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Preview text
        """
        lines = ["📋 JaCoCo 配置预览", ""]
        
        build_plugins = config.get('build_plugins', [])
        if build_plugins:
            lines.append("📝 将要添加的插件配置:")
            lines.append("")
            
            for plugin in build_plugins:
                lines.append("+ <plugin>")
                lines.append(f"+     <groupId>{plugin.get('group_id', 'org.jacoco')}</groupId>")
                lines.append(f"+     <artifactId>{plugin.get('artifact_id', 'jacoco-maven-plugin')}</artifactId>")
                lines.append(f"+     <version>{plugin.get('version', self.DEFAULT_JACOCO_VERSION)}</version>")
                
                executions = plugin.get('executions', [])
                if executions:
                    lines.append("+     <executions>")
                    for exec_config in executions:
                        lines.append("+         <execution>")
                        exec_id = exec_config.get('id')
                        if exec_id:
                            lines.append(f"+             <id>{exec_id}</id>")
                        phase = exec_config.get('phase')
                        if phase:
                            lines.append(f"+             <phase>{phase}</phase>")
                        goals = exec_config.get('goals', [])
                        if goals:
                            lines.append("+             <goals>")
                            for goal in goals:
                                lines.append(f"+                 <goal>{goal}</goal>")
                            lines.append("+             </goals>")
                        lines.append("+         </execution>")
                    lines.append("+     </executions>")
                
                lines.append("+ </plugin>")
                lines.append("")
        
        # Add explanation
        explanation = config.get('explanation', '')
        if explanation:
            lines.append("📖 配置说明:")
            lines.append(f"  {explanation}")
            lines.append("")
        
        # Add warnings
        warnings = config.get('warnings', [])
        if warnings:
            lines.append("⚠️ 警告:")
            for warning in warnings:
                lines.append(f"  - {warning}")
            lines.append("")
        
        lines.append("✅ 配置将在以下阶段生效:")
        lines.append("  - test-compile: 准备 JaCoCo agent")
        lines.append("  - test: 生成覆盖率报告")
        lines.append("")
        lines.append("📁 报告将生成在: target/site/jacoco/index.html")
        
        return "\n".join(lines)
    
    def restore_backup(self, backup_path: str) -> bool:
        """Restore pom.xml from backup.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if restored successfully
        """
        try:
            result = self.pom_editor.restore_pom(backup_path)
            if result:
                logger.info(f"[JacocoConfigService] Restored from backup: {backup_path}")
            return result
        except Exception as e:
            logger.error(f"[JacocoConfigService] Failed to restore backup: {e}")
            return False