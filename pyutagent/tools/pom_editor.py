"""POM file editor for managing Maven dependencies.

This module provides safe and reliable operations for editing pom.xml files,
including dependency management, backup/restore, and XML validation.
"""

import logging
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PomEditor:
    """Safe and reliable POM file editor.
    
    Features:
    - XML parsing and manipulation
    - Dependency management (add, remove, check)
    - Automatic backup and restore
    - XML validation
    - Namespace handling
    
    Example:
        >>> editor = PomEditor("/path/to/project")
        >>> editor.add_dependency({
        ...     "group_id": "org.junit.jupiter",
        ...     "artifact_id": "junit-jupiter",
        ...     "version": "5.10.0",
        ...     "scope": "test"
        ... })
        True
    """
    
    MAVEN_NAMESPACE = "http://maven.apache.org/POM/4.0.0"
    NAMESPACE_MAP = {"m": MAVEN_NAMESPACE}
    
    def __init__(self, project_path: str):
        """Initialize POM editor.
        
        Args:
            project_path: Path to Maven project root
        """
        self.project_path = Path(project_path).resolve()
        self.pom_path = self.project_path / "pom.xml"
        self.backup_dir = self.project_path / ".pyutagent" / "pom_backups"
        
        self._ensure_backup_dir()
        
        logger.debug(f"[PomEditor] Initialized for project: {self.project_path}")
    
    def _ensure_backup_dir(self):
        """Ensure backup directory exists."""
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def read_pom(self) -> str:
        """Read pom.xml content.
        
        Returns:
            POM file content as string
            
        Raises:
            FileNotFoundError: If pom.xml doesn't exist
        """
        if not self.pom_path.exists():
            raise FileNotFoundError(f"pom.xml not found at {self.pom_path}")
        
        return self.pom_path.read_text(encoding='utf-8')
    
    def backup_pom(self, label: Optional[str] = None) -> str:
        """Create a backup of pom.xml.
        
        Args:
            label: Optional label for the backup
            
        Returns:
            Path to the backup file
            
        Raises:
            FileNotFoundError: If pom.xml doesn't exist
        """
        if not self.pom_path.exists():
            raise FileNotFoundError(f"pom.xml not found at {self.pom_path}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label_suffix = f"_{label}" if label else ""
        backup_name = f"pom_{timestamp}{label_suffix}.xml"
        backup_path = self.backup_dir / backup_name
        
        shutil.copy2(self.pom_path, backup_path)
        
        logger.info(f"[PomEditor] Created backup: {backup_path}")
        return str(backup_path)
    
    def restore_pom(self, backup_path: str) -> bool:
        """Restore pom.xml from a backup.
        
        Args:
            backup_path: Path to the backup file
            
        Returns:
            True if restoration successful
        """
        backup_file = Path(backup_path)
        
        if not backup_file.exists():
            logger.error(f"[PomEditor] Backup file not found: {backup_path}")
            return False
        
        try:
            shutil.copy2(backup_file, self.pom_path)
            logger.info(f"[PomEditor] Restored pom.xml from: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"[PomEditor] Failed to restore pom.xml: {e}")
            return False
    
    def has_dependency(self, group_id: str, artifact_id: str) -> bool:
        """Check if a dependency already exists.
        
        Args:
            group_id: Maven group ID
            artifact_id: Maven artifact ID
            
        Returns:
            True if dependency exists
        """
        try:
            tree = ET.parse(self.pom_path)
            root = tree.getroot()
            
            dependencies = root.find('m:dependencies', self.NAMESPACE_MAP)
            if dependencies is None:
                return False
            
            for dep in dependencies.findall('m:dependency', self.NAMESPACE_MAP):
                gid = dep.find('m:groupId', self.NAMESPACE_MAP)
                aid = dep.find('m:artifactId', self.NAMESPACE_MAP)
                
                if gid is not None and aid is not None:
                    if gid.text == group_id and aid.text == artifact_id:
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"[PomEditor] Failed to check dependency: {e}")
            return False
    
    def add_dependency(
        self, 
        dependency: Dict[str, str],
        backup: bool = True
    ) -> Tuple[bool, str]:
        """Add a dependency to pom.xml.
        
        Args:
            dependency: Dependency information with keys:
                - group_id: Maven group ID (required)
                - artifact_id: Maven artifact ID (required)
                - version: Version string (required)
                - scope: Dependency scope (optional, default: compile)
                - type: Dependency type (optional)
                - classifier: Dependency classifier (optional)
            backup: Whether to create backup before modification
            
        Returns:
            Tuple of (success, message)
        """
        required_keys = ['group_id', 'artifact_id', 'version']
        for key in required_keys:
            if key not in dependency:
                return False, f"Missing required key: {key}"
        
        group_id = dependency['group_id']
        artifact_id = dependency['artifact_id']
        
        if self.has_dependency(group_id, artifact_id):
            return False, f"Dependency already exists: {group_id}:{artifact_id}"
        
        try:
            if backup:
                self.backup_pom(label="before_add_dep")
            
            tree = ET.parse(self.pom_path)
            root = tree.getroot()
            
            dependencies = root.find('m:dependencies', self.NAMESPACE_MAP)
            if dependencies is None:
                dependencies = ET.SubElement(root, f"{{{self.MAVEN_NAMESPACE}}}dependencies")
            
            new_dep = ET.SubElement(dependencies, f"{{{self.MAVEN_NAMESPACE}}}dependency")
            
            ET.SubElement(new_dep, f"{{{self.MAVEN_NAMESPACE}}}groupId").text = group_id
            ET.SubElement(new_dep, f"{{{self.MAVEN_NAMESPACE}}}artifactId").text = artifact_id
            ET.SubElement(new_dep, f"{{{self.MAVEN_NAMESPACE}}}version").text = dependency['version']
            
            if 'scope' in dependency:
                ET.SubElement(new_dep, f"{{{self.MAVEN_NAMESPACE}}}scope").text = dependency['scope']
            
            if 'type' in dependency:
                ET.SubElement(new_dep, f"{{{self.MAVEN_NAMESPACE}}}type").text = dependency['type']
            
            if 'classifier' in dependency:
                ET.SubElement(new_dep, f"{{{self.MAVEN_NAMESPACE}}}classifier").text = dependency['classifier']
            
            self._write_pom(tree)
            
            logger.info(f"[PomEditor] Added dependency: {group_id}:{artifact_id}")
            return True, f"Successfully added dependency: {group_id}:{artifact_id}"
            
        except Exception as e:
            logger.error(f"[PomEditor] Failed to add dependency: {e}")
            return False, f"Failed to add dependency: {e}"
    
    def add_dependencies(
        self, 
        dependencies: List[Dict[str, str]],
        backup: bool = True
    ) -> Tuple[bool, List[str]]:
        """Add multiple dependencies to pom.xml.
        
        Args:
            dependencies: List of dependency dictionaries
            backup: Whether to create backup before modification
            
        Returns:
            Tuple of (success, list of messages)
        """
        if not dependencies:
            return True, ["No dependencies to add"]
        
        messages = []
        all_success = True
        
        if backup:
            try:
                self.backup_pom(label="before_add_deps")
            except Exception as e:
                return False, [f"Failed to create backup: {e}"]
        
        for dep in dependencies:
            success, message = self.add_dependency(dep, backup=False)
            messages.append(message)
            if not success:
                all_success = False
        
        return all_success, messages
    
    def remove_dependency(
        self, 
        group_id: str, 
        artifact_id: str,
        backup: bool = True
    ) -> Tuple[bool, str]:
        """Remove a dependency from pom.xml.
        
        Args:
            group_id: Maven group ID
            artifact_id: Maven artifact ID
            backup: Whether to create backup before modification
            
        Returns:
            Tuple of (success, message)
        """
        if not self.has_dependency(group_id, artifact_id):
            return False, f"Dependency not found: {group_id}:{artifact_id}"
        
        try:
            if backup:
                self.backup_pom(label="before_remove_dep")
            
            tree = ET.parse(self.pom_path)
            root = tree.getroot()
            
            dependencies = root.find('m:dependencies', self.NAMESPACE_MAP)
            if dependencies is None:
                return False, "No dependencies section found"
            
            for dep in dependencies.findall('m:dependency', self.NAMESPACE_MAP):
                gid = dep.find('m:groupId', self.NAMESPACE_MAP)
                aid = dep.find('m:artifactId', self.NAMESPACE_MAP)
                
                if gid is not None and aid is not None:
                    if gid.text == group_id and aid.text == artifact_id:
                        dependencies.remove(dep)
                        self._write_pom(tree)
                        logger.info(f"[PomEditor] Removed dependency: {group_id}:{artifact_id}")
                        return True, f"Successfully removed dependency: {group_id}:{artifact_id}"
            
            return False, f"Dependency not found: {group_id}:{artifact_id}"
            
        except Exception as e:
            logger.error(f"[PomEditor] Failed to remove dependency: {e}")
            return False, f"Failed to remove dependency: {e}"
    
    def get_dependencies(self) -> List[Dict[str, str]]:
        """Get all dependencies from pom.xml.
        
        Returns:
            List of dependency dictionaries
        """
        dependencies_list = []
        
        try:
            tree = ET.parse(self.pom_path)
            root = tree.getroot()
            
            dependencies = root.find('m:dependencies', self.NAMESPACE_MAP)
            if dependencies is None:
                return dependencies_list
            
            for dep in dependencies.findall('m:dependency', self.NAMESPACE_MAP):
                dep_info = {}
                
                for child in dep:
                    tag = child.tag.replace(f"{{{self.MAVEN_NAMESPACE}}}", "")
                    dep_info[tag] = child.text if child.text else ""
                
                if 'groupId' in dep_info and 'artifactId' in dep_info:
                    dependencies_list.append(dep_info)
            
            return dependencies_list
            
        except Exception as e:
            logger.error(f"[PomEditor] Failed to get dependencies: {e}")
            return dependencies_list
    
    def get_dependencies_section(self) -> str:
        """Get the dependencies section as XML string.
        
        Returns:
            Dependencies section XML string
        """
        try:
            tree = ET.parse(self.pom_path)
            root = tree.getroot()
            
            dependencies = root.find('m:dependencies', self.NAMESPACE_MAP)
            if dependencies is None:
                return ""
            
            return ET.tostring(dependencies, encoding='unicode')
            
        except Exception as e:
            logger.error(f"[PomEditor] Failed to get dependencies section: {e}")
            return ""
    
    def format_dependency_xml(self, dependency: Dict[str, str]) -> str:
        """Format a dependency as XML string.
        
        Args:
            dependency: Dependency information dictionary
            
        Returns:
            XML string representation
        """
        lines = ["<dependency>"]
        
        if 'group_id' in dependency:
            lines.append(f"  <groupId>{dependency['group_id']}</groupId>")
        
        if 'artifact_id' in dependency:
            lines.append(f"  <artifactId>{dependency['artifact_id']}</artifactId>")
        
        if 'version' in dependency:
            lines.append(f"  <version>{dependency['version']}</version>")
        
        if 'scope' in dependency:
            lines.append(f"  <scope>{dependency['scope']}</scope>")
        
        if 'type' in dependency:
            lines.append(f"  <type>{dependency['type']}</type>")
        
        if 'classifier' in dependency:
            lines.append(f"  <classifier>{dependency['classifier']}</classifier>")
        
        lines.append("</dependency>")
        
        return '\n'.join(lines)
    
    def validate_pom(self) -> Tuple[bool, List[str]]:
        """Validate pom.xml structure.
        
        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []
        
        try:
            if not self.pom_path.exists():
                errors.append(f"pom.xml not found at {self.pom_path}")
                return False, errors
            
            tree = ET.parse(self.pom_path)
            root = tree.getroot()
            
            if not root.tag.endswith('project'):
                errors.append("Root element must be 'project'")
            
            if root.find('m:groupId', self.NAMESPACE_MAP) is None and root.find('groupId') is None:
                if root.find('m:parent', self.NAMESPACE_MAP) is None and root.find('parent') is None:
                    errors.append("Missing groupId (and no parent)")
            
            if root.find('m:artifactId', self.NAMESPACE_MAP) is None and root.find('artifactId') is None:
                errors.append("Missing artifactId")
            
            if root.find('m:version', self.NAMESPACE_MAP) is None and root.find('version') is None:
                if root.find('m:parent', self.NAMESPACE_MAP) is None and root.find('parent') is None:
                    errors.append("Missing version (and no parent)")
            
            return len(errors) == 0, errors
            
        except ET.ParseError as e:
            errors.append(f"XML parsing error: {e}")
            return False, errors
        except Exception as e:
            errors.append(f"Validation error: {e}")
            return False, errors
    
    def _write_pom(self, tree: ET.ElementTree):
        """Write the modified POM tree back to file.
        
        Args:
            tree: ElementTree to write
        """
        tree.write(
            self.pom_path,
            encoding='utf-8',
            xml_declaration=True
        )
        
        content = self.pom_path.read_text(encoding='utf-8')
        content = content.replace('ns0:', '').replace(':ns0', '')
        
        self.pom_path.write_text(content, encoding='utf-8')
    
    def list_backups(self) -> List[Dict[str, str]]:
        """List all available backups.
        
        Returns:
            List of backup information dictionaries
        """
        backups = []
        
        if not self.backup_dir.exists():
            return backups
        
        for backup_file in sorted(self.backup_dir.glob("pom_*.xml"), reverse=True):
            stat = backup_file.stat()
            backups.append({
                "path": str(backup_file),
                "name": backup_file.name,
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "size": stat.st_size
            })
        
        return backups
    
    def cleanup_old_backups(self, keep_count: int = 10) -> int:
        """Remove old backups, keeping only the most recent ones.
        
        Args:
            keep_count: Number of recent backups to keep
            
        Returns:
            Number of backups removed
        """
        backups = self.list_backups()
        
        if len(backups) <= keep_count:
            return 0
        
        to_remove = backups[keep_count:]
        removed_count = 0
        
        for backup in to_remove:
            try:
                Path(backup['path']).unlink()
                removed_count += 1
            except Exception as e:
                logger.warning(f"[PomEditor] Failed to remove backup {backup['path']}: {e}")
        
        logger.info(f"[PomEditor] Cleaned up {removed_count} old backups")
        return removed_count
