"""File icon provider for different file types and languages."""

from pathlib import Path
from typing import Dict, Optional

from PyQt6.QtGui import QIcon, QPixmap, QColor
from PyQt6.QtCore import QSize


class FileIconProvider:
    """Provides icons for files and folders based on type."""
    
    # File extension to icon mapping (using emoji as fallback)
    FILE_ICONS: Dict[str, str] = {
        # Programming languages
        '.java': '☕',
        '.py': '🐍',
        '.js': '📜',
        '.jsx': '⚛️',
        '.ts': '🔷',
        '.tsx': '⚛️',
        '.go': '🐹',
        '.rs': '🦀',
        '.c': '🔧',
        '.cpp': '🔧',
        '.cc': '🔧',
        '.h': '📋',
        '.hpp': '📋',
        '.cs': '🔵',
        '.rb': '💎',
        '.php': '🐘',
        '.swift': '🦉',
        '.kt': '🎯',
        '.scala': '🔴',
        '.r': '📊',
        '.m': '📱',
        '.mm': '📱',
        
        # Web
        '.html': '🌐',
        '.htm': '🌐',
        '.css': '🎨',
        '.scss': '🎨',
        '.sass': '🎨',
        '.less': '🎨',
        '.vue': '💚',
        '.svelte': '🧡',
        
        # Data/Config
        '.json': '📋',
        '.xml': '📋',
        '.yaml': '⚙️',
        '.yml': '⚙️',
        '.toml': '⚙️',
        '.ini': '⚙️',
        '.conf': '⚙️',
        '.config': '⚙️',
        '.env': '🔐',
        
        # Documentation
        '.md': '📝',
        '.markdown': '📝',
        '.rst': '📝',
        '.txt': '📄',
        '.pdf': '📕',
        '.doc': '📘',
        '.docx': '📘',
        
        # Images
        '.png': '🖼️',
        '.jpg': '🖼️',
        '.jpeg': '🖼️',
        '.gif': '🖼️',
        '.svg': '🎨',
        '.ico': '🎯',
        
        # Database
        '.sql': '🗄️',
        '.db': '🗄️',
        '.sqlite': '🗄️',
        
        # Build/Package
        '.gradle': '🐘',
        '.maven': '🟠',
        '.pom': '🟠',
        '.mk': '🔨',
        '.cmake': '🔨',
        '.makefile': '🔨',
        '.dockerfile': '🐳',
        
        # Shell/Scripts
        '.sh': '⌨️',
        '.bash': '⌨️',
        '.zsh': '⌨️',
        '.fish': '⌨️',
        '.ps1': '💻',
        '.bat': '💻',
        '.cmd': '💻',
        
        # Git
        '.gitignore': '🙈',
        '.gitattributes': '📝',
        
        # Test
        '.test': '🧪',
        '.spec': '🧪',
    }
    
    # Folder name to icon mapping
    FOLDER_ICONS: Dict[str, str] = {
        'project': '📦',
        'src': '📂',
        'source': '📂',
        'sources': '📂',
        'test': '🧪',
        'tests': '🧪',
        'spec': '🧪',
        'specs': '🧪',
        'doc': '📚',
        'docs': '📚',
        'documentation': '📚',
        'lib': '📚',
        'libs': '📚',
        'library': '📚',
        'libraries': '📚',
        'bin': '⚙️',
        'build': '🔨',
        'dist': '📦',
        'out': '📦',
        'target': '🎯',
        'node_modules': '📦',
        'venv': '🐍',
        '.git': '🔀',
        '.github': '🐙',
        '.vscode': '🔷',
        '.idea': '💡',
        'config': '⚙️',
        'configs': '⚙️',
        'configuration': '⚙️',
        'assets': '🎨',
        'resources': '📁',
        'static': '📁',
        'public': '🌐',
        'private': '🔒',
        'scripts': '⌨️',
        'tools': '🔧',
        'utils': '🛠️',
        'helpers': '🛠️',
        'components': '🧩',
        'pages': '📄',
        'views': '👁️',
        'models': '📊',
        'controllers': '🎮',
        'services': '⚡',
        'middleware': '🔗',
        'routes': '🛣️',
        'api': '🔌',
        'interfaces': '🔌',
        'types': '🏷️',
        'interfaces': '🔌',
        'migrations': '🗄️',
        'seeds': '🌱',
        'fixtures': '🔧',
        'mocks': '🎭',
        'stubs': '📋',
        'examples': '💡',
        'samples': '💡',
        'demo': '🎬',
        'benchmarks': '📊',
        'perf': '📊',
        'performance': '📊',
        'profiling': '📊',
        'coverage': '📈',
        'reports': '📊',
        'logs': '📋',
        'temp': '⏱️',
        'tmp': '⏱️',
        'cache': '💾',
        'caches': '💾',
    }
    
    def __init__(self):
        self._icon_cache: Dict[str, QIcon] = {}
        
    def get_file_icon(self, extension: str) -> QIcon:
        """Get icon for a file extension.
        
        Args:
            extension: File extension (e.g., '.py', '.java')
            
        Returns:
            QIcon for the file type
        """
        ext_lower = extension.lower()
        
        if ext_lower in self._icon_cache:
            return self._icon_cache[ext_lower]
        
        # Get emoji/icon text
        icon_text = self.FILE_ICONS.get(ext_lower, '📄')
        
        # Create icon from emoji (simplified - in production would use actual icon files)
        icon = self._create_icon_from_emoji(icon_text)
        self._icon_cache[ext_lower] = icon
        
        return icon
        
    def get_folder_icon(self, folder_name: str) -> QIcon:
        """Get icon for a folder.
        
        Args:
            folder_name: Name of the folder
            
        Returns:
            QIcon for the folder
        """
        name_lower = folder_name.lower()
        
        if name_lower in self._icon_cache:
            return self._icon_cache[name_lower]
        
        # Get emoji/icon text
        icon_text = self.FOLDER_ICONS.get(name_lower, '📁')
        
        # Create icon from emoji
        icon = self._create_icon_from_emoji(icon_text)
        self._icon_cache[name_lower] = icon
        
        return icon
        
    def _create_icon_from_emoji(self, emoji: str) -> QIcon:
        """Create a QIcon from an emoji character.
        
        This is a simplified implementation. In production, you would use
        actual icon files or a proper icon font.
        
        Args:
            emoji: Emoji character
            
        Returns:
            QIcon
        """
        # For now, return an empty icon
        # In production, this would render the emoji to a pixmap
        return QIcon()
        
    def get_icon_for_path(self, path: str) -> QIcon:
        """Get appropriate icon for a file system path.
        
        Args:
            path: File or folder path
            
        Returns:
            QIcon
        """
        p = Path(path)
        
        if p.is_dir():
            return self.get_folder_icon(p.name)
        else:
            return self.get_file_icon(p.suffix)
