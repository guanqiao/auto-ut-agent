"""Language support for multi-language project detection and handling."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LanguageConfig:
    """Configuration for a programming language."""
    name: str
    extensions: List[str]
    icon: str
    build_tools: List[str]
    config_files: List[str]
    source_dirs: List[str]
    test_dirs: List[str]
    

# Language configurations
LANGUAGE_SUPPORT: Dict[str, LanguageConfig] = {
    'java': LanguageConfig(
        name='Java',
        extensions=['.java'],
        icon='☕',
        build_tools=['maven', 'gradle', 'ant'],
        config_files=['pom.xml', 'build.gradle', 'build.gradle.kts', 'build.xml'],
        source_dirs=['src/main/java', 'src/test/java', 'src'],
        test_dirs=['src/test/java', 'test', 'tests']
    ),
    'python': LanguageConfig(
        name='Python',
        extensions=['.py'],
        icon='🐍',
        build_tools=['pip', 'poetry', 'pipenv', 'conda'],
        config_files=['requirements.txt', 'pyproject.toml', 'setup.py', 'setup.cfg', 'Pipfile', 'environment.yml'],
        source_dirs=['src', '.'],
        test_dirs=['tests', 'test', 'src/tests']
    ),
    'javascript': LanguageConfig(
        name='JavaScript',
        extensions=['.js', '.jsx', '.mjs'],
        icon='📜',
        build_tools=['npm', 'yarn', 'pnpm'],
        config_files=['package.json', '.nvmrc'],
        source_dirs=['src', 'lib', '.'],
        test_dirs=['test', 'tests', '__tests__', 'spec', 'specs']
    ),
    'typescript': LanguageConfig(
        name='TypeScript',
        extensions=['.ts', '.tsx'],
        icon='🔷',
        build_tools=['npm', 'yarn', 'pnpm', 'tsc'],
        config_files=['tsconfig.json', 'package.json'],
        source_dirs=['src', 'lib', '.'],
        test_dirs=['test', 'tests', '__tests__']
    ),
    'go': LanguageConfig(
        name='Go',
        extensions=['.go'],
        icon='🐹',
        build_tools=['go modules', 'dep', 'glide'],
        config_files=['go.mod', 'go.sum', 'Gopkg.toml', 'glide.yaml'],
        source_dirs=['.', 'cmd', 'pkg', 'internal'],
        test_dirs=['.', '_test.go']
    ),
    'rust': LanguageConfig(
        name='Rust',
        extensions=['.rs'],
        icon='🦀',
        build_tools=['cargo'],
        config_files=['Cargo.toml', 'Cargo.lock'],
        source_dirs=['src'],
        test_dirs=['tests', 'src']
    ),
    'csharp': LanguageConfig(
        name='C#',
        extensions=['.cs'],
        icon='🔵',
        build_tools=['dotnet', 'msbuild', 'nuget'],
        config_files=['.csproj', '.sln', 'packages.config'],
        source_dirs=['.', 'src'],
        test_dirs=['tests', 'Tests', '.Tests']
    ),
    'cpp': LanguageConfig(
        name='C++',
        extensions=['.cpp', '.cc', '.cxx', '.c++', '.hpp', '.h'],
        icon='🔧',
        build_tools=['cmake', 'make', 'bazel', 'conan'],
        config_files=['CMakeLists.txt', 'Makefile', 'BUILD', 'conanfile.txt'],
        source_dirs=['src', 'source', 'include', '.'],
        test_dirs=['test', 'tests', '_test']
    ),
    'ruby': LanguageConfig(
        name='Ruby',
        extensions=['.rb'],
        icon='💎',
        build_tools=['bundler', 'gem'],
        config_files=['Gemfile', 'Gemfile.lock', '.gemspec'],
        source_dirs=['lib', '.'],
        test_dirs=['test', 'spec']
    ),
    'php': LanguageConfig(
        name='PHP',
        extensions=['.php'],
        icon='🐘',
        build_tools=['composer'],
        config_files=['composer.json', 'composer.lock'],
        source_dirs=['src', 'lib', '.'],
        test_dirs=['tests', 'test']
    ),
}


class LanguageSupport:
    """Detects and provides information about project languages."""
    
    def __init__(self):
        self._language_configs = LANGUAGE_SUPPORT
        
    def detect_project(self, project_path: str) -> Dict[str, Any]:
        """Detect the primary language of a project.
        
        Args:
            project_path: Path to the project directory
            
        Returns:
            Dictionary with project info:
            - language: Primary language name
            - config: LanguageConfig object
            - confidence: Detection confidence (0-1)
            - source_dirs: List of source directories
            - build_tool: Detected build tool
        """
        path = Path(project_path)
        
        if not path.exists():
            return {'language': 'Unknown', 'confidence': 0}
        
        # Check for config files first (highest confidence)
        for lang_id, config in self._language_configs.items():
            for config_file in config.config_files:
                if (path / config_file).exists():
                    source_dirs = self._find_source_dirs(path, config)
                    build_tool = self._detect_build_tool(path, config)
                    
                    return {
                        'language': config.name,
                        'language_id': lang_id,
                        'config': config,
                        'confidence': 1.0,
                        'source_dirs': source_dirs,
                        'build_tool': build_tool
                    }
        
        # Count files by extension
        extension_counts: Dict[str, int] = {}
        for lang_id, config in self._language_configs.items():
            for ext in config.extensions:
                extension_counts[lang_id] = 0
        
        total_files = 0
        for file_path in path.rglob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                for lang_id, config in self._language_configs.items():
                    if ext in config.extensions:
                        extension_counts[lang_id] = extension_counts.get(lang_id, 0) + 1
                        total_files += 1
                        break
        
        # Find dominant language
        if total_files > 0:
            dominant_lang = max(extension_counts.items(), key=lambda x: x[1])
            lang_id, count = dominant_lang
            confidence = count / total_files
            
            config = self._language_configs[lang_id]
            source_dirs = self._find_source_dirs(path, config)
            
            return {
                'language': config.name,
                'language_id': lang_id,
                'config': config,
                'confidence': confidence,
                'source_dirs': source_dirs,
                'build_tool': None
            }
        
        return {'language': 'Unknown', 'confidence': 0, 'source_dirs': [path]}
    
    def _find_source_dirs(self, project_path: Path, config: LanguageConfig) -> List[Path]:
        """Find source directories for a language.
        
        Args:
            project_path: Project root path
            config: Language configuration
            
        Returns:
            List of source directory paths
        """
        source_dirs = []
        
        for src_dir in config.source_dirs:
            full_path = project_path / src_dir
            if full_path.exists() and full_path.is_dir():
                source_dirs.append(full_path)
        
        # If no standard source dirs found, use project root
        if not source_dirs:
            source_dirs = [project_path]
        
        return source_dirs
    
    def _detect_build_tool(self, project_path: Path, config: LanguageConfig) -> Optional[str]:
        """Detect the build tool being used.
        
        Args:
            project_path: Project root path
            config: Language configuration
            
        Returns:
            Build tool name or None
        """
        build_tool_files = {
            'maven': ['pom.xml'],
            'gradle': ['build.gradle', 'build.gradle.kts'],
            'npm': ['package.json'],
            'poetry': ['pyproject.toml'],
            'pipenv': ['Pipfile'],
            'cargo': ['Cargo.toml'],
            'dotnet': ['.csproj', '.sln'],
            'cmake': ['CMakeLists.txt'],
        }
        
        for tool, files in build_tool_files.items():
            for file in files:
                if (project_path / file).exists():
                    return tool
        
        return None
    
    def get_language_config(self, language_id: str) -> Optional[LanguageConfig]:
        """Get configuration for a language.
        
        Args:
            language_id: Language identifier (e.g., 'python', 'java')
            
        Returns:
            LanguageConfig or None
        """
        return self._language_configs.get(language_id.lower())
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language IDs."""
        return list(self._language_configs.keys())
    
    def is_supported_extension(self, extension: str) -> bool:
        """Check if a file extension is supported.
        
        Args:
            extension: File extension (e.g., '.py', '.java')
            
        Returns:
            True if supported
        """
        ext_lower = extension.lower()
        for config in self._language_configs.values():
            if ext_lower in config.extensions:
                return True
        return False
    
    def get_language_for_extension(self, extension: str) -> Optional[str]:
        """Get language ID for a file extension.
        
        Args:
            extension: File extension
            
        Returns:
            Language ID or None
        """
        ext_lower = extension.lower()
        for lang_id, config in self._language_configs.items():
            if ext_lower in config.extensions:
                return lang_id
        return None
