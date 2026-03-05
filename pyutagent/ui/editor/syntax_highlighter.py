"""Syntax highlighter for code editor using Pygments."""

import logging
from typing import Optional, Dict

from PyQt6.QtCore import Qt, QRegularExpression
from PyQt6.QtGui import (
    QColor, QTextCharFormat, QFont, QSyntaxHighlighter,
    QFontDatabase
)

logger = logging.getLogger(__name__)

# Try to import pygments
try:
    from pygments import lex
    from pygments.lexers import get_lexer_by_name, guess_lexer_for_filename
    from pygments.token import Token
    PYGMENTS_AVAILABLE = True
except ImportError:
    PYGMENTS_AVAILABLE = False
    logger.warning("Pygments not available, syntax highlighting disabled")


class SyntaxHighlighter(QSyntaxHighlighter):
    """Syntax highlighter using Pygments.
    
    Supports multiple languages through Pygments lexers.
    Falls back to basic highlighting if Pygments is not available.
    """
    
    # Color schemes
    LIGHT_THEME = {
        Token.Keyword: '#0000FF',
        Token.Keyword.Constant: '#0000FF',
        Token.Keyword.Declaration: '#0000FF',
        Token.Keyword.Namespace: '#0000FF',
        Token.Keyword.Pseudo: '#0000FF',
        Token.Keyword.Reserved: '#0000FF',
        Token.Keyword.Type: '#0000FF',
        
        Token.Name.Class: '#2B91AF',
        Token.Name.Function: '#74531F',
        Token.Name.Namespace: '#2B91AF',
        Token.Name.Tag: '#0000FF',
        Token.Name.Variable: '#1F377F',
        
        Token.String: '#A31515',
        Token.String.Doc: '#008000',
        Token.String.Escape: '#FF00FF',
        Token.String.Interpol: '#FF00FF',
        
        Token.Number: '#098658',
        Token.Number.Float: '#098658',
        Token.Number.Hex: '#098658',
        Token.Number.Integer: '#098658',
        
        Token.Comment: '#008000',
        Token.Comment.Preproc: '#0000FF',
        Token.Comment.Special: '#FF0000',
        
        Token.Operator: '#000000',
        Token.Operator.Word: '#0000FF',
        
        Token.Punctuation: '#000000',
    }
    
    DARK_THEME = {
        Token.Keyword: '#569CD6',
        Token.Keyword.Constant: '#569CD6',
        Token.Keyword.Declaration: '#569CD6',
        Token.Keyword.Namespace: '#569CD6',
        Token.Keyword.Pseudo: '#569CD6',
        Token.Keyword.Reserved: '#569CD6',
        Token.Keyword.Type: '#4EC9B0',
        
        Token.Name.Class: '#4EC9B0',
        Token.Name.Function: '#DCDCAA',
        Token.Name.Namespace: '#4EC9B0',
        Token.Name.Tag: '#569CD6',
        Token.Name.Variable: '#9CDCFE',
        
        Token.String: '#CE9178',
        Token.String.Doc: '#6A9955',
        Token.String.Escape: '#D7BA7D',
        Token.String.Interpol: '#D7BA7D',
        
        Token.Number: '#B5CEA8',
        Token.Number.Float: '#B5CEA8',
        Token.Number.Hex: '#B5CEA8',
        Token.Number.Integer: '#B5CEA8',
        
        Token.Comment: '#6A9955',
        Token.Comment.Preproc: '#569CD6',
        Token.Comment.Special: '#FF0000',
        
        Token.Operator: '#D4D4D4',
        Token.Operator.Word: '#569CD6',
        
        Token.Punctuation: '#D4D4D4',
    }
    
    # Language to Pygments lexer mapping
    LANGUAGE_MAP = {
        'python': 'python',
        'java': 'java',
        'javascript': 'javascript',
        'typescript': 'typescript',
        'go': 'go',
        'rust': 'rust',
        'cpp': 'cpp',
        'c': 'c',
        'csharp': 'csharp',
        'ruby': 'ruby',
        'php': 'php',
        'html': 'html',
        'css': 'css',
        'sql': 'sql',
        'yaml': 'yaml',
        'json': 'json',
        'xml': 'xml',
        'markdown': 'markdown',
        'shell': 'bash',
        'bash': 'bash',
        'powershell': 'powershell',
    }
    
    def __init__(self, parent=None, language: str = '', dark_mode: bool = False):
        super().__init__(parent)
        self._language = language
        self._dark_mode = dark_mode
        self._lexer = None
        self._theme = self.DARK_THEME if dark_mode else self.LIGHT_THEME
        
        self._formats: Dict[Token, QTextCharFormat] = {}
        self._setup_formats()
        
        if PYGMENTS_AVAILABLE and language:
            self.set_language(language)
    
    def _setup_formats(self):
        """Setup text formats for tokens."""
        for token, color in self._theme.items():
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(color))
            
            # Set font style based on token type
            if token in Token.Comment:
                fmt.setFontItalic(True)
            elif token in Token.Keyword:
                fmt.setFontWeight(QFont.Weight.Bold)
            
            self._formats[token] = fmt
    
    def set_language(self, language: str):
        """Set the programming language.
        
        Args:
            language: Language identifier (e.g., 'python', 'java')
        """
        self._language = language.lower()
        
        if not PYGMENTS_AVAILABLE:
            return
        
        try:
            lexer_name = self.LANGUAGE_MAP.get(self._language, self._language)
            self._lexer = get_lexer_by_name(lexer_name)
            logger.debug(f"Set language: {language}")
        except Exception as e:
            logger.warning(f"Failed to get lexer for {language}: {e}")
            self._lexer = None
    
    def set_dark_mode(self, dark_mode: bool):
        """Set dark mode.
        
        Args:
            dark_mode: Whether to use dark theme
        """
        self._dark_mode = dark_mode
        self._theme = self.DARK_THEME if dark_mode else self.LIGHT_THEME
        self._setup_formats()
        self.rehighlight()
    
    def highlightBlock(self, text: str):
        """Highlight a block of text.
        
        Args:
            text: Text to highlight
        """
        if not PYGMENTS_AVAILABLE or not self._lexer:
            # Fallback: basic highlighting
            self._basic_highlight(text)
            return
        
        try:
            index = 0
            for token, value in lex(text, self._lexer):
                length = len(value)
                
                # Find the best matching format
                fmt = self._get_format(token)
                if fmt:
                    self.setFormat(index, length, fmt)
                
                index += length
        except Exception as e:
            logger.debug(f"Highlighting error: {e}")
    
    def _get_format(self, token) -> Optional[QTextCharFormat]:
        """Get format for a token.
        
        Args:
            token: Pygments token
            
        Returns:
            Text format or None
        """
        # Try exact match first
        if token in self._formats:
            return self._formats[token]
        
        # Try parent token types
        while token:
            if token in self._formats:
                return self._formats[token]
            token = token.parent
        
        return None
    
    def _basic_highlight(self, text: str):
        """Basic highlighting without Pygments.
        
        Args:
            text: Text to highlight
        """
        # Simple string highlighting
        string_fmt = QTextCharFormat()
        string_fmt.setForeground(QColor('#CE9178' if self._dark_mode else '#A31515'))
        
        # Find strings
        import re
        for match in re.finditer(r'["\'](.*?)["\']', text):
            self.setFormat(match.start(), match.end() - match.start(), string_fmt)
        
        # Simple comment highlighting
        comment_fmt = QTextCharFormat()
        comment_fmt.setForeground(QColor('#6A9955' if self._dark_mode else '#008000'))
        comment_fmt.setFontItalic(True)
        
        # Find comments
        for match in re.finditer(r'(#|//).*$', text):
            self.setFormat(match.start(), match.end() - match.start(), comment_fmt)
    
    def guess_language(self, filename: str):
        """Guess language from filename.
        
        Args:
            filename: Filename to guess from
        """
        if not PYGMENTS_AVAILABLE:
            return
        
        try:
            self._lexer = guess_lexer_for_filename(filename, '')
            logger.debug(f"Guessed language from {filename}: {self._lexer.name}")
        except Exception as e:
            logger.debug(f"Failed to guess language: {e}")
            self._lexer = None


class CodeEditorStyle:
    """Style configuration for code editor."""
    
    @staticmethod
    def get_light_style() -> str:
        """Get light theme stylesheet."""
        return """
            QTextEdit {
                background-color: #FFFFFF;
                color: #000000;
                border: none;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 13px;
                line-height: 1.5;
            }
        """
    
    @staticmethod
    def get_dark_style() -> str:
        """Get dark theme stylesheet."""
        return """
            QTextEdit {
                background-color: #1E1E1E;
                color: #D4D4D4;
                border: none;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 13px;
                line-height: 1.5;
            }
        """
