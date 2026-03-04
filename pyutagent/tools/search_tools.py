"""Search and web tools for enhanced capabilities.

This module provides:
- WebSearchTool: Search the web for information
- WebFetchTool: Fetch and parse web pages
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .tool import (
    Tool,
    ToolCategory,
    ToolDefinition,
    ToolResult,
    create_tool_parameter,
)

logger = logging.getLogger(__name__)


class WebSearchTool(Tool):
    """Tool for searching the web."""
    
    def __init__(self, timeout: int = 30):
        """Initialize web search tool.
        
        Args:
            timeout: Request timeout in seconds
        """
        super().__init__()
        self._timeout = timeout
        self._definition = ToolDefinition(
            name="web_search",
            description="Search the web for information. Use this to find documentation, "
                       "solutions to problems, or general knowledge.",
            category=ToolCategory.SEARCH,
            parameters=[
                create_tool_parameter(
                    name="query",
                    param_type="string",
                    description="Search query",
                    required=True
                ),
                create_tool_parameter(
                    name="num_results",
                    param_type="integer",
                    description="Number of results to return",
                    required=False,
                    default=5
                ),
                create_tool_parameter(
                    name="engine",
                    param_type="string",
                    description="Search engine to use (google, bing, duckduckgo)",
                    required=False,
                    default="duckduckgo"
                )
            ],
            examples=[
                {
                    "params": {"query": "JUnit 5 mockito example"},
                    "description": "Search for JUnit 5 mockito examples"
                },
                {
                    "params": {"query": "Java 17 new features", "num_results": 3},
                    "description": "Search for Java 17 features"
                }
            ],
            tags=["search", "web", "internet", "documentation"]
        )
    
    @property
    def definition(self) -> ToolDefinition:
        return self._definition
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute web search."""
        query = kwargs.get("query")
        num_results = kwargs.get("num_results", 5)
        engine = kwargs.get("engine", "duckduckgo")
        
        if not query:
            return ToolResult(success=False, error="Query is required")
        
        try:
            if engine == "duckduckgo":
                return await self._search_duckduckgo(query, num_results)
            elif engine == "google":
                return await self._search_google(query, num_results)
            elif engine == "bing":
                return await self._search_bing(query, num_results)
            else:
                return await self._search_duckduckgo(query, num_results)
                
        except Exception as e:
            logger.exception(f"[WebSearchTool] Search failed: {e}")
            return ToolResult(success=False, error=f"Search failed: {str(e)}")
    
    async def _search_duckduckgo(self, query: str, num_results: int) -> ToolResult:
        """Search using DuckDuckGo."""
        try:
            import aiohttp
            
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=self._timeout) as response:
                    if response.status != 200:
                        return ToolResult(success=False, error=f"HTTP {response.status}")
                    
                    data = await response.json()
                    
                    results = []
                    for item in data.get("RelatedTopics", [])[:num_results]:
                        if "Text" in item and "FirstURL" in item:
                            results.append({
                                "title": item.get("Text", "").split(" - ")[0][:100],
                                "url": item.get("FirstURL", ""),
                                "snippet": item.get("Text", "")[:200]
                            })
                    
                    if not results:
                        results = [{"message": "No results found", "query": query}]
                    
                    return ToolResult(
                        success=True,
                        output=json.dumps(results, indent=2, ensure_ascii=False),
                        metadata={"query": query, "engine": "duckduckgo", "count": len(results)}
                    )
                    
        except ImportError:
            return ToolResult(
                success=False,
                error="aiohttp not installed. Install with: pip install aiohttp"
            )
        except Exception as e:
            return ToolResult(success=False, error=f"DuckDuckGo search failed: {str(e)}")
    
    async def _search_google(self, query: str, num_results: int) -> ToolResult:
        """Search using Google (requires pip install google-search-results)."""
        return ToolResult(
            success=False,
            error="Google search not implemented. Use duckduckgo or bing."
        )
    
    async def _search_bing(self, query: str, num_results: int) -> ToolResult:
        """Search using Bing (requires API key)."""
        return ToolResult(
            success=False,
            error="Bing search requires API key. Use duckduckgo for free search."
        )


class WebFetchTool(Tool):
    """Tool for fetching web pages."""
    
    def __init__(self, timeout: int = 30):
        """Initialize web fetch tool.
        
        Args:
            timeout: Request timeout in seconds
        """
        super().__init__()
        self._timeout = timeout
        self._definition = ToolDefinition(
            name="web_fetch",
            description="Fetch and extract content from a web page. "
                       "Use this to read documentation, articles, or any online resource.",
            category=ToolCategory.SEARCH,
            parameters=[
                create_tool_parameter(
                    name="url",
                    param_type="string",
                    description="URL to fetch",
                    required=True
                ),
                create_tool_parameter(
                    name="extract_text",
                    param_type="boolean",
                    description="Extract only text content (no HTML)",
                    required=False,
                    default=True
                ),
                create_tool_parameter(
                    name="max_length",
                    param_type="integer",
                    description="Maximum characters to return",
                    required=False,
                    default=5000
                )
            ],
            examples=[
                {
                    "params": {"url": "https://junit.org/junit5/docs/current/user-guide/"},
                    "description": "Fetch JUnit 5 documentation"
                },
                {
                    "params": {"url": "https://github.com", "max_length": 2000},
                    "description": "Fetch GitHub homepage (limited)"
                }
            ],
            tags=["web", "fetch", "download", "read"]
        )
    
    @property
    def definition(self) -> ToolDefinition:
        return self._definition
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute web fetch."""
        url = kwargs.get("url")
        extract_text = kwargs.get("extract_text", True)
        max_length = kwargs.get("max_length", 5000)
        
        if not url:
            return ToolResult(success=False, error="URL is required")
        
        try:
            import aiohttp
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=self._timeout) as response:
                    if response.status != 200:
                        return ToolResult(success=False, error=f"HTTP {response.status}")
                    
                    content = await response.text()
                    
                    if extract_text:
                        content = self._extract_text(content, max_length)
                    else:
                        content = content[:max_length]
                    
                    return ToolResult(
                        success=True,
                        output=content,
                        metadata={
                            "url": url,
                            "original_length": len(content),
                            "returned_length": len(content)
                        }
                    )
                    
        except ImportError:
            return ToolResult(
                success=False,
                error="aiohttp not installed. Install with: pip install aiohttp"
            )
        except Exception as e:
            logger.exception(f"[WebFetchTool] Fetch failed: {e}")
            return ToolResult(success=False, error=f"Fetch failed: {str(e)}")
    
    def _extract_text(self, html: str, max_length: int) -> str:
        """Extract text from HTML."""
        import re
        
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        return text


def get_all_search_tools() -> List[Tool]:
    """Get all search tools.
    
    Returns:
        List of search tool instances
    """
    return [
        WebSearchTool(),
        WebFetchTool(),
    ]
