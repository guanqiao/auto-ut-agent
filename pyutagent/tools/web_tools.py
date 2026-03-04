"""Web tools for internet search and browsing.

This module provides:
- WebSearchTool: Search the web using various search engines
- WebFetchTool: Fetch and parse web pages
"""

import asyncio
import json
import logging
import re
import urllib.parse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .tool import (
    Tool,
    ToolCategory,
    ToolDefinition,
    ToolResult,
    create_tool_parameter
)

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Web search result."""
    title: str
    url: str
    snippet: str
    source: str  # search engine


class WebSearchTool(Tool):
    """Web search tool using multiple search engines.

    Supports:
    - DuckDuckGo (default, no API key)
    - Bing (requires API key)
    - Google (requires API key)
    """

    def __init__(
        self,
        default_engine: str = "duckduckgo",
        api_keys: Optional[Dict[str, str]] = None
    ):
        """Initialize web search tool.

        Args:
            default_engine: Default search engine
            api_keys: API keys for search engines
        """
        super().__init__()
        self._definition = ToolDefinition(
            name="search_web",
            description="Search the web for information. Returns search results with titles, URLs, and snippets.",
            category=ToolCategory.WEB,
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
                    description="Number of results to return (1-10). Default: 5",
                    required=False,
                    default=5
                ),
                create_tool_parameter(
                    name="engine",
                    param_type="string",
                    description="Search engine: duckduckgo, bing, google. Default: duckduckgo",
                    required=False,
                    default="duckduckgo"
                )
            ],
            examples=[
                {
                    "params": {"query": "Python asyncio tutorial"},
                    "description": "Search for Python asyncio tutorial"
                },
                {
                    "params": {"query": "best practices", "num_results": 3},
                    "description": "Get top 3 results"
                }
            ],
            tags=["web", "search", "internet"]
        )

        self.default_engine = default_engine
        self.api_keys = api_keys or {}

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, **kwargs) -> ToolResult:
        """Execute web search."""
        query = kwargs.get("query")
        num_results = min(max(kwargs.get("num_results", 5), 1), 10)
        engine = kwargs.get("engine", self.default_engine)

        if not query:
            return ToolResult(success=False, error="Search query is required")

        try:
            if engine == "duckduckgo":
                results = await self._search_duckduckgo(query, num_results)
            elif engine == "bing":
                results = await self._search_bing(query, num_results)
            elif engine == "google":
                results = await self._search_google(query, num_results)
            else:
                return ToolResult(success=False, error=f"Unknown search engine: {engine}")

            logger.info(f"[WebSearchTool] Found {len(results)} results for: {query[:50]}...")

            return ToolResult(
                success=True,
                output=[r.__dict__ for r in results],
                metadata={
                    "query": query,
                    "engine": engine,
                    "count": len(results)
                }
            )

        except Exception as e:
            logger.exception(f"[WebSearchTool] Search failed: {e}")
            return ToolResult(success=False, error=f"Search failed: {str(e)}")

    async def _search_duckduckgo(
        self,
        query: str,
        num_results: int
    ) -> List[SearchResult]:
        """Search using DuckDuckGo HTML interface.

        Args:
            query: Search query
            num_results: Number of results

        Returns:
            List of search results
        """
        import aiohttp

        encoded_query = urllib.parse.quote(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=10) as response:
                html = await response.text()

        return self._parse_duckduckgo_results(html, num_results)

    def _parse_duckduckgo_results(
        self,
        html: str,
        num_results: int
    ) -> List[SearchResult]:
        """Parse DuckDuckGo HTML results.

        Args:
            html: HTML content
            num_results: Number of results to extract

        Returns:
            List of search results
        """
        results = []

        # Simple regex-based parsing
        # DuckDuckGo result structure
        result_pattern = r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>'
        snippet_pattern = r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>'

        urls = re.findall(result_pattern, html)
        snippets = re.findall(snippet_pattern, html, re.DOTALL)

        for i in range(min(len(urls), len(snippets), num_results)):
            url = urls[i][0] if isinstance(urls[i], tuple) else urls[i]
            title = urls[i][1] if isinstance(urls[i], tuple) else urls[i]
            snippet = snippets[i]

            # Clean up HTML tags
            title = re.sub(r'<[^>]+>', '', title)
            snippet = re.sub(r'<[^>]+>', '', snippet)

            results.append(SearchResult(
                title=title.strip(),
                url=url.strip(),
                snippet=snippet.strip(),
                source="duckduckgo"
            ))

        return results

    async def _search_bing(
        self,
        query: str,
        num_results: int
    ) -> List[SearchResult]:
        """Search using Bing API.

        Args:
            query: Search query
            num_results: Number of results

        Returns:
            List of search results
        """
        import aiohttp

        api_key = self.api_keys.get("bing")
        if not api_key:
            raise ValueError("Bing API key not configured")

        endpoint = "https://api.bing.microsoft.com/v7.0/search"
        headers = {"Ocp-Apim-Subscription-Key": api_key}
        params = {
            "q": query,
            "count": num_results,
            "textDecorations": False,
            "textFormat": "HTML"
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(endpoint, headers=headers, params=params) as response:
                data = await response.json()

        results = []
        for item in data.get("webPages", {}).get("value", []):
            results.append(SearchResult(
                title=item.get("name", ""),
                url=item.get("url", ""),
                snippet=item.get("snippet", ""),
                source="bing"
            ))

        return results

    async def _search_google(
        self,
        query: str,
        num_results: int
    ) -> List[SearchResult]:
        """Search using Google Custom Search API.

        Args:
            query: Search query
            num_results: Number of results

        Returns:
            List of search results
        """
        import aiohttp

        api_key = self.api_keys.get("google")
        cx = self.api_keys.get("google_cx")

        if not api_key or not cx:
            raise ValueError("Google API key or CX not configured")

        endpoint = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": api_key,
            "cx": cx,
            "q": query,
            "num": num_results
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(endpoint, params=params) as response:
                data = await response.json()

        results = []
        for item in data.get("items", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                source="google"
            ))

        return results


class WebFetchTool(Tool):
    """Fetch and parse web pages."""

    def __init__(
        self,
        max_content_length: int = 10000,
        timeout: int = 30
    ):
        """Initialize web fetch tool.

        Args:
            max_content_length: Maximum content length to fetch
            timeout: Request timeout in seconds
        """
        super().__init__()
        self._definition = ToolDefinition(
            name="fetch_web",
            description="Fetch and parse content from a web page. Returns the page content in markdown format.",
            category=ToolCategory.WEB,
            parameters=[
                create_tool_parameter(
                    name="url",
                    param_type="string",
                    description="URL to fetch",
                    required=True
                ),
                create_tool_parameter(
                    name="max_length",
                    param_type="integer",
                    description="Maximum content length. Default: 10000",
                    required=False,
                    default=10000
                )
            ],
            examples=[
                {
                    "params": {"url": "https://example.com"},
                    "description": "Fetch example.com content"
                }
            ],
            tags=["web", "fetch", "browser"]
        )

        self.max_content_length = max_content_length
        self.timeout = timeout

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, **kwargs) -> ToolResult:
        """Execute web fetch."""
        url = kwargs.get("url")
        max_length = kwargs.get("max_length", self.max_content_length)

        if not url:
            return ToolResult(success=False, error="URL is required")

        try:
            content = await self._fetch_url(url, max_length)

            logger.info(f"[WebFetchTool] Fetched {len(content)} chars from: {url[:50]}...")

            return ToolResult(
                success=True,
                output=content,
                metadata={
                    "url": url,
                    "length": len(content),
                    "truncated": len(content) >= max_length
                }
            )

        except Exception as e:
            logger.exception(f"[WebFetchTool] Fetch failed: {e}")
            return ToolResult(success=False, error=f"Failed to fetch URL: {str(e)}")

    async def _fetch_url(self, url: str, max_length: int) -> str:
        """Fetch URL content.

        Args:
            url: URL to fetch
            max_length: Maximum content length

        Returns:
            Content as markdown
        """
        import aiohttp

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=self.timeout) as response:
                content_type = response.headers.get("Content-Type", "")

                if "text/html" in content_type:
                    html = await response.text()
                    return self._html_to_markdown(html, max_length)
                elif "application/json" in content_type:
                    json_data = await response.json()
                    return json.dumps(json_data, indent=2)[:max_length]
                else:
                    text = await response.text()
                    return text[:max_length]

    def _html_to_markdown(self, html: str, max_length: int) -> str:
        """Convert HTML to markdown.

        Args:
            html: HTML content
            max_length: Maximum length

        Returns:
            Markdown content
        """
        # Simple HTML to markdown conversion
        # Remove script and style tags
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)

        # Convert common tags
        html = re.sub(r'<h1[^>]*>(.*?)</h1>', r'# \1\n\n', html, flags=re.DOTALL)
        html = re.sub(r'<h2[^>]*>(.*?)</h2>', r'## \1\n\n', html, flags=re.DOTALL)
        html = re.sub(r'<h3[^>]*>(.*?)</h3>', r'### \1\n\n', html, flags=re.DOTALL)
        html = re.sub(r'<p[^>]*>(.*?)</p>', r'\1\n\n', html, flags=re.DOTALL)
        html = re.sub(r'<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>', r'[\2](\1)', html, flags=re.DOTALL)
        html = re.sub(r'<code[^>]*>(.*?)</code>', r'`\1`', html, flags=re.DOTALL)
        html = re.sub(r'<pre[^>]*>(.*?)</pre>', r'```\n\1\n```', html, flags=re.DOTALL)
        html = re.sub(r'<li[^>]*>(.*?)</li>', r'- \1\n', html, flags=re.DOTALL)

        # Remove remaining tags
        html = re.sub(r'<[^>]+>', '', html)

        # Clean up
        html = re.sub(r'\n\n+', '\n\n', html)
        html = html.strip()

        # Truncate if needed
        if len(html) > max_length:
            html = html[:max_length] + "\n\n[Content truncated...]"

        return html


def get_web_tools(
    search_api_keys: Optional[Dict[str, str]] = None
) -> List[Tool]:
    """Get all web tools.

    Args:
        search_api_keys: API keys for search engines

    Returns:
        List of web tool instances
    """
    return [
        WebSearchTool(api_keys=search_api_keys),
        WebFetchTool()
    ]
