"""Semantic Search Tool - AI-enhanced grep with understanding.

This module provides:
- SemanticGrepTool: Understand natural language queries
- Context-aware search results
- LLM-powered result interpretation
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..llm.client import LLMClient
from .tool import Tool, ToolDefinition, ToolCategory, ToolResult, create_tool_parameter
from .standard_tools import GrepTool

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A search result with context."""
    file: str
    line: int
    content: str
    context_before: List[str] = field(default_factory=list)
    context_after: List[str] = field(default_factory=list)
    relevance_score: float = 1.0


class SemanticGrepTool(Tool):
    """AI-enhanced semantic search tool.

    Features:
    - Natural language query understanding
    - Context-aware results
    - LLM-powered result interpretation
    - Smart query expansion
    """

    def __init__(
        self,
        base_path: Optional[str] = None,
        llm_client: Optional[LLMClient] = None
    ):
        """Initialize semantic grep tool.

        Args:
            base_path: Base path for search
            llm_client: LLM client for semantic understanding
        """
        super().__init__()
        self._base_path = Path(base_path) if base_path else None
        self._llm_client = llm_client
        self._definition = ToolDefinition(
            name="semantic_grep",
            description="AI-enhanced semantic search. Understands natural language queries "
                       "and returns context-aware results with explanations.",
            category=ToolCategory.SEARCH,
            parameters=[
                create_tool_parameter(
                    name="query",
                    param_type="string",
                    description="Natural language search query (e.g., 'find authentication logic')",
                    required=True
                ),
                create_tool_parameter(
                    name="path",
                    param_type="string",
                    description="Directory or file path to search in",
                    required=False,
                    default=None
                ),
                create_tool_parameter(
                    name="language",
                    param_type="string",
                    description="Programming language for context (java, python, etc.)",
                    required=False,
                    default=None
                ),
                create_tool_parameter(
                    name="max_results",
                    param_type="integer",
                    description="Maximum number of results",
                    required=False,
                    default=10
                ),
                create_tool_parameter(
                    name="context_lines",
                    param_type="integer",
                    description="Number of context lines before/after",
                    required=False,
                    default=3
                ),
                create_tool_parameter(
                    name="interpret_results",
                    param_type="boolean",
                    description="Use LLM to interpret results",
                    required=False,
                    default=True
                )
            ],
            examples=[
                {
                    "params": {"query": "find authentication logic", "language": "java"},
                    "description": "Find authentication related code"
                },
                {
                    "params": {"query": "where is the database connection", "language": "python"},
                    "description": "Find database connection code"
                }
            ],
            tags=["search", "semantic", "ai", "grep", "understand"]
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, **kwargs) -> ToolResult:
        """Execute semantic search."""
        query = kwargs.get("query")
        search_path = kwargs.get("path")
        language = kwargs.get("language")
        max_results = kwargs.get("max_results", 10)
        context_lines = kwargs.get("context_lines", 3)
        interpret_results = kwargs.get("interpret_results", True)

        if not query:
            return ToolResult(success=False, error="query is required")

        try:
            if search_path:
                base = Path(search_path)
            elif self._base_path:
                base = self._base_path
            else:
                base = Path.cwd()

            technical_query = await self._expand_query(query, language or "code")

            results = await self._search(
                technical_query,
                base,
                max_results,
                context_lines
            )

            if interpret_results and results and self._llm_client:
                interpretation = await self._interpret_results(query, results)
            else:
                interpretation = None

            logger.info(f"[SemanticGrepTool] Query: {query}, results: {len(results)}")

            return ToolResult(
                success=True,
                output={
                    "results": [self._result_to_dict(r) for r in results],
                    "query": query,
                    "technical_query": technical_query,
                    "interpretation": interpretation,
                    "total_matches": len(results)
                },
                metadata={
                    "query": query,
                    "interpreted": interpret_results
                }
            )

        except Exception as e:
            logger.exception(f"[SemanticGrepTool] Failed: {e}")
            return ToolResult(success=False, error=str(e))

    async def _expand_query(self, query: str, language: str) -> str:
        """Expand natural language query to technical search terms."""
        if not self._llm_client:
            return query

        try:
            prompt = f"""Convert this natural language search query to technical search terms for {language} code.

Query: {query}

Output only the technical search terms (keywords, function names, patterns), one per line. Be concise."""

            response = await self._llm_client.agenerate(
                prompt=prompt,
                max_tokens=100,
                temperature=0.3
            )

            terms = [line.strip() for line in response.strip().split("\n") if line.strip()]
            return "|".join(terms) if terms else query

        except Exception as e:
            logger.warning(f"[SemanticGrepTool] Query expansion failed: {e}")
            return query

    async def _search(
        self,
        query: str,
        base: Path,
        max_results: int,
        context_lines: int
    ) -> List[SearchResult]:
        """Perform the search."""
        if not base.exists():
            return []

        results = []

        if query.endswith("|"):
            patterns = query[:-1].split("|")
        else:
            patterns = [query]

        files_to_search = []
        if base.is_file():
            files_to_search = [base]
        elif base.is_dir():
            files_to_search = [f for f in base.rglob("*") if f.is_file() and f.suffix in [".py", ".java", ".js", ".ts", ".go", ".rs"]]

        for file_path in files_to_search:
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                lines = content.split("\n")

                for line_num, line in enumerate(lines, start=1):
                    for pattern in patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            context_before = lines[max(0, line_num - context_lines - 1):line_num - 1]
                            context_after = lines[line_num:min(len(lines), line_num + context_lines)]

                            results.append(SearchResult(
                                file=str(file_path.relative_to(base) if base.is_dir() else file_path),
                                line=line_num,
                                content=line.strip(),
                                context_before=context_before,
                                context_after=context_after
                            ))

                            if len(results) >= max_results:
                                return results

            except Exception:
                continue

        return results

    async def _interpret_results(
        self,
        query: str,
        results: List[SearchResult]
    ) -> str:
        """Use LLM to interpret search results."""
        if not self._llm_client:
            return ""

        try:
            context = "\n".join([
                f"{r.file}:{r.line} - {r.content}"
                for r in results[:5]
            ])

            prompt = f"""Given the search query "{query}", analyze these results:

{context}

Provide a brief interpretation of what these results mean and how they relate to the query. Be concise."""

            return await self._llm_client.agenerate(
                prompt=prompt,
                max_tokens=200,
                temperature=0.5
            )

        except Exception as e:
            logger.warning(f"[SemanticGrepTool] Interpretation failed: {e}")
            return ""

    def _result_to_dict(self, result: SearchResult) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "file": result.file,
            "line": result.line,
            "content": result.content,
            "context_before": result.context_before,
            "context_after": result.context_after,
            "relevance_score": result.relevance_score
        }


class SmartGrepTool(GrepTool):
    """Enhanced grep tool with smart features.

    Adds:
    - Context lines
    - Result grouping
    - Multiple output formats
    """

    def __init__(self, base_path: Optional[str] = None):
        super().__init__(base_path)
        self._definition = ToolDefinition(
            name="smart_grep",
            description="Enhanced grep with smart features: context lines, grouping, multiple formats",
            category=ToolCategory.SEARCH,
            parameters=[
                create_tool_parameter(
                    name="pattern",
                    param_type="string",
                    description="Search pattern",
                    required=True
                ),
                create_tool_parameter(
                    name="path",
                    param_type="string",
                    description="Search path",
                    required=False,
                    default=None
                ),
                create_tool_parameter(
                    name="glob",
                    param_type="string",
                    description="File filter",
                    required=False,
                    default=None
                ),
                create_tool_parameter(
                    name="context_lines",
                    param_type="integer",
                    description="Lines of context before/after",
                    required=False,
                    default=3
                ),
                create_tool_parameter(
                    name="group_by_file",
                    param_type="boolean",
                    description="Group results by file",
                    required=False,
                    default=True
                ),
                create_tool_parameter(
                    name="output_format",
                    param_type="string",
                    description="Output format: json, simple, detailed",
                    required=False,
                    default="json"
                )
            ],
            tags=["search", "grep", "smart", "enhanced"]
        )

    async def execute(self, **kwargs) -> ToolResult:
        """Execute enhanced grep."""
        context_lines = kwargs.get("context_lines", 3)
        group_by_file = kwargs.get("group_by_file", True)
        output_format = kwargs.get("output_format", "json")

        base_result = await super().execute(**kwargs)

        if not base_result.success:
            return base_result

        results = base_result.output

        if group_by_file:
            grouped = {}
            for r in results:
                file = r.get("file", "unknown")
                if file not in grouped:
                    grouped[file] = []
                grouped[file].append(r)
            results = grouped

        return ToolResult(
            success=True,
            output=results,
            metadata={"format": output_format}
        )
