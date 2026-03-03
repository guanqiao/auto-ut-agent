"""Agent Helpers - Utility and helper methods."""

import logging
import hashlib
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pyutagent.core.config import get_settings

logger = logging.getLogger(__name__)


class AgentHelpers:
    """Helper methods for ReActAgent.
    
    Provides utility functions for:
    - Code extraction
    - Test failure parsing
    - Coverage analysis
    - File operations
    - Embedding generation
    - Build tool info
    """
    
    def __init__(self, agent_core: Any, components: Dict[str, Any]):
        """Initialize helpers.
        
        Args:
            agent_core: AgentCore instance
            components: Dictionary of all components
        """
        self.agent_core = agent_core
        self.components = components
        
        logger.debug("[AgentHelpers] Initialized")
    
    def get_build_tool_info(self) -> Dict[str, Any]:
        """Get information about the detected build tool.
        
        Returns:
            Build tool information
        """
        if self.components.get("_build_tool_info_cache") is not None:
            return self.components["_build_tool_info_cache"]
        
        build_tool_info = self.components["build_tool_info"]
        self.components["_build_tool_info_cache"] = {
            "tool_type": build_tool_info.tool_type.name,
            "version": build_tool_info.version,
            "config_file": str(build_tool_info.config_file) if build_tool_info.config_file else None,
            "wrapper_available": build_tool_info.wrapper_available,
            "executable_path": build_tool_info.executable_path
        }
        return self.components["_build_tool_info_cache"]
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using available embedding model.
        
        Uses caching to avoid recomputing embeddings for the same text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if not available
        """
        import struct
        
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self.components["_embedding_cache"]:
            return self.components["_embedding_cache"][cache_key]
        
        try:
            if hasattr(self.agent_core, 'embedding_model') and self.agent_core.embedding_model:
                embedding = self.agent_core.embedding_model.embed(text)
                self.components["_embedding_cache"][cache_key] = embedding
                return embedding
            
            hash_bytes = hashlib.sha256(text.encode()).digest()
            embedding = []
            for i in range(0, min(len(hash_bytes) * 8, 384), 4):
                if i + 4 <= len(hash_bytes):
                    val = struct.unpack('f', hash_bytes[i:i+4])[0]
                else:
                    val = 0.0
                embedding.append(val)
            
            while len(embedding) < 384:
                embedding.append(0.0)
            
            embedding = embedding[:384]
            self.components["_embedding_cache"][cache_key] = embedding
            return embedding
        except Exception as e:
            logger.debug(f"[AgentHelpers] Embedding generation failed: {e}")
            return None
    
    def get_uncovered_info(self, report) -> Dict[str, Any]:
        """Get information about uncovered code.
        
        Args:
            report: Coverage report
            
        Returns:
            Dictionary with uncovered methods, lines, and branches
        """
        uncovered_info = {
            "methods": [],
            "lines": [],
            "branches": []
        }
        
        if report and report.files:
            for file_coverage in report.files:
                for line_num, is_covered in file_coverage.lines:
                    if not is_covered:
                        uncovered_info["lines"].append(line_num)
        
        logger.debug(f"[AgentHelpers] Uncovered info - Lines: {len(uncovered_info['lines'])}")
        return uncovered_info
    
    def parse_test_failures(self) -> List[Dict[str, Any]]:
        """Parse test failures from Maven output.
        
        Returns:
            List of test failures
        """
        failures = []
        settings = get_settings()
        surefire_dir = Path(self.agent_core.project_path) / settings.project_paths.target_surefire_reports
        
        if surefire_dir.exists():
            for report_file in surefire_dir.glob("*.txt"):
                content = report_file.read_text()
                if "FAILURE" in content or "ERROR" in content:
                    failures.append({
                        "test_name": report_file.stem,
                        "error": content[:500]
                    })
        
        logger.debug(f"[AgentHelpers] Parsed test failures - Failures: {len(failures)}")
        return failures
    
    async def run_tests_with_build_tool(
        self,
        test_class: Optional[str] = None,
        test_method: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run tests using the detected build tool.
        
        Args:
            test_class: Specific test class to run
            test_method: Specific test method to run
            
        Returns:
            Test results
        """
        if not self.components.get("build_runner"):
            logger.error("[AgentHelpers] No build runner available")
            return {"success": False, "error": "No build runner available"}
        
        try:
            build_runner = self.components["build_runner"]
            result = await build_runner.run_tests(
                test_class=test_class,
                test_method=test_method
            )
            
            logger.info(f"[AgentHelpers] Build tool test run - "
                       f"Success: {result.success}, "
                       f"Tests: {result.test_count}, "
                       f"Failures: {result.failure_count}")
            
            return {
                "success": result.success,
                "test_count": result.test_count,
                "failure_count": result.failure_count,
                "error_count": result.error_count,
                "skipped_count": result.skipped_count,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except Exception as e:
            logger.error(f"[AgentHelpers] Build tool test run failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def semantic_search_code(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Perform semantic search over code using vector store.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching code snippets
        """
        if not self.components.get("vector_store"):
            logger.warning("[AgentHelpers] VectorStore not available for semantic search")
            return []
        
        try:
            query_embedding = self._generate_embedding(query)
            if query_embedding is None:
                logger.warning("[AgentHelpers] Could not generate embedding for query")
                return []
            
            vector_store = self.components["vector_store"]
            results = vector_store.search(query_embedding=query_embedding, k=limit)
            
            logger.info(f"[AgentHelpers] Semantic search - "
                       f"Query: '{query[:50]}...', Results: {len(results)}")
            
            return [
                {
                    "id": r.id,
                    "content": r.text,
                    "score": r.score,
                    "metadata": r.metadata
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"[AgentHelpers] Semantic search failed: {e}")
            return []
    
    async def index_code_for_search(
        self,
        code: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Index code snippet for semantic search.
        
        Args:
            code: Code snippet to index
            metadata: Additional metadata
            
        Returns:
            True if indexed successfully
        """
        if not self.components.get("vector_store"):
            return False
        
        try:
            embedding = self._generate_embedding(code)
            if embedding is None:
                logger.warning("[AgentHelpers] Could not generate embedding for code")
                return False
            
            vector_store = self.components["vector_store"]
            vector_store.add(
                texts=[code],
                embeddings=[embedding],
                metadatas=[metadata or {}]
            )
            
            logger.debug(f"[AgentHelpers] Indexed code snippet - Length: {len(code)}")
            return True
        except Exception as e:
            logger.error(f"[AgentHelpers] Failed to index code: {e}")
            return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get collected performance metrics.
        
        Returns:
            Performance metrics summary
        """
        try:
            metrics_collector = self.components["metrics_collector"]
            report = metrics_collector.generate_report()
            
            logger.debug(f"[AgentHelpers] Performance metrics - "
                        f"Operations: {report.get('total_operations', 0)}")
            
            return report
        except Exception as e:
            logger.error(f"[AgentHelpers] Failed to get performance metrics: {e}")
            return {"error": str(e)}
    
    def get_adaptive_strategy_recommendation(
        self,
        error_category: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get strategy recommendation from adaptive strategy manager.
        
        Args:
            error_category: Category of the error
            context: Additional context for strategy selection
            
        Returns:
            Strategy recommendation with confidence
        """
        try:
            from pyutagent.core.parallel_recovery import RecoveryStrategy
            
            available_strategies = list(RecoveryStrategy)
            
            adaptive_strategy_manager = self.components["adaptive_strategy_manager"]
            selected_strategy = adaptive_strategy_manager.select_strategy(
                error_category=error_category,
                available_strategies=available_strategies,
                context=context or {},
                allow_exploration=True
            )
            
            logger.info(f"[AgentHelpers] Adaptive strategy recommendation - "
                       f"Strategy: {selected_strategy.name}, "
                       f"Category: {error_category}")
            
            return {
                "strategy": selected_strategy.name,
                "strategy_value": selected_strategy.value,
                "confidence": 0.8
            }
        except Exception as e:
            logger.error(f"[AgentHelpers] Failed to get strategy recommendation: {e}")
            return {"strategy": "DEFAULT", "confidence": 0.0, "error": str(e)}
    
    def record_strategy_outcome(
        self,
        strategy_name: str,
        success: bool,
        execution_time_ms: float,
        error_category: Optional[str] = None
    ) -> None:
        """Record the outcome of a strategy execution.
        
        Args:
            strategy_name: Name of the strategy
            success: Whether it succeeded
            execution_time_ms: Execution time in milliseconds
            error_category: Category of the error being handled
        """
        try:
            from pyutagent.core.parallel_recovery import RecoveryStrategy
            
            try:
                strategy = RecoveryStrategy[strategy_name.upper()]
            except KeyError:
                strategy = RecoveryStrategy.DEFAULT
            
            adaptive_strategy_manager = self.components["adaptive_strategy_manager"]
            adaptive_strategy_manager.record_attempt(
                strategy=strategy,
                error_category=error_category or "unknown",
                success=success,
                execution_time_ms=execution_time_ms,
                context={}
            )
            
            logger.debug(f"[AgentHelpers] Recorded strategy outcome - "
                        f"Strategy: {strategy_name}, Success: {success}")
        except Exception as e:
            logger.error(f"[AgentHelpers] Failed to record strategy outcome: {e}")
