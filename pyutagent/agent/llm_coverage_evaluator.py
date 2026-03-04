"""LLM-based coverage evaluator for when JaCoCo is not available.

This module provides coverage estimation using LLM analysis when
traditional coverage tools (JaCoCo, etc.) are not available.
"""

import logging
import re
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from enum import Enum, auto

logger = logging.getLogger(__name__)


class CoverageSource(Enum):
    """Source of coverage data."""
    JACOCO = "jacoco"
    LLM_ESTIMATED = "llm_estimated"
    HYBRID = "hybrid"


@dataclass
class LLMCoverageReport:
    """Coverage report from LLM estimation."""
    line_coverage: float
    branch_coverage: float
    method_coverage: float
    source: CoverageSource = CoverageSource.LLM_ESTIMATED
    uncovered_methods: List[str] = field(default_factory=list)
    uncovered_branches: List[str] = field(default_factory=list)
    covered_methods: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""
    recommendations: List[str] = field(default_factory=list)


class LLMCoverageEvaluator:
    """Evaluates test coverage using LLM when JaCoCo is not available.
    
    This provides a fallback mechanism for projects without JaCoCo
    or when JaCoCo reports are not available.
    """
    
    def __init__(self, llm_client: Any):
        """Initialize the LLM coverage evaluator.
        
        Args:
            llm_client: LLM client for generation
        """
        self.llm_client = llm_client
        logger.info("[LLMCoverageEvaluator] Initialized")
    
    async def evaluate_coverage(
        self,
        source_code: str,
        test_code: str,
        class_info: Optional[Dict[str, Any]] = None
    ) -> LLMCoverageReport:
        """Evaluate test coverage using LLM.
        
        Args:
            source_code: The source code being tested
            test_code: The test code
            class_info: Optional class information from parsing
            
        Returns:
            LLMCoverageReport with estimated coverage
        """
        logger.info("[LLMCoverageEvaluator] Starting coverage evaluation")
        
        methods_info = self._extract_methods_info(source_code, class_info)
        test_methods_info = self._extract_test_methods_info(test_code)
        
        prompt = self._build_evaluation_prompt(
            source_code, test_code, methods_info, test_methods_info
        )
        
        try:
            response = await self.llm_client.agenerate(
                prompt=prompt,
                system_prompt=self._get_system_prompt()
            )
            
            report = self._parse_llm_response(response, methods_info)
            logger.info(
                f"[LLMCoverageEvaluator] Coverage evaluation complete - "
                f"Line: {report.line_coverage:.1%}, "
                f"Branch: {report.branch_coverage:.1%}, "
                f"Method: {report.method_coverage:.1%}"
            )
            return report
            
        except Exception as e:
            logger.exception(f"[LLMCoverageEvaluator] LLM evaluation failed: {e}")
            return self._create_fallback_report(methods_info, test_methods_info)
    
    def evaluate_coverage_sync(
        self,
        source_code: str,
        test_code: str,
        class_info: Optional[Dict[str, Any]] = None
    ) -> LLMCoverageReport:
        """Synchronous version of coverage evaluation.
        
        Args:
            source_code: The source code being tested
            test_code: The test code
            class_info: Optional class information from parsing
            
        Returns:
            LLMCoverageReport with estimated coverage
        """
        logger.info("[LLMCoverageEvaluator] Starting sync coverage evaluation")
        
        methods_info = self._extract_methods_info(source_code, class_info)
        test_methods_info = self._extract_test_methods_info(test_code)
        
        prompt = self._build_evaluation_prompt(
            source_code, test_code, methods_info, test_methods_info
        )
        
        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=self._get_system_prompt()
            )
            
            report = self._parse_llm_response(response, methods_info)
            logger.info(
                f"[LLMCoverageEvaluator] Sync coverage evaluation complete - "
                f"Line: {report.line_coverage:.1%}, "
                f"Branch: {report.branch_coverage:.1%}, "
                f"Method: {report.method_coverage:.1%}"
            )
            return report
            
        except Exception as e:
            logger.exception(f"[LLMCoverageEvaluator] Sync LLM evaluation failed: {e}")
            return self._create_fallback_report(methods_info, test_methods_info)
    
    def quick_estimate(
        self,
        source_code: str,
        test_code: str,
        class_info: Optional[Dict[str, Any]] = None
    ) -> LLMCoverageReport:
        """Quick heuristic-based coverage estimation without LLM.
        
        This is used as a fast fallback when LLM is not available.
        
        Args:
            source_code: The source code being tested
            test_code: The test code
            class_info: Optional class information from parsing
            
        Returns:
            LLMCoverageReport with heuristic coverage estimate
        """
        logger.info("[LLMCoverageEvaluator] Starting quick heuristic estimate")
        
        methods_info = self._extract_methods_info(source_code, class_info)
        test_methods_info = self._extract_test_methods_info(test_code)
        
        return self._create_fallback_report(methods_info, test_methods_info)
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM evaluation."""
        return """你是一个代码覆盖率分析专家。你的任务是分析源代码和测试代码，评估测试覆盖率。

请仔细分析：
1. 源代码中的所有方法（包括公共、私有、静态方法）
2. 源代码中的分支逻辑（if/else、switch、循环等）
3. 测试代码中测试的方法和场景
4. 测试覆盖的边界情况

请返回 JSON 格式的评估结果，包含以下字段：
- line_coverage: 行覆盖率估计值 (0.0-1.0)
- branch_coverage: 分支覆盖率估计值 (0.0-1.0)
- method_coverage: 方法覆盖率估计值 (0.0-1.0)
- covered_methods: 已覆盖的方法名列表
- uncovered_methods: 未覆盖的方法名列表
- uncovered_branches: 未覆盖的分支描述列表
- confidence: 评估置信度 (0.0-1.0)
- reasoning: 评估理由说明
- recommendations: 提高覆盖率的建议列表

只返回 JSON，不要包含其他文字。"""
    
    def _build_evaluation_prompt(
        self,
        source_code: str,
        test_code: str,
        methods_info: Dict[str, Any],
        test_methods_info: Dict[str, Any]
    ) -> str:
        """Build the evaluation prompt for LLM."""
        source_preview = source_code[:3000] if len(source_code) > 3000 else source_code
        test_preview = test_code[:3000] if len(test_code) > 3000 else test_code
        
        prompt = f"""请分析以下代码并评估测试覆盖率：

## 源代码信息

### 方法列表
共 {methods_info.get('total_count', 0)} 个方法：
{self._format_methods_list(methods_info)}

### 源代码
```java
{source_preview}
```

## 测试代码信息

### 测试方法列表
共 {test_methods_info.get('total_count', 0)} 个测试方法：
{self._format_test_methods_list(test_methods_info)}

### 测试代码
```java
{test_preview}
```

请评估测试覆盖率并返回 JSON 格式的结果。"""
        
        return prompt
    
    def _format_methods_list(self, methods_info: Dict[str, Any]) -> str:
        """Format methods list for prompt."""
        methods = methods_info.get('methods', [])
        if not methods:
            return "无方法信息"
        
        lines = []
        for m in methods[:20]:
            name = m.get('name', 'unknown')
            visibility = m.get('visibility', 'unknown')
            params = m.get('params', 0)
            lines.append(f"- {visibility} {name}() [{params} params]")
        
        if len(methods) > 20:
            lines.append(f"... 还有 {len(methods) - 20} 个方法")
        
        return "\n".join(lines)
    
    def _format_test_methods_list(self, test_methods_info: Dict[str, Any]) -> str:
        """Format test methods list for prompt."""
        methods = test_methods_info.get('methods', [])
        if not methods:
            return "无测试方法"
        
        lines = []
        for m in methods[:15]:
            name = m.get('name', 'unknown')
            lines.append(f"- {name}()")
        
        if len(methods) > 15:
            lines.append(f"... 还有 {len(methods) - 15} 个测试方法")
        
        return "\n".join(lines)
    
    def _extract_methods_info(
        self,
        source_code: str,
        class_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Extract method information from source code."""
        methods = []
        
        if class_info and 'methods' in class_info:
            for m in class_info['methods']:
                methods.append({
                    'name': m.get('name', 'unknown'),
                    'visibility': m.get('visibility', 'public'),
                    'params': len(m.get('parameters', [])),
                    'is_static': m.get('is_static', False),
                    'return_type': m.get('return_type', 'void'),
                })
        else:
            method_pattern = re.compile(
                r'(public|private|protected)?\s*(static)?\s*(\w+(?:<[^>]+>)?)\s+(\w+)\s*\(([^)]*)\)',
                re.MULTILINE
            )
            
            for match in method_pattern.finditer(source_code):
                visibility = match.group(1) or 'package'
                is_static = bool(match.group(2))
                return_type = match.group(3)
                name = match.group(4)
                params_str = match.group(5)
                param_count = len([p for p in params_str.split(',') if p.strip()])
                
                if name not in ['if', 'for', 'while', 'switch', 'catch', 'class']:
                    methods.append({
                        'name': name,
                        'visibility': visibility,
                        'params': param_count,
                        'is_static': is_static,
                        'return_type': return_type,
                    })
        
        return {
            'methods': methods,
            'total_count': len(methods),
        }
    
    def _extract_test_methods_info(self, test_code: str) -> Dict[str, Any]:
        """Extract test method information from test code."""
        methods = []
        
        test_method_pattern = re.compile(
            r'@Test\s*(?:\([^)]*\))?\s*(?:public\s+)?void\s+(\w+)\s*\(',
            re.MULTILINE
        )
        
        for match in test_method_pattern.finditer(test_code):
            methods.append({
                'name': match.group(1),
            })
        
        return {
            'methods': methods,
            'total_count': len(methods),
        }
    
    def _parse_llm_response(
        self,
        response: str,
        methods_info: Dict[str, Any]
    ) -> LLMCoverageReport:
        """Parse LLM response into coverage report."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
            
            line_coverage = float(data.get('line_coverage', 0.0))
            branch_coverage = float(data.get('branch_coverage', 0.0))
            method_coverage = float(data.get('method_coverage', 0.0))
            
            line_coverage = max(0.0, min(1.0, line_coverage))
            branch_coverage = max(0.0, min(1.0, branch_coverage))
            method_coverage = max(0.0, min(1.0, method_coverage))
            
            return LLMCoverageReport(
                line_coverage=line_coverage,
                branch_coverage=branch_coverage,
                method_coverage=method_coverage,
                source=CoverageSource.LLM_ESTIMATED,
                uncovered_methods=data.get('uncovered_methods', []),
                uncovered_branches=data.get('uncovered_branches', []),
                covered_methods=data.get('covered_methods', []),
                confidence=float(data.get('confidence', 0.7)),
                reasoning=data.get('reasoning', ''),
                recommendations=data.get('recommendations', []),
            )
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"[LLMCoverageEvaluator] Failed to parse LLM response: {e}")
            return self._create_fallback_report(methods_info, {'methods': [], 'total_count': 0})
    
    def _create_fallback_report(
        self,
        methods_info: Dict[str, Any],
        test_methods_info: Dict[str, Any]
    ) -> LLMCoverageReport:
        """Create a heuristic-based fallback report."""
        methods = methods_info.get('methods', [])
        test_methods = test_methods_info.get('methods', [])
        
        method_names = {m.get('name') for m in methods}
        covered_methods = set()
        uncovered_methods = set()
        
        for test_m in test_methods:
            test_name = test_m.get('name', '').lower()
            for method_name in method_names:
                if method_name and method_name.lower() in test_name:
                    covered_methods.add(method_name)
        
        uncovered_methods = method_names - covered_methods
        
        total_methods = len(methods)
        if total_methods > 0:
            method_coverage = len(covered_methods) / total_methods
        else:
            method_coverage = 0.0
        
        test_count = len(test_methods)
        if total_methods > 0:
            line_coverage = min(1.0, (test_count / total_methods) * 0.7)
        else:
            line_coverage = 0.0
        
        branch_coverage = line_coverage * 0.6
        
        recommendations = []
        if uncovered_methods:
            recommendations.append(f"添加测试覆盖以下方法: {', '.join(list(uncovered_methods)[:5])}")
        if branch_coverage < 0.5:
            recommendations.append("添加边界条件测试以提高分支覆盖率")
        
        return LLMCoverageReport(
            line_coverage=line_coverage,
            branch_coverage=branch_coverage,
            method_coverage=method_coverage,
            source=CoverageSource.LLM_ESTIMATED,
            uncovered_methods=list(uncovered_methods),
            uncovered_branches=[],
            covered_methods=list(covered_methods),
            confidence=0.5,
            reasoning=f"基于启发式估算：{test_count} 个测试方法覆盖 {len(covered_methods)}/{total_methods} 个方法",
            recommendations=recommendations,
        )


def create_llm_coverage_report(
    line_coverage: float,
    branch_coverage: float,
    method_coverage: float,
    source: CoverageSource = CoverageSource.LLM_ESTIMATED,
    **kwargs
) -> LLMCoverageReport:
    """Factory function to create LLM coverage report.
    
    Args:
        line_coverage: Line coverage (0.0-1.0)
        branch_coverage: Branch coverage (0.0-1.0)
        method_coverage: Method coverage (0.0-1.0)
        source: Coverage source
        **kwargs: Additional fields
        
    Returns:
        LLMCoverageReport instance
    """
    return LLMCoverageReport(
        line_coverage=line_coverage,
        branch_coverage=branch_coverage,
        method_coverage=method_coverage,
        source=source,
        **kwargs
    )
