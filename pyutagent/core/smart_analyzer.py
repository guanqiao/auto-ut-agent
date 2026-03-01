"""
P3.5 智能代码分析器 (Smart Code Analyzer)

功能：
1. 代码语义分析 - 理解代码意图和逻辑
2. 依赖关系图 - 分析代码间的依赖关系
3. 变更影响分析 - 评估代码变更的影响范围
4. 智能代码搜索 - 基于语义而非文本的代码搜索
"""

import ast
import re
import json
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple, Callable
import hashlib
from collections import defaultdict


class AnalysisType(Enum):
    """分析类型"""
    SEMANTIC = auto()         # 语义分析
    DEPENDENCY = auto()       # 依赖分析
    IMPACT = auto()           # 影响分析
    SIMILARITY = auto()       # 相似性分析
    COMPLEXITY = auto()       # 复杂度分析


class CodeEntityType(Enum):
    """代码实体类型"""
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    IMPORT = "import"
    DECORATOR = "decorator"


@dataclass
class CodeEntity:
    """代码实体"""
    entity_id: str
    name: str
    entity_type: CodeEntityType
    file_path: str
    line_start: int
    line_end: int
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    signature: Optional[str] = None
    docstring: Optional[str] = None


@dataclass
class DependencyEdge:
    """依赖边"""
    source_id: str
    target_id: str
    dependency_type: str  # "calls", "inherits", "uses", "imports"
    strength: float  # 0-1
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticContext:
    """语义上下文"""
    entity_id: str
    purpose: str  # 代码实体的目的描述
    inputs: List[str]  # 输入参数/数据
    outputs: List[str]  # 输出/返回值
    side_effects: List[str]  # 副作用
    preconditions: List[str]  # 前置条件
    postconditions: List[str]  # 后置条件
    related_concepts: List[str]  # 相关概念/关键词


@dataclass
class ImpactAnalysisResult:
    """影响分析结果"""
    changed_entity_id: str
    directly_affected: List[str]  # 直接受影响的实体
    indirectly_affected: List[str]  # 间接受影响的实体
    test_entities_to_check: List[str]  # 需要检查的测试实体
    risk_score: float  # 0-1，变更风险
    estimated_effort: int  # 估计工作量（分钟）


@dataclass
class CodeSearchResult:
    """代码搜索结果"""
    entity_id: str
    relevance_score: float
    match_type: str  # "semantic", "syntactic", "name"
    matched_concepts: List[str]
    snippet: str


class ASTAnalyzer:
    """AST分析器"""

    def __init__(self):
        self.entities: Dict[str, CodeEntity] = {}
        self.dependencies: List[DependencyEdge] = []

    def analyze_file(self, file_path: str, source_code: str) -> List[CodeEntity]:
        """分析单个文件"""
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            return []

        file_entities = []
        current_class = None

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                entity = self._create_entity_from_class(node, file_path)
                self.entities[entity.entity_id] = entity
                file_entities.append(entity)
                current_class = entity.entity_id

            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                entity = self._create_entity_from_function(node, file_path, current_class)
                self.entities[entity.entity_id] = entity
                file_entities.append(entity)

                if current_class:
                    self.entities[current_class].children_ids.append(entity.entity_id)
                    entity.parent_id = current_class

        # 分析依赖关系
        self._analyze_dependencies(tree, file_entities)

        return file_entities

    def _create_entity_from_class(self, node: ast.ClassDef, file_path: str) -> CodeEntity:
        """从类定义创建实体"""
        entity_id = self._generate_entity_id(file_path, node.name, node.lineno)

        # 获取基类
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(f"{base.value.id}.{base.attr}" if isinstance(base.value, ast.Name) else str(base.attr))

        # 获取文档字符串
        docstring = ast.get_docstring(node)

        return CodeEntity(
            entity_id=entity_id,
            name=node.name,
            entity_type=CodeEntityType.CLASS,
            file_path=file_path,
            line_start=node.lineno,
            line_end=node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
            docstring=docstring,
            metadata={
                'bases': bases,
                'decorators': [self._get_name(d) for d in node.decorator_list],
                'method_count': len([n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))])
            }
        )

    def _create_entity_from_function(
        self,
        node: ast.FunctionDef,
        file_path: str,
        class_id: Optional[str] = None
    ) -> CodeEntity:
        """从函数定义创建实体"""
        name = node.name
        if class_id:
            entity_id = f"{class_id}.{name}"
            entity_type = CodeEntityType.METHOD
        else:
            entity_id = self._generate_entity_id(file_path, name, node.lineno)
            entity_type = CodeEntityType.FUNCTION

        # 获取签名
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {self._get_annotation_name(arg.annotation)}"
            args.append(arg_str)

        # 获取返回类型
        returns = ""
        if node.returns:
            returns = f" -> {self._get_annotation_name(node.returns)}"

        signature = f"def {name}({', '.join(args)}){returns}"
        docstring = ast.get_docstring(node)

        return CodeEntity(
            entity_id=entity_id,
            name=name,
            entity_type=entity_type,
            file_path=file_path,
            line_start=node.lineno,
            line_end=node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
            parent_id=class_id,
            signature=signature,
            docstring=docstring,
            metadata={
                'is_async': isinstance(node, ast.AsyncFunctionDef),
                'decorators': [self._get_name(d) for d in node.decorator_list],
                'arg_count': len(node.args.args),
                'has_return': any(isinstance(n, ast.Return) and n.value for n in ast.walk(node))
            }
        )

    def _analyze_dependencies(self, tree: ast.AST, file_entities: List[CodeEntity]):
        """分析依赖关系"""
        entity_map = {e.line_start: e for e in file_entities}

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # 函数调用
                if isinstance(node.func, ast.Name):
                    # 简单函数调用
                    pass
                elif isinstance(node.func, ast.Attribute):
                    # 方法调用
                    pass

            elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                # 导入语句
                pass

    def _generate_entity_id(self, file_path: str, name: str, line: int) -> str:
        """生成实体ID"""
        unique_str = f"{file_path}:{name}:{line}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:16]

    def _get_name(self, node) -> str:
        """获取节点名称"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return str(node)

    def _get_annotation_name(self, node) -> str:
        """获取注解名称"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_annotation_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            return f"{self._get_annotation_name(node.value)}[{self._get_annotation_name(node.slice)}]"
        return str(node)


class SemanticAnalyzer:
    """语义分析器"""

    def __init__(self):
        self.semantic_contexts: Dict[str, SemanticContext] = {}

    def analyze_entity(self, entity: CodeEntity) -> SemanticContext:
        """分析实体的语义上下文"""
        purpose = self._extract_purpose(entity)
        inputs, outputs = self._extract_io(entity)
        side_effects = self._extract_side_effects(entity)
        preconditions, postconditions = self._extract_conditions(entity)
        related_concepts = self._extract_concepts(entity)

        context = SemanticContext(
            entity_id=entity.entity_id,
            purpose=purpose,
            inputs=inputs,
            outputs=outputs,
            side_effects=side_effects,
            preconditions=preconditions,
            postconditions=postconditions,
            related_concepts=related_concepts
        )

        self.semantic_contexts[entity.entity_id] = context
        return context

    def _extract_purpose(self, entity: CodeEntity) -> str:
        """从文档字符串提取目的"""
        if entity.docstring:
            # 提取第一行作为目的
            first_line = entity.docstring.split('\n')[0].strip()
            return first_line

        # 从名称推断
        name_parts = self._split_camel_or_snake(entity.name)
        return f"{' '.join(name_parts)}"

    def _extract_io(self, entity: CodeEntity) -> Tuple[List[str], List[str]]:
        """提取输入输出"""
        inputs = []
        outputs = []

        if entity.signature:
            # 从签名提取参数
            param_match = re.search(r'\((.*?)\)', entity.signature)
            if param_match:
                params = param_match.group(1).split(',')
                inputs = [p.strip().split(':')[0] for p in params if p.strip()]

            # 从签名提取返回类型
            if '->' in entity.signature:
                return_match = re.search(r'->\s*(\w+)', entity.signature)
                if return_match:
                    outputs = [return_match.group(1)]

        return inputs, outputs

    def _extract_side_effects(self, entity: CodeEntity) -> List[str]:
        """提取副作用"""
        side_effects = []

        # 从文档字符串查找副作用描述
        if entity.docstring:
            side_effect_patterns = [
                r'[Ss]ide [Ee]ffects?:\s*(.+?)(?=\n\n|\n[A-Z]|$)',
                r'[Mm]odifies:\s*(.+?)(?=\n\n|\n[A-Z]|$)',
                r'[Nn]ote:\s*(.+?)(?=\n\n|\n[A-Z]|$)'
            ]
            for pattern in side_effect_patterns:
                match = re.search(pattern, entity.docstring, re.DOTALL)
                if match:
                    side_effects.append(match.group(1).strip())

        return side_effects

    def _extract_conditions(self, entity: CodeEntity) -> Tuple[List[str], List[str]]:
        """提取前置和后置条件"""
        preconditions = []
        postconditions = []

        if entity.docstring:
            # 查找前置条件
            pre_patterns = [
                r'[Rr]equires?:\s*(.+?)(?=\n\n|\n[A-Z]|$)',
                r'[Pp]recondition:\s*(.+?)(?=\n\n|\n[A-Z]|$)',
                r'[Aa]rgs?:\s*(.+?)(?=\n\n|\n[A-Z]|$)'
            ]
            for pattern in pre_patterns:
                match = re.search(pattern, entity.docstring, re.DOTALL)
                if match:
                    preconditions.append(match.group(1).strip())

            # 查找后置条件
            post_patterns = [
                r'[Rr]eturns?:\s*(.+?)(?=\n\n|\n[A-Z]|$)',
                r'[Pp]ostcondition:\s*(.+?)(?=\n\n|\n[A-Z]|$)',
                r'[Ee]nsures?:\s*(.+?)(?=\n\n|\n[A-Z]|$)'
            ]
            for pattern in post_patterns:
                match = re.search(pattern, entity.docstring, re.DOTALL)
                if match:
                    postconditions.append(match.group(1).strip())

        return preconditions, postconditions

    def _extract_concepts(self, entity: CodeEntity) -> List[str]:
        """提取相关概念"""
        concepts = set()

        # 从名称提取
        name_parts = self._split_camel_or_snake(entity.name)
        concepts.update(name_parts)

        # 从文档字符串提取关键词
        if entity.docstring:
            # 提取大写缩写
            acronyms = re.findall(r'\b[A-Z]{2,}\b', entity.docstring)
            concepts.update(acronyms)

            # 提取代码相关的词
            code_words = re.findall(r'`(\w+)`', entity.docstring)
            concepts.update(code_words)

        return list(concepts)

    def _split_camel_or_snake(self, name: str) -> List[str]:
        """分割驼峰或蛇形命名"""
        # 先处理蛇形
        if '_' in name:
            return name.split('_')

        # 处理驼峰
        parts = re.findall(r'[A-Z][^A-Z]*', name)
        return [p.lower() for p in parts] if parts else [name.lower()]


class DependencyGraph:
    """依赖关系图"""

    def __init__(self):
        self.nodes: Dict[str, CodeEntity] = {}
        self.edges: List[DependencyEdge] = []
        self._adjacency: Dict[str, List[str]] = defaultdict(list)
        self._reverse_adjacency: Dict[str, List[str]] = defaultdict(list)

    def add_entity(self, entity: CodeEntity):
        """添加实体节点"""
        self.nodes[entity.entity_id] = entity

    def add_dependency(self, edge: DependencyEdge):
        """添加依赖边"""
        self.edges.append(edge)
        self._adjacency[edge.source_id].append(edge.target_id)
        self._reverse_adjacency[edge.target_id].append(edge.source_id)

        # 更新实体的依赖列表
        if edge.source_id in self.nodes:
            self.nodes[edge.source_id].dependencies.append(edge.target_id)
        if edge.target_id in self.nodes:
            self.nodes[edge.target_id].dependents.append(edge.source_id)

    def get_dependencies(self, entity_id: str, depth: int = 1) -> List[str]:
        """获取实体的依赖（指定深度）"""
        if depth == 0:
            return []

        direct = self._adjacency.get(entity_id, [])
        if depth == 1:
            return direct

        all_deps = set(direct)
        for dep in direct:
            all_deps.update(self.get_dependencies(dep, depth - 1))

        return list(all_deps)

    def get_dependents(self, entity_id: str, depth: int = 1) -> List[str]:
        """获取依赖于该实体的实体（指定深度）"""
        if depth == 0:
            return []

        direct = self._reverse_adjacency.get(entity_id, [])
        if depth == 1:
            return direct

        all_deps = set(direct)
        for dep in direct:
            all_deps.update(self.get_dependents(dep, depth - 1))

        return list(all_deps)

    def find_cycles(self) -> List[List[str]]:
        """查找循环依赖"""
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node_id: str, path: List[str]):
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)

            for neighbor in self._adjacency.get(node_id, []):
                if neighbor not in visited:
                    dfs(neighbor, path)
                elif neighbor in rec_stack:
                    # 发现循环
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:] + [neighbor])

            path.pop()
            rec_stack.remove(node_id)

        for node_id in self.nodes:
            if node_id not in visited:
                dfs(node_id, [])

        return cycles

    def compute_centrality(self) -> Dict[str, float]:
        """计算节点中心性（基于依赖数量）"""
        centrality = {}
        for entity_id in self.nodes:
            in_degree = len(self._reverse_adjacency.get(entity_id, []))
            out_degree = len(self._adjacency.get(entity_id, []))
            centrality[entity_id] = in_degree + out_degree
        return centrality


class ImpactAnalyzer:
    """影响分析器"""

    def __init__(self, dependency_graph: DependencyGraph):
        self.graph = dependency_graph

    def analyze_impact(self, changed_entity_id: str) -> ImpactAnalysisResult:
        """分析变更影响"""
        # 直接受影响
        direct_dependents = self.graph.get_dependents(changed_entity_id, depth=1)

        # 间接受影响
        indirect_dependents = self.graph.get_dependents(changed_entity_id, depth=3)
        indirect_dependents = [d for d in indirect_dependents if d not in direct_dependents]

        # 需要检查的测试
        test_entities = self._identify_test_entities(
            direct_dependents + indirect_dependents
        )

        # 计算风险分数
        risk_score = self._calculate_risk(
            changed_entity_id,
            direct_dependents,
            indirect_dependents
        )

        # 估计工作量
        estimated_effort = self._estimate_effort(
            len(direct_dependents),
            len(indirect_dependents),
            len(test_entities)
        )

        return ImpactAnalysisResult(
            changed_entity_id=changed_entity_id,
            directly_affected=direct_dependents,
            indirectly_affected=indirect_dependents,
            test_entities_to_check=test_entities,
            risk_score=risk_score,
            estimated_effort=estimated_effort
        )

    def _identify_test_entities(self, affected_entities: List[str]) -> List[str]:
        """识别需要检查的测试实体"""
        test_entities = []

        for entity_id in affected_entities:
            entity = self.graph.nodes.get(entity_id)
            if entity:
                # 检查是否是测试
                if 'test' in entity.name.lower() or entity.file_path.endswith('_test.py'):
                    test_entities.append(entity_id)
                else:
                    # 查找相关的测试
                    related_tests = self._find_related_tests(entity)
                    test_entities.extend(related_tests)

        return list(set(test_entities))

    def _find_related_tests(self, entity: CodeEntity) -> List[str]:
        """查找与实体相关的测试"""
        related = []

        for test_id, test_entity in self.graph.nodes.items():
            if 'test' in test_entity.name.lower():
                # 检查测试是否引用了该实体
                if entity.name in test_entity.metadata.get('test_targets', []):
                    related.append(test_id)

        return related

    def _calculate_risk(
        self,
        changed_id: str,
        direct: List[str],
        indirect: List[str]
    ) -> float:
        """计算变更风险"""
        # 基于受影响实体数量和中心性计算风险
        base_risk = min(1.0, (len(direct) * 0.1 + len(indirect) * 0.05))

        # 考虑被修改实体的重要性
        centrality = self.graph.compute_centrality()
        max_centrality = max(centrality.values()) if centrality else 1
        entity_centrality = centrality.get(changed_id, 0)

        importance_factor = entity_centrality / max_centrality if max_centrality > 0 else 0

        return min(1.0, base_risk + importance_factor * 0.3)

    def _estimate_effort(
        self,
        direct_count: int,
        indirect_count: int,
        test_count: int
    ) -> int:
        """估计工作量（分钟）"""
        # 简单估算：每个直接受影响实体5分钟，间接2分钟，测试10分钟
        return direct_count * 5 + indirect_count * 2 + test_count * 10


class SemanticCodeSearch:
    """语义代码搜索"""

    def __init__(
        self,
        entities: Dict[str, CodeEntity],
        semantic_contexts: Dict[str, SemanticContext]
    ):
        self.entities = entities
        self.semantic_contexts = semantic_contexts

    def search(self, query: str, top_k: int = 10) -> List[CodeSearchResult]:
        """语义搜索"""
        query_concepts = self._extract_query_concepts(query)
        results = []

        for entity_id, entity in self.entities.items():
            score = self._compute_relevance(entity_id, query_concepts, query)
            if score > 0:
                context = self.semantic_contexts.get(entity_id)
                results.append(CodeSearchResult(
                    entity_id=entity_id,
                    relevance_score=score,
                    match_type="semantic",
                    matched_concepts=query_concepts,
                    snippet=self._generate_snippet(entity, context)
                ))

        # 排序并返回前k个
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:top_k]

    def _extract_query_concepts(self, query: str) -> List[str]:
        """提取查询中的概念"""
        # 简单的关键词提取
        words = re.findall(r'\b\w+\b', query.lower())
        # 过滤停用词
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                      'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                      'through', 'during', 'before', 'after', 'above', 'below',
                      'between', 'under', 'and', 'but', 'or', 'yet', 'so'}
        return [w for w in words if w not in stop_words]

    def _compute_relevance(
        self,
        entity_id: str,
        query_concepts: List[str],
        original_query: str
    ) -> float:
        """计算相关性分数"""
        entity = self.entities.get(entity_id)
        context = self.semantic_contexts.get(entity_id)

        if not entity:
            return 0.0

        score = 0.0

        # 名称匹配
        name_parts = self._split_name(entity.name)
        name_matches = sum(1 for c in query_concepts if any(c in p for p in name_parts))
        score += name_matches * 0.3

        # 文档字符串匹配
        if entity.docstring:
            doc_lower = entity.docstring.lower()
            doc_matches = sum(1 for c in query_concepts if c in doc_lower)
            score += doc_matches * 0.2

        # 语义上下文匹配
        if context:
            concept_matches = sum(
                1 for c in query_concepts if any(c in rc.lower() for rc in context.related_concepts)
            )
            score += concept_matches * 0.25

            purpose_match = sum(1 for c in query_concepts if c in context.purpose.lower())
            score += purpose_match * 0.15

        # 完全匹配加分
        if original_query.lower() in entity.name.lower():
            score += 0.5

        return min(1.0, score)

    def _split_name(self, name: str) -> List[str]:
        """分割名称"""
        # 驼峰命名
        parts = re.findall(r'[A-Z][^A-Z]*', name)
        if parts:
            return [p.lower() for p in parts]
        # 蛇形命名
        return name.lower().split('_')

    def _generate_snippet(
        self,
        entity: CodeEntity,
        context: Optional[SemanticContext]
    ) -> str:
        """生成代码片段"""
        lines = []

        if entity.signature:
            lines.append(entity.signature)

        if context:
            lines.append(f"# {context.purpose}")

        if entity.docstring:
            first_line = entity.docstring.split('\n')[0][:100]
            lines.append(f'"""{first_line}..."""')

        return '\n'.join(lines)


class SmartCodeAnalyzer:
    """智能代码分析器主类"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(
            Path.home() / ".pyutagent" / "code_analysis.db"
        )
        self.ast_analyzer = ASTAnalyzer()
        self.semantic_analyzer = SemanticAnalyzer()
        self.dependency_graph = DependencyGraph()
        self.impact_analyzer: Optional[ImpactAnalyzer] = None
        self.code_search: Optional[SemanticCodeSearch] = None

        self._init_db()

    def _init_db(self):
        """初始化数据库"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok