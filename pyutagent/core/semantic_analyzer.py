"""代码语义分析模块 - 深层代码理解和测试场景识别

本模块提供代码语义分析功能，包括:
- 方法调用图构建
- 业务逻辑链路识别
- 数据流分析
- 测试场景智能推导
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from pathlib import Path
from enum import Enum, auto
from collections import defaultdict

from .cache import get_global_cache

logger = logging.getLogger(__name__)


class LogicType(Enum):
    """业务逻辑类型"""
    VALIDATION = auto()      # 参数验证
    TRANSFORMATION = auto()  # 数据转换
    PERSISTENCE = auto()     # 数据持久化
    RETRIEVAL = auto()       # 数据检索
    CALCULATION = auto()     # 计算逻辑
    STATE_CHANGE = auto()    # 状态变更
    EXTERNAL_CALL = auto()   # 外部调用
    EXCEPTION_HANDLING = auto()  # 异常处理


class BoundaryType(Enum):
    """边界条件类型"""
    NULL_CHECK = auto()          # 空值检查
    EMPTY_CHECK = auto()         # 空集合检查
    RANGE_CHECK = auto()         # 范围检查
    TYPE_CHECK = auto()          # 类型检查
    STATE_CHECK = auto()         # 状态检查
    PERMISSION_CHECK = auto()    # 权限检查


@dataclass
class MethodCall:
    """方法调用信息"""
    caller: str
    callee: str
    line_number: int
    arguments: List[str] = field(default_factory=list)
    return_value_used: bool = False


@dataclass
class DataFlow:
    """数据流信息"""
    source: str
    sink: str
    path: List[str] = field(default_factory=list)
    transformations: List[str] = field(default_factory=list)


@dataclass
class BusinessLogic:
    """业务逻辑片段"""
    method_name: str
    logic_type: LogicType
    description: str
    start_line: int
    end_line: int
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    side_effects: List[str] = field(default_factory=list)


@dataclass
class BoundaryCondition:
    """边界条件"""
    parameter_name: str
    boundary_type: BoundaryType
    description: str
    test_value: Any
    expected_behavior: str


@dataclass
class TestScenario:
    """测试场景"""
    scenario_id: str
    description: str
    target_method: str
    test_type: str  # "normal", "edge", "exception"
    setup_steps: List[str] = field(default_factory=list)
    test_steps: List[str] = field(default_factory=list)
    expected_result: str = ""
    priority: int = 1  # 1-5, 1 最高
    related_methods: List[str] = field(default_factory=list)
    boundary_conditions: List[BoundaryCondition] = field(default_factory=list)


@dataclass
class CallGraphNode:
    """调用图节点"""
    method_name: str
    class_name: str
    file_path: str
    callers: Set[str] = field(default_factory=set)
    callees: Set[str] = field(default_factory=set)
    complexity: int = 1
    is_entry_point: bool = False
    is_test_target: bool = True


@dataclass
class CallGraph:
    """方法调用图"""
    nodes: Dict[str, CallGraphNode] = field(default_factory=dict)
    edges: List[MethodCall] = field(default_factory=list)
    entry_points: Set[str] = field(default_factory=set)
    
    def add_node(self, node: CallGraphNode):
        """添加节点"""
        key = f"{node.class_name}.{node.method_name}"
        self.nodes[key] = node
    
    def add_edge(self, call: MethodCall):
        """添加边"""
        self.edges.append(call)
        caller_key = call.caller
        callee_key = call.callee
        
        if caller_key in self.nodes:
            self.nodes[caller_key].callees.add(callee_key)
        if callee_key in self.nodes:
            self.nodes[callee_key].callers.add(caller_key)


class SemanticAnalyzer:
    """代码语义分析器
    
    功能:
    - 构建方法调用图
    - 识别业务逻辑
    - 分析数据流
    - 推导测试场景
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.call_graph = CallGraph()
        self.business_logic: Dict[str, List[BusinessLogic]] = defaultdict(list)
        self.data_flows: Dict[str, List[DataFlow]] = defaultdict(list)
        self.test_scenarios: List[TestScenario] = []
        self.cache = get_global_cache()
        
    def analyze_file(self, file_path: str, java_class: Any) -> Dict[str, Any]:
        """分析 Java 文件的语义
        
        Args:
            file_path: Java 文件路径
            java_class: 解析后的 JavaClass 对象
            
        Returns:
            分析结果
        """
        self.logger.info(f"[SemanticAnalyzer] Analyzing file: {file_path}")
        
        # 生成缓存键
        cache_key = self.cache.generate_key("semantic_analysis", file_path, str(java_class))
        
        # 尝试从缓存获取
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            self.logger.info(f"[SemanticAnalyzer] Cache hit for file: {file_path}")
            return cached_result
        
        # 1. 构建调用图
        self._build_call_graph_for_class(file_path, java_class)
        
        # 2. 识别业务逻辑
        self._identify_business_logic(java_class)
        
        # 3. 分析数据流
        self._analyze_data_flow(java_class)
        
        # 4. 识别边界条件
        boundary_conditions = self._identify_boundary_conditions(java_class)
        
        # 5. 生成测试场景
        test_scenarios = self._generate_test_scenarios(java_class, boundary_conditions)
        
        result = {
            "file": file_path,
            "class_name": java_class.name,
            "call_graph": self._call_graph_to_dict(),
            "business_logic": self._business_logic_to_dict(java_class.name),
            "data_flows": self._data_flows_to_dict(java_class.name),
            "boundary_conditions": [
                {
                    "parameter": bc.parameter_name,
                    "type": bc.boundary_type.name,
                    "description": bc.description,
                    "test_value": bc.test_value,
                    "expected_behavior": bc.expected_behavior
                }
                for bc in boundary_conditions
            ],
            "test_scenarios": [
                {
                    "id": ts.scenario_id,
                    "description": ts.description,
                    "target": ts.target_method,
                    "type": ts.test_type,
                    "priority": ts.priority,
                    "setup": ts.setup_steps,
                    "steps": ts.test_steps,
                    "expected": ts.expected_result
                }
                for ts in test_scenarios
            ]
        }
        
        # 缓存结果
        self.cache.set(cache_key, result, ttl_l1=1800, ttl_l2=43200)
        
        self.logger.info(f"[SemanticAnalyzer] Analysis complete - "
                        f"Scenarios: {len(test_scenarios)}, "
                        f"Boundaries: {len(boundary_conditions)}")
        
        return result
    
    def _build_call_graph_for_class(self, file_path: str, java_class: Any):
        """为类构建调用图"""
        class_name = java_class.name
        
        # 添加类的方法到调用图
        for method in java_class.methods:
            node = CallGraphNode(
                method_name=method.name,
                class_name=class_name,
                file_path=file_path,
                complexity=self._calculate_method_complexity(method),
                is_entry_point=self._is_entry_point(method)
            )
            self.call_graph.add_node(node)
            
            if node.is_entry_point:
                self.call_graph.entry_points.add(f"{class_name}.{method.name}")
        
        # 分析方法体中的方法调用 (需要源代码)
        # 这里简化处理，实际实现需要解析方法体
    
    def _calculate_method_complexity(self, method: Any) -> int:
        """计算方法复杂度 (圈复杂度)"""
        # 简化版本，基于参数数量和返回值
        complexity = 1
        complexity += len(method.parameters)  # 每个参数增加复杂度
        # 实际实现需要分析方法体中的分支语句
        return complexity
    
    def _is_entry_point(self, method: Any) -> bool:
        """判断是否为入口方法"""
        # Public 方法通常是入口点
        return 'public' in method.modifiers and not method.name.startswith('_')
    
    def _identify_business_logic(self, java_class: Any):
        """识别业务逻辑"""
        class_name = java_class.name
        
        for method in java_class.methods:
            # 基于方法名和注解识别业务逻辑类型
            logic_types = self._infer_logic_type(method)
            
            for logic_type in logic_types:
                business_logic = BusinessLogic(
                    method_name=method.name,
                    logic_type=logic_type,
                    description=self._describe_logic(method, logic_type),
                    start_line=method.start_line,
                    end_line=method.end_line,
                    preconditions=self._extract_preconditions(method),
                    postconditions=self._extract_postconditions(method),
                    side_effects=self._identify_side_effects(method)
                )
                self.business_logic[class_name].append(business_logic)
    
    def _infer_logic_type(self, method: Any) -> List[LogicType]:
        """推断业务逻辑类型"""
        logic_types = []
        method_name = method.name.lower()
        
        # 基于方法名推断
        if any(x in method_name for x in ['valid', 'check', 'verify', 'ensure']):
            logic_types.append(LogicType.VALIDATION)
        
        if any(x in method_name for x in ['convert', 'transform', 'parse', 'format']):
            logic_types.append(LogicType.TRANSFORMATION)
        
        if any(x in method_name for x in ['save', 'store', 'insert', 'update', 'delete']):
            logic_types.append(LogicType.PERSISTENCE)
        
        if any(x in method_name for x in ['get', 'find', 'search', 'query', 'load']):
            logic_types.append(LogicType.RETRIEVAL)
        
        if any(x in method_name for x in ['calculate', 'compute', 'count', 'sum']):
            logic_types.append(LogicType.CALCULATION)
        
        if any(x in method_name for x in ['set', 'change', 'modify', 'enable', 'disable']):
            logic_types.append(LogicType.STATE_CHANGE)
        
        if any(x in method_name for x in ['call', 'invoke', 'send', 'fetch']):
            logic_types.append(LogicType.EXTERNAL_CALL)
        
        if not logic_types:
            logic_types.append(LogicType.TRANSFORMATION)  # 默认
        
        return logic_types
    
    def _describe_logic(self, method: Any, logic_type: LogicType) -> str:
        """描述业务逻辑"""
        descriptions = {
            LogicType.VALIDATION: f"Validates {method.name.replace('_', ' ')}",
            LogicType.TRANSFORMATION: f"Transforms data in {method.name.replace('_', ' ')}",
            LogicType.PERSISTENCE: f"Persists data in {method.name.replace('_', ' ')}",
            LogicType.RETRIEVAL: f"Retrieves data in {method.name.replace('_', ' ')}",
            LogicType.CALCULATION: f"Calculates result in {method.name.replace('_', ' ')}",
            LogicType.STATE_CHANGE: f"Changes state in {method.name.replace('_', ' ')}",
            LogicType.EXTERNAL_CALL: f"Calls external service in {method.name.replace('_', ' ')}",
            LogicType.EXCEPTION_HANDLING: f"Handles exceptions in {method.name.replace('_', ' ')}"
        }
        return descriptions.get(logic_type, f"Executes {method.name.replace('_', ' ')}")
    
    def _extract_preconditions(self, method: Any) -> List[str]:
        """提取前置条件"""
        preconditions = []
        
        # 检查参数验证
        for param_type, param_name in method.parameters:
            if 'NotNull' in method.annotations or 'NonNull' in method.annotations:
                preconditions.append(f"{param_name} must not be null")
            if 'NotEmpty' in method.annotations:
                preconditions.append(f"{param_name} must not be empty")
        
        return preconditions
    
    def _extract_postconditions(self, method: Any) -> List[str]:
        """提取后置条件"""
        postconditions = []
        
        if method.return_type:
            if 'NotNull' in method.annotations or 'NonNull' in method.annotations:
                postconditions.append("Returns non-null value")
            if 'NotEmpty' in method.annotations:
                postconditions.append("Returns non-empty result")
        
        return postconditions
    
    def _identify_side_effects(self, method: Any) -> List[str]:
        """识别副作用"""
        side_effects = []
        
        # 基于方法名和类型推断
        if any(x in method.name.lower() for x in ['save', 'update', 'delete', 'set']):
            side_effects.append("Modifies persistent state")
        
        if any(x in method.name.lower() for x in ['send', 'publish', 'notify']):
            side_effects.append("Triggers external communication")
        
        return side_effects
    
    def _analyze_data_flow(self, java_class: Any):
        """分析数据流"""
        # 简化版本，实际实现需要分析方法体
        class_name = java_class.name
        
        # 识别字段读写
        for field_type, field_name, field_modifiers in java_class.fields:
            # 检查是否有 getter/setter
            getter_name = f"get{field_name.capitalize()}"
            setter_name = f"set{field_name.capitalize()}"
            
            has_getter = any(m.name == getter_name for m in java_class.methods)
            has_setter = any(m.name == setter_name for m in java_class.methods)
            
            if has_getter and has_setter:
                data_flow = DataFlow(
                    source=getter_name,
                    sink=setter_name,
                    path=[field_name],
                    transformations=[]
                )
                self.data_flows[class_name].append(data_flow)
    
    def _identify_boundary_conditions(self, java_class: Any) -> List[BoundaryCondition]:
        """识别边界条件"""
        boundary_conditions = []
        
        for method in java_class.methods:
            for param_type, param_name in method.parameters:
                # 识别常见边界条件
                conditions = self._extract_parameter_boundaries(param_type, param_name)
                boundary_conditions.extend(conditions)
        
        return boundary_conditions
    
    def _extract_parameter_boundaries(self, param_type: str, param_name: str) -> List[BoundaryCondition]:
        """提取参数边界条件"""
        conditions = []
        
        # 字符串类型
        if param_type in ['String', 'CharSequence']:
            conditions.extend([
                BoundaryCondition(
                    parameter_name=param_name,
                    boundary_type=BoundaryType.NULL_CHECK,
                    description=f"Test {param_name} with null value",
                    test_value=None,
                    expected_behavior="Should throw IllegalArgumentException or handle gracefully"
                ),
                BoundaryCondition(
                    parameter_name=param_name,
                    boundary_type=BoundaryType.EMPTY_CHECK,
                    description=f"Test {param_name} with empty string",
                    test_value="",
                    expected_behavior="Should handle empty string appropriately"
                )
            ])
        
        # 数值类型
        elif param_type in ['int', 'Integer', 'long', 'Long', 'double', 'Double', 'float', 'Float']:
            conditions.extend([
                BoundaryCondition(
                    parameter_name=param_name,
                    boundary_type=BoundaryType.RANGE_CHECK,
                    description=f"Test {param_name} with zero value",
                    test_value=0,
                    expected_behavior="Should handle zero correctly"
                ),
                BoundaryCondition(
                    parameter_name=param_name,
                    boundary_type=BoundaryType.RANGE_CHECK,
                    description=f"Test {param_name} with negative value",
                    test_value=-1,
                    expected_behavior="Should validate negative values if not allowed"
                ),
                BoundaryCondition(
                    parameter_name=param_name,
                    boundary_type=BoundaryType.RANGE_CHECK,
                    description=f"Test {param_name} with maximum value",
                    test_value=self._get_max_value(param_type),
                    expected_behavior="Should handle maximum value without overflow"
                )
            ])
        
        # 集合类型
        elif any(x in param_type for x in ['List', 'Set', 'Map', 'Collection', 'Array']):
            conditions.extend([
                BoundaryCondition(
                    parameter_name=param_name,
                    boundary_type=BoundaryType.NULL_CHECK,
                    description=f"Test {param_name} with null",
                    test_value=None,
                    expected_behavior="Should handle null collection"
                ),
                BoundaryCondition(
                    parameter_name=param_name,
                    boundary_type=BoundaryType.EMPTY_CHECK,
                    description=f"Test {param_name} with empty collection",
                    test_value=[],
                    expected_behavior="Should handle empty collection gracefully"
                )
            ])
        
        return conditions
    
    def _get_max_value(self, param_type: str) -> Any:
        """获取类型的最大值"""
        max_values = {
            'int': 2147483647,
            'Integer': 2147483647,
            'long': 9223372036854775807,
            'Long': 9223372036854775807,
            'float': 3.4028235e38,
            'Float': 3.4028235e38,
            'double': 1.7976931348623157e308,
            'Double': 1.7976931348623157e308
        }
        return max_values.get(param_type, float('inf'))
    
    def _generate_test_scenarios(
        self,
        java_class: Any,
        boundary_conditions: List[BoundaryCondition]
    ) -> List[TestScenario]:
        """生成测试场景"""
        scenarios = []
        class_name = java_class.name
        scenario_counter = 1
        
        for method in java_class.methods:
            # 1. 正常场景测试
            normal_scenario = TestScenario(
                scenario_id=f"{class_name}_{scenario_counter:03d}",
                description=f"Test {method.name} with normal input",
                target_method=method.name,
                test_type="normal",
                setup_steps=[f"Create instance of {class_name}"],
                test_steps=[
                    f"Prepare valid input parameters",
                    f"Call {method.name}() with prepared inputs",
                    "Verify the result matches expected output"
                ],
                expected_result="Method executes successfully and returns correct result",
                priority=1,
                related_methods=[m.name for m in java_class.methods if m.name != method.name]
            )
            scenarios.append(normal_scenario)
            scenario_counter += 1
            
            # 2. 边界条件测试
            method_boundaries = [
                bc for bc in boundary_conditions
                if any(p[1] == bc.parameter_name for p in method.parameters)
            ]
            
            for boundary in method_boundaries:
                boundary_scenario = TestScenario(
                    scenario_id=f"{class_name}_{scenario_counter:03d}",
                    description=boundary.description,
                    target_method=method.name,
                    test_type="edge",
                    setup_steps=[f"Create instance of {class_name}"],
                    test_steps=[
                        f"Prepare {boundary.parameter_name} with {boundary.test_value}",
                        f"Call {method.name}()",
                        "Verify behavior matches expectation"
                    ],
                    expected_result=boundary.expected_behavior,
                    priority=2,
                    boundary_conditions=[boundary]
                )
                scenarios.append(boundary_scenario)
                scenario_counter += 1
            
            # 3. 异常场景测试
            if 'throws' in str(method.annotations).lower() or method.return_type is None:
                exception_scenario = TestScenario(
                    scenario_id=f"{class_name}_{scenario_counter:03d}",
                    description=f"Test {method.name} with invalid input to trigger exception",
                    target_method=method.name,
                    test_type="exception",
                    setup_steps=[f"Create instance of {class_name}"],
                    test_steps=[
                        "Prepare invalid input that should trigger exception",
                        f"Call {method.name}()",
                        "Verify exception is thrown with correct type and message"
                    ],
                    expected_result="Throws appropriate exception",
                    priority=2
                )
                scenarios.append(exception_scenario)
                scenario_counter += 1
        
        return scenarios
    
    def _call_graph_to_dict(self) -> Dict[str, Any]:
        """将调用图转换为字典"""
        return {
            "nodes": [
                {
                    "method": node.method_name,
                    "class": node.class_name,
                    "file": node.file_path,
                    "callers": list(node.callers),
                    "callees": list(node.callees),
                    "complexity": node.complexity,
                    "is_entry_point": node.is_entry_point
                }
                for node in self.call_graph.nodes.values()
            ],
            "edges": [
                {
                    "caller": edge.caller,
                    "callee": edge.callee,
                    "line": edge.line_number
                }
                for edge in self.call_graph.edges
            ],
            "entry_points": list(self.call_graph.entry_points)
        }
    
    def _business_logic_to_dict(self, class_name: str) -> List[Dict[str, Any]]:
        """将业务逻辑转换为字典"""
        logic_list = self.business_logic.get(class_name, [])
        return [
            {
                "method": bl.method_name,
                "type": bl.logic_type.name,
                "description": bl.description,
                "preconditions": bl.preconditions,
                "postconditions": bl.postconditions,
                "side_effects": bl.side_effects
            }
            for bl in logic_list
        ]
    
    def _data_flows_to_dict(self, class_name: str) -> List[Dict[str, Any]]:
        """将数据流转换为字典"""
        flows = self.data_flows.get(class_name, [])
        return [
            {
                "source": df.source,
                "sink": df.sink,
                "path": df.path,
                "transformations": df.transformations
            }
            for df in flows
        ]
    
    def get_test_scenarios(self) -> List[TestScenario]:
        """获取所有测试场景"""
        return self.test_scenarios
    
    def get_call_graph(self) -> CallGraph:
        """获取调用图"""
        return self.call_graph
    
    def clear(self):
        """清除分析结果"""
        self.call_graph = CallGraph()
        self.business_logic.clear()
        self.data_flows.clear()
        self.test_scenarios.clear()
        self.logger.info("[SemanticAnalyzer] Cleared analysis results")
