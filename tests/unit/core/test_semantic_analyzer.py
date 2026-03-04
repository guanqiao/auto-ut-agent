"""单元测试 - SemanticAnalyzer"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock

from pyutagent.core.semantic_analyzer import (
    SemanticAnalyzer,
    LogicType,
    BoundaryType,
    MethodCall,
    DataFlow,
    BusinessLogic,
    BoundaryCondition,
    TestScenario,
    CallGraphNode,
    CallGraph
)


class TestLogicType:
    """测试 LogicType 枚举"""
    
    def test_logic_type_values(self):
        """测试业务逻辑类型值"""
        assert LogicType.VALIDATION.name == "VALIDATION"
        assert LogicType.TRANSFORMATION.name == "TRANSFORMATION"
        assert LogicType.PERSISTENCE.name == "PERSISTENCE"
        assert LogicType.RETRIEVAL.name == "RETRIEVAL"
        assert LogicType.CALCULATION.name == "CALCULATION"
        assert LogicType.STATE_CHANGE.name == "STATE_CHANGE"
        assert LogicType.EXTERNAL_CALL.name == "EXTERNAL_CALL"
        assert LogicType.EXCEPTION_HANDLING.name == "EXCEPTION_HANDLING"


class TestBoundaryType:
    """测试 BoundaryType 枚举"""
    
    def test_boundary_type_values(self):
        """测试边界条件类型值"""
        assert BoundaryType.NULL_CHECK.name == "NULL_CHECK"
        assert BoundaryType.EMPTY_CHECK.name == "EMPTY_CHECK"
        assert BoundaryType.RANGE_CHECK.name == "RANGE_CHECK"
        assert BoundaryType.TYPE_CHECK.name == "TYPE_CHECK"
        assert BoundaryType.STATE_CHECK.name == "STATE_CHECK"
        assert BoundaryType.PERMISSION_CHECK.name == "PERMISSION_CHECK"


class TestMethodCall:
    """测试 MethodCall 数据类"""
    
    def test_method_call_creation(self):
        """测试方法调用创建"""
        call = MethodCall(
            caller="ClassA.method1",
            callee="ClassB.method2",
            line_number=10,
            arguments=["arg1", "arg2"],
            return_value_used=True
        )
        
        assert call.caller == "ClassA.method1"
        assert call.callee == "ClassB.method2"
        assert call.line_number == 10
        assert len(call.arguments) == 2
        assert call.return_value_used is True


class TestDataFlow:
    """测试 DataFlow 数据类"""
    
    def test_data_flow_creation(self):
        """测试数据流创建"""
        flow = DataFlow(
            source="getter",
            sink="setter",
            path=["field1"],
            transformations=["transform1"]
        )
        
        assert flow.source == "getter"
        assert flow.sink == "setter"
        assert len(flow.path) == 1
        assert len(flow.transformations) == 1


class TestBusinessLogic:
    """测试 BusinessLogic 数据类"""
    
    def test_business_logic_creation(self):
        """测试业务逻辑创建"""
        logic = BusinessLogic(
            method_name="calculateTotal",
            logic_type=LogicType.CALCULATION,
            description="Calculates total amount",
            start_line=10,
            end_line=20,
            preconditions=["amount > 0"],
            postconditions=["returns positive value"],
            side_effects=["updates cache"]
        )
        
        assert logic.method_name == "calculateTotal"
        assert logic.logic_type == LogicType.CALCULATION
        assert len(logic.preconditions) == 1
        assert len(logic.postconditions) == 1
        assert len(logic.side_effects) == 1


class TestBoundaryCondition:
    """测试 BoundaryCondition 数据类"""
    
    def test_boundary_condition_creation(self):
        """测试边界条件创建"""
        boundary = BoundaryCondition(
            parameter_name="username",
            boundary_type=BoundaryType.NULL_CHECK,
            description="Test username with null value",
            test_value=None,
            expected_behavior="Should throw IllegalArgumentException"
        )
        
        assert boundary.parameter_name == "username"
        assert boundary.boundary_type == BoundaryType.NULL_CHECK
        assert boundary.test_value is None


class TestTestScenario:
    """测试 TestScenario 数据类"""
    
    def test_test_scenario_creation(self):
        """测试测试场景创建"""
        scenario = TestScenario(
            scenario_id="TestClass_001",
            description="Test method with normal input",
            target_method="TestMethod",
            test_type="normal",
            setup_steps=["Create instance"],
            test_steps=["Call method", "Verify result"],
            expected_result="Success",
            priority=1
        )
        
        assert scenario.scenario_id == "TestClass_001"
        assert scenario.test_type == "normal"
        assert scenario.priority == 1
        assert len(scenario.setup_steps) == 1
        assert len(scenario.test_steps) == 2


class TestCallGraphNode:
    """测试 CallGraphNode 数据类"""
    
    def test_call_graph_node_creation(self):
        """测试调用图节点创建"""
        node = CallGraphNode(
            method_name="method1",
            class_name="TestClass",
            file_path="/path/to/file.java",
            complexity=5,
            is_entry_point=True
        )
        
        assert node.method_name == "method1"
        assert node.class_name == "TestClass"
        assert node.complexity == 5
        assert node.is_entry_point is True
        assert len(node.callers) == 0
        assert len(node.callees) == 0


class TestCallGraph:
    """测试 CallGraph 数据类"""
    
    def test_call_graph_creation(self):
        """测试调用图创建"""
        graph = CallGraph()
        
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
        assert len(graph.entry_points) == 0
    
    def test_add_node(self):
        """测试添加节点"""
        graph = CallGraph()
        node = CallGraphNode(
            method_name="method1",
            class_name="TestClass",
            file_path="/path/to/file.java"
        )
        
        graph.add_node(node)
        
        assert len(graph.nodes) == 1
        assert "TestClass.method1" in graph.nodes
    
    def test_add_edge(self):
        """测试添加边"""
        graph = CallGraph()
        
        # 先添加节点
        node1 = CallGraphNode(method_name="method1", class_name="ClassA", file_path="file1.java")
        node2 = CallGraphNode(method_name="method2", class_name="ClassB", file_path="file2.java")
        graph.add_node(node1)
        graph.add_node(node2)
        
        # 添加边
        call = MethodCall(
            caller="ClassA.method1",
            callee="ClassB.method2",
            line_number=10
        )
        graph.add_edge(call)
        
        assert len(graph.edges) == 1
        assert "ClassB.method2" in graph.nodes["ClassA.method1"].callees
        assert "ClassA.method1" in graph.nodes["ClassB.method2"].callers


class TestSemanticAnalyzer:
    """测试 SemanticAnalyzer 类"""
    
    def setup_method(self):
        """测试前准备"""
        self.analyzer = SemanticAnalyzer()
    
    def teardown_method(self):
        """测试后清理"""
        self.analyzer.clear()
    
    def test_initialization(self):
        """测试初始化"""
        analyzer = SemanticAnalyzer()
        
        assert analyzer.call_graph is not None
        assert len(analyzer.business_logic) == 0
        assert len(analyzer.data_flows) == 0
        assert len(analyzer.test_scenarios) == 0
    
    def test_analyze_file_basic(self):
        """测试基本文件分析"""
        # 创建模拟的 JavaClass 对象
        mock_class = Mock()
        mock_class.name = "UserService"
        mock_class.package = "com.example.service"
        mock_class.methods = []
        mock_class.fields = []
        mock_class.imports = []
        mock_class.annotations = []
        
        result = self.analyzer.analyze_file("/path/to/UserService.java", mock_class)
        
        assert result["file"] == "/path/to/UserService.java"
        assert result["class_name"] == "UserService"
        assert "call_graph" in result
        assert "business_logic" in result
        assert "test_scenarios" in result
    
    def test_analyze_file_with_methods(self):
        """测试带方法的文件分析"""
        # 创建带方法的模拟对象
        mock_method = Mock()
        mock_method.name = "getUserById"
        mock_method.return_type = "User"
        mock_method.parameters = [("Long", "userId")]
        mock_method.modifiers = ["public"]
        mock_method.annotations = []
        mock_method.start_line = 10
        mock_method.end_line = 20
        
        mock_class = Mock()
        mock_class.name = "UserService"
        mock_class.package = "com.example.service"
        mock_class.methods = [mock_method]
        mock_class.fields = [("User", "userRepository", "private")]
        mock_class.imports = []
        mock_class.annotations = []
        
        result = self.analyzer.analyze_file("/path/to/UserService.java", mock_class)
        
        assert len(result["test_scenarios"]) > 0
        # 应该有正常场景和边界条件场景
        assert any(s["type"] == "normal" for s in result["test_scenarios"])
        assert any(s["type"] == "edge" for s in result["test_scenarios"])
    
    def test_infer_logic_type_validation(self):
        """测试推断业务逻辑类型 - 验证"""
        mock_method = Mock()
        mock_method.name = "validateUser"
        mock_method.annotations = []
        
        logic_types = self.analyzer._infer_logic_type(mock_method)
        
        assert LogicType.VALIDATION in logic_types
    
    def test_infer_logic_type_transformation(self):
        """测试推断业务逻辑类型 - 转换"""
        mock_method = Mock()
        mock_method.name = "convertToDTO"
        mock_method.annotations = []
        
        logic_types = self.analyzer._infer_logic_type(mock_method)
        
        assert LogicType.TRANSFORMATION in logic_types
    
    def test_infer_logic_type_persistence(self):
        """测试推断业务逻辑类型 - 持久化"""
        mock_method = Mock()
        mock_method.name = "saveUser"
        mock_method.annotations = []
        
        logic_types = self.analyzer._infer_logic_type(mock_method)
        
        assert LogicType.PERSISTENCE in logic_types
    
    def test_infer_logic_type_retrieval(self):
        """测试推断业务逻辑类型 - 检索"""
        mock_method = Mock()
        mock_method.name = "findUser"
        mock_method.annotations = []
        
        logic_types = self.analyzer._infer_logic_type(mock_method)
        
        assert LogicType.RETRIEVAL in logic_types
    
    def test_infer_logic_type_calculation(self):
        """测试推断业务逻辑类型 - 计算"""
        mock_method = Mock()
        mock_method.name = "calculateTotal"
        mock_method.annotations = []
        
        logic_types = self.analyzer._infer_logic_type(mock_method)
        
        assert LogicType.CALCULATION in logic_types
    
    def test_infer_logic_type_default(self):
        """测试推断业务逻辑类型 - 默认"""
        mock_method = Mock()
        mock_method.name = "execute"
        mock_method.annotations = []
        
        logic_types = self.analyzer._infer_logic_type(mock_method)
        
        assert len(logic_types) > 0
    
    def test_extract_parameter_boundaries_string(self):
        """测试提取参数边界条件 - 字符串类型"""
        boundaries = self.analyzer._extract_parameter_boundaries("String", "username")
        
        assert len(boundaries) > 0
        assert any(b.boundary_type == BoundaryType.NULL_CHECK for b in boundaries)
        assert any(b.boundary_type == BoundaryType.EMPTY_CHECK for b in boundaries)
    
    def test_extract_parameter_boundaries_numeric(self):
        """测试提取参数边界条件 - 数值类型"""
        boundaries = self.analyzer._extract_parameter_boundaries("int", "count")
        
        assert len(boundaries) > 0
        assert any(b.boundary_type == BoundaryType.RANGE_CHECK for b in boundaries)
        # 应该包含 0、负数、最大值测试
        assert any(b.test_value == 0 for b in boundaries)
        assert any(b.test_value == -1 for b in boundaries)
    
    def test_extract_parameter_boundaries_collection(self):
        """测试提取参数边界条件 - 集合类型"""
        boundaries = self.analyzer._extract_parameter_boundaries("List<String>", "items")
        
        assert len(boundaries) > 0
        assert any(b.boundary_type == BoundaryType.NULL_CHECK for b in boundaries)
        assert any(b.boundary_type == BoundaryType.EMPTY_CHECK for b in boundaries)
    
    def test_get_max_value(self):
        """测试获取最大值"""
        assert self.analyzer._get_max_value("int") == 2147483647
        assert self.analyzer._get_max_value("long") == 9223372036854775807
        assert isinstance(self.analyzer._get_max_value("double"), float)
    
    def test_is_entry_point(self):
        """测试判断入口方法"""
        public_method = Mock()
        public_method.name = "getUser"
        public_method.modifiers = ["public"]
        
        private_method = Mock()
        private_method.name = "_internalMethod"
        private_method.modifiers = ["private"]
        
        assert self.analyzer._is_entry_point(public_method) is True
        assert self.analyzer._is_entry_point(private_method) is False
    
    def test_calculate_method_complexity(self):
        """测试计算方法复杂度"""
        simple_method = Mock()
        simple_method.name = "simple"
        simple_method.parameters = []
        
        complex_method = Mock()
        complex_method.name = "complex"
        complex_method.parameters = [("String", "a"), ("int", "b"), ("List", "c")]
        
        simple_complexity = self.analyzer._calculate_method_complexity(simple_method)
        complex_complexity = self.analyzer._calculate_method_complexity(complex_method)
        
        assert simple_complexity >= 1
        assert complex_complexity > simple_complexity
    
    def test_generate_test_scenarios(self):
        """测试生成测试场景"""
        mock_method = Mock()
        mock_method.name = "processData"
        mock_method.return_type = "Result"
        mock_method.parameters = [("String", "data"), ("int", "count")]
        mock_method.modifiers = ["public"]
        mock_method.annotations = []
        mock_method.start_line = 1
        mock_method.end_line = 10
        
        mock_class = Mock()
        mock_class.name = "DataProcessor"
        mock_class.methods = [mock_method]
        
        boundary_conditions = [
            BoundaryCondition(
                parameter_name="data",
                boundary_type=BoundaryType.NULL_CHECK,
                description="Test with null",
                test_value=None,
                expected_behavior="Handle null"
            )
        ]
        
        scenarios = self.analyzer._generate_test_scenarios(mock_class, boundary_conditions)
        
        assert len(scenarios) > 1
        # 应该有正常场景
        assert any(s.test_type == "normal" for s in scenarios)
        # 应该有边界场景
        assert any(s.test_type == "edge" for s in scenarios)
    
    def test_clear(self):
        """测试清除分析结果"""
        # 先添加一些数据
        self.analyzer.call_graph.add_node(
            CallGraphNode("method1", "Class1", "file1.java")
        )
        self.analyzer.business_logic["Class1"].append(
            BusinessLogic("method1", LogicType.VALIDATION, "desc", 1, 10)
        )
        
        self.analyzer.clear()
        
        assert len(self.analyzer.call_graph.nodes) == 0
        assert len(self.analyzer.business_logic) == 0
        assert len(self.analyzer.data_flows) == 0
        assert len(self.analyzer.test_scenarios) == 0
    
    def test_get_test_scenarios(self):
        """测试获取测试场景"""
        scenarios = self.analyzer.get_test_scenarios()
        assert isinstance(scenarios, list)
    
    def test_get_call_graph(self):
        """测试获取调用图"""
        graph = self.analyzer.get_call_graph()
        assert isinstance(graph, CallGraph)
    
    def test_analyze_file_integration(self):
        """测试文件分析集成"""
        # 创建一个更完整的模拟类
        mock_method1 = Mock()
        mock_method1.name = "createUser"
        mock_method1.return_type = "User"
        mock_method1.parameters = [("String", "name"), ("String", "email")]
        mock_method1.modifiers = ["public"]
        mock_method1.annotations = ["NotNull"]
        mock_method1.start_line = 10
        mock_method1.end_line = 25
        
        mock_method2 = Mock()
        mock_method2.name = "deleteUser"
        mock_method2.return_type = "void"
        mock_method2.parameters = [("Long", "userId")]
        mock_method2.modifiers = ["public"]
        mock_method2.annotations = []
        mock_method2.start_line = 27
        mock_method2.end_line = 35
        
        mock_class = Mock()
        mock_class.name = "UserService"
        mock_class.package = "com.example.service"
        mock_class.methods = [mock_method1, mock_method2]
        mock_class.fields = [
            ("UserRepository", "userRepository", "private")
        ]
        mock_class.imports = ["com.example.User"]
        mock_class.annotations = ["Service"]
        
        result = self.analyzer.analyze_file("/path/to/UserService.java", mock_class)
        
        # 验证结果完整性
        assert result["class_name"] == "UserService"
        assert len(result["test_scenarios"]) >= 2  # 至少每个方法一个正常场景
        assert len(result["boundary_conditions"]) > 0
        assert "call_graph" in result
        assert result["call_graph"]["entry_points"] is not None


class TestSemanticAnalyzerEdgeCases:
    """测试边界情况"""
    
    def setup_method(self):
        self.analyzer = SemanticAnalyzer()
    
    def test_empty_class(self):
        """测试空类分析"""
        mock_class = Mock()
        mock_class.name = "EmptyClass"
        mock_class.package = "com.example"
        mock_class.methods = []
        mock_class.fields = []
        mock_class.imports = []
        mock_class.annotations = []
        
        result = self.analyzer.analyze_file("/path/to/EmptyClass.java", mock_class)
        
        assert result["class_name"] == "EmptyClass"
        assert len(result["test_scenarios"]) == 0
    
    def test_method_with_no_parameters(self):
        """测试无参数方法"""
        mock_method = Mock()
        mock_method.name = "doSomething"
        mock_method.return_type = "void"
        mock_method.parameters = []
        mock_method.modifiers = ["public"]
        mock_method.annotations = []
        mock_method.start_line = 1
        mock_method.end_line = 5
        
        mock_class = Mock()
        mock_class.name = "TestClass"
        mock_class.methods = [mock_method]
        mock_class.fields = []
        mock_class.imports = []
        mock_class.annotations = []
        
        result = self.analyzer.analyze_file("/path/to/TestClass.java", mock_class)
        
        # 应该仍然生成正常场景
        assert any(s["type"] == "normal" for s in result["test_scenarios"])
    
    def test_method_with_complex_return_type(self):
        """测试复杂返回值类型"""
        mock_method = Mock()
        mock_method.name = "getUsers"
        mock_method.return_type = "List<Map<String, Object>>"
        mock_method.parameters = []
        mock_method.modifiers = ["public"]
        mock_method.annotations = []
        mock_method.start_line = 1
        mock_method.end_line = 10
        
        mock_class = Mock()
        mock_class.name = "UserService"
        mock_class.methods = [mock_method]
        mock_class.fields = []
        mock_class.imports = []
        mock_class.annotations = []
        
        result = self.analyzer.analyze_file("/path/to/UserService.java", mock_class)
        
        assert len(result["test_scenarios"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
