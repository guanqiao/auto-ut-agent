"""单元测试 - IntelligenceEnhancedCoT"""

import pytest
from pathlib import Path
from unittest.mock import Mock

from pyutagent.llm.intelligence_cot import (
    IntelligenceEnhancedCoT,
    IntelligenceContext,
    create_intelligence_enhanced_cot
)
from pyutagent.llm.chain_of_thought import PromptCategory


class TestIntelligenceContext:
    """测试 IntelligenceContext 数据类"""
    
    def test_intelligence_context_creation(self):
        """测试智能上下文创建"""
        context = IntelligenceContext(
            test_scenarios=[{"id": "1", "description": "Test scenario"}],
            boundary_conditions=[{"parameter": "param1", "type": "NULL_CHECK"}],
            business_logic=[{"method": "method1", "type": "VALIDATION"}],
            call_graph_info={"nodes": [], "edges": []},
            root_cause_analysis=None,
            code_complexity=3,
            recommended_focus_areas=["High complexity methods"]
        )
        
        assert len(context.test_scenarios) == 1
        assert len(context.boundary_conditions) == 1
        assert context.code_complexity == 3


class TestIntelligenceEnhancedCoT:
    """测试 IntelligenceEnhancedCoT 类"""
    
    def setup_method(self):
        """测试前准备"""
        self.cot = IntelligenceEnhancedCoT()
    
    def test_initialization(self):
        """测试初始化"""
        cot = IntelligenceEnhancedCoT()
        
        assert cot.cot_engine is not None
        assert cot.semantic_analyzer is not None
        assert cot.root_cause_analyzer is not None
    
    def test_generate_enhanced_test_prompt(self):
        """测试生成增强的测试提示"""
        # 创建模拟对象
        mock_method = Mock()
        mock_method.name = "calculateTotal"
        mock_method.return_type = "int"
        mock_method.parameters = [("int", "amount"), ("int", "tax")]
        mock_method.modifiers = ["public"]
        mock_method.annotations = []
        mock_method.start_line = 10
        mock_method.end_line = 20
        
        mock_class = Mock()
        mock_class.name = "Calculator"
        mock_class.package = "com.example"
        mock_class.methods = [mock_method]
        mock_class.fields = []
        mock_class.imports = []
        mock_class.annotations = []
        
        source_code = """
public class Calculator {
    public int calculateTotal(int amount, int tax) {
        return amount + tax;
    }
}
        """
        
        enhanced_prompt = self.cot.generate_enhanced_test_prompt(
            source_code,
            mock_class,
            "/path/to/Calculator.java"
        )
        
        assert enhanced_prompt is not None
        assert len(enhanced_prompt) > 0
        assert "Semantic Analysis Results" in enhanced_prompt
        assert "Recommended Test Scenarios" in enhanced_prompt
        assert "Boundary Conditions" in enhanced_prompt
    
    def test_generate_enhanced_fix_prompt(self):
        """测试生成增强的修复提示"""
        error_message = """
Tests run: 5, Failures: 1
CalculatorTest.testCalculateTotal
Expected: 100 but was: 90
        """
        
        test_code = """
@Test
void testCalculateTotal() {
    Calculator calc = new Calculator();
    assertEquals(100, calc.calculateTotal(90, 10));
}
        """
        
        source_code = """
public class Calculator {
    public int calculateTotal(int amount, int tax) {
        return amount + tax;
    }
}
        """
        
        enhanced_prompt = self.cot.generate_enhanced_fix_prompt(
            error_message,
            test_code,
            source_code,
            "testCalculateTotal"
        )
        
        assert enhanced_prompt is not None
        assert len(enhanced_prompt) > 0
        assert "Root Cause Analysis" in enhanced_prompt
        assert "Recommended Fix Strategies" in enhanced_prompt
    
    def test_generate_boundary_focused_prompt(self):
        """测试生成边界聚焦提示"""
        mock_method = Mock()
        mock_method.name = "processData"
        mock_method.return_type = "String"
        mock_method.parameters = [("String", "data"), ("int", "count")]
        mock_method.modifiers = ["public"]
        mock_method.annotations = []
        mock_method.start_line = 1
        mock_method.end_line = 10
        
        mock_class = Mock()
        mock_class.name = "DataProcessor"
        mock_class.package = "com.example"
        mock_class.methods = [mock_method]
        mock_class.fields = []
        mock_class.imports = []
        mock_class.annotations = []
        
        source_code = """
public class DataProcessor {
    public String processData(String data, int count) {
        return data.repeat(count);
    }
}
        """
        
        enhanced_prompt = self.cot.generate_boundary_focused_prompt(
            source_code,
            mock_class,
            "/path/to/DataProcessor.java",
            target_method="processData"
        )
        
        assert enhanced_prompt is not None
        assert "Boundary Condition" in enhanced_prompt
        assert "Parameter Analysis" in enhanced_prompt
    
    def test_build_intelligence_context(self):
        """测试构建智能上下文"""
        semantic_result = {
            "test_scenarios": [
                {"id": "1", "description": "Scenario 1", "target": "method1", "type": "normal", "priority": 1}
            ],
            "boundary_conditions": [
                {"parameter": "param1", "type": "NULL_CHECK", "test_value": None, "expected_behavior": "Throw exception"}
            ],
            "business_logic": [
                {"method": "method1", "type": "VALIDATION", "description": "Validates input"}
            ],
            "call_graph": {
                "nodes": [
                    {"method": "method1", "complexity": 5}
                ],
                "edges": []
            },
            "data_flows": []
        }
        
        context = self.cot._build_intelligence_context(semantic_result)
        
        assert isinstance(context, IntelligenceContext)
        assert len(context.test_scenarios) == 1
        assert len(context.boundary_conditions) == 1
        assert context.code_complexity > 0
    
    def test_format_business_logic_types(self):
        """测试格式化业务逻辑类型"""
        business_logic = [
            {"type": "VALIDATION"},
            {"type": "TRANSFORMATION"},
            {"type": "PERSISTENCE"}
        ]
        
        formatted = self.cot._format_business_logic_types(business_logic)
        
        assert "VALIDATION" in formatted
        assert "TRANSFORMATION" in formatted
        assert "PERSISTENCE" in formatted
    
    def test_format_complexity_scores(self):
        """测试格式化复杂度分数"""
        call_graph_info = {
            "nodes": [
                {"method": "method1", "complexity": 5},
                {"method": "method2", "complexity": 3}
            ]
        }
        
        formatted = self.cot._format_complexity_scores(call_graph_info)
        
        assert "method1" in formatted
        assert "5" in formatted
        assert "method2" in formatted
        assert "3" in formatted
    
    def test_format_test_scenarios(self):
        """测试格式化测试场景"""
        scenarios = [
            {
                "id": "1",
                "description": "Test normal case",
                "target": "method1",
                "type": "normal",
                "priority": 1
            },
            {
                "id": "2",
                "description": "Test edge case",
                "target": "method1",
                "type": "edge",
                "priority": 2
            }
        ]
        
        formatted = self.cot._format_test_scenarios(scenarios)
        
        assert "Test normal case" in formatted
        assert "Test edge case" in formatted
        assert "Priority: 1" in formatted
    
    def test_format_boundary_conditions(self):
        """测试格式化边界条件"""
        boundaries = [
            {
                "parameter": "username",
                "type": "NULL_CHECK",
                "test_value": None,
                "expected_behavior": "Throw IllegalArgumentException"
            },
            {
                "parameter": "count",
                "type": "RANGE_CHECK",
                "test_value": 0,
                "expected_behavior": "Handle zero correctly"
            }
        ]
        
        formatted = self.cot._format_boundary_conditions(boundaries)
        
        assert "username" in formatted
        assert "NULL_CHECK" in formatted
        assert "count" in formatted
        assert "RANGE_CHECK" in formatted
    
    def test_format_recommended_fixes(self):
        """测试格式化推荐修复"""
        from pyutagent.core.root_cause_analyzer import FixStrategy, FixStrategyType
        
        fixes = [
            FixStrategy(
                strategy_id="fix_001",
                strategy_type=FixStrategyType.SYNTAX_FIX,
                description="Fix syntax error",
                priority=1,
                estimated_effort="low",
                success_probability=0.95
            ),
            FixStrategy(
                strategy_id="fix_002",
                strategy_type=FixStrategyType.TYPE_CORRECTION,
                description="Correct type mismatch",
                priority=2,
                estimated_effort="medium",
                success_probability=0.85
            )
        ]
        
        formatted = self.cot._format_recommended_fixes(fixes)
        
        assert "Fix syntax error" in formatted
        assert "Correct type mismatch" in formatted
        assert "Priority: 1" in formatted
        assert "Success: 95%" in formatted
    
    def test_calculate_overall_complexity(self):
        """测试计算整体复杂度"""
        semantic_result = {
            "call_graph": {
                "nodes": [
                    {"complexity": 5},
                    {"complexity": 3},
                    {"complexity": 7}
                ]
            }
        }
        
        complexity = self.cot._calculate_overall_complexity(semantic_result)
        
        # 平均值 (5+3+7)/3 = 5
        assert complexity == 5
    
    def test_identify_focus_areas(self):
        """测试识别重点关注区域"""
        semantic_result = {
            "call_graph": {
                "nodes": [
                    {"method": "complexMethod", "complexity": 8},
                    {"method": "simpleMethod", "complexity": 2}
                ]
            },
            "boundary_conditions": [{"parameter": "p1"}] * 10,  # 10 个边界条件
            "business_logic": [
                {"type": "VALIDATION"},
                {"type": "TRANSFORMATION"},
                {"type": "PERSISTENCE"}
            ]
        }
        
        focus_areas = self.cot._identify_focus_areas(semantic_result)
        
        assert len(focus_areas) > 0
        assert any("High complexity" in area for area in focus_areas)
        assert any("boundary" in area.lower() for area in focus_areas)
        assert any("business logic" in area.lower() for area in focus_areas)
    
    def test_extract_method_signature(self):
        """测试提取方法签名"""
        mock_method = Mock()
        mock_method.name = "calculate"
        mock_method.return_type = "int"
        mock_method.parameters = [("int", "a"), ("int", "b")]
        
        mock_class = Mock()
        mock_class.name = "Calculator"
        mock_class.methods = [mock_method]
        
        signature = self.cot._extract_method_signature(mock_class, "calculate")
        
        assert "int calculate(int a, int b)" in signature
    
    def test_get_available_enhanced_prompts(self):
        """测试获取可用的增强提示"""
        prompts = self.cot.get_available_enhanced_prompts()
        
        assert len(prompts) >= 4
        assert any(p["name"] == "semantic_enhanced_test_generation" for p in prompts)
        assert any(p["name"] == "rca_enhanced_error_fix" for p in prompts)
        assert any(p["name"] == "boundary_enhanced_generation" for p in prompts)


class TestIntelligenceEnhancedCoTIntegration:
    """测试 IntelligenceEnhancedCoT 集成"""
    
    def setup_method(self):
        self.cot = IntelligenceEnhancedCoT()
    
    def test_full_test_generation_workflow(self):
        """测试完整的测试生成工作流"""
        # 1. 创建模拟类
        mock_method = Mock()
        mock_method.name = "processUser"
        mock_method.return_type = "User"
        mock_method.parameters = [("UserDTO", "dto"), ("boolean", "validate")]
        mock_method.modifiers = ["public"]
        mock_method.annotations = ["NotNull"]
        mock_method.start_line = 10
        mock_method.end_line = 30
        
        mock_class = Mock()
        mock_class.name = "UserService"
        mock_class.package = "com.example.service"
        mock_class.methods = [mock_method]
        mock_class.fields = [("UserRepository", "repo", "private")]
        mock_class.imports = ["com.example.User"]
        mock_class.annotations = ["Service"]
        
        source_code = """
@Service
public class UserService {
    @Autowired
    private UserRepository repo;
    
    public User processUser(UserDTO dto, boolean validate) {
        if (validate && dto == null) {
            throw new IllegalArgumentException("DTO cannot be null");
        }
        return repo.save(dto.toEntity());
    }
}
        """
        
        # 2. 生成增强提示
        enhanced_prompt = self.cot.generate_enhanced_test_prompt(
            source_code,
            mock_class,
            "/path/to/UserService.java"
        )
        
        # 3. 验证提示质量
        assert enhanced_prompt is not None
        assert len(enhanced_prompt) > 500  # 应该是详细的提示
        assert "UserService" in enhanced_prompt
        assert "processUser" in enhanced_prompt
        assert "boundary" in enhanced_prompt.lower()
        assert "test scenario" in enhanced_prompt.lower()
    
    def test_full_error_fix_workflow(self):
        """测试完整的错误修复工作流"""
        error_message = """
Tests run: 10, Failures: 2, Errors: 1
UserServiceTest::testProcessUserWithNullDto
Expected: IllegalArgumentException but was: NullPointerException

UserServiceTest::testProcessUserWithValidDto
Expected: User but was: null
        """
        
        test_code = """
@Test
void testProcessUserWithNullDto() {
    UserService service = new UserService();
    assertThrows(IllegalArgumentException.class, () -> {
        service.processUser(null, true);
    });
}
        """
        
        source_code = """
public class UserService {
    public User processUser(UserDTO dto, boolean validate) {
        if (validate && dto == null) {
            throw new IllegalArgumentException("DTO cannot be null");
        }
        return null;
    }
}
        """
        
        enhanced_prompt = self.cot.generate_enhanced_fix_prompt(
            error_message,
            test_code,
            source_code,
            "testProcessUserWithNullDto"
        )
        
        assert enhanced_prompt is not None
        assert "Root Cause Analysis" in enhanced_prompt
        assert "error" in enhanced_prompt.lower()
        assert "fix" in enhanced_prompt.lower()


class TestFactoryFunction:
    """测试工厂函数"""
    
    def test_create_intelligence_enhanced_cot(self):
        """测试创建智能化增强 CoT 引擎"""
        cot = create_intelligence_enhanced_cot()
        
        assert isinstance(cot, IntelligenceEnhancedCoT)
        assert cot.cot_engine is not None
        assert cot.semantic_analyzer is not None


class TestIntelligenceEnhancedCoTEdgeCases:
    """测试边界情况"""
    
    def setup_method(self):
        self.cot = IntelligenceEnhancedCoT()
    
    def test_empty_class_test_generation(self):
        """测试空类的测试生成提示"""
        mock_class = Mock()
        mock_class.name = "EmptyClass"
        mock_class.methods = []
        mock_class.fields = []
        mock_class.imports = []
        mock_class.annotations = []
        
        source_code = "public class EmptyClass {}"
        
        enhanced_prompt = self.cot.generate_enhanced_test_prompt(
            source_code,
            mock_class,
            "/path/to/EmptyClass.java"
        )
        
        assert enhanced_prompt is not None
        # 应该仍然生成有效的提示，即使没有内容
    
    def test_method_not_found_boundary_prompt(self):
        """测试方法未找到的边界提示"""
        mock_class = Mock()
        mock_class.name = "TestClass"
        mock_class.methods = []
        mock_class.fields = []
        mock_class.imports = []
        mock_class.annotations = []
        
        source_code = "public class TestClass {}"
        
        enhanced_prompt = self.cot.generate_boundary_focused_prompt(
            source_code,
            mock_class,
            "/path/to/TestClass.java",
            target_method="nonExistentMethod"
        )
        
        assert enhanced_prompt is not None
        assert "not found" in enhanced_prompt.lower() or "No specific method" in enhanced_prompt
    
    def test_no_root_causes_fix_prompt(self):
        """测试没有根因的修复提示"""
        error_message = "BUILD SUCCESS"
        
        enhanced_prompt = self.cot.generate_enhanced_fix_prompt(
            error_message,
            "",
            "",
            "testMethod"
        )
        
        assert enhanced_prompt is not None
        # 应该处理没有错误的情况


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
