"""测试增量式修复器"""
import pytest
from typing import List, Dict
from pyutagent.agent.incremental_fixer import (
    IncrementalFixer,
    TestFailure,
    TestFailureCluster,
    TestResults
)


class TestFailure:
    """测试失败数据类"""
    def __init__(self, test_name: str, error_type: str, message: str):
        self.test_name = test_name
        self.error_type = error_type
        self.message = message


class TestResults:
    """测试结果数据类"""
    def __init__(self, failures: List[TestFailure]):
        self.failures = failures


class MockLLMClient:
    """模拟 LLM 客户端"""
    
    def __init__(self, response: str = None):
        self.response = response or """```java
public int add(int a, int b) {
    return a + b;
}
```"""
        self.call_count = 0
    
    async def agenerate(self, prompt: str) -> str:
        self.call_count += 1
        return self.response


class TestIncrementalFixer:
    """测试增量式修复器"""
    
    def test_group_failures_by_type(self):
        """测试按失败类型分组"""
        fixer = IncrementalFixer()
        
        test_results = TestResults(
            failures=[
                TestFailure("test1", "AssertionError", "Expected 1 but got 2"),
                TestFailure("test2", "AssertionError", "Expected 3 but got 4"),
                TestFailure("test3", "NullPointerException", "obj is null"),
            ]
        )
        
        groups = fixer.group_failures_by_type(test_results)
        
        assert len(groups) == 2
        assert len(groups["AssertionError"]) == 2
        assert len(groups["NullPointerException"]) == 1
    
    def test_cluster_by_root_cause(self):
        """测试按根本原因聚类"""
        fixer = IncrementalFixer()
        
        failures = [
            TestFailure("test1", "AssertionError", "Expected 1 but got 2"),
            TestFailure("test2", "AssertionError", "Expected 1 but got 3"),
            TestFailure("test3", "AssertionError", "obj is null"),
        ]
        
        clusters = fixer.cluster_by_root_cause(failures)
        
        # 应该聚类成 2 组（相似的错误消息）
        assert len(clusters) == 2
    
    def test_string_similarity(self):
        """测试字符串相似度计算"""
        fixer = IncrementalFixer()
        
        # 相似字符串
        s1 = "Expected 1 but got 2"
        s2 = "Expected 1 but got 3"
        similarity = fixer._string_similarity(s1, s2)
        assert similarity > 0.7
        
        # 不相似字符串
        s3 = "obj is null"
        similarity2 = fixer._string_similarity(s1, s3)
        assert similarity2 < 0.7
    
    def test_is_similar(self):
        """测试失败相似性判断"""
        fixer = IncrementalFixer()
        
        f1 = TestFailure("test1", "AssertionError", "Expected 1 but got 2")
        f2 = TestFailure("test2", "AssertionError", "Expected 1 but got 3")
        f3 = TestFailure("test3", "NullPointerException", "obj is null")
        
        # 相似的失败
        assert fixer._is_similar(f1, f2) is True
        
        # 不相似的失败（错误类型不同）
        assert fixer._is_similar(f1, f3) is False
    
    def test_format_failures(self):
        """测试格式化失败信息"""
        fixer = IncrementalFixer()
        
        failures = [
            TestFailure("test1", "AssertionError", "Expected 1 but got 2"),
            TestFailure("test2", "AssertionError", "Expected 3 but got 4"),
        ]
        
        formatted = fixer._format_failures(failures)
        
        assert "test1" in formatted
        assert "test2" in formatted
        assert "AssertionError" in formatted
    
    def test_extract_code(self):
        """测试从响应中提取代码"""
        fixer = IncrementalFixer()
        
        response = """Here's the fixed code:
```java
public int add(int a, int b) {
    return a + b;
}
```
Hope this helps!"""
        
        code = fixer._extract_code(response)
        
        assert "public int add" in code
        assert "return a + b" in code
    
    @pytest.mark.asyncio
    async def test_generate_targeted_fix(self):
        """测试生成针对性修复"""
        mock_llm = MockLLMClient()
        fixer = IncrementalFixer(llm_client=mock_llm)
        
        cluster = TestFailureCluster(
            failures=[TestFailure("test1", "AssertionError", "Expected 1 but got 2")],
            root_cause="Incorrect return value"
        )
        
        current_code = """
public int add(int a, int b) {
    return a - b;  // Bug: should be a + b
}
"""
        
        fixed_code = await fixer.generate_targeted_fix(cluster, current_code)
        
        # 验证 LLM 被调用
        assert mock_llm.call_count == 1
        # 验证返回了修复代码
        assert "add" in fixed_code
    
    def test_create_failure_cluster(self):
        """测试创建失败聚类"""
        failures = [
            TestFailure("test1", "AssertionError", "Expected 1 but got 2"),
            TestFailure("test2", "AssertionError", "Expected 1 but got 3"),
        ]
        
        cluster = TestFailureCluster(
            failures=failures,
            root_cause="Incorrect calculation"
        )
        
        assert len(cluster.failures) == 2
        assert cluster.root_cause == "Incorrect calculation"
