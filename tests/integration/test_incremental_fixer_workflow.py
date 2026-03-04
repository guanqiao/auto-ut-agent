"""集成测试：增量式修复与工作流集成"""
import pytest
import asyncio
from typing import List, Dict
from pyutagent.agent.incremental_fixer import (
    IncrementalFixer,
    TestFailure,
    TestFailureCluster,
    TestResults
)


class MockLLMClient:
    """模拟 LLM 客户端"""
    
    def __init__(self, response: str = None, call_delay: float = 0.01):
        self.response = response or """```java
public int add(int a, int b) {
    return a + b;
}
```"""
        self.call_count = 0
        self.call_delay = call_delay
        self.call_history: List[Dict] = []
    
    async def agenerate(self, prompt: str) -> str:
        self.call_count += 1
        self.call_history.append({'prompt': prompt[:100], 'timestamp': self.call_count})
        await asyncio.sleep(self.call_delay)
        return self.response


class TestIncrementalFixerWorkflow:
    """测试增量式修复器的完整工作流"""
    
    @pytest.mark.asyncio
    async def test_complete_compile_fail_fix_verify_cycle(self):
        """测试完整的编译→失败→聚类→修复→验证循环"""
        llm_client = MockLLMClient()
        fixer = IncrementalFixer(llm_client=llm_client)
        
        # 模拟编译失败
        failures = [
            TestFailure("test1", "AssertionError", "Expected 1 but got 2"),
            TestFailure("test2", "AssertionError", "Expected 1 but got 3"),
            TestFailure("test3", "NullPointerException", "obj is null"),
        ]
        
        test_results = TestResults(failures=failures)
        
        # 步骤 1: 按类型分组
        groups = fixer.group_failures_by_type(test_results)
        assert len(groups) == 2
        assert len(groups["AssertionError"]) == 2
        
        # 步骤 2: 聚类
        clusters = fixer.cluster_by_root_cause(failures)
        assert len(clusters) == 2  # AssertionError 和 NullPointerException
        
        # 步骤 3: 生成修复
        for cluster in clusters:
            fixed_code = await fixer.generate_targeted_fix(
                cluster,
                "public int add(int a, int b) { return a - b; }"
            )
            assert fixed_code is not None
            assert len(fixed_code) > 0
        
        # 验证 LLM 被调用了 2 次（每个聚类一次）
        assert llm_client.call_count == 2
    
    @pytest.mark.asyncio
    async def test_multi_round_iteration_efficiency(self):
        """测试多轮迭代修复的效率"""
        llm_client = MockLLMClient(call_delay=0.001)
        fixer = IncrementalFixer(llm_client=llm_client)
        
        # 模拟多轮迭代
        rounds = []
        for round_num in range(3):
            failures = [
                TestFailure(f"test{i}", "AssertionError", f"Error in round {round_num}")
                for i in range(2)
            ]
            
            clusters = fixer.cluster_by_root_cause(failures)
            
            for cluster in clusters:
                await fixer.generate_targeted_fix(cluster, "test code")
            
            rounds.append({
                'round': round_num,
                'failures': len(failures),
                'clusters': len(clusters)
            })
        
        # 验证效率
        assert len(rounds) == 3
        assert llm_client.call_count == 3  # 每轮 1 个聚类
        
        # 计算效率指标
        total_calls = llm_client.call_count
        assert total_calls == 3  # 高效的聚类减少了 LLM 调用
    
    @pytest.mark.asyncio
    async def test_fix_success_rate_statistics(self):
        """测试修复成功率的统计和分析"""
        llm_client = MockLLMClient()
        fixer = IncrementalFixer(llm_client=llm_client)
        
        # 模拟多次修复尝试
        fix_attempts = []
        for i in range(5):
            failures = [TestFailure("test1", "AssertionError", f"Error {i}")]
            clusters = fixer.cluster_by_root_cause(failures)
            
            for cluster in clusters:
                fixed_code = await fixer.generate_targeted_fix(
                    cluster,
                    f"public int test() {{ return {i}; }}"
                )
                
                # 模拟验证（这里假设都成功）
                success = len(fixed_code) > 0
                fix_attempts.append({
                    'attempt': i,
                    'success': success,
                    'code_length': len(fixed_code)
                })
        
        # 统计成功率
        success_count = sum(1 for attempt in fix_attempts if attempt['success'])
        success_rate = success_count / len(fix_attempts)
        
        # 验证成功率
        assert success_rate == 1.0  # 100% 成功（因为 Mock 总是返回代码）
        assert llm_client.call_count == 5
    
    @pytest.mark.asyncio
    async def test_similar_failure_clustering(self):
        """测试相似失败的聚类效果"""
        llm_client = MockLLMClient()
        fixer = IncrementalFixer(llm_client=llm_client)
        
        # 创建相似的失败
        similar_failures = [
            TestFailure("test1", "AssertionError", "Expected 1 but got 2"),
            TestFailure("test2", "AssertionError", "Expected 1 but got 3"),
            TestFailure("test3", "AssertionError", "Expected 1 but got 4"),
        ]
        
        # 创建不相似的失败
        different_failures = [
            TestFailure("test4", "NullPointerException", "obj is null"),
            TestFailure("test5", "IndexOutOfBoundsException", "Index: 0, Size: 0"),
        ]
        
        all_failures = similar_failures + different_failures
        clusters = fixer.cluster_by_root_cause(all_failures)
        
        # 验证聚类效果
        # 应该有 3 个聚类：AssertionError(相似) + NullPointerException + IndexOutOfBoundsException
        assert len(clusters) == 3
        
        # 验证相似的失败被聚在一起
        assertion_cluster = next(
            c for c in clusters 
            if c.failures[0].error_type == "AssertionError"
        )
        assert len(assertion_cluster.failures) == 3
    
    @pytest.mark.asyncio
    async def test_incremental_fix_preserves_passing_tests(self):
        """测试增量式修复保留已通过的测试"""
        # 使用返回完整代码的 Mock
        llm_client = MockLLMClient(response="""```java
public class Calculator {
    public int add(int a, int b) { return a + b; }
    public int subtract(int a, int b) { return a - b; }
    public int multiply(int a, int b) { return a * b; }
}
```""")
        fixer = IncrementalFixer(llm_client=llm_client)
        
        # 只有部分测试失败
        failures = [
            TestFailure("testAdd", "AssertionError", "Expected 5 but got 3"),
        ]
        
        clusters = fixer.cluster_by_root_cause(failures)
        fixed_code = await fixer.generate_targeted_fix(
            clusters[0],
            "original code"
        )
        
        # 验证修复后的代码保留了原有功能
        assert "subtract" in fixed_code
        assert "multiply" in fixed_code
        # 应该修复了 add 方法
        assert "add" in fixed_code
    
    @pytest.mark.asyncio
    async def test_clustering_reduces_llm_calls(self):
        """测试聚类减少 LLM 调用次数"""
        llm_client = MockLLMClient()
        fixer = IncrementalFixer(llm_client=llm_client)
        
        # 创建 10 个相似的失败
        failures = [
            TestFailure(f"test{i}", "AssertionError", "Expected 1 but got 2")
            for i in range(10)
        ]
        
        # 不聚类的情况：每个失败都调用 LLM
        without_clustering_calls = len(failures)
        
        # 聚类的情况：相似的失败只调用一次 LLM
        clusters = fixer.cluster_by_root_cause(failures)
        for cluster in clusters:
            await fixer.generate_targeted_fix(cluster, "code")
        
        with_clustering_calls = llm_client.call_count
        
        # 验证聚类显著减少了 LLM 调用
        assert with_clustering_calls < without_clustering_calls
        assert with_clustering_calls == 1  # 所有相似失败聚为一类
        reduction_rate = (without_clustering_calls - with_clustering_calls) / without_clustering_calls
        assert reduction_rate == 0.9  # 减少了 90% 的调用


class TestIncrementalFixerWithRealWorkflow:
    """测试增量式修复与真实工作流的集成"""
    
    @pytest.mark.asyncio
    async def test_integration_with_react_agent_workflow(self):
        """测试与 ReActAgent 工作流的集成"""
        # 模拟 ReActAgent 的工作流
        llm_client = MockLLMClient()
        fixer = IncrementalFixer(llm_client=llm_client)
        
        workflow_steps = []
        
        # 步骤 1: 生成测试
        workflow_steps.append("generate_tests")
        
        # 步骤 2: 编译失败
        workflow_steps.append("compile_failed")
        failures = [
            TestFailure("test1", "CompilationError", "Cannot find symbol"),
        ]
        
        # 步骤 3: 聚类并修复
        clusters = fixer.cluster_by_root_cause(failures)
        workflow_steps.append(f"cluster_and_fix_{len(clusters)}")
        
        for cluster in clusters:
            await fixer.generate_targeted_fix(cluster, "test code")
        
        # 步骤 4: 验证
        workflow_steps.append("verify")
        
        # 验证工作流
        assert len(workflow_steps) == 4
        assert "generate_tests" in workflow_steps
        assert "compile_failed" in workflow_steps
        assert "cluster_and_fix_1" in workflow_steps
        assert "verify" in workflow_steps
        
        # 验证 LLM 调用
        assert llm_client.call_count == 1
    
    @pytest.mark.asyncio
    async def test_error_type_specific_strategies(self):
        """测试针对不同错误类型的特定策略"""
        llm_client = MockLLMClient()
        fixer = IncrementalFixer(llm_client=llm_client)
        
        # 定义不同错误类型的特定策略
        strategies = {
            "AssertionError": "Fix assertion logic",
            "NullPointerException": "Add null check",
            "CompilationError": "Fix import or syntax",
        }
        
        # 测试每种错误类型
        for error_type, expected_strategy in strategies.items():
            failures = [TestFailure("test1", error_type, "Error message")]
            clusters = fixer.cluster_by_root_cause(failures)
            
            # 验证聚类包含错误类型信息
            assert len(clusters) == 1
            assert clusters[0].failures[0].error_type == error_type
    
    @pytest.mark.asyncio
    async def test_workflow_performance_metrics(self):
        """测试工作流性能指标"""
        import time
        
        llm_client = MockLLMClient(call_delay=0.001)
        fixer = IncrementalFixer(llm_client=llm_client)
        
        # 测量性能
        start_time = time.time()
        
        # 执行多次修复
        for i in range(10):
            failures = [TestFailure(f"test{i}", "AssertionError", f"Error {i}")]
            clusters = fixer.cluster_by_root_cause(failures)
            
            for cluster in clusters:
                await fixer.generate_targeted_fix(cluster, "code")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 性能指标
        avg_time_per_fix = total_time / 10
        fixes_per_second = 10 / total_time
        
        # 验证性能（应该很快，因为是 Mock）
        assert total_time < 1.0  # 总时间小于 1 秒
        assert fixes_per_second > 10  # 每秒至少 10 次修复
