"""增量式修复器 - 仅修复失败的部分"""
from typing import List, Dict, Any
from pyutagent.llm.client import LLMClient
import logging

logger = logging.getLogger(__name__)


class TestFailure:
    """测试失败"""
    
    def __init__(self, test_name: str, error_type: str, message: str):
        self.test_name = test_name
        self.error_type = error_type
        self.message = message


class TestFailureCluster:
    """测试失败聚类"""
    
    def __init__(
        self,
        failures: List[TestFailure],
        root_cause: str
    ):
        self.failures = failures
        self.root_cause = root_cause


class TestResults:
    """测试结果"""
    
    def __init__(self, failures: List[TestFailure]):
        self.failures = failures


class IncrementalFixer:
    """增量式修复器"""
    
    def __init__(self, llm_client: LLMClient = None):
        self.llm_client = llm_client
    
    def group_failures_by_type(
        self,
        test_results: TestResults
    ) -> Dict[str, List[TestFailure]]:
        """按失败类型分组"""
        groups: Dict[str, List[TestFailure]] = {}
        
        for failure in test_results.failures:
            error_type = failure.error_type
            if error_type not in groups:
                groups[error_type] = []
            groups[error_type].append(failure)
        
        return groups
    
    def cluster_by_root_cause(
        self,
        failures: List[TestFailure]
    ) -> List[TestFailureCluster]:
        """按根本原因聚类"""
        clusters = []
        
        for failure in failures:
            # 查找相似的聚类
            similar_cluster = None
            for cluster in clusters:
                if self._is_similar(failure, cluster.failures[0]):
                    similar_cluster = cluster
                    break
            
            if similar_cluster:
                similar_cluster.failures.append(failure)
            else:
                clusters.append(TestFailureCluster(
                    failures=[failure],
                    root_cause=failure.message
                ))
        
        return clusters
    
    def _is_similar(
        self,
        failure1: TestFailure,
        failure2: TestFailure
    ) -> bool:
        """判断两个失败是否相似"""
        # 错误类型必须相同
        if failure1.error_type != failure2.error_type:
            return False
        
        # 消息相似度>0.7
        similarity = self._string_similarity(
            failure1.message,
            failure2.message
        )
        return similarity > 0.7
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """计算字符串相似度"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, s1, s2).ratio()
    
    async def generate_targeted_fix(
        self,
        cluster: TestFailureCluster,
        current_code: str
    ) -> str:
        """生成针对性修复"""
        if not self.llm_client:
            logger.warning("No LLM client available, returning original code")
            return current_code
        
        prompt = f"""Fix the following code to address these test failures:

Failures:
{self._format_failures(cluster.failures)}

Root Cause: {cluster.root_cause}

Current Code:
```java
{current_code}
```

Output only the fixed code:"""
        
        response = await self.llm_client.agenerate(prompt)
        return self._extract_code(response)
    
    def _format_failures(self, failures: List[TestFailure]) -> str:
        """格式化失败信息"""
        lines = []
        for f in failures:
            lines.append(f"- {f.test_name}: {f.error_type} - {f.message}")
        return "\n".join(lines)
    
    def _extract_code(self, response: str) -> str:
        """从响应中提取代码"""
        import re
        match = re.search(r'```java\s*(.*?)\s*```', response, re.DOTALL)
        if match:
            return match.group(1)
        return response
