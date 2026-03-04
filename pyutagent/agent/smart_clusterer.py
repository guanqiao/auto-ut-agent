"""智能错误聚类算法 - 基于词向量和语义相似度"""
import hashlib
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import logging

from pyutagent.agent.incremental_fixer import TestFailure, TestFailureCluster

logger = logging.getLogger(__name__)


@dataclass
class ClusteringConfig:
    """聚类配置"""
    similarity_threshold: float = 0.7
    min_cluster_size: int = 1
    dimensions: int = 100
    use_semantic: bool = True
    use_keyword: bool = True
    keyword_weight: float = 0.3


class WordEmbedding:
    """词嵌入 - 简化的词向量表示"""
    
    def __init__(self, dimensions: int = 100):
        self.dimensions = dimensions
        self._cache: Dict[str, List[float]] = {}
    
    def get_vector(self, word: str) -> List[float]:
        """获取词向量"""
        if word in self._cache:
            return self._cache[word]
        
        # 使用哈希生成确定性向量
        vector = self._generate_vector(word)
        self._cache[word] = vector
        
        return vector
    
    def _generate_vector(self, word: str) -> List[float]:
        """生成词向量"""
        # 使用 MD5 哈希生成确定性随机向量
        word_hash = hashlib.md5(word.encode()).hexdigest()
        
        vector = []
        for i in range(self.dimensions):
            # 从哈希中提取字节并转换为 -1 到 1 之间的值
            byte_val = int(word_hash[i % len(word_hash)], 16)
            normalized = (byte_val / 15.0) * 2 - 1  # 归一化到 [-1, 1]
            vector.append(normalized)
        
        # 归一化向量
        magnitude = math.sqrt(sum(v * v for v in vector))
        if magnitude > 0:
            vector = [v / magnitude for v in vector]
        
        return vector


class SemanticSimilarity:
    """语义相似度计算器"""
    
    def __init__(self):
        self._word_embedding = WordEmbedding()
    
    def cosine_similarity(
        self,
        vector1: List[float],
        vector2: List[float]
    ) -> float:
        """计算余弦相似度"""
        if len(vector1) != len(vector2):
            raise ValueError("向量维度必须相同")
        
        # 计算点积
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        
        # 计算模长
        magnitude1 = math.sqrt(sum(a * a for a in vector1))
        magnitude2 = math.sqrt(sum(b * b for b in vector2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def text_similarity(self, text1: str, text2: str) -> float:
        """计算文本语义相似度"""
        # 分词
        words1 = self._tokenize(text1)
        words2 = self._tokenize(text2)
        
        # 生成文本向量（词向量平均）
        vector1 = self._text_to_vector(words1)
        vector2 = self._text_to_vector(words2)
        
        # 计算余弦相似度
        semantic_sim = self.cosine_similarity(vector1, vector2)
        
        # 结合 Jaccard 相似度
        jaccard_sim = self._jaccard_similarity(set(words1), set(words2))
        
        # 加权组合
        return 0.7 * semantic_sim + 0.3 * jaccard_sim
    
    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        # 转为小写并提取单词
        words = re.findall(r'\b[a-z]+\b', text.lower())
        
        # 移除常见停用词
        stopwords = {'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or'}
        return [w for w in words if w not in stopwords]
    
    def _text_to_vector(self, words: List[str]) -> List[float]:
        """将文本转换为向量"""
        if not words:
            return [0.0] * self._word_embedding.dimensions
        
        # 平均所有词向量
        vectors = [self._word_embedding.get_vector(word) for word in words]
        
        result = [0.0] * self._word_embedding.dimensions
        for vector in vectors:
            for i, val in enumerate(vector):
                result[i] += val
        
        # 平均
        result = [v / len(vectors) for v in result]
        
        # 归一化
        magnitude = math.sqrt(sum(v * v for v in result))
        if magnitude > 0:
            result = [v / magnitude for v in result]
        
        return result
    
    def _jaccard_similarity(
        self,
        set1: Set[str],
        set2: Set[str]
    ) -> float:
        """计算 Jaccard 相似度"""
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0


class FailureEmbedding:
    """失败嵌入 - 将测试失败转换为向量"""
    
    def __init__(self, dimensions: int = 100):
        self.dimensions = dimensions
        self._word_embedding = WordEmbedding(dimensions)
        self._semantic = SemanticSimilarity()
    
    def embed(self, failure: TestFailure) -> List[float]:
        """嵌入测试失败"""
        # 组合错误类型和消息
        text = f"{failure.error_type} {failure.message}"
        
        # 使用语义相似度计算器的文本向量
        return self._semantic._text_to_vector(
            self._semantic._tokenize(text)
        )


class SmartClusterer:
    """智能聚类器 - 使用语义相似度进行聚类"""
    
    def __init__(self, config: Optional[ClusteringConfig] = None):
        self.config = config or ClusteringConfig()
        self._embedding = FailureEmbedding(dimensions=self.config.dimensions)
        self._semantic = SemanticSimilarity()
    
    def cluster_failures(
        self,
        failures: List[TestFailure]
    ) -> List[TestFailureCluster]:
        """聚类失败"""
        if not failures:
            return []
        
        start_time = time.time()
        logger.debug(f"Starting clustering for {len(failures)} failures")
        
        clusters: List[TestFailureCluster] = []
        
        for failure in failures:
            # 找到最相似的聚类
            best_cluster = None
            best_similarity = 0.0
            
            for cluster in clusters:
                similarity = self._compute_cluster_similarity(
                    failure,
                    cluster
                )
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = cluster
            
            # 如果找到相似的聚类，加入；否则创建新聚类
            if best_similarity >= self.config.similarity_threshold:
                best_cluster.failures.append(failure)
            else:
                new_cluster = TestFailureCluster(
                    failures=[failure],
                    root_cause=self._extract_root_cause([failure])
                )
                clusters.append(new_cluster)
        
        elapsed = time.time() - start_time
        logger.debug(
            f"Clustering completed in {elapsed:.3f}s, "
            f"created {len(clusters)} clusters"
        )
        
        return clusters
    
    def _compute_cluster_similarity(
        self,
        failure: TestFailure,
        cluster: TestFailureCluster
    ) -> float:
        """计算失败与聚类的相似度"""
        if not cluster.failures:
            return 0.0
        
        # 与聚类中所有失败的相似度平均
        similarities = []
        for existing_failure in cluster.failures:
            sim = self._compute_failure_similarity(failure, existing_failure)
            similarities.append(sim)
        
        return sum(similarities) / len(similarities)
    
    def _compute_failure_similarity(
        self,
        failure1: TestFailure,
        failure2: TestFailure
    ) -> float:
        """计算两个失败的相似度"""
        # 错误类型必须相同或相似
        if failure1.error_type != failure2.error_type:
            # 如果是完全不同的错误类型，直接返回低相似度
            return 0.0
        
        # 计算语义相似度
        semantic_sim = self._semantic_similarity(failure1, failure2)
        
        # 计算关键词相似度
        keyword_sim = self._keyword_similarity(failure1, failure2)
        
        # 加权组合
        if self.config.use_semantic and self.config.use_keyword:
            return (
                (1 - self.config.keyword_weight) * semantic_sim +
                self.config.keyword_weight * keyword_sim
            )
        elif self.config.use_semantic:
            return semantic_sim
        else:
            return keyword_sim
    
    def _semantic_similarity(
        self,
        failure1: TestFailure,
        failure2: TestFailure
    ) -> float:
        """计算语义相似度"""
        vector1 = self._embedding.embed(failure1)
        vector2 = self._embedding.embed(failure2)
        
        return self._semantic.cosine_similarity(vector1, vector2)
    
    def _keyword_similarity(
        self,
        failure1: TestFailure,
        failure2: TestFailure
    ) -> float:
        """计算关键词相似度"""
        # 提取错误消息中的关键词
        keywords1 = self._extract_keywords(failure1.message)
        keywords2 = self._extract_keywords(failure2.message)
        
        if not keywords1 and not keywords2:
            return 0.0
        
        # 计算 Jaccard 相似度
        intersection = len(set(keywords1) & set(keywords2))
        union = len(set(keywords1) | set(keywords2))
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 转为小写
        text = text.lower()
        
        # 提取单词和数字
        tokens = re.findall(r'\b[a-z]+|\d+\b', text)
        
        # 移除停用词
        stopwords = {
            'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'expected', 'but', 'got', 'actual', 'value'
        }
        keywords = [t for t in tokens if t not in stopwords]
        
        # 返回最常见的关键词
        return keywords
    
    def _extract_root_cause(
        self,
        failures: List[TestFailure]
    ) -> str:
        """提取根本原因"""
        if not failures:
            return "Unknown root cause"
        
        # 找到最常见的错误类型
        error_types = Counter(f.error_type for f in failures)
        most_common_type = error_types.most_common(1)[0][0]
        
        # 提取共同的模式
        messages = [f.message for f in failures]
        common_pattern = self._find_common_pattern(messages)
        
        if common_pattern:
            return f"{most_common_type}: {common_pattern}"
        else:
            return f"{most_common_type}: Multiple occurrences detected"
    
    def _find_common_pattern(
        self,
        messages: List[str]
    ) -> str:
        """找到共同模式"""
        if not messages:
            return ""
        
        if len(messages) == 1:
            return messages[0]
        
        # 找到所有消息的共同部分
        common_words = None
        for message in messages:
            words = set(self._extract_keywords(message))
            if common_words is None:
                common_words = words
            else:
                common_words &= words
        
        if common_words:
            return " ".join(sorted(common_words))
        
        # 如果没有共同词，返回第一个消息的摘要
        return messages[0][:50] + "..." if len(messages[0]) > 50 else messages[0]
