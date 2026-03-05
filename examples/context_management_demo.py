"""智能上下文管理演示脚本

演示代码索引、语义搜索和智能上下文组装功能。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from pathlib import Path
from typing import List, Dict, Any

from pyutagent.indexing import (
    CodeIndexer,
    IndexConfig,
    CodeChunker,
    ChunkingConfig,
    ContextAssembler,
    AssemblerConfig,
    ContextStrategy,
)


class MockEmbeddingModel:
    """模拟 Embedding 模型"""
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """生成模拟的 embedding 向量"""
        import hashlib
        embeddings = []
        for text in texts:
            # 使用文本哈希生成确定性向量
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
            # 生成 384 维向量
            vector = [(hash_val % 1000) / 1000.0 + (i % 10) / 100.0 for i in range(384)]
            embeddings.append(vector)
        return embeddings


class MockVectorStore:
    """模拟向量存储"""
    
    def __init__(self):
        self.data: List[Dict[str, Any]] = []
    
    def add(self, texts: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]]):
        """添加数据到向量存储"""
        for text, embedding, metadata in zip(texts, embeddings, metadatas):
            self.data.append({
                "text": text,
                "embedding": embedding,
                "metadata": metadata,
            })
    
    def search(self, query_embedding: List[float], k: int = 10) -> List[tuple]:
        """模拟向量搜索 - 返回最相似的 k 个结果"""
        # 计算余弦相似度
        def cosine_similarity(a: List[float], b: List[float]) -> float:
            import math
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                return 0
            return dot / (norm_a * norm_b)
        
        # 计算所有相似度并排序
        results = []
        for item in self.data:
            similarity = cosine_similarity(query_embedding, item["embedding"])
            results.append((item["text"], 1.0 - similarity, item["metadata"]))
        
        results.sort(key=lambda x: x[1])  # 按距离排序
        return results[:k]
    
    def delete(self, file_path: str):
        """删除指定文件的数据"""
        self.data = [
            item for item in self.data 
            if item["metadata"].get("file_path") != file_path
        ]
    
    def clear(self):
        """清空所有数据"""
        self.data.clear()


def demo_code_chunking():
    """演示代码分块"""
    print("\n" + "="*60)
    print("演示 1: 代码分块 (Code Chunking)")
    print("="*60)
    
    chunker = CodeChunker(ChunkingConfig(max_chunk_lines=100))
    
    # 示例 Java 代码
    java_code = '''
public class UserService {
    private UserRepository userRepository;
    private EmailService emailService;
    
    public UserService(UserRepository userRepository, EmailService emailService) {
        this.userRepository = userRepository;
        this.emailService = emailService;
    }
    
    public User createUser(String email, String name) {
        if (email == null || email.isEmpty()) {
            throw new IllegalArgumentException("Email is required");
        }
        
        User user = new User(email, name);
        userRepository.save(user);
        emailService.sendWelcomeEmail(user);
        
        return user;
    }
    
    public User findById(Long id) {
        return userRepository.findById(id)
            .orElseThrow(() -> new UserNotFoundException(id));
    }
}
'''
    
    # 创建临时文件
    temp_file = Path("/tmp/UserService.java")
    temp_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file.write_text(java_code, encoding="utf-8")
    
    # 分块
    chunks = chunker.chunk_file(str(temp_file))
    
    print(f"\n文件: UserService.java")
    print(f"代码行数: {len(java_code.splitlines())}")
    print(f"分块数量: {len(chunks)}")
    print("\n分块详情:")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n  块 {i}:")
        print(f"    类型: {chunk.chunk_type.value}")
        print(f"    名称: {chunk.name}")
        print(f"    行范围: {chunk.start_line}-{chunk.end_line}")
        print(f"    内容预览: {chunk.content[:80]}...")
    
    # 清理
    temp_file.unlink()


def demo_index_config():
    """演示索引配置"""
    print("\n" + "="*60)
    print("演示 2: 索引配置 (Index Config)")
    print("="*60)
    
    config = IndexConfig(
        chunking_config=ChunkingConfig(max_chunk_lines=50),
        embedding_dimension=384,
        batch_size=100,
        max_index_size=100000,
        enable_incremental=True,
        persist_index=True,
        index_path=".pyutagent/code_index"
    )
    
    print(f"\n索引配置:")
    print(f"  Embedding 维度: {config.embedding_dimension}")
    print(f"  批处理大小: {config.batch_size}")
    print(f"  最大索引大小: {config.max_index_size}")
    print(f"  增量索引: {'启用' if config.enable_incremental else '禁用'}")
    print(f"  持久化: {'启用' if config.persist_index else '禁用'}")
    print(f"  索引路径: {config.index_path}")


async def demo_code_indexing():
    """演示代码索引"""
    print("\n" + "="*60)
    print("演示 3: 代码索引 (Code Indexing)")
    print("="*60)
    
    # 使用当前项目作为示例
    project_path = Path(__file__).parent.parent
    
    # 创建模拟的 embedding 模型和向量存储
    embedding_model = MockEmbeddingModel()
    vector_store = MockVectorStore()
    
    # 创建索引器
    config = IndexConfig(
        chunking_config=ChunkingConfig(max_chunk_lines=30),
        embedding_dimension=384,
        enable_incremental=True,
        persist_index=False,  # 演示时不持久化
    )
    
    indexer = CodeIndexer(
        project_path=str(project_path),
        vector_store=vector_store,
        embedding_model=embedding_model,
        config=config
    )
    
    print(f"\n项目路径: {project_path}")
    print(f"索引器状态:")
    stats = indexer.get_stats()
    print(f"  已索引文件: {stats['total_files']}")
    print(f"  已索引块: {stats['total_chunks']}")
    print(f"  索引大小: {stats['index_size']}")
    
    # 索引几个示例文件
    print("\n开始索引示例文件...")
    
    # 创建一些示例文件
    example_dir = Path("/tmp/demo_project")
    example_dir.mkdir(parents=True, exist_ok=True)
    
    files = [
        ("UserService.java", '''
public class UserService {
    public User createUser(String email, String name) {
        // 创建用户逻辑
        return new User(email, name);
    }
    
    public User findById(Long id) {
        // 查询用户逻辑
        return userRepository.findById(id);
    }
}
'''),
        ("OrderService.java", '''
public class OrderService {
    public Order createOrder(Long userId, List<Item> items) {
        // 创建订单逻辑
        Order order = new Order(userId);
        items.forEach(order::addItem);
        return order;
    }
    
    public void cancelOrder(Long orderId) {
        // 取消订单逻辑
        Order order = orderRepository.findById(orderId);
        order.cancel();
    }
}
'''),
        ("PaymentService.java", '''
public class PaymentService {
    public Payment processPayment(Long orderId, PaymentMethod method) {
        // 处理支付逻辑
        Payment payment = new Payment(orderId, method);
        payment.process();
        return payment;
    }
}
'''),
    ]
    
    for filename, content in files:
        file_path = example_dir / filename
        file_path.write_text(content, encoding="utf-8")
    
    # 索引这些文件
    result = await indexer.index_project(
        file_patterns=["**/*.java"],
        exclude_patterns=[],
        progress_callback=lambda p: print(f"  进度: {p['current']}/{p['total']} - {p['file']}")
    )
    
    print(f"\n索引完成:")
    print(f"  成功: {result['success']}")
    print(f"  文件数: {result['total_files']}")
    print(f"  块数: {result['total_chunks']}")
    
    if result['errors']:
        print(f"  错误: {len(result['errors'])}")
    
    # 显示更新后的状态
    stats = indexer.get_stats()
    print(f"\n更新后的索引状态:")
    print(f"  已索引文件: {stats['total_files']}")
    print(f"  已索引块: {stats['total_chunks']}")
    
    # 清理
    import shutil
    shutil.rmtree(example_dir, ignore_errors=True)


def demo_context_assembler():
    """演示上下文组装"""
    print("\n" + "="*60)
    print("演示 4: 上下文组装 (Context Assembler)")
    print("="*60)
    
    # 创建配置
    config = AssemblerConfig(
        max_tokens=2000,
        max_chunks=10,
        strategy=ContextStrategy.RELEVANCE,
        include_imports=True,
        include_signatures=True,
    )
    
    print(f"\n组装器配置:")
    print(f"  最大 Token: {config.max_tokens}")
    print(f"  最大块数: {config.max_chunks}")
    print(f"  策略: {config.strategy.value}")
    print(f"  包含导入: {config.include_imports}")
    print(f"  包含签名: {config.include_signatures}")
    
    # 创建模拟的索引器
    class MockIndexer:
        async def search(self, query: str, k: int = 10, filter_dict=None):
            # 模拟搜索结果
            return [
                {
                    "content": "public User createUser(String email, String name) { ... }",
                    "file_path": "/src/UserService.java",
                    "chunk_type": "method",
                    "name": "createUser",
                    "score": 0.95,
                },
                {
                    "content": "public class UserService { ... }",
                    "file_path": "/src/UserService.java",
                    "chunk_type": "class",
                    "name": "UserService",
                    "score": 0.90,
                },
                {
                    "content": "public User findById(Long id) { ... }",
                    "file_path": "/src/UserService.java",
                    "chunk_type": "method",
                    "name": "findById",
                    "score": 0.85,
                },
            ]
    
    assembler = ContextAssembler(code_indexer=MockIndexer(), config=config)
    
    print("\n上下文组装策略:")
    for strategy in ContextStrategy:
        print(f"  - {strategy.value}: ", end="")
        if strategy == ContextStrategy.RELEVANCE:
            print("基于相关性排序")
        elif strategy == ContextStrategy.PROXIMITY:
            print("基于代码邻近性")
        elif strategy == ContextStrategy.HIERARCHICAL:
            print("分层结构（类 -> 方法）")
        elif strategy == ContextStrategy.COMPREHENSIVE:
            print("综合模式")


def demo_token_estimation():
    """演示 Token 估算"""
    print("\n" + "="*60)
    print("演示 5: Token 估算")
    print("="*60)
    
    assembler = ContextAssembler(config=AssemblerConfig())
    
    test_contents = [
        ("短文本", "Hello world"),
        ("中等文本", "public class UserService { private UserRepository repository; }"),
        ("长文本", "public class UserService {\n" + "    private UserRepository repository;\n" * 20 + "}"),
    ]
    
    print("\nToken 估算示例:")
    for name, content in test_contents:
        tokens = assembler._estimate_tokens(content)
        words = len(content.split())
        print(f"  {name}:")
        print(f"    单词数: {words}")
        print(f"    估算 Token: {tokens}")
        print(f"    比例: {tokens/words:.2f}x")


async def demo_full_workflow():
    """演示完整工作流程"""
    print("\n" + "="*60)
    print("演示 6: 完整工作流程")
    print("="*60)
    
    print("""
完整工作流程示例:

1. 初始化索引器
   ```python
   from pyutagent.indexing import CodeIndexer, IndexConfig
   
   indexer = CodeIndexer(
       project_path="/path/to/project",
       vector_store=vector_store,
       embedding_model=embedding_model,
       config=IndexConfig()
   )
   ```

2. 构建代码索引
   ```python
   result = await indexer.index_project(
       file_patterns=["**/*.java", "**/*.py"],
       exclude_patterns=["**/test/**", "**/node_modules/**"]
   )
   ```

3. 语义搜索
   ```python
   results = await indexer.search(
       query="用户创建逻辑",
       k=10,
       filter_dict={"chunk_type": "method"}
   )
   ```

4. 组装上下文
   ```python
   from pyutagent.indexing import ContextAssembler, AssemblerConfig
   
   assembler = ContextAssembler(
       code_indexer=indexer,
       config=AssemblerConfig(
           max_tokens=8000,
           strategy=ContextStrategy.RELEVANCE
       )
   )
   
   context = await assembler.assemble_context(
       query="如何创建用户",
       target_file="/src/UserService.java"
   )
   ```

5. 在 Agent 中使用
   ```python
   from pyutagent.agent import UniversalCodingAgent
   
   agent = UniversalCodingAgent()
   agent.enable_code_indexing(indexer)
   
   # Agent 会自动使用索引进行上下文选择
   result = await agent.execute("为用户服务添加邮箱验证")
   ```
""")


async def main():
    """主函数"""
    print("\n" + "="*60)
    print("智能上下文管理演示")
    print("代码索引 + 语义搜索 + 智能上下文组装")
    print("="*60)
    
    demo_code_chunking()
    demo_index_config()
    await demo_code_indexing()
    demo_context_assembler()
    demo_token_estimation()
    await demo_full_workflow()
    
    print("\n" + "="*60)
    print("智能上下文管理演示完成!")
    print("="*60)
    print("""
核心组件:
1. CodeChunker - 智能代码分块
2. CodeIndexer - 代码索引构建
3. ContextAssembler - 上下文智能组装

支持策略:
- RELEVANCE: 基于语义相关性
- PROXIMITY: 基于代码邻近性  
- HIERARCHICAL: 分层结构
- COMPREHENSIVE: 综合模式
""")


if __name__ == "__main__":
    asyncio.run(main())
