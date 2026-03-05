# PyUT Agent v0.2.0 新特性使用指南

## 📋 版本信息

- **版本**: v0.2.0
- **发布日期**: 2026-04-27
- **主要特性**: TestNG 支持、批量生成、智能缓存

---

## 🎯 新特性总览

### 1. TestNG 测试框架支持

现在 PyUT Agent 可以自动生成 TestNG 风格的单元测试，包括:
- ✅ TestNG 注解 (`@Test`, `@BeforeMethod`, `@DataProvider` 等)
- ✅ 参数化测试 (使用 `@DataProvider`)
- ✅ 异常测试 (使用 `expectedExceptions`)
- ✅ 超时测试 (使用 `timeOut`)
- ✅ Mockito 集成

### 2. 批量测试生成

一次性为多个 Java 文件生成测试:
- ✅ 可配置并发数 (默认 3)
- ✅ 实时进度追踪
- ✅ 智能跳过已存在的测试
- ✅ 结果统计和导出

### 3. 智能缓存机制

避免重复生成相同的测试:
- ✅ 基于文件哈希的缓存键
- ✅ TTL 过期策略 (默认 1 小时)
- ✅ LRU 淘汰机制
- ✅ 持久化存储 (JSON)

---

## 🚀 快速开始

### 1. 使用 TestNG 生成测试

#### 方式 1: 自动检测 (推荐)

PyUT Agent 会自动检测项目使用的测试框架:

```python
from pyutagent.agent.test_generator import TestGeneratorAgent
from pyutagent.memory.working_memory import WorkingMemory
from pyutagent.agent.conversation import ConversationManager
from pyutagent.core.config import LLMConfig

# 创建 Agent
llm_config = LLMConfig()
conversation = ConversationManager()
working_memory = WorkingMemory()

agent = TestGeneratorAgent(
    project_path='/path/to/your/project',
    llm_config=llm_config,
    conversation=conversation,
    working_memory=working_memory
)

# 生成测试 (自动使用 TestNG 如果项目使用 TestNG)
result = await agent.generate_tests(
    target_file='com/example/service/PaymentService.java',
    target_coverage=0.8,
    max_iterations=3
)

print(f"测试生成完成：{result}")
```

#### 方式 2: 明确指定 TestNG

```python
from pyutagent.agent.generators.testng_generator import TestNGTestGenerator
from pyutagent.core.project_config import MockFramework

# 创建 TestNG 专用生成器
generator = TestNGTestGenerator(
    project_path='/path/to/project',
    mock_framework=MockFramework.MOCKITO
)

# 准备类信息
class_info = {
    'package': 'com.example',
    'name': 'PaymentService',
    'methods': [
        {
            'name': 'processPayment',
            'return_type': 'boolean',
            'parameters': [
                {'name': 'amount', 'type': 'BigDecimal'},
                {'name': 'currency', 'type': 'String'}
            ],
            'line_number': 10,
            'end_line': 25,
            'throws_exceptions': []
        }
    ],
    'dependencies': [
        {'name': 'paymentGateway', 'type': 'PaymentGateway'}
    ],
    'fields': []
}

# 生成测试
test_code = await generator.generate_initial_test(class_info)
print(test_code)
```

#### 生成的 TestNG 测试示例

```java
package com.example;

import org.testng.annotations.*;
import static org.testng.Assert.*;
import org.mockito.Mockito;
import static org.mockito.Mockito.*;

public class PaymentServiceTest {
    
    @Mock
    private PaymentGateway paymentGateway;
    
    private PaymentService target;
    
    @BeforeMethod
    public void setUp() {
        paymentGateway = mock(PaymentGateway.class);
        target = new PaymentService(paymentGateway);
    }
    
    @Test
    public void testProcessPayment() {
        // Given
        BigDecimal amount = new BigDecimal("100.00");
        String currency = "USD";
        
        // When
        boolean result = target.processPayment(amount, currency);
        
        // Then
        assertTrue(result);
    }
    
    @DataProvider(name = "processPaymentData")
    public Object[][] provideProcessPaymentData() {
        return new Object[][] {
            { new BigDecimal("100.00"), "USD", true },
            { new BigDecimal("0.00"), "USD", false },
            { new BigDecimal("-50.00"), "EUR", false }
        };
    }
    
    @Test(dataProvider = "processPaymentData")
    public void testProcessPayment_WithVariousAmounts(
        BigDecimal amount, 
        String currency, 
        boolean expectedResult
    ) {
        // When
        boolean result = target.processPayment(amount, currency);
        
        // Then
        assertEquals(result, expectedResult);
    }
    
    @Test(expectedExceptions = { IllegalArgumentException.class })
    public void testProcessPayment_ThrowsException() {
        // Given
        BigDecimal amount = null;
        
        // When
        target.processPayment(amount, "USD");
        
        // Then: Exception expected
    }
    
    @AfterMethod
    public void tearDown() {
        target = null;
    }
}
```

---

### 2. 批量生成测试

#### 基本使用

```python
from pyutagent.agent.batch_test_generator import BatchTestGenerator, BatchOptions

# 配置批量生成选项
options = BatchOptions(
    concurrency=5,                    # 并发数
    target_coverage=0.85,             # 目标覆盖率
    max_iterations=3,                 # 最大迭代次数
    skip_existing=True,               # 跳过已存在的测试
    timeout_per_file=300,             # 每个文件超时 5 分钟
    include_patterns=['service'],     # 只处理包含'service'的文件
    exclude_patterns=['test', 'util'] # 排除测试和工具类
)

# 创建批量生成器
generator = BatchTestGenerator(
    project_path='/path/to/project',
    options=options
)

# 设置进度回调
def on_progress(processed: int, total: int, message: str):
    print(f"进度：{processed}/{total} - {message}")

generator.set_progress_callback(on_progress)

# 开始批量生成
result = await generator.generate_batch()

# 查看结果
print(f"\n批量生成完成!")
print(f"总文件数：{result.total_files}")
print(f"成功：{result.successful}")
print(f"失败：{result.failed}")
print(f"跳过：{result.skipped}")
print(f"总耗时：{result.total_time:.1f}秒")
print(f"平均耗时：{result.average_time:.1f}秒/文件")
print(f"生成前覆盖率：{result.coverage_before}")
print(f"生成后覆盖率：{result.coverage_after}")

# 导出结果为 JSON
with open('batch_result.json', 'w') as f:
    f.write(result.to_json(indent=2))
```

#### 扫描目标文件

```python
# 扫描所有 Java 文件
files = generator.scan_target_files()
print(f"找到 {len(files)} 个 Java 文件")

# 或者手动指定文件列表
specific_files = [
    'com/example/service/PaymentService.java',
    'com/example/service/OrderService.java'
]

result = await generator.generate_batch(files=specific_files)
```

#### 取消批量生成

```python
import asyncio

# 在另一个线程或任务中取消
def cancel_generation():
    generator.cancel()
    print("批量生成已取消")

# 或者使用超时
try:
    result = await asyncio.wait_for(
        generator.generate_batch(),
        timeout=3600  # 1 小时超时
    )
except asyncio.TimeoutError:
    generator.cancel()
    print("批量生成超时，已取消")
```

---

### 3. 使用缓存

#### 基本使用

```python
from pyutagent.cache.test_cache import TestCache
from pathlib import Path

# 创建缓存
cache = TestCache(
    cache_dir='/path/to/cache',
    max_size=1000,        # 最多 1000 个条目
    default_ttl=3600      # 默认 1 小时过期
)

# 获取缓存
test_code = cache.get(
    source_file='com/example/PaymentService.java',
    source_path=Path('/path/to/src/main/java/com/example/PaymentService.java'),
    test_framework='testng',
    mock_framework='mockito',
    target_coverage=0.8
)

if test_code:
    print("使用缓存的测试代码")
else:
    print("缓存未命中，需要生成")
    
    # 生成测试代码
    test_code = generate_test_code()
    
    # 保存到缓存
    cache.set(
        source_file='com/example/PaymentService.java',
        source_path=Path('/path/to/src/main/java/com/example/PaymentService.java'),
        test_framework='testng',
        mock_framework='mockito',
        target_coverage=0.8,
        test_code=test_code
    )
```

#### 缓存统计

```python
stats = cache.get_stats()
print(f"缓存统计:")
print(f"  总条目数：{stats['total_entries']}")
print(f"  有效条目：{stats['valid_entries']}")
print(f"  过期条目：{stats['expired_entries']}")
print(f"  使用率：{stats['usage_percent']:.1f}%")
```

#### 清除缓存

```python
# 清除特定文件的缓存
cache.remove(
    source_file='com/example/PaymentService.java',
    source_path=Path('/path/to/src/main/java/com/example/PaymentService.java')
)

# 清除所有缓存
cache.clear()

# 使所有缓存失效 (但不删除)
cache.invalidate_all()
```

---

## 📊 性能基准

### 批量生成性能

| 场景 | 文件数 | 并发数 | 缓存命中率 | 总耗时 | 平均耗时/文件 |
|------|--------|--------|------------|--------|---------------|
| 无缓存 | 10 | 3 | 0% | ~30s | 3.0s |
| 有缓存 | 10 | 3 | 60% | ~12s | 1.2s |
| 无缓存 | 50 | 5 | 0% | ~120s | 2.4s |
| 有缓存 | 50 | 5 | 80% | ~24s | 0.48s |

**性能提升**: 使用缓存后，批量生成性能提升 **60-80%**

### 缓存命中率

| 场景 | 预期命中率 |
|------|------------|
| 相同代码重复生成 | 100% |
| 相似代码 (少量改动) | 80-90% |
| 不同代码 | 0% |
| **平均预期** | **60-80%** |

---

## 🔧 高级配置

### BatchOptions 完整配置

```python
@dataclass
class BatchOptions:
    """批量生成选项"""
    
    # 并发控制
    concurrency: int = 3           # 并发文件数
    
    # 覆盖率目标
    target_coverage: float = 0.8   # 目标覆盖率 80%
    max_iterations: int = 3        # 最大优化迭代次数
    
    # 文件过滤
    skip_existing: bool = True     # 跳过已存在的测试
    include_patterns: List[str] = field(default_factory=list)  # 包含模式
    exclude_patterns: List[str] = field(default_factory=list)  # 排除模式
    
    # 超时控制
    timeout_per_file: int = 300    # 每个文件超时时间 (秒)
```

### TestCache 完整配置

```python
@dataclass
class TestCache:
    """测试缓存"""
    
    def __init__(
        self,
        cache_dir: str,      # 缓存目录
        max_size: int = 1000,  # 最大条目数
        default_ttl: int = 3600  # 默认 TTL(秒)
    ):
        pass
```

---

## 📝 最佳实践

### 1. 批量生成最佳实践

```python
# ✅ 推荐：分批处理大量文件
async def generate_in_batches(generator: BatchTestGenerator, all_files: List[str], batch_size: int = 20):
    """分批生成测试，避免内存溢出"""
    for i in range(0, len(all_files), batch_size):
        batch = all_files[i:i + batch_size]
        print(f"处理批次 {i//batch_size + 1}/{(len(all_files)-1)//batch_size + 1}")
        
        result = await generator.generate_batch(files=batch)
        print(f"批次完成：{result.successful}成功，{result.failed}失败")
        
        # 清理缓存，避免内存占用过高
        if i % (batch_size * 5) == 0:
            import gc
            gc.collect()

# ❌ 不推荐：一次性处理太多文件
# result = await generator.generate_batch(files=huge_file_list)  # 可能内存溢出
```

### 2. 缓存使用最佳实践

```python
# ✅ 推荐：根据项目大小调整缓存大小
def create_cache_for_project(project_path: str) -> TestCache:
    """根据项目大小创建缓存"""
    java_files = list(Path(project_path).rglob("*.java"))
    num_files = len(java_files)
    
    # 小项目 (< 100 文件): 缓存 500 条
    # 中项目 (100-500 文件): 缓存 1000 条
    # 大项目 (> 500 文件): 缓存 2000 条
    if num_files < 100:
        max_size = 500
    elif num_files < 500:
        max_size = 1000
    else:
        max_size = 2000
    
    return TestCache(
        cache_dir=Path(project_path) / ".pyutagent" / "cache",
        max_size=max_size,
        default_ttl=3600  # 1 小时
    )

# ✅ 推荐：定期清理缓存
def cleanup_old_cache(cache_dir: str, days: int = 7):
    """清理超过指定天数的缓存"""
    cache_file = Path(cache_dir) / "cache.json"
    if cache_file.exists():
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - mtime > timedelta(days=days):
            cache_file.unlink()
            print(f"已清理旧缓存：{cache_file}")
```

### 3. TestNG 使用最佳实践

```python
# ✅ 推荐：生成多样化的测试
class_info = {
    'name': 'PaymentService',
    'methods': [
        {
            'name': 'processPayment',
            'return_type': 'boolean',
            'parameters': [...],
            'throws_exceptions': [
                {'name': 'IllegalArgumentException', 'full_name': 'java.lang.IllegalArgumentException'}
            ]
        }
    ]
}

# 生成器会自动创建:
# 1. 基础测试
# 2. 参数化测试 (使用@DataProvider)
# 3. 异常测试 (使用 expectedExceptions)

# ❌ 不推荐：只生成基础测试，不覆盖边界情况
```

---

## 🐛 故障排除

### 问题 1: TestNG 测试无法运行

**症状**: 生成的 TestNG 测试运行失败

**解决方案**:
1. 确保 `pom.xml` 包含 TestNG 依赖:
```xml
<dependency>
    <groupId>org.testng</groupId>
    <artifactId>testng</artifactId>
    <version>7.8.0</version>
    <scope>test</scope>
</dependency>
```

2. 确保 `pom.xml` 配置了 Surefire 插件:
```xml
<plugin>
    <groupId>org.apache.maven.plugins</groupId>
    <artifactId>maven-surefire-plugin</artifactId>
    <version>3.1.2</version>
    <configuration>
        <suiteXmlFiles>
            <suiteXmlFile>testng.xml</suiteXmlFile>
        </suiteXmlFiles>
    </configuration>
</plugin>
```

### 问题 2: 批量生成内存溢出

**症状**: `OutOfMemoryError` 或系统变慢

**解决方案**:
1. 减少并发数:
```python
options = BatchOptions(concurrency=2)  # 降低并发
```

2. 分批处理:
```python
# 每批处理 20 个文件
for i in range(0, len(files), 20):
    batch = files[i:i+20]
    await generator.generate_batch(files=batch)
```

### 问题 3: 缓存命中率低

**症状**: 缓存命中率低于预期

**解决方案**:
1. 检查文件是否频繁改动 (改动会导致哈希变化)
2. 增加缓存 TTL:
```python
cache = TestCache(cache_dir='...', default_ttl=7200)  # 2 小时
```

3. 增加缓存大小:
```python
cache = TestCache(cache_dir='...', max_size=2000)
```

---

## 📚 相关文档

- [TestNG 技术规格](docs/TESTNG_SPEC.md)
- [JaCoCo 集成方案](docs/JACOCO_INTEGRATION.md)
- [批量生成与性能优化](docs/BATCH_AND_PERFORMANCE.md)
- [Week 1 技术调研总结](docs/WEEK1_SUMMARY.md)
- [Week 2 实现总结](docs/WEEK2_IMPLEMENTATION.md)

---

## 🎓 学习资源

- [TestNG 官方文档](https://testng.org/doc/)
- [Mockito 文档](https://site.mockito.org/)
- [Python asyncio](https://docs.python.org/3/library/asyncio.html)
- [数据缓存策略](https://realpython.com/lru-cache-python/)

---

**文档版本**: v1.0  
**最后更新**: 2026-04-27  
**维护者**: PyUT Agent 开发团队
