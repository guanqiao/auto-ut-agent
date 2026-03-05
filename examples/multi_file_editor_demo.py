"""多文件编辑器演示脚本

演示多文件编辑、依赖分析和影响分析功能。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from pathlib import Path
from typing import Dict, Any

from pyutagent.tools import (
    DependencyAnalyzer,
    MultiFileEditor,
    MultiFileEditResult,
    FileNode,
)


class MockLLMClient:
    """模拟 LLM 客户端"""
    
    async def complete(self, messages, temperature=0.1):
        """模拟 LLM 完成"""
        # 返回简单的编辑响应
        return """<<<<<<< SEARCH
    public User createUser(String email, String name) {
=======
    public User createUser(String email, String name, String phone) {
>>>>>>> REPLACE
"""


def create_demo_project(base_path: Path):
    """创建演示项目文件"""
    # User.java
    user_java = '''
package com.example.model;

public class User {
    private Long id;
    private String email;
    private String name;
    
    public User(String email, String name) {
        this.email = email;
        this.name = name;
    }
    
    // Getters and setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getEmail() { return email; }
    public void setEmail(String email) { this.email = email; }
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
}
'''
    
    # UserRepository.java
    user_repo_java = '''
package com.example.repository;

import com.example.model.User;
import java.util.*;

public class UserRepository {
    private Map<Long, User> users = new HashMap<>();
    private Long nextId = 1L;
    
    public User save(User user) {
        if (user.getId() == null) {
            user.setId(nextId++);
        }
        users.put(user.getId(), user);
        return user;
    }
    
    public Optional<User> findById(Long id) {
        return Optional.ofNullable(users.get(id));
    }
    
    public List<User> findAll() {
        return new ArrayList<>(users.values());
    }
}
'''
    
    # UserService.java
    user_service_java = '''
package com.example.service;

import com.example.model.User;
import com.example.repository.UserRepository;

public class UserService {
    private UserRepository userRepository;
    
    public UserService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }
    
    public User createUser(String email, String name) {
        User user = new User(email, name);
        return userRepository.save(user);
    }
    
    public User findById(Long id) {
        return userRepository.findById(id)
            .orElseThrow(() -> new RuntimeException("User not found"));
    }
}
'''
    
    # Order.java
    order_java = '''
package com.example.model;

import java.util.List;

public class Order {
    private Long id;
    private Long userId;
    private List<String> items;
    private double totalAmount;
    
    public Order(Long userId, List<String> items) {
        this.userId = userId;
        this.items = items;
        this.totalAmount = calculateTotal();
    }
    
    private double calculateTotal() {
        return items.size() * 10.0; // Simplified
    }
    
    // Getters
    public Long getId() { return id; }
    public Long getUserId() { return userId; }
    public List<String> getItems() { return items; }
    public double getTotalAmount() { return totalAmount; }
}
'''
    
    # OrderService.java
    order_service_java = '''
package com.example.service;

import com.example.model.Order;
import com.example.model.User;
import java.util.List;

public class OrderService {
    private UserService userService;
    
    public OrderService(UserService userService) {
        this.userService = userService;
    }
    
    public Order createOrder(Long userId, List<String> items) {
        User user = userService.findById(userId);
        Order order = new Order(userId, items);
        // Save order logic
        return order;
    }
}
'''
    
    # 创建目录结构
    (base_path / "com" / "example" / "model").mkdir(parents=True, exist_ok=True)
    (base_path / "com" / "example" / "repository").mkdir(parents=True, exist_ok=True)
    (base_path / "com" / "example" / "service").mkdir(parents=True, exist_ok=True)
    
    # 写入文件
    (base_path / "com" / "example" / "model" / "User.java").write_text(user_java, encoding="utf-8")
    (base_path / "com" / "example" / "repository" / "UserRepository.java").write_text(user_repo_java, encoding="utf-8")
    (base_path / "com" / "example" / "service" / "UserService.java").write_text(user_service_java, encoding="utf-8")
    (base_path / "com" / "example" / "model" / "Order.java").write_text(order_java, encoding="utf-8")
    (base_path / "com" / "example" / "service" / "OrderService.java").write_text(order_service_java, encoding="utf-8")
    
    return [
        str(base_path / "com" / "example" / "model" / "User.java"),
        str(base_path / "com" / "example" / "repository" / "UserRepository.java"),
        str(base_path / "com" / "example" / "service" / "UserService.java"),
        str(base_path / "com" / "example" / "model" / "Order.java"),
        str(base_path / "com" / "example" / "service" / "OrderService.java"),
    ]


def demo_dependency_analysis():
    """演示依赖分析"""
    print("\n" + "="*60)
    print("演示 1: 依赖分析 (Dependency Analysis)")
    print("="*60)
    
    # 创建临时项目
    temp_dir = Path("/tmp/demo_project")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        file_paths = create_demo_project(temp_dir)
        
        # 创建依赖分析器
        analyzer = DependencyAnalyzer(str(temp_dir))
        
        # 分析文件
        nodes = analyzer.analyze_files(file_paths)
        
        print(f"\n分析了 {len(nodes)} 个文件:")
        for path, node in nodes.items():
            filename = Path(path).name
            deps_count = len(node.dependencies)
            dependents_count = len(node.dependents)
            print(f"  📄 {filename}")
            print(f"     依赖: {deps_count} 个文件")
            print(f"     被依赖: {dependents_count} 个文件")
            if node.dependencies:
                print(f"     依赖列表: {', '.join([Path(d).name for d in node.dependencies])}")
        
        # 拓扑排序
        sorted_files = analyzer.topological_sort()
        print(f"\n拓扑排序结果 (依赖优先):")
        for i, path in enumerate(sorted_files, 1):
            print(f"  {i}. {Path(path).name}")
        
        # 获取相关文件
        user_service_path = str(temp_dir / "com" / "example" / "service" / "UserService.java")
        if user_service_path in nodes:
            related = analyzer.get_related_files(user_service_path, depth=2)
            print(f"\n与 UserService.java 相关的文件 (深度=2):")
            for path in related:
                print(f"  - {Path(path).name}")
        
    finally:
        # 清理
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def demo_edit_plan():
    """演示编辑计划"""
    print("\n" + "="*60)
    print("演示 2: 编辑计划 (Edit Plan)")
    print("="*60)
    
    temp_dir = Path("/tmp/demo_project")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        file_paths = create_demo_project(temp_dir)
        
        # 创建多文件编辑器
        llm_client = MockLLMClient()
        editor = MultiFileEditor(llm_client, str(temp_dir))
        
        # 获取编辑计划
        plan = editor.get_edit_plan(file_paths)
        
        print(f"\n编辑计划:")
        print(f"  总文件数: {plan['total_files']}")
        print(f"\n  编辑顺序 (考虑依赖):")
        for i, path in enumerate(plan['edit_order'], 1):
            filename = Path(path).name
            deps = plan['dependencies'].get(path, [])
            deps_str = f" (依赖: {', '.join([Path(d).name for d in deps])})" if deps else ""
            print(f"    {i}. {filename}{deps_str}")
        
        print(f"\n  依赖关系图:")
        for path, deps in plan['dependencies'].items():
            if deps:
                filename = Path(path).name
                dep_names = [Path(d).name for d in deps]
                print(f"    {filename} -> {', '.join(dep_names)}")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def demo_file_node():
    """演示文件节点"""
    print("\n" + "="*60)
    print("演示 3: 文件节点 (File Node)")
    print("="*60)
    
    # 创建文件节点
    node = FileNode(
        path="/src/UserService.java",
        content="public class UserService { ... }",
        dependencies={"/src/UserRepository.java", "/src/User.java"},
        dependents={"/src/OrderService.java"}
    )
    
    print(f"\n文件节点信息:")
    print(f"  路径: {node.path}")
    print(f"  内容长度: {len(node.content)} 字符")
    print(f"  依赖数: {len(node.dependencies)}")
    print(f"  被依赖数: {len(node.dependents)}")
    print(f"  是否有编辑内容: {node.edit_content is not None}")


def demo_multi_file_edit_result():
    """演示编辑结果"""
    print("\n" + "="*60)
    print("演示 4: 多文件编辑结果 (MultiFileEditResult)")
    print("="*60)
    
    # 成功的编辑
    success_result = MultiFileEditResult(
        success=True,
        edited_files=[
            "/src/User.java",
            "/src/UserService.java"
        ],
        failed_files=[],
        rollback_applied=False
    )
    
    print(f"\n成功的编辑:")
    print(f"  成功: {success_result.success}")
    print(f"  编辑文件数: {len(success_result.edited_files)}")
    print(f"  失败文件数: {len(success_result.failed_files)}")
    print(f"  回滚: {success_result.rollback_applied}")
    
    # 失败的编辑
    failed_result = MultiFileEditResult(
        success=False,
        edited_files=["/src/User.java"],
        failed_files=[("/src/UserService.java", "编译错误: 缺少分号")],
        rollback_applied=True,
        error_message="编辑 UserService.java 失败"
    )
    
    print(f"\n失败的编辑:")
    print(f"  成功: {failed_result.success}")
    print(f"  编辑文件数: {len(failed_result.edited_files)}")
    print(f"  失败文件数: {len(failed_result.failed_files)}")
    print(f"  回滚: {failed_result.rollback_applied}")
    print(f"  错误信息: {failed_result.error_message}")


def demo_integration_with_agent():
    """演示与 Agent 集成"""
    print("\n" + "="*60)
    print("演示 5: 与 Agent 集成")
    print("="*60)
    
    print("""
多文件编辑器与 Agent 集成:

1. 在 UniversalTaskPlanner 中使用
   ```python
   from pyutagent.agent import UniversalTaskPlanner
   from pyutagent.tools import MultiFileEditor
   
   planner = UniversalTaskPlanner(...)
   
   # 注册多文件编辑处理器
   async def handle_refactoring(subtask):
       editor = MultiFileEditor(llm_client, project_path)
       
       # 分析影响范围
       plan = editor.get_edit_plan(target_files)
       
       # 执行编辑
       result = await editor.edit_files(edits_by_file)
       
       return result
   
   planner.register_task_handler(
       TaskType.CODE_REFACTORING, 
       handle_refactoring
   )
   ```

2. 依赖感知编辑
   ```python
   # 编辑器会自动:
   # - 分析文件间依赖
   # - 按拓扑排序执行
   # - 支持并发编辑
   # - 失败时自动回滚
   
   edits = {
       "User.java": "添加 phone 字段",
       "UserService.java": "更新 createUser 方法",
       "OrderService.java": "更新订单创建逻辑"
   }
   
   result = await editor.edit_files(edits)
   ```

3. 与代码索引集成
   ```python
   from pyutagent.indexing import CodeIndexer
   
   # 使用索引分析影响范围
   indexer = CodeIndexer(project_path, ...)
   await indexer.index_project()
   
   # 找到所有受影响文件
   affected_files = await indexer.find_references("User.createUser")
   
   # 批量编辑
   plan = editor.get_edit_plan(affected_files)
   ```
""")


async def main():
    """主函数"""
    print("\n" + "="*60)
    print("多文件编辑器演示")
    print("依赖分析 + 批量编辑 + 影响分析")
    print("="*60)
    
    demo_dependency_analysis()
    demo_edit_plan()
    demo_file_node()
    demo_multi_file_edit_result()
    demo_integration_with_agent()
    
    print("\n" + "="*60)
    print("多文件编辑器演示完成!")
    print("="*60)
    print("""
核心功能:
1. DependencyAnalyzer - 分析文件间依赖关系
2. MultiFileEditor - 批量编辑多个文件
3. 拓扑排序 - 确保按依赖顺序编辑
4. 自动回滚 - 失败时恢复原始状态
5. 并发编辑 - 提高编辑效率

使用场景:
- 重构操作 (重命名、提取接口等)
- 批量添加字段/方法
- 更新 API 签名
- 代码迁移
""")


if __name__ == "__main__":
    asyncio.run(main())
