"""项目配置系统演示脚本

演示 PYUT.md 配置生成、项目自动检测和编码规范定制。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from typing import Dict, Any

from pyutagent.core import (
    BuildTool,
    TestFramework,
    MockFramework,
    BuildConfig,
    TestConfig,
    AgentPreferences,
    CodeStyle,
    DependencyInfo,
    ProjectContext,
    ProjectConfig,
    ProjectConfigLoader,
    load_project_config,
    create_config_template,
)


def demo_build_config():
    """演示构建配置"""
    print("\n" + "="*60)
    print("演示 1: 构建配置 (Build Config)")
    print("="*60)
    
    # Maven 配置
    maven_config = BuildConfig(
        tool=BuildTool.MAVEN,
        java_version="17",
        build_command="mvn clean install -DskipTests",
        test_command="mvn test",
        compile_command="mvn compile"
    )
    
    print(f"\nMaven 配置:")
    print(f"  工具: {maven_config.tool.value}")
    print(f"  Java 版本: {maven_config.java_version}")
    print(f"  构建命令: {maven_config.build_command}")
    print(f"  测试命令: {maven_config.test_command}")
    print(f"  编译命令: {maven_config.compile_command}")
    
    # Gradle 配置
    gradle_config = BuildConfig(
        tool=BuildTool.GRADLE,
        java_version="21",
        build_command="./gradlew build -x test",
        test_command="./gradlew test",
        compile_command="./gradlew compileJava"
    )
    
    print(f"\nGradle 配置:")
    print(f"  工具: {gradle_config.tool.value}")
    print(f"  Java 版本: {gradle_config.java_version}")
    print(f"  构建命令: {gradle_config.build_command}")


def demo_test_config():
    """演示测试配置"""
    print("\n" + "="*60)
    print("演示 2: 测试配置 (Test Config)")
    print("="*60)
    
    # JUnit 5 配置
    junit5_config = TestConfig(
        framework=TestFramework.JUNIT5,
        target_coverage=0.8,
        test_directory="src/test/java",
        test_suffix="Test",
        generate_integration_tests=False,
        mock_framework="mockito"
    )
    
    print(f"\nJUnit 5 配置:")
    print(f"  框架: {junit5_config.framework.value}")
    print(f"  目标覆盖率: {junit5_config.target_coverage:.0%}")
    print(f"  测试目录: {junit5_config.test_directory}")
    print(f"  测试后缀: {junit5_config.test_suffix}")
    print(f"  Mock 框架: {junit5_config.mock_framework}")
    print(f"  生成集成测试: {junit5_config.generate_integration_tests}")
    
    # TestNG 配置
    testng_config = TestConfig(
        framework=TestFramework.TESTNG,
        target_coverage=0.85,
        mock_framework="mockito"
    )
    
    print(f"\nTestNG 配置:")
    print(f"  框架: {testng_config.framework.value}")
    print(f"  目标覆盖率: {testng_config.target_coverage:.0%}")


def demo_agent_preferences():
    """演示 Agent 偏好设置"""
    print("\n" + "="*60)
    print("演示 3: Agent 偏好设置 (Agent Preferences)")
    print("="*60)
    
    preferences = AgentPreferences(
        enable_multi_agent=True,
        enable_error_prediction=True,
        enable_self_reflection=True,
        enable_pattern_library=True,
        enable_chain_of_thought=True,
        max_iterations=10,
        preferred_strategies=["boundary", "positive", "mutation"],
        timeout_per_file=300
    )
    
    print(f"\nAgent 偏好:")
    print(f"  启用多 Agent: {preferences.enable_multi_agent}")
    print(f"  启用错误预测: {preferences.enable_error_prediction}")
    print(f"  启用自我反思: {preferences.enable_self_reflection}")
    print(f"  启用模式库: {preferences.enable_pattern_library}")
    print(f"  启用思维链: {preferences.enable_chain_of_thought}")
    print(f"  最大迭代次数: {preferences.max_iterations}")
    print(f"  首选策略: {', '.join(preferences.preferred_strategies)}")
    print(f"  每文件超时: {preferences.timeout_per_file}秒")


def demo_code_style():
    """演示编码规范"""
    print("\n" + "="*60)
    print("演示 4: 编码规范 (Code Style)")
    print("="*60)
    
    style = CodeStyle(
        naming_convention="camelCase",
        max_line_length=120,
        indent_size=4,
        use_lombok=True,
        generate_javadoc=True
    )
    
    print(f"\n编码规范:")
    print(f"  命名规范: {style.naming_convention}")
    print(f"  最大行长度: {style.max_line_length}")
    print(f"  缩进大小: {style.indent_size}")
    print(f"  使用 Lombok: {style.use_lombok}")
    print(f"  生成 Javadoc: {style.generate_javadoc}")


def demo_project_context():
    """演示项目上下文"""
    print("\n" + "="*60)
    print("演示 5: 项目上下文 (Project Context)")
    print("="*60)
    
    context = ProjectContext(
        project_name="MyProject",
        project_path=Path("/path/to/project"),
        build_config=BuildConfig(
            tool=BuildTool.MAVEN,
            java_version="17"
        ),
        test_config=TestConfig(
            framework=TestFramework.JUNIT5,
            mock_framework="mockito"
        )
    )
    
    print(f"\n项目上下文:")
    print(f"  项目名称: {context.project_name}")
    print(f"  项目路径: {context.project_path}")
    print(f"  构建工具: {context.build_config.tool.value}")
    print(f"  Java 版本: {context.build_config.java_version}")
    print(f"  测试框架: {context.test_config.framework.value}")
    print(f"  Mock 框架: {context.test_config.mock_framework}")


def demo_project_config():
    """演示完整项目配置"""
    print("\n" + "="*60)
    print("演示 6: 完整项目配置 (Project Config)")
    print("="*60)
    
    config = ProjectConfig(
        project_name="ECommerce System",
        project_root=Path("/projects/ecommerce"),
        build=BuildConfig(
            tool=BuildTool.MAVEN,
            java_version="17",
            build_command="mvn clean package -DskipTests",
            test_command="mvn test",
            compile_command="mvn compile"
        ),
        testing=TestConfig(
            framework=TestFramework.JUNIT5,
            target_coverage=0.85,
            test_directory="src/test/java",
            test_suffix="Test",
            generate_integration_tests=True,
            mock_framework="mockito"
        ),
        agent=AgentPreferences(
            enable_multi_agent=True,
            enable_error_prediction=True,
            max_iterations=15,
            preferred_strategies=["boundary", "mutation", "exception"],
            timeout_per_file=600
        ),
        code_style=CodeStyle(
            naming_convention="camelCase",
            max_line_length=120,
            indent_size=4
        ),
        dependencies={
            "Spring Boot": DependencyInfo(name="Spring Boot", version="3.2.0", enabled=True),
            "Lombok": DependencyInfo(name="Lombok", version="1.18.30", enabled=True),
            "MapStruct": DependencyInfo(name="MapStruct", version="1.5.5", enabled=False),
        },
        custom_instructions=[
            "优先使用构造函数注入",
            "所有服务类必须添加 @Service 注解",
            "使用 Optional 处理可能为空的返回值"
        ],
        ignore_patterns=[
            "**/generated/**",
            "**/target/**",
            "**/*.min.js"
        ]
    )
    
    print(f"\n项目配置:")
    print(f"  项目名称: {config.project_name}")
    print(f"  项目路径: {config.project_root}")
    
    print(f"\n  构建配置:")
    print(f"    工具: {config.build.tool.value}")
    print(f"    Java 版本: {config.build.java_version}")
    
    print(f"\n  测试配置:")
    print(f"    框架: {config.testing.framework.value}")
    print(f"    目标覆盖率: {config.testing.target_coverage:.0%}")
    
    print(f"\n  Agent 配置:")
    print(f"    多 Agent: {config.agent.enable_multi_agent}")
    print(f"    最大迭代: {config.agent.max_iterations}")
    
    print(f"\n  依赖项:")
    for name, dep in config.dependencies.items():
        status = "✅" if dep.enabled else "⏸️"
        print(f"    {status} {dep.name}: {dep.version}")
    
    print(f"\n  自定义指令:")
    for instruction in config.custom_instructions:
        print(f"    • {instruction}")
    
    # 转换为上下文
    context = ProjectContext.from_project_config(config)
    print(f"\n  转换为上下文:")
    print(f"    项目名称: {context.project_name}")
    print(f"    构建工具: {context.build_config.tool.value}")


def demo_config_loader():
    """演示配置加载器"""
    print("\n" + "="*60)
    print("演示 7: 配置加载器 (Config Loader)")
    print("="*60)
    
    loader = ProjectConfigLoader()
    
    print(f"\n配置加载器功能:")
    print(f"  配置文件名:")
    for filename in loader.CONFIG_FILENAMES:
        print(f"    - {filename}")
    
    print(f"\n  支持的格式:")
    print(f"    - PYUT.md (Markdown)")
    print(f"    - .pyut.md (隐藏文件)")


def demo_create_config_template():
    """演示创建配置模板"""
    print("\n" + "="*60)
    print("演示 8: 创建配置模板")
    print("="*60)
    
    # 创建临时文件
    temp_file = Path("/tmp/PYUT.md")
    
    try:
        create_config_template(temp_file)
        
        print(f"\n已创建配置模板: {temp_file}")
        print(f"\n模板内容预览:")
        content = temp_file.read_text(encoding="utf-8")
        lines = content.splitlines()[:30]
        for line in lines:
            print(f"  {line}")
        if len(content.splitlines()) > 30:
            print(f"  ... (共 {len(content.splitlines())} 行)")
        
        # 清理
        temp_file.unlink()
        
    except Exception as e:
        print(f"创建模板失败: {e}")


def demo_config_to_dict():
    """演示配置转换为字典"""
    print("\n" + "="*60)
    print("演示 9: 配置转换为字典")
    print("="*60)
    
    config = ProjectConfig(
        project_name="Demo Project",
        build=BuildConfig(tool=BuildTool.MAVEN),
        testing=TestConfig(framework=TestFramework.JUNIT5),
    )
    
    config_dict = config.to_dict()
    
    print(f"\n配置字典结构:")
    print(f"  project_name: {config_dict['project_name']}")
    print(f"  build:")
    print(f"    tool: {config_dict['build']['tool']}")
    print(f"    java_version: {config_dict['build']['java_version']}")
    print(f"  testing:")
    print(f"    framework: {config_dict['testing']['framework']}")
    print(f"    target_coverage: {config_dict['testing']['target_coverage']}")


def demo_integration_with_cli():
    """演示与 CLI 集成"""
    print("\n" + "="*60)
    print("演示 10: 与 CLI 集成")
    print("="*60)
    
    print("""
CLI 命令示例:

1. 初始化项目配置
   $ pyutagent-cli project init
   $ pyutagent-cli project init --path /path/to/project --force

2. 查看配置
   $ pyutagent-cli project show
   $ pyutagent-cli project show --format json

3. 验证配置
   $ pyutagent-cli project validate

4. 修改配置
   $ pyutagent-cli project set testing.target_coverage 0.9
   $ pyutagent-cli project set agent.max_iterations 15

5. 列出项目文件
   $ pyutagent-cli project list

6. 查看项目信息
   $ pyutagent-cli project info
""")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("项目配置系统演示")
    print("PYUT.md - 类似 CLAUDE.md 的项目配置")
    print("="*60)
    
    demo_build_config()
    demo_test_config()
    demo_agent_preferences()
    demo_code_style()
    demo_project_context()
    demo_project_config()
    demo_config_loader()
    demo_create_config_template()
    demo_config_to_dict()
    demo_integration_with_cli()
    
    print("\n" + "="*60)
    print("项目配置系统演示完成!")
    print("="*60)
    print("""
核心功能:
1. BuildConfig - 构建工具配置 (Maven/Gradle)
2. TestConfig - 测试框架配置 (JUnit/TestNG)
3. AgentPreferences - Agent 行为偏好
4. CodeStyle - 编码规范定制
5. ProjectContext - 项目上下文信息
6. ProjectConfig - 完整项目配置
7. ProjectConfigLoader - 配置加载器
8. create_config_template - 生成配置模板

使用场景:
- 项目初始化时自动生成配置
- 持久化项目特定的设置
- 团队成员共享配置
- CI/CD 集成
""")


if __name__ == "__main__":
    main()
