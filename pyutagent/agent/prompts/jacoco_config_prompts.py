"""Prompts for JaCoCo configuration generation."""

JACOCO_CONFIG_GENERATION_PROMPT = """你是一个 Maven 配置专家。请分析以下 pom.xml 内容，并生成 JaCoCo 代码覆盖率工具的配置。

当前 pom.xml:
```xml
{pom_content}
```

请生成以下 JSON 格式的配置建议:
{{
    "dependencies": [],
    "build_plugins": [
        {{
            "group_id": "org.jacoco",
            "artifact_id": "jacoco-maven-plugin",
            "version": "0.8.11",
            "executions": [
                {{
                    "id": "prepare-agent",
                    "goals": ["prepare-agent"],
                    "phase": "test-compile"
                }},
                {{
                    "id": "report",
                    "goals": ["report"],
                    "phase": "test"
                }}
            ],
            "configuration": {{
                "excludes": []
            }}
        }}
    ],
    "explanation": "配置说明...",
    "warnings": []
}}

注意:
1. 版本号应该与项目中的其他依赖版本兼容，推荐使用 0.8.11 或更高版本
2. 如果已有 JaCoCo 配置，请分析现有配置并提供升级建议
3. 确保配置符合 Maven 标准
4. 如果 pom.xml 中已经有 build/plugins 部分，请提供插件配置建议
5. 如果项目使用 Spring Boot，可能需要特殊配置
6. 保持 JSON 格式正确，不要添加注释

请只返回 JSON 格式的配置，不要有其他内容。"""


JACOCO_CONFIG_ANALYSIS_PROMPT = """你是一个 Maven 配置分析专家。请分析以下 pom.xml 内容，判断是否已经配置了 JaCoCo 代码覆盖率工具。

当前 pom.xml:
```xml
{pom_content}
```

请分析以下方面:
1. 是否已经配置了 jacoco-maven-plugin
2. 插件版本是多少
3. 配置了哪些 execution goals
4. 是否有任何配置问题

请以 JSON 格式返回分析结果:
{{
    "is_configured": true/false,
    "plugin_version": "版本号或 null",
    "executions": ["prepare-agent", "report"],
    "issues": [],
    "recommendations": []
}}

只返回 JSON 格式的结果，不要有其他内容。"""


JACOCO_CONFIG_PREVIEW_PROMPT = """你是一个 Maven 配置专家。请根据以下原始 pom.xml 和建议的 JaCoCo 配置，生成一个配置预览，展示 pom.xml 将会如何被修改。

原始 pom.xml:
```xml
{original_pom}
```

建议的 JaCoCo 配置:
```json
{config}
```

请生成一个用户友好的配置预览，包含:
1. 将要添加的内容（以 + 标记）
2. 将要修改的内容（以 ~ 标记）
3. 简要的配置说明

格式示例:
```
📋 JaCoCo 配置预览

📝 将要添加的插件配置:
+ <plugin>
+     <groupId>org.jacoco</groupId>
+     <artifactId>jacoco-maven-plugin</artifactId>
+     <version>0.8.11</version>
+     ...
+ </plugin>

📖 配置说明:
- 此配置将在 test-compile 阶段准备 JaCoCo agent
- 在 test 阶段生成覆盖率报告
- 报告将生成在 target/site/jacoco 目录
```
"""
