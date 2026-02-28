## 修改计划

需要在两个文件中修改系统提示词，添加 `@DisplayName` 注解的要求：

### 1. 修改 `pyutagent/agent/prompts.py`
在 `_build_system_prompt()` 方法的 Guidelines 中添加一条关于 `@DisplayName` 的要求：
- 位置：第18-27行之间
- 添加内容：使用 `@DisplayName` 注解为每个测试方法提供描述性名称

### 2. 修改 `pyutagent/tools/aider_integration.py`
在 `AiderTestGenerator.generate_initial_test()` 方法的 `default_system_prompt` 中添加 `@DisplayName` 要求：
- 位置：第596-602行
- 添加内容：使用 `@DisplayName` 注解描述测试目的

### 修改内容示例
在两个提示词的 guidelines 中添加类似：
```
- 使用 @DisplayName 注解为每个测试方法添加描述性名称，说明测试的目的和场景
```

这样可以确保生成的Java单元测试用例会带上 `@DisplayName("描述信息")` 注解。