"""Unified error classification service.

This module provides a single point for all error classification logic,
eliminating duplication across the codebase.
"""

import logging
import re
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from pyutagent.core.error_recovery import ErrorCategory, ErrorClassifier

logger = logging.getLogger(__name__)


class ErrorSubCategory(Enum):
    """错误子类型细分，用于更精确的错误诊断和恢复策略选择"""
    # 编译错误子类型
    MISSING_DEPENDENCY = auto()       # 缺少依赖包
    MISSING_IMPORT = auto()           # 缺少导入语句
    TYPE_MISMATCH = auto()            # 类型不匹配
    SYNTAX_ERROR = auto()             # 语法错误
    SYMBOL_NOT_FOUND = auto()         # 符号未找到
    INCOMPATIBLE_TYPES = auto()       # 类型不兼容
    METHOD_NOT_FOUND = auto()         # 方法未找到
    
    # 工具执行错误子类型
    MAVEN_DEPENDENCY_ERROR = auto()   # Maven依赖解析错误
    MAVEN_BUILD_ERROR = auto()        # Maven构建错误
    MAVEN_TEST_ERROR = auto()         # Maven测试错误
    MAVEN_NETWORK_ERROR = auto()      # Maven网络错误
    
    # 测试错误子类型
    ASSERTION_FAILED = auto()         # 断言失败
    NULL_POINTER = auto()             # 空指针异常
    MOCK_INVOCATION_ERROR = auto()    # Mock调用错误
    TEST_TIMEOUT = auto()             # 测试超时
    
    # 环境错误子类型
    NETWORK_UNREACHABLE = auto()      # 网络不可达
    PERMISSION_DENIED = auto()        # 权限被拒绝
    RESOURCE_EXHAUSTED = auto()       # 资源耗尽
    
    # 未知子类型
    UNKNOWN = auto()


DEPENDENCY_ERROR_PATTERNS = [
    (r"package\s+([\w.]+)\s+does not exist", "missing_package"),
    (r"cannot find symbol.*package\s+([\w.]+)", "missing_package"),
    (r"could not resolve dependencies", "resolve_failed"),
    (r"Failed to execute goal.*dependency", "maven_dependency"),
    (r"Failed to collect dependencies", "collect_failed"),
    (r"Could not find artifact\s+([\w.:-]+)", "artifact_not_found"),
    (r"Cannot resolve\s+([\w.:-]+)", "cannot_resolve"),
    (r"cannot find symbol\s*:\s*class\s+([\w.]+)", "missing_class"),
    (r"cannot find symbol\s*:\s*method\s+([\w.]+)", "missing_method"),
    (r"cannot find symbol\s*:\s*variable\s+([\w.]+)", "missing_variable"),
    (r"package\s+([\w.]+(?:\.[\w.]+)*)\s+does\s+not\s+exist", "missing_package"),
]

COMPILATION_ERROR_PATTERNS = [
    (r"cannot find symbol\s*:\s*method\s+(\w+)", "missing_method"),
    (r"cannot find symbol\s*:\s*constructor\s+(\w+)", "missing_constructor"),
    (r"cannot resolve method\s+'([^']+)'", "missing_method"),
    (r"incompatible types.*found\s*:\s*([\w.]+).*required\s*:\s*([\w.]+)", "type_mismatch"),
    (r"cannot be applied to\s*([\w()]+)", "method_signature_mismatch"),
    (r"no suitable method found\s+for\s+([\w.]+)", "no_suitable_method"),
    (r"(\w+)\s+has\s+private\s+access\s+in\s+(\w+)", "private_access"),
    (r"(\w+)\s+has\s+protected\s+access\s+in\s+(\w+)", "protected_access"),
    (r"illegal start of type", "illegal_start"),
    (r"'.class'\s+expected", "class_expected"),
    (r"cannot assign a value to final variable", "final_assignment"),
    (r"possible lossy conversion from\s+(\w+)\s+to\s+(\w+)", "narrowing_conversion"),
    (r"bad operand types for binary operator\s+'(\w+)'", "binary_operator_type_error"),
    (r"operator\s+\w+\s+cannot be applied\s+to\s+([\w(), ]+)", "operator_type_error"),
]

IMPORT_ERROR_PATTERNS = [
    (r"cannot find symbol\s*:\s*class\s+([\w.]+)", "missing_class"),
    (r"cannot find symbol\s*:\s*variable\s+([\w.]+)", "missing_variable"),
    (r"cannot find symbol\s*:\s*method\s+([\w.]+)", "missing_method"),
    (r"symbol:\s*class\s+([\w.]+)", "missing_class"),
    (r"symbol:\s*variable\s+([\w.]+)", "missing_variable"),
]

TEST_DEPENDENCY_PACKAGES = {
    "org.junit": "junit-jupiter",
    "org.junit.jupiter": "junit-jupiter",
    "org.mockito": "mockito-core",
    "org.assertj": "assertj-core",
    "org.hamcrest": "hamcrest",
    "org.powermock": "powermock",
}

COMMON_JAVA_CLASS_MAPPINGS = {
    "org.junit.jupiter.api.Test": ("org.junit.jupiter", "junit-jupiter", "5.10.0"),
    "org.junit.jupiter.api.BeforeEach": ("org.junit.jupiter", "junit-jupiter", "5.10.0"),
    "org.junit.jupiter.api.AfterEach": ("org.junit.jupiter", "junit-jupiter", "5.10.0"),
    "org.junit.jupiter.api.BeforeAll": ("org.junit.jupiter", "junit-jupiter", "5.10.0"),
    "org.junit.jupiter.api.AfterAll": ("org.junit.jupiter", "junit-jupiter", "5.10.0"),
    "org.junit.jupiter.api.Assertions": ("org.junit.jupiter", "junit-jupiter", "5.10.0"),
    "org.junit.jupiter.api.Assumptions": ("org.junit.jupiter", "junit-jupiter", "5.10.0"),
    "org.junit.jupiter.api.Disabled": ("org.junit.jupiter", "junit-jupiter", "5.10.0"),
    "org.junit.jupiter.api.Tag": ("org.junit.jupiter", "junit-jupiter", "5.10.0"),
    "org.junit.jupiter.api.DisplayName": ("org.junit.jupiter", "junit-jupiter", "5.10.0"),
    "org.junit.jupiter.api.extension.ExtendWith": ("org.junit.jupiter", "junit-jupiter", "5.10.0"),
    "org.junit.jupiter.params.ParameterizedTest": ("org.junit.jupiter", "junit-jupiter-params", "5.10.0"),
    "org.junit.jupiter.params.provider.ValueSource": ("org.junit.jupiter", "junit-jupiter-params", "5.10.0"),
    "org.junit.jupiter.params.provider.CsvSource": ("org.junit.jupiter", "junit-jupiter-params", "5.10.0"),
    "org.junit.platform.commons.annotation.Testable": ("org.junit.platform", "junit-platform-commons", "5.10.0"),
    
    "org.mockito.Mockito": ("org.mockito", "mockito-core", "5.8.0"),
    "org.mockito.InjectMocks": ("org.mockito", "mockito-core", "5.8.0"),
    "org.mockito.Mock": ("org.mockito", "mockito-core", "5.8.0"),
    "org.mockito.Spy": ("org.mockito", "mockito-core", "5.8.0"),
    "org.mockito.MockitoAnnotations": ("org.mockito", "mockito-core", "5.8.0"),
    "org.mockito.junit.jupiter.MockitoExtension": ("org.mockito", "mockito-junit-jupiter", "5.8.0"),
    "org.mockito.junit5.MockitoExtension": ("org.mockito", "mockito-junit-jupiter", "5.8.0"),
    "org.mockito.ArgumentMatchers": ("org.mockito", "mockito-core", "5.8.0"),
    "org.mockitoMatchers": ("org.mockito", "mockito-core", "5.8.0"),
    "org.mockito.Any": ("org.mockito", "mockito-core", "5.8.0"),
    "org.mockito.ArgumentCaptor": ("org.mockito", "mockito-core", "5.8.0"),
    "org.mockito.verification.VerificationMode": ("org.mockito", "mockito-core", "5.8.0"),
    "org.mockito.BDDMockito": ("org.mockito", "mockito-core", "5.8.0"),
    "org.mockito.stubbing.Answer": ("org.mockito", "mockito-core", "5.8.0"),
    "org.mockito.stubbing.OngoingStubbing": ("org.mockito", "mockito-core", "5.8.0"),
    
    "org.assertj.core.api.Assertions": ("org.assertj", "assertj-core", "3.25.0"),
    "org.assertj.core.api.SoftAssertions": ("org.assertj", "assertj-core", "3.25.0"),
    "org.assertj.core.api.AbstractSoftAssertions": ("org.assertj", "assertj-core", "3.25.0"),
    "org.assertj.core.api.JUnitSoftAssertions": ("org.assertj", "assertj-soft-assertions", "3.25.0"),
    "org.assertj.core.api.WithAssertions": ("org.assertj", "assertj-core", "3.25.0"),
    
    "org.hamcrest.MatcherAssert": ("org.hamcrest", "hamcrest", "2.2"),
    "org.hamcrest.Matchers": ("org.hamcrest", "hamcrest", "2.2"),
    "org.hamcrest.CoreMatchers": ("org.hamcrest", "hamcrest", "2.2"),
    "org.hamcrest.Description": ("org.hamcrest", "hamcrest", "2.2"),
    "org.hamcrest.SelfDescribing": ("org.hamcrest", "hamcrest", "2.2"),
    "org.hamcrest.Matcher": ("org.hamcrest", "hamcrest", "2.2"),
    
    "org.springframework.boot.SpringApplication": ("org.springframework.boot", "spring-boot-starter-test", "3.2.0"),
    "org.springframework.boot.test.context.SpringBootTest": ("org.springframework.boot", "spring-boot-starter-test", "3.2.0"),
    "org.springframework.boot.test.mock.mockito.MockBean": ("org.springframework.boot", "spring-boot-starter-test", "3.2.0"),
    "org.springframework.context.ApplicationContext": ("org.springframework", "spring-context", "6.1.0"),
    "org.springframework.beans.factory.annotation.Autowired": ("org.springframework", "spring-beans", "6.1.0"),
    "org.springframework.test.context.junit.jupiter.SpringExtension": ("org.springframework", "spring-test", "6.1.0"),
    "org.springframework.test.context.TestPropertySource": ("org.springframework", "spring-test", "6.1.0"),
    "org.springframework.test.web.servlet.MockMvc": ("org.springframework", "spring-test", "6.1.0"),
    "org.springframework.http.ResponseEntity": ("org.springframework", "spring-web", "6.1.0"),
    "org.springframework.web.client.RestTemplate": ("org.springframework", "spring-web", "6.1.0"),
    
    "org.jline.utils.AttributedString": ("org.jline", "jline", "3.25.0"),
    "org.jline.utils.AttributedStyle": ("org.jline", "jline", "3.25.0"),
    "org.jline.reader.LineReader": ("org.jline", "jline", "3.25.0"),
    "org.jline.reader.LineReaderBuilder": ("org.jline", "jline", "3.25.0"),
    "org.jline.terminal.Terminal": ("org.jline", "jline", "3.25.0"),
    
    "ch.qos.logback.core.Appender": ("ch.qos.logback", "logback-classic", "1.4.14"),
    "ch.qos.logback.classic.Logger": ("ch.qos.logback", "logback-classic", "1.4.14"),
    "org.slf4j.Logger": ("org.slf4j", "slf4j-api", "2.0.9"),
    "org.slf4j.LoggerFactory": ("org.slf4j", "slf4j-api", "2.0.9"),
    
    "org.apache.commons.lang3.StringUtils": ("org.apache.commons", "commons-lang3", "3.14.0"),
    "org.apache.commons.lang3.ObjectUtils": ("org.apache.commons", "commons-lang3", "3.14.0"),
    "org.apache.commons.collections4.CollectionUtils": ("org.apache.commons", "commons-collections4", "4.4"),
    "org.apache.commons.io.FileUtils": ("org.apache.commons", "commons-io", "2.15.0"),
    "org.apache.commons.io.IOUtils": ("org.apache.commons", "commons-io", "2.15.0"),
    
    "com.fasterxml.jackson.databind.ObjectMapper": ("com.fasterxml.jackson.core", "jackson-databind", "2.16.0"),
    "com.fasterxml.jackson.databind.JsonNode": ("com.fasterxml.jackson.core", "jackson-databind", "2.16.0"),
    "com.fasterxml.jackson.core.JsonProcessingException": ("com.fasterxml.jackson.core", "jackson-databind", "2.16.0"),
    "com.google.gson.Gson": ("com.google.code.gson", "gson", "2.10.1"),
    "com.google.gson.JsonObject": ("com.google.code.gson", "gson", "2.10.1"),
    
    "java.io.File": ("java-sdk", "java-sdk", "8"),
    "java.nio.file.Path": ("java-sdk", "java-sdk", "8"),
    "java.util.List": ("java-sdk", "java-sdk", "8"),
    "java.util.Map": ("java-sdk", "java-sdk", "8"),
    "java.util.Set": ("java-sdk", "java-sdk", "8"),
    "java.util.Optional": ("java-sdk", "java-sdk", "8"),
    "java.util.stream.Collectors": ("java-sdk", "java-sdk", "8"),
}

COMMON_PACKAGE_PREFIXES = {
    "org.junit.jupiter": ("org.junit.jupiter", "junit-jupiter", "5.10.0"),
    "org.junit.vintage": ("org.junit.vintage", "junit-vintage-engine", "5.10.0"),
    "org.mockito": ("org.mockito", "mockito-core", "5.8.0"),
    "org.assertj.core": ("org.assertj", "assertj-core", "3.25.0"),
    "org.hamcrest": ("org.hamcrest", "hamcrest", "2.2"),
    "org.springframework.boot": ("org.springframework.boot", "spring-boot-starter", "3.2.0"),
    "org.springframework.test": ("org.springframework", "spring-test", "6.1.0"),
    "org.springframework.context": ("org.springframework", "spring-context", "6.1.0"),
    "org.jline": ("org.jline", "jline", "3.25.0"),
    "ch.qos.logback": ("ch.qos.logback", "logback-classic", "1.4.14"),
    "org.slf4j": ("org.slf4j", "slf4j-api", "2.0.9"),
    "org.apache.commons.lang3": ("org.apache.commons", "commons-lang3", "3.14.0"),
    "org.apache.commons.collections4": ("org.apache.commons", "commons-collections4", "4.4"),
    "org.apache.commons.io": ("org.apache.commons", "commons-io", "2.15.0"),
    "com.fasterxml.jackson": ("com.fasterxml.jackson.core", "jackson-databind", "2.16.0"),
    "com.google.gson": ("com.google.code.gson", "gson", "2.10.1"),
}


def detect_missing_dependencies(compiler_output: str) -> Dict[str, Any]:
    """检测编译输出中缺失的依赖包
    
    Args:
        compiler_output: 编译器输出字符串
        
    Returns:
        包含缺失依赖信息的字典：
        - missing_packages: 缺失的包名列表
        - suggested_dependencies: 建议添加的Maven依赖
        - is_test_dependency: 是否为测试依赖
        - sub_category: 错误子类型
    """
    missing_packages = []
    suggested_dependencies = []
    is_test_dependency = False
    
    for pattern, error_type in DEPENDENCY_ERROR_PATTERNS:
        matches = re.findall(pattern, compiler_output, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0] if match else ""
            
            match = match.strip()
            if match and match not in missing_packages:
                missing_packages.append(match)
                
                dep_info = get_dependency_info(match)
                if dep_info:
                    artifact = dep_info.get('artifact_id')
                    if artifact and artifact not in suggested_dependencies:
                        suggested_dependencies.append(artifact)
                    is_test_dependency = True
                else:
                    for pkg_prefix, dep_name in TEST_DEPENDENCY_PACKAGES.items():
                        if match.startswith(pkg_prefix):
                            is_test_dependency = True
                            if dep_name not in suggested_dependencies:
                                suggested_dependencies.append(dep_name)
                            break
    
    sub_category = ErrorSubCategory.UNKNOWN
    if missing_packages:
        if any("symbol" in p.lower() for p in missing_packages):
            sub_category = ErrorSubCategory.SYMBOL_NOT_FOUND
        else:
            sub_category = ErrorSubCategory.MISSING_DEPENDENCY
    elif "could not resolve dependencies" in compiler_output.lower():
        sub_category = ErrorSubCategory.MAVEN_DEPENDENCY_ERROR
    
    return {
        "missing_packages": missing_packages,
        "suggested_dependencies": suggested_dependencies,
        "is_test_dependency": is_test_dependency,
        "sub_category": sub_category,
        "error_count": len(missing_packages),
    }


def detect_missing_imports(compiler_output: str) -> List[str]:
    """从编译错误中检测缺失的导入语句
    
    Args:
        compiler_output: 编译器输出字符串
        
    Returns:
        缺失的导入语句列表
    """
    missing_imports = []
    
    for pattern, error_type in IMPORT_ERROR_PATTERNS:
        matches = re.findall(pattern, compiler_output, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0] if match else ""
            
            match = match.strip()
            if match and match not in missing_imports:
                full_class_path = _resolve_full_class_path(match, compiler_output)
                if full_class_path:
                    if full_class_path not in missing_imports:
                        missing_imports.append(full_class_path)
                elif match in COMMON_JAVA_CLASS_MAPPINGS:
                    missing_imports.append(match)
                elif not match.startswith('java.'):
                    missing_imports.append(match)
    
    return missing_imports


def _resolve_full_class_path(class_name: str, compiler_output: str) -> Optional[str]:
    """根据类名和编译输出解析完整的类路径
    
    Args:
        class_name: 简化的类名（如 Assertions）
        compiler_output: 编译器输出
        
    Returns:
        完整的类路径（如 org.junit.jupiter.api.Assertions）或 None
    """
    import re
    
    pattern = rf"{re.escape(class_name)}.*?location.*?package\s+([\w.]+)"
    match = re.search(pattern, compiler_output, re.IGNORECASE | re.DOTALL)
    
    if match:
        package = match.group(1)
        return f"{package}.{class_name}"
    
    common_packages = [
        "org.junit.jupiter.api",
        "org.junit.jupiter.api.Assertions",
        "org.mockito.Mockito",
        "org.assertj.core.api",
        "org.springframework.boot",
        "org.springframework.context",
    ]
    
    for pkg in common_packages:
        potential = f"{pkg}.{class_name}"
        if potential in COMMON_JAVA_CLASS_MAPPINGS:
            return potential
    
    return None


def get_dependency_for_class(class_name: str) -> Optional[Dict[str, str]]:
    """根据类名获取对应的Maven依赖信息
    
    Args:
        class_name: 完整的类名 (如 org.junit.jupiter.api.Test)
        
    Returns:
        包含 group_id, artifact_id, version 的字典，或 None
    """
    if class_name in COMMON_JAVA_CLASS_MAPPINGS:
        group_id, artifact_id, version = COMMON_JAVA_CLASS_MAPPINGS[class_name]
        return {
            "group_id": group_id,
            "artifact_id": artifact_id,
            "version": version,
            "scope": "test"
        }
    
    return get_dependency_by_package_prefix(class_name)


def get_dependency_by_package_prefix(class_name: str) -> Optional[Dict[str, str]]:
    """根据类名的包前缀获取对应的Maven依赖信息
    
    Args:
        class_name: 类名 (如 org.mockito.Mockito 或 org.junit.jupiter)
        
    Returns:
        包含 group_id, artifact_id, version 的字典，或 None
    """
    if not class_name:
        return None
    
    for prefix, (group_id, artifact_id, version) in COMMON_PACKAGE_PREFIXES.items():
        if class_name.startswith(prefix):
            return {
                "group_id": group_id,
                "artifact_id": artifact_id,
                "version": version,
                "scope": "test"
            }
    
    return None


def get_dependency_info(class_name: str) -> Optional[Dict[str, str]]:
    """获取依赖信息的统一入口函数
    
    优先使用精确匹配，失败则使用包前缀匹配
    
    Args:
        class_name: 类名
        
    Returns:
        包含 group_id, artifact_id, version 的字典，或 None
    """
    result = get_dependency_for_class(class_name)
    if result:
        return result
    
    parts = class_name.split('.')
    for i in range(len(parts) - 1, 0, -1):
        prefix = '.'.join(parts[:i])
        result = get_dependency_by_package_prefix(prefix)
        if result:
            return result
    
    return None


def detect_compilation_error_type(compiler_output: str) -> ErrorSubCategory:
    """检测编译错误的具体类型
    
    Args:
        compiler_output: 编译器输出字符串
        
    Returns:
        ErrorSubCategory 枚举值
    """
    output_lower = compiler_output.lower()
    
    if "package" in output_lower and "does not exist" in output_lower:
        return ErrorSubCategory.MISSING_DEPENDENCY
    
    if "cannot find symbol" in output_lower:
        if "package" in output_lower:
            return ErrorSubCategory.MISSING_DEPENDENCY
        return ErrorSubCategory.SYMBOL_NOT_FOUND
    
    if "incompatible types" in output_lower:
        return ErrorSubCategory.INCOMPATIBLE_TYPES
    
    if "method" in output_lower and "cannot be applied" in output_lower:
        return ErrorSubCategory.METHOD_NOT_FOUND
    
    if "expected" in output_lower and (";" in output_lower or "{" in output_lower or "}" in output_lower):
        return ErrorSubCategory.SYNTAX_ERROR
    
    if "type mismatch" in output_lower:
        return ErrorSubCategory.TYPE_MISMATCH
    
    return ErrorSubCategory.UNKNOWN


def extract_required_imports_from_code(code: str, existing_imports: Optional[Set[str]] = None) -> List[str]:
    """从Java代码中提取需要但未导入的类
    
    使用静态分析从代码中提取完全限定类名，与现有导入对比，
    返回需要添加的导入列表
    
    Args:
        code: Java源代码
        existing_imports: 已存在的导入集合（可选）
        
    Returns:
        需要添加的导入列表
    """
    if existing_imports is None:
        existing_imports = set(re.findall(r'^import\s+([\w.]+);', code, re.MULTILINE))
    
    required_imports = []
    
    fully_qualified_patterns = [
        r'\b([a-z][a-z0-9_]*\.[A-Z][a-zA-Z0-9_]*(?:\.[A-Z][a-zA-Z0-9_]*)+\b)',
        r'\borg\.[a-z]+\.[A-Z][A-Za-z0-9_]+\b',
        r'\bcom\.[a-z]+\.[A-Z][A-Za-z0-9_]+\b',
        r'\bio\.[a-z]+\.[A-Z][A-Za-z0-9_]+\b',
    ]
    
    for pattern in fully_qualified_patterns:
        matches = re.findall(pattern, code)
        for match in matches:
            if match in existing_imports:
                continue
            if match.startswith('java.lang.'):
                continue
            if match in COMMON_JAVA_CLASS_MAPPINGS or match in COMMON_PACKAGE_PREFIXES:
                if match not in required_imports:
                    required_imports.append(match)
    
    return required_imports


def suggest_dependency_coordinates(class_name: str, pom_content: Optional[str] = None) -> Optional[Dict[str, str]]:
    """返回完整的Maven依赖坐标建议
    
    Args:
        class_name: 完整的类名
        pom_content: pom.xml内容（可选，用于推断版本）
        
    Returns:
        包含完整Maven坐标的字典，或None
    """
    dep_info = get_dependency_info(class_name)
    if not dep_info:
        return None
    
    result = {
        "groupId": dep_info["group_id"],
        "artifactId": dep_info["artifact_id"],
        "version": dep_info["version"],
        "scope": dep_info.get("scope", "test"),
        "className": class_name
    }
    
    if pom_content:
        inferred_version = infer_version_from_pom(class_name, pom_content)
        if inferred_version:
            result["version"] = inferred_version
            result["inferred"] = True
    
    return result


def infer_version_from_pom(class_name: str, pom_content: str) -> Optional[str]:
    """从pom.xml推断类所属依赖的版本
    
    Args:
        class_name: 完整类名
        pom_content: pom.xml内容
        
    Returns:
        依赖版本字符串，或None
    """
    dep_info = get_dependency_info(class_name)
    if not dep_info:
        return None
    
    group_id = dep_info["group_id"]
    artifact_id = dep_info["artifact_id"]
    
    artifact_pattern = rf'<artifactId>{re.escape(artifact_id)}</artifactId>'
    if not re.search(artifact_pattern, pom_content):
        return None
    
    version_pattern = rf'<artifactId>{re.escape(artifact_id)}</artifactId>.*?<version>([^<]+)</version>'
    version_match = re.search(version_pattern, pom_content, re.DOTALL)
    
    if version_match:
        return version_match.group(1)
    
    parent_pattern = r'<parent>.*?<version>([^<]+)</version>.*?</parent>'
    parent_version = re.search(parent_pattern, pom_content, re.DOTALL)
    if parent_version:
        return parent_version.group(1)
    
    return None


def batch_resolve_compilation_errors(compiler_output: str, code: Optional[str] = None) -> Dict[str, Any]:
    """批量解析编译错误，返回完整的修复建议
    
    Args:
        compiler_output: 编译器输出
        code: 源代码（可选）
        
    Returns:
        包含所有修复建议的字典
    """
    result = {
        "missing_imports": [],
        "missing_dependencies": [],
        "error_types": [],
        "suggestions": []
    }
    
    missing_imports = detect_missing_imports(compiler_output)
    result["missing_imports"] = missing_imports
    
    for imp in missing_imports:
        coords = suggest_dependency_coordinates(imp)
        if coords:
            result["suggestions"].append({
                "type": "import_and_dependency",
                "import": imp,
                "dependency": coords
            })
            if coords not in result["missing_dependencies"]:
                result["missing_dependencies"].append(coords)
        else:
            result["suggestions"].append({
                "type": "import_only",
                "import": imp,
                "dependency": None
            })
    
    for pattern, error_type in COMPILATION_ERROR_PATTERNS:
        if re.search(pattern, compiler_output, re.IGNORECASE):
            if error_type not in result["error_types"]:
                result["error_types"].append(error_type)
    
    if code:
        required_from_code = extract_required_imports_from_code(code)
        for imp in required_from_code:
            if imp not in result["missing_imports"]:
                result["missing_imports"].append(imp)
                coords = suggest_dependency_coordinates(imp)
                if coords:
                    result["suggestions"].append({
                        "type": "import_and_dependency",
                        "import": imp,
                        "dependency": coords
                    })
    
    return result


def detect_maven_error_type(maven_output: str) -> ErrorSubCategory:
    """检测Maven错误的具体类型
    
    Args:
        maven_output: Maven输出字符串
        
    Returns:
        ErrorSubCategory 枚举值
    """
    output_lower = maven_output.lower()
    
    if "dependency" in output_lower or "resolve" in output_lower:
        if "could not" in output_lower or "failed" in output_lower:
            return ErrorSubCategory.MAVEN_DEPENDENCY_ERROR
    
    if "connection" in output_lower or "network" in output_lower or "timeout" in output_lower:
        return ErrorSubCategory.MAVEN_NETWORK_ERROR
    
    if "test" in output_lower and "failed" in output_lower:
        return ErrorSubCategory.MAVEN_TEST_ERROR
    
    if "build" in output_lower and "failed" in output_lower:
        return ErrorSubCategory.MAVEN_BUILD_ERROR
    
    return ErrorSubCategory.UNKNOWN


class ErrorClassificationService:
    """Unified service for error classification.
    
    This service provides a single point for all error classification
    logic, eliminating duplication across the codebase.
    
    Features:
    - Error classification by type and message
    - Retryability checking
    - Recovery strategy recommendation
    - Singleton pattern for global access
    
    Example:
        >>> service = get_error_classification_service()
        >>> category = service.classify(ValueError("test"))
        >>> print(category)
        ErrorCategory.VALIDATION
    """
    
    _instance: Optional['ErrorClassificationService'] = None
    
    def __new__(cls) -> 'ErrorClassificationService':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def classify(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> ErrorCategory:
        """Classify an error into a category.
        
        Args:
            error: The error to classify
            context: Optional context information (step name, etc.)
            
        Returns:
            ErrorCategory enum value
        """
        category = ErrorClassifier.classify(error)
        
        if context:
            step = context.get("step", "").lower()
            if "compile" in step:
                return ErrorCategory.COMPILATION_ERROR
            elif "test" in step and "fail" in str(error).lower():
                return ErrorCategory.TEST_FAILURE
        
        return category
    
    def categorize_by_message(self, error_message: str, error_details: Dict[str, Any]) -> ErrorCategory:
        """Categorize error by message and details.
        
        Args:
            error_message: Error message string
            error_details: Additional error details
            
        Returns:
            ErrorCategory enum value
        """
        return ErrorClassifier.categorize_error(error_message, error_details)
    
    def is_retryable(self, error: Exception) -> bool:
        """Check if an error is retryable.
        
        Args:
            error: The error to check
            
        Returns:
            True if the error is retryable
        """
        return ErrorClassifier.is_retryable(error)
    
    def is_retryable_category(self, category: ErrorCategory) -> bool:
        """Check if an error category is retryable.
        
        Args:
            category: Error category to check
            
        Returns:
            True if the category is retryable
        """
        retryable_categories = {
            ErrorCategory.TRANSIENT,
            ErrorCategory.NETWORK,
            ErrorCategory.TIMEOUT,
            ErrorCategory.RESOURCE,
            ErrorCategory.COMPILATION_ERROR,
            ErrorCategory.TEST_FAILURE,
            ErrorCategory.LLM_API_ERROR,
        }
        return category in retryable_categories
    
    def get_recovery_strategy(self, error: Exception, attempt_count: int = 0) -> str:
        """Get recommended recovery strategy for an error.
        
        Args:
            error: The error
            attempt_count: Number of previous attempts
            
        Returns:
            Strategy name string
        """
        category = self.classify(error)
        return self.get_strategy_for_category(category, attempt_count)
    
    def get_strategy_for_category(self, category: ErrorCategory, attempt_count: int = 0) -> str:
        """Get recommended recovery strategy for an error category.
        
        Args:
            category: Error category
            attempt_count: Number of previous attempts
            
        Returns:
            Strategy name string
        """
        if category == ErrorCategory.NETWORK:
            return "RETRY_WITH_BACKOFF"
        elif category == ErrorCategory.TIMEOUT:
            return "RETRY_WITH_BACKOFF"
        elif category == ErrorCategory.COMPILATION_ERROR:
            return "ANALYZE_AND_FIX"
        elif category == ErrorCategory.TEST_FAILURE:
            if attempt_count < 3:
                return "ANALYZE_AND_FIX"
            else:
                return "RESET_AND_REGENERATE"
        elif category == ErrorCategory.LLM_API_ERROR:
            if attempt_count < 2:
                return "RETRY_WITH_BACKOFF"
            else:
                return "FALLBACK_ALTERNATIVE"
        elif category in (ErrorCategory.TRANSIENT, ErrorCategory.RESOURCE):
            return "RETRY_IMMEDIATE"
        elif category == ErrorCategory.PARSING_ERROR:
            return "RESET_AND_REGENERATE"
        elif category == ErrorCategory.FILE_IO_ERROR:
            return "SKIP_AND_CONTINUE"
        else:
            return "ANALYZE_AND_FIX"
    
    def get_error_severity(self, error: Exception) -> str:
        """Get severity level for an error.
        
        Args:
            error: The error
            
        Returns:
            Severity level string: "low", "medium", "high", "critical"
        """
        category = self.classify(error)
        
        severity_mapping = {
            ErrorCategory.TRANSIENT: "low",
            ErrorCategory.NETWORK: "medium",
            ErrorCategory.TIMEOUT: "medium",
            ErrorCategory.RESOURCE: "high",
            ErrorCategory.VALIDATION: "medium",
            ErrorCategory.SYNTAX: "high",
            ErrorCategory.LOGIC: "medium",
            ErrorCategory.COMPILATION_ERROR: "medium",
            ErrorCategory.TEST_FAILURE: "medium",
            ErrorCategory.TOOL_EXECUTION_ERROR: "high",
            ErrorCategory.PARSING_ERROR: "high",
            ErrorCategory.GENERATION_ERROR: "high",
            ErrorCategory.FILE_IO_ERROR: "high",
            ErrorCategory.LLM_API_ERROR: "high",
            ErrorCategory.PERMANENT: "critical",
            ErrorCategory.UNKNOWN: "medium",
        }
        
        return severity_mapping.get(category, "medium")
    
    def should_skip_recovery(self, error: Exception) -> bool:
        """Check if recovery should be skipped for this error.
        
        Args:
            error: The error
            
        Returns:
            True if recovery should be skipped
        """
        error_message = str(error).lower()
        
        skip_patterns = [
            "no compilation errors",
            "no test failures",
            "all tests passed",
            "compilation successful",
            "tests passed",
        ]
        
        for pattern in skip_patterns:
            if pattern in error_message:
                return True
        
        return False
    
    def get_error_info(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get comprehensive error information.
        
        Args:
            error: The error
            context: Optional context information
            
        Returns:
            Dictionary with error information
        """
        category = self.classify(error, context)
        
        return {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "category": category.name,
            "is_retryable": self.is_retryable_category(category),
            "recommended_strategy": self.get_strategy_for_category(category),
            "severity": self.get_error_severity(error),
            "should_skip_recovery": self.should_skip_recovery(error),
        }
    
    def get_detailed_error_info(
        self, 
        error: Exception, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """获取详细的错误信息，包括子类型和依赖检测
        
        Args:
            error: 错误对象
            context: 上下文信息，可包含 compiler_output, step 等
            
        Returns:
            包含详细错误信息的字典
        """
        base_info = self.get_error_info(error, context)
        category = self.classify(error, context)
        
        sub_category = ErrorSubCategory.UNKNOWN
        dependency_info = {}
        
        compiler_output = context.get("compiler_output", "") if context else ""
        if not compiler_output:
            compiler_output = str(error)
        
        if category == ErrorCategory.COMPILATION_ERROR:
            sub_category = detect_compilation_error_type(compiler_output)
            if sub_category == ErrorSubCategory.MISSING_DEPENDENCY:
                dependency_info = detect_missing_dependencies(compiler_output)
        
        elif category == ErrorCategory.TOOL_EXECUTION_ERROR:
            tool_name = context.get("tool", "") if context else ""
            if "maven" in tool_name.lower() or "mvn" in tool_name.lower():
                sub_category = detect_maven_error_type(compiler_output)
                if sub_category == ErrorSubCategory.MAVEN_DEPENDENCY_ERROR:
                    dependency_info = detect_missing_dependencies(compiler_output)
        
        elif category == ErrorCategory.TEST_FAILURE:
            failures = context.get("failures", []) if context else []
            if failures:
                first_failure = failures[0] if failures else {}
                error_msg = first_failure.get("error", "")
                if "assertion" in error_msg.lower():
                    sub_category = ErrorSubCategory.ASSERTION_FAILED
                elif "nullpointer" in error_msg.lower():
                    sub_category = ErrorSubCategory.NULL_POINTER
                elif "mock" in error_msg.lower():
                    sub_category = ErrorSubCategory.MOCK_INVOCATION_ERROR
        
        result = {
            **base_info,
            "sub_category": sub_category.name if sub_category != ErrorSubCategory.UNKNOWN else None,
            "is_environment_issue": self._is_environment_issue(category, sub_category),
            "needs_dependency_resolution": sub_category in (
                ErrorSubCategory.MISSING_DEPENDENCY,
                ErrorSubCategory.MAVEN_DEPENDENCY_ERROR,
            ),
        }
        
        if dependency_info:
            result["dependency_info"] = dependency_info
        
        return result
    
    def _is_environment_issue(self, category: ErrorCategory, sub_category: ErrorSubCategory) -> bool:
        """判断是否为环境问题（非代码问题）
        
        Args:
            category: 错误类别
            sub_category: 错误子类别
            
        Returns:
            如果是环境问题返回 True
        """
        environment_sub_categories = {
            ErrorSubCategory.MISSING_DEPENDENCY,
            ErrorSubCategory.MAVEN_DEPENDENCY_ERROR,
            ErrorSubCategory.MAVEN_NETWORK_ERROR,
            ErrorSubCategory.NETWORK_UNREACHABLE,
            ErrorSubCategory.PERMISSION_DENIED,
            ErrorSubCategory.RESOURCE_EXHAUSTED,
        }
        
        if sub_category in environment_sub_categories:
            return True
        
        if category in (ErrorCategory.NETWORK, ErrorCategory.RESOURCE):
            return True
        
        return False
    
    def get_strategy_for_sub_category(
        self, 
        sub_category: ErrorSubCategory,
        attempt_count: int = 0
    ) -> str:
        """根据错误子类型获取恢复策略
        
        Args:
            sub_category: 错误子类型
            attempt_count: 尝试次数
            
        Returns:
            策略名称字符串
        """
        strategy_mapping = {
            ErrorSubCategory.MISSING_DEPENDENCY: "INSTALL_DEPENDENCIES",
            ErrorSubCategory.MAVEN_DEPENDENCY_ERROR: "RESOLVE_DEPENDENCIES",
            ErrorSubCategory.MAVEN_NETWORK_ERROR: "RETRY_WITH_BACKOFF",
            ErrorSubCategory.NETWORK_UNREACHABLE: "RETRY_WITH_BACKOFF",
            ErrorSubCategory.PERMISSION_DENIED: "ESCALATE_TO_USER",
            ErrorSubCategory.RESOURCE_EXHAUSTED: "BACKOFF",
            ErrorSubCategory.SYNTAX_ERROR: "ANALYZE_AND_FIX",
            ErrorSubCategory.TYPE_MISMATCH: "ANALYZE_AND_FIX",
            ErrorSubCategory.INCOMPATIBLE_TYPES: "ANALYZE_AND_FIX",
            ErrorSubCategory.SYMBOL_NOT_FOUND: "ANALYZE_AND_FIX",
            ErrorSubCategory.METHOD_NOT_FOUND: "ANALYZE_AND_FIX",
            ErrorSubCategory.ASSERTION_FAILED: "ANALYZE_AND_FIX",
            ErrorSubCategory.NULL_POINTER: "ANALYZE_AND_FIX",
            ErrorSubCategory.MOCK_INVOCATION_ERROR: "ANALYZE_AND_FIX",
            ErrorSubCategory.TEST_TIMEOUT: "RETRY_WITH_BACKOFF",
        }
        
        return strategy_mapping.get(sub_category, "ANALYZE_AND_FIX")


error_classification_service = ErrorClassificationService()


def get_error_classification_service() -> ErrorClassificationService:
    """Get the global error classification service instance.
    
    Returns:
        ErrorClassificationService singleton instance
    """
    return error_classification_service


def classify_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> ErrorCategory:
    """Convenience function to classify an error.
    
    Args:
        error: The error to classify
        context: Optional context information
        
    Returns:
        ErrorCategory enum value
    """
    return get_error_classification_service().classify(error, context)


def is_retryable_error(error: Exception) -> bool:
    """Convenience function to check if an error is retryable.
    
    Args:
        error: The error to check
        
    Returns:
        True if the error is retryable
    """
    return get_error_classification_service().is_retryable(error)


def get_recovery_strategy(error: Exception, attempt_count: int = 0) -> str:
    """Convenience function to get recovery strategy.
    
    Args:
        error: The error
        attempt_count: Number of previous attempts
        
    Returns:
        Strategy name string
    """
    return get_error_classification_service().get_recovery_strategy(error, attempt_count)


def get_detailed_error_info(error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience function to get detailed error info.
    
    Args:
        error: The error
        context: Optional context information
        
    Returns:
        Dictionary with detailed error information
    """
    return get_error_classification_service().get_detailed_error_info(error, context)
