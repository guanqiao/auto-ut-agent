"""错误根因分析器 - 智能错误分析和修复策略推荐

本模块提供错误根因分析功能，包括:
- 编译错误语义分析
- 测试失败模式识别
- 自动定位根本原因
- 修复策略推荐
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from pathlib import Path
from enum import Enum, auto
from collections import defaultdict

from .cache import get_global_cache

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """错误类别"""
    SYNTAX_ERROR = auto()           # 语法错误
    TYPE_ERROR = auto()             # 类型错误
    IMPORT_ERROR = auto()           # 导入错误
    REFERENCE_ERROR = auto()        # 引用错误
    ACCESS_ERROR = auto()           # 访问权限错误
    OVERRIDE_ERROR = auto()         # 重写错误
    INITIALIZATION_ERROR = auto()   # 初始化错误
    RESOURCE_ERROR = auto()         # 资源错误
    ASSERTION_ERROR = auto()        # 断言错误
    MOCK_ERROR = auto()             # Mock 错误
    NULL_POINTER_ERROR = auto()     # 空指针错误
    LOGIC_ERROR = auto()            # 逻辑错误


class ErrorSeverity(Enum):
    """错误严重程度"""
    CRITICAL = "critical"    # 关键错误，必须修复
    HIGH = "high"           # 高优先级错误
    MEDIUM = "medium"       # 中等优先级
    LOW = "low"             # 低优先级


class FixStrategyType(Enum):
    """修复策略类型"""
    SYNTAX_FIX = auto()           # 语法修复
    TYPE_CORRECTION = auto()      # 类型修正
    IMPORT_ADDITION = auto()      # 添加导入
    VISIBILITY_CHANGE = auto()    # 可见性修改
    SIGNATURE_UPDATE = auto()     # 签名更新
    INITIALIZATION_FIX = auto()   # 初始化修复
    RESOURCE_MANAGEMENT = auto()  # 资源管理修复
    ASSERTION_FIX = auto()        # 断言修复
    MOCK_SETUP = auto()           # Mock 设置
    NULL_CHECK = auto()           # 空值检查
    LOGIC_CORRECTION = auto()     # 逻辑修正
    CODE_REMOVAL = auto()         # 代码删除
    CODE_ADDITION = auto()        # 代码添加
    CODE_MODIFICATION = auto()    # 代码修改


@dataclass
class CompilationError:
    """编译错误信息"""
    error_id: str
    error_type: str
    message: str
    file_path: str
    line_number: Optional[int]
    column_number: Optional[int]
    category: ErrorCategory
    severity: ErrorSeverity
    error_code: Optional[str] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestFailure:
    """测试失败信息"""
    failure_id: str
    test_class: str
    test_method: str
    error_type: str
    message: str
    stack_trace: str
    category: ErrorCategory
    severity: ErrorSeverity
    expected_value: Any = None
    actual_value: Any = None


@dataclass
class RootCause:
    """根本原因"""
    cause_id: str
    description: str
    confidence: float  # 0.0 - 1.0
    category: ErrorCategory
    location: Optional[str] = None
    contributing_factors: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)


@dataclass
class FixStrategy:
    """修复策略"""
    strategy_id: str
    strategy_type: FixStrategyType
    description: str
    priority: int  # 1-5, 1 最高
    estimated_effort: str  # "low", "medium", "high"
    code_changes: List[str] = field(default_factory=list)
    success_probability: float = 0.0
    side_effects: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)


@dataclass
class RootCauseAnalysis:
    """根因分析结果"""
    errors: List[Any]  # CompilationError or TestFailure
    root_causes: List[RootCause]
    suggested_fixes: List[FixStrategy]
    analysis_summary: str
    confidence_score: float
    requires_manual_review: bool = False


class RootCauseAnalyzer:
    """错误根因分析器
    
    功能:
    - 分析编译错误
    - 分析测试失败
    - 识别根本原因
    - 推荐修复策略
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.error_patterns = self._load_error_patterns()
        self.fix_strategies = self._load_fix_strategies()
        self.analysis_history: List[RootCauseAnalysis] = []
        self.cache = get_global_cache()
        
    def _load_error_patterns(self) -> Dict[ErrorCategory, List[str]]:
        """加载错误模式"""
        return {
            ErrorCategory.SYNTAX_ERROR: [
                r".*syntax error.*",
                r".*';' expected.*",
                r".*'}' expected.*",
                r".*unexpected token.*",
                r".*illegal start of.*",
            ],
            ErrorCategory.TYPE_ERROR: [
                r".*incompatible types.*",
                r".*cannot convert.*",
                r".*required.*found.*",
                r".*type mismatch.*",
            ],
            ErrorCategory.IMPORT_ERROR: [
                r".*cannot find symbol.*",
                r".*package .* does not exist.*",
                r".*cannot access.*",
                r".*class file for .* not found.*",
            ],
            ErrorCategory.REFERENCE_ERROR: [
                r".*cannot find symbol.*",
                r".*variable .* not found.*",
                r".*method .* not found.*",
            ],
            ErrorCategory.ACCESS_ERROR: [
                r".*has private access in.*",
                r".*is not public.*",
                r".*cannot be accessed from outside.*",
            ],
            ErrorCategory.OVERRIDE_ERROR: [
                r".*does not override.*",
                r".*method .* overrides.*",
                r".*cannot override.*",
            ],
            ErrorCategory.INITIALIZATION_ERROR: [
                r".*variable .* might not have been initialized.*",
                r".*recursive constructor invocation.*",
            ],
            ErrorCategory.RESOURCE_ERROR: [
                r".*resource .* never closed.*",
                r".*unclosed resource.*",
            ],
            ErrorCategory.ASSERTION_ERROR: [
                r".*assertion failed.*",
                r".*expected.*but was.*",
                r".*AssertionError.*",
            ],
            ErrorCategory.MOCK_ERROR: [
                r".*MissingMethodCallException.*",
                r".*UnnecessaryStubbingException.*",
                r".*StrictStubbingException.*",
            ],
            ErrorCategory.NULL_POINTER_ERROR: [
                r".*NullPointerException.*",
                r".*null.*",
            ],
        }
    
    def _load_fix_strategies(self) -> Dict[ErrorCategory, List[FixStrategyType]]:
        """加载修复策略"""
        return {
            ErrorCategory.SYNTAX_ERROR: [
                FixStrategyType.SYNTAX_FIX,
                FixStrategyType.CODE_MODIFICATION,
            ],
            ErrorCategory.TYPE_ERROR: [
                FixStrategyType.TYPE_CORRECTION,
                FixStrategyType.CODE_MODIFICATION,
            ],
            ErrorCategory.IMPORT_ERROR: [
                FixStrategyType.IMPORT_ADDITION,
                FixStrategyType.CODE_ADDITION,
            ],
            ErrorCategory.REFERENCE_ERROR: [
                FixStrategyType.CODE_ADDITION,
                FixStrategyType.CODE_MODIFICATION,
            ],
            ErrorCategory.ACCESS_ERROR: [
                FixStrategyType.VISIBILITY_CHANGE,
                FixStrategyType.CODE_MODIFICATION,
            ],
            ErrorCategory.OVERRIDE_ERROR: [
                FixStrategyType.SIGNATURE_UPDATE,
                FixStrategyType.CODE_MODIFICATION,
            ],
            ErrorCategory.INITIALIZATION_ERROR: [
                FixStrategyType.INITIALIZATION_FIX,
                FixStrategyType.CODE_ADDITION,
            ],
            ErrorCategory.RESOURCE_ERROR: [
                FixStrategyType.RESOURCE_MANAGEMENT,
                FixStrategyType.CODE_ADDITION,
            ],
            ErrorCategory.ASSERTION_ERROR: [
                FixStrategyType.ASSERTION_FIX,
                FixStrategyType.LOGIC_CORRECTION,
            ],
            ErrorCategory.MOCK_ERROR: [
                FixStrategyType.MOCK_SETUP,
                FixStrategyType.CODE_MODIFICATION,
            ],
            ErrorCategory.NULL_POINTER_ERROR: [
                FixStrategyType.NULL_CHECK,
                FixStrategyType.CODE_ADDITION,
            ],
        }
    
    def analyze_compilation_errors(
        self,
        compiler_output: str,
        source_code: Optional[str] = None
    ) -> RootCauseAnalysis:
        """分析编译错误
        
        Args:
            compiler_output: 编译器输出
            source_code: 可选的源代码
            
        Returns:
            根因分析结果
        """
        self.logger.info("[RootCauseAnalyzer] Analyzing compilation errors")
        
        # 生成缓存键
        cache_key = self.cache.generate_key("compilation_analysis", compiler_output)
        
        # 尝试从缓存获取
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            self.logger.info("[RootCauseAnalyzer] Cache hit for compilation analysis")
            return self._dict_to_analysis(cached_result)
        
        # 1. 解析编译错误
        errors = self._parse_compilation_errors(compiler_output)
        
        if not errors:
            return RootCauseAnalysis(
                errors=[],
                root_causes=[],
                suggested_fixes=[],
                analysis_summary="No compilation errors detected",
                confidence_score=1.0
            )
        
        # 2. 识别根本原因
        root_causes = self._identify_root_causes(errors, source_code)
        
        # 3. 生成修复策略
        suggested_fixes = self._generate_fix_strategies(root_causes, source_code)
        
        # 4. 计算置信度
        confidence_score = self._calculate_confidence(root_causes)
        
        # 5. 生成摘要
        summary = self._generate_summary(errors, root_causes, suggested_fixes)
        
        # 6. 判断是否需要人工审查
        requires_review = self._requires_manual_review(root_causes, confidence_score)
        
        analysis = RootCauseAnalysis(
            errors=errors,
            root_causes=root_causes,
            suggested_fixes=suggested_fixes,
            analysis_summary=summary,
            confidence_score=confidence_score,
            requires_manual_review=requires_review
        )
        
        # 缓存结果
        self.cache.set(cache_key, self._analysis_to_dict(analysis), ttl_l1=1800, ttl_l2=43200)
        
        self.analysis_history.append(analysis)
        self.logger.info(f"[RootCauseAnalyzer] Analysis complete - "
                        f"Errors: {len(errors)}, "
                        f"Root Causes: {len(root_causes)}, "
                        f"Confidence: {confidence_score:.2f}")
        
        return analysis
    
    def analyze_test_failures(
        self,
        test_output: str,
        test_code: Optional[str] = None,
        source_code: Optional[str] = None
    ) -> RootCauseAnalysis:
        """分析测试失败
        
        Args:
            test_output: 测试输出
            test_code: 可选的测试代码
            source_code: 可选的源代码
            
        Returns:
            根因分析结果
        """
        self.logger.info("[RootCauseAnalyzer] Analyzing test failures")
        
        # 生成缓存键
        cache_key = self.cache.generate_key("test_failure_analysis", test_output)
        
        # 尝试从缓存获取
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            self.logger.info("[RootCauseAnalyzer] Cache hit for test failure analysis")
            return self._dict_to_analysis(cached_result)
        
        # 1. 解析测试失败
        failures = self._parse_test_failures(test_output)
        
        if not failures:
            return RootCauseAnalysis(
                errors=[],
                root_causes=[],
                suggested_fixes=[],
                analysis_summary="No test failures detected",
                confidence_score=1.0
            )
        
        # 2. 识别根本原因
        root_causes = self._identify_test_failure_root_causes(failures, test_code, source_code)
        
        # 3. 生成修复策略
        suggested_fixes = self._generate_test_fix_strategies(root_causes, test_code, source_code)
        
        # 4. 计算置信度
        confidence_score = self._calculate_confidence(root_causes)
        
        # 5. 生成摘要
        summary = self._generate_test_summary(failures, root_causes, suggested_fixes)
        
        # 6. 判断是否需要人工审查
        requires_review = self._requires_manual_review(root_causes, confidence_score)
        
        analysis = RootCauseAnalysis(
            errors=failures,
            root_causes=root_causes,
            suggested_fixes=suggested_fixes,
            analysis_summary=summary,
            confidence_score=confidence_score,
            requires_manual_review=requires_review
        )
        
        # 缓存结果
        self.cache.set(cache_key, self._analysis_to_dict(analysis), ttl_l1=1800, ttl_l2=43200)
        
        self.analysis_history.append(analysis)
        self.logger.info(f"[RootCauseAnalyzer] Analysis complete - "
                        f"Failures: {len(failures)}, "
                        f"Root Causes: {len(root_causes)}, "
                        f"Confidence: {confidence_score:.2f}")
        
        return analysis
    
    def _parse_compilation_errors(self, output: str) -> List[CompilationError]:
        """解析编译错误"""
        errors = []
        error_counter = 0
        
        # Maven/Gradle 编译错误模式
        # 例如：[ERROR] /path/to/File.java:[10,5] error: ';' expected
        maven_pattern = r'\[ERROR\]\s+(.*?\.java):(\d+):(\d+):\s*(.*)'
        
        # javac 错误模式
        # 例如：File.java:10:5: error: ';' expected
        javac_pattern = r'(.*?\.java):(\d+):(\d+):\s*error:\s*(.*)'
        
        # 简单错误模式 (没有行号)
        simple_pattern = r'error:\s*(.*)'
        
        for line in output.split('\n'):
            error_counter += 1
            
            # 尝试匹配 Maven 格式
            match = re.search(maven_pattern, line)
            if match:
                file_path, line_num, col_num, message = match.groups()
                category = self._categorize_error(message)
                severity = self._determine_severity(category, message)
                
                error = CompilationError(
                    error_id=f"comp_{error_counter:03d}",
                    error_type="compilation",
                    message=message.strip(),
                    file_path=file_path,
                    line_number=int(line_num),
                    column_number=int(col_num),
                    category=category,
                    severity=severity
                )
                errors.append(error)
                continue
            
            # 尝试匹配 javac 格式
            match = re.search(javac_pattern, line)
            if match:
                file_path, line_num, col_num, message = match.groups()
                category = self._categorize_error(message)
                severity = self._determine_severity(category, message)
                
                error = CompilationError(
                    error_id=f"comp_{error_counter:03d}",
                    error_type="compilation",
                    message=message.strip(),
                    file_path=file_path,
                    line_number=int(line_num),
                    column_number=int(col_num),
                    category=category,
                    severity=severity
                )
                errors.append(error)
                continue
            
            # 尝试匹配简单格式
            match = re.search(simple_pattern, line, re.IGNORECASE)
            if match and 'warning' not in line.lower():
                message = match.group(1)
                category = self._categorize_error(message)
                severity = self._determine_severity(category, message)
                
                error = CompilationError(
                    error_id=f"comp_{error_counter:03d}",
                    error_type="compilation",
                    message=message.strip(),
                    file_path="unknown",
                    line_number=None,
                    column_number=None,
                    category=category,
                    severity=severity
                )
                errors.append(error)
        
        return errors
    
    def _parse_test_failures(self, output: str) -> List[TestFailure]:
        """解析测试失败"""
        failures = []
        failure_counter = 0
        
        # JUnit 测试失败模式
        # 例如：Test class::testMethod 或 TestClass.testMethod
        test_pattern = r'([A-Z]\w*Test)[\.:](\w+)'
        
        # 断言失败模式
        # 例如：Expected: <5> but was: <3>
        assertion_pattern = r'Expected:\s*<?([^>]+)>?\s*but was:\s*<?([^>]+)>?'
        
        # 异常模式
        # 例如：thrown (expected )?(\w+Exception|\w+Error) 或 (\w+Exception) was thrown 或 NullPointerException
        exception_pattern = r'(\w+Exception|\w+Error)(?:\s+was\s+thrown)?'
        
        current_test_class = None
        current_test_method = None
        
        lines = output.split('\n')
        for i, line in enumerate(lines):
            # 检测测试类和方法
            match = re.search(test_pattern, line)
            if match:
                current_test_class = match.group(1)
                current_test_method = match.group(2)
            
            # 检测断言失败
            match = re.search(assertion_pattern, line)
            if match and current_test_class and current_test_method:
                failure_counter += 1
                expected, actual = match.groups()
                
                failure = TestFailure(
                    failure_id=f"fail_{failure_counter:03d}",
                    test_class=current_test_class,
                    test_method=current_test_method,
                    error_type="AssertionError",
                    message=f"Expected {expected} but was {actual}",
                    stack_trace=line,
                    category=ErrorCategory.ASSERTION_ERROR,
                    severity=ErrorSeverity.HIGH,
                    expected_value=expected,
                    actual_value=actual
                )
                failures.append(failure)
                continue
            
            # 检测异常
            match = re.search(exception_pattern, line, re.IGNORECASE)
            if match and current_test_class and current_test_method:
                failure_counter += 1
                exception_type = match.group(1)
                
                category = self._categorize_test_exception(exception_type)
                
                failure = TestFailure(
                    failure_id=f"fail_{failure_counter:03d}",
                    test_class=current_test_class,
                    test_method=current_test_method,
                    error_type=exception_type,
                    message=line.strip(),
                    stack_trace=line,
                    category=category,
                    severity=ErrorSeverity.HIGH
                )
                failures.append(failure)
        
        return failures
    
    def _categorize_error(self, message: str) -> ErrorCategory:
        """将错误分类"""
        message_lower = message.lower()
        
        for category, patterns in self.error_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    return category
        
        # 默认类别
        if 'cannot find symbol' in message_lower:
            return ErrorCategory.REFERENCE_ERROR
        elif 'incompatible types' in message_lower:
            return ErrorCategory.TYPE_ERROR
        elif 'expected' in message_lower and 'found' in message_lower:
            return ErrorCategory.TYPE_ERROR
        elif 'has private access' in message_lower or 'is not public' in message_lower:
            return ErrorCategory.ACCESS_ERROR
        
        return ErrorCategory.SYNTAX_ERROR
    
    def _categorize_test_exception(self, exception_type: str) -> ErrorCategory:
        """将测试异常分类"""
        exception_lower = exception_type.lower()
        
        if 'nullpointer' in exception_lower:
            return ErrorCategory.NULL_POINTER_ERROR
        elif 'assertion' in exception_lower:
            return ErrorCategory.ASSERTION_ERROR
        elif 'mockito' in exception_lower or 'missingmethod' in exception_lower:
            return ErrorCategory.MOCK_ERROR
        elif 'illegalargument' in exception_lower:
            return ErrorCategory.LOGIC_ERROR
        elif 'illegalstate' in exception_lower:
            return ErrorCategory.LOGIC_ERROR
        
        return ErrorCategory.LOGIC_ERROR
    
    def _determine_severity(self, category: ErrorCategory, message: str) -> ErrorSeverity:
        """确定错误严重程度"""
        # 关键错误
        if category in [ErrorCategory.SYNTAX_ERROR, ErrorCategory.TYPE_ERROR]:
            return ErrorSeverity.CRITICAL
        
        # 高优先级错误
        if category in [ErrorCategory.REFERENCE_ERROR, ErrorCategory.IMPORT_ERROR]:
            return ErrorSeverity.HIGH
        
        # 中等优先级
        if category in [ErrorCategory.ACCESS_ERROR, ErrorCategory.OVERRIDE_ERROR]:
            return ErrorSeverity.MEDIUM
        
        # 低优先级
        return ErrorSeverity.LOW
    
    def _identify_root_causes(
        self,
        errors: List[CompilationError],
        source_code: Optional[str] = None
    ) -> List[RootCause]:
        """识别根本原因"""
        root_causes = []
        cause_counter = 0
        
        # 按类别分组错误
        errors_by_category = defaultdict(list)
        for error in errors:
            errors_by_category[error.category].append(error)
        
        # 为每个类别识别根本原因
        for category, category_errors in errors_by_category.items():
            cause_counter += 1
            
            # 分析错误模式
            contributing_factors = self._analyze_error_pattern(category, category_errors)
            evidence = [e.message for e in category_errors[:3]]  # 前 3 个错误作为证据
            
            # 确定位置
            locations = set()
            for error in category_errors:
                if error.file_path and error.line_number:
                    locations.add(f"{error.file_path}:{error.line_number}")
            
            location = ", ".join(list(locations)[:3]) if locations else None
            
            # 计算置信度
            confidence = min(0.95, 0.6 + (len(category_errors) * 0.05))
            
            root_cause = RootCause(
                cause_id=f"cause_{cause_counter:03d}",
                description=self._describe_root_cause(category),
                confidence=confidence,
                category=category,
                location=location,
                contributing_factors=contributing_factors,
                evidence=evidence
            )
            root_causes.append(root_cause)
        
        return root_causes
    
    def _identify_test_failure_root_causes(
        self,
        failures: List[TestFailure],
        test_code: Optional[str] = None,
        source_code: Optional[str] = None
    ) -> List[RootCause]:
        """识别测试失败的根本原因"""
        root_causes = []
        cause_counter = 0
        
        # 按类别分组失败
        failures_by_category = defaultdict(list)
        for failure in failures:
            failures_by_category[failure.category].append(failure)
        
        # 为每个类别识别根本原因
        for category, category_failures in failures_by_category.items():
            cause_counter += 1
            
            # 分析失败模式
            contributing_factors = self._analyze_failure_pattern(category, category_failures)
            evidence = [f.message for f in category_failures[:3]]
            
            # 确定位置
            locations = [f"{f.test_class}::{f.test_method}" for f in category_failures[:3]]
            location = ", ".join(locations) if locations else None
            
            # 计算置信度
            confidence = min(0.95, 0.6 + (len(category_failures) * 0.05))
            
            root_cause = RootCause(
                cause_id=f"cause_{cause_counter:03d}",
                description=self._describe_test_root_cause(category),
                confidence=confidence,
                category=category,
                location=location,
                contributing_factors=contributing_factors,
                evidence=evidence
            )
            root_causes.append(root_cause)
        
        return root_causes
    
    def _analyze_error_pattern(
        self,
        category: ErrorCategory,
        errors: List[CompilationError]
    ) -> List[str]:
        """分析错误模式"""
        factors = []
        
        if category == ErrorCategory.SYNTAX_ERROR:
            if any(';' in e.message for e in errors):
                factors.append("Missing semicolons")
            if any('}' in e.message for e in errors):
                factors.append("Unbalanced braces")
            if any('unexpected' in e.message.lower() for e in errors):
                factors.append("Unexpected tokens or syntax")
        
        elif category == ErrorCategory.TYPE_ERROR:
            if any('incompatible types' in e.message.lower() for e in errors):
                factors.append("Type mismatches in assignments or returns")
            if any('cannot convert' in e.message.lower() for e in errors):
                factors.append("Incompatible type conversions")
        
        elif category == ErrorCategory.REFERENCE_ERROR:
            if any('cannot find symbol' in e.message.lower() for e in errors):
                factors.append("Missing declarations or imports")
        
        if not factors:
            factors.append(f"Multiple {category.name.replace('_', ' ').lower()} detected")
        
        return factors
    
    def _analyze_failure_pattern(
        self,
        category: ErrorCategory,
        failures: List[TestFailure]
    ) -> List[str]:
        """分析失败模式"""
        factors = []
        
        if category == ErrorCategory.ASSERTION_ERROR:
            if all(f.expected_value is not None for f in failures):
                factors.append("Incorrect expected values in assertions")
            factors.append("Logic errors in test or implementation")
        
        elif category == ErrorCategory.NULL_POINTER_ERROR:
            factors.append("Missing null checks")
            factors.append("Uninitialized objects or mock setup issues")
        
        elif category == ErrorCategory.MOCK_ERROR:
            factors.append("Incorrect Mockito usage")
            factors.append("Missing or unnecessary stubbings")
        
        if not factors:
            factors.append(f"Multiple {category.name.replace('_', ' ').lower()} detected")
        
        return factors
    
    def _describe_root_cause(self, category: ErrorCategory) -> str:
        """描述根本原因"""
        descriptions = {
            ErrorCategory.SYNTAX_ERROR: "Syntax errors in code structure",
            ErrorCategory.TYPE_ERROR: "Type incompatibilities or mismatches",
            ErrorCategory.IMPORT_ERROR: "Missing or incorrect imports",
            ErrorCategory.REFERENCE_ERROR: "Undefined variables, methods, or classes",
            ErrorCategory.ACCESS_ERROR: "Visibility or access restriction violations",
            ErrorCategory.OVERRIDE_ERROR: "Incorrect method override signatures",
            ErrorCategory.INITIALIZATION_ERROR: "Uninitialized variables or constructors",
            ErrorCategory.RESOURCE_ERROR: "Resource management issues",
            ErrorCategory.ASSERTION_ERROR: "Assertion failures or incorrect expectations",
            ErrorCategory.MOCK_ERROR: "Mock setup or usage errors",
            ErrorCategory.NULL_POINTER_ERROR: "Null pointer dereferences",
            ErrorCategory.LOGIC_ERROR: "Logical errors in code flow",
        }
        return descriptions.get(category, "Unknown error cause")
    
    def _describe_test_root_cause(self, category: ErrorCategory) -> str:
        """描述测试失败根本原因"""
        descriptions = {
            ErrorCategory.ASSERTION_ERROR: "Test assertions do not match actual behavior",
            ErrorCategory.NULL_POINTER_ERROR: "Null references in test or target code",
            ErrorCategory.MOCK_ERROR: "Incorrect mock configuration or usage",
            ErrorCategory.LOGIC_ERROR: "Logic errors causing unexpected behavior",
        }
        return descriptions.get(category, "Test failure due to unexpected behavior")
    
    def _generate_fix_strategies(
        self,
        root_causes: List[RootCause],
        source_code: Optional[str] = None
    ) -> List[FixStrategy]:
        """生成修复策略"""
        strategies = []
        strategy_counter = 0
        
        for cause in root_causes:
            # 获取适用的修复策略类型
            applicable_types = self.fix_strategies.get(cause.category, [])
            
            for strategy_type in applicable_types:
                strategy_counter += 1
                
                strategy = FixStrategy(
                    strategy_id=f"fix_{strategy_counter:03d}",
                    strategy_type=strategy_type,
                    description=self._describe_fix_strategy(strategy_type, cause),
                    priority=self._calculate_fix_priority(cause),
                    estimated_effort=self._estimate_effort(strategy_type),
                    code_changes=self._suggest_code_changes(strategy_type, cause, source_code),
                    success_probability=self._estimate_success_probability(strategy_type, cause),
                    side_effects=self._identify_side_effects(strategy_type)
                )
                strategies.append(strategy)
        
        # 按优先级排序
        strategies.sort(key=lambda s: s.priority)
        
        return strategies
    
    def _generate_test_fix_strategies(
        self,
        root_causes: List[RootCause],
        test_code: Optional[str] = None,
        source_code: Optional[str] = None
    ) -> List[FixStrategy]:
        """生成测试失败修复策略"""
        strategies = []
        strategy_counter = 0
        
        for cause in root_causes:
            # 获取适用的修复策略类型
            applicable_types = self.fix_strategies.get(cause.category, [])
            
            for strategy_type in applicable_types:
                strategy_counter += 1
                
                strategy = FixStrategy(
                    strategy_id=f"fix_{strategy_counter:03d}",
                    strategy_type=strategy_type,
                    description=self._describe_test_fix_strategy(strategy_type, cause),
                    priority=self._calculate_fix_priority(cause),
                    estimated_effort=self._estimate_effort(strategy_type),
                    code_changes=self._suggest_test_code_changes(strategy_type, cause, test_code, source_code),
                    success_probability=self._estimate_success_probability(strategy_type, cause),
                    side_effects=self._identify_side_effects(strategy_type)
                )
                strategies.append(strategy)
        
        # 按优先级排序
        strategies.sort(key=lambda s: s.priority)
        
        return strategies
    
    def _describe_fix_strategy(self, strategy_type: FixStrategyType, cause: RootCause) -> str:
        """描述修复策略"""
        descriptions = {
            FixStrategyType.SYNTAX_FIX: f"Fix syntax errors: {cause.description}",
            FixStrategyType.TYPE_CORRECTION: f"Correct type mismatches: {cause.description}",
            FixStrategyType.IMPORT_ADDITION: f"Add missing imports: {cause.description}",
            FixStrategyType.VISIBILITY_CHANGE: f"Modify visibility: {cause.description}",
            FixStrategyType.SIGNATURE_UPDATE: f"Update method signature: {cause.description}",
            FixStrategyType.INITIALIZATION_FIX: f"Fix initialization: {cause.description}",
            FixStrategyType.RESOURCE_MANAGEMENT: f"Improve resource management: {cause.description}",
            FixStrategyType.ASSERTION_FIX: f"Fix assertions: {cause.description}",
            FixStrategyType.MOCK_SETUP: f"Configure mocks properly: {cause.description}",
            FixStrategyType.NULL_CHECK: f"Add null checks: {cause.description}",
            FixStrategyType.LOGIC_CORRECTION: f"Correct logic: {cause.description}",
            FixStrategyType.CODE_REMOVAL: "Remove problematic code",
            FixStrategyType.CODE_ADDITION: "Add missing code",
            FixStrategyType.CODE_MODIFICATION: "Modify existing code",
        }
        return descriptions.get(strategy_type, "Apply appropriate fix")
    
    def _describe_test_fix_strategy(self, strategy_type: FixStrategyType, cause: RootCause) -> str:
        """描述测试修复策略"""
        descriptions = {
            FixStrategyType.ASSERTION_FIX: f"Update test assertions: {cause.description}",
            FixStrategyType.MOCK_SETUP: f"Fix mock configuration: {cause.description}",
            FixStrategyType.NULL_CHECK: f"Add null checks in test: {cause.description}",
            FixStrategyType.LOGIC_CORRECTION: f"Correct test logic: {cause.description}",
        }
        return descriptions.get(strategy_type, "Apply test-specific fix")
    
    def _calculate_fix_priority(self, cause: RootCause) -> int:
        """计算修复优先级"""
        # 基于置信度和严重程度
        if cause.confidence > 0.8:
            return 1
        elif cause.confidence > 0.6:
            return 2
        elif cause.confidence > 0.4:
            return 3
        else:
            return 4
    
    def _estimate_effort(self, strategy_type: FixStrategyType) -> str:
        """估算修复工作量"""
        effort_map = {
            FixStrategyType.SYNTAX_FIX: "low",
            FixStrategyType.IMPORT_ADDITION: "low",
            FixStrategyType.CODE_ADDITION: "medium",
            FixStrategyType.CODE_MODIFICATION: "medium",
            FixStrategyType.TYPE_CORRECTION: "medium",
            FixStrategyType.SIGNATURE_UPDATE: "medium",
            FixStrategyType.VISIBILITY_CHANGE: "low",
            FixStrategyType.INITIALIZATION_FIX: "medium",
            FixStrategyType.RESOURCE_MANAGEMENT: "medium",
            FixStrategyType.ASSERTION_FIX: "low",
            FixStrategyType.MOCK_SETUP: "low",
            FixStrategyType.NULL_CHECK: "low",
            FixStrategyType.LOGIC_CORRECTION: "high",
            FixStrategyType.CODE_REMOVAL: "low",
        }
        return effort_map.get(strategy_type, "medium")
    
    def _suggest_code_changes(
        self,
        strategy_type: FixStrategyType,
        cause: RootCause,
        source_code: Optional[str] = None
    ) -> List[str]:
        """建议代码修改"""
        changes = []
        
        if strategy_type == FixStrategyType.SYNTAX_FIX:
            changes.append("Add missing semicolons or braces")
            changes.append("Fix syntax structure")
        
        elif strategy_type == FixStrategyType.IMPORT_ADDITION:
            changes.append("Add required import statements")
        
        elif strategy_type == FixStrategyType.TYPE_CORRECTION:
            changes.append("Cast to correct type or change variable type")
        
        elif strategy_type == FixStrategyType.NULL_CHECK:
            changes.append("Add null check before accessing object")
        
        elif strategy_type == FixStrategyType.ASSERTION_FIX:
            changes.append("Update assertion expectations")
        
        elif strategy_type == FixStrategyType.MOCK_SETUP:
            changes.append("Configure mock behavior with when().thenReturn()")
        
        if not changes:
            changes.append("Apply appropriate code modification")
        
        return changes
    
    def _suggest_test_code_changes(
        self,
        strategy_type: FixStrategyType,
        cause: RootCause,
        test_code: Optional[str] = None,
        source_code: Optional[str] = None
    ) -> List[str]:
        """建议测试代码修改"""
        changes = []
        
        if strategy_type == FixStrategyType.ASSERTION_FIX:
            changes.append("Update assertion to match actual behavior")
            changes.append("Verify expected values are correct")
        
        elif strategy_type == FixStrategyType.MOCK_SETUP:
            changes.append("Add missing mock stubbings")
            changes.append("Remove unnecessary stubbings")
        
        elif strategy_type == FixStrategyType.NULL_CHECK:
            changes.append("Initialize mock objects properly")
            changes.append("Add null checks before assertions")
        
        if not changes:
            changes.append("Modify test code to fix failure")
        
        return changes
    
    def _estimate_success_probability(
        self,
        strategy_type: FixStrategyType,
        cause: RootCause
    ) -> float:
        """估算成功概率"""
        base_probabilities = {
            FixStrategyType.SYNTAX_FIX: 0.95,
            FixStrategyType.IMPORT_ADDITION: 0.90,
            FixStrategyType.CODE_ADDITION: 0.80,
            FixStrategyType.CODE_MODIFICATION: 0.75,
            FixStrategyType.TYPE_CORRECTION: 0.85,
            FixStrategyType.SIGNATURE_UPDATE: 0.85,
            FixStrategyType.VISIBILITY_CHANGE: 0.80,
            FixStrategyType.INITIALIZATION_FIX: 0.85,
            FixStrategyType.RESOURCE_MANAGEMENT: 0.80,
            FixStrategyType.ASSERTION_FIX: 0.90,
            FixStrategyType.MOCK_SETUP: 0.85,
            FixStrategyType.NULL_CHECK: 0.90,
            FixStrategyType.LOGIC_CORRECTION: 0.70,
            FixStrategyType.CODE_REMOVAL: 0.75,
        }
        
        base_prob = base_probabilities.get(strategy_type, 0.75)
        
        # 根据置信度调整
        adjusted_prob = base_prob * cause.confidence
        
        return min(0.95, adjusted_prob)
    
    def _identify_side_effects(self, strategy_type: FixStrategyType) -> List[str]:
        """识别副作用"""
        side_effects = {
            FixStrategyType.VISIBILITY_CHANGE: ["May expose internal implementation"],
            FixStrategyType.CODE_REMOVAL: ["May remove necessary functionality"],
            FixStrategyType.TYPE_CORRECTION: ["May affect other type usages"],
            FixStrategyType.SIGNATURE_UPDATE: ["May break existing callers"],
        }
        return side_effects.get(strategy_type, [])
    
    def _calculate_confidence(self, root_causes: List[RootCause]) -> float:
        """计算总体置信度"""
        if not root_causes:
            return 1.0
        
        # 加权平均
        total_confidence = sum(rc.confidence for rc in root_causes)
        return total_confidence / len(root_causes)
    
    def _generate_summary(
        self,
        errors: List[CompilationError],
        root_causes: List[RootCause],
        suggested_fixes: List[FixStrategy]
    ) -> str:
        """生成分析摘要"""
        summary_parts = []
        
        # 错误统计
        summary_parts.append(f"Found {len(errors)} compilation error(s)")
        
        # 根本原因
        if root_causes:
            categories = set(rc.category for rc in root_causes)
            summary_parts.append(f"Identified {len(root_causes)} root cause(s): "
                               f"{', '.join(c.name.replace('_', ' ') for c in categories)}")
        
        # 修复建议
        if suggested_fixes:
            summary_parts.append(f"Suggested {len(suggested_fixes)} fix strategy(ies)")
            top_fix = suggested_fixes[0]
            summary_parts.append(f"Recommended: {top_fix.description} "
                               f"(Priority: {top_fix.priority}, "
                               f"Effort: {top_fix.estimated_effort})")
        
        return ". ".join(summary_parts)
    
    def _generate_test_summary(
        self,
        failures: List[TestFailure],
        root_causes: List[RootCause],
        suggested_fixes: List[FixStrategy]
    ) -> str:
        """生成测试失败分析摘要"""
        summary_parts = []
        
        # 失败统计
        summary_parts.append(f"Found {len(failures)} test failure(s)")
        
        # 根本原因
        if root_causes:
            categories = set(rc.category for rc in root_causes)
            summary_parts.append(f"Identified {len(root_causes)} root cause(s): "
                               f"{', '.join(c.name.replace('_', ' ') for c in categories)}")
        
        # 修复建议
        if suggested_fixes:
            summary_parts.append(f"Suggested {len(suggested_fixes)} fix strategy(ies)")
            top_fix = suggested_fixes[0]
            summary_parts.append(f"Recommended: {top_fix.description} "
                               f"(Priority: {top_fix.priority}, "
                               f"Effort: {top_fix.estimated_effort})")
        
        return ". ".join(summary_parts)
    
    def _requires_manual_review(self, root_causes: List[RootCause], confidence: float) -> bool:
        """判断是否需要人工审查"""
        # 置信度低需要人工审查
        if confidence < 0.5:
            return True
        
        # 有多个不确定的根本原因需要人工审查
        if len(root_causes) > 3:
            return True
        
        # 有低置信度的根本原因
        if any(rc.confidence < 0.4 for rc in root_causes):
            return True
        
        return False
    
    def get_analysis_history(self) -> List[RootCauseAnalysis]:
        """获取分析历史"""
        return self.analysis_history
    
    def clear_history(self):
        """清除分析历史"""
        self.analysis_history.clear()
        self.logger.info("[RootCauseAnalyzer] Analysis history cleared")
    
    def _analysis_to_dict(self, analysis: RootCauseAnalysis) -> Dict[str, Any]:
        """将 RootCauseAnalysis 转换为可 JSON 序列化的字典"""
        errors_list = []
        for error in analysis.errors:
            if hasattr(error, '__dict__'):
                error_dict = {
                    k: (v.value if isinstance(v, Enum) else v)
                    for k, v in error.__dict__.items()
                }
                errors_list.append(error_dict)
            else:
                errors_list.append(error)
        
        root_causes_list = []
        for rc in analysis.root_causes:
            rc_dict = {
                k: (v.value if isinstance(v, Enum) else v)
                for k, v in rc.__dict__.items()
            }
            root_causes_list.append(rc_dict)
        
        suggested_fixes_list = []
        for fix in analysis.suggested_fixes:
            fix_dict = {
                k: (v.value if isinstance(v, Enum) else v)
                for k, v in fix.__dict__.items()
            }
            suggested_fixes_list.append(fix_dict)
        
        return {
            "errors": errors_list,
            "root_causes": root_causes_list,
            "suggested_fixes": suggested_fixes_list,
            "analysis_summary": analysis.analysis_summary,
            "confidence_score": analysis.confidence_score,
            "requires_manual_review": analysis.requires_manual_review
        }
    
    def _dict_to_analysis(self, data: Dict[str, Any]) -> RootCauseAnalysis:
        """将字典转换回 RootCauseAnalysis"""
        errors = []
        for error_data in data.get("errors", []):
            if error_data.get("error_type") == "compilation":
                error = CompilationError(**error_data)
            else:
                error = TestFailure(**error_data)
            errors.append(error)
        
        root_causes = [RootCause(**rc_data) for rc_data in data.get("root_causes", [])]
        suggested_fixes = [FixStrategy(**fix_data) for fix_data in data.get("suggested_fixes", [])]
        
        return RootCauseAnalysis(
            errors=errors,
            root_causes=root_causes,
            suggested_fixes=suggested_fixes,
            analysis_summary=data.get("analysis_summary", ""),
            confidence_score=data.get("confidence_score", 0.0),
            requires_manual_review=data.get("requires_manual_review", False)
        )
