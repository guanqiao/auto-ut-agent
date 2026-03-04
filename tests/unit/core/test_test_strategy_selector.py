"""Unit tests for TestStrategySelector module."""

import pytest

from pyutagent.core.test_strategy_selector import (
    TestStrategySelector,
    TestStrategy,
    CodeCharacteristic,
    StrategyScore,
    CodeAnalysis,
    StrategyRecommendation,
)


class TestTestStrategySelector:
    """Tests for TestStrategySelector class."""

    def test_init(self):
        """Test initialization."""
        selector = TestStrategySelector()
        
        assert selector._strategy_weights is not None
        assert selector._characteristic_detectors is not None
        assert len(selector._strategy_weights) > 0

    def test_analyze_code_simple(self):
        """Test analyzing simple code."""
        selector = TestStrategySelector()
        
        source_code = '''
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
'''
        
        analysis = selector.analyze_code(source_code)
        
        assert isinstance(analysis, CodeAnalysis)
        assert CodeCharacteristic.IS_STATELESS in analysis.characteristics or \
               CodeCharacteristic.IS_PURE_FUNCTION in analysis.characteristics
        assert analysis.complexity_score >= 0.0

    def test_analyze_code_with_dependencies(self):
        """Test analyzing code with dependencies."""
        selector = TestStrategySelector()
        
        source_code = '''
@Service
public class UserService {
    @Autowired
    private UserRepository userRepo;
    
    public User getUser(Long id) {
        return userRepo.findById(id).orElse(null);
    }
}
'''
        
        analysis = selector.analyze_code(source_code)
        
        assert CodeCharacteristic.HAS_DEPENDENCIES in analysis.characteristics
        assert CodeCharacteristic.IS_SERVICE_CLASS in analysis.characteristics

    def test_analyze_code_with_exceptions(self):
        """Test analyzing code with exceptions."""
        selector = TestStrategySelector()
        
        source_code = '''
public class Validator {
    public void validate(String input) throws IllegalArgumentException {
        if (input == null) {
            throw new IllegalArgumentException("Input cannot be null");
        }
    }
}
'''
        
        analysis = selector.analyze_code(source_code)
        
        assert CodeCharacteristic.HAS_EXCEPTIONS in analysis.characteristics

    def test_analyze_code_with_conditionals(self):
        """Test analyzing code with conditionals."""
        selector = TestStrategySelector()
        
        source_code = '''
public class Processor {
    public String process(int value) {
        if (value > 0) {
            return "positive";
        } else if (value < 0) {
            return "negative";
        } else {
            return "zero";
        }
    }
}
'''
        
        analysis = selector.analyze_code(source_code)
        
        assert CodeCharacteristic.HAS_CONDITIONALS in analysis.characteristics

    def test_analyze_code_with_loops(self):
        """Test analyzing code with loops."""
        selector = TestStrategySelector()
        
        source_code = '''
public class SumCalculator {
    public int sum(List<Integer> numbers) {
        int total = 0;
        for (int n : numbers) {
            total += n;
        }
        return total;
    }
}
'''
        
        analysis = selector.analyze_code(source_code)
        
        assert CodeCharacteristic.HAS_LOOPS in analysis.characteristics
        assert CodeCharacteristic.HAS_COLLECTIONS in analysis.characteristics

    def test_analyze_code_with_validation(self):
        """Test analyzing code with validation."""
        selector = TestStrategySelector()
        
        source_code = '''
public class UserValidator {
    public boolean isValid(@NotNull User user) {
        if (user.getName() == null || user.getName().isEmpty()) {
            return false;
        }
        if (user.getAge() < 0 || user.getAge() > 150) {
            return false;
        }
        return true;
    }
}
'''
        
        analysis = selector.analyze_code(source_code)
        
        assert CodeCharacteristic.HAS_VALIDATION in analysis.characteristics
        assert CodeCharacteristic.HAS_BOUNDARY_CONDITIONS in analysis.characteristics

    def test_select_strategy_simple_code(self):
        """Test strategy selection for simple code."""
        selector = TestStrategySelector()
        
        source_code = '''
public class MathUtils {
    public static int square(int x) {
        return x * x;
    }
}
'''
        
        recommendation = selector.select_strategy(source_code)
        
        assert isinstance(recommendation, StrategyRecommendation)
        assert recommendation.primary_strategy in [TestStrategy.UNIT_BASIC, TestStrategy.PROPERTY_BASED]
        assert recommendation.confidence > 0.0

    def test_select_strategy_with_dependencies(self):
        """Test strategy selection for code with dependencies."""
        selector = TestStrategySelector()
        
        source_code = '''
@Service
public class OrderService {
    @Autowired
    private OrderRepository orderRepo;
    @Autowired
    private PaymentService paymentService;
    
    public Order createOrder(OrderRequest request) {
        Order order = new Order(request);
        return orderRepo.save(order);
    }
}
'''
        
        recommendation = selector.select_strategy(source_code)
        
        assert recommendation.primary_strategy == TestStrategy.UNIT_MOCK

    def test_select_strategy_with_exceptions(self):
        """Test strategy selection for code with exceptions."""
        selector = TestStrategySelector()
        
        source_code = '''
public class FileProcessor {
    public void process(String filename) throws IOException, FileNotFoundException {
        Files.readAllLines(Paths.get(filename));
    }
}
'''
        
        recommendation = selector.select_strategy(source_code)
        
        assert TestStrategy.EXCEPTION in recommendation.secondary_strategies or \
               recommendation.primary_strategy == TestStrategy.EXCEPTION

    def test_select_strategy_with_preferences(self):
        """Test strategy selection with user preferences."""
        selector = TestStrategySelector()
        
        source_code = '''
public class Calculator {
    public int add(int a, int b) { return a + b; }
}
'''
        
        preferences = {
            "preferred_strategies": ["unit_parameterized"],
            "avoided_strategies": ["integration"]
        }
        
        recommendation = selector.select_strategy(source_code, preferences=preferences)
        
        assert isinstance(recommendation, StrategyRecommendation)

    def test_calculate_complexity_simple(self):
        """Test complexity calculation for simple code."""
        selector = TestStrategySelector()
        
        simple_code = '''
public int add(int a, int b) {
    return a + b;
}
'''
        
        complexity = selector._calculate_complexity(simple_code)
        
        assert complexity < 0.3

    def test_calculate_complexity_complex(self):
        """Test complexity calculation for complex code."""
        selector = TestStrategySelector()
        
        complex_code = '''
public void processData(Data data) {
    if (data == null) {
        throw new IllegalArgumentException();
    }
    for (Item item : data.getItems()) {
        if (item.isActive()) {
            if (item.getValue() > 0 && item.getValue() < 100) {
                processItem(item);
            } else if (item.getValue() >= 100) {
                handleLargeItem(item);
            }
        }
    }
}
'''
        
        complexity = selector._calculate_complexity(complex_code)
        
        assert complexity > 0.3

    def test_detect_dependencies(self):
        """Test dependency detection."""
        selector = TestStrategySelector()
        
        code_with_deps = '''
@Service
public class Service {
    @Autowired private Repo repo;
    private Helper helper = new Helper();
}
'''
        
        has_deps = selector._detect_dependencies(code_with_deps, None)
        
        assert has_deps is True

    def test_detect_exceptions(self):
        """Test exception detection."""
        selector = TestStrategySelector()
        
        code_with_exceptions = '''
public void process() throws IOException {
    throw new RuntimeException();
}
'''
        
        has_exceptions = selector._detect_exceptions(code_with_exceptions, None)
        
        assert has_exceptions is True

    def test_detect_collections(self):
        """Test collection detection."""
        selector = TestStrategySelector()
        
        code_with_collections = '''
public List<String> getNames() {
    return new ArrayList<>();
}
'''
        
        has_collections = selector._detect_collections(code_with_collections, None)
        
        assert has_collections is True

    def test_detect_service_class(self):
        """Test service class detection."""
        selector = TestStrategySelector()
        
        service_code = '''
@Service
public class UserService {
}
'''
        
        is_service = selector._detect_service_class(service_code, None)
        
        assert is_service is True

    def test_detect_utility_class(self):
        """Test utility class detection."""
        selector = TestStrategySelector()
        
        utility_code = '''
public class StringUtils {
    private StringUtils() {}
    
    public static boolean isEmpty(String s) {
        return s == null || s.isEmpty();
    }
}
'''
        
        is_utility = selector._detect_utility_class(utility_code, None)
        
        assert is_utility is True

    def test_get_applicable_patterns(self):
        """Test getting applicable patterns for strategies."""
        selector = TestStrategySelector()
        
        patterns = selector._get_applicable_patterns(TestStrategy.UNIT_MOCK)
        
        assert "mock_test" in patterns
        assert "mockito_pattern" in patterns

    def test_estimate_effort_low(self):
        """Test effort estimation for simple code."""
        selector = TestStrategySelector()
        
        analysis = CodeAnalysis(
            characteristics=[CodeCharacteristic.IS_PURE_FUNCTION],
            complexity_score=0.1,
            dependency_count=0,
            method_count=1,
            branch_count=0,
            exception_types=[],
            parameter_types=[],
            return_type="int",
            annotations=[],
            suggested_strategies=[TestStrategy.UNIT_BASIC]
        )
        
        effort = selector._estimate_effort(TestStrategy.UNIT_BASIC, analysis)
        
        assert effort == "low"

    def test_estimate_effort_high(self):
        """Test effort estimation for complex code."""
        selector = TestStrategySelector()
        
        analysis = CodeAnalysis(
            characteristics=[CodeCharacteristic.HAS_DEPENDENCIES, CodeCharacteristic.HAS_EXCEPTIONS],
            complexity_score=0.8,
            dependency_count=5,
            method_count=10,
            branch_count=8,
            exception_types=["IOException", "SQLException"],
            parameter_types=[],
            return_type="void",
            annotations=[],
            suggested_strategies=[TestStrategy.INTEGRATION]
        )
        
        effort = selector._estimate_effort(TestStrategy.INTEGRATION, analysis)
        
        assert effort in ["high", "very_high"]


class TestStrategyScore:
    """Tests for StrategyScore dataclass."""

    def test_strategy_score_creation(self):
        """Test strategy score creation."""
        score = StrategyScore(
            strategy=TestStrategy.UNIT_MOCK,
            score=0.85,
            reasons=["Has dependencies", "Service class"],
            applicable_patterns=["mock_test"],
            estimated_effort="medium"
        )
        
        assert score.strategy == TestStrategy.UNIT_MOCK
        assert score.score == 0.85
        assert len(score.reasons) == 2


class TestCodeAnalysis:
    """Tests for CodeAnalysis dataclass."""

    def test_code_analysis_creation(self):
        """Test code analysis creation."""
        analysis = CodeAnalysis(
            characteristics=[CodeCharacteristic.HAS_DEPENDENCIES],
            complexity_score=0.5,
            dependency_count=2,
            method_count=3,
            branch_count=1,
            exception_types=["IOException"],
            parameter_types=["String", "int"],
            return_type="void",
            annotations=["@Service"],
            suggested_strategies=[TestStrategy.UNIT_MOCK]
        )
        
        assert CodeCharacteristic.HAS_DEPENDENCIES in analysis.characteristics
        assert analysis.complexity_score == 0.5
        assert len(analysis.exception_types) == 1


class TestStrategyRecommendation:
    """Tests for StrategyRecommendation dataclass."""

    def test_strategy_recommendation_creation(self):
        """Test strategy recommendation creation."""
        recommendation = StrategyRecommendation(
            primary_strategy=TestStrategy.UNIT_MOCK,
            secondary_strategies=[TestStrategy.EXCEPTION, TestStrategy.BOUNDARY],
            analysis=CodeAnalysis(
                characteristics=[],
                complexity_score=0.5,
                dependency_count=1,
                method_count=1,
                branch_count=0,
                exception_types=[],
                parameter_types=[],
                return_type="void",
                annotations=[],
                suggested_strategies=[]
            ),
            scores=[
                StrategyScore(TestStrategy.UNIT_MOCK, 0.9, [], [], "medium")
            ],
            confidence=0.85,
            reasoning="Selected unit_mock strategy"
        )
        
        assert recommendation.primary_strategy == TestStrategy.UNIT_MOCK
        assert len(recommendation.secondary_strategies) == 2
        assert recommendation.confidence == 0.85


class TestTestStrategy:
    """Tests for TestStrategy enum."""

    def test_strategy_values(self):
        """Test strategy enum values."""
        assert TestStrategy.UNIT_BASIC.value == "unit_basic"
        assert TestStrategy.UNIT_MOCK.value == "unit_mock"
        assert TestStrategy.EXCEPTION.value == "exception"
        assert TestStrategy.INTEGRATION.value == "integration"


class TestCodeCharacteristic:
    """Tests for CodeCharacteristic enum."""

    def test_characteristic_values(self):
        """Test characteristic enum values."""
        assert CodeCharacteristic.HAS_DEPENDENCIES.value == "has_dependencies"
        assert CodeCharacteristic.HAS_EXCEPTIONS.value == "has_exceptions"
        assert CodeCharacteristic.IS_SERVICE_CLASS.value == "is_service_class"
