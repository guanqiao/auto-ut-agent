"""Smart Mock Data Generator for intelligent test data creation.

This module provides intelligent mock data generation based on type analysis,
context understanding, and pattern recognition.
"""

import logging
import random
import re
import string
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Set, Type, get_type_hints
from collections import defaultdict

logger = logging.getLogger(__name__)


class DataType(Enum):
    """Types of data to generate."""
    STRING = "string"
    INTEGER = "integer"
    LONG = "long"
    DOUBLE = "double"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    LIST = "list"
    SET = "set"
    MAP = "map"
    OBJECT = "object"
    ENUM = "enum"


class GenerationContext(Enum):
    """Context for data generation."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    BOUNDARY = "boundary"
    EDGE_CASE = "edge_case"
    PERFORMANCE = "performance"


@dataclass
class MockConfig:
    """Configuration for mock generation."""
    data_type: DataType
    constraints: Dict[str, Any] = field(default_factory=dict)
    context: GenerationContext = GenerationContext.POSITIVE
    field_name: Optional[str] = None
    class_name: Optional[str] = None
    annotations: List[str] = field(default_factory=list)


@dataclass
class GeneratedMock:
    """Result of mock generation."""
    value: Any
    data_type: DataType
    generation_method: str
    description: str
    constraints_applied: List[str] = field(default_factory=list)


@dataclass
class MockSetup:
    """Complete mock setup for a test."""
    mock_declarations: List[str]
    mock_initializations: List[str]
    stub_configurations: List[str]
    test_data: Dict[str, Any]


class SmartMockGenerator:
    """Intelligent mock data generator.
    
    Features:
    - Type-aware data generation
    - Constraint-based generation
    - Context-sensitive values
    - Pattern-based generation
    - Realistic test data
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the smart mock generator.
        
        Args:
            seed: Optional random seed for reproducible generation
        """
        self._random = random.Random(seed)
        self._type_handlers = self._initialize_type_handlers()
        self._field_patterns = self._initialize_field_patterns()
        self._value_cache: Dict[str, Any] = {}
        
        logger.info("[SmartMockGenerator] Initialized")
    
    def _initialize_type_handlers(self) -> Dict[DataType, callable]:
        """Initialize handlers for different data types."""
        return {
            DataType.STRING: self._generate_string,
            DataType.INTEGER: self._generate_integer,
            DataType.LONG: self._generate_long,
            DataType.DOUBLE: self._generate_double,
            DataType.BOOLEAN: self._generate_boolean,
            DataType.DATE: self._generate_date,
            DataType.DATETIME: self._generate_datetime,
            DataType.LIST: self._generate_list,
            DataType.SET: self._generate_set,
            DataType.MAP: self._generate_map,
            DataType.OBJECT: self._generate_object,
            DataType.ENUM: self._generate_enum,
        }
    
    def _initialize_field_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize patterns for common field names."""
        return {
            "email": {
                "pattern": r"[a-z]{5,10}@[a-z]{3,8}\.(com|org|net)",
                "examples": ["test@example.com", "user@domain.org"]
            },
            "phone": {
                "pattern": r"\d{3}-\d{3}-\d{4}",
                "examples": ["123-456-7890", "555-123-4567"]
            },
            "name": {
                "examples": ["John Doe", "Jane Smith", "Test User"]
            },
            "firstname": {
                "examples": ["John", "Jane", "Mike", "Sarah"]
            },
            "lastname": {
                "examples": ["Doe", "Smith", "Johnson", "Williams"]
            },
            "address": {
                "examples": ["123 Main St", "456 Oak Ave", "789 Pine Rd"]
            },
            "city": {
                "examples": ["New York", "Los Angeles", "Chicago", "Houston"]
            },
            "zipcode": {
                "pattern": r"\d{5}",
                "examples": ["10001", "90210", "60601"]
            },
            "url": {
                "examples": ["https://example.com", "http://test.org/page"]
            },
            "id": {
                "type": "identifier",
                "examples": [1, 2, 100]
            },
            "userid": {
                "type": "identifier",
                "examples": [1, 2, 100]
            },
            "age": {
                "min": 0,
                "max": 120,
                "examples": [25, 30, 45]
            },
            "price": {
                "min": 0.0,
                "max": 10000.0,
                "examples": [9.99, 19.99, 99.99]
            },
            "quantity": {
                "min": 0,
                "max": 1000,
                "examples": [1, 10, 100]
            },
            "status": {
                "examples": ["ACTIVE", "INACTIVE", "PENDING"]
            },
            "description": {
                "examples": ["Test description", "Sample text for testing"]
            },
            "title": {
                "examples": ["Test Title", "Sample Title"]
            },
            "code": {
                "examples": ["CODE001", "TEST-123", "ABC-XYZ"]
            }
        }
    
    def generate(self, config: MockConfig) -> GeneratedMock:
        """Generate mock data based on configuration.
        
        Args:
            config: Mock configuration
            
        Returns:
            GeneratedMock with the generated value
        """
        handler = self._type_handlers.get(config.data_type, self._generate_object)
        
        value = handler(config)
        
        constraints_applied = []
        if config.constraints:
            value, constraints_applied = self._apply_constraints(value, config)
        
        if config.context == GenerationContext.NEGATIVE:
            value = self._make_negative(value, config)
        elif config.context == GenerationContext.BOUNDARY:
            value = self._make_boundary(value, config)
        
        description = self._generate_description(config, value)
        
        return GeneratedMock(
            value=value,
            data_type=config.data_type,
            generation_method=handler.__name__,
            description=description,
            constraints_applied=constraints_applied
        )
    
    def generate_for_field(
        self,
        field_name: str,
        field_type: str,
        annotations: Optional[List[str]] = None,
        context: GenerationContext = GenerationContext.POSITIVE
    ) -> GeneratedMock:
        """Generate mock data for a specific field.
        
        Args:
            field_name: Name of the field
            field_type: Type of the field
            annotations: Optional list of field annotations
            context: Generation context
            
        Returns:
            GeneratedMock with appropriate value
        """
        data_type = self._detect_data_type(field_type)
        
        constraints = self._extract_constraints(annotations or [])
        
        if field_name.lower() in self._field_patterns:
            pattern_config = self._field_patterns[field_name.lower()]
            if "examples" in pattern_config:
                constraints["examples"] = pattern_config["examples"]
            if "pattern" in pattern_config:
                constraints["pattern"] = pattern_config["pattern"]
            if "min" in pattern_config:
                constraints["min"] = pattern_config["min"]
            if "max" in pattern_config:
                constraints["max"] = pattern_config["max"]
        
        config = MockConfig(
            data_type=data_type,
            constraints=constraints,
            context=context,
            field_name=field_name,
            annotations=annotations or []
        )
        
        return self.generate(config)
    
    def generate_test_data(
        self,
        class_info: Dict[str, Any],
        context: GenerationContext = GenerationContext.POSITIVE
    ) -> Dict[str, GeneratedMock]:
        """Generate test data for all fields of a class.
        
        Args:
            class_info: Class information with fields
            context: Generation context
            
        Returns:
            Dictionary mapping field names to generated mocks
        """
        result = {}
        
        fields = class_info.get("fields", [])
        for field in fields:
            field_name = field.get("name", "")
            field_type = field.get("type", "Object")
            annotations = field.get("annotations", [])
            
            if field_name:
                result[field_name] = self.generate_for_field(
                    field_name, field_type, annotations, context
                )
        
        return result
    
    def generate_mock_setup(
        self,
        class_under_test: str,
        dependencies: List[Dict[str, str]],
        test_scenarios: Optional[List[str]] = None
    ) -> MockSetup:
        """Generate complete mock setup for a test class.
        
        Args:
            class_under_test: Name of the class being tested
            dependencies: List of dependencies with name and type
            test_scenarios: Optional list of test scenarios
            
        Returns:
            MockSetup with all necessary mock code
        """
        declarations = []
        initializations = []
        stubs = []
        test_data = {}
        
        declarations.append(f"@ExtendWith(MockitoExtension.class)")
        declarations.append(f"class {class_under_test}Test {{")
        declarations.append("")
        
        for dep in dependencies:
            dep_name = dep.get("name", "dependency")
            dep_type = dep.get("type", "Object")
            
            declarations.append(f"    @Mock")
            declarations.append(f"    private {dep_type} {dep_name};")
            test_data[dep_name] = {"type": dep_type, "mocked": True}
        
        declarations.append("")
        declarations.append(f"    @InjectMocks")
        declarations.append(f"    private {class_under_test} {self._to_camel_case(class_under_test)};")
        declarations.append("")
        
        initializations.append("    @BeforeEach")
        initializations.append("    void setUp() {")
        
        for dep in dependencies:
            dep_name = dep.get("name", "dependency")
            dep_type = dep.get("type", "Object")
            
            mock_value = self.generate_for_field(dep_name, dep_type)
            initializations.append(f"        // {dep_name}: {mock_value.description}")
        
        initializations.append("    }")
        initializations.append("")
        
        if test_scenarios:
            for scenario in test_scenarios:
                stub = self._generate_stub_for_scenario(scenario, dependencies)
                if stub:
                    stubs.append(stub)
        
        return MockSetup(
            mock_declarations=declarations,
            mock_initializations=initializations,
            stub_configurations=stubs,
            test_data=test_data
        )
    
    def _generate_string(self, config: MockConfig) -> str:
        """Generate a string value."""
        constraints = config.constraints
        
        if "examples" in constraints:
            return self._random.choice(constraints["examples"])
        
        if "pattern" in constraints:
            return self._generate_from_pattern(constraints["pattern"])
        
        if config.field_name:
            field_lower = config.field_name.lower()
            if field_lower in self._field_patterns:
                pattern_config = self._field_patterns[field_lower]
                if "examples" in pattern_config:
                    return self._random.choice(pattern_config["examples"])
                if "pattern" in pattern_config:
                    return self._generate_from_pattern(pattern_config["pattern"])
        
        min_len = constraints.get("min_length", 5)
        max_len = constraints.get("max_length", 20)
        length = self._random.randint(min_len, max_len)
        
        chars = string.ascii_letters + string.digits
        return ''.join(self._random.choice(chars) for _ in range(length))
    
    def _generate_integer(self, config: MockConfig) -> int:
        """Generate an integer value."""
        constraints = config.constraints
        
        if "examples" in constraints:
            return self._random.choice(constraints["examples"])
        
        min_val = constraints.get("min", -1000)
        max_val = constraints.get("max", 1000)
        
        if config.field_name and config.field_name.lower() in self._field_patterns:
            pattern_config = self._field_patterns[config.field_name.lower()]
            min_val = pattern_config.get("min", min_val)
            max_val = pattern_config.get("max", max_val)
        
        return self._random.randint(min_val, max_val)
    
    def _generate_long(self, config: MockConfig) -> int:
        """Generate a long value."""
        return self._generate_integer(config) * 1000
    
    def _generate_double(self, config: MockConfig) -> float:
        """Generate a double value."""
        constraints = config.constraints
        
        if "examples" in constraints:
            return self._random.choice(constraints["examples"])
        
        min_val = constraints.get("min", 0.0)
        max_val = constraints.get("max", 1000.0)
        
        value = self._random.uniform(min_val, max_val)
        return round(value, 2)
    
    def _generate_boolean(self, config: MockConfig) -> bool:
        """Generate a boolean value."""
        return self._random.choice([True, False])
    
    def _generate_date(self, config: MockConfig) -> str:
        """Generate a date value."""
        constraints = config.constraints
        
        if "examples" in constraints:
            return self._random.choice(constraints["examples"])
        
        base = datetime.now()
        delta = timedelta(days=self._random.randint(-365, 365))
        date = base + delta
        
        return date.strftime("%Y-%m-%d")
    
    def _generate_datetime(self, config: MockConfig) -> str:
        """Generate a datetime value."""
        constraints = config.constraints
        
        if "examples" in constraints:
            return self._random.choice(constraints["examples"])
        
        base = datetime.now()
        delta = timedelta(
            days=self._random.randint(-30, 30),
            hours=self._random.randint(0, 23),
            minutes=self._random.randint(0, 59)
        )
        dt = base + delta
        
        return dt.strftime("%Y-%m-%dT%H:%M:%S")
    
    def _generate_list(self, config: MockConfig) -> List[Any]:
        """Generate a list value."""
        constraints = config.constraints
        
        element_type = constraints.get("element_type", "String")
        min_size = constraints.get("min_size", 0)
        max_size = constraints.get("max_size", 5)
        
        size = self._random.randint(min_size, max_size)
        
        element_config = MockConfig(
            data_type=self._detect_data_type(element_type),
            context=config.context
        )
        
        return [self.generate(element_config).value for _ in range(size)]
    
    def _generate_set(self, config: MockConfig) -> Set[Any]:
        """Generate a set value."""
        return set(self._generate_list(config))
    
    def _generate_map(self, config: MockConfig) -> Dict[Any, Any]:
        """Generate a map value."""
        constraints = config.constraints
        
        key_type = constraints.get("key_type", "String")
        value_type = constraints.get("value_type", "Object")
        min_size = constraints.get("min_size", 0)
        max_size = constraints.get("max_size", 3)
        
        size = self._random.randint(min_size, max_size)
        
        key_config = MockConfig(
            data_type=self._detect_data_type(key_type),
            context=config.context
        )
        value_config = MockConfig(
            data_type=self._detect_data_type(value_type),
            context=config.context
        )
        
        return {
            self.generate(key_config).value: self.generate(value_config).value
            for _ in range(size)
        }
    
    def _generate_object(self, config: MockConfig) -> Any:
        """Generate an object value."""
        if config.class_name:
            return f"new {config.class_name}()"
        return None
    
    def _generate_enum(self, config: MockConfig) -> str:
        """Generate an enum value."""
        constraints = config.constraints
        
        if "values" in constraints:
            return self._random.choice(constraints["values"])
        
        if "examples" in constraints:
            return self._random.choice(constraints["examples"])
        
        return "ENUM_VALUE"
    
    def _detect_data_type(self, type_str: str) -> DataType:
        """Detect data type from string."""
        type_lower = type_str.lower()
        
        if type_lower in ("string", "charsequence"):
            return DataType.STRING
        elif type_lower in ("int", "integer"):
            return DataType.INTEGER
        elif type_lower == "long":
            return DataType.LONG
        elif type_lower in ("double", "float", "bigdecimal"):
            return DataType.DOUBLE
        elif type_lower == "boolean":
            return DataType.BOOLEAN
        elif "localdate" in type_lower or type_lower == "date":
            return DataType.DATE
        elif "localdatetime" in type_lower or "timestamp" in type_lower:
            return DataType.DATETIME
        elif type_lower in ("list", "arraylist", "linkedlist", "collection"):
            return DataType.LIST
        elif type_lower in ("set", "hashset", "linkedhashset", "treeset"):
            return DataType.SET
        elif type_lower in ("map", "hashmap", "linkedhashmap", "treemap"):
            return DataType.MAP
        else:
            return DataType.OBJECT
    
    def _extract_constraints(self, annotations: List[str]) -> Dict[str, Any]:
        """Extract constraints from annotations."""
        constraints = {}
        
        for annotation in annotations:
            if "@NotNull" in annotation or "@NonNull" in annotation:
                constraints["not_null"] = True
            elif "@NotEmpty" in annotation:
                constraints["not_empty"] = True
            elif "@NotBlank" in annotation:
                constraints["not_blank"] = True
            elif "@Size" in annotation:
                match = re.search(r'min\s*=\s*(\d+)', annotation)
                if match:
                    constraints["min_size"] = int(match.group(1))
                match = re.search(r'max\s*=\s*(\d+)', annotation)
                if match:
                    constraints["max_size"] = int(match.group(1))
            elif "@Min" in annotation:
                match = re.search(r'@Min\s*\(\s*(\d+)', annotation)
                if match:
                    constraints["min"] = int(match.group(1))
            elif "@Max" in annotation:
                match = re.search(r'@Max\s*\(\s*(\d+)', annotation)
                if match:
                    constraints["max"] = int(match.group(1))
            elif "@Pattern" in annotation:
                match = re.search(r'regexp\s*=\s*"([^"]+)"', annotation)
                if match:
                    constraints["pattern"] = match.group(1)
            elif "@Email" in annotation:
                constraints["pattern"] = r"[a-z]+@[a-z]+\.[a-z]+"
            elif "@Positive" in annotation:
                constraints["min"] = 1
            elif "@PositiveOrZero" in annotation:
                constraints["min"] = 0
            elif "@Negative" in annotation:
                constraints["max"] = -1
        
        return constraints
    
    def _apply_constraints(
        self,
        value: Any,
        config: MockConfig
    ) -> Tuple[Any, List[str]]:
        """Apply constraints to a generated value."""
        constraints = config.constraints
        applied = []
        
        if isinstance(value, str):
            if "min_length" in constraints and len(value) < constraints["min_length"]:
                value = value + "x" * (constraints["min_length"] - len(value))
                applied.append("min_length")
            
            if "max_length" in constraints and len(value) > constraints["max_length"]:
                value = value[:constraints["max_length"]]
                applied.append("max_length")
            
            if "pattern" in constraints:
                value = self._generate_from_pattern(constraints["pattern"])
                applied.append("pattern")
        
        elif isinstance(value, (int, float)):
            if "min" in constraints and value < constraints["min"]:
                value = constraints["min"]
                applied.append("min")
            
            if "max" in constraints and value > constraints["max"]:
                value = constraints["max"]
                applied.append("max")
        
        return value, applied
    
    def _make_negative(self, value: Any, config: MockConfig) -> Any:
        """Make a value negative/invalid for negative testing."""
        if value is None:
            return None
        
        if isinstance(value, str):
            if config.constraints.get("not_blank"):
                return ""
            elif config.constraints.get("not_empty"):
                return ""
            elif config.constraints.get("pattern"):
                return "INVALID_FORMAT"
            else:
                return None
        
        elif isinstance(value, (int, float)):
            if config.constraints.get("min") is not None:
                return config.constraints["min"] - 1
            elif config.constraints.get("max") is not None:
                return config.constraints["max"] + 1
            elif config.constraints.get("positive"):
                return -1
            else:
                return -abs(value)
        
        elif isinstance(value, bool):
            return not value
        
        elif isinstance(value, list):
            return None
        
        return None
    
    def _make_boundary(self, value: Any, config: MockConfig) -> Any:
        """Make a value a boundary value."""
        if isinstance(value, (int, float)):
            constraints = config.constraints
            
            if "min" in constraints:
                return constraints["min"]
            elif "max" in constraints:
                return constraints["max"]
            else:
                if isinstance(value, int):
                    return self._random.choice([0, 1, -1, 2147483647, -2147483648])
                else:
                    return self._random.choice([0.0, 0.01, -0.01])
        
        elif isinstance(value, str):
            return ""
        
        elif isinstance(value, list):
            return []
        
        return value
    
    def _generate_from_pattern(self, pattern: str) -> str:
        """Generate a string from a regex pattern."""
        result = []
        i = 0
        
        while i < len(pattern):
            char = pattern[i]
            
            if char == '[':
                end = pattern.find(']', i)
                if end != -1:
                    chars = pattern[i+1:end]
                    chars = chars.replace('\\d', string.digits)
                    chars = chars.replace('\\w', string.ascii_letters + string.digits)
                    result.append(self._random.choice(chars))
                    i = end + 1
                    continue
            
            elif char == '\\':
                if i + 1 < len(pattern):
                    next_char = pattern[i + 1]
                    if next_char == 'd':
                        result.append(self._random.choice(string.digits))
                    elif next_char == 'w':
                        result.append(self._random.choice(string.ascii_letters + string.digits))
                    elif next_char == 's':
                        result.append(' ')
                    i += 2
                    continue
            
            elif char == '.':
                result.append(self._random.choice(string.ascii_letters))
            
            elif char in '+*?':
                pass
            
            elif char == '{':
                pass
            
            else:
                result.append(char)
            
            i += 1
        
        return ''.join(result)
    
    def _generate_description(self, config: MockConfig, value: Any) -> str:
        """Generate a description for the generated value."""
        parts = [f"Generated {config.data_type.value}"]
        
        if config.field_name:
            parts.append(f"for field '{config.field_name}'")
        
        if config.context != GenerationContext.POSITIVE:
            parts.append(f"({config.context.value} context)")
        
        parts.append(f": {repr(value)[:50]}")
        
        return " ".join(parts)
    
    def _generate_stub_for_scenario(
        self,
        scenario: str,
        dependencies: List[Dict[str, str]]
    ) -> Optional[str]:
        """Generate stub configuration for a test scenario."""
        lines = []
        lines.append(f"    // Stubs for: {scenario}")
        
        for dep in dependencies[:2]:
            dep_name = dep.get("name", "dep")
            dep_type = dep.get("type", "Object")
            lines.append(f"    when({dep_name}.method()).thenReturn(value);")
        
        return "\n".join(lines)
    
    def _to_camel_case(self, name: str) -> str:
        """Convert a class name to camelCase."""
        if not name:
            return name
        return name[0].lower() + name[1:]
