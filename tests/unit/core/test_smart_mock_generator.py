"""Unit tests for SmartMockGenerator module."""

import pytest

from pyutagent.core.smart_mock_generator import (
    SmartMockGenerator,
    MockConfig,
    GeneratedMock,
    GenerationContext,
    DataType,
    MockSetup,
)


class TestSmartMockGenerator:
    """Tests for SmartMockGenerator class."""

    def test_init(self):
        """Test initialization."""
        generator = SmartMockGenerator()
        
        assert generator._random is not None
        assert generator._type_handlers is not None
        assert generator._field_patterns is not None

    def test_init_with_seed(self):
        """Test initialization with seed for reproducibility."""
        generator1 = SmartMockGenerator(seed=42)
        generator2 = SmartMockGenerator(seed=42)
        
        result1 = generator1.generate(MockConfig(data_type=DataType.INTEGER))
        result2 = generator2.generate(MockConfig(data_type=DataType.INTEGER))
        
        assert result1.value == result2.value

    def test_generate_string(self):
        """Test generating string value."""
        generator = SmartMockGenerator()
        
        result = generator.generate(MockConfig(data_type=DataType.STRING))
        
        assert isinstance(result.value, str)
        assert result.data_type == DataType.STRING

    def test_generate_string_with_constraints(self):
        """Test generating string with constraints."""
        generator = SmartMockGenerator()
        
        config = MockConfig(
            data_type=DataType.STRING,
            constraints={"min_length": 5, "max_length": 10}
        )
        
        result = generator.generate(config)
        
        assert 5 <= len(result.value) <= 10

    def test_generate_integer(self):
        """Test generating integer value."""
        generator = SmartMockGenerator()
        
        result = generator.generate(MockConfig(data_type=DataType.INTEGER))
        
        assert isinstance(result.value, int)
        assert result.data_type == DataType.INTEGER

    def test_generate_integer_with_constraints(self):
        """Test generating integer with constraints."""
        generator = SmartMockGenerator()
        
        config = MockConfig(
            data_type=DataType.INTEGER,
            constraints={"min": 10, "max": 20}
        )
        
        result = generator.generate(config)
        
        assert 10 <= result.value <= 20

    def test_generate_long(self):
        """Test generating long value."""
        generator = SmartMockGenerator()
        
        result = generator.generate(MockConfig(data_type=DataType.LONG))
        
        assert isinstance(result.value, int)
        assert result.data_type == DataType.LONG

    def test_generate_double(self):
        """Test generating double value."""
        generator = SmartMockGenerator()
        
        result = generator.generate(MockConfig(data_type=DataType.DOUBLE))
        
        assert isinstance(result.value, float)
        assert result.data_type == DataType.DOUBLE

    def test_generate_boolean(self):
        """Test generating boolean value."""
        generator = SmartMockGenerator()
        
        result = generator.generate(MockConfig(data_type=DataType.BOOLEAN))
        
        assert isinstance(result.value, bool)
        assert result.data_type == DataType.BOOLEAN

    def test_generate_date(self):
        """Test generating date value."""
        generator = SmartMockGenerator()
        
        result = generator.generate(MockConfig(data_type=DataType.DATE))
        
        assert isinstance(result.value, str)
        assert result.data_type == DataType.DATE

    def test_generate_datetime(self):
        """Test generating datetime value."""
        generator = SmartMockGenerator()
        
        result = generator.generate(MockConfig(data_type=DataType.DATETIME))
        
        assert isinstance(result.value, str)
        assert result.data_type == DataType.DATETIME

    def test_generate_list(self):
        """Test generating list value."""
        generator = SmartMockGenerator()
        
        config = MockConfig(
            data_type=DataType.LIST,
            constraints={"element_type": "String", "min_size": 0, "max_size": 5}
        )
        
        result = generator.generate(config)
        
        assert isinstance(result.value, list)
        assert len(result.value) <= 5

    def test_generate_set(self):
        """Test generating set value."""
        generator = SmartMockGenerator()
        
        config = MockConfig(
            data_type=DataType.SET,
            constraints={"element_type": "Integer"}
        )
        
        result = generator.generate(config)
        
        assert isinstance(result.value, set)

    def test_generate_map(self):
        """Test generating map value."""
        generator = SmartMockGenerator()
        
        config = MockConfig(
            data_type=DataType.MAP,
            constraints={"key_type": "String", "value_type": "Integer"}
        )
        
        result = generator.generate(config)
        
        assert isinstance(result.value, dict)

    def test_generate_for_field_email(self):
        """Test generating value for email field."""
        generator = SmartMockGenerator()
        
        result = generator.generate_for_field("email", "String")
        
        assert "@" in result.value
        assert "." in result.value

    def test_generate_for_field_phone(self):
        """Test generating value for phone field."""
        generator = SmartMockGenerator()
        
        result = generator.generate_for_field("phone", "String")
        
        assert "-" in result.value

    def test_generate_for_field_name(self):
        """Test generating value for name field."""
        generator = SmartMockGenerator()
        
        result = generator.generate_for_field("name", "String")
        
        assert isinstance(result.value, str)
        assert len(result.value) > 0

    def test_generate_for_field_age(self):
        """Test generating value for age field."""
        generator = SmartMockGenerator()
        
        result = generator.generate_for_field("age", "int")
        
        assert 0 <= result.value <= 120

    def test_generate_for_field_price(self):
        """Test generating value for price field."""
        generator = SmartMockGenerator()
        
        result = generator.generate_for_field("price", "double")
        
        assert result.value >= 0.0

    def test_generate_for_field_with_annotations(self):
        """Test generating value with validation annotations."""
        generator = SmartMockGenerator()
        
        result = generator.generate_for_field(
            "username",
            "String",
            annotations=["@NotNull", "@Size(min=3, max=20)"]
        )
        
        assert result.value is not None
        assert 3 <= len(result.value) <= 20

    def test_generate_negative_context(self):
        """Test generating negative value."""
        generator = SmartMockGenerator()
        
        result = generator.generate_for_field(
            "email",
            "String",
            context=GenerationContext.NEGATIVE
        )
        
        assert result.value is None or "@" not in str(result.value)

    def test_generate_boundary_context(self):
        """Test generating boundary value."""
        generator = SmartMockGenerator()
        
        result = generator.generate_for_field(
            "count",
            "int",
            context=GenerationContext.BOUNDARY
        )
        
        assert result.value in [0, 1, -1, 2147483647, -2147483648]

    def test_generate_test_data(self):
        """Test generating test data for class."""
        generator = SmartMockGenerator()
        
        class_info = {
            "fields": [
                {"name": "id", "type": "Long"},
                {"name": "name", "type": "String"},
                {"name": "active", "type": "boolean"}
            ]
        }
        
        result = generator.generate_test_data(class_info)
        
        assert "id" in result
        assert "name" in result
        assert "active" in result

    def test_generate_mock_setup(self):
        """Test generating mock setup."""
        generator = SmartMockGenerator()
        
        setup = generator.generate_mock_setup(
            class_under_test="UserService",
            dependencies=[
                {"name": "userRepository", "type": "UserRepository"},
                {"name": "emailService", "type": "EmailService"}
            ]
        )
        
        assert isinstance(setup, MockSetup)
        assert len(setup.mock_declarations) > 0
        assert len(setup.test_data) == 2

    def test_detect_data_type_integer(self):
        """Test detecting integer data type."""
        generator = SmartMockGenerator()
        
        assert generator._detect_data_type("int") == DataType.INTEGER
        assert generator._detect_data_type("Integer") == DataType.INTEGER

    def test_detect_data_type_string(self):
        """Test detecting string data type."""
        generator = SmartMockGenerator()
        
        assert generator._detect_data_type("String") == DataType.STRING
        assert generator._detect_data_type("CharSequence") == DataType.STRING

    def test_detect_data_type_collection(self):
        """Test detecting collection data type."""
        generator = SmartMockGenerator()
        
        assert generator._detect_data_type("List") == DataType.LIST
        assert generator._detect_data_type("Set") == DataType.SET
        assert generator._detect_data_type("Map") == DataType.MAP

    def test_detect_data_type_date(self):
        """Test detecting date data type."""
        generator = SmartMockGenerator()
        
        assert generator._detect_data_type("LocalDate") == DataType.DATE
        assert generator._detect_data_type("Date") == DataType.DATE
        assert generator._detect_data_type("LocalDateTime") == DataType.DATETIME

    def test_extract_constraints_not_null(self):
        """Test extracting NotNull constraint."""
        generator = SmartMockGenerator()
        
        constraints = generator._extract_constraints(["@NotNull", "@NonNull"])
        
        assert "not_null" in constraints

    def test_extract_constraints_size(self):
        """Test extracting Size constraint."""
        generator = SmartMockGenerator()
        
        constraints = generator._extract_constraints(["@Size(min=1, max=100)"])
        
        assert "min_size" in constraints or "max_size" in constraints

    def test_extract_constraints_min_max(self):
        """Test extracting Min/Max constraints."""
        generator = SmartMockGenerator()
        
        constraints = generator._extract_constraints(["@Min(0)", "@Max(100)"])
        
        assert "min" in constraints
        assert "max" in constraints

    def test_extract_constraints_pattern(self):
        """Test extracting Pattern constraint."""
        generator = SmartMockGenerator()
        
        constraints = generator._extract_constraints(["@Pattern(regexp=\"[a-z]+\")"])
        
        assert "pattern" in constraints

    def test_extract_constraints_email(self):
        """Test extracting Email constraint."""
        generator = SmartMockGenerator()
        
        constraints = generator._extract_constraints(["@Email"])
        
        assert "pattern" in constraints

    def test_apply_constraints_string(self):
        """Test applying constraints to string."""
        generator = SmartMockGenerator()
        
        value, applied = generator._apply_constraints(
            "test",
            MockConfig(
                data_type=DataType.STRING,
                constraints={"min_length": 5}
            )
        )
        
        assert len(value) >= 5

    def test_apply_constraints_integer(self):
        """Test applying constraints to integer."""
        generator = SmartMockGenerator()
        
        value, applied = generator._apply_constraints(
            50,
            MockConfig(
                data_type=DataType.INTEGER,
                constraints={"min": 0, "max": 10}
            )
        )
        
        assert 0 <= value <= 10

    def test_make_negative_string(self):
        """Test making string negative."""
        generator = SmartMockGenerator()
        
        result = generator._make_negative(
            "test@example.com",
            MockConfig(
                data_type=DataType.STRING,
                constraints={"not_blank": True}
            )
        )
        
        assert result is None or result == ""

    def test_make_negative_integer(self):
        """Test making integer negative."""
        generator = SmartMockGenerator()
        
        result = generator._make_negative(
            5,
            MockConfig(
                data_type=DataType.INTEGER,
                constraints={"positive": True}
            )
        )
        
        assert result < 0

    def test_make_boundary_integer(self):
        """Test making integer boundary value."""
        generator = SmartMockGenerator()
        
        result = generator._make_boundary(
            50,
            MockConfig(
                data_type=DataType.INTEGER,
                constraints={"min": 0, "max": 100}
            )
        )
        
        assert result in [0, 100]

    def test_generate_description(self):
        """Test generating description."""
        generator = SmartMockGenerator()
        
        description = generator._generate_description(
            MockConfig(data_type=DataType.STRING, field_name="email"),
            "test@example.com"
        )
        
        assert "email" in description
        assert "string" in description.lower()


class TestMockConfig:
    """Tests for MockConfig dataclass."""

    def test_config_creation(self):
        """Test config creation."""
        config = MockConfig(
            data_type=DataType.STRING,
            constraints={"min_length": 5},
            context=GenerationContext.POSITIVE,
            field_name="username"
        )
        
        assert config.data_type == DataType.STRING
        assert config.constraints["min_length"] == 5
        assert config.context == GenerationContext.POSITIVE

    def test_config_defaults(self):
        """Test config defaults."""
        config = MockConfig(data_type=DataType.INTEGER)
        
        assert config.constraints == {}
        assert config.context == GenerationContext.POSITIVE
        assert config.field_name is None


class TestGeneratedMock:
    """Tests for GeneratedMock dataclass."""

    def test_mock_creation(self):
        """Test mock creation."""
        mock = GeneratedMock(
            value="test@example.com",
            data_type=DataType.STRING,
            generation_method="_generate_string",
            description="Generated string for email field",
            constraints_applied=["pattern"]
        )
        
        assert mock.value == "test@example.com"
        assert mock.data_type == DataType.STRING
        assert len(mock.constraints_applied) == 1


class TestDataType:
    """Tests for DataType enum."""

    def test_data_type_values(self):
        """Test data type enum values."""
        assert DataType.STRING.value == "string"
        assert DataType.INTEGER.value == "integer"
        assert DataType.BOOLEAN.value == "boolean"
        assert DataType.LIST.value == "list"
        assert DataType.DATE.value == "date"


class TestGenerationContext:
    """Tests for GenerationContext enum."""

    def test_context_values(self):
        """Test context enum values."""
        assert GenerationContext.POSITIVE.value == "positive"
        assert GenerationContext.NEGATIVE.value == "negative"
        assert GenerationContext.BOUNDARY.value == "boundary"
        assert GenerationContext.EDGE_CASE.value == "edge_case"


class TestMockSetup:
    """Tests for MockSetup dataclass."""

    def test_setup_creation(self):
        """Test setup creation."""
        setup = MockSetup(
            mock_declarations=["@Mock private Repo repo;"],
            mock_initializations=["MockitoAnnotations.openMocks(this);"],
            stub_configurations=["when(repo.find()).thenReturn(list);"],
            test_data={"repo": {"type": "Repo", "mocked": True}}
        )
        
        assert len(setup.mock_declarations) == 1
        assert len(setup.test_data) == 1


class TestSmartMockGeneratorIntegration:
    """Integration tests for smart mock generator."""

    def test_full_generation_workflow(self):
        """Test full generation workflow."""
        generator = SmartMockGenerator(seed=42)
        
        class_info = {
            "fields": [
                {"name": "id", "type": "Long", "annotations": ["@NotNull"]},
                {"name": "email", "type": "String", "annotations": ["@Email"]},
                {"name": "age", "type": "int", "annotations": ["@Min(0)", "@Max(150)"]},
                {"name": "active", "type": "boolean"},
                {"name": "tags", "type": "List<String>"}
            ]
        }
        
        result = generator.generate_test_data(class_info)
        
        assert "id" in result
        assert "email" in result
        assert "age" in result
        assert "active" in result
        assert "tags" in result
        
        assert result["id"].value is not None
        assert "@" in result["email"].value
        assert 0 <= result["age"].value <= 150
        assert isinstance(result["active"].value, bool)

    def test_mock_setup_workflow(self):
        """Test mock setup workflow."""
        generator = SmartMockGenerator()
        
        setup = generator.generate_mock_setup(
            class_under_test="OrderService",
            dependencies=[
                {"name": "orderRepository", "type": "OrderRepository"},
                {"name": "paymentGateway", "type": "PaymentGateway"},
                {"name": "notificationService", "type": "NotificationService"}
            ],
            test_scenarios=["create_order", "process_payment"]
        )
        
        assert "OrderService" in setup.mock_declarations[0]
        assert len(setup.test_data) == 3
