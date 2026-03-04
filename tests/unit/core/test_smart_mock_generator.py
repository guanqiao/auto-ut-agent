"""Unit tests for SmartMockGenerator module."""

import pytest

from pyutagent.core.smart_mock_generator import (
    SmartMockGenerator,
    DataType,
    GenerationContext,
    MockConfig,
    GeneratedMock,
    MockSetup,
)


class TestSmartMockGenerator:
    """Tests for SmartMockGenerator class."""

    def test_init(self):
        """Test initialization."""
        generator = SmartMockGenerator()
        
        assert generator._type_handlers is not None
        assert generator._pattern_matchers is not None

    def test_init_with_seed(self):
        """Test initialization with seed."""
        generator = SmartMockGenerator(seed=42)
        
        assert generator._seed == 42

    def test_generate_string(self):
        """Test generating string."""
        generator = SmartMockGenerator()
        
        config = MockConfig(data_type=DataType.STRING)
        result = generator.generate(config)
        
        assert result is not None
        assert result.data_type == DataType.STRING

    def test_generate_integer(self):
        """Test generating integer."""
        generator = SmartMockGenerator()
        
        config = MockConfig(data_type=DataType.INTEGER)
        result = generator.generate(config)
        
        assert result is not None
        assert result.data_type == DataType.INTEGER
        assert isinstance(result.value, int)

    def test_generate_boolean(self):
        """Test generating boolean."""
        generator = SmartMockGenerator()
        
        config = MockConfig(data_type=DataType.BOOLEAN)
        result = generator.generate(config)
        
        assert result is not None
        assert result.data_type == DataType.BOOLEAN
        assert isinstance(result.value, bool)

    def test_generate_date(self):
        """Test generating date."""
        generator = SmartMockGenerator()
        
        config = MockConfig(data_type=DataType.DATE)
        result = generator.generate(config)
        
        assert result is not None
        assert result.data_type == DataType.DATE

    def test_generate_datetime(self):
        """Test generating datetime."""
        generator = SmartMockGenerator()
        
        config = MockConfig(data_type=DataType.DATETIME)
        result = generator.generate(config)
        
        assert result is not None
        assert result.data_type == DataType.DATETIME

    def test_generate_list(self):
        """Test generating list."""
        generator = SmartMockGenerator()
        
        config = MockConfig(data_type=DataType.LIST)
        result = generator.generate(config)
        
        assert result is not None
        assert result.data_type == DataType.LIST
        assert isinstance(result.value, list)

    def test_generate_with_constraints(self):
        """Test generating with constraints."""
        generator = SmartMockGenerator()
        
        config = MockConfig(
            data_type=DataType.INTEGER,
            constraints={"min": 10, "max": 20}
        )
        result = generator.generate(config)
        
        assert result is not None
        assert 10 <= result.value <= 20

    def test_generate_for_field_string(self):
        """Test generating for string field."""
        generator = SmartMockGenerator()
        
        result = generator.generate_for_field("name", "String")
        
        assert result is not None
        assert result.data_type == DataType.STRING

    def test_generate_for_field_integer(self):
        """Test generating for integer field."""
        generator = SmartMockGenerator()
        
        result = generator.generate_for_field("age", "int")
        
        assert result is not None
        assert result.data_type == DataType.INTEGER

    def test_generate_for_field_boolean(self):
        """Test generating for boolean field."""
        generator = SmartMockGenerator()
        
        result = generator.generate_for_field("active", "boolean")
        
        assert result is not None
        assert result.data_type == DataType.BOOLEAN

    def test_detect_data_type_string(self):
        """Test detecting string data type."""
        generator = SmartMockGenerator()
        
        assert generator._detect_data_type("String") == DataType.STRING
        assert generator._detect_data_type("CharSequence") == DataType.STRING

    def test_detect_data_type_integer(self):
        """Test detecting integer data type."""
        generator = SmartMockGenerator()
        
        assert generator._detect_data_type("int") == DataType.INTEGER
        assert generator._detect_data_type("Integer") == DataType.INTEGER

    def test_detect_data_type_long(self):
        """Test detecting long data type."""
        generator = SmartMockGenerator()
        
        assert generator._detect_data_type("long") == DataType.LONG
        assert generator._detect_data_type("Long") == DataType.LONG

    def test_detect_data_type_double(self):
        """Test detecting double data type."""
        generator = SmartMockGenerator()
        
        assert generator._detect_data_type("double") == DataType.DOUBLE
        assert generator._detect_data_type("Double") == DataType.DOUBLE

    def test_detect_data_type_boolean(self):
        """Test detecting boolean data type."""
        generator = SmartMockGenerator()
        
        assert generator._detect_data_type("boolean") == DataType.BOOLEAN
        assert generator._detect_data_type("Boolean") == DataType.BOOLEAN

    def test_detect_data_type_date(self):
        """Test detecting date data type."""
        generator = SmartMockGenerator()
        
        assert generator._detect_data_type("LocalDate") == DataType.DATE
        assert generator._detect_data_type("Date") == DataType.DATE

    def test_detect_data_type_datetime(self):
        """Test detecting datetime data type."""
        generator = SmartMockGenerator()
        
        assert generator._detect_data_type("LocalDateTime") == DataType.DATETIME
        assert generator._detect_data_type("ZonedDateTime") == DataType.DATETIME

    def test_detect_data_type_list(self):
        """Test detecting list data type."""
        generator = SmartMockGenerator()
        
        assert generator._detect_data_type("List") == DataType.LIST
        assert generator._detect_data_type("ArrayList") == DataType.LIST

    def test_detect_data_type_set(self):
        """Test detecting set data type."""
        generator = SmartMockGenerator()
        
        assert generator._detect_data_type("Set") == DataType.SET
        assert generator._detect_data_type("HashSet") == DataType.SET

    def test_detect_data_type_map(self):
        """Test detecting map data type."""
        generator = SmartMockGenerator()
        
        assert generator._detect_data_type("Map") == DataType.MAP
        assert generator._detect_data_type("HashMap") == DataType.MAP

    def test_generate_mock_setup(self):
        """Test generating mock setup."""
        generator = SmartMockGenerator()
        
        fields = {
            "name": "String",
            "age": "int",
            "active": "boolean"
        }
        
        setup = generator.generate_mock_setup(fields)
        
        assert setup is not None
        assert isinstance(setup, MockSetup)
        assert len(setup.test_data) >= 3


class TestMockConfig:
    """Tests for MockConfig dataclass."""

    def test_config_creation(self):
        """Test config creation."""
        config = MockConfig(
            data_type=DataType.STRING,
            constraints={"min_length": 5},
            context=GenerationContext.POSITIVE,
            field_name="testField"
        )
        
        assert config.data_type == DataType.STRING
        assert config.constraints["min_length"] == 5
        assert config.context == GenerationContext.POSITIVE

    def test_config_minimal(self):
        """Test minimal config."""
        config = MockConfig(data_type=DataType.INTEGER)
        
        assert config.data_type == DataType.INTEGER
        assert config.constraints == {}
        assert config.context == GenerationContext.POSITIVE


class TestGeneratedMock:
    """Tests for GeneratedMock dataclass."""

    def test_mock_creation(self):
        """Test mock creation."""
        mock = GeneratedMock(
            value="test_value",
            data_type=DataType.STRING,
            generation_method="_generate_string",
            description="Generated string",
            constraints_applied=["min_length"]
        )
        
        assert mock.value == "test_value"
        assert mock.data_type == DataType.STRING
        assert len(mock.constraints_applied) == 1


class TestMockSetup:
    """Tests for MockSetup dataclass."""

    def test_setup_creation(self):
        """Test setup creation."""
        setup = MockSetup(
            mock_declarations=["@Mock UserService userService;"],
            mock_initializations=["MockitoAnnotations.openMocks(this);"],
            stub_configurations=["when(userService.getUser()).thenReturn(user);"],
            test_data={"name": "test"}
        )
        
        assert len(setup.mock_declarations) == 1
        assert len(setup.test_data) == 1


class TestDataType:
    """Tests for DataType enum."""

    def test_data_type_values(self):
        """Test data type enum values."""
        assert DataType.STRING.value == "string"
        assert DataType.INTEGER.value == "integer"
        assert DataType.LONG.value == "long"
        assert DataType.DOUBLE.value == "double"
        assert DataType.BOOLEAN.value == "boolean"
        assert DataType.DATE.value == "date"
        assert DataType.DATETIME.value == "datetime"
        assert DataType.LIST.value == "list"


class TestGenerationContext:
    """Tests for GenerationContext enum."""

    def test_context_values(self):
        """Test context enum values."""
        assert GenerationContext.POSITIVE.value == "positive"
        assert GenerationContext.NEGATIVE.value == "negative"
        assert GenerationContext.BOUNDARY.value == "boundary"
        assert GenerationContext.EDGE_CASE.value == "edge_case"


class TestSmartMockGeneratorIntegration:
    """Integration tests for smart mock generator."""

    def test_full_mock_generation_workflow(self):
        """Test full mock generation workflow."""
        generator = SmartMockGenerator()
        
        fields = {
            "id": "Long",
            "name": "String",
            "email": "String",
            "age": "int",
            "active": "boolean",
            "createdAt": "LocalDateTime"
        }
        
        setup = generator.generate_mock_setup(fields)
        
        assert setup is not None
        assert "id" in setup.test_data
        assert "name" in setup.test_data
        assert "email" in setup.test_data
        assert "age" in setup.test_data
        assert "active" in setup.test_data
        assert "createdAt" in setup.test_data

    def test_boundary_value_generation(self):
        """Test boundary value generation."""
        generator = SmartMockGenerator()
        
        config = MockConfig(
            data_type=DataType.INTEGER,
            context=GenerationContext.BOUNDARY
        )
        result = generator.generate(config)
        
        assert result is not None

    def test_negative_value_generation(self):
        """Test negative value generation."""
        generator = SmartMockGenerator()
        
        config = MockConfig(
            data_type=DataType.INTEGER,
            context=GenerationContext.NEGATIVE
        )
        result = generator.generate(config)
        
        assert result is not None
