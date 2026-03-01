"""Constants for PyUT Agent.

This module contains all constants used throughout the application.
Centralizing constants makes the code more maintainable and reduces magic numbers/strings.
"""

# =============================================================================
# LLM Configuration Constants
# =============================================================================

# Timeout settings (in seconds)
DEFAULT_LLM_TIMEOUT = 300
MIN_LLM_TIMEOUT = 10
MAX_LLM_TIMEOUT = 600

# Retry settings
DEFAULT_MAX_RETRIES = 5
MIN_MAX_RETRIES = 0
MAX_MAX_RETRIES = 10

# Temperature settings
DEFAULT_TEMPERATURE = 0.7
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0

# Token settings
DEFAULT_MAX_TOKENS = 4096
MIN_MAX_TOKENS = 100
MAX_MAX_TOKENS = 32000

# =============================================================================
# Generation Constants
# =============================================================================

# Coverage targets
DEFAULT_TARGET_COVERAGE = 0.8
MIN_TARGET_COVERAGE = 0.0
MAX_TARGET_COVERAGE = 1.0

# Iteration limits
DEFAULT_MAX_ITERATIONS = 10
MIN_MAX_ITERATIONS = 1
MAX_MAX_ITERATIONS = 50

# Aider settings
DEFAULT_AIDER_MAX_ATTEMPTS = 3
DEFAULT_AIDER_TIMEOUT = 120

# =============================================================================
# Cache Constants
# =============================================================================

# File cache settings
DEFAULT_CACHE_SIZE = 100
MAX_CACHE_SIZE = 1000
DEFAULT_CACHE_TTL = 3600  # 1 hour in seconds

# =============================================================================
# UI Constants
# =============================================================================

# Window sizes
DEFAULT_WINDOW_WIDTH = 1200
DEFAULT_WINDOW_HEIGHT = 800
MIN_WINDOW_WIDTH = 800
MIN_WINDOW_HEIGHT = 600

# Status colors
STATUS_COLORS = {
    "IDLE": "#666666",
    "PARSING": "#2196F3",      # Blue
    "GENERATING": "#9C27B0",   # Purple
    "COMPILING": "#FF9800",    # Orange
    "TESTING": "#4CAF50",      # Green
    "FIXING": "#F44336",       # Red
    "ANALYZING": "#00BCD4",    # Cyan
    "OPTIMIZING": "#FFEB3B",   # Yellow
    "COMPLETED": "#4CAF50",    # Green
    "FAILED": "#F44336",       # Red
    "PAUSED": "#FFC107",       # Amber
}

# =============================================================================
# Maven Constants
# =============================================================================

# Maven goals
MAVEN_GOAL_TEST = "test"
MAVEN_GOAL_COMPILE = "compile"
MAVEN_GOAL_CLEAN = "clean"
MAVEN_GOAL_VERIFY = "verify"

# Maven options
MAVEN_OPTION_QUIET = "-q"
MAVEN_OPTION_BATCH = "-B"
MAVEN_OPTION_SKIP_TESTS = "-DskipTests"

# Default paths
DEFAULT_SOURCE_DIR = "src/main/java"
DEFAULT_TEST_DIR = "src/test/java"
DEFAULT_TARGET_DIR = "target"
DEFAULT_CLASSES_DIR = "target/classes"
DEFAULT_TEST_CLASSES_DIR = "target/test-classes"

# =============================================================================
# Java Constants
# =============================================================================

# Required imports for JUnit 5 tests
REQUIRED_JUNIT_IMPORTS = frozenset({
    'org.junit.jupiter.api.Test',
    'org.junit.jupiter.api.BeforeEach',
    'org.junit.jupiter.api.AfterEach',
    'org.junit.jupiter.api.BeforeAll',
    'org.junit.jupiter.api.AfterAll',
    'org.junit.jupiter.api.Assertions',
})

# Common assertion methods
COMMON_ASSERTION_METHODS = frozenset({
    'assertEquals',
    'assertTrue',
    'assertFalse',
    'assertNull',
    'assertNotNull',
    'assertThrows',
    'assertDoesNotThrow',
    'fail',
})

# =============================================================================
# File Constants
# =============================================================================

# Supported file extensions
JAVA_EXTENSION = ".java"
XML_EXTENSION = ".xml"
PROPERTIES_EXTENSION = ".properties"
YAML_EXTENSIONS = frozenset({".yml", ".yaml"})
JSON_EXTENSION = ".json"

# Encoding
DEFAULT_ENCODING = "utf-8"

# =============================================================================
# Security Constants
# =============================================================================

# Path validation
MAX_PATH_LENGTH = 4096
MAX_FILENAME_LENGTH = 255

# Shell metacharacters (dangerous characters that should not appear in paths)
SHELL_METACHARACTERS = frozenset({
    ';', '|', '&', '`', '$', '(', ')', '{', '}', '[', ']',
    '<', '>', '!', '#', '*', '?', '\\', '\n', '\r', '\x00'
})

# Sensitive field names (for logging)
SENSITIVE_FIELD_NAMES = frozenset({
    'api_key', 'apikey', 'api-key',
    'password', 'passwd', 'pwd',
    'secret', 'secret_key', 'secretkey',
    'token', 'access_token', 'refresh_token',
    'private_key', 'privatekey',
    'credential', 'credentials',
    'auth', 'authorization',
})

# =============================================================================
# Logging Constants
# =============================================================================

# Log formats
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEBUG_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"

# Log levels
LOG_LEVEL_DEBUG = "DEBUG"
LOG_LEVEL_INFO = "INFO"
LOG_LEVEL_WARNING = "WARNING"
LOG_LEVEL_ERROR = "ERROR"

# =============================================================================
# Retry Constants
# =============================================================================

# Exponential backoff defaults
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 60.0
DEFAULT_EXPONENTIAL_BASE = 2.0

# Circuit breaker defaults
DEFAULT_CIRCUIT_BREAKER_THRESHOLD = 5
DEFAULT_CIRCUIT_BREAKER_TIMEOUT = 60.0

# =============================================================================
# Vector Store Constants
# =============================================================================

# Embedding dimensions (depends on the model used)
DEFAULT_EMBEDDING_DIMENSION = 768

# Similarity thresholds
DEFAULT_SIMILARITY_THRESHOLD = 0.7
MIN_SIMILARITY_THRESHOLD = 0.0
MAX_SIMILARITY_THRESHOLD = 1.0

# =============================================================================
# Batch Processing Constants
# =============================================================================

# Concurrency limits
DEFAULT_MAX_CONCURRENT_GENERATIONS = 3
MIN_MAX_CONCURRENT_GENERATIONS = 1
MAX_MAX_CONCURRENT_GENERATIONS = 10

# Batch sizes
DEFAULT_BATCH_SIZE = 10
MAX_BATCH_SIZE = 100
