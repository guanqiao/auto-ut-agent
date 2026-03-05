"""Constants and default configuration values for PyUT Agent."""


class AgentConstants:
    """Constants for agent behavior and configuration."""
    
    # Context Management
    DEFAULT_MAX_TOKENS = 8000
    DEFAULT_TARGET_TOKENS = 6000
    DEFAULT_COMPRESSION_THRESHOLD = 0.8
    
    # Retry Configuration
    DEFAULT_BASE_DELAY = 2.0
    DEFAULT_MAX_DELAY = 30.0
    DEFAULT_MAX_RETRIES = 100
    DEFAULT_BACKOFF_FACTOR = 2.0
    
    # Circuit Breaker
    DEFAULT_FAILURE_THRESHOLD = 5
    DEFAULT_RECOVERY_TIMEOUT = 60.0
    
    # Timeout Configuration
    DEFAULT_TIMEOUT_SECONDS = 300
    DEFAULT_TEST_TIMEOUT = 120
    
    # Coverage
    DEFAULT_TARGET_COVERAGE = 0.8
    DEFAULT_MAX_ITERATIONS = 50
    
    # Subtask Execution
    DEFAULT_MAX_SUBTASKS = 50
    DEFAULT_BATCH_SIZE = 10
    DEFAULT_CHECKPOINT_INTERVAL = 5
    
    # LLM Generation
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS_LLM = 4096
    
    # Memory Management
    DEFAULT_MAX_HISTORY_SIZE = 100
    DEFAULT_MEMORY_CLEANUP_INTERVAL = 3600  # seconds
    
    # File Operations
    DEFAULT_ENCODING = "utf-8"
    MAX_FILE_SIZE_MB = 10
    
    # UI
    DEFAULT_WINDOW_WIDTH = 1400
    DEFAULT_WINDOW_HEIGHT = 900
    DEFAULT_FONT_SIZE = 10


class ErrorMessages:
    """Standard error messages."""
    
    MAVEN_NOT_FOUND = "Maven not found. Please ensure mvn is in PATH."
    COMPILATION_FAILED = "Compilation failed. Please check the error messages."
    TEST_EXECUTION_FAILED = "Test execution failed."
    COVERAGE_ANALYSIS_FAILED = "Failed to analyze coverage report."
    LLM_GENERATION_FAILED = "LLM generation failed."
    FILE_NOT_FOUND = "File not found: {path}"
    INVALID_CONFIGURATION = "Invalid configuration: {reason}"


class LogMessages:
    """Standard log message templates."""
    
    AGENT_STARTING = "[Agent] Starting test generation - Target: {target}"
    AGENT_COMPLETED = "[Agent] Test generation completed - Duration: {duration:.2f}s"
    LLM_CALL_START = "[LLM] Starting {operation} - Model: {model}"
    LLM_CALL_COMPLETE = "[LLM] {operation} completed - Duration: {duration:.2f}s"
    RETRY_ATTEMPT = "[Retry] Attempt {attempt}/{max_attempts} - Operation: {operation}"
    FALLBACK_ACTIVATED = "[Fallback] Activated - Level: {level}, Reason: {reason}"
