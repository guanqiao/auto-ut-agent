"""Mock object factories for testing.

This module provides factory functions for creating mock objects used in tests.
"""

from unittest.mock import AsyncMock, Mock, MagicMock
from typing import Optional, Any, Dict


def create_mock_llm_client(
    model: str = "gpt-4",
    generate_response: str = "generated test code",
    test_connection_result: tuple = (True, "Connection successful")
) -> Mock:
    """Create a mock LLM client.
    
    Args:
        model: Model name
        generate_response: Response for agenerate method
        test_connection_result: Result for test_connection method
        
    Returns:
        Mock LLM client
    """
    client = Mock()
    client.model = model
    client.agenerate = AsyncMock(return_value=generate_response)
    client.astream = AsyncMock(return_value=iter(["chunk1", "chunk2", "chunk3"]))
    client.test_connection = AsyncMock(return_value=test_connection_result)
    return client


def create_mock_settings(
    target_coverage: float = 0.8,
    max_iterations: int = 3,
    timeout_per_file: int = 300,
    maven_path: str = "mvn",
    java_home: str = "/usr/lib/jvm/java-11"
) -> Mock:
    """Create mock settings.
    
    Args:
        target_coverage: Target coverage percentage
        max_iterations: Maximum iterations
        timeout_per_file: Timeout per file in seconds
        maven_path: Maven executable path
        java_home: Java home path
        
    Returns:
        Mock settings object
    """
    settings = Mock()
    
    # Coverage settings
    coverage = Mock()
    coverage.target_coverage = target_coverage
    coverage.max_iterations = max_iterations
    coverage.timeout_per_file = timeout_per_file
    settings.coverage = coverage
    
    # Maven settings
    maven = Mock()
    maven.maven_path = maven_path
    settings.maven = maven
    
    # JDK settings
    jdk = Mock()
    jdk.java_home = java_home
    settings.jdk = jdk
    
    return settings


def create_mock_agent_worker(
    success: bool = True,
    test_file: str = "/path/to/Test.java",
    coverage: float = 0.85,
    iterations: int = 2,
    message: str = "Success"
) -> Mock:
    """Create a mock agent worker.
    
    Args:
        success: Whether the generation was successful
        test_file: Path to generated test file
        coverage: Coverage percentage
        iterations: Number of iterations
        message: Status message
        
    Returns:
        Mock agent worker
    """
    worker = Mock()
    worker.isRunning = Mock(return_value=False)
    worker.start = Mock()
    worker.stop = Mock()
    worker.pause = Mock()
    worker.resume = Mock()
    
    # Signals
    worker.progress_updated = Mock()
    worker.state_changed = Mock()
    worker.log_message = Mock()
    worker.completed = Mock()
    worker.error = Mock()
    worker.paused = Mock()
    worker.resumed = Mock()
    worker.terminated = Mock()
    
    # Result
    worker.result = {
        "success": success,
        "test_file": test_file,
        "coverage": coverage,
        "iterations": iterations,
        "message": message
    }
    
    return worker


def create_mock_java_parser(
    class_name: str = "TestClass",
    methods: Optional[list] = None,
    package: str = "com.example"
) -> Mock:
    """Create a mock Java parser.
    
    Args:
        class_name: Name of the parsed class
        methods: List of method dictionaries
        package: Package name
        
    Returns:
        Mock Java parser
    """
    if methods is None:
        methods = [
            {"name": "add", "return_type": "int", "parameters": ["int a", "int b"]},
            {"name": "subtract", "return_type": "int", "parameters": ["int a", "int b"]}
        ]
    
    parser = Mock()
    parser.parse_class = Mock(return_value={
        "name": class_name,
        "package": package,
        "methods": methods,
        "imports": [],
        "fields": []
    })
    parser.parse_file = Mock(return_value={
        "classes": [{"name": class_name, "methods": methods}]
    })
    
    return parser


def create_mock_maven_runner(
    compile_success: bool = True,
    test_success: bool = True,
    coverage_report: Optional[Dict[str, Any]] = None
) -> Mock:
    """Create a mock Maven runner.
    
    Args:
        compile_success: Whether compilation succeeds
        test_success: Whether tests pass
        coverage_report: Coverage report data
        
    Returns:
        Mock Maven runner
    """
    if coverage_report is None:
        coverage_report = {
            "line_coverage": 0.85,
            "branch_coverage": 0.75,
            "instruction_coverage": 0.80
        }
    
    runner = Mock()
    runner.compile = Mock(return_value=(compile_success, "" if compile_success else "Compilation error"))
    runner.test = Mock(return_value=(test_success, "" if test_success else "Test failed"))
    runner.get_coverage = Mock(return_value=coverage_report)
    
    return runner


def create_mock_test_history(
    records: Optional[list] = None
) -> Mock:
    """Create a mock test history.
    
    Args:
        records: List of history records
        
    Returns:
        Mock test history
    """
    if records is None:
        records = []
    
    history = Mock()
    history.records = records
    history.add_record = Mock()
    history.get_record = Mock(return_value=None)
    history.delete_record = Mock(return_value=True)
    history.clear = Mock()
    history.get_stats = Mock(return_value={
        "total": len(records),
        "successful": sum(1 for r in records if r.get("status") == "success"),
        "failed": sum(1 for r in records if r.get("status") == "failed"),
        "success_rate": 0.8,
        "avg_coverage": 0.75,
        "avg_duration": 120.0
    })
    
    return history


def create_mock_notification_manager() -> Mock:
    """Create a mock notification manager.
    
    Returns:
        Mock notification manager
    """
    manager = Mock()
    manager.show = Mock()
    manager.show_success = Mock()
    manager.show_warning = Mock()
    manager.show_error = Mock()
    manager.show_info = Mock()
    manager.clear_all = Mock()
    
    return manager


def create_mock_style_manager(theme: str = "light") -> Mock:
    """Create a mock style manager.
    
    Args:
        theme: Current theme name
        
    Returns:
        Mock style manager
    """
    manager = Mock()
    manager.current_theme = theme
    manager.get_color = Mock(return_value="#2196F3")
    manager.get_font_family = Mock(return_value="Arial")
    manager.get_stylesheet = Mock(return_value="")
    manager.apply_stylesheet = Mock()
    manager.set_theme = Mock(return_value=True)
    
    return manager
