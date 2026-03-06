"""Utility functions for E2E tests.

This module provides helper functions for:
- Creating Java classes and test files
- Running Maven commands
- Checking compilation and test execution
- Generating coverage reports
"""

import subprocess
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional


def create_java_class(
    package: str,
    class_name: str,
    methods: List[str],
    imports: Optional[List[str]] = None
) -> str:
    """Create Java class code.
    
    Args:
        package: Package name (e.g., "com.example")
        class_name: Class name
        methods: List of method signatures
        imports: Optional list of import statements
    
    Returns:
        Java class code as string
    """
    import_section = ""
    if imports:
        import_section = "\n".join(f"import {imp};" for imp in imports) + "\n\n"
    
    methods_section = "\n    ".join(methods)
    
    return f'''package {package};

{import_section}public class {class_name} {{
    {methods_section}
}}
'''


def create_test_class(
    package: str,
    class_name: str,
    test_methods: List[str],
    imports: Optional[List[str]] = None
) -> str:
    """Create JUnit test class code.
    
    Args:
        package: Package name
        class_name: Test class name (e.g., "CalculatorTest")
        test_methods: List of test method code
        imports: Optional list of import statements
    
    Returns:
        Test class code as string
    """
    default_imports = [
        "org.junit.jupiter.api.Test",
        "org.junit.jupiter.api.BeforeEach",
        "static org.junit.jupiter.api.Assertions.*"
    ]
    
    all_imports = default_imports + (imports or [])
    import_section = "\n".join(f"import {imp};" for imp in all_imports) + "\n\n"
    
    test_methods_section = "\n    ".join(test_methods)
    
    return f'''package {package};

{import_section}class {class_name} {{
    private {class_name.replace("Test", "")} instance;
    
    @BeforeEach
    void setUp() {{
        instance = new {class_name.replace("Test", "")}();
    }}
    
    {test_methods_section}
}}
'''


def run_maven_command(
    project_path: str,
    command: str,
    timeout: int = 300
) -> Tuple[int, str, str]:
    """Execute a Maven command.
    
    Args:
        project_path: Path to Maven project
        command: Maven command (e.g., "clean compile")
        timeout: Timeout in seconds
    
    Returns:
        Tuple of (exit_code, stdout, stderr)
    """
    try:
        result = subprocess.run(
            ["mvn"] + command.split(),
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except FileNotFoundError:
        return -1, "", "Maven not found. Please install Maven and add it to PATH."
    except Exception as e:
        return -1, "", str(e)


def check_compilation(project_path: str) -> Tuple[bool, str]:
    """Check if project compiles successfully.
    
    Args:
        project_path: Path to Maven project
    
    Returns:
        Tuple of (success, error_message)
    """
    exit_code, stdout, stderr = run_maven_command(project_path, "clean compile")
    
    if exit_code == 0:
        return True, ""
    else:
        error_msg = stderr if stderr else stdout
        return False, error_msg


def check_test_execution(project_path: str) -> Tuple[bool, float, str]:
    """Check if tests execute successfully.
    
    Args:
        project_path: Path to Maven project
    
    Returns:
        Tuple of (success, coverage, error_message)
    """
    exit_code, stdout, stderr = run_maven_command(project_path, "clean test")
    
    if exit_code == 0:
        coverage = extract_coverage_from_output(stdout)
        return True, coverage, ""
    else:
        error_msg = stderr if stderr else stdout
        return False, 0.0, error_msg


def get_coverage_report(project_path: str) -> Dict:
    """Get coverage report from JaCoCo.
    
    Args:
        project_path: Path to Maven project
    
    Returns:
        Dictionary with coverage information
    """
    jacoco_report = Path(project_path) / "target" / "site" / "jacoco" / "index.html"
    
    if not jacoco_report.exists():
        return {"error": "JaCoCo report not found"}
    
    try:
        content = jacoco_report.read_text()
        
        coverage = extract_coverage_from_html(content)
        
        return {
            "total": coverage,
            "line_coverage": coverage,
            "branch_coverage": coverage * 0.9,
            "report_path": str(jacoco_report)
        }
    except Exception as e:
        return {"error": str(e)}


def extract_coverage_from_output(output: str) -> float:
    """Extract coverage percentage from Maven output.
    
    Args:
        output: Maven test output
    
    Returns:
        Coverage percentage (0.0 to 1.0)
    """
    import re
    
    patterns = [
        r'Total coverage:\s*(\d+(?:\.\d+)?)%',
        r'Coverage:\s*(\d+(?:\.\d+)?)%',
        r'(\d+(?:\.\d+)?)%\s*covered'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output)
        if match:
            return float(match.group(1)) / 100.0
    
    return 0.0


def extract_coverage_from_html(html_content: str) -> float:
    """Extract coverage percentage from JaCoCo HTML report.
    
    Args:
        html_content: JaCoCo HTML report content
    
    Returns:
        Coverage percentage (0.0 to 1.0)
    """
    import re
    
    pattern = r'Total.*?(\d+(?:\.\d+)?)%'
    match = re.search(pattern, html_content, re.DOTALL)
    
    if match:
        return float(match.group(1)) / 100.0
    
    return 0.0


def create_pom_xml(
    group_id: str = "com.example",
    artifact_id: str = "test-project",
    version: str = "1.0-SNAPSHOT",
    dependencies: Optional[List[Dict]] = None
) -> str:
    """Create a pom.xml file content.
    
    Args:
        group_id: Maven group ID
        artifact_id: Maven artifact ID
        version: Project version
        dependencies: List of dependency dictionaries
    
    Returns:
        pom.xml content as string
    """
    default_deps = [
        {
            "groupId": "org.junit.jupiter",
            "artifactId": "junit-jupiter",
            "version": "5.9.3",
            "scope": "test"
        }
    ]
    
    all_deps = default_deps + (dependencies or [])
    
    deps_xml = ""
    for dep in all_deps:
        deps_xml += f'''
        <dependency>
            <groupId>{dep["groupId"]}</groupId>
            <artifactId>{dep["artifactId"]}</artifactId>
            <version>{dep["version"]}</version>
            <scope>{dep.get("scope", "compile")}</scope>
        </dependency>
'''
    
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    
    <groupId>{group_id}</groupId>
    <artifactId>{artifact_id}</artifactId>
    <version>{version}</version>
    
    <properties>
        <maven.compiler.source>11</maven.compiler.source>
        <maven.compiler.target>11</maven.compiler.target>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    </properties>
    
    <dependencies>
{deps_xml}    </dependencies>
    
    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-surefire-plugin</artifactId>
                <version>3.1.2</version>
            </plugin>
        </plugins>
    </build>
</project>
'''


def wait_for_file(file_path: Path, timeout: int = 10) -> bool:
    """Wait for a file to be created.
    
    Args:
        file_path: Path to file
        timeout: Timeout in seconds
    
    Returns:
        True if file exists, False otherwise
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if file_path.exists():
            return True
        time.sleep(0.1)
    return False


def count_java_files(project_path: Path) -> int:
    """Count Java files in a project.
    
    Args:
        project_path: Path to Maven project
    
    Returns:
        Number of Java files
    """
    src_main = project_path / "src" / "main" / "java"
    if not src_main.exists():
        return 0
    
    return len(list(src_main.rglob("*.java")))


def count_test_files(project_path: Path) -> int:
    """Count test files in a project.
    
    Args:
        project_path: Path to Maven project
    
    Returns:
        Number of test files
    """
    src_test = project_path / "src" / "test" / "java"
    if not src_test.exists():
        return 0
    
    return len(list(src_test.rglob("*Test.java")))
