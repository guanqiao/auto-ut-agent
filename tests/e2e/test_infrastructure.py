"""Simple test to verify E2E test infrastructure."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.e2e.utils import create_java_class, create_pom_xml, count_java_files


def create_temp_maven_project():
    """Create a temporary Maven project for testing."""
    tmpdir = tempfile.mkdtemp()
    project_path = Path(tmpdir)
    
    src_main = project_path / "src" / "main" / "java" / "com" / "example"
    src_main.mkdir(parents=True, exist_ok=True)
    
    src_test = project_path / "src" / "test" / "java" / "com" / "example"
    src_test.mkdir(parents=True, exist_ok=True)
    
    pom_xml = create_pom_xml()
    (project_path / "pom.xml").write_text(pom_xml)
    
    calculator_code = create_java_class(
        package="com.example",
        class_name="Calculator",
        methods=[
            "public int add(int a, int b) { return a + b; }",
            "public int subtract(int a, int b) { return a - b; }"
        ]
    )
    (src_main / "Calculator.java").write_text(calculator_code)
    
    return project_path


def test_project_creation():
    """Test that project creation works correctly."""
    print("Testing project creation...")
    
    project_path = create_temp_maven_project()
    print(f"✓ temp_maven_project created: {project_path}")
    assert project_path.exists()
    assert (project_path / "pom.xml").exists()
    assert (project_path / "src" / "main" / "java" / "com" / "example" / "Calculator.java").exists()
    
    java_count = count_java_files(project_path)
    print(f"✓ Found {java_count} Java files")
    assert java_count == 1
    
    print("✓ Project creation works correctly!")


def test_utils():
    """Test utility functions."""
    print("\nTesting utility functions...")
    
    java_code = create_java_class(
        package="com.example",
        class_name="TestService",
        methods=[
            "public String getName() { return \"Test\"; }",
            "public int calculate(int a, int b) { return a + b; }"
        ]
    )
    print("✓ create_java_class works")
    assert "package com.example" in java_code
    assert "class TestService" in java_code
    
    pom = create_pom_xml(
        group_id="com.test",
        artifact_id="test-app",
        version="1.0.0"
    )
    print("✓ create_pom_xml works")
    assert "<groupId>com.test</groupId>" in pom
    assert "<artifactId>test-app</artifactId>" in pom
    
    print("✓ All utility functions work correctly!")


def test_mock_llm():
    """Test mock LLM client."""
    print("\nTesting mock LLM client...")
    
    from tests.e2e.conftest import mock_llm_client
    
    client = mock_llm_client()
    
    async def test_generate():
        result = await client.agenerate("Generate test for Calculator")
        print(f"✓ Mock LLM generated code: {len(result)} chars")
        assert "CalculatorTest" in result
        assert "@Test" in result
    
    import asyncio
    asyncio.run(test_generate())
    
    print("✓ Mock LLM client works correctly!")


if __name__ == "__main__":
    print("=" * 60)
    print("E2E Test Infrastructure Verification")
    print("=" * 60)
    
    try:
        test_project_creation()
        test_utils()
        test_mock_llm()
        
        print("\n" + "=" * 60)
        print("✅ All E2E test infrastructure checks passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
