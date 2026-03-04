"""Test fixtures package.

This package provides reusable test data and fixtures for PyUT Agent tests.
"""

from .java_code_samples import (
    SIMPLE_JAVA_CLASS,
    JAVA_CLASS_WITH_DEPENDENCIES,
    JAVA_INTERFACE,
    JAVA_ABSTRACT_CLASS,
    JAVA_ENUM,
    JAVA_EXCEPTION,
)

from .maven_projects import (
    MINIMAL_POM_XML,
    SPRING_BOOT_POM_XML,
    create_minimal_maven_project,
    create_spring_boot_project,
)

from .mock_objects import (
    create_mock_llm_client,
    create_mock_settings,
    create_mock_agent_worker,
)

__all__ = [
    # Java code samples
    "SIMPLE_JAVA_CLASS",
    "JAVA_CLASS_WITH_DEPENDENCIES",
    "JAVA_INTERFACE",
    "JAVA_ABSTRACT_CLASS",
    "JAVA_ENUM",
    "JAVA_EXCEPTION",
    # Maven projects
    "MINIMAL_POM_XML",
    "SPRING_BOOT_POM_XML",
    "create_minimal_maven_project",
    "create_spring_boot_project",
    # Mock objects
    "create_mock_llm_client",
    "create_mock_settings",
    "create_mock_agent_worker",
]
