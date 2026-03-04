"""Maven project fixtures for testing.

This module provides functions to create Maven project structures for testing.
"""

from pathlib import Path
from typing import Optional


# Minimal pom.xml content
MINIMAL_POM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    
    <groupId>com.example</groupId>
    <artifactId>test-project</artifactId>
    <version>1.0.0</version>
    <packaging>jar</packaging>
    
    <properties>
        <maven.compiler.source>11</maven.compiler.source>
        <maven.compiler.target>11</maven.compiler.target>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    </properties>
    
    <dependencies>
        <dependency>
            <groupId>org.junit.jupiter</groupId>
            <artifactId>junit-jupiter</artifactId>
            <version>5.9.0</version>
            <scope>test</scope>
        </dependency>
    </dependencies>
    
    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-surefire-plugin</artifactId>
                <version>3.0.0</version>
            </plugin>
        </plugins>
    </build>
</project>
"""

# Spring Boot pom.xml content
SPRING_BOOT_POM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    
    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>3.1.0</version>
        <relativePath/>
    </parent>
    
    <groupId>com.example</groupId>
    <artifactId>spring-boot-test</artifactId>
    <version>1.0.0</version>
    <packaging>jar</packaging>
    
    <properties>
        <java.version>17</java.version>
    </properties>
    
    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-data-jpa</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-test</artifactId>
            <scope>test</scope>
        </dependency>
    </dependencies>
    
    <build>
        <plugins>
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
            </plugin>
        </plugins>
    </build>
</project>
"""


def create_minimal_maven_project(
    base_path: Path,
    group_id: str = "com.example",
    artifact_id: str = "test-project",
    version: str = "1.0.0"
) -> Path:
    """Create a minimal Maven project structure.
    
    Args:
        base_path: Base directory for the project
        group_id: Maven group ID
        artifact_id: Maven artifact ID
        version: Project version
        
    Returns:
        Path to the created project
    """
    # Create directory structure
    src_main = base_path / "src" / "main" / "java" / "com" / "example"
    src_test = base_path / "src" / "test" / "java" / "com" / "example"
    src_resources = base_path / "src" / "main" / "resources"
    test_resources = base_path / "src" / "test" / "resources"
    
    src_main.mkdir(parents=True, exist_ok=True)
    src_test.mkdir(parents=True, exist_ok=True)
    src_resources.mkdir(parents=True, exist_ok=True)
    test_resources.mkdir(parents=True, exist_ok=True)
    
    # Create pom.xml
    pom_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    
    <groupId>{group_id}</groupId>
    <artifactId>{artifact_id}</artifactId>
    <version>{version}</version>
    <packaging>jar</packaging>
    
    <properties>
        <maven.compiler.source>11</maven.compiler.source>
        <maven.compiler.target>11</maven.compiler.target>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    </properties>
    
    <dependencies>
        <dependency>
            <groupId>org.junit.jupiter</groupId>
            <artifactId>junit-jupiter</artifactId>
            <version>5.9.0</version>
            <scope>test</scope>
        </dependency>
    </dependencies>
</project>
"""
    (base_path / "pom.xml").write_text(pom_content)
    
    return base_path


def create_spring_boot_project(
    base_path: Path,
    group_id: str = "com.example",
    artifact_id: str = "spring-boot-app",
    version: str = "1.0.0"
) -> Path:
    """Create a Spring Boot project structure.
    
    Args:
        base_path: Base directory for the project
        group_id: Maven group ID
        artifact_id: Maven artifact ID
        version: Project version
        
    Returns:
        Path to the created project
    """
    # Create directory structure
    src_main = base_path / "src" / "main" / "java" / "com" / "example"
    src_test = base_path / "src" / "test" / "java" / "com" / "example"
    src_resources = base_path / "src" / "main" / "resources"
    test_resources = base_path / "src" / "test" / "resources"
    
    src_main.mkdir(parents=True, exist_ok=True)
    src_test.mkdir(parents=True, exist_ok=True)
    src_resources.mkdir(parents=True, exist_ok=True)
    test_resources.mkdir(parents=True, exist_ok=True)
    
    # Create pom.xml
    pom_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    
    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>3.1.0</version>
        <relativePath/>
    </parent>
    
    <groupId>{group_id}</groupId>
    <artifactId>{artifact_id}</artifactId>
    <version>{version}</version>
    <packaging>jar</packaging>
    
    <properties>
        <java.version>17</java.version>
    </properties>
    
    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-test</artifactId>
            <scope>test</scope>
        </dependency>
    </dependencies>
    
    <build>
        <plugins>
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
            </plugin>
        </plugins>
    </build>
</project>
"""
    (base_path / "pom.xml").write_text(pom_content)
    
    # Create application.properties
    app_properties = """# Spring Boot Application Properties
spring.application.name=test-app
server.port=8080
"""
    (src_resources / "application.properties").write_text(app_properties)
    
    # Create main application class
    main_class = f"""package com.example;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class Application {{
    public static void main(String[] args) {{
        SpringApplication.run(Application.class, args);
    }}
}}
"""
    (src_main / "Application.java").write_text(main_class)
    
    return base_path


def add_java_file(
    project_path: Path,
    class_name: str,
    content: str,
    package: str = "com.example"
) -> Path:
    """Add a Java file to a Maven project.
    
    Args:
        project_path: Path to the Maven project
        class_name: Name of the Java class (without .java extension)
        content: Java code content
        package: Java package name
        
    Returns:
        Path to the created Java file
    """
    # Convert package to path
    package_path = package.replace(".", "/")
    src_dir = project_path / "src" / "main" / "java" / package_path
    src_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Java file
    java_file = src_dir / f"{class_name}.java"
    java_file.write_text(content)
    
    return java_file


def add_test_file(
    project_path: Path,
    class_name: str,
    content: str,
    package: str = "com.example"
) -> Path:
    """Add a test Java file to a Maven project.
    
    Args:
        project_path: Path to the Maven project
        class_name: Name of the test class (without .java extension)
        content: Java code content
        package: Java package name
        
    Returns:
        Path to the created test file
    """
    # Convert package to path
    package_path = package.replace(".", "/")
    test_dir = project_path / "src" / "test" / "java" / package_path
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test file
    test_file = test_dir / f"{class_name}.java"
    test_file.write_text(content)
    
    return test_file
