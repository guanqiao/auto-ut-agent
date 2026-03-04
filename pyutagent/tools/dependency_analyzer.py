"""LLM-powered dependency analyzer for intelligent dependency detection.

This module uses LLM to analyze compilation errors and identify missing dependencies
with complete Maven coordinates, including groupId, artifactId, version, and scope.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DependencyAnalyzer:
    """LLM-powered dependency analyzer.
    
    Uses LLM to intelligently analyze compilation errors and identify
    missing dependencies with complete Maven coordinates.
    
    Features:
    - LLM-based intelligent analysis
    - Complete Maven coordinate generation
    - Version recommendation
    - Scope determination
    - Confidence scoring
    
    Example:
        >>> analyzer = DependencyAnalyzer(llm_client, prompt_builder)
        >>> result = await analyzer.analyze_missing_dependencies(
        ...     compiler_output="package org.junit.jupiter.api does not exist",
        ...     current_pom_content="<dependencies>...</dependencies>"
        ... )
        >>> print(result['missing_dependencies'])
    """
    
    def __init__(self, llm_client: Optional[Any] = None, prompt_builder: Optional[Any] = None):
        """Initialize dependency analyzer.
        
        Args:
            llm_client: LLM client for intelligent analysis
            prompt_builder: Prompt builder for creating analysis prompts
        """
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder
        
        logger.debug("[DependencyAnalyzer] Initialized")
    
    async def analyze_missing_dependencies(
        self, 
        compiler_output: str,
        current_pom_content: str
    ) -> Dict[str, Any]:
        """Analyze compilation errors and identify missing dependencies.
        
        Args:
            compiler_output: Compilation error output
            current_pom_content: Current pom.xml content
            
        Returns:
            Analysis result dictionary:
            {
                "missing_dependencies": [
                    {
                        "group_id": "org.junit.jupiter",
                        "artifact_id": "junit-jupiter",
                        "version": "5.10.0",
                        "scope": "test",
                        "reason": "JUnit 5 testing framework"
                    }
                ],
                "confidence": 0.95,
                "analysis": "Brief analysis",
                "suggested_fixes": ["..."]
            }
        """
        if self.llm_client and self.prompt_builder:
            return await self._llm_analysis(compiler_output, current_pom_content)
        else:
            return self._rule_based_analysis(compiler_output, current_pom_content)
    
    async def _llm_analysis(
        self, 
        compiler_output: str,
        current_pom_content: str
    ) -> Dict[str, Any]:
        """Perform LLM-based dependency analysis.
        
        Args:
            compiler_output: Compilation error output
            current_pom_content: Current pom.xml content
            
        Returns:
            Analysis result dictionary
        """
        try:
            prompt = self.build_dependency_analysis_prompt(
                compiler_output,
                current_pom_content
            )
            
            response = await self.llm_client.agenerate(prompt)
            
            result = self._parse_llm_response(response)
            
            logger.info(f"[DependencyAnalyzer] LLM analysis found {len(result.get('missing_dependencies', []))} dependencies")
            return result
            
        except Exception as e:
            logger.error(f"[DependencyAnalyzer] LLM analysis failed: {e}")
            return self._rule_based_analysis(compiler_output, current_pom_content)
    
    def _rule_based_analysis(
        self, 
        compiler_output: str,
        current_pom_content: str
    ) -> Dict[str, Any]:
        """Perform rule-based dependency analysis (fallback).
        
        Args:
            compiler_output: Compilation error output
            current_pom_content: Current pom.xml content
            
        Returns:
            Analysis result dictionary
        """
        missing_packages = self._extract_missing_packages(compiler_output)
        
        dependencies = []
        for package in missing_packages:
            dep_info = self._map_package_to_dependency(package)
            if dep_info:
                dependencies.append(dep_info)
        
        return {
            "missing_dependencies": dependencies,
            "confidence": 0.6,
            "analysis": f"Found {len(dependencies)} missing dependencies using rule-based analysis",
            "suggested_fixes": []
        }
    
    def _extract_missing_packages(self, compiler_output: str) -> List[str]:
        """Extract missing package names from compiler output.
        
        Args:
            compiler_output: Compiler error output
            
        Returns:
            List of missing package names
        """
        packages = []
        
        patterns = [
            r"package\s+([\w.]+)\s+does not exist",
            r"cannot find symbol.*package\s+([\w.]+)",
            r"import\s+([\w.]+)\s*;",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, compiler_output, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if match and match not in packages:
                    packages.append(match)
        
        return packages
    
    def _map_package_to_dependency(self, package: str) -> Optional[Dict[str, str]]:
        """Map a package name to Maven dependency coordinates.
        
        Args:
            package: Package name
            
        Returns:
            Dependency information dictionary or None
        """
        package_mappings = {
            "org.junit": {
                "group_id": "org.junit.jupiter",
                "artifact_id": "junit-jupiter",
                "version": "5.10.0",
                "scope": "test",
                "reason": "JUnit 5 testing framework"
            },
            "org.junit.jupiter": {
                "group_id": "org.junit.jupiter",
                "artifact_id": "junit-jupiter",
                "version": "5.10.0",
                "scope": "test",
                "reason": "JUnit 5 testing framework"
            },
            "org.mockito": {
                "group_id": "org.mockito",
                "artifact_id": "mockito-core",
                "version": "5.8.0",
                "scope": "test",
                "reason": "Mockito mocking framework"
            },
            "org.assertj": {
                "group_id": "org.assertj",
                "artifact_id": "assertj-core",
                "version": "3.25.0",
                "scope": "test",
                "reason": "AssertJ assertion library"
            },
            "org.hamcrest": {
                "group_id": "org.hamcrest",
                "artifact_id": "hamcrest",
                "version": "2.2",
                "scope": "test",
                "reason": "Hamcrest matcher library"
            },
            "org.powermock": {
                "group_id": "org.powermock",
                "artifact_id": "powermock-module-junit4",
                "version": "2.0.9",
                "scope": "test",
                "reason": "PowerMock testing framework"
            },
            "org.springframework": {
                "group_id": "org.springframework",
                "artifact_id": "spring-context",
                "version": "6.1.0",
                "scope": "compile",
                "reason": "Spring Framework"
            },
            "org.springframework.boot": {
                "group_id": "org.springframework.boot",
                "artifact_id": "spring-boot-starter-test",
                "version": "3.2.0",
                "scope": "test",
                "reason": "Spring Boot Test starter"
            },
            "jakarta.persistence": {
                "group_id": "jakarta.persistence",
                "artifact_id": "jakarta.persistence-api",
                "version": "3.1.0",
                "scope": "compile",
                "reason": "Jakarta Persistence API"
            },
            "javax.persistence": {
                "group_id": "javax.persistence",
                "artifact_id": "javax.persistence-api",
                "version": "2.2",
                "scope": "compile",
                "reason": "Java Persistence API"
            },
            "org.hibernate": {
                "group_id": "org.hibernate",
                "artifact_id": "hibernate-core",
                "version": "6.4.0",
                "scope": "compile",
                "reason": "Hibernate ORM"
            },
            "com.fasterxml.jackson": {
                "group_id": "com.fasterxml.jackson.core",
                "artifact_id": "jackson-databind",
                "version": "2.16.0",
                "scope": "compile",
                "reason": "Jackson JSON processor"
            },
            "org.slf4j": {
                "group_id": "org.slf4j",
                "artifact_id": "slf4j-api",
                "version": "2.0.9",
                "scope": "compile",
                "reason": "SLF4J logging API"
            },
            "ch.qos.logback": {
                "group_id": "ch.qos.logback",
                "artifact_id": "logback-classic",
                "version": "1.4.11",
                "scope": "compile",
                "reason": "Logback logging framework"
            },
            "org.apache.commons": {
                "group_id": "org.apache.commons",
                "artifact_id": "commons-lang3",
                "version": "3.14.0",
                "scope": "compile",
                "reason": "Apache Commons Lang"
            },
            "com.google.guava": {
                "group_id": "com.google.guava",
                "artifact_id": "guava",
                "version": "33.0.0",
                "scope": "compile",
                "reason": "Google Guava library"
            },
        }
        
        for prefix, dep_info in package_mappings.items():
            if package.startswith(prefix):
                return dep_info.copy()
        
        return None
    
    def build_dependency_analysis_prompt(
        self,
        compiler_output: str,
        current_pom_content: str
    ) -> str:
        """Build prompt for dependency analysis.
        
        Args:
            compiler_output: Compilation error output
            current_pom_content: Current pom.xml content
            
        Returns:
            Prompt string
        """
        return f"""You are a Maven dependency expert. Analyze the following compilation errors and identify missing dependencies.

Compilation Errors:
```
{compiler_output}
```

Current pom.xml:
```
{current_pom_content}
```

Task:
1. Identify all missing dependencies from the compilation errors
2. For each missing dependency, provide complete Maven coordinates
3. Determine the appropriate scope (test, compile, provided, runtime)
4. Recommend stable versions

Output in JSON format:
{{
  "missing_dependencies": [
    {{
      "group_id": "org.junit.jupiter",
      "artifact_id": "junit-jupiter",
      "version": "5.10.0",
      "scope": "test",
      "reason": "JUnit 5 testing framework"
    }}
  ],
  "confidence": 0.95,
  "analysis": "Brief analysis of missing dependencies",
  "suggested_fixes": ["Additional suggestions for fixing the errors"]
}}

Important:
- Use latest stable versions for common libraries
- Test dependencies should have scope "test"
- Be precise with groupId and artifactId
- If uncertain, set lower confidence score
- Only output the JSON, no additional text"""
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract dependency information.
        
        Args:
            response: LLM response string
            
        Returns:
            Parsed result dictionary
        """
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                
                if 'missing_dependencies' not in result:
                    result['missing_dependencies'] = []
                
                if 'confidence' not in result:
                    result['confidence'] = 0.5
                
                if 'analysis' not in result:
                    result['analysis'] = ""
                
                if 'suggested_fixes' not in result:
                    result['suggested_fixes'] = []
                
                return result
            else:
                logger.warning("[DependencyAnalyzer] No JSON found in LLM response")
                return {
                    "missing_dependencies": [],
                    "confidence": 0.3,
                    "analysis": "Failed to parse LLM response",
                    "suggested_fixes": []
                }
                
        except json.JSONDecodeError as e:
            logger.error(f"[DependencyAnalyzer] Failed to parse JSON: {e}")
            return {
                "missing_dependencies": [],
                "confidence": 0.3,
                "analysis": f"JSON parsing error: {e}",
                "suggested_fixes": []
            }
        except Exception as e:
            logger.error(f"[DependencyAnalyzer] Failed to parse LLM response: {e}")
            return {
                "missing_dependencies": [],
                "confidence": 0.3,
                "analysis": f"Parsing error: {e}",
                "suggested_fixes": []
            }
    
    def validate_dependency(self, dependency: Dict[str, str]) -> Tuple[bool, List[str]]:
        """Validate a dependency dictionary.
        
        Args:
            dependency: Dependency information dictionary
            
        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []
        
        required_fields = ['group_id', 'artifact_id', 'version']
        for field in required_fields:
            if field not in dependency or not dependency[field]:
                errors.append(f"Missing or empty required field: {field}")
        
        if 'scope' in dependency:
            valid_scopes = ['compile', 'test', 'provided', 'runtime', 'system', 'import']
            if dependency['scope'] not in valid_scopes:
                errors.append(f"Invalid scope: {dependency['scope']}")
        
        if 'group_id' in dependency:
            if not re.match(r'^[\w.]+$', dependency['group_id']):
                errors.append(f"Invalid group_id format: {dependency['group_id']}")
        
        if 'artifact_id' in dependency:
            if not re.match(r'^[\w-]+$', dependency['artifact_id']):
                errors.append(f"Invalid artifact_id format: {dependency['artifact_id']}")
        
        if 'version' in dependency:
            if not re.match(r'^[\w.-]+$', dependency['version']):
                errors.append(f"Invalid version format: {dependency['version']}")
        
        return len(errors) == 0, errors
    
    def deduplicate_dependencies(
        self, 
        dependencies: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Remove duplicate dependencies.
        
        Args:
            dependencies: List of dependency dictionaries
            
        Returns:
            Deduplicated list of dependencies
        """
        seen = set()
        unique_deps = []
        
        for dep in dependencies:
            key = f"{dep.get('group_id', '')}:{dep.get('artifact_id', '')}"
            
            if key not in seen:
                seen.add(key)
                unique_deps.append(dep)
        
        return unique_deps


from typing import Tuple
