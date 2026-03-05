"""Test Fix Agent for multi-agent collaboration.

Specialized agent for fixing compilation and test execution errors.
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Any, Set
from pathlib import Path

from .specialized_agent import SpecializedAgent, AgentCapability, AgentTask
from .message_bus import MessageBus
from .shared_knowledge import SharedKnowledgeBase, ExperienceReplay

logger = logging.getLogger(__name__)


class TestFixAgent(SpecializedAgent):
    """Agent specialized in fixing test errors.
    
    Capabilities:
    - Fix compilation errors
    - Fix test failures
    - Fix import issues
    - Fix mock configuration problems
    """
    
    def __init__(
        self,
        agent_id: str,
        message_bus: MessageBus,
        knowledge_base: SharedKnowledgeBase,
        experience_replay: Optional[ExperienceReplay] = None,
        llm_client=None
    ):
        """Initialize test fix agent.
        
        Args:
            agent_id: Unique agent identifier
            message_bus: Message bus for communication
            knowledge_base: Shared knowledge base
            experience_replay: Optional experience replay buffer
            llm_client: LLM client for fix generation
        """
        super().__init__(
            agent_id=agent_id,
            capabilities={
                AgentCapability.ERROR_FIXING,
                AgentCapability.TEST_REVIEW
            },
            message_bus=message_bus,
            knowledge_base=knowledge_base,
            experience_replay=experience_replay
        )
        
        self.llm_client = llm_client
        self._fix_cache: Dict[str, Any] = {}
        
        logger.info(f"[TestFixAgent:{agent_id}] Initialized")
    
    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute test fix task.
        
        Args:
            task: Task containing fix parameters
            
        Returns:
            Fix results
        """
        task_type = task.task_type
        payload = task.payload
        
        logger.info(f"[TestFixAgent:{self.agent_id}] Executing task: {task_type}")
        
        try:
            if task_type == "fix_compilation_error":
                return await self._fix_compilation_error(payload)
            elif task_type == "fix_test_failure":
                return await self._fix_test_failure(payload)
            elif task_type == "fix_import_error":
                return await self._fix_import_error(payload)
            elif task_type == "fix_mock_error":
                return await self._fix_mock_error(payload)
            elif task_type == "analyze_error":
                return await self._analyze_error(payload)
            else:
                return {
                    "success": False,
                    "error": f"Unknown task type: {task_type}"
                }
        except Exception as e:
            logger.exception(f"[TestFixAgent:{self.agent_id}] Task execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _fix_compilation_error(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Fix compilation errors in test code.
        
        Args:
            payload: Contains error_info, test_code, and source_code
            
        Returns:
            Fixed test code
        """
        error_info = payload.get("error_info", {})
        test_code = payload.get("test_code", "")
        source_code = payload.get("source_code", "")
        class_info = payload.get("class_info", {})
        
        if not test_code:
            return {"success": False, "error": "No test code provided"}
        
        error_message = error_info.get("message", "")
        error_line = error_info.get("line", 0)
        
        logger.info(f"[TestFixAgent:{self.agent_id}] Fixing compilation error: {error_message[:100]}...")
        
        try:
            # Analyze error type
            error_type = self._classify_compilation_error(error_message)
            
            # Try rule-based fixes first
            fixed_code = self._apply_rule_based_fix(test_code, error_type, error_message, error_line)
            
            # If rule-based fix didn't work, use LLM
            if fixed_code == test_code and self.llm_client:
                fixed_code = await self._generate_fix_with_llm(
                    test_code=test_code,
                    source_code=source_code,
                    error_message=error_message,
                    error_type=error_type,
                    class_info=class_info
                )
            
            # Verify the fix is different
            if fixed_code != test_code:
                result = {
                    "success": True,
                    "output": {
                        "fixed_code": fixed_code,
                        "error_type": error_type,
                        "changes_made": self._calculate_diff(test_code, fixed_code),
                        "fix_strategy": "rule_based" if fixed_code != test_code else "llm"
                    }
                }
                
                # Share knowledge
                self.share_knowledge(
                    item_type="error_fix",
                    content={
                        "error_type": error_type,
                        "error_message": error_message[:200],
                        "fix_applied": True
                    },
                    confidence=0.8,
                    tags=["error_fix", "compilation", error_type]
                )
                
                return result
            else:
                return {
                    "success": False,
                    "error": "Could not fix compilation error",
                    "error_type": error_type
                }
                
        except Exception as e:
            logger.exception(f"Failed to fix compilation error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _fix_test_failure(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Fix test execution failures.
        
        Args:
            payload: Contains failure_info, test_code, and source_code
            
        Returns:
            Fixed test code
        """
        failure_info = payload.get("failure_info", {})
        test_code = payload.get("test_code", "")
        source_code = payload.get("source_code", "")
        
        failure_message = failure_info.get("message", "")
        test_method = failure_info.get("test_method", "")
        
        logger.info(f"[TestFixAgent:{self.agent_id}] Fixing test failure in {test_method}")
        
        try:
            # Classify failure type
            failure_type = self._classify_test_failure(failure_message)
            
            # Apply rule-based fixes
            fixed_code = self._apply_failure_fix(test_code, failure_type, failure_message, test_method)
            
            # If needed, use LLM
            if fixed_code == test_code and self.llm_client:
                fixed_code = await self._generate_failure_fix_with_llm(
                    test_code=test_code,
                    source_code=source_code,
                    failure_message=failure_message,
                    failure_type=failure_type,
                    test_method=test_method
                )
            
            if fixed_code != test_code:
                return {
                    "success": True,
                    "output": {
                        "fixed_code": fixed_code,
                        "failure_type": failure_type,
                        "changes_made": self._calculate_diff(test_code, fixed_code)
                    }
                }
            else:
                return {
                    "success": False,
                    "error": "Could not fix test failure",
                    "failure_type": failure_type
                }
                
        except Exception as e:
            logger.exception(f"Failed to fix test failure: {e}")
            return {"success": False, "error": str(e)}
    
    async def _fix_import_error(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Fix import-related errors.
        
        Args:
            payload: Contains error_info and test_code
            
        Returns:
            Fixed test code with correct imports
        """
        error_info = payload.get("error_info", {})
        test_code = payload.get("test_code", "")
        available_classes = payload.get("available_classes", [])
        
        error_message = error_info.get("message", "")
        
        logger.info(f"[TestFixAgent:{self.agent_id}] Fixing import error")
        
        try:
            # Extract missing class from error
            missing_class = self._extract_missing_class(error_message)
            
            if missing_class:
                # Find the correct import
                correct_import = self._find_correct_import(missing_class, available_classes)
                
                if correct_import:
                    fixed_code = self._add_import(test_code, correct_import)
                    
                    return {
                        "success": True,
                        "output": {
                            "fixed_code": fixed_code,
                            "added_import": correct_import,
                            "missing_class": missing_class
                        }
                    }
            
            # Try to fix common import issues
            fixed_code = self._fix_common_imports(test_code)
            
            if fixed_code != test_code:
                return {
                    "success": True,
                    "output": {
                        "fixed_code": fixed_code,
                        "fix_type": "common_import_fix"
                    }
                }
            
            return {
                "success": False,
                "error": "Could not resolve import error"
            }
            
        except Exception as e:
            logger.exception(f"Failed to fix import error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _fix_mock_error(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Fix mock configuration errors.
        
        Args:
            payload: Contains error_info and test_code
            
        Returns:
            Fixed test code with correct mock configuration
        """
        error_info = payload.get("error_info", {})
        test_code = payload.get("test_code", "")
        
        error_message = error_info.get("message", "")
        
        logger.info(f"[TestFixAgent:{self.agent_id}] Fixing mock error")
        
        try:
            # Apply mock-specific fixes
            fixed_code = self._apply_mock_fixes(test_code, error_message)
            
            if fixed_code != test_code:
                return {
                    "success": True,
                    "output": {
                        "fixed_code": fixed_code,
                        "fix_type": "mock_configuration"
                    }
                }
            
            return {
                "success": False,
                "error": "Could not fix mock error"
            }
            
        except Exception as e:
            logger.exception(f"Failed to fix mock error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _analyze_error(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze error and provide diagnostic information.
        
        Args:
            payload: Contains error_info and context
            
        Returns:
            Error analysis
        """
        error_info = payload.get("error_info", {})
        context = payload.get("context", {})
        
        error_message = error_info.get("message", "")
        error_type = error_info.get("type", "unknown")
        
        analysis = {
            "error_type": error_type,
            "error_category": self._classify_error_category(error_message),
            "severity": self._determine_severity(error_message),
            "likely_causes": self._identify_likely_causes(error_message, context),
            "suggested_fixes": self._suggest_fixes(error_message, error_type),
            "requires_llm": len(error_message) > 200 or error_type == "complex"
        }
        
        return {
            "success": True,
            "output": analysis
        }
    
    def _classify_compilation_error(self, error_message: str) -> str:
        """Classify compilation error type.
        
        Args:
            error_message: Error message
            
        Returns:
            Error type
        """
        error_lower = error_message.lower()
        
        if "cannot find symbol" in error_lower:
            return "symbol_not_found"
        elif "cannot be applied to" in error_lower:
            return "method_signature_mismatch"
        elif "incompatible types" in error_lower:
            return "type_mismatch"
        elif "package does not exist" in error_lower:
            return "missing_import"
        elif "has private access" in error_lower:
            return "access_violation"
        elif "non-static" in error_lower and "static context" in error_lower:
            return "static_context_error"
        elif "exception" in error_lower and "never thrown" in error_lower:
            return "unnecessary_exception"
        elif "variable might not have been initialized" in error_lower:
            return "uninitialized_variable"
        else:
            return "unknown_compilation_error"
    
    def _classify_test_failure(self, failure_message: str) -> str:
        """Classify test failure type.
        
        Args:
            failure_message: Failure message
            
        Returns:
            Failure type
        """
        message_lower = failure_message.lower()
        
        if "assertionfailed" in message_lower or "expected:" in message_lower:
            return "assertion_failure"
        elif "nullpointer" in message_lower:
            return "null_pointer"
        elif "indexoutofbounds" in message_lower:
            return "index_out_of_bounds"
        elif "illegalargument" in message_lower:
            return "illegal_argument"
        elif "classcast" in message_lower:
            return "class_cast"
        elif "timeout" in message_lower:
            return "timeout"
        elif "mock" in message_lower:
            return "mock_failure"
        else:
            return "unknown_failure"
    
    def _classify_error_category(self, error_message: str) -> str:
        """Classify error into high-level category.
        
        Args:
            error_message: Error message
            
        Returns:
            Error category
        """
        error_lower = error_message.lower()
        
        if any(x in error_lower for x in ["syntax", "unexpected", "expected"]):
            return "syntax"
        elif any(x in error_lower for x in ["symbol", "cannot find", "not found"]):
            return "reference"
        elif any(x in error_lower for x in ["type", "cast", "convert"]):
            return "type"
        elif any(x in error_lower for x in ["access", "private", "protected"]):
            return "access"
        elif any(x in error_lower for x in ["null", "pointer"]):
            return "null_safety"
        else:
            return "other"
    
    def _apply_rule_based_fix(
        self,
        test_code: str,
        error_type: str,
        error_message: str,
        error_line: int
    ) -> str:
        """Apply rule-based fix for compilation error.
        
        Args:
            test_code: Test code to fix
            error_type: Type of error
            error_message: Error message
            error_line: Line number of error
            
        Returns:
            Fixed test code
        """
        fixed_code = test_code
        
        if error_type == "missing_import":
            # Extract package from error message
            match = re.search(r"package ([\w.]+) does not exist", error_message)
            if match:
                package = match.group(1)
                fixed_code = self._add_import(fixed_code, f"import {package}.*;")
        
        elif error_type == "static_context_error":
            # Add static import or fix method call
            fixed_code = self._fix_static_context(fixed_code, error_message)
        
        elif error_type == "access_violation":
            # Change to use getter or make accessible
            fixed_code = self._fix_access_violation(fixed_code, error_message)
        
        elif error_type == "unnecessary_exception":
            # Remove unnecessary throws declaration
            fixed_code = self._remove_unnecessary_throws(fixed_code, error_message)
        
        return fixed_code
    
    def _apply_failure_fix(
        self,
        test_code: str,
        failure_type: str,
        failure_message: str,
        test_method: str
    ) -> str:
        """Apply fix for test failure.
        
        Args:
            test_code: Test code to fix
            failure_type: Type of failure
            failure_message: Failure message
            test_method: Test method name
            
        Returns:
            Fixed test code
        """
        fixed_code = test_code
        
        if failure_type == "null_pointer":
            # Add null check or initialization
            fixed_code = self._add_null_handling(fixed_code, test_method)
        
        elif failure_type == "assertion_failure":
            # Extract expected vs actual and suggest fix
            fixed_code = self._fix_assertion(fixed_code, failure_message, test_method)
        
        elif failure_type == "mock_failure":
            # Fix mock setup
            fixed_code = self._fix_mock_setup(fixed_code, failure_message)
        
        return fixed_code
    
    def _apply_mock_fixes(self, test_code: str, error_message: str) -> str:
        """Apply mock-specific fixes.
        
        Args:
            test_code: Test code to fix
            error_message: Error message
            
        Returns:
            Fixed test code
        """
        fixed_code = test_code
        error_lower = error_message.lower()
        
        # Fix missing mock initialization
        if "mock" in error_lower and "not initialized" in error_lower:
            if "MockitoAnnotations.openMocks" not in test_code:
                fixed_code = self._add_mockito_init(test_code)
        
        # Fix wrong mock method
        if "wrong type of return value" in error_lower:
            fixed_code = self._fix_mock_return_type(test_code, error_message)
        
        return fixed_code
    
    def _add_import(self, code: str, import_statement: str) -> str:
        """Add import statement to code.
        
        Args:
            code: Source code
            import_statement: Import statement to add
            
        Returns:
            Code with import added
        """
        lines = code.split('\n')
        
        # Find last import or package declaration
        insert_index = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('package '):
                insert_index = i + 1
        
        # Insert new import
        lines.insert(insert_index, import_statement)
        
        return '\n'.join(lines)
    
    def _fix_static_context(self, code: str, error_message: str) -> str:
        """Fix static context errors.
        
        Args:
            code: Source code
            error_message: Error message
            
        Returns:
            Fixed code
        """
        # Extract method name from error
        match = re.search(r"non-static method (\w+)", error_message)
        if match:
            method_name = match.group(1)
            # This is a complex fix that may require object instantiation
            # For now, add a comment indicating the issue
            return code.replace(
                f"{method_name}(",
                f"/* TODO: Fix static context - need instance to call */ {method_name}("
            )
        return code
    
    def _fix_access_violation(self, code: str, error_message: str) -> str:
        """Fix access violation errors.
        
        Args:
            code: Source code
            error_message: Error message
            
        Returns:
            Fixed code
        """
        # Extract field/method name
        match = re.search(r"([\w]+) has private access", error_message)
        if match:
            name = match.group(1)
            # Add reflection-based access or comment
            return code.replace(
                f".{name}",
                f"./* TODO: Fix access - {name} is private */get{name.capitalize()}()"
            )
        return code
    
    def _remove_unnecessary_throws(self, code: str, error_message: str) -> str:
        """Remove unnecessary throws declarations.
        
        Args:
            code: Source code
            error_message: Error message
            
        Returns:
            Fixed code
        """
        # Extract exception name
        match = re.search(r"exception (\w+) is never thrown", error_message)
        if match:
            exception_name = match.group(1)
            # Remove from throws clause
            pattern = rf"throws\s+[^{{{{]*?{exception_name}[^{{{{]*?{{"
            return re.sub(pattern, "{", code)
        return code
    
    def _add_null_handling(self, code: str, test_method: str) -> str:
        """Add null handling to test method.
        
        Args:
            code: Source code
            test_method: Test method name
            
        Returns:
            Fixed code
        """
        # Find the test method and add null check
        pattern = rf"(@Test\s+void\s+{test_method}\(\)\s*{{)"
        replacement = r"\1\n        // Added null handling\n        assertNotNull(target);"
        return re.sub(pattern, replacement, code)
    
    def _fix_assertion(self, code: str, failure_message: str, test_method: str) -> str:
        """Fix assertion in test method.
        
        Args:
            code: Source code
            failure_message: Failure message
            test_method: Test method name
            
        Returns:
            Fixed code
        """
        # Extract expected and actual values
        match = re.search(r"expected:\s*<(.*?)>\s+but was:\s*<(.*?)>", failure_message)
        if match:
            expected = match.group(1)
            actual = match.group(2)
            # Add comment about the mismatch
            return code.replace(
                f"void {test_method}(",
                f"/* TODO: Fix assertion - expected: {expected}, actual: {actual} */\n    void {test_method}("
            )
        return code
    
    def _fix_mock_setup(self, code: str, failure_message: str) -> str:
        """Fix mock setup issues.
        
        Args:
            code: Source code
            failure_message: Failure message
            
        Returns:
            Fixed code
        """
        # Add mockito init if missing
        if "@BeforeEach" in code and "MockitoAnnotations" not in code:
            return self._add_mockito_init(code)
        return code
    
    def _add_mockito_init(self, code: str) -> str:
        """Add Mockito initialization to setup method.
        
        Args:
            code: Source code
            
        Returns:
            Code with mockito init added
        """
        if "MockitoAnnotations.openMocks(this)" not in code:
            # Find @BeforeEach method and add init
            pattern = r"(@BeforeEach\s+void\s+setUp\(\)\s*{{)"
            replacement = r"\1\n        MockitoAnnotations.openMocks(this);"
            return re.sub(pattern, replacement, code)
        return code
    
    def _fix_mock_return_type(self, code: str, error_message: str) -> str:
        """Fix mock return type issues.
        
        Args:
            code: Source code
            error_message: Error message
            
        Returns:
            Fixed code
        """
        # This would require more sophisticated analysis
        # For now, add a comment
        return code.replace(
            "when(",
            "/* TODO: Fix mock return type */\n        when("
        )
    
    def _fix_common_imports(self, code: str) -> str:
        """Fix common import issues.
        
        Args:
            code: Source code
            
        Returns:
            Fixed code
        """
        # Add common JUnit imports if missing
        if "@Test" in code and "import org.junit" not in code:
            code = "import org.junit.jupiter.api.Test;\n" + code
        
        if "@BeforeEach" in code and "import org.junit.jupiter.api.BeforeEach" not in code:
            code = "import org.junit.jupiter.api.BeforeEach;\n" + code
        
        if "assert" in code and "import static org.junit.jupiter.api.Assertions" not in code:
            code = "import static org.junit.jupiter.api.Assertions.*;\n" + code
        
        if "when(" in code and "import static org.mockito.Mockito" not in code:
            code = "import static org.mockito.Mockito.*;\n" + code
        
        return code
    
    def _extract_missing_class(self, error_message: str) -> Optional[str]:
        """Extract missing class name from error message.
        
        Args:
            error_message: Error message
            
        Returns:
            Missing class name or None
        """
        match = re.search(r"cannot find symbol\s+.*?class\s+(\w+)", error_message, re.DOTALL)
        if match:
            return match.group(1)
        
        match = re.search(r"package ([\w.]+) does not exist", error_message)
        if match:
            return match.group(1).split('.')[-1]
        
        return None
    
    def _find_correct_import(self, class_name: str, available_classes: List[str]) -> Optional[str]:
        """Find correct import for a class.
        
        Args:
            class_name: Class name to find
            available_classes: List of available classes with full package
            
        Returns:
            Correct import statement or None
        """
        for full_class in available_classes:
            if full_class.endswith(f".{class_name}"):
                return f"import {full_class};"
        
        # Common framework classes
        common_imports = {
            "List": "java.util.List",
            "ArrayList": "java.util.ArrayList",
            "Map": "java.util.Map",
            "HashMap": "java.util.HashMap",
            "Set": "java.util.Set",
            "Optional": "java.util.Optional",
            "Mock": "org.mockito.Mock",
            "InjectMocks": "org.mockito.InjectMocks",
        }
        
        if class_name in common_imports:
            return f"import {common_imports[class_name]};"
        
        return None
    
    def _determine_severity(self, error_message: str) -> str:
        """Determine error severity.
        
        Args:
            error_message: Error message
            
        Returns:
            Severity level
        """
        error_lower = error_message.lower()
        
        if any(x in error_lower for x in ["fatal", "critical"]):
            return "critical"
        elif any(x in error_lower for x in ["cannot find", "does not exist", "required"]):
            return "high"
        elif any(x in error_lower for x in ["deprecated", "warning"]):
            return "low"
        else:
            return "medium"
    
    def _identify_likely_causes(self, error_message: str, context: Dict) -> List[str]:
        """Identify likely causes of error.
        
        Args:
            error_message: Error message
            context: Error context
            
        Returns:
            List of likely causes
        """
        causes = []
        error_lower = error_message.lower()
        
        if "cannot find symbol" in error_lower:
            causes.append("Missing import statement")
            causes.append("Typo in class/method name")
            causes.append("Dependency not in classpath")
        
        if "incompatible types" in error_lower:
            causes.append("Wrong variable type assigned")
            causes.append("Missing type conversion")
        
        if "null" in error_lower:
            causes.append("Object not initialized")
            causes.append("Method returned null unexpectedly")
        
        return causes
    
    def _suggest_fixes(self, error_message: str, error_type: str) -> List[str]:
        """Suggest fixes for error.
        
        Args:
            error_message: Error message
            error_type: Error type
            
        Returns:
            List of suggested fixes
        """
        suggestions = []
        
        if error_type == "symbol_not_found":
            suggestions.append("Add missing import statement")
            suggestions.append("Check for typos in class name")
        
        elif error_type == "type_mismatch":
            suggestions.append("Add explicit type casting")
            suggestions.append("Change variable type declaration")
        
        elif error_type == "null_pointer":
            suggestions.append("Add null check before using object")
            suggestions.append("Initialize object in setUp method")
        
        return suggestions
    
    async def _generate_fix_with_llm(
        self,
        test_code: str,
        source_code: str,
        error_message: str,
        error_type: str,
        class_info: Dict
    ) -> str:
        """Generate fix using LLM.
        
        Args:
            test_code: Test code to fix
            source_code: Source code of class under test
            error_message: Error message
            error_type: Error type
            class_info: Class information
            
        Returns:
            Fixed code
        """
        if not self.llm_client:
            return test_code
        
        prompt = f"""Fix the following compilation error in the test code:

Error Type: {error_type}
Error Message: {error_message}

Source Code (Class Under Test):
```java
{source_code[:1000]}
```

Test Code to Fix:
```java
{test_code}
```

Please provide the fixed test code only, without explanations.
"""
        
        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                temperature=0.2,
                max_tokens=1500
            )
            
            fixed_code = response.get("content", test_code)
            fixed_code = self._extract_code_from_markdown(fixed_code)
            
            return fixed_code
            
        except Exception as e:
            logger.error(f"LLM fix generation failed: {e}")
            return test_code
    
    async def _generate_failure_fix_with_llm(
        self,
        test_code: str,
        source_code: str,
        failure_message: str,
        failure_type: str,
        test_method: str
    ) -> str:
        """Generate fix for test failure using LLM.
        
        Args:
            test_code: Test code to fix
            source_code: Source code of class under test
            failure_message: Failure message
            failure_type: Failure type
            test_method: Test method name
            
        Returns:
            Fixed code
        """
        if not self.llm_client:
            return test_code
        
        prompt = f"""Fix the following test failure:

Failure Type: {failure_type}
Failure Message: {failure_message}
Test Method: {test_method}

Source Code (Class Under Test):
```java
{source_code[:1000]}
```

Test Code to Fix:
```java
{test_code}
```

Please provide the fixed test code only, without explanations.
"""
        
        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                temperature=0.2,
                max_tokens=1500
            )
            
            fixed_code = response.get("content", test_code)
            fixed_code = self._extract_code_from_markdown(fixed_code)
            
            return fixed_code
            
        except Exception as e:
            logger.error(f"LLM failure fix generation failed: {e}")
            return test_code
    
    def _calculate_diff(self, original: str, fixed: str) -> List[Dict]:
        """Calculate differences between original and fixed code.
        
        Args:
            original: Original code
            fixed: Fixed code
            
        Returns:
            List of changes
        """
        import difflib
        
        original_lines = original.split('\n')
        fixed_lines = fixed.split('\n')
        
        diff = list(difflib.unified_diff(original_lines, fixed_lines, lineterm=''))
        
        changes = []
        for line in diff:
            if line.startswith('+') and not line.startswith('+++'):
                changes.append({"type": "added", "content": line[1:]})
            elif line.startswith('-') and not line.startswith('---'):
                changes.append({"type": "removed", "content": line[1:]})
        
        return changes
    
    def _extract_code_from_markdown(self, text: str) -> str:
        """Extract code from markdown code blocks.
        
        Args:
            text: Text that may contain markdown code blocks
            
        Returns:
            Extracted code
        """
        match = re.search(r'```(?:java)?\s*\n(.*?)\n```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()
