"""Code Analysis Agent for multi-agent collaboration.

Specialized agent for analyzing Java code and extracting test-relevant information.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
from pathlib import Path

from .specialized_agent import SpecializedAgent, AgentCapability, AgentTask
from .message_bus import MessageBus
from .shared_knowledge import SharedKnowledgeBase, ExperienceReplay

logger = logging.getLogger(__name__)


class CodeAnalysisAgent(SpecializedAgent):
    """Agent specialized in analyzing Java source code.
    
    Capabilities:
    - Parse and analyze Java classes
    - Extract method signatures and dependencies
    - Identify testable units
    - Analyze code complexity and coverage hotspots
    """
    
    def __init__(
        self,
        agent_id: str,
        message_bus: MessageBus,
        knowledge_base: SharedKnowledgeBase,
        experience_replay: Optional[ExperienceReplay] = None,
        java_parser=None
    ):
        """Initialize code analysis agent.
        
        Args:
            agent_id: Unique agent identifier
            message_bus: Message bus for communication
            knowledge_base: Shared knowledge base
            experience_replay: Optional experience replay buffer
            java_parser: Java code parser instance
        """
        super().__init__(
            agent_id=agent_id,
            capabilities={
                AgentCapability.DEPENDENCY_ANALYSIS,
                AgentCapability.TEST_DESIGN
            },
            message_bus=message_bus,
            knowledge_base=knowledge_base,
            experience_replay=experience_replay
        )
        
        self.java_parser = java_parser
        self._analysis_cache: Dict[str, Any] = {}
        
        logger.info(f"[CodeAnalysisAgent:{agent_id}] Initialized")
    
    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute code analysis task.
        
        Args:
            task: Task containing analysis parameters
            
        Returns:
            Analysis results
        """
        task_type = task.task_type
        payload = task.payload
        
        logger.info(f"[CodeAnalysisAgent:{self.agent_id}] Executing task: {task_type}")
        
        try:
            if task_type == "analyze_code":
                return await self._analyze_code(payload)
            elif task_type == "extract_methods":
                return await self._extract_methods(payload)
            elif task_type == "analyze_dependencies":
                return await self._analyze_dependencies(payload)
            elif task_type == "identify_test_targets":
                return await self._identify_test_targets(payload)
            else:
                return {
                    "success": False,
                    "error": f"Unknown task type: {task_type}"
                }
        except Exception as e:
            logger.exception(f"[CodeAnalysisAgent:{self.agent_id}] Task execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _analyze_code(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Java source code.
        
        Args:
            payload: Contains 'file_path' and optional 'source_code'
            
        Returns:
            Analysis results with class info, methods, fields
        """
        file_path = payload.get("file_path")
        source_code = payload.get("source_code")
        
        if not file_path:
            return {"success": False, "error": "No file_path provided"}
        
        # Check cache
        cache_key = f"{file_path}:{hash(source_code) if source_code else 'file'}"
        if cache_key in self._analysis_cache:
            logger.debug(f"[CodeAnalysisAgent:{self.agent_id}] Using cached analysis for {file_path}")
            return {
                "success": True,
                "output": self._analysis_cache[cache_key]
            }
        
        try:
            # Read file if source not provided
            if source_code is None:
                path = Path(file_path)
                if path.exists():
                    source_code = path.read_text(encoding='utf-8')
                else:
                    return {"success": False, "error": f"File not found: {file_path}"}
            
            # Parse code using java_parser if available
            analysis_result = await self._parse_java_code(file_path, source_code)
            
            # Cache result
            self._analysis_cache[cache_key] = analysis_result
            
            # Share knowledge
            self.share_knowledge(
                item_type="code_analysis",
                content={
                    "file_path": file_path,
                    "class_name": analysis_result.get("class_name"),
                    "method_count": len(analysis_result.get("methods", [])),
                    "complexity": analysis_result.get("complexity", 0)
                },
                confidence=0.9,
                tags=["code_analysis", "java"]
            )
            
            return {
                "success": True,
                "output": analysis_result
            }
            
        except Exception as e:
            logger.exception(f"Code analysis failed for {file_path}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _parse_java_code(self, file_path: str, source_code: str) -> Dict[str, Any]:
        """Parse Java code to extract structure.
        
        Args:
            file_path: Path to Java file
            source_code: Java source code
            
        Returns:
            Parsed code structure
        """
        result = {
            "file_path": file_path,
            "class_name": None,
            "package": None,
            "imports": [],
            "methods": [],
            "fields": [],
            "complexity": 0,
            "testability_score": 0.0
        }
        
        try:
            # Use java_parser if available
            if self.java_parser:
                parsed = await asyncio.get_event_loop().run_in_executor(
                    None, self._parse_with_parser, source_code
                )
                if parsed:
                    result.update(parsed)
            else:
                # Fallback: basic regex-based parsing
                result.update(self._basic_parse(source_code))
            
            # Calculate testability score
            result["testability_score"] = self._calculate_testability(result)
            
        except Exception as e:
            logger.warning(f"Parsing failed, using basic analysis: {e}")
            result.update(self._basic_parse(source_code))
        
        return result
    
    def _parse_with_parser(self, source_code: str) -> Optional[Dict[str, Any]]:
        """Parse code using tree-sitter Java parser.
        
        Args:
            source_code: Java source code
            
        Returns:
            Parsed structure or None if parsing fails
        """
        try:
            from tree_sitter import Language, Parser
            from tree_sitter_java import language
            
            # Initialize parser
            java_language = Language(language())
            parser = Parser(java_language)
            
            # Parse source code
            tree = parser.parse(source_code.encode('utf-8'))
            root_node = tree.root_node
            
            result = {
                "class_name": None,
                "package": None,
                "imports": [],
                "methods": [],
                "fields": [],
                "annotations": [],
                "complexity": 0,
                "line_count": len(source_code.split('\n'))
            }
            
            # Walk the AST
            self._walk_java_ast(root_node, source_code, result)
            
            # Calculate cyclomatic complexity
            result["complexity"] = self._calculate_cyclomatic_complexity(root_node, source_code)
            
            return result
            
        except ImportError:
            logger.debug("[CodeAnalysisAgent] tree-sitter not available, falling back to regex parsing")
            return None
        except Exception as e:
            logger.warning(f"[CodeAnalysisAgent] Tree-sitter parsing failed: {e}")
            return None
    
    def _walk_java_ast(self, node, source_code: str, result: Dict[str, Any]) -> None:
        """Walk Java AST and extract information.
        
        Args:
            node: Current AST node
            source_code: Original source code
            result: Result dictionary to populate
        """
        if node is None:
            return
        
        node_type = node.type
        node_text = source_code[node.start_byte:node.end_byte]
        
        # Extract package declaration
        if node_type == "package_declaration":
            # Find scoped_identifier child
            for child in node.children:
                if child.type == "scoped_identifier":
                    result["package"] = source_code[child.start_byte:child.end_byte]
                    break
        
        # Extract imports
        elif node_type == "import_declaration":
            # Find scoped_identifier child
            for child in node.children:
                if child.type == "scoped_identifier":
                    import_text = source_code[child.start_byte:child.end_byte]
                    result["imports"].append(import_text)
                    break
        
        # Extract class/interface/enum declaration
        elif node_type in ["class_declaration", "interface_declaration", "enum_declaration"]:
            # Find identifier child for class name
            for child in node.children:
                if child.type == "identifier":
                    result["class_name"] = source_code[child.start_byte:child.end_byte]
                    break
            
            # Extract class-level annotations
            for child in node.children:
                if child.type == "modifiers":
                    for modifier_child in child.children:
                        if modifier_child.type == "annotation":
                            annotation_text = source_code[modifier_child.start_byte:modifier_child.end_byte]
                            result["annotations"].append(annotation_text)
        
        # Extract method declarations
        elif node_type == "method_declaration":
            method_info = self._extract_method_info(node, source_code)
            if method_info:
                result["methods"].append(method_info)
        
        # Extract field declarations
        elif node_type == "field_declaration":
            field_info = self._extract_field_info(node, source_code)
            if field_info:
                result["fields"].append(field_info)
        
        # Recursively walk children
        for child in node.children:
            self._walk_java_ast(child, source_code, result)
    
    def _extract_method_info(self, node, source_code: str) -> Optional[Dict[str, Any]]:
        """Extract method information from AST node.
        
        Args:
            node: Method declaration node
            source_code: Original source code
            
        Returns:
            Method information dictionary
        """
        method_info = {
            "name": None,
            "return_type": "void",
            "parameters": [],
            "modifiers": [],
            "annotations": [],
            "signature": "",
            "line_start": node.start_point[0] + 1,
            "line_end": node.end_point[0] + 1
        }
        
        for child in node.children:
            child_type = child.type
            child_text = source_code[child.start_byte:child.end_byte]
            
            if child_type == "identifier":
                method_info["name"] = child_text
            elif child_type == "type_identifier" or child_type == "void_type":
                method_info["return_type"] = child_text
            elif child_type == "formal_parameters":
                # Extract parameters
                method_info["parameters"] = self._extract_parameters(child, source_code)
            elif child_type == "modifiers":
                # Extract modifiers and annotations
                for modifier in child.children:
                    if modifier.type == "annotation":
                        method_info["annotations"].append(source_code[modifier.start_byte:modifier.end_byte])
                    elif modifier.type in ["public", "private", "protected", "static", "final"]:
                        method_info["modifiers"].append(modifier.type)
        
        # Build signature
        if method_info["name"]:
            params_str = ", ".join([f"{p['type']} {p['name']}" for p in method_info["parameters"]])
            method_info["signature"] = f"{method_info['return_type']} {method_info['name']}({params_str})"
            return method_info
        
        return None
    
    def _extract_parameters(self, node, source_code: str) -> List[Dict[str, str]]:
        """Extract parameters from formal_parameters node.
        
        Args:
            node: Formal parameters node
            source_code: Original source code
            
        Returns:
            List of parameter dictionaries
        """
        parameters = []
        
        for child in node.children:
            if child.type == "formal_parameter":
                param_info = {"type": "", "name": ""}
                
                for param_child in child.children:
                    param_child_text = source_code[param_child.start_byte:param_child.end_byte]
                    
                    if param_child.type in ["type_identifier", "scoped_type_identifier", "generic_type"]:
                        param_info["type"] = param_child_text
                    elif param_child.type == "identifier":
                        param_info["name"] = param_child_text
                
                if param_info["name"]:
                    parameters.append(param_info)
        
        return parameters
    
    def _extract_field_info(self, node, source_code: str) -> Optional[Dict[str, Any]]:
        """Extract field information from AST node.
        
        Args:
            node: Field declaration node
            source_code: Original source code
            
        Returns:
            Field information dictionary
        """
        field_info = {
            "name": None,
            "type": None,
            "modifiers": [],
            "annotations": []
        }
        
        for child in node.children:
            child_type = child.type
            child_text = source_code[child.start_byte:child.end_byte]
            
            if child_type in ["type_identifier", "scoped_type_identifier", "generic_type"]:
                field_info["type"] = child_text
            elif child_type == "variable_declarator":
                # Extract field name
                for var_child in child.children:
                    if var_child.type == "identifier":
                        field_info["name"] = source_code[var_child.start_byte:var_child.end_byte]
                        break
            elif child_type == "modifiers":
                for modifier in child.children:
                    if modifier.type == "annotation":
                        field_info["annotations"].append(source_code[modifier.start_byte:modifier.end_byte])
                    elif modifier.type in ["public", "private", "protected", "static", "final"]:
                        field_info["modifiers"].append(modifier.type)
        
        if field_info["name"]:
            return field_info
        
        return None
    
    def _calculate_cyclomatic_complexity(self, node, source_code: str) -> int:
        """Calculate cyclomatic complexity from AST.
        
        Args:
            node: AST node
            source_code: Original source code
            
        Returns:
            Cyclomatic complexity value
        """
        complexity = 1  # Base complexity
        
        decision_nodes = [
            "if_statement",
            "while_statement",
            "do_statement",
            "for_statement",
            "enhanced_for_statement",
            "switch_statement",
            "catch_clause",
            "conditional_expression",
            "||",  # Logical OR
            "&&"   # Logical AND
        ]
        
        def count_decisions(n):
            nonlocal complexity
            if n.type in decision_nodes:
                complexity += 1
            for child in n.children:
                count_decisions(child)
        
        count_decisions(node)
        return complexity
    
    def _basic_parse(self, source_code: str) -> Dict[str, Any]:
        """Basic parsing using regex patterns.
        
        Args:
            source_code: Java source code
            
        Returns:
            Basic parsed structure
        """
        import re
        
        result = {
            "class_name": None,
            "package": None,
            "imports": [],
            "methods": [],
            "fields": [],
            "complexity": 0
        }
        
        # Extract package
        package_match = re.search(r'package\s+([\w.]+);', source_code)
        if package_match:
            result["package"] = package_match.group(1)
        
        # Extract imports
        result["imports"] = re.findall(r'import\s+([\w.]+);', source_code)
        
        # Extract class name
        class_match = re.search(r'(?:public\s+)?(?:class|interface|enum)\s+(\w+)', source_code)
        if class_match:
            result["class_name"] = class_match.group(1)
        
        # Extract methods
        method_pattern = r'(?:public|private|protected)?\s*(?:static\s+)?(?:<[^>]+>\s*)?(\w+[\w<>,\s]*)\s+(\w+)\s*\(([^)]*)\)'
        for match in re.finditer(method_pattern, source_code):
            method = {
                "return_type": match.group(1).strip(),
                "name": match.group(2),
                "parameters": match.group(3),
                "signature": f"{match.group(1).strip()} {match.group(2)}({match.group(3)})"
            }
            result["methods"].append(method)
        
        # Calculate complexity (simple: count of branches)
        result["complexity"] = (
            source_code.count('if') +
            source_code.count('for') +
            source_code.count('while') +
            source_code.count('switch') +
            source_code.count('catch')
        )
        
        return result
    
    def _calculate_testability(self, analysis: Dict[str, Any]) -> float:
        """Calculate testability score.
        
        Args:
            analysis: Code analysis result
            
        Returns:
            Testability score (0.0 - 1.0)
        """
        score = 0.5  # Base score
        
        # More methods = more testable (up to a point)
        method_count = len(analysis.get("methods", []))
        if method_count > 0:
            score += min(0.2, method_count * 0.02)
        
        # Lower complexity = more testable
        complexity = analysis.get("complexity", 0)
        if complexity < 5:
            score += 0.15
        elif complexity < 10:
            score += 0.05
        else:
            score -= min(0.2, (complexity - 10) * 0.01)
        
        # Has dependencies = may need mocking
        imports = analysis.get("imports", [])
        if len(imports) > 5:
            score -= 0.05
        
        return max(0.0, min(1.0, score))
    
    async def _extract_methods(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extract method information from code.
        
        Args:
            payload: Contains analysis result or file path
            
        Returns:
            Method information
        """
        analysis = payload.get("analysis")
        
        if not analysis:
            # First analyze the code
            file_path = payload.get("file_path")
            if file_path:
                result = await self._analyze_code({"file_path": file_path})
                if result["success"]:
                    analysis = result["output"]
        
        if not analysis:
            return {"success": False, "error": "No analysis available"}
        
        methods = analysis.get("methods", [])
        
        # Enhance method info with test relevance
        enhanced_methods = []
        for method in methods:
            enhanced = {
                **method,
                "test_priority": self._calculate_method_priority(method),
                "suggested_tests": self._suggest_tests_for_method(method)
            }
            enhanced_methods.append(enhanced)
        
        return {
            "success": True,
            "output": {
                "methods": enhanced_methods,
                "total_methods": len(methods),
                "testable_methods": len([m for m in enhanced_methods if m["test_priority"] > 0.5])
            }
        }
    
    def _calculate_method_priority(self, method: Dict[str, Any]) -> float:
        """Calculate test priority for a method.
        
        Args:
            method: Method information
            
        Returns:
            Priority score (0.0 - 1.0)
        """
        name = method.get("name", "").lower()
        return_type = method.get("return_type", "").lower()
        
        # Skip getters/setters (low priority)
        if name.startswith("get") or name.startswith("set") or name.startswith("is"):
            return 0.3
        
        # Public methods are higher priority
        if "public" in method.get("modifiers", ""):
            return 0.9
        
        # Methods with return values are good test targets
        if return_type != "void":
            return 0.8
        
        return 0.6
    
    def _suggest_tests_for_method(self, method: Dict[str, Any]) -> List[str]:
        """Suggest test cases for a method.
        
        Args:
            method: Method information
            
        Returns:
            List of suggested test descriptions
        """
        suggestions = []
        name = method.get("name", "")
        return_type = method.get("return_type", "")
        
        # Basic test suggestions
        suggestions.append(f"Test {name} with valid input")
        suggestions.append(f"Test {name} with null input")
        
        if "void" not in return_type:
            suggestions.append(f"Test {name} return value")
        
        # Add edge case suggestions based on method name
        if "save" in name.lower() or "create" in name.lower():
            suggestions.append(f"Test {name} with duplicate data")
        elif "delete" in name.lower() or "remove" in name.lower():
            suggestions.append(f"Test {name} with non-existent item")
        elif "find" in name.lower() or "get" in name.lower():
            suggestions.append(f"Test {name} with not found scenario")
        
        return suggestions
    
    async def _analyze_dependencies(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code dependencies.
        
        Args:
            payload: Contains file_path or analysis
            
        Returns:
            Dependency analysis
        """
        analysis = payload.get("analysis")
        
        if not analysis:
            file_path = payload.get("file_path")
            if file_path:
                result = await self._analyze_code({"file_path": file_path})
                if result["success"]:
                    analysis = result["output"]
        
        if not analysis:
            return {"success": False, "error": "No analysis available"}
        
        imports = analysis.get("imports", [])
        
        # Categorize dependencies
        external_deps = []
        internal_deps = []
        java_deps = []
        
        for imp in imports:
            if imp.startswith("java."):
                java_deps.append(imp)
            elif imp.startswith("javax.") or imp.startswith("org.") or imp.startswith("com."):
                external_deps.append(imp)
            else:
                internal_deps.append(imp)
        
        return {
            "success": True,
            "output": {
                "total_dependencies": len(imports),
                "external_dependencies": external_deps,
                "internal_dependencies": internal_deps,
                "java_standard_deps": java_deps,
                "needs_mocking": len(external_deps) > 0
            }
        }
    
    async def _identify_test_targets(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Identify high-value test targets.
        
        Args:
            payload: Contains project path or file list
            
        Returns:
            Prioritized test targets
        """
        file_paths = payload.get("file_paths", [])
        project_path = payload.get("project_path")
        
        if not file_paths and project_path:
            # Find all Java files in project
            project_dir = Path(project_path)
            file_paths = [
                str(f) for f in project_dir.rglob("*.java")
                if "test" not in str(f).lower() and "Test" not in str(f).name
            ]
        
        targets = []
        
        for file_path in file_paths:
            result = await self._analyze_code({"file_path": file_path})
            if result["success"]:
                analysis = result["output"]
                
                target = {
                    "file_path": file_path,
                    "class_name": analysis.get("class_name"),
                    "testability_score": analysis.get("testability_score", 0),
                    "method_count": len(analysis.get("methods", [])),
                    "complexity": analysis.get("complexity", 0),
                    "priority": 0.0
                }
                
                # Calculate priority
                target["priority"] = (
                    target["testability_score"] * 0.4 +
                    min(1.0, target["method_count"] / 10) * 0.3 +
                    (1.0 - min(1.0, target["complexity"] / 20)) * 0.3
                )
                
                targets.append(target)
        
        # Sort by priority
        targets.sort(key=lambda x: x["priority"], reverse=True)
        
        return {
            "success": True,
            "output": {
                "targets": targets[:20],  # Top 20 targets
                "total_files": len(file_paths),
                "high_priority_count": len([t for t in targets if t["priority"] > 0.7])
            }
        }
