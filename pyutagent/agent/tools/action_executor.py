"""Action Executor - Execute LLM-recommended actions for error recovery.

This module provides intelligent action execution based on LLM analysis results.
It translates LLM recommendations into concrete tool calls and code modifications.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of actions that can be executed."""
    FIX_IMPORTS = auto()
    ADD_DEPENDENCY = auto()
    FIX_SYNTAX = auto()
    FIX_TYPE_ERROR = auto()
    REGENERATE_TEST = auto()
    FIX_TEST_LOGIC = auto()
    ADD_MOCK = auto()
    FIX_ASSERTION = auto()
    SKIP_TEST = auto()
    MODIFY_CODE = auto()
    INSTALL_DEPENDENCY = auto()
    RESOLVE_DEPENDENCY = auto()
    UNKNOWN = auto()


@dataclass
class ActionResult:
    """Result of an action execution."""
    action_type: ActionType
    success: bool
    message: str = ""
    modified_code: str = ""
    modified_file: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ActionPlan:
    """A plan of actions to execute."""
    actions: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.5
    source: str = "llm"


class ActionExecutor:
    """Executes LLM-recommended actions for error recovery.
    
    This class translates LLM recommendations into concrete operations:
    - Code modifications (fix imports, syntax, logic)
    - Dependency management (add to pom.xml, resolve)
    - Test adjustments (skip, regenerate, fix)
    """
    
    ACTION_TYPE_MAP = {
        'fix_imports': ActionType.FIX_IMPORTS,
        'add_import': ActionType.FIX_IMPORTS,
        'add_imports': ActionType.FIX_IMPORTS,
        'add_dependency': ActionType.ADD_DEPENDENCY,
        'add_dependencies': ActionType.ADD_DEPENDENCY,
        'fix_syntax': ActionType.FIX_SYNTAX,
        'fix_syntax_error': ActionType.FIX_SYNTAX,
        'fix_type_error': ActionType.FIX_TYPE_ERROR,
        'fix_type': ActionType.FIX_TYPE_ERROR,
        'regenerate_test': ActionType.REGENERATE_TEST,
        'regenerate': ActionType.REGENERATE_TEST,
        'fix_test_logic': ActionType.FIX_TEST_LOGIC,
        'fix_logic': ActionType.FIX_TEST_LOGIC,
        'add_mock': ActionType.ADD_MOCK,
        'add_mocks': ActionType.ADD_MOCK,
        'fix_mock': ActionType.ADD_MOCK,
        'fix_assertion': ActionType.FIX_ASSERTION,
        'fix_assertions': ActionType.FIX_ASSERTION,
        'skip_test': ActionType.SKIP_TEST,
        'skip_tests': ActionType.SKIP_TEST,
        'modify_code': ActionType.MODIFY_CODE,
        'apply_fix': ActionType.MODIFY_CODE,
        'install_dependency': ActionType.INSTALL_DEPENDENCY,
        'install_dependencies': ActionType.INSTALL_DEPENDENCY,
        'resolve_dependency': ActionType.RESOLVE_DEPENDENCY,
        'resolve_dependencies': ActionType.RESOLVE_DEPENDENCY,
    }
    
    def __init__(
        self,
        project_path: str = "",
        llm_client: Optional[Any] = None,
        progress_callback: Optional[Callable[[str, str], None]] = None
    ):
        self.project_path = project_path
        self.llm_client = llm_client
        self.progress_callback = progress_callback
        
        self._execution_history: List[ActionResult] = []
        self._action_success_rates: Dict[ActionType, Dict[str, int]] = {}
    
    def get_available_actions(self) -> List[str]:
        return list(self.ACTION_TYPE_MAP.keys())
    
    def parse_action_type(self, action_str: str) -> ActionType:
        action_lower = action_str.lower().replace(' ', '_').replace('-', '_')
        return self.ACTION_TYPE_MAP.get(action_lower, ActionType.UNKNOWN)
    
    async def execute_action(
        self,
        action: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ActionResult:
        action_type_str = action.get('action', action.get('type', 'unknown'))
        action_type = self.parse_action_type(action_type_str)
        
        logger.info(f"[ActionExecutor] Executing action: {action_type.name}")
        
        if self.progress_callback:
            self.progress_callback("EXECUTING_ACTION", f"Executing: {action_type.name}")
        
        result = await self._execute_action_internal(action_type, action, context or {})
        
        self._execution_history.append(result)
        self._update_success_rate(result)
        
        return result
    
    async def execute_action_plan(
        self,
        plan: ActionPlan,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ActionResult]:
        results = []
        
        logger.info(
            f"[ActionExecutor] Executing action plan with {len(plan.actions)} actions "
            f"(confidence: {plan.confidence:.2f})"
        )
        
        for i, action in enumerate(plan.actions):
            logger.info(f"[ActionExecutor] Action {i+1}/{len(plan.actions)}: {action.get('action', 'unknown')}")
            
            result = await self.execute_action(action, context)
            results.append(result)
            
            if not result.success and action.get('critical', False):
                logger.warning(f"[ActionExecutor] Critical action failed, stopping plan execution")
                break
            
            if not result.success:
                logger.warning(f"[ActionExecutor] Action {i+1} failed: {result.message}")
            
            await asyncio.sleep(0.1)
        
        success_count = sum(1 for r in results if r.success)
        logger.info(
            f"[ActionExecutor] Action plan complete - "
            f"Success: {success_count}/{len(results)}"
        )
        
        return results
    
    async def _execute_action_internal(
        self,
        action_type: ActionType,
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ActionResult:
        try:
            if action_type == ActionType.FIX_IMPORTS:
                return await self._fix_imports(action, context)
            elif action_type == ActionType.ADD_DEPENDENCY:
                return await self._add_dependency(action, context)
            elif action_type == ActionType.FIX_SYNTAX:
                return await self._fix_syntax(action, context)
            elif action_type == ActionType.FIX_TYPE_ERROR:
                return await self._fix_type_error(action, context)
            elif action_type == ActionType.REGENERATE_TEST:
                return await self._regenerate_test(action, context)
            elif action_type == ActionType.FIX_TEST_LOGIC:
                return await self._fix_test_logic(action, context)
            elif action_type == ActionType.ADD_MOCK:
                return await self._add_mock(action, context)
            elif action_type == ActionType.FIX_ASSERTION:
                return await self._fix_assertion(action, context)
            elif action_type == ActionType.SKIP_TEST:
                return await self._skip_test(action, context)
            elif action_type == ActionType.MODIFY_CODE:
                return await self._modify_code(action, context)
            elif action_type == ActionType.INSTALL_DEPENDENCY:
                return await self._install_dependency(action, context)
            elif action_type == ActionType.RESOLVE_DEPENDENCY:
                return await self._resolve_dependency(action, context)
            else:
                return ActionResult(
                    action_type=ActionType.UNKNOWN,
                    success=False,
                    message=f"Unknown action type: {action_type}"
                )
        except Exception as e:
            logger.exception(f"[ActionExecutor] Action execution failed: {e}")
            return ActionResult(
                action_type=action_type,
                success=False,
                message=f"Exception: {str(e)}"
            )
    
    async def _fix_imports(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ActionResult:
        imports_to_add = action.get('imports', action.get('import_list', []))
        test_file = action.get('file') or context.get('test_file', '')
        
        if not imports_to_add:
            return ActionResult(
                action_type=ActionType.FIX_IMPORTS,
                success=False,
                message="No imports specified to add"
            )
        
        if not test_file:
            return ActionResult(
                action_type=ActionType.FIX_IMPORTS,
                success=False,
                message="No test file specified"
            )
        
        try:
            test_path = Path(self.project_path) / test_file if self.project_path else Path(test_file)
            
            if not test_path.exists():
                return ActionResult(
                    action_type=ActionType.FIX_IMPORTS,
                    success=False,
                    message=f"Test file not found: {test_file}"
                )
            
            content = test_path.read_text(encoding='utf-8')
            
            existing_imports = set(re.findall(r'import\s+([\w.]+);', content))
            
            new_imports = []
            for imp in imports_to_add:
                imp_str = imp if imp.endswith(';') else f"{imp};"
                if not imp_str.startswith('import '):
                    imp_str = f"import {imp_str}"
                
                import_path = re.search(r'import\s+([\w.]+);', imp_str)
                if import_path and import_path.group(1) not in existing_imports:
                    new_imports.append(imp_str)
            
            if not new_imports:
                return ActionResult(
                    action_type=ActionType.FIX_IMPORTS,
                    success=True,
                    message="All imports already exist",
                    modified_file=str(test_file)
                )
            
            package_match = re.search(r'package\s+[\w.]+;', content)
            package_line = package_match.group(0) if package_match else ""
            
            if package_line:
                insert_pos = content.find(package_line) + len(package_line)
                import_block = "\n\n" + "\n".join(new_imports)
                modified_content = content[:insert_pos] + import_block + content[insert_pos:]
            else:
                import_block = "\n".join(new_imports) + "\n\n"
                modified_content = import_block + content
            
            test_path.write_text(modified_content, encoding='utf-8')
            
            logger.info(f"[ActionExecutor] Added {len(new_imports)} imports to {test_file}")
            
            return ActionResult(
                action_type=ActionType.FIX_IMPORTS,
                success=True,
                message=f"Added {len(new_imports)} imports",
                modified_code=modified_content,
                modified_file=str(test_file),
                details={"added_imports": new_imports}
            )
        except Exception as e:
            return ActionResult(
                action_type=ActionType.FIX_IMPORTS,
                success=False,
                message=f"Failed to fix imports: {str(e)}"
            )
    
    async def _add_dependency(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ActionResult:
        group_id = action.get('group_id', '')
        artifact_id = action.get('artifact_id', '')
        version = action.get('version', '')
        scope = action.get('scope', 'test')
        
        if not group_id or not artifact_id:
            return ActionResult(
                action_type=ActionType.ADD_DEPENDENCY,
                success=False,
                message="Missing group_id or artifact_id"
            )
        
        try:
            pom_path = Path(self.project_path) / "pom.xml" if self.project_path else Path("pom.xml")
            if not pom_path.exists():
                return ActionResult(
                    action_type=ActionType.ADD_DEPENDENCY,
                    success=False,
                    message="pom.xml not found"
                )
            
            pom_content = pom_path.read_text(encoding='utf-8')
            
            dep_pattern = rf'<groupId>{re.escape(group_id)}</groupId>\s*<artifactId>{re.escape(artifact_id)}</artifactId>'
            if re.search(dep_pattern, pom_content):
                return ActionResult(
                    action_type=ActionType.ADD_DEPENDENCY,
                    success=True,
                    message="Dependency already exists in pom.xml"
                )
            
            dependency_xml = f"""
    <dependency>
        <groupId>{group_id}</groupId>
        <artifactId>{artifact_id}</artifactId>
        <version>{version}</version>
        <scope>{scope}</scope>
    </dependency>"""
            
            dependencies_match = re.search(r'<dependencies>', pom_content)
            if dependencies_match:
                insert_pos = dependencies_match.end()
                modified_content = (
                    pom_content[:insert_pos] + 
                    dependency_xml + 
                    pom_content[insert_pos:]
                )
            else:
                dependencies_block = f"<dependencies>{dependency_xml}\n    </dependencies>"
                if '</build>' in pom_content:
                    modified_content = pom_content.replace(
                        '</build>',
                        f'</build>\n    {dependencies_block}'
                    )
                else:
                    modified_content = pom_content.replace(
                        '</project>',
                        f'    {dependencies_block}\n</project>'
                    )
            
            pom_path.write_text(modified_content, encoding='utf-8')
            
            logger.info(f"[ActionExecutor] Added dependency: {group_id}:{artifact_id}")
            
            return ActionResult(
                action_type=ActionType.ADD_DEPENDENCY,
                success=True,
                message=f"Added dependency {group_id}:{artifact_id}",
                details={
                    "group_id": group_id,
                    "artifact_id": artifact_id,
                    "version": version,
                    "scope": scope
                }
            )
        except Exception as e:
            return ActionResult(
                action_type=ActionType.ADD_DEPENDENCY,
                success=False,
                message=f"Failed to add dependency: {str(e)}"
            )
    
    async def _fix_syntax(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ActionResult:
        fixed_code = action.get('fixed_code', action.get('code', ''))
        test_file = action.get('file') or context.get('test_file', '')
        
        if not fixed_code:
            return ActionResult(
                action_type=ActionType.FIX_SYNTAX,
                success=False,
                message="No fixed code provided"
            )
        
        if test_file:
            try:
                test_path = Path(self.project_path) / test_file if self.project_path else Path(test_file)
                test_path.write_text(fixed_code, encoding='utf-8')
                
                return ActionResult(
                    action_type=ActionType.FIX_SYNTAX,
                    success=True,
                    message="Applied syntax fix",
                    modified_code=fixed_code,
                    modified_file=str(test_file)
                )
            except Exception as e:
                return ActionResult(
                    action_type=ActionType.FIX_SYNTAX,
                    success=False,
                    message=f"Failed to write fixed code: {str(e)}"
                )
        
        return ActionResult(
            action_type=ActionType.FIX_SYNTAX,
            success=True,
            message="Syntax fix ready to apply",
            modified_code=fixed_code
        )
    
    async def _fix_type_error(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ActionResult:
        return await self._fix_syntax(action, context)
    
    async def _regenerate_test(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ActionResult:
        return ActionResult(
            action_type=ActionType.REGENERATE_TEST,
            success=True,
            message="Test regeneration requested",
            details={"trigger_regeneration": True}
        )
    
    async def _fix_test_logic(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ActionResult:
        fixed_code = action.get('fixed_code', action.get('code', ''))
        test_file = action.get('file') or context.get('test_file', '')
        
        if not fixed_code:
            return ActionResult(
                action_type=ActionType.FIX_TEST_LOGIC,
                success=False,
                message="No fixed code provided"
            )
        
        if test_file:
            try:
                test_path = Path(self.project_path) / test_file if self.project_path else Path(test_file)
                test_path.write_text(fixed_code, encoding='utf-8')
                
                return ActionResult(
                    action_type=ActionType.FIX_TEST_LOGIC,
                    success=True,
                    message="Applied test logic fix",
                    modified_code=fixed_code,
                    modified_file=str(test_file)
                )
            except Exception as e:
                return ActionResult(
                    action_type=ActionType.FIX_TEST_LOGIC,
                    success=False,
                    message=f"Failed to apply fix: {str(e)}"
                )
        
        return ActionResult(
            action_type=ActionType.FIX_TEST_LOGIC,
            success=True,
            message="Test logic fix ready",
            modified_code=fixed_code
        )
    
    async def _add_mock(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ActionResult:
        mock_code = action.get('mock_code', '')
        mock_setup = action.get('mock_setup', action.get('setup', ''))
        test_file = action.get('file') or context.get('test_file', '')
        
        if not test_file:
            return ActionResult(
                action_type=ActionType.ADD_MOCK,
                success=False,
                message="No test file specified"
            )
        
        try:
            test_path = Path(self.project_path) / test_file if self.project_path else Path(test_file)
            
            if not test_path.exists():
                return ActionResult(
                    action_type=ActionType.ADD_MOCK,
                    success=False,
                    message=f"Test file not found: {test_file}"
                )
            
            content = test_path.read_text(encoding='utf-8')
            
            if mock_setup:
                setup_match = re.search(r'@BeforeEach\s+void\s+\w+\([^)]*\)\s*\{', content)
                if setup_match:
                    insert_pos = setup_match.end()
                    modified_content = (
                        content[:insert_pos] + 
                        "\n        " + mock_setup + 
                        content[insert_pos:]
                    )
                else:
                    modified_content = content + "\n\n" + mock_setup
            else:
                modified_content = content
            
            test_path.write_text(modified_content, encoding='utf-8')
            
            return ActionResult(
                action_type=ActionType.ADD_MOCK,
                success=True,
                message="Added mock configuration",
                modified_code=modified_content,
                modified_file=str(test_file)
            )
        except Exception as e:
            return ActionResult(
                action_type=ActionType.ADD_MOCK,
                success=False,
                message=f"Failed to add mock: {str(e)}"
            )
    
    async def _fix_assertion(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ActionResult:
        fixed_code = action.get('fixed_code', action.get('code', ''))
        test_file = action.get('file') or context.get('test_file', '')
        
        if not fixed_code:
            return ActionResult(
                action_type=ActionType.FIX_ASSERTION,
                success=False,
                message="No fixed code provided"
            )
        
        if test_file:
            try:
                test_path = Path(self.project_path) / test_file if self.project_path else Path(test_file)
                test_path.write_text(fixed_code, encoding='utf-8')
                
                return ActionResult(
                    action_type=ActionType.FIX_ASSERTION,
                    success=True,
                    message="Applied assertion fix",
                    modified_code=fixed_code,
                    modified_file=str(test_file)
                )
            except Exception as e:
                return ActionResult(
                    action_type=ActionType.FIX_ASSERTION,
                    success=False,
                    message=f"Failed to apply fix: {str(e)}"
                )
        
        return ActionResult(
            action_type=ActionType.FIX_ASSERTION,
            success=True,
            message="Assertion fix ready",
            modified_code=fixed_code
        )
    
    async def _skip_test(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ActionResult:
        test_method = action.get('test_method', '')
        test_file = action.get('file') or context.get('test_file', '')
        reason = action.get('reason', 'Skipped due to persistent failure')
        
        if not test_file:
            return ActionResult(
                action_type=ActionType.SKIP_TEST,
                success=False,
                message="No test file specified"
            )
        
        try:
            test_path = Path(self.project_path) / test_file if self.project_path else Path(test_file)
            
            if not test_path.exists():
                return ActionResult(
                    action_type=ActionType.SKIP_TEST,
                    success=False,
                    message=f"Test file not found: {test_file}"
                )
            
            content = test_path.read_text(encoding='utf-8')
            
            if test_method:
                pattern = rf'(@Test\s+)'
                def add_disabled(match):
                    return f'@Disabled("{reason}")\n{match.group(1)}'
                
                modified_content = re.sub(pattern, add_disabled, content, count=1)
            else:
                modified_content = content
            
            test_path.write_text(modified_content, encoding='utf-8')
            
            return ActionResult(
                action_type=ActionType.SKIP_TEST,
                success=True,
                message=f"Skipped test: {test_method or 'all'}",
                modified_code=modified_content,
                modified_file=str(test_file),
                details={"test_method": test_method, "reason": reason}
            )
        except Exception as e:
            return ActionResult(
                action_type=ActionType.SKIP_TEST,
                success=False,
                message=f"Failed to skip test: {str(e)}"
            )
    
    async def _modify_code(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ActionResult:
        fixed_code = action.get('fixed_code', action.get('code', ''))
        test_file = action.get('file') or context.get('test_file', '')
        
        if not fixed_code:
            return ActionResult(
                action_type=ActionType.MODIFY_CODE,
                success=False,
                message="No code provided"
            )
        
        if test_file:
            try:
                test_path = Path(self.project_path) / test_file if self.project_path else Path(test_file)
                test_path.write_text(fixed_code, encoding='utf-8')
                
                return ActionResult(
                    action_type=ActionType.MODIFY_CODE,
                    success=True,
                    message="Code modified successfully",
                    modified_code=fixed_code,
                    modified_file=str(test_file)
                )
            except Exception as e:
                return ActionResult(
                    action_type=ActionType.MODIFY_CODE,
                    success=False,
                    message=f"Failed to modify code: {str(e)}"
                )
        
        return ActionResult(
            action_type=ActionType.MODIFY_CODE,
            success=True,
            message="Code modification ready",
            modified_code=fixed_code
        )
    
    async def _install_dependency(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ActionResult:
        try:
            process = await asyncio.create_subprocess_exec(
                "mvn", "dependency:resolve", "-q",
                cwd=self.project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=120
            )
            
            if process.returncode == 0:
                return ActionResult(
                    action_type=ActionType.INSTALL_DEPENDENCY,
                    success=True,
                    message="Dependencies resolved successfully"
                )
            else:
                error_msg = stderr.decode() if stderr else "Unknown error"
                return ActionResult(
                    action_type=ActionType.INSTALL_DEPENDENCY,
                    success=False,
                    message=f"Failed to resolve dependencies: {error_msg[:200]}"
                )
        except asyncio.TimeoutError:
            return ActionResult(
                action_type=ActionType.INSTALL_DEPENDENCY,
                success=False,
                message="Dependency resolution timed out"
            )
        except Exception as e:
            return ActionResult(
                action_type=ActionType.INSTALL_DEPENDENCY,
                success=False,
                message=f"Failed to install dependency: {str(e)}"
            )
    
    async def _resolve_dependency(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ActionResult:
        return await self._install_dependency(action, context)
    
    def _update_success_rate(self, result: ActionResult):
        if result.action_type not in self._action_success_rates:
            self._action_success_rates[result.action_type] = {"success": 0, "total": 0}
        
        self._action_success_rates[result.action_type]["total"] += 1
        if result.success:
            self._action_success_rates[result.action_type]["success"] += 1
    
    def get_success_rate(self, action_type: ActionType) -> float:
        if action_type not in self._action_success_rates:
            return 0.0
        
        stats = self._action_success_rates[action_type]
        if stats["total"] == 0:
            return 0.0
        
        return stats["success"] / stats["total"]
    
    def get_execution_history(self) -> List[ActionResult]:
        return self._execution_history.copy()
    
    def clear_history(self):
        self._execution_history.clear()


def parse_llm_action_plan(llm_response: str) -> ActionPlan:
    """Parse LLM response into an ActionPlan.
    
    Args:
        llm_response: Raw LLM response text
        
    Returns:
        ActionPlan with parsed actions
    """
    actions = []
    reasoning = ""
    confidence = 0.5
    
    lines = llm_response.split('\n')
    current_section = None
    
    for line in lines:
        line_lower = line.lower().strip()
        
        if 'action_plan:' in line_lower or 'actions:' in line_lower:
            current_section = 'actions'
        elif 'reasoning:' in line_lower:
            current_section = 'reasoning'
            reasoning = line.split(':', 1)[1].strip() if ':' in line else ""
        elif 'confidence:' in line_lower:
            try:
                conf_str = line.split(':', 1)[1].strip()
                confidence = float(conf_str.replace('%', '')) / 100 if '%' in conf_str else float(conf_str)
            except ValueError:
                confidence = 0.5
        elif current_section == 'actions' and line.strip():
            action = _parse_single_action(line)
            if action:
                actions.append(action)
        elif current_section == 'reasoning' and line.strip():
            reasoning += " " + line.strip()
    
    return ActionPlan(
        actions=actions,
        reasoning=reasoning.strip(),
        confidence=confidence
    )


def _parse_single_action(line: str) -> Optional[Dict[str, Any]]:
    """Parse a single action line.
    
    Args:
        line: Action line text
        
    Returns:
        Parsed action dict or None
    """
    line = line.strip()
    if not line or line.startswith('#'):
        return None
    
    if line.startswith('-') or line.startswith('*'):
        line = line[1:].strip()
    
    if ':' in line:
        parts = line.split(':', 1)
        action_type = parts[0].strip()
        action_details = parts[1].strip() if len(parts) > 1 else ""
        
        action = {
            'action': action_type,
            'description': action_details
        }
        
        detail_patterns = {
            'imports': r'imports?:\s*\[([^\]]+)\]',
            'import_list': r'imports?:\s*\[([^\]]+)\]',
            'group_id': r'group_id:\s*([\w.]+)',
            'artifact_id': r'artifact_id:\s*([\w-]+)',
            'version': r'version:\s*([\w.-]+)',
            'scope': r'scope:\s*(\w+)',
            'file': r'file:\s*([\w/\\.-]+)',
            'test_file': r'test_file:\s*([\w/\\.-]+)',
            'test_method': r'test_method:\s*(\w+)',
            'fixed_code': r'code:\s*```(?:java)?\s*([^`]+)```',
        }
        
        for key, pattern in detail_patterns.items():
            match = re.search(pattern, action_details, re.IGNORECASE)
            if match:
                action[key] = match.group(1).strip()
        
        return action
    
    return {'action': line}
