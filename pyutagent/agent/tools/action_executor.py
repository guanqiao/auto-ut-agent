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

from .action_definitions import (
    ACTION_DEFINITIONS,
    ACTION_TYPE_MAP,
    ActionCategory,
    ActionDefinition,
    get_action_definition,
    get_all_action_names,
    is_valid_action_name,
    generate_prompt_action_list,
    generate_prompt_examples,
)

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


_ACTION_TYPE_ENUM_MAP = {
    "fix_imports": ActionType.FIX_IMPORTS,
    "add_dependency": ActionType.ADD_DEPENDENCY,
    "fix_syntax": ActionType.FIX_SYNTAX,
    "fix_type_error": ActionType.FIX_TYPE_ERROR,
    "regenerate_test": ActionType.REGENERATE_TEST,
    "fix_test_logic": ActionType.FIX_TEST_LOGIC,
    "add_mock": ActionType.ADD_MOCK,
    "fix_assertion": ActionType.FIX_ASSERTION,
    "skip_test": ActionType.SKIP_TEST,
    "modify_code": ActionType.MODIFY_CODE,
    "install_dependency": ActionType.INSTALL_DEPENDENCY,
    "resolve_dependency": ActionType.RESOLVE_DEPENDENCY,
}


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
        """Get all valid action names (including aliases)."""
        return get_all_action_names()
    
    def get_action_definition(self, action_name: str) -> Optional[ActionDefinition]:
        """Get action definition by name or alias."""
        return get_action_definition(action_name)
    
    def _is_valid_test_code(self, code: str) -> bool:
        """Check if code looks like valid Java test code.
        
        Args:
            code: Code to validate
            
        Returns:
            True if code appears valid
        """
        if not code:
            return False
        
        code = code.strip()
        
        if len(code) < 50:
            logger.warning(f"[ActionExecutor] Code validation failed: too short ({len(code)} chars)")
            return False
        
        if 'class ' not in code:
            logger.warning("[ActionExecutor] Code validation failed: missing class declaration")
            return False
        
        has_test = '@Test' in code or 'void test' in code or '@BeforeEach' in code
        if not has_test:
            logger.warning("[ActionExecutor] Code validation failed: no test annotations/methods found")
            return False
        
        open_braces = code.count('{')
        close_braces = code.count('}')
        if open_braces != close_braces or open_braces == 0:
            logger.warning(f"[ActionExecutor] Code validation failed: unbalanced braces ({{={open_braces}, }}={close_braces})")
            return False
        
        return True
    
    def _backup_and_validate_write(
        self,
        test_path: Path,
        new_code: str,
        action_type: ActionType
    ) -> tuple[bool, str, Optional[str]]:
        """Backup existing file and validate new code before writing.
        
        Args:
            test_path: Path to test file
            new_code: New code to write
            action_type: Type of action being performed
            
        Returns:
            Tuple of (should_write, reason, backup_content)
        """
        backup_content = None
        
        if test_path.exists():
            try:
                backup_content = test_path.read_text(encoding='utf-8')
                logger.debug(f"[ActionExecutor] Backed up existing file: {test_path} ({len(backup_content)} chars)")
            except Exception as e:
                logger.warning(f"[ActionExecutor] Failed to backup existing file: {e}")
        
        if not self._is_valid_test_code(new_code):
            logger.error(f"[ActionExecutor] Refusing to write invalid code for {action_type.name}")
            return False, "Invalid test code - validation failed", backup_content
        
        return True, "Code validated successfully", backup_content
    
    def parse_action_type(self, action_str: str) -> ActionType:
        """Parse action string to ActionType enum.
        
        Uses the centralized ACTION_TYPE_MAP from action_definitions.
        """
        action_lower = action_str.lower().replace(' ', '_').replace('-', '_')
        canonical_type = ACTION_TYPE_MAP.get(action_lower)
        if canonical_type:
            return _ACTION_TYPE_ENUM_MAP.get(canonical_type, ActionType.UNKNOWN)
        return ActionType.UNKNOWN
    
    async def execute_action(
        self,
        action: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ActionResult:
        action_type_str = action.get('action', action.get('type', 'unknown'))
        action_type = self.parse_action_type(action_type_str)
        
        logger.info(f"🛠️ 执行 Action: {action_type.name}")
        logger.debug(f"Action 参数: {action}")
        
        if self.progress_callback:
            self.progress_callback("EXECUTING_ACTION", f"Executing: {action_type.name}")
        
        start_time = datetime.now()
        result = await self._execute_action_internal(action_type, action, context or {})
        duration = (datetime.now() - start_time).total_seconds()
        
        self._execution_history.append(result)
        self._update_success_rate(result)
        
        if result.success:
            logger.info(f"✅ Action 完成: {action_type.name} - {result.message} (耗时: {duration:.2f}s)")
        else:
            logger.warning(f"❌ Action 失败: {action_type.name} - {result.message}")
        
        return result
    
    async def execute_action_plan(
        self,
        plan: ActionPlan,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ActionResult]:
        results = []
        
        logger.info(
            f"📋 执行 Action 计划 - 共 {len(plan.actions)} 个动作, "
            f"置信度: {plan.confidence:.2f}, 来源: {plan.source}"
        )
        
        if plan.reasoning:
            logger.info(f"💭 推理: {plan.reasoning[:100]}...")
        
        for i, action in enumerate(plan.actions):
            action_name = action.get('action', 'unknown')
            logger.info(f"🔹 Action {i+1}/{len(plan.actions)}: {action_name}")
            
            result = await self.execute_action(action, context)
            results.append(result)
            
            if not result.success and action.get('critical', False):
                logger.error(f"🛑 关键 Action 失败，停止执行计划")
                break
            
            if not result.success:
                logger.warning(f"⚠️ Action {i+1} 失败: {result.message}")
            
            await asyncio.sleep(0.1)
        
        success_count = sum(1 for r in results if r.success)
        logger.info(
            f"📊 Action 计划完成 - 成功: {success_count}/{len(results)}, "
            f"成功率: {success_count/len(results)*100:.1f}%"
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
        compiler_output = context.get('compiler_output', '')
        
        if isinstance(imports_to_add, str):
            imports_to_add = [i.strip() for i in imports_to_add.split(',')]
        
        if not imports_to_add and compiler_output:
            from pyutagent.core.error_classification import detect_missing_imports
            imports_to_add = detect_missing_imports(compiler_output)
            logger.info(f"🔧 从编译错误中提取到的导入: {imports_to_add}")
        
        logger.info(f"🔧 修复导入 - 文件: {test_file}, 导入数: {len(imports_to_add)}")
        
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
            
            existing_regular_imports = set(re.findall(r'^import\s+([\w.]+);', content, re.MULTILINE))
            existing_static_imports = set(re.findall(r'^import\s+static\s+([\w.]+);', content, re.MULTILINE))
            
            new_regular_imports = []
            new_static_imports = []
            
            for imp in imports_to_add:
                cleaned_import = self._clean_import_statement(imp)
                if not cleaned_import:
                    continue
                
                is_static = cleaned_import.startswith('import static ')
                import_content = cleaned_import.replace('import static ', '').replace('import ', '').rstrip(';')
                
                if is_static:
                    if import_content not in existing_static_imports:
                        new_static_imports.append(f"import static {import_content};")
                else:
                    if import_content not in existing_regular_imports:
                        new_regular_imports.append(f"import {import_content};")
            
            all_new_imports = new_regular_imports + new_static_imports
            
            if not all_new_imports:
                logger.info(f"✓ 所有导入已存在")
                return ActionResult(
                    action_type=ActionType.FIX_IMPORTS,
                    success=True,
                    message="All imports already exist",
                    modified_file=str(test_file)
                )
            
            new_regular_imports.sort()
            new_static_imports.sort()
            
            logger.info(f"➕ 添加 {len(all_new_imports)} 个新导入 (普通: {len(new_regular_imports)}, 静态: {len(new_static_imports)})")
            
            package_match = re.search(r'package\s+[\w.]+;', content)
            package_line = package_match.group(0) if package_match else ""
            
            regular_block = "\n".join(new_regular_imports)
            static_block = "\n".join(new_static_imports)
            
            if package_line:
                insert_pos = content.find(package_line) + len(package_line)
                if new_regular_imports and new_static_imports:
                    import_block = f"\n\n{regular_block}\n\n{static_block}\n"
                elif new_regular_imports:
                    import_block = f"\n\n{regular_block}\n"
                else:
                    import_block = f"\n\n{static_block}\n"
                modified_content = content[:insert_pos] + import_block + content[insert_pos:]
            else:
                if new_regular_imports and new_static_imports:
                    import_block = f"{regular_block}\n\n{static_block}\n\n"
                elif new_regular_imports:
                    import_block = f"{regular_block}\n\n"
                else:
                    import_block = f"{static_block}\n\n"
                modified_content = import_block + content
            
            test_path.write_text(modified_content, encoding='utf-8')
            
            logger.info(f"✅ 成功添加导入: {len(all_new_imports)} 个")
            
            return ActionResult(
                action_type=ActionType.FIX_IMPORTS,
                success=True,
                message=f"Added {len(all_new_imports)} imports ({len(new_regular_imports)} regular, {len(new_static_imports)} static)",
                modified_code=modified_content,
                modified_file=str(test_file),
                details={"added_imports": all_new_imports}
            )
        except Exception as e:
            logger.error(f"❌ 添加导入失败: {str(e)}")
            return ActionResult(
                action_type=ActionType.FIX_IMPORTS,
                success=False,
                message=f"Failed to fix imports: {str(e)}"
            )
    
    def _clean_import_statement(self, imp: str) -> Optional[str]:
        """Clean and normalize an import statement from various LLM output formats.
        
        Handles formats like:
        - "import java.sql.Connection;"
        - ["import java.sql.Connection;"]
        - import "import java.sql.Connection;"];
        - java.sql.Connection
        - "java.sql.Connection"
        
        Args:
            imp: Raw import statement from LLM
            
        Returns:
            Cleaned import statement or None if invalid
        """
        if not imp:
            return None
        
        imp = str(imp).strip()
        
        imp = imp.strip('"\'')
        imp = imp.strip('"\'')
        
        imp = re.sub(r'^\[["\']?', '', imp)
        imp = re.sub(r'["\']?\]$', '', imp)
        
        imp = re.sub(r'^["\']import\s+', 'import ', imp)
        imp = re.sub(r';["\']$', ';', imp)
        
        if imp.startswith('import "import '):
            imp = re.sub(r'^import\s+"import\s+', 'import ', imp)
            imp = imp.rstrip('";') + ';'
        
        if imp.startswith('import "'):
            imp = re.sub(r'^import\s+"', 'import ', imp)
            imp = imp.rstrip('"') + ';'
        
        imp = re.sub(r';["\']?\]?\s*;?\s*$', ';', imp)
        imp = re.sub(r'\s+', ' ', imp).strip()
        
        if not imp.startswith('import '):
            if re.match(r'^[\w.]+\.\w+$', imp):
                imp = f'import {imp};'
            elif re.match(r'^static\s+[\w.]+', imp):
                imp = f'import {imp};'
            elif re.match(r'^[\w.]+\.\*(?:;)?$', imp):
                imp = f'import {imp};' if not imp.endswith(';') else f'import {imp}'
        
        if not re.match(r'^import\s+(?:static\s+)?[\w.]+(?:\.\*)?;$', imp):
            logger.warning(f"[ActionExecutor] Invalid import format after cleaning: '{imp}'")
            return None
        
        return imp
    
    def _clean_code_block(self, code: str) -> str:
        """Clean code block from various LLM output formats.
        
        Handles formats like:
        - ```java\\ncode\\n```
        - ```\\ncode\\n```
        - code with extra whitespace
        
        Args:
            code: Raw code from LLM
            
        Returns:
            Cleaned code
        """
        if not code:
            return ""
        
        code = str(code).strip()
        
        code = re.sub(r'^```(?:java)?\s*\n', '', code)
        code = re.sub(r'\n```\s*$', '', code)
        
        code = re.sub(r'^```(?:java)?\s*', '', code)
        code = re.sub(r'```\s*$', '', code)
        
        code = code.strip()
        
        return code
    
    async def _add_dependency(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ActionResult:
        group_id = action.get('group_id', '')
        artifact_id = action.get('artifact_id', '')
        version = action.get('version', '')
        scope = action.get('scope', 'test')
        class_name = action.get('class_name', '')
        
        if not version and class_name:
            from pyutagent.core.error_classification import get_dependency_info
            dep_info = get_dependency_info(class_name)
            if dep_info:
                group_id = dep_info.get('group_id', group_id)
                artifact_id = dep_info.get('artifact_id', artifact_id)
                version = dep_info.get('version', version)
                logger.info(f"🔍 从类映射表获取依赖信息: {group_id}:{artifact_id}:{version}")
            else:
                logger.warning(f"⚠️ 未找到类 {class_name} 对应的依赖信息")
        
        logger.info(f"📦 添加依赖 - {group_id}:{artifact_id}:{version} (scope: {scope})")
        
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
                logger.info(f"✓ 依赖已存在")
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
            
            logger.info(f"✅ 成功添加依赖: {group_id}:{artifact_id}")
            
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
            logger.error(f"❌ 添加依赖失败: {str(e)}")
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
        
        logger.info(f"🔧 修复语法错误 - 文件: {test_file}")
        
        if not fixed_code:
            return ActionResult(
                action_type=ActionType.FIX_SYNTAX,
                success=False,
                message="No fixed code provided"
            )
        
        fixed_code = self._clean_code_block(fixed_code)
        
        if test_file:
            try:
                test_path = Path(self.project_path) / test_file if self.project_path else Path(test_file)
                
                should_write, reason, backup = self._backup_and_validate_write(
                    test_path, fixed_code, ActionType.FIX_SYNTAX
                )
                
                if not should_write:
                    logger.error(f"[ActionExecutor] Refusing to write invalid syntax fix: {reason}")
                    return ActionResult(
                        action_type=ActionType.FIX_SYNTAX,
                        success=False,
                        message=f"Invalid code - {reason}"
                    )
                
                test_path.write_text(fixed_code, encoding='utf-8')
                
                logger.info(f"✅ 语法修复已应用 - 文件: {test_file}, 长度: {len(fixed_code)}")
                return ActionResult(
                    action_type=ActionType.FIX_SYNTAX,
                    success=True,
                    message="Applied syntax fix",
                    modified_code=fixed_code,
                    modified_file=str(test_file)
                )
            except Exception as e:
                logger.error(f"❌ 应用语法修复失败: {str(e)}")
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
        logger.info(f"🔄 请求重新生成测试")
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
        
        logger.info(f"🔧 修复测试逻辑 - 文件: {test_file}")
        
        if not fixed_code:
            return ActionResult(
                action_type=ActionType.FIX_TEST_LOGIC,
                success=False,
                message="No fixed code provided"
            )
        
        fixed_code = self._clean_code_block(fixed_code)
        
        if test_file:
            try:
                test_path = Path(self.project_path) / test_file if self.project_path else Path(test_file)
                
                should_write, reason, backup = self._backup_and_validate_write(
                    test_path, fixed_code, ActionType.FIX_TEST_LOGIC
                )
                
                if not should_write:
                    logger.error(f"[ActionExecutor] Refusing to write invalid test logic fix: {reason}")
                    return ActionResult(
                        action_type=ActionType.FIX_TEST_LOGIC,
                        success=False,
                        message=f"Invalid code - {reason}"
                    )
                
                test_path.write_text(fixed_code, encoding='utf-8')
                
                logger.info(f"✅ 测试逻辑修复已应用 - 文件: {test_file}, 长度: {len(fixed_code)}")
                return ActionResult(
                    action_type=ActionType.FIX_TEST_LOGIC,
                    success=True,
                    message="Applied test logic fix",
                    modified_code=fixed_code,
                    modified_file=str(test_file)
                )
            except Exception as e:
                logger.error(f"❌ 应用测试逻辑修复失败: {str(e)}")
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
        
        logger.info(f"🎭 添加 Mock - 文件: {test_file}")
        
        if not test_file:
            return ActionResult(
                action_type=ActionType.ADD_MOCK,
                success=False,
                message="No test file specified"
            )
        
        if mock_setup:
            mock_setup = self._clean_code_block(mock_setup)
        
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
            
            logger.info(f"✅ Mock 配置已添加")
            return ActionResult(
                action_type=ActionType.ADD_MOCK,
                success=True,
                message="Added mock configuration",
                modified_code=modified_content,
                modified_file=str(test_file)
            )
        except Exception as e:
            logger.error(f"❌ 添加 Mock 失败: {str(e)}")
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
        
        fixed_code = self._clean_code_block(fixed_code)
        
        if test_file:
            try:
                test_path = Path(self.project_path) / test_file if self.project_path else Path(test_file)
                
                should_write, reason, backup = self._backup_and_validate_write(
                    test_path, fixed_code, ActionType.FIX_ASSERTION
                )
                
                if not should_write:
                    logger.error(f"[ActionExecutor] Refusing to write invalid assertion fix: {reason}")
                    return ActionResult(
                        action_type=ActionType.FIX_ASSERTION,
                        success=False,
                        message=f"Invalid code - {reason}"
                    )
                
                test_path.write_text(fixed_code, encoding='utf-8')
                
                logger.info(f"✅ Assertion fix applied - 文件: {test_file}, 长度: {len(fixed_code)}")
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
        
        logger.info(f"⏭️ 跳过测试 - 方法: {test_method or 'all'}, 文件: {test_file}")
        
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
            
            logger.info(f"✅ 测试已跳过: {test_method or 'all'}")
            return ActionResult(
                action_type=ActionType.SKIP_TEST,
                success=True,
                message=f"Skipped test: {test_method or 'all'}",
                modified_code=modified_content,
                modified_file=str(test_file),
                details={"test_method": test_method, "reason": reason}
            )
        except Exception as e:
            logger.error(f"❌ 跳过测试失败: {str(e)}")
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
        
        fixed_code = self._clean_code_block(fixed_code)
        
        if test_file:
            try:
                test_path = Path(self.project_path) / test_file if self.project_path else Path(test_file)
                
                should_write, reason, backup = self._backup_and_validate_write(
                    test_path, fixed_code, ActionType.MODIFY_CODE
                )
                
                if not should_write:
                    logger.error(f"[ActionExecutor] Refusing to write invalid code modification: {reason}")
                    return ActionResult(
                        action_type=ActionType.MODIFY_CODE,
                        success=False,
                        message=f"Invalid code - {reason}"
                    )
                
                test_path.write_text(fixed_code, encoding='utf-8')
                
                logger.info(f"✅ Code modified - 文件: {test_file}, 长度: {len(fixed_code)}")
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
    
    Supports both single-line and multi-line YAML formats:
    - Single-line: "- action: fix_imports imports: [...]"
    - Multi-line: 
      - action: fix_imports
        imports: [...]
    
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
    current_action: Optional[Dict[str, Any]] = None
    
    for line in lines:
        line_stripped = line.strip()
        line_lower = line_stripped.lower()
        
        if ('action' in line_lower and 'plan' in line_lower) or 'actions:' in line_lower:
            current_section = 'actions'
            continue
        elif 'reasoning:' in line_lower:
            if current_action:
                if _is_valid_action(current_action):
                    actions.append(current_action)
                current_action = None
            current_section = 'reasoning'
            reasoning = line.split(':', 1)[1].strip() if ':' in line else ""
            continue
        elif 'confidence:' in line_lower:
            try:
                conf_str = line.split(':', 1)[1].strip()
                confidence = float(conf_str.replace('%', '')) / 100 if '%' in conf_str else float(conf_str)
            except ValueError:
                confidence = 0.5
            continue
        
        if current_section == 'actions':
            if line_stripped.startswith('- action:') or line_stripped.startswith('-action:'):
                if current_action:
                    if _is_valid_action(current_action):
                        actions.append(current_action)
                
                action_type = line_stripped.split(':', 1)[1].strip() if ':' in line_stripped else ""
                current_action = {'action': action_type}
            elif current_action and line_stripped and ':' in line_stripped:
                key_value = line_stripped.split(':', 1)
                if len(key_value) == 2:
                    key = key_value[0].strip()
                    value = key_value[1].strip()
                    current_action[key] = value
        elif current_section == 'reasoning' and line_stripped:
            reasoning += " " + line_stripped
    
    if current_action:
        if _is_valid_action(current_action):
            actions.append(current_action)
    
    filtered_actions = _filter_and_merge_actions(actions)
    
    logger.info(f"[ActionExecutor] Parsed {len(actions)} raw actions, filtered to {len(filtered_actions)} valid actions")
    
    return ActionPlan(
        actions=filtered_actions,
        reasoning=reasoning.strip(),
        confidence=confidence
    )


def _is_valid_action(action: Dict[str, Any]) -> bool:
    """Validate if an action is meaningful and should be executed.
    
    Validates against the centralized action definitions to ensure
    LLM only returns predefined action types.
    
    Args:
        action: Parsed action dictionary
        
    Returns:
        True if action is valid and meaningful
    """
    action_type = action.get('action', '').lower().strip()
    
    if not action_type:
        return False
    
    invalid_action_types = {
        'action', 'imports', 'group_id', 'artifact_id', 'version', 'scope',
        'file', 'test_file', 'test_method', 'fixed_code', 'reasoning',
        'confidence', 'root_cause', 'analysis', 'description', 'package',
        'import', 'code', 'class', 'test', 'method', 'result', 'error'
    }
    
    if action_type in invalid_action_types:
        return False
    
    if len(action_type) < 3:
        return False
    
    if action_type.startswith('import '):
        return False
    
    if action_type.startswith('package '):
        return False
    
    if action_type.startswith('public ') or action_type.startswith('private '):
        return False
    
    code_indicators = ['{', '}', 'void ', 'class ', 'public ', 'private ', '@', '//']
    if any(action_type.startswith(indicator) for indicator in code_indicators):
        return False
    
    if _is_import_statement(action_type):
        return False
    
    if not is_valid_action_name(action_type):
        logger.warning(f"[ActionExecutor] Invalid action type not in predefined list: '{action_type}'")
        return False
    
    return True


def _is_import_statement(text: str) -> bool:
    """Check if text is an import statement.
    
    Args:
        text: Text to check
        
    Returns:
        True if it's an import statement
    """
    return (
        text.strip().startswith('import ') or
        text.strip().startswith('import static') or
        bool(re.match(r'^import\s+[\w.]+;?\s*$', text.strip()))
    )


def _filter_and_merge_actions(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter and merge similar actions to avoid redundant operations.
    
    Args:
        actions: List of parsed actions
        
    Returns:
        Filtered and merged action list
    """
    if not actions:
        return []
    
    valid_actions = []
    import_statements = []
    dependency_actions = []
    
    for action in actions:
        action_type = action.get('action', '').lower()
        
        if action_type in ('fix_imports', 'add_import', 'add_imports'):
            import_value = action.get('imports', action.get('import_list', ''))
            if isinstance(import_value, list):
                import_statements.extend(import_value)
            elif import_value and isinstance(import_value, str):
                imports = [i.strip() for i in import_value.split(',')]
                import_statements.extend(imports)
        
        elif action_type in ('add_dependency', 'add_dependencies'):
            dep_info = {
                'group_id': action.get('group_id', ''),
                'artifact_id': action.get('artifact_id', ''),
                'version': action.get('version', ''),
                'scope': action.get('scope', 'test')
            }
            if dep_info.get('group_id') and dep_info.get('artifact_id'):
                dependency_actions.append(dep_info)
        
        else:
            valid_actions.append(action)
    
    if import_statements:
        unique_imports = list(dict.fromkeys(import_statements))
        valid_actions.insert(0, {
            'action': 'fix_imports',
            'imports': unique_imports,
            'description': f'Add {len(unique_imports)} imports'
        })
    
    if dependency_actions:
        unique_deps = []
        seen = set()
        for dep in dependency_actions:
            key = (dep.get('group_id'), dep.get('artifact_id'))
            if key not in seen:
                seen.add(key)
                unique_deps.append(dep)
        
        for dep in unique_deps:
            valid_actions.insert(0, {
                'action': 'add_dependency',
                'group_id': dep.get('group_id'),
                'artifact_id': dep.get('artifact_id'),
                'version': dep.get('version', ''),
                'scope': dep.get('scope', 'test'),
                'description': f'Add dependency {dep.get("group_id")}:{dep.get("artifact_id")}'
            })
    
    return valid_actions


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
            'imports': r'imports?[:\s]+(?:\[?([^\]\n]+)\]?)?',
            'import_list': r'imports?[:\s]+(?:\[?([^\]\n]+)\]?)?',
            'group_id': r'group_id[:\s]+([\w.]+)',
            'artifact_id': r'artifact_id[:\s]+([\w-]+)',
            'version': r'version[:\s]+([\w.-]+)',
            'scope': r'scope[:\s]+(\w+)',
            'file': r'file[:\s]+([\w/\\.-]+)',
            'test_file': r'test_file[:\s]+([\w/\\.-]+)',
            'test_method': r'test_method[:\s]+(\w+)',
            'fixed_code': r'code:\s*```(?:java)?\s*([^`]+)```',
        }
        
        for key, pattern in detail_patterns.items():
            match = re.search(pattern, action_details, re.IGNORECASE)
            if match:
                action[key] = match.group(1).strip()
        
        return action
    
    return {'action': line}
