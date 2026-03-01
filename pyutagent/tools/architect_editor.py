"""Architect/Editor dual-model pattern for Aider integration.

This module implements the Architect/Editor pattern where:
- Architect (powerful model): Analyzes and plans edits
- Editor (fast/cheap model): Converts plans to actual code changes

This approach provides higher quality edits at lower cost.
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Callable
import time


class ArchitectMode(Enum):
    """Architect/Editor operation modes."""
    SINGLE_MODEL = "single"  # Use single model for everything
    DUAL_MODEL = "dual"  # Use Architect + Editor


@dataclass
class EditPlan:
    """Represents an edit plan created by the Architect."""
    description: str
    reasoning: str
    changes: List[Dict[str, Any]] = field(default_factory=list)
    affected_files: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class ArchitectEditorResult:
    """Result from Architect/Editor process."""
    success: bool
    edit_plan: Optional[EditPlan] = None
    diff_text: Optional[str] = None
    error_message: Optional[str] = None
    architect_time: float = 0.0
    editor_time: float = 0.0
    total_cost: float = 0.0


class ArchitectEditor:
    """Implements the Architect/Editor dual-model pattern."""

    ARCHITECT_SYSTEM_PROMPT = """You are an expert Java developer and software architect.
Your role is to analyze code issues and create detailed edit plans.

DO NOT write actual code. Instead, describe:
1. What needs to be changed and why
2. The specific locations (class, method, line context)
3. The nature of each change (add, remove, modify)
4. Any dependencies or considerations

Format your response as:

## Analysis
[Your analysis of the issue]

## Edit Plan

### Change 1: [Brief description]
- **Location**: [Class.Method or line context]
- **Action**: [add/modify/remove]
- **Details**: [Specific description of what to change]
- **Reasoning**: [Why this change is needed]

### Change 2: [Brief description]
...

## Dependencies
- [List any dependencies or related files]

## Notes
[Any special considerations or warnings]
"""

    EDITOR_SYSTEM_PROMPT = """You are a precise code editor.
Your role is to convert edit plans into actual code changes using SEARCH/REPLACE format.

You will receive:
1. The original code
2. An edit plan describing what needs to change

Your task:
1. Follow the edit plan exactly
2. Create minimal, precise SEARCH/REPLACE blocks
3. Ensure the changes are syntactically correct
4. Include only the changed lines in SEARCH blocks

Use this format for each change:

<<<<<<< SEARCH
exact original code to find
=======
new code to replace
>>>>>>> REPLACE

Rules:
- SEARCH content must match exactly (including whitespace and indentation)
- Keep SEARCH blocks minimal - only changed lines
- Multiple blocks can be used for different locations
- Do not include explanations outside the blocks
"""

    def __init__(
        self,
        architect_llm,
        editor_llm,
        mode: ArchitectMode = ArchitectMode.DUAL_MODEL
    ):
        """Initialize Architect/Editor.

        Args:
            architect_llm: LLM client for Architect (powerful model)
            editor_llm: LLM client for Editor (fast/cheap model)
            mode: Operation mode (single or dual model)
        """
        self.architect_llm = architect_llm
        self.editor_llm = editor_llm
        self.mode = mode

    async def generate_fix(
        self,
        context: Dict[str, Any],
        original_code: str,
        error_analysis: Optional[str] = None,
        failure_analysis: Optional[str] = None
    ) -> ArchitectEditorResult:
        """Generate fix using Architect/Editor pattern.

        Args:
            context: Context information (file path, class info, etc.)
            original_code: The original code to fix
            error_analysis: Optional compilation error analysis
            failure_analysis: Optional test failure analysis

        Returns:
            ArchitectEditorResult with edit plan and diff
        """
        start_time = time.time()

        try:
            # Step 1: Architect phase
            architect_start = time.time()
            edit_plan = await self._architect_phase(
                context, original_code, error_analysis, failure_analysis
            )
            architect_time = time.time() - architect_start

            if not edit_plan or not edit_plan.changes:
                return ArchitectEditorResult(
                    success=False,
                    error_message="Architect could not create an edit plan",
                    architect_time=architect_time
                )

            # Step 2: Editor phase
            editor_start = time.time()
            diff_text = await self._editor_phase(edit_plan, original_code)
            editor_time = time.time() - editor_start

            total_time = time.time() - start_time

            return ArchitectEditorResult(
                success=True,
                edit_plan=edit_plan,
                diff_text=diff_text,
                architect_time=architect_time,
                editor_time=editor_time,
                total_cost=self._estimate_cost(architect_time, editor_time)
            )

        except Exception as e:
            return ArchitectEditorResult(
                success=False,
                error_message=str(e),
                architect_time=time.time() - start_time
            )

    async def _architect_phase(
        self,
        context: Dict[str, Any],
        original_code: str,
        error_analysis: Optional[str],
        failure_analysis: Optional[str]
    ) -> Optional[EditPlan]:
        """Execute Architect phase to create edit plan."""

        # Build prompt for Architect
        prompt = self._build_architect_prompt(
            context, original_code, error_analysis, failure_analysis
        )

        # Call Architect LLM
        response = await self.architect_llm.complete(
            messages=[
                {"role": "system", "content": self.ARCHITECT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3  # Lower temperature for more consistent analysis
        )

        # Parse the edit plan from response
        return self._parse_architect_response(response)

    async def _editor_phase(
        self,
        edit_plan: EditPlan,
        original_code: str
    ) -> str:
        """Execute Editor phase to convert plan to diff."""

        if self.mode == ArchitectMode.SINGLE_MODEL:
            # In single model mode, use architect model for everything
            return await self._single_model_edit(edit_plan, original_code)

        # Build prompt for Editor
        prompt = self._build_editor_prompt(edit_plan, original_code)

        # Call Editor LLM
        response = await self.editor_llm.complete(
            messages=[
                {"role": "system", "content": self.EDITOR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1  # Very low temperature for precise edits
        )

        return response

    async def _single_model_edit(
        self,
        edit_plan: EditPlan,
        original_code: str
    ) -> str:
        """Generate edit using single model (fallback mode)."""

        prompt = f"""Based on the following edit plan, create SEARCH/REPLACE blocks to fix the code.

## Edit Plan

{edit_plan.description}

### Changes Needed:
"""
        for i, change in enumerate(edit_plan.changes, 1):
            prompt += f"""
{i}. **{change.get('description', 'Change')}**
   - Location: {change.get('location', 'Unknown')}
   - Action: {change.get('action', 'modify')}
   - Details: {change.get('details', '')}
"""

        prompt += f"""

## Original Code

```java
{original_code}
```

Generate SEARCH/REPLACE blocks to implement these changes.
"""

        response = await self.architect_llm.complete(
            messages=[
                {"role": "system", "content": self.EDITOR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )

        return response

    def _build_architect_prompt(
        self,
        context: Dict[str, Any],
        original_code: str,
        error_analysis: Optional[str],
        failure_analysis: Optional[str]
    ) -> str:
        """Build prompt for Architect phase."""

        file_path = context.get('file_path', 'Unknown')
        class_info = context.get('class_info', {})

        prompt = f"""Please analyze the following Java code and create an edit plan to fix the issues.

## File Information
- Path: {file_path}
- Class: {class_info.get('name', 'Unknown')}
- Package: {class_info.get('package', 'Unknown')}
"""

        if error_analysis:
            prompt += f"""

## Compilation Errors
{error_analysis}
"""

        if failure_analysis:
            prompt += f"""

## Test Failures
{failure_analysis}
"""

        prompt += f"""

## Original Code

```java
{original_code}
```

Please analyze the issues and create a detailed edit plan following the format specified in your instructions.
"""

        return prompt

    def _build_editor_prompt(
        self,
        edit_plan: EditPlan,
        original_code: str
    ) -> str:
        """Build prompt for Editor phase."""

        prompt = f"""Please convert the following edit plan into SEARCH/REPLACE blocks.

## Edit Plan

{edit_plan.description}

### Reasoning
{edit_plan.reasoning}

### Changes to Implement:
"""

        for i, change in enumerate(edit_plan.changes, 1):
            prompt += f"""
{i}. **{change.get('description', f'Change {i}')}**
   - Location: {change.get('location', 'Unknown')}
   - Action: {change.get('action', 'modify')}
   - Details: {change.get('details', '')}
"""

        if edit_plan.dependencies:
            prompt += "\n### Dependencies\n"
            for dep in edit_plan.dependencies:
                prompt += f"- {dep}\n"

        prompt += f"""

## Original Code

```java
{original_code}
```

Please generate SEARCH/REPLACE blocks to implement these changes exactly as described.
"""

        return prompt

    def _parse_architect_response(self, response: str) -> Optional[EditPlan]:
        """Parse Architect response into EditPlan."""

        try:
            # Extract sections
            analysis = self._extract_section(response, "Analysis", "Edit Plan")
            reasoning = analysis if analysis else ""

            # Parse changes
            changes = []
            change_pattern = r"### Change \d+: ([^\n]+)\n.*?-\s*\*\*Location\*\*:\s*([^\n]+)"
            change_details = r"-\s*\*\*Action\*\*:\s*([^\n]+)\n.*?-\s*\*\*Details\*\*:\s*([^\n]+)"

            # Simple parsing - can be enhanced with more sophisticated regex
            lines = response.split('\n')
            current_change = None

            for line in lines:
                line = line.strip()

                if line.startswith('### Change'):
                    if current_change:
                        changes.append(current_change)
                    current_change = {
                        'description': line.replace('### Change', '').strip().split(':', 1)[-1].strip(),
                        'location': '',
                        'action': 'modify',
                        'details': ''
                    }
                elif current_change and 'Location:' in line:
                    current_change['location'] = line.split(':', 1)[-1].strip()
                elif current_change and 'Action:' in line:
                    current_change['action'] = line.split(':', 1)[-1].strip()
                elif current_change and 'Details:' in line:
                    current_change['details'] = line.split(':', 1)[-1].strip()

            if current_change:
                changes.append(current_change)

            # Extract dependencies
            dependencies = []
            dep_section = self._extract_section(response, "Dependencies", "Notes")
            if dep_section:
                for line in dep_section.split('\n'):
                    line = line.strip()
                    if line.startswith('- ') or line.startswith('* '):
                        dependencies.append(line[2:])

            return EditPlan(
                description=self._extract_section(response, "Edit Plan", "Dependencies") or response[:500],
                reasoning=reasoning,
                changes=changes,
                affected_files=[],  # Can be populated from changes
                dependencies=dependencies
            )

        except Exception as e:
            # Fallback: return basic plan
            return EditPlan(
                description=response[:1000],
                reasoning="Parsed from architect response",
                changes=[{'description': 'See full response', 'details': response}]
            )

    def _extract_section(self, text: str, section_name: str, next_section: str) -> Optional[str]:
        """Extract a section from markdown-style text."""

        patterns = [
            rf"## {section_name}\n(.*?)(?=## {next_section}|\Z)",
            rf"### {section_name}\n(.*?)(?=### {next_section}|\Z)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def _estimate_cost(self, architect_time: float, editor_time: float) -> float:
        """Estimate cost based on time (simplified)."""
        # This is a simplified cost estimation
        # In production, you'd use actual token counts and pricing
        architect_cost = architect_time * 0.01  # Assume $0.01 per second
        editor_cost = editor_time * 0.005  # Editor is cheaper
        return architect_cost + editor_cost


# Import re at module level for _extract_section
import re
