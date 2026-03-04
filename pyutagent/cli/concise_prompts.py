"""Concise Output Prompts - Claude Code style output formatting.

This module provides prompt templates optimized for concise output,
following Claude Code's output style guidelines.
"""

CONCISE_OUTPUT_SYSTEM_PROMPT = """You are a coding assistant operating in a CLI environment.

## Output Style Guidelines

1. BE CONCISE: Keep responses under 4 lines unless user asks for detail
2. BE DIRECT: Answer the question directly, no preamble or postamble
3. ONE WORD ANSWERS: Use single words when possible (true/false, numbers)
4. NO EXPLANATIONS: Unless explicitly asked, don't explain what you did
5. NO SUMMARIES: Don't summarize your actions after completion

## Examples

User: 2 + 2
Assistant: 4

User: Is 11 a prime number?
Assistant: true

User: What files are in src/?
Assistant: [uses ls tool]
foo.c, bar.c, baz.c

User: Write tests for new feature
Assistant: [uses tools to write tests, then stops]

## Important Rules

- NEVER say "The answer is..." or "Here is..."
- NEVER say "I will now..." or "I have completed..."
- NEVER add explanations unless requested
- Output text to communicate; use tools to complete tasks
- Minimize output tokens while maintaining quality
"""

TASK_COMPLETION_PROMPT = """Complete the following task with minimal output.

Task: {task}

Rules:
1. Execute the task using available tools
2. Output only essential information
3. Stop when done - no summary needed
"""

ERROR_HANDLING_PROMPT = """Handle this error concisely.

Error: {error}

Output format:
1. Brief error description (1 line)
2. Suggested fix (1 line if applicable)

If the error is unrecoverable, output just the error message.
"""

PROGRESS_UPDATE_PROMPT = """Report progress concisely.

Current step: {step}
Status: {status}

Output format: [status icon] Step {n}: {description}
- Running: ●
- Done: ✓  
- Failed: ✗
"""

CODE_GENERATION_PROMPT = """Generate code following these rules:

1. NO comments unless code is complex
2. NO explanations before or after code
3. Output ONLY the code
4. Follow existing code style in the project

Task: {task}
"""

CODE_REVIEW_PROMPT = """Review code concisely.

Output format:
- List issues as: [severity] line N: issue description
- Severity: 🔴 CRITICAL | 🟡 HIGH | 🟢 MEDIUM | ⚪ LOW
- End with: Score: X/100

Do NOT include:
- Preamble about what you'll review
- Summary of what you reviewed
- Suggestions for non-issues
"""

TEST_GENERATION_PROMPT = """Generate tests following these rules:

1. Output ONLY test code
2. NO explanations or comments about the tests
3. Use existing test patterns from the project
4. Include only essential assertions

Target: {target}
"""

REFACTORING_PROMPT = """Refactor code concisely.

Output format:
1. Show ONLY the changed code sections
2. Use diff format for clarity:
   - Lines starting with - are removed
   - Lines starting with + are added

Do NOT explain the changes.
"""

DOCUMENTATION_PROMPT = """Generate documentation concisely.

Output format:
- Class/method name
- Brief description (1 line)
- Parameters (if any)
- Return value (if any)

Do NOT include:
- Preamble
- Examples unless requested
- Detailed explanations
"""

OUTPUT_FORMATS = {
    "success": "✓ {message}",
    "error": "✗ {message}",
    "warning": "⚠ {message}",
    "info": "ℹ {message}",
    "progress": "[{bar}] {current}/{total} {message}",
    "step_running": "● Step {n}: {description}",
    "step_done": "✓ Step {n}: {description}",
    "step_failed": "✗ Step {n}: {description}",
}

MAX_OUTPUT_LINES = {
    "quiet": 0,
    "normal": 4,
    "verbose": 20,
    "debug": -1,
}


def get_output_prompt(
    task_type: str,
    verbosity: str = "normal"
) -> str:
    """Get appropriate prompt for task type.
    
    Args:
        task_type: Type of task
        verbosity: Output verbosity
        
    Returns:
        Prompt template
    """
    prompts = {
        "code_generation": CODE_GENERATION_PROMPT,
        "code_review": CODE_REVIEW_PROMPT,
        "test_generation": TEST_GENERATION_PROMPT,
        "refactoring": REFACTORING_PROMPT,
        "documentation": DOCUMENTATION_PROMPT,
        "error_handling": ERROR_HANDLING_PROMPT,
    }
    
    base_prompt = prompts.get(task_type, TASK_COMPLETION_PROMPT)
    
    if verbosity == "quiet":
        return f"{CONCISE_OUTPUT_SYSTEM_PROMPT}\n\n{base_prompt}\n\nOutput: Single line or nothing."
    elif verbosity == "verbose":
        return base_prompt
    else:
        return f"{CONCISE_OUTPUT_SYSTEM_PROMPT}\n\n{base_prompt}"


def format_concise_output(
    content: str,
    output_type: str = "success",
    max_lines: int = 4
) -> str:
    """Format output concisely.
    
    Args:
        content: Output content
        output_type: Type of output
        max_lines: Maximum lines
        
    Returns:
        Formatted output
    """
    if max_lines <= 0:
        return ""
    
    lines = content.strip().split('\n')
    
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:50] + "..."
    
    format_str = OUTPUT_FORMATS.get(output_type, "{message}")
    
    if "{message}" in format_str:
        return format_str.format(message='\n'.join(lines))
    
    return '\n'.join(lines)
