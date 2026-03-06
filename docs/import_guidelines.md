# Import Guidelines - 导入规范

This document defines the import structure guidelines for the PyUT Agent project.

## Import Order

All imports should be organized in the following order, separated by blank lines:

```python
"""Module docstring."""

# 1. Standard library imports
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# 2. Third-party imports
import pydantic
from rich import print

# 3. Local project imports (pyutagent)
from pyutagent.core.interfaces import IAgent, ITool
from pyutagent.agent.unified_agent_base import UnifiedAgentBase
```

## Import Style

### Preferred: Absolute Imports

Always use absolute imports for clarity:

```python
# Good
from pyutagent.core.interfaces import IAgent
from pyutagent.agent.unified_agent_base import UnifiedAgentBase

# Avoid
from ..core.interfaces import IAgent
from .unified_agent_base import UnifiedAgentBase
```

### Preferred: Explicit Imports

Import only what you need:

```python
# Good
from pyutagent.core.interfaces import IAgent, ITool

# Avoid
from pyutagent.core.interfaces import *
```

### Type Imports

For type-only imports, use `TYPE_CHECKING`:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyutagent.core.interfaces import IAgent
```

## Import Categories

### Standard Library

Common stdlib modules used in the project:

- `abc` - Abstract base classes
- `asyncio` - Async I/O
- `collections` - Data structures
- `dataclasses` - Data classes
- `datetime` - Date and time
- `enum` - Enumerations
- `functools` - Higher-order functions
- `json` - JSON handling
- `logging` - Logging
- `os` - OS interface
- `pathlib` - Path manipulation
- `re` - Regular expressions
- `sys` - System-specific parameters
- `typing` - Type hints
- `uuid` - UUID generation

### Third-Party

Key third-party dependencies:

- `pydantic` - Data validation
- `pytest` - Testing framework
- `PySide6` - GUI framework
- `rich` - Rich text in terminal
- `yaml` - YAML parsing

### Local (pyutagent)

Import structure:

```
pyutagent/
├── core/          - Core infrastructure
├── agent/         - Agent implementations
├── tools/         - Tool implementations
├── skills/        - Skill implementations
├── memory/        - Memory systems
├── llm/           - LLM clients
└── utils/         - Utilities
```

## Circular Import Prevention

To avoid circular imports:

1. **Use TYPE_CHECKING**: Import types only when type checking
2. **Import at function level**: Move imports inside functions when necessary
3. **Use protocols**: Define protocols in a separate module
4. **Refactor**: Split modules if they have circular dependencies

Example:

```python
# interfaces.py
from typing import Protocol

class IAgent(Protocol):
    async def execute(self, task: str) -> Any: ...

# agent.py
from typing import TYPE_CHECKING
from pyutagent.core.interfaces import IAgent

if TYPE_CHECKING:
    from pyutagent.tools.core.tool_base import ToolBase

class MyAgent(IAgent):
    def __init__(self):
        # Import here to avoid circular import
        from pyutagent.tools.core.tool_base import ToolBase
        self.tool = ToolBase()
```

## __init__.py Structure

Package `__init__.py` files should:

1. Define `__all__` explicitly
2. Group exports logically
3. Include deprecation warnings for old imports
4. Import from submodules, not define classes

Example:

```python
"""Core module - 核心基础设施"""

# Import from submodules
from pyutagent.core.container import DIContainer
from pyutagent.core.event_bus import EventBus
from pyutagent.core.interfaces import IAgent, ITool

__all__ = [
    # Container
    'DIContainer',
    # Event
    'EventBus',
    # Interfaces
    'IAgent',
    'ITool',
]
```

## Deprecation Pattern

When deprecating old imports:

```python
import warnings

def __getattr__(name: str):
    if name == 'OldClass':
        warnings.warn(
            "OldClass is deprecated. Use NewClass instead.",
            DeprecationWarning,
            stacklevel=2
        )
        from pyutagent.core.new_module import NewClass
        return NewClass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

## Import Checking

Use the import optimization script:

```bash
# Check all imports
python scripts/optimize_imports.py pyutagent

# Check specific module
python scripts/optimize_imports.py pyutagent/core
```

## Best Practices

1. **Keep imports at the top**: Unless avoiding circular imports
2. **Sort imports**: Use isort or similar tools
3. **Remove unused imports**: Use autoflake or similar tools
4. **Use type hints**: Import types for better IDE support
5. **Document complex imports**: Add comments for non-obvious imports
6. **Minimize import side effects**: Avoid code execution at import time
