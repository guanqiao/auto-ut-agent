# 测试代码生成流程优化计划

> 基于 2026-03-04 的流程分析报告，本计划旨在解决发现的问题并优化整体架构。

## 一、问题概述

### 1.1 高优先级问题 (P0)

| 问题 | 位置 | 影响 | 风险等级 |
|------|------|------|---------|
| 重试机制不统一 | `execution_steps.py:40-154` | 可能陷入无限循环 | 🔴 高 |
| 终止条件分散 | 多个文件 | 状态不一致 | 🔴 高 |
| 无全局最大尝试次数 | `execute_with_recovery()` | 资源耗尽 | 🔴 高 |

### 1.2 中优先级问题 (P1)

| 问题 | 位置 | 影响 | 风险等级 |
|------|------|------|---------|
| 错误分类逻辑重复 | `recovery_manager.py`, `execution_steps.py` | 代码冗余 | 🟡 中 |
| 状态更新冗余 | `execution_steps.py:622-638` | UI闪烁 | 🟡 中 |
| 缺少状态一致性验证 | 全局 | 状态混乱 | 🟡 中 |

---

## 二、修改计划

### Phase 1: 统一重试机制 (P0)

#### 2.1.1 创建 RetryConfig 配置类

**文件**: `pyutagent/core/retry_config.py`

```python
"""Unified retry configuration for the entire system."""

from dataclasses import dataclass, field
from typing import Dict, Optional
from enum import Enum


class RetryStrategy(Enum):
    """Retry strategies."""
    IMMEDIATE = "immediate"
    LINEAR_BACKOFF = "linear"
    EXPONENTIAL_BACKOFF = "exponential"
    FIXED_DELAY = "fixed"


@dataclass
class RetryConfig:
    """Unified retry configuration.
    
    This configuration is used across all retry mechanisms to ensure
    consistent behavior and prevent infinite loops.
    """
    
    max_total_attempts: int = 50
    max_step_attempts: int = 5
    max_compilation_attempts: int = 5
    max_test_attempts: int = 5
    
    backoff_base: float = 2.0
    backoff_max: float = 60.0
    backoff_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    
    retryable_errors: Dict[str, bool] = field(default_factory=lambda: {
        "network": True,
        "timeout": True,
        "transient": True,
        "resource": True,
        "compilation": True,
        "test_failure": True,
    })
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        if self.backoff_strategy == RetryStrategy.IMMEDIATE:
            return 0
        elif self.backoff_strategy == RetryStrategy.LINEAR_BACKOFF:
            return min(self.backoff_base * attempt, self.backoff_max)
        elif self.backoff_strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            return min(self.backoff_base * (2 ** (attempt - 1)), self.backoff_max)
        else:
            return self.backoff_base
    
    def is_retryable(self, error_category: str) -> bool:
        """Check if an error category is retryable."""
        return self.retryable_errors.get(error_category, False)
    
    def should_stop(self, attempt: int, step_name: str) -> bool:
        """Check if should stop based on attempt count."""
        if attempt >= self.max_total_attempts:
            return True
        
        step_limits = {
            "compilation": self.max_compilation_attempts,
            "test_execution": self.max_test_attempts,
        }
        
        step_limit = step_limits.get(step_name, self.max_step_attempts)
        return attempt >= step_limit


# Global default configuration
DEFAULT_RETRY_CONFIG = RetryConfig()
```

#### 2.1.2 重构 execute_with_recovery

**文件**: `pyutagent/agent/components/execution_steps.py`

**修改内容**:

```python
# 在文件开头添加导入
from pyutagent.core.retry_config import DEFAULT_RETRY_CONFIG, RetryConfig

# 修改 StepExecutor.__init__ 方法
def __init__(self, agent_core: Any, components: Dict[str, Any], retry_config: Optional[RetryConfig] = None):
    self.agent_core = agent_core
    self.components = components
    self.retry_config = retry_config or DEFAULT_RETRY_CONFIG

# 修改 execute_with_recovery 方法
async def execute_with_recovery(
    self,
    operation,
    *args,
    step_name: str = "operation",
    **kwargs
) -> StepResult:
    """Execute an operation with automatic error recovery.
    
    Now includes unified maximum attempt limits to prevent infinite loops.
    """
    attempt = 0
    
    logger.info(f"[StepExecutor] Starting step execution - Step: {step_name}")
    
    while not self.agent_core._stop_requested and not self.agent_core._terminated:
        attempt += 1
        
        # 🔴 新增: 统一的最大尝试次数检查
        if self.retry_config.should_stop(attempt, step_name):
            logger.error(f"[StepExecutor] Exceeded maximum attempts - Step: {step_name}, Attempts: {attempt}")
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Exceeded maximum attempts ({attempt}) for {step_name}"
            )
        
        logger.debug(f"[StepExecutor] Step attempt - Step: {step_name}, Attempt: {attempt}")
        
        try:
            if asyncio.iscoroutinefunction(operation):
                result = await operation(*args, **kwargs)
            else:
                result = operation(*args, **kwargs)
            
            if result.success:
                logger.info(f"[StepExecutor] Step executed successfully - Step: {step_name}, Attempt: {attempt}")
                return result
            else:
                logger.warning(f"[StepExecutor] Step returned failure - Step: {step_name}, Attempt: {attempt}")
                
                # ... 恢复逻辑保持不变 ...
                
        except Exception as e:
            # ... 异常处理保持不变 ...
    
    # ... 其余代码保持不变 ...
```

---

### Phase 2: 统一终止条件检查 (P0)

#### 2.2.1 创建 TerminationChecker 类

**文件**: `pyutagent/core/termination.py`

```python
"""Unified termination condition checker."""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class TerminationReason(Enum):
    """Reasons for termination."""
    MAX_ITERATIONS = auto()
    TARGET_COVERAGE_REACHED = auto()
    USER_STOPPED = auto()
    USER_TERMINATED = auto()
    MAX_ATTEMPTS_EXCEEDED = auto()
    ERROR_THRESHOLD_EXCEEDED = auto()
    TIMEOUT_EXCEEDED = auto()


@dataclass
class TerminationState:
    """Current termination state."""
    should_stop: bool = False
    reason: Optional[TerminationReason] = None
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class TerminationChecker:
    """Unified termination condition checker.
    
    Centralizes all termination condition checks to ensure consistency
    and prevent scattered logic across multiple files.
    """
    
    def __init__(
        self,
        max_iterations: int = 10,
        target_coverage: float = 0.8,
        max_total_attempts: int = 50,
        max_error_count: int = 10,
        timeout_seconds: Optional[float] = None
    ):
        self.max_iterations = max_iterations
        self.target_coverage = target_coverage
        self.max_total_attempts = max_total_attempts
        self.max_error_count = max_error_count
        self.timeout_seconds = timeout_seconds
        
        self._start_time: Optional[float] = None
        self._error_count = 0
        self._total_attempts = 0
        self._callbacks: List[Callable[[TerminationState], None]] = []
    
    def start(self):
        """Start the timer."""
        import time
        self._start_time = time.time()
    
    def register_callback(self, callback: Callable[[TerminationState], None]):
        """Register a callback for termination events."""
        self._callbacks.append(callback)
    
    def record_error(self):
        """Record an error occurrence."""
        self._error_count += 1
    
    def record_attempt(self):
        """Record an attempt."""
        self._total_attempts += 1
    
    def check(
        self,
        current_iteration: int,
        current_coverage: float,
        is_stopped: bool,
        is_terminated: bool
    ) -> TerminationState:
        """Check all termination conditions.
        
        Args:
            current_iteration: Current iteration number
            current_coverage: Current coverage percentage
            is_stopped: Whether user requested stop
            is_terminated: Whether user requested termination
            
        Returns:
            TerminationState with stop decision and reason
        """
        # Check user termination first
        if is_terminated:
            return self._create_state(True, TerminationReason.USER_TERMINATED, "User terminated")
        
        # Check user stop
        if is_stopped:
            return self._create_state(True, TerminationReason.USER_STOPPED, "User stopped")
        
        # Check max iterations
        if current_iteration > self.max_iterations:
            return self._create_state(
                True, 
                TerminationReason.MAX_ITERATIONS,
                f"Max iterations ({self.max_iterations}) reached"
            )
        
        # Check target coverage
        if current_coverage >= self.target_coverage:
            return self._create_state(
                True,
                TerminationReason.TARGET_COVERAGE_REACHED,
                f"Target coverage ({self.target_coverage:.1%}) reached"
            )
        
        # Check max attempts
        if self._total_attempts >= self.max_total_attempts:
            return self._create_state(
                True,
                TerminationReason.MAX_ATTEMPTS_EXCEEDED,
                f"Max attempts ({self.max_total_attempts}) exceeded"
            )
        
        # Check error count
        if self._error_count >= self.max_error_count:
            return self._create_state(
                True,
                TerminationReason.ERROR_THRESHOLD_EXCEEDED,
                f"Error threshold ({self.max_error_count}) exceeded"
            )
        
        # Check timeout
        if self.timeout_seconds and self._start_time:
            import time
            elapsed = time.time() - self._start_time
            if elapsed >= self.timeout_seconds:
                return self._create_state(
                    True,
                    TerminationReason.TIMEOUT_EXCEEDED,
                    f"Timeout ({self.timeout_seconds}s) exceeded"
                )
        
        return TerminationState(should_stop=False)
    
    def _create_state(
        self, 
        should_stop: bool, 
        reason: TerminationReason, 
        message: str,
        details: Optional[Dict] = None
    ) -> TerminationState:
        """Create termination state and notify callbacks."""
        state = TerminationState(
            should_stop=should_stop,
            reason=reason,
            message=message,
            details=details or {}
        )
        
        if should_stop:
            logger.info(f"[TerminationChecker] Stopping - Reason: {reason.name}, Message: {message}")
            for callback in self._callbacks:
                try:
                    callback(state)
                except Exception as e:
                    logger.warning(f"[TerminationChecker] Callback error: {e}")
        
        return state
    
    def reset(self):
        """Reset the checker state."""
        self._start_time = None
        self._error_count = 0
        self._total_attempts = 0
```

#### 2.2.2 集成到 FeedbackLoopExecutor

**文件**: `pyutagent/agent/components/feedback_loop.py`

**修改内容**:

```python
# 在文件开头添加导入
from pyutagent.core.termination import TerminationChecker, TerminationReason

# 修改 FeedbackLoopExecutor 类
class FeedbackLoopExecutor:
    def __init__(self, agent_core: Any, step_executor: Any):
        self.agent_core = agent_core
        self.step_executor = step_executor
        
        # 🔴 新增: 创建统一的终止检查器
        self.termination_checker = TerminationChecker(
            max_iterations=agent_core.max_iterations,
            target_coverage=agent_core.target_coverage
        )
        
        logger.debug("[FeedbackLoopExecutor] Initialized")

# 修改 _phase_feedback_loop 方法
async def _phase_feedback_loop(self) -> AgentResult:
    """Phase 3-6: Compile-Test-Analyze-Optimize loop."""
    loop_start_time = asyncio.get_event_loop().time()
    
    # 🔴 新增: 启动终止检查器
    self.termination_checker.start()
    
    self.agent_core._update_state(AgentState.COMPILING, "🔨 Step 3/6: Compiling generated tests...")
    logger.info("[FeedbackLoopExecutor] 🔨 Step 3: Starting compile-test loop")
    
    while not self.agent_core._stop_requested and not self.agent_core._terminated:
        # 🔴 修改: 使用统一的终止条件检查
        term_state = self.termination_checker.check(
            current_iteration=self.agent_core.current_iteration,
            current_coverage=self.agent_core.working_memory.current_coverage,
            is_stopped=self.agent_core._stop_requested,
            is_terminated=self.agent_core._terminated
        )
        
        if term_state.should_stop:
            logger.info(f"[FeedbackLoopExecutor] Termination requested - {term_state.message}")
            break
        
        # ... 其余逻辑保持不变 ...
```

---

### Phase 3: 提取错误分类服务 (P1)

#### 2.3.1 创建 ErrorClassificationService

**文件**: `pyutagent/core/error_classification.py`

```python
"""Unified error classification service."""

import logging
from typing import Any, Dict, Optional

from pyutagent.core.error_recovery import ErrorCategory, ErrorClassifier

logger = logging.getLogger(__name__)


class ErrorClassificationService:
    """Unified service for error classification.
    
    This service provides a single point for all error classification
    logic, eliminating duplication across the codebase.
    """
    
    _instance: Optional['ErrorClassificationService'] = None
    
    def __new__(cls) -> 'ErrorClassificationService':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def classify(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> ErrorCategory:
        """Classify an error into a category.
        
        Args:
            error: The error to classify
            context: Optional context information
            
        Returns:
            ErrorCategory enum value
        """
        return ErrorClassifier.classify(error)
    
    def categorize_by_message(self, error_message: str, error_details: Dict[str, Any]) -> ErrorCategory:
        """Categorize error by message and details.
        
        Args:
            error_message: Error message string
            error_details: Additional error details
            
        Returns:
            ErrorCategory enum value
        """
        return ErrorClassifier.categorize_error(error_message, error_details)
    
    def is_retryable(self, error: Exception) -> bool:
        """Check if an error is retryable.
        
        Args:
            error: The error to check
            
        Returns:
            True if the error is retryable
        """
        return ErrorClassifier.is_retryable(error)
    
    def get_recovery_strategy(self, error: Exception, attempt_count: int = 0) -> str:
        """Get recommended recovery strategy for an error.
        
        Args:
            error: The error
            attempt_count: Number of previous attempts
            
        Returns:
            Strategy name string
        """
        category = self.classify(error)
        
        # Strategy selection based on category and attempt count
        if category == ErrorCategory.NETWORK:
            return "RETRY_WITH_BACKOFF"
        elif category == ErrorCategory.TIMEOUT:
            return "RETRY_WITH_BACKOFF"
        elif category == ErrorCategory.COMPILATION_ERROR:
            return "ANALYZE_AND_FIX"
        elif category == ErrorCategory.TEST_FAILURE:
            if attempt_count < 3:
                return "ANALYZE_AND_FIX"
            else:
                return "RESET_AND_REGENERATE"
        elif category in (ErrorCategory.TRANSIENT, ErrorCategory.RESOURCE):
            return "RETRY_IMMEDIATE"
        else:
            return "ANALYZE_AND_FIX"


# Global service instance
error_classification_service = ErrorClassificationService()


def get_error_classification_service() -> ErrorClassificationService:
    """Get the global error classification service instance."""
    return error_classification_service
```

#### 2.3.2 移除重复的错误分类代码

**文件**: `pyutagent/agent/components/execution_steps.py`

**修改内容**:

```python
# 在文件开头添加导入
from pyutagent.core.error_classification import get_error_classification_service

# 修改 StepExecutor 类
class StepExecutor:
    def __init__(self, agent_core: Any, components: Dict[str, Any], retry_config: Optional[RetryConfig] = None):
        self.agent_core = agent_core
        self.components = components
        self.retry_config = retry_config or DEFAULT_RETRY_CONFIG
        # 🔴 新增: 使用统一的错误分类服务
        self.error_classifier = get_error_classification_service()

# 🔴 删除: 移除 _categorize_error 方法 (第 1217-1245 行)
# 改为使用 self.error_classifier.classify(error, context)

# 修改 _try_recover 方法
async def _try_recover(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
    # ...
    # 🔴 修改: 使用统一的错误分类服务
    error_category = self.error_classifier.classify(error, context)
    # ...
```

**文件**: `pyutagent/agent/components/recovery_manager.py`

**修改内容**:

```python
# 在文件开头添加导入
from pyutagent.core.error_classification import get_error_classification_service

# 修改 AgentRecoveryManager 类
class AgentRecoveryManager:
    def __init__(self, components: Dict[str, Any], agent_core: Any):
        self.components = components
        self.agent_core = agent_core
        # 🔴 新增: 使用统一的错误分类服务
        self.error_classifier = get_error_classification_service()
        logger.debug("[AgentRecoveryManager] Initialized")

# 🔴 删除: 移除 _categorize_error 方法 (第 119-145 行)
# 改为使用 self.error_classifier.classify(error, context)

# 修改 recover_from_error 方法
async def recover_from_error(self, error, context, step_name, attempt):
    # ...
    # 🔴 修改: 使用统一的错误分类服务
    error_category = self.error_classifier.classify(error, context)
    # ...
```

---

### Phase 4: 简化状态更新 (P1)

#### 2.4.1 创建 StateMachine 状态机

**文件**: `pyutagent/core/state_machine.py`

```python
"""State machine for agent state management."""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent states."""
    IDLE = auto()
    PARSING = auto()
    GENERATING = auto()
    COMPILING = auto()
    TESTING = auto()
    ANALYZING = auto()
    FIXING = auto()
    OPTIMIZING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass
class StateTransition:
    """Represents a state transition."""
    from_state: AgentState
    to_state: AgentState
    message: str = ""
    timestamp: float = 0.0


class StateMachine:
    """State machine for managing agent state transitions.
    
    Features:
    - Valid transition checking
    - Transition history
    - Observer pattern for state changes
    - Debounced state updates
    """
    
    # Valid state transitions
    VALID_TRANSITIONS: Dict[AgentState, Set[AgentState]] = {
        AgentState.IDLE: {AgentState.PARSING, AgentState.PAUSED},
        AgentState.PARSING: {AgentState.GENERATING, AgentState.FAILED, AgentState.PAUSED},
        AgentState.GENERATING: {AgentState.COMPILING, AgentState.FIXING, AgentState.FAILED, AgentState.PAUSED},
        AgentState.COMPILING: {AgentState.TESTING, AgentState.FIXING, AgentState.FAILED, AgentState.PAUSED},
        AgentState.TESTING: {AgentState.ANALYZING, AgentState.FIXING, AgentState.FAILED, AgentState.PAUSED},
        AgentState.ANALYZING: {AgentState.OPTIMIZING, AgentState.COMPLETED, AgentState.FAILED, AgentState.PAUSED},
        AgentState.OPTIMIZING: {AgentState.COMPILING, AgentState.COMPLETED, AgentState.FAILED, AgentState.PAUSED},
        AgentState.FIXING: {AgentState.COMPILING, AgentState.TESTING, AgentState.GENERATING, AgentState.FAILED, AgentState.PAUSED},
        AgentState.PAUSED: {AgentState.IDLE, AgentState.PARSING, AgentState.GENERATING, AgentState.COMPILING, 
                           AgentState.TESTING, AgentState.ANALYZING, AgentState.OPTIMIZING, AgentState.FIXING},
        AgentState.COMPLETED: {AgentState.IDLE},
        AgentState.FAILED: {AgentState.IDLE, AgentState.PARSING},
    }
    
    def __init__(self, initial_state: AgentState = AgentState.IDLE):
        self._state = initial_state
        self._transition_history: List[StateTransition] = []
        self._observers: List[Callable[[AgentState, str], None]] = []
        self._last_update_time: float = 0.0
        self._debounce_seconds: float = 0.1  # Minimum time between updates
    
    @property
    def current_state(self) -> AgentState:
        """Get current state."""
        return self._state
    
    def can_transition_to(self, new_state: AgentState) -> bool:
        """Check if transition to new state is valid."""
        valid_targets = self.VALID_TRANSITIONS.get(self._state, set())
        return new_state in valid_targets
    
    def transition(self, new_state: AgentState, message: str = "") -> bool:
        """Attempt to transition to a new state.
        
        Args:
            new_state: Target state
            message: Optional message for the transition
            
        Returns:
            True if transition was successful
        """
        import time
        current_time = time.time()
        
        # Debounce check - skip if same state and too soon
        if new_state == self._state:
            if current_time - self._last_update_time < self._debounce_seconds:
                logger.debug(f"[StateMachine] Debouncing state update - State: {new_state.name}")
                return True  # Not an error, just skipped
        
        # Validate transition
        if not self.can_transition_to(new_state):
            logger.warning(f"[StateMachine] Invalid transition - From: {self._state.name}, To: {new_state.name}")
            return False
        
        # Record transition
        old_state = self._state
        self._state = new_state
        self._last_update_time = current_time
        
        transition = StateTransition(
            from_state=old_state,
            to_state=new_state,
            message=message,
            timestamp=current_time
        )
        self._transition_history.append(transition)
        
        logger.info(f"[StateMachine] State transition - {old_state.name} → {new_state.name}: {message}")
        
        # Notify observers
        self._notify_observers(new_state, message)
        
        return True
    
    def force_transition(self, new_state: AgentState, message: str = ""):
        """Force a transition without validation (use with caution)."""
        import time
        
        old_state = self._state
        self._state = new_state
        self._last_update_time = time.time()
        
        logger.warning(f"[StateMachine] Forced transition - {old_state.name} → {new_state.name}: {message}")
        self._notify_observers(new_state, message)
    
    def add_observer(self, observer: Callable[[AgentState, str], None]):
        """Add an observer for state changes."""
        self._observers.append(observer)
    
    def remove_observer(self, observer: Callable[[AgentState, str], None]):
        """Remove an observer."""
        if observer in self._observers:
            self._observers.remove(observer)
    
    def _notify_observers(self, new_state: AgentState, message: str):
        """Notify all observers of state change."""
        for observer in self._observers:
            try:
                observer(new_state, message)
            except Exception as e:
                logger.warning(f"[StateMachine] Observer error: {e}")
    
    def get_history(self, limit: int = 10) -> List[StateTransition]:
        """Get recent transition history."""
        return self._transition_history[-limit:]
    
    def reset(self):
        """Reset to initial state."""
        self._state = AgentState.IDLE
        self._transition_history.clear()
        self._last_update_time = 0.0
```

#### 2.4.2 集成到 AgentCore

**文件**: `pyutagent/agent/components/core_agent.py`

**修改内容**:

```python
# 在文件开头添加导入
from pyutagent.core.state_machine import StateMachine, AgentState as SMAgentState

# 修改 AgentCore 类
class AgentCore:
    def __init__(self, llm_client, working_memory, project_path, progress_callback):
        # ... 现有初始化代码 ...
        
        # 🔴 新增: 使用状态机管理状态
        self._state_machine = StateMachine()
        
        # 注册观察者
        if progress_callback:
            self._state_machine.add_observer(self._on_state_change)
    
    def _on_state_change(self, new_state: SMAgentState, message: str):
        """Handle state machine state changes."""
        if self.progress_callback:
            self.progress_callback({
                "state": new_state.name,
                "message": message
            })
    
    def _update_state(self, state: AgentState, message: str):
        """Update agent state using state machine."""
        # 🔴 修改: 使用状态机进行状态转换
        try:
            sm_state = SMAgentState[state.name]
            if not self._state_machine.transition(sm_state, message):
                logger.warning(f"[AgentCore] State transition rejected - State: {state.name}")
        except KeyError:
            logger.warning(f"[AgentCore] Unknown state: {state.name}")
    
    @property
    def current_state(self) -> AgentState:
        """Get current state."""
        return AgentState[self._state_machine.current_state.name]
```

---

### Phase 5: 增加状态一致性验证 (P1)

#### 2.5.1 创建 StateValidator

**文件**: `pyutagent/core/state_validator.py`

```python
"""State consistency validator."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Represents a validation error."""
    field: str
    expected: Any
    actual: Any
    message: str


@dataclass
class ValidationResult:
    """Result of state validation."""
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, field: str, expected: Any, actual: Any, message: str = ""):
        self.errors.append(ValidationError(field, expected, actual, message))
        self.is_valid = False
    
    def add_warning(self, message: str):
        self.warnings.append(message)


class StateValidator:
    """Validates state consistency across the agent.
    
    Checks:
    - Required fields presence
    - Field type consistency
    - Cross-field dependencies
    - State transition validity
    """
    
    REQUIRED_FIELDS: Dict[str, Set[str]] = {
        "parsing": {"target_file"},
        "generating": {"target_file", "class_info"},
        "compiling": {"target_file", "class_info", "test_file"},
        "testing": {"target_file", "class_info", "test_file"},
        "analyzing": {"target_file", "class_info", "test_file"},
        "optimizing": {"target_file", "class_info", "test_file", "coverage_data"},
    }
    
    FIELD_TYPES: Dict[str, type] = {
        "target_file": str,
        "test_file": str,
        "class_info": dict,
        "coverage_data": dict,
        "current_iteration": int,
        "current_coverage": float,
    }
    
    def validate(self, state: Dict[str, Any], current_phase: str) -> ValidationResult:
        """Validate state for a given phase.
        
        Args:
            state: Current state dictionary
            current_phase: Current phase name
            
        Returns:
            ValidationResult with any errors or warnings
        """
        result = ValidationResult(is_valid=True)
        
        # Check required fields
        required = self.REQUIRED_FIELDS.get(current_phase.lower(), set())
        for field in required:
            if field not in state or state[field] is None:
                result.add_error(field, "present", "missing", f"Required field '{field}' is missing")
        
        # Check field types
        for field, expected_type in self.FIELD_TYPES.items():
            if field in state and state[field] is not None:
                if not isinstance(state[field], expected_type):
                    result.add_error(
                        field, 
                        expected_type.__name__, 
                        type(state[field]).__name__,
                        f"Field '{field}' has wrong type"
                    )
        
        # Check cross-field dependencies
        self._validate_dependencies(state, result)
        
        # Check value ranges
        self._validate_ranges(state, result)
        
        return result
    
    def _validate_dependencies(self, state: Dict[str, Any], result: ValidationResult):
        """Validate cross-field dependencies."""
        # test_file should match class_info.name
        if "test_file" in state and "class_info" in state:
            test_file = state["test_file"]
            class_name = state["class_info"].get("name", "")
            if class_name and class_name not in test_file:
                result.add_warning(f"Test file name doesn't match class name: {test_file} vs {class_name}")
        
        # coverage should be between 0 and 1
        if "current_coverage" in state:
            coverage = state["current_coverage"]
            if not (0 <= coverage <= 1):
                result.add_error("current_coverage", "0-1", coverage, "Coverage should be between 0 and 1")
    
    def _validate_ranges(self, state: Dict[str, Any], result: ValidationResult):
        """Validate value ranges."""
        if "current_iteration" in state:
            iteration = state["current_iteration"]
            if iteration < 0:
                result.add_error("current_iteration", ">=0", iteration, "Iteration cannot be negative")
            
            max_iterations = state.get("max_iterations", 10)
            if iteration > max_iterations:
                result.add_warning(f"Iteration ({iteration}) exceeds max ({max_iterations})")
    
    def validate_transition(
        self, 
        from_phase: str, 
        to_phase: str, 
        state: Dict[str, Any]
    ) -> ValidationResult:
        """Validate a phase transition.
        
        Args:
            from_phase: Current phase
            to_phase: Target phase
            state: Current state
            
        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)
        
        # Define valid transitions
        valid_transitions = {
            "idle": {"parsing"},
            "parsing": {"generating", "failed"},
            "generating": {"compiling", "fixing", "failed"},
            "compiling": {"testing", "fixing", "failed"},
            "testing": {"analyzing", "fixing", "failed"},
            "analyzing": {"optimizing", "completed", "failed"},
            "optimizing": {"compiling", "completed", "failed"},
            "fixing": {"compiling", "testing", "generating", "failed"},
            "completed": {"idle"},
            "failed": {"idle", "parsing"},
        }
        
        from_lower = from_phase.lower()
        to_lower = to_phase.lower()
        
        valid_targets = valid_transitions.get(from_lower, set())
        if to_lower not in valid_targets:
            result.add_error(
                "transition",
                f"{from_phase} -> {valid_targets}",
                f"{from_phase} -> {to_phase}",
                f"Invalid phase transition"
            )
        
        return result
```

#### 2.5.2 集成到流程中

**文件**: `pyutagent/agent/components/feedback_loop.py`

**修改内容**:

```python
# 在文件开头添加导入
from pyutagent.core.state_validator import StateValidator

# 修改 FeedbackLoopExecutor 类
class FeedbackLoopExecutor:
    def __init__(self, agent_core: Any, step_executor: Any):
        self.agent_core = agent_core
        self.step_executor = step_executor
        self.termination_checker = TerminationChecker(...)
        
        # 🔴 新增: 创建状态验证器
        self.state_validator = StateValidator()
        
        logger.debug("[FeedbackLoopExecutor] Initialized")
    
    def _validate_state(self, phase: str) -> bool:
        """Validate current state before proceeding."""
        state = {
            "target_file": getattr(self.agent_core, 'target_file', None),
            "test_file": self.agent_core.current_test_file,
            "class_info": self.agent_core.target_class_info,
            "current_iteration": self.agent_core.current_iteration,
            "current_coverage": self.agent_core.working_memory.current_coverage,
        }
        
        result = self.state_validator.validate(state, phase)
        
        if not result.is_valid:
            logger.error(f"[FeedbackLoopExecutor] State validation failed for {phase}:")
            for error in result.errors:
                logger.error(f"  - {error.field}: {error.message}")
            return False
        
        if result.warnings:
            for warning in result.warnings:
                logger.warning(f"[FeedbackLoopExecutor] State warning: {warning}")
        
        return True

# 在各阶段开始前添加验证
async def _phase_generate_initial_tests(self) -> AgentResult:
    # 🔴 新增: 状态验证
    if not self._validate_state("generating"):
        return AgentResult(success=False, message="State validation failed")
    
    # ... 原有代码 ...
```

---

## 三、修改文件清单

| 文件 | 操作 | 优先级 |
|------|------|--------|
| `pyutagent/core/retry_config.py` | 新建 | P0 |
| `pyutagent/core/termination.py` | 新建 | P0 |
| `pyutagent/core/error_classification.py` | 新建 | P1 |
| `pyutagent/core/state_machine.py` | 新建 | P1 |
| `pyutagent/core/state_validator.py` | 新建 | P1 |
| `pyutagent/agent/components/execution_steps.py` | 修改 | P0 |
| `pyutagent/agent/components/feedback_loop.py` | 修改 | P0 |
| `pyutagent/agent/components/recovery_manager.py` | 修改 | P1 |
| `pyutagent/agent/components/core_agent.py` | 修改 | P1 |

---

## 四、测试计划

### 4.1 单元测试

| 测试文件 | 测试内容 |
|---------|---------|
| `tests/unit/core/test_retry_config.py` | RetryConfig 配置和延迟计算 |
| `tests/unit/core/test_termination.py` | TerminationChecker 各条件检查 |
| `tests/unit/core/test_error_classification.py` | ErrorClassificationService 分类逻辑 |
| `tests/unit/core/test_state_machine.py` | StateMachine 状态转换和验证 |
| `tests/unit/core/test_state_validator.py` | StateValidator 状态验证 |

### 4.2 集成测试

| 测试文件 | 测试内容 |
|---------|---------|
| `tests/integration/test_retry_integration.py` | 重试机制集成测试 |
| `tests/integration/test_termination_integration.py` | 终止条件集成测试 |

### 4.3 回归测试

运行现有测试套件确保修改不影响现有功能：

```bash
pytest tests/ -v --cov=pyutagent
```

---

## 五、实施顺序

1. **Phase 1 (P0)**: 统一重试机制
   - 创建 `retry_config.py`
   - 修改 `execution_steps.py`
   - 编写单元测试

2. **Phase 2 (P0)**: 统一终止条件检查
   - 创建 `termination.py`
   - 修改 `feedback_loop.py`
   - 编写单元测试

3. **Phase 3 (P1)**: 提取错误分类服务
   - 创建 `error_classification.py`
   - 移除重复代码
   - 编写单元测试

4. **Phase 4 (P1)**: 简化状态更新
   - 创建 `state_machine.py`
   - 修改 `core_agent.py`
   - 编写单元测试

5. **Phase 5 (P1)**: 增加状态一致性验证
   - 创建 `state_validator.py`
   - 集成到流程中
   - 编写单元测试

6. **验证**: 运行完整测试套件

---

## 六、预期收益

| 改进点 | 预期收益 |
|-------|---------|
| 统一重试机制 | 消除无限循环风险，资源使用可控 |
| 统一终止条件 | 状态一致性提升，调试更容易 |
| 错误分类服务 | 代码冗余减少，维护成本降低 |
| 状态机管理 | 状态转换清晰，UI 体验更好 |
| 状态验证 | 提前发现问题，减少运行时错误 |

---

*计划创建时间: 2026-03-04*
*预计实施周期: 2-3 天*
