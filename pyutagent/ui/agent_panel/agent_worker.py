"""Agent Worker - 状态同步机制，连接 AutonomousLoop 到 UI.

提供实时状态广播、信号定义和 AutonomousLoop 集成.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Callable

from PyQt6.QtCore import QObject, pyqtSignal, QThread

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent 执行状态."""
    IDLE = auto()
    STARTING = auto()
    OBSERVING = auto()
    THINKING = auto()
    ACTING = auto()
    VERIFYING = auto()
    LEARNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    PAUSED = auto()


class ToolCallStatus(Enum):
    """工具调用状态."""
    PENDING = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass
class ToolCallInfo:
    """工具调用信息."""
    id: str
    tool_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None
    status: ToolCallStatus = ToolCallStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def duration(self) -> Optional[float]:
        """计算执行时长."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


@dataclass
class ThinkingStepInfo:
    """思考步骤信息."""
    id: str
    title: str
    description: str = ""
    status: str = "pending"
    details: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None


@dataclass
class AgentProgress:
    """Agent 进度信息."""
    current_step: int
    total_steps: int
    current_state: AgentState
    task_name: str
    progress_percent: float = 0.0
    message: str = ""


@dataclass
class AgentError:
    """Agent 错误信息."""
    step_id: str
    error_message: str
    error_type: str
    retryable: bool = True
    context: Dict[str, Any] = field(default_factory=dict)


class AgentStateSignals(QObject):
    """Agent 状态信号定义.

    所有信号都在主线程中发射，确保 UI 更新线程安全.
    """

    # 状态变更信号
    state_changed = pyqtSignal(AgentState, str)

    # 进度更新信号
    progress_updated = pyqtSignal(AgentProgress)

    # 思考步骤信号
    thinking_step_added = pyqtSignal(ThinkingStepInfo)
    thinking_step_updated = pyqtSignal(str, str, object)

    # 工具调用信号
    tool_call_started = pyqtSignal(ToolCallInfo)
    tool_call_completed = pyqtSignal(ToolCallInfo)
    tool_call_failed = pyqtSignal(ToolCallInfo, str)

    # 错误信号
    error_occurred = pyqtSignal(AgentError)

    # 原始输出信号
    raw_output = pyqtSignal(str)

    # 任务完成/失败信号
    task_completed = pyqtSignal(str, object)
    task_failed = pyqtSignal(str, str)

    # 学习记录信号
    learning_recorded = pyqtSignal(str)


class AgentWorker(QThread):
    """Agent 工作线程.

    在独立线程中运行 AutonomousLoop，通过信号与 UI 通信.
    """

    def __init__(self, autonomous_loop: Any, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.autonomous_loop = autonomous_loop
        self.signals = AgentStateSignals()

        self._current_task: Optional[str] = None
        self._context: Dict[str, Any] = {}
        self._running = False
        self._paused = False
        self._tool_calls: Dict[str, ToolCallInfo] = {}
        self._thinking_steps: Dict[str, ThinkingStepInfo] = {}
        self._step_counter = 0
        self._tool_counter = 0

    def run_task(self, task: str, context: Optional[Dict[str, Any]] = None):
        """启动任务.

        Args:
            task: 任务描述
            context: 任务上下文
        """
        self._current_task = task
        self._context = context or {}
        self._running = True
        self._paused = False
        self._tool_calls.clear()
        self._thinking_steps.clear()
        self._step_counter = 0
        self._tool_counter = 0

        self.start()

    def run(self):
        """运行 Agent 循环."""
        if not self._current_task:
            logger.error("No task to run")
            return

        try:
            # 创建事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # 运行任务
            result = loop.run_until_complete(
                self._execute_task(self._current_task, self._context)
            )

            loop.close()

            if result.get("success"):
                self.signals.task_completed.emit(
                    self._current_task,
                    result.get("result")
                )
            else:
                self.signals.task_failed.emit(
                    self._current_task,
                    result.get("error", "Unknown error")
                )

        except Exception as e:
            logger.exception(f"Agent worker error: {e}")
            self.signals.task_failed.emit(
                self._current_task or "unknown",
                str(e)
            )

    async def _execute_task(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行任务.

        Args:
            task: 任务描述
            context: 任务上下文

        Returns:
            执行结果
        """
        self._emit_state_changed(AgentState.STARTING, f"Starting task: {task}")

        try:
            # 包装 progress_callback
            def progress_callback(progress: Dict[str, Any]):
                self._handle_progress(progress)

            # 运行 AutonomousLoop
            if hasattr(self.autonomous_loop, 'run'):
                result = await self.autonomous_loop.run(
                    task=task,
                    context=context,
                    progress_callback=progress_callback
                )

                # 转换结果
                return {
                    "success": getattr(result, 'success', False),
                    "result": result,
                    "error": getattr(result, 'error', None)
                }
            else:
                # 模拟执行用于测试
                return await self._simulate_execution(task, context)

        except Exception as e:
            logger.exception(f"Task execution error: {e}")
            return {"success": False, "error": str(e)}

    async def _simulate_execution(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """模拟执行（用于测试）."""
        self._emit_state_changed(AgentState.OBSERVING, "Observing current state...")
        await asyncio.sleep(0.5)

        self._add_thinking_step("Analyze Task", f"Analyzing: {task}")
        self._emit_state_changed(AgentState.THINKING, "Thinking about approach...")
        await asyncio.sleep(0.5)

        self._add_thinking_step("Plan Actions", "Planning execution steps")

        # 模拟工具调用
        self._emit_state_changed(AgentState.ACTING, "Executing tools...")
        tool_id = self._start_tool_call("analyze_code", {"task": task})
        await asyncio.sleep(0.5)
        self._complete_tool_call(tool_id, {"status": "analyzed"})

        self._emit_state_changed(AgentState.VERIFYING, "Verifying results...")
        await asyncio.sleep(0.3)

        self._emit_state_changed(AgentState.LEARNING, "Recording learnings...")
        self.signals.learning_recorded.emit("Task completed successfully")
        await asyncio.sleep(0.2)

        self._emit_state_changed(AgentState.COMPLETED, "Task completed!")

        return {"success": True, "result": {"task": task, "status": "done"}}

    def _handle_progress(self, progress: Dict[str, Any]):
        """处理进度更新."""
        iteration = progress.get("iteration", 0)
        max_iterations = progress.get("max_iterations", 1)
        state_name = progress.get("state", "IDLE")

        # 转换状态
        try:
            state = AgentState[state_name]
        except KeyError:
            state = AgentState.IDLE

        # 计算进度百分比
        progress_percent = (iteration / max_iterations) * 100 if max_iterations > 0 else 0

        # 发射进度信号
        agent_progress = AgentProgress(
            current_step=iteration,
            total_steps=max_iterations,
            current_state=state,
            task_name=self._current_task or "",
            progress_percent=progress_percent,
            message=f"Iteration {iteration}/{max_iterations}"
        )
        self.signals.progress_updated.emit(agent_progress)

        # 发射状态变更信号
        self._emit_state_changed(state, f"Iteration {iteration}/{max_iterations}")

        # 发射原始输出
        self.signals.raw_output.emit(
            f"[{state_name}] Iteration {iteration}/{max_iterations}\n"
        )

    def _emit_state_changed(self, state: AgentState, message: str):
        """发射状态变更信号."""
        self.signals.state_changed.emit(state, message)
        logger.debug(f"Agent state: {state.name} - {message}")

    def _add_thinking_step(self, title: str, description: str = "",
                          parent_id: Optional[str] = None) -> str:
        """添加思考步骤."""
        self._step_counter += 1
        step_id = f"step_{self._step_counter}"

        step_info = ThinkingStepInfo(
            id=step_id,
            title=title,
            description=description,
            status="pending",
            parent_id=parent_id
        )
        self._thinking_steps[step_id] = step_info

        self.signals.thinking_step_added.emit(step_info)
        return step_id

    def _update_thinking_step(self, step_id: str, status: str,
                              details: Optional[str] = None):
        """更新思考步骤状态."""
        if step_id in self._thinking_steps:
            self._thinking_steps[step_id].status = status
            if details:
                self._thinking_steps[step_id].details.append(details)

        self.signals.thinking_step_updated.emit(step_id, status, details)

    def _start_tool_call(self, tool_name: str,
                         parameters: Dict[str, Any]) -> str:
        """开始工具调用."""
        self._tool_counter += 1
        tool_id = f"tool_{self._tool_counter}"

        tool_info = ToolCallInfo(
            id=tool_id,
            tool_name=tool_name,
            parameters=parameters,
            status=ToolCallStatus.RUNNING,
            start_time=time.time()
        )
        self._tool_calls[tool_id] = tool_info

        self.signals.tool_call_started.emit(tool_info)
        return tool_id

    def _complete_tool_call(self, tool_id: str, result: Any):
        """完成工具调用."""
        if tool_id in self._tool_calls:
            tool_info = self._tool_calls[tool_id]
            tool_info.result = result
            tool_info.status = ToolCallStatus.SUCCESS
            tool_info.end_time = time.time()

            self.signals.tool_call_completed.emit(tool_info)

    def _fail_tool_call(self, tool_id: str, error: str):
        """标记工具调用失败."""
        if tool_id in self._tool_calls:
            tool_info = self._tool_calls[tool_id]
            tool_info.error = error
            tool_info.status = ToolCallStatus.FAILED
            tool_info.end_time = time.time()

            self.signals.tool_call_failed.emit(tool_info, error)

    def pause(self):
        """暂停执行."""
        self._paused = True
        self._emit_state_changed(AgentState.PAUSED, "Execution paused")

    def resume(self):
        """恢复执行."""
        self._paused = False
        self._emit_state_changed(AgentState.STARTING, "Execution resumed")

    def stop(self):
        """停止执行."""
        self._running = False
        if hasattr(self.autonomous_loop, 'interrupt'):
            self.autonomous_loop.interrupt()

    def retry_step(self, step_id: str):
        """重试步骤."""
        logger.info(f"Retrying step: {step_id}")
        # 实现重试逻辑
        self.signals.raw_output.emit(f"[Retry] Step {step_id}\n")

    def skip_step(self, step_id: str):
        """跳过步骤."""
        logger.info(f"Skipping step: {step_id}")
        if step_id in self._thinking_steps:
            self._thinking_steps[step_id].status = "skipped"
        self.signals.thinking_step_updated.emit(step_id, "skipped", None)


def create_agent_worker(autonomous_loop: Any,
                        parent: Optional[QObject] = None) -> AgentWorker:
    """创建 AgentWorker 实例.

    Args:
        autonomous_loop: AutonomousLoop 实例
        parent: 父对象

    Returns:
        AgentWorker 实例
    """
    return AgentWorker(autonomous_loop, parent)
