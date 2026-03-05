"""Unified Autonomous Loop - 统一的自主循环基类

整合所有自主循环实现：
- 基础循环 (Observe-Think-Act-Verify-Learn)
- LLM增强循环
- 委派循环
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar
from uuid import uuid4

logger = logging.getLogger(__name__)

T = TypeVar('T')


class LoopState(Enum):
    """循环状态枚举"""
    IDLE = auto()
    OBSERVING = auto()
    THINKING = auto()
    ACTING = auto()
    VERIFYING = auto()
    LEARNING = auto()
    DELEGATING = auto()
    COMPLETED = auto()
    FAILED = auto()
    PAUSED = auto()


class DecisionStrategy(Enum):
    """决策策略枚举"""
    RULE_BASED = auto()
    LLM_BASED = auto()
    HYBRID = auto()
    ADAPTIVE = auto()


class LoopFeature(Enum):
    """循环特性枚举"""
    LLM_REASONING = "llm_reasoning"
    SELF_CORRECTION = "self_correction"
    DELEGATION = "delegation"
    LEARNING = "learning"
    PARALLEL_EXECUTION = "parallel_execution"
    PROGRESS_REPORTING = "progress_reporting"


@dataclass
class Observation:
    """观察结果"""
    timestamp: datetime
    state_summary: str
    relevant_data: Dict[str, Any]
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Thought:
    """思考结果"""
    timestamp: datetime
    reasoning: str
    decision: str
    confidence: float
    plan: List[Dict[str, Any]] = field(default_factory=list)
    tool_recommendations: List[str] = field(default_factory=list)
    risk_assessment: str = ""
    alternative_approaches: List[str] = field(default_factory=list)


@dataclass
class Action:
    """执行动作"""
    tool_name: str
    parameters: Dict[str, Any]
    expected_outcome: str
    is_delegation: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Verification:
    """验证结果"""
    success: bool
    actual_outcome: str
    expected_outcome: str
    differences: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningEntry:
    """学习记录"""
    timestamp: datetime
    situation: str
    action_taken: str
    outcome: str
    lesson: str
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoopResult:
    """循环结果"""
    success: bool
    iterations: int
    final_state: LoopState
    observations: List[Observation] = field(default_factory=list)
    thoughts: List[Thought] = field(default_factory=list)
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    learnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    execution_time_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoopConfig:
    """循环配置"""
    max_iterations: int = 10
    confidence_threshold: float = 0.8
    user_interruptible: bool = True
    decision_strategy: DecisionStrategy = DecisionStrategy.HYBRID
    features: Set[LoopFeature] = field(default_factory=lambda: {
        LoopFeature.LEARNING,
        LoopFeature.PROGRESS_REPORTING
    })
    timeout: int = 300
    max_retries: int = 3

    def has_feature(self, feature: LoopFeature) -> bool:
        """检查是否启用某特性"""
        return feature in self.features

    def enable_feature(self, feature: LoopFeature) -> None:
        """启用特性"""
        self.features.add(feature)

    def disable_feature(self, feature: LoopFeature) -> None:
        """禁用特性"""
        self.features.discard(feature)


class UnifiedAutonomousLoop(ABC):
    """统一的自主循环基类
    
    提供所有自主循环的公共功能：
    - Observe-Think-Act-Verify-Learn 循环
    - 状态管理
    - 进度报告
    - 错误处理
    - 学习机制
    """
    
    def __init__(
        self,
        config: Optional[LoopConfig] = None,
        progress_callback: Optional[Callable[[str, float, str], None]] = None,
        result_callback: Optional[Callable[[LoopResult], None]] = None
    ):
        """初始化自主循环
        
        Args:
            config: 循环配置
            progress_callback: 进度回调 (phase, progress, message)
            result_callback: 结果回调
        """
        self.config = config or LoopConfig()
        self.id = str(uuid4())
        
        self._state = LoopState.IDLE
        self._current_iteration = 0
        self._stop_requested = False
        self._pause_requested = False
        self._pause_event = asyncio.Event()
        self._pause_event.set()
        
        self._progress_callback = progress_callback
        self._result_callback = result_callback
        
        self._observations: List[Observation] = []
        self._thoughts: List[Thought] = []
        self._actions_taken: List[Dict[str, Any]] = []
        self._learnings: List[LearningEntry] = []
        self._errors: List[str] = []
        
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        
        self._execution_history: List[LoopResult] = []
        
        logger.info(f"[UnifiedAutonomousLoop:{self.id}] Initialized")
    
    @property
    def state(self) -> LoopState:
        """获取当前状态"""
        return self._state
    
    @state.setter
    def state(self, value: LoopState) -> None:
        """设置状态"""
        old_state = self._state
        self._state = value
        logger.debug(f"[UnifiedAutonomousLoop:{self.id}] State: {old_state.name} -> {value.name}")
    
    def _report_progress(self, phase: str, progress: float, message: str) -> None:
        """报告进度"""
        if self._progress_callback and self.config.has_feature(LoopFeature.PROGRESS_REPORTING):
            self._progress_callback(phase, progress, message)
    
    async def _check_pause(self) -> None:
        """检查暂停状态"""
        if self._pause_requested or not self._pause_event.is_set():
            self.state = LoopState.PAUSED
            self._report_progress("paused", 0, "Execution paused")
            await self._pause_event.wait()
            self.state = LoopState.THINKING
    
    def _should_continue(self) -> bool:
        """检查是否应该继续"""
        if self._stop_requested:
            return False
        if self._current_iteration >= self.config.max_iterations:
            return False
        return True
    
    def stop(self) -> None:
        """请求停止"""
        self._stop_requested = True
        logger.info(f"[UnifiedAutonomousLoop:{self.id}] Stop requested")
    
    def pause(self) -> None:
        """请求暂停"""
        self._pause_requested = True
        self._pause_event.clear()
        logger.info(f"[UnifiedAutonomousLoop:{self.id}] Pause requested")
    
    def resume(self) -> None:
        """恢复执行"""
        self._pause_requested = False
        self._pause_event.set()
        self.state = LoopState.THINKING
        logger.info(f"[UnifiedAutonomousLoop:{self.id}] Resumed")
    
    def reset(self) -> None:
        """重置循环"""
        self._stop_requested = False
        self._pause_requested = False
        self._pause_event.set()
        self._current_iteration = 0
        self._state = LoopState.IDLE
        self._observations.clear()
        self._thoughts.clear()
        self._actions_taken.clear()
        self._learnings.clear()
        self._errors.clear()
        logger.info(f"[UnifiedAutonomousLoop:{self.id}] Reset")
    
    @abstractmethod
    async def _observe(self, task: str, context: Dict[str, Any]) -> Observation:
        """观察当前状态
        
        Args:
            task: 当前任务
            context: 执行上下文
            
        Returns:
            Observation: 观察结果
        """
        pass
    
    @abstractmethod
    async def _think(self, task: str, observation: Observation, context: Dict[str, Any]) -> Thought:
        """思考下一步行动
        
        Args:
            task: 当前任务
            observation: 观察结果
            context: 执行上下文
            
        Returns:
            Thought: 思考结果
        """
        pass
    
    @abstractmethod
    async def _act(self, action: Action, context: Dict[str, Any]) -> Any:
        """执行动作
        
        Args:
            action: 要执行的动作
            context: 执行上下文
            
        Returns:
            Any: 执行结果
        """
        pass
    
    @abstractmethod
    async def _verify(self, result: Any, expected: str, context: Dict[str, Any]) -> Verification:
        """验证执行结果
        
        Args:
            result: 执行结果
            expected: 期望结果
            context: 执行上下文
            
        Returns:
            Verification: 验证结果
        """
        pass
    
    async def _learn(self, entry: LearningEntry) -> None:
        """从执行中学习
        
        Args:
            entry: 学习记录
        """
        if self.config.has_feature(LoopFeature.LEARNING):
            self._learnings.append(entry)
            logger.info(f"[UnifiedAutonomousLoop:{self.id}] Learned: {entry.lesson}")
    
    async def _self_correct(self, error: str, context: Dict[str, Any]) -> Optional[Thought]:
        """自我纠正
        
        Args:
            error: 错误信息
            context: 执行上下文
            
        Returns:
            Optional[Thought]: 纠正后的思考
        """
        if not self.config.has_feature(LoopFeature.SELF_CORRECTION):
            return None
        
        self._errors.append(error)
        
        correction_thought = Thought(
            timestamp=datetime.now(),
            reasoning=f"Self-correction triggered by error: {error}",
            decision="retry_with_adjustment",
            confidence=0.6,
            plan=[{"type": "retry", "adjustments": {"error_context": error}}]
        )
        
        return correction_thought
    
    async def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> LoopResult:
        """运行自主循环
        
        Args:
            task: 要执行的任务
            context: 执行上下文
            
        Returns:
            LoopResult: 循环结果
        """
        import time
        
        self._start_time = time.time()
        context = context or {}
        
        try:
            self.state = LoopState.OBSERVING
            self._report_progress("starting", 0, f"Starting task: {task[:50]}...")
            
            while self._should_continue():
                await self._check_pause()
                
                self._current_iteration += 1
                progress = self._current_iteration / self.config.max_iterations
                
                self.state = LoopState.OBSERVING
                self._report_progress("observing", progress, f"Observing (iteration {self._current_iteration})")
                observation = await self._observe(task, context)
                self._observations.append(observation)
                
                self.state = LoopState.THINKING
                self._report_progress("thinking", progress, "Thinking...")
                thought = await self._think(task, observation, context)
                self._thoughts.append(thought)
                
                if thought.confidence >= self.config.confidence_threshold:
                    self._report_progress("completed", 1.0, "Confidence threshold reached")
                    break
                
                for action_plan in thought.plan:
                    await self._check_pause()
                    
                    action = Action(
                        tool_name=action_plan.get("tool_name", "unknown"),
                        parameters=action_plan.get("parameters", {}),
                        expected_outcome=action_plan.get("expected_outcome", ""),
                        is_delegation=action_plan.get("is_delegation", False)
                    )
                    
                    self.state = LoopState.ACTING
                    self._report_progress("acting", progress, f"Executing: {action.tool_name}")
                    
                    try:
                        result = await self._act(action, context)
                        
                        self.state = LoopState.VERIFYING
                        verification = await self._verify(result, action.expected_outcome, context)
                        
                        action_record = {
                            "tool_name": action.tool_name,
                            "parameters": action.parameters,
                            "success": verification.success,
                            "result": result,
                            "iteration": self._current_iteration
                        }
                        self._actions_taken.append(action_record)
                        
                        if verification.success:
                            self.state = LoopState.LEARNING
                            await self._learn(LearningEntry(
                                timestamp=datetime.now(),
                                situation=str(observation.relevant_data)[:200],
                                action_taken=action.tool_name,
                                outcome=verification.actual_outcome[:200],
                                lesson=f"Action {action.tool_name} succeeded",
                                success=True
                            ))
                        else:
                            if self.config.has_feature(LoopFeature.SELF_CORRECTION):
                                correction = await self._self_correct(
                                    verification.differences[0] if verification.differences else "Unknown error",
                                    context
                                )
                                if correction:
                                    self._thoughts.append(correction)
                    
                    except Exception as e:
                        logger.error(f"[UnifiedAutonomousLoop:{self.id}] Action failed: {e}")
                        
                        if self.config.has_feature(LoopFeature.SELF_CORRECTION):
                            correction = await self._self_correct(str(e), context)
                            if correction:
                                self._thoughts.append(correction)
            
            self.state = LoopState.COMPLETED
            result = LoopResult(
                success=True,
                iterations=self._current_iteration,
                final_state=self._state,
                observations=self._observations.copy(),
                thoughts=self._thoughts.copy(),
                actions_taken=self._actions_taken.copy(),
                learnings=[l.lesson for l in self._learnings]
            )
            
        except Exception as e:
            logger.exception(f"[UnifiedAutonomousLoop:{self.id}] Loop failed: {e}")
            self.state = LoopState.FAILED
            
            result = LoopResult(
                success=False,
                iterations=self._current_iteration,
                final_state=self._state,
                observations=self._observations.copy(),
                thoughts=self._thoughts.copy(),
                actions_taken=self._actions_taken.copy(),
                learnings=[l.lesson for l in self._learnings],
                error=str(e)
            )
        
        finally:
            self._end_time = time.time()
            if self._start_time:
                result.execution_time_ms = int((self._end_time - self._start_time) * 1000)
            
            self._execution_history.append(result)
            
            if self._result_callback:
                self._result_callback(result)
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "id": self.id,
            "state": self._state.name,
            "current_iteration": self._current_iteration,
            "max_iterations": self.config.max_iterations,
            "observations_count": len(self._observations),
            "thoughts_count": len(self._thoughts),
            "actions_count": len(self._actions_taken),
            "learnings_count": len(self._learnings),
            "errors_count": len(self._errors),
            "execution_count": len(self._execution_history),
            "features": [f.value for f in self.config.features]
        }


def create_loop_config(
    max_iterations: int = 10,
    confidence_threshold: float = 0.8,
    features: Optional[List[LoopFeature]] = None,
    **kwargs
) -> LoopConfig:
    """创建循环配置"""
    config = LoopConfig(
        max_iterations=max_iterations,
        confidence_threshold=confidence_threshold,
        **kwargs
    )
    
    if features:
        config.features = set(features)
    
    return config
