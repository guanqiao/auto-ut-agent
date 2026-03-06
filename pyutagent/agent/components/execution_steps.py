"""Step Executor - Individual execution steps for the feedback loop."""

import logging
import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pyutagent.agent.base_agent import StepResult
from pyutagent.core.protocols import AgentState
from pyutagent.core.config import get_settings
from pyutagent.core.retry_config import RetryConfig as CoreRetryConfig, DEFAULT_RETRY_CONFIG
from pyutagent.core.error_classification import get_error_classification_service
from pyutagent.core.failure_pattern_tracker import FailurePatternTracker
from pyutagent.agent.thinking_engine import ThinkingEngine, ThinkingType, ThinkingSession
from pyutagent.agent.execution.retry import RetryConfig

logger = logging.getLogger(__name__)


class StepExecutor:
    """Executes individual steps in the feedback loop.
    
    Handles:
    - Target file parsing
    - Test generation (initial and additional)
    - Compilation with recovery
    - Test execution with recovery
    - Coverage analysis
    - Incremental fix handling
    """
    
    def __init__(self, agent_core: Any, components: Dict[str, Any], retry_config: Optional[RetryConfig] = None):
        """Initialize step executor.
        
        Args:
            agent_core: AgentCore instance
            components: Dictionary of all components
            retry_config: Optional retry configuration
        """
        self.agent_core = agent_core
        self.components = components
        
        self._capability_registry = components.get("capability_registry")
        
        if retry_config is None:
            if self._capability_registry:
                self.retry_config = self._get_capability_retry_config()
            else:
                settings = get_settings()
                self.retry_config = RetryConfig.from_core(CoreRetryConfig(
                    max_step_attempts=settings.coverage.max_step_attempts,
                    max_compilation_attempts=settings.coverage.max_compilation_attempts,
                    max_test_attempts=settings.coverage.max_test_attempts,
                ))
        else:
            self.retry_config = retry_config
        
        self.error_classifier = get_error_classification_service()

        self.failure_tracker = FailurePatternTracker(
            max_repeated_failures=3,
            pattern_expiry_seconds=600
        )

        self._thinking_engine: Optional[ThinkingEngine] = None
        self._thinking_session: Optional[ThinkingSession] = None
        self._enable_thinking = False

        self._global_attempt_count: int = 0
        self._step_attempt_counts: Dict[str, int] = {}

        logger.debug("[StepExecutor] Initialized with failure pattern tracking")
    
    def _get_capability_retry_config(self) -> RetryConfig:
        """Get retry config from capability system.
        
        Merges retry configurations from all ready capabilities,
        using the most conservative (strictest) settings.
        """
        configs = []
        for cap in self._capability_registry.get_all_ready():
            cap_config = cap.get_retry_config()
            if cap_config:
                configs.append(cap_config)
        
        if not configs:
            return RetryConfig(max_attempts=3)
        
        return RetryConfig(
            max_attempts=min(c.max_attempts for c in configs),
            base_delay=max(c.base_delay for c in configs),
            max_delay=max(c.max_delay for c in configs),
        )
    
    def reset_attempt_counts(self):
        """Reset all attempt counters."""
        self._global_attempt_count = 0
        self._step_attempt_counts.clear()
        logger.info("[StepExecutor] Attempt counters reset")
    
    def enable_thinking(self, llm_client: Any = None):
        """Enable thinking engine for enhanced error analysis.
        
        Args:
            llm_client: Optional LLM client for thinking engine
        """
        if self._thinking_engine is None:
            self._thinking_engine = ThinkingEngine(
                llm_client=llm_client,
                enable_deep_thinking=True,
                max_reasoning_steps=5,
                thinking_timeout=15.0,
                enable_prediction=True
            )
            self._thinking_session = self._thinking_engine.create_session()
            self._enable_thinking = True
            logger.info("[StepExecutor] Thinking engine enabled")
        else:
            logger.debug("[StepExecutor] Thinking engine already enabled")
    
    def get_attempt_stats(self) -> Dict[str, Any]:
        """Get current attempt statistics."""
        return {
            "global_attempts": self._global_attempt_count,
            "max_global_attempts": self.retry_config.max_total_attempts,
            "step_attempts": dict(self._step_attempt_counts),
        }
    
    async def execute_with_recovery(
        self,
        operation,
        *args,
        step_name: str = "operation",
        reset_count: int = 0,
        **kwargs
    ) -> StepResult:
        """Execute an operation with automatic error recovery.
        
        Now includes unified maximum attempt limits to prevent infinite loops.
        Also includes reset count limit to prevent infinite recursion.
        
        Args:
            operation: The operation to execute
            *args: Positional arguments
            step_name: Name of the step for logging
            reset_count: Current reset count (for preventing infinite recursion)
            **kwargs: Keyword arguments
            
        Returns:
            StepResult
        """
        attempt = 0
        max_attempts = self.retry_config.get_max_attempts(step_name)
        
        if self.retry_config.should_stop_reset(reset_count):
            logger.error(f"[StepExecutor] Exceeded maximum reset count - Step: {step_name}, ResetCount: {reset_count}/{self.retry_config.max_reset_count}")
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Exceeded maximum reset count ({self.retry_config.max_reset_count})"
            )
        
        if self._global_attempt_count >= self.retry_config.max_total_attempts:
            logger.error(f"[StepExecutor] Exceeded global attempt limit - GlobalAttempts: {self._global_attempt_count}/{self.retry_config.max_total_attempts}")
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Exceeded global attempt limit ({self.retry_config.max_total_attempts})"
            )
        
        logger.info(f"[StepExecutor] Starting step execution - Step: {step_name}, MaxAttempts: {max_attempts}, ResetCount: {reset_count}/{self.retry_config.max_reset_count}, GlobalAttempts: {self._global_attempt_count}/{self.retry_config.max_total_attempts}")
        
        while not self.agent_core._stop_requested and not self.agent_core._terminated:
            attempt += 1
            self._global_attempt_count += 1
            
            if self._global_attempt_count >= self.retry_config.max_total_attempts:
                logger.error(f"[StepExecutor] Exceeded global attempt limit - Step: {step_name}, GlobalAttempts: {self._global_attempt_count}/{self.retry_config.max_total_attempts}")
                return StepResult(
                    success=False,
                    state=AgentState.FAILED,
                    message=f"Exceeded global attempt limit ({self.retry_config.max_total_attempts})"
                )
            
            if self.retry_config.should_stop(attempt, step_name):
                logger.error(f"[StepExecutor] Exceeded maximum attempts - Step: {step_name}, Attempts: {attempt}/{max_attempts}")
                return StepResult(
                    success=False,
                    state=AgentState.FAILED,
                    message=f"Exceeded maximum attempts ({attempt}) for {step_name}"
                )
            
            logger.debug(f"[StepExecutor] Step attempt - Step: {step_name}, Attempt: {attempt}/{max_attempts}")
            
            try:
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)
                
                if result.success:
                    logger.info(f"[StepExecutor] Step executed successfully - Step: {step_name}, Attempt: {attempt}")
                    return result
                else:
                    logger.warning(f"[StepExecutor] Step returned failure - Step: {step_name}, Attempt: {attempt}, Message: {result.message}")
                    
                    error = Exception(result.message)
                    recovery_result = await self._try_recover(
                        error,
                        {"step": step_name, "attempt": attempt, "result": result, "reset_count": reset_count}
                    )
                    
                    if not recovery_result.get("should_continue", True):
                        logger.error(f"[StepExecutor] Recovery failed, step terminated - Step: {step_name}")
                        return StepResult(
                            success=False,
                            state=AgentState.FAILED,
                            message=f"Recovery failed for {step_name}"
                        )
                    
                    action = recovery_result.get("action", "retry")
                    logger.info(f"[StepExecutor] Applying recovery action - Action: {action}")
                    
                    if action == "fix":
                        fixed_code = recovery_result.get("fixed_code")
                        if fixed_code:
                            await self._write_test_file(fixed_code)
                    elif action == "reset":
                        if self.retry_config.can_reset(reset_count):
                            logger.info(f"[StepExecutor] Resetting and regenerating - ResetCount: {reset_count + 1}/{self.retry_config.max_reset_count}")
                            return await self.execute_with_recovery(
                                self.generate_initial_tests,
                                step_name="regenerating tests",
                                reset_count=reset_count + 1
                            )
                        else:
                            logger.error(f"[StepExecutor] Reset denied - max reset count reached ({self.retry_config.max_reset_count})")
                            return StepResult(
                                success=False,
                                state=AgentState.FAILED,
                                message=f"Cannot reset: max reset count ({self.retry_config.max_reset_count}) reached"
                            )
                    
                    delay = self.retry_config.get_delay(attempt)
                    if delay > 0:
                        logger.debug(f"[StepExecutor] Waiting {delay:.1f}s before retry")
                        await asyncio.sleep(delay)
                    
                    continue
                    
            except Exception as e:
                logger.exception(f"[StepExecutor] Step execution exception - Step: {step_name}, Attempt: {attempt}, Error: {e}")

                # Record failure pattern for intelligent retry
                self.failure_tracker.record_failure(e, step_name, {"attempt": attempt})

                # Check if we should stop retrying this pattern
                if self.failure_tracker.should_stop_retrying(e, step_name):
                    recommendation = self.failure_tracker.get_recommendation(e, step_name)
                    logger.warning(f"[StepExecutor] Repeated failure detected - Step: {step_name}, Recommendation: {recommendation}")

                    if recommendation == "skip_file":
                        return StepResult(
                            success=False,
                            state=AgentState.FAILED,
                            message=f"Repeated compilation failures, skipping file"
                        )
                    elif recommendation == "accept_partial":
                        return StepResult(
                            success=True,
                            state=AgentState.COMPLETED,
                            message=f"Accepting partial results after repeated test failures",
                            data={"partial_success": True}
                        )

                recovery_result = await self._try_recover(
                    e,
                    {"step": step_name, "attempt": attempt, "reset_count": reset_count}
                )
                
                if not recovery_result.get("should_continue", True):
                    logger.error(f"[StepExecutor] Recovery failed, step terminated - Step: {step_name}")
                    return StepResult(
                        success=False,
                        state=AgentState.FAILED,
                        message=f"Recovery failed for {step_name}: {str(e)}"
                    )
                
                action = recovery_result.get("action", "retry")
                if action == "fix":
                    fixed_code = recovery_result.get("fixed_code")
                    if fixed_code:
                        await self._write_test_file(fixed_code)
                elif action == "skip":
                    logger.info(f"[StepExecutor] Skipping step - Step: {step_name}")
                    return StepResult(
                        success=True,
                        state=AgentState.COMPLETED,
                        message=f"Skipped {step_name}",
                        data={}
                    )
                elif action == "reset":
                    if self.retry_config.can_reset(reset_count):
                        logger.info(f"[StepExecutor] Resetting and regenerating - ResetCount: {reset_count + 1}/{self.retry_config.max_reset_count}")
                        return await self.execute_with_recovery(
                            self.generate_initial_tests,
                            step_name="regenerating tests",
                            reset_count=reset_count + 1
                        )
                    else:
                        logger.error(f"[StepExecutor] Reset denied - max reset count reached ({self.retry_config.max_reset_count})")
                        return StepResult(
                            success=False,
                            state=AgentState.FAILED,
                            message=f"Cannot reset: max reset count ({self.retry_config.max_reset_count}) reached"
                        )
                
                delay = self.retry_config.get_delay(attempt)
                if delay > 0:
                    logger.debug(f"[StepExecutor] Waiting {delay:.1f}s before retry")
                    await asyncio.sleep(delay)
                
                continue
        
        if self.agent_core._terminated:
            logger.info(f"[StepExecutor] Step terminated - Step: {step_name}")
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message="Operation terminated by user"
            )
        
        logger.info(f"[StepExecutor] User stopped step - Step: {step_name}")
        return StepResult(
            success=False,
            state=AgentState.PAUSED,
            message="Operation stopped by user"
        )
    
    async def parse_target_file(self, target_file: str) -> StepResult:
        """Parse the target Java file.
        
        Args:
            target_file: Path to target file
            
        Returns:
            StepResult with parsing results
        """
        logger.info(f"[StepExecutor] Parsing target file - File: {target_file}")
        
        try:
            file_path = Path(self.agent_core.project_path) / target_file
            if not file_path.exists():
                logger.error(f"[StepExecutor] Target file not found - Path: {file_path}")
                return StepResult(
                    success=False,
                    state=AgentState.FAILED,
                    message=f"File not found: {target_file}"
                )
            
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            logger.debug(f"[StepExecutor] Read file content - Length: {len(source_code)}")
            
            parsed_class = self.components["java_parser"].parse(source_code.encode('utf-8'))
            
            class_info = {
                'name': parsed_class.name,
                'package': parsed_class.package,
                'methods': [
                    {
                        'name': m.name,
                        'return_type': m.return_type,
                        'parameters': m.parameters,
                        'modifiers': m.modifiers,
                        'annotations': m.annotations,
                    }
                    for m in parsed_class.methods
                ],
                'fields': parsed_class.fields,
                'imports': parsed_class.imports,
                'annotations': parsed_class.annotations,
                'source': source_code,
            }
            
            logger.info(f"[StepExecutor] Parsing complete - Class: {class_info.get('name', 'unknown')}, Methods: {len(class_info.get('methods', []))}")
            
            return StepResult(
                success=True,
                state=AgentState.PARSING,
                message=f"Successfully parsed {class_info.get('name', 'unknown')}",
                data={"class_info": class_info, "source_code": source_code}
            )
        except Exception as e:
            logger.exception(f"[StepExecutor] Failed to parse file: {e}")
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Error parsing file: {str(e)}"
            )
    
    async def generate_initial_tests(self, use_streaming: bool = True, max_retries: int = 2) -> StepResult:
        """Generate initial test cases with context management, streaming and quality evaluation.
        
        Enhanced with P4 intelligent features:
        - Strategy selection before generation
        - Self-reflection after generation
        - Feedback recording for learning
        
        Args:
            use_streaming: Whether to use streaming generation
            max_retries: Maximum number of retries on timeout
            
        Returns:
            StepResult with generation results
        """
        class_name = self.agent_core.target_class_info.get('name', 'Unknown')
        logger.info(f"[StepExecutor] 🎯 Starting test generation for class: {class_name}")
        logger.info(f"[StepExecutor] ⚙️ Configuration - Streaming: {use_streaming}, MaxRetries: {max_retries}")
        
        self.agent_core._update_state(AgentState.GENERATING, f"🚀 正在为 {class_name} 生成测试代码...")
        
        # P4: Strategy Selection - Choose optimal test strategy before generation
        selected_strategy = None
        if hasattr(self.agent_core, 'strategy_selector') and self.agent_core.strategy_selector:
            try:
                source_code = self.agent_core.target_class_info.get("source", "")
                strategy_result = self.agent_core.strategy_selector.select_strategy(
                    source_code=source_code,
                    class_info=self.agent_core.target_class_info
                )
                selected_strategy = strategy_result.primary_strategy.value
                logger.info(f"[StepExecutor] 🧠 P4 Strategy selected: {selected_strategy} "
                           f"(confidence: {strategy_result.confidence:.2f})")
                self.agent_core._update_state(
                    AgentState.GENERATING, 
                    f"🧠 智能策略选择: {selected_strategy}"
                )
            except Exception as e:
                logger.warning(f"[StepExecutor] Strategy selection failed: {e}")
        
        if hasattr(self.agent_core.llm_client, 'set_progress_callback'):
            self.agent_core.llm_client.set_progress_callback(self._llm_progress_callback)
            self.agent_core.llm_client.reset_cancel()
        
        try:
            source_code = self.agent_core.target_class_info.get("source", "")
            
            compressed_context = self.components["context_compressor"].build_context(
                query=f"Generate tests for {self.agent_core.target_class_info.get('name', 'class')}",
                target_file=None,
                additional_context={"class_info": self.agent_core.target_class_info}
            )
            
            if compressed_context.snippets_included > 0:
                logger.info(f"[StepExecutor] Context compression - "
                           f"Snippets: {compressed_context.snippets_included}, "
                           f"Tokens: {compressed_context.total_tokens}")
            
            effective_source = compressed_context.content if compressed_context.content else source_code
            
            base_prompt = self.components["prompt_builder"].build_initial_test_prompt(
                class_info=self.agent_core.target_class_info,
                source_code=effective_source
            )
            
            # Phase 2: Use IntelligenceEnhancedCoT for enhanced prompts
            if "intelligence_enhanced_cot" in self.components:
                try:
                    self.agent_core._update_state(AgentState.GENERATING, "🧠 正在生成智能增强提示词...")
                    
                    source_code = self.agent_core.target_class_info.get("source", "")
                    
                    # Create a simple JavaClass-like object for semantic analysis
                    class JavaClassProxy:
                        def __init__(self, class_info):
                            self.name = class_info.get('name', '')
                            self.package = class_info.get('package', '')
                            self.methods = []
                            for m in class_info.get('methods', []):
                                method_proxy = type('obj', (object,), {
                                    'name': m.get('name', ''),
                                    'return_type': m.get('return_type', ''),
                                    'parameters': m.get('parameters', []),
                                    'modifiers': m.get('modifiers', []),
                                    'annotations': m.get('annotations', [])
                                })
                                self.methods.append(method_proxy)
                    
                    java_class_proxy = JavaClassProxy(self.agent_core.target_class_info)
                    target_file_path = str(Path(self.agent_core.project_path) / self.agent_core.working_memory.current_file)
                    
                    enhanced_prompt = self.components["intelligence_enhanced_cot"].generate_enhanced_test_prompt(
                        source_code=source_code,
                        java_class=java_class_proxy,
                        file_path=target_file_path
                    )
                    
                    # Combine base prompt with enhanced insights
                    prompt = f"{base_prompt}\n\n---\n\n{enhanced_prompt}"
                    logger.info("[StepExecutor] ✅ Phase 2: Enhanced prompt generation complete")
                except Exception as e:
                    logger.warning(f"[StepExecutor] Phase 2: Failed to generate enhanced prompt, falling back: {e}")
                    prompt = self._optimize_prompt(base_prompt, "test_generation")
            else:
                prompt = self._optimize_prompt(base_prompt, "test_generation")
            
            logger.debug(f"[StepExecutor] Initial test prompt - Length: {len(prompt)}, Model: {self.agent_core.model_name}")
            
            test_code = None
            last_error = None
            
            # Retry loop for handling timeouts
            for attempt in range(max_retries + 1):
                # Check for termination before each attempt
                if self.agent_core._terminated or self.agent_core._stop_requested:
                    logger.warning(f"[StepExecutor] ⏹️ Generation cancelled by user before attempt {attempt + 1}")
                    return StepResult(
                        success=False,
                        state=AgentState.PAUSED,
                        message="⏹️ 测试生成已被用户取消"
                    )
                
                if attempt > 0:
                    logger.warning(f"[StepExecutor] 🔄 Retry attempt {attempt}/{max_retries} after previous failure")
                    self.agent_core._update_state(AgentState.GENERATING, f"🔄 重试 {attempt}/{max_retries}: 重新生成测试代码...")
                    # Add a small delay before retry to allow any transient issues to resolve
                    await asyncio.sleep(2.0 * attempt)  # Exponential backoff: 2s, 4s, ...
                    
                    # Check again after sleep
                    if self.agent_core._terminated or self.agent_core._stop_requested:
                        logger.warning(f"[StepExecutor] ⏹️ Generation cancelled by user after sleep")
                        return StepResult(
                            success=False,
                            state=AgentState.PAUSED,
                            message="⏹️ 测试生成已被用户取消"
                        )
                
                try:
                    if use_streaming:
                        logger.info("[StepExecutor] Using streaming generation")
                        self.agent_core._update_state(AgentState.GENERATING, "🚀 正在使用流式生成测试代码...")
                        
                        chunk_count = 0
                        total_chars = 0
                        last_update_time = time.time()
                        streaming_start_time = time.time()
                        
                        def on_chunk(chunk: str):
                            nonlocal chunk_count, total_chars, last_update_time
                            chunk_count += 1
                            total_chars += len(chunk)
                            
                            if self.agent_core.progress_callback:
                                self.agent_core.progress_callback({
                                    "type": "streaming_chunk",
                                    "chunk": chunk[:100] + "..." if len(chunk) > 100 else chunk
                                })
                            
                            current_time = time.time()
                            if current_time - last_update_time >= 3:
                                logger.info(f"[StepExecutor] 📥 已接收 {chunk_count} 个数据块，累计 {total_chars} 字符")
                                last_update_time = current_time
                        
                        streaming_timeout = float(self.agent_core.llm_client.timeout)
                        
                        # Create streaming task for cancellation support
                        streaming_task = asyncio.create_task(
                            self.components["streaming_generator"].generate_with_streaming(
                                prompt=prompt,
                                on_chunk=on_chunk,
                                on_progress=lambda p: logger.debug(f"[StepExecutor] Streaming progress: {p:.1%}")
                            )
                        )
                        
                        streaming_result = None
                        streaming_error = None
                        
                        try:
                            # Wait with periodic termination checks
                            while not streaming_task.done():
                                # Check for termination
                                if self.agent_core._terminated or self.agent_core._stop_requested:
                                    logger.warning("[StepExecutor] ⏹️ Cancelling streaming task due to user request")
                                    streaming_task.cancel()
                                    raise asyncio.CancelledError("User terminated")
                                
                                try:
                                    streaming_result = await asyncio.wait_for(
                                        asyncio.shield(streaming_task),
                                        timeout=1.0  # Check every second
                                    )
                                    break
                                except asyncio.TimeoutError:
                                    continue
                            
                            if streaming_result is None:
                                streaming_result = streaming_task.result()
                        except asyncio.TimeoutError as e:
                            logger.warning(f"[StepExecutor] ⏰ Streaming generation timeout ({streaming_timeout/60:.0f} minutes), falling back to normal generation")
                            self.agent_core._update_state(AgentState.GENERATING, f"⏰ 流式生成超时 (>{streaming_timeout/60:.0f}分钟),切换到普通模式...")
                            streaming_error = e
                        except Exception as e:
                            # Catch all other exceptions (404, connection errors, etc.)
                            error_str = str(e).lower()
                            if "404" in error_str or "not found" in error_str:
                                logger.warning(f"[StepExecutor] ❌ Streaming endpoint not found (404), falling back to normal generation: {e}")
                                self.agent_core._update_state(AgentState.GENERATING, "⚠️ 流式端点不可用 (404)，切换到普通模式...")
                            else:
                                logger.warning(f"[StepExecutor] ⚠️ Streaming generation failed ({type(e).__name__}), falling back to normal generation: {e}")
                                self.agent_core._update_state(AgentState.GENERATING, f"⚠️ 流式生成失败，切换到普通模式...")
                            streaming_error = e
                        
                        if streaming_result and streaming_result.success:
                            test_code = self.agent_core._extract_java_code(streaming_result.complete_code)
                            self.agent_core._update_state(AgentState.GENERATING, f"✅ 流式生成完成 - {len(test_code)} 字符")
                            logger.info(f"[StepExecutor] Streaming generation complete - "
                                       f"Tokens: {streaming_result.total_tokens}, "
                                       f"Time: {streaming_result.total_time:.2f}s")
                        else:
                            if streaming_result:
                                logger.warning(f"[StepExecutor] Streaming generation failed: {streaming_result.state}")
                            # Fallback to normal generation using agenerate (same as GUI test_connection)
                            logger.info("[StepExecutor] 🔄 Falling back to normal generation using agenerate (OpenAI compatible)")
                            self.agent_core._update_state(AgentState.GENERATING, "🔄 使用普通模式生成 (兼容模式)...")
                            try:
                                response = await self.agent_core.llm_client.agenerate(prompt)
                                test_code = self.agent_core._extract_java_code(response)
                            except Exception as agenerate_error:
                                logger.error(f"[StepExecutor] ❌ Normal generation also failed: {agenerate_error}")
                                raise agenerate_error
                    else:
                        self.agent_core._update_state(AgentState.GENERATING, "🚀 正在调用 LLM 生成测试代码...")
                        response = await self.agent_core.llm_client.agenerate(prompt)
                        test_code = self.agent_core._extract_java_code(response)
                    
                    # If we got test code, break out of retry loop
                    if test_code and len(test_code) > 0:
                        logger.info(f"[StepExecutor] ✅ Test code generated successfully on attempt {attempt + 1}")
                        break
                    else:
                        logger.warning(f"[StepExecutor] ⚠️ Generated test code is empty on attempt {attempt + 1}")
                        last_error = Exception("Empty test code generated")
                        
                except asyncio.TimeoutError as e:
                    elapsed_time = time.time() - streaming_start_time if use_streaming else time.time() - last_update_time
                    logger.error(f"[StepExecutor] ⏰ Timeout on attempt {attempt + 1}/{max_retries + 1} (elapsed: {elapsed_time:.1f}s): {e}")
                    self.agent_core._update_state(AgentState.GENERATING, f"⏰ 第 {attempt + 1} 次尝试超时 (耗时 {elapsed_time:.1f}秒)")
                    last_error = e
                    if attempt < max_retries:
                        logger.info(f"[StepExecutor] 🔄 Will retry ({max_retries - attempt} attempts remaining)...")
                        continue  # Retry
                    else:
                        logger.error(f"[StepExecutor] ❌ All {max_retries + 1} attempts exhausted due to timeouts")
                        raise  # Re-raise if all retries exhausted
                except Exception as e:
                    logger.exception(f"[StepExecutor] ❌ Error on attempt {attempt + 1}/{max_retries + 1}: {e}")
                    self.agent_core._update_state(AgentState.GENERATING, f"❌ 第 {attempt + 1} 次尝试失败：{str(e)[:100]}")
                    last_error = e
                    if attempt < max_retries:
                        logger.info(f"[StepExecutor] 🔄 Will retry ({max_retries - attempt} attempts remaining)...")
                        continue  # Retry
                    else:
                        logger.error(f"[StepExecutor] ❌ All {max_retries + 1} attempts exhausted due to errors")
                        raise  # Re-raise if all retries exhausted
            
            # If we exhausted all retries and still no test code
            if not test_code or len(test_code) == 0:
                logger.error(f"[StepExecutor] ❌ Failed to generate test code after {max_retries + 1} attempts")
                self.agent_core._update_state(AgentState.FAILED, f"❌ 无法生成测试代码 (已尝试 {max_retries + 1} 次)")
                raise last_error or Exception("Failed to generate test code after all retries")
            
            logger.debug(f"[StepExecutor] Extracted test code - Length: {len(test_code)}")
            
            eval_result = self.components["generation_evaluator"].evaluate(
                test_code=test_code,
                target_class_info=self.agent_core.target_class_info
            )
            
            logger.info(f"[StepExecutor] Generation evaluation - Score: {eval_result.overall_score:.2f}, "
                       f"Acceptable: {eval_result.is_acceptable}")
            
            if not eval_result.is_acceptable:
                critical_issues = eval_result.get_critical_issues()
                if critical_issues:
                    logger.warning(f"[StepExecutor] Critical issues detected: {len(critical_issues)}")
            
            if eval_result.coverage_estimate:
                logger.info(f"[StepExecutor] Estimated coverage potential - "
                           f"Line: {eval_result.coverage_estimate.line_coverage_potential:.1%}, "
                           f"Method: {eval_result.coverage_estimate.method_coverage_potential:.1%}")
            
            prediction_result = self.components["error_predictor"].predict_compilation_errors(test_code)
            if prediction_result.predicted_errors:
                logger.info(f"[StepExecutor] Error prediction - "
                           f"Predictions: {len(prediction_result.predicted_errors)}, "
                           f"Risk: {prediction_result.overall_risk_score:.2f}")
                
                if prediction_result.overall_risk_score > 0.7:
                    logger.warning(f"[StepExecutor] High risk detected, may need additional review")
            
            class_name = self.agent_core.target_class_info.get("name", "Unknown")
            test_file_name = f"{class_name}Test.java"
            
            settings = get_settings()
            test_dir = Path(self.agent_core.project_path) / settings.project_paths.src_test_java
            package_path = self.agent_core.target_class_info.get("package", "").replace(".", "/")
            if package_path:
                test_dir = test_dir / package_path
            
            test_dir.mkdir(parents=True, exist_ok=True)
            test_file_path = test_dir / test_file_name
            
            with open(test_file_path, 'w', encoding='utf-8') as f:
                f.write(test_code)
            
            self.agent_core.current_test_file = str(test_file_path.relative_to(self.agent_core.project_path))
            self.agent_core.working_memory.add_generated_test(
                file=self.agent_core.current_test_file,
                method="initial",
                code=test_code
            )
            
            logger.info(f"[StepExecutor] Initial test generation complete - TestFile: {self.agent_core.current_test_file}")
            
            # P4: Self-Reflection - Critique generated test code
            critique_result = None
            if hasattr(self.agent_core, 'self_reflection') and self.agent_core.self_reflection:
                try:
                    self.agent_core._update_state(AgentState.GENERATING, "🧠 正在进行自我反思评估...")
                    critique_result = await self.agent_core.self_reflection.critique_generated_test(
                        test_code=test_code,
                        source_code=self.agent_core.target_class_info.get("source"),
                        class_info=self.agent_core.target_class_info
                    )
                    logger.info(f"[StepExecutor] 🧠 P4 Self-reflection - "
                               f"Quality: {critique_result.overall_quality_score:.2f}, "
                               f"Issues: {len(critique_result.identified_issues)}, "
                               f"Should regenerate: {critique_result.should_regenerate}")
                    
                    if critique_result.should_regenerate:
                        logger.warning(f"[StepExecutor] ⚠️ Self-reflection suggests regeneration")
                except Exception as e:
                    logger.warning(f"[StepExecutor] Self-reflection failed: {e}")
            
            # P4: Feedback Recording - Record generation result for learning
            if hasattr(self.agent_core, 'feedback_loop') and self.agent_core.feedback_loop:
                try:
                    from ...core.enhanced_feedback_loop import FeedbackType
                    self.agent_core.feedback_loop.record_feedback(
                        feedback_type=FeedbackType.TEST_PASS if eval_result.is_acceptable else FeedbackType.TEST_FAILURE,
                        context={
                            "class_name": class_name,
                            "strategy": selected_strategy,
                            "test_file": self.agent_core.current_test_file
                        },
                        outcome="success" if eval_result.is_acceptable else "needs_improvement",
                        details={
                            "quality_score": eval_result.overall_score,
                            "critique_score": critique_result.overall_quality_score if critique_result else None
                        }
                    )
                    logger.debug(f"[StepExecutor] 🧠 P4 Feedback recorded")
                except Exception as e:
                    logger.warning(f"[StepExecutor] Feedback recording failed: {e}")
            
            if self.agent_core.ab_test_id and hasattr(self.agent_core, 'current_ab_variant_id'):
                self._record_generation_result(success=True, response_time_ms=0)
            
            return StepResult(
                success=True,
                state=AgentState.GENERATING,
                message=f"Generated initial tests: {self.agent_core.current_test_file}",
                data={"test_file": self.agent_core.current_test_file, "test_code": test_code}
            )
        except asyncio.CancelledError:
            logger.warning("[StepExecutor] ⚠️ Test generation was cancelled by user")
            return StepResult(
                success=False,
                state=AgentState.PAUSED,
                message="⏹️ 测试生成已被用户取消"
            )
        except Exception as e:
            logger.exception(f"[StepExecutor] Failed to generate initial tests: {e}")
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Error generating tests: {str(e)}"
            )
        finally:
            if hasattr(self.agent_core.llm_client, 'set_progress_callback'):
                self.agent_core.llm_client.set_progress_callback(None)
    
    def _llm_progress_callback(self, message: str):
        """Callback for LLM progress updates."""
        logger.info(f"[StepExecutor] {message}")
        if self.agent_core.progress_callback:
            self.agent_core.progress_callback({
                "state": AgentState.GENERATING.name,
                "message": message,
                "progress": {
                    "iteration": f"{self.agent_core.working_memory.current_iteration}/{self.agent_core.working_memory.max_iterations}",
                    "coverage": f"{self.agent_core.working_memory.current_coverage:.1%}",
                    "target": f"{self.agent_core.working_memory.target_coverage:.1%}"
                }
            })
    
    async def generate_incremental_tests(
        self,
        incremental_context: Any,
        use_streaming: bool = True,
        max_retries: int = 2
    ) -> StepResult:
        """Generate tests in incremental mode, preserving existing passing tests.
        
        Args:
            incremental_context: IncrementalContext with existing test analysis
            use_streaming: Whether to use streaming generation
            max_retries: Maximum number of retries on timeout
            
        Returns:
            StepResult with generation results
        """
        class_name = incremental_context.class_info.get('name', 'Unknown')
        logger.info(f"[StepExecutor] 🔄 Starting INCREMENTAL test generation for class: {class_name}")
        logger.info(f"[StepExecutor] ⚙️ Preserved tests: {len(incremental_context.preserved_test_names)}, "
                   f"Tests to fix: {len(incremental_context.tests_to_fix)}")
        
        self.agent_core._update_state(
            AgentState.GENERATING, 
            f"🔄 增量模式: 保留 {len(incremental_context.preserved_test_names)} 个通过的测试..."
        )
        
        if hasattr(self.agent_core.llm_client, 'set_progress_callback'):
            self.agent_core.llm_client.set_progress_callback(self._llm_progress_callback)
            self.agent_core.llm_client.reset_cancel()
        
        try:
            source_code = incremental_context.source_code
            
            uncovered_info = {
                "lines": [item.get("line") for item in incremental_context.uncovered_code if item.get("type") == "line"],
                "methods": [],
            }
            
            base_prompt = self.components["prompt_builder"].build_incremental_preserve_prompt(
                class_info=incremental_context.class_info,
                source_code=source_code,
                preserved_tests=incremental_context.preserved_test_names,
                preserved_test_code=incremental_context.preserved_test_code,
                tests_to_fix=incremental_context.tests_to_fix,
                uncovered_info=uncovered_info,
                current_coverage=1.0 - incremental_context.target_coverage_gap,
                target_coverage=self.agent_core.target_coverage
            )
            
            prompt = self._optimize_prompt(base_prompt, "incremental_test_generation")
            
            logger.debug(f"[StepExecutor] Incremental test prompt - Length: {len(prompt)}")
            
            test_code = None
            last_error = None
            
            for attempt in range(max_retries + 1):
                if self.agent_core._terminated or self.agent_core._stop_requested:
                    logger.warning(f"[StepExecutor] ⏹️ Incremental generation cancelled by user")
                    return StepResult(
                        success=False,
                        state=AgentState.PAUSED,
                        message="⏹️ 增量测试生成已被用户取消"
                    )
                
                if attempt > 0:
                    logger.warning(f"[StepExecutor] 🔄 Retry attempt {attempt}/{max_retries}")
                    self.agent_core._update_state(AgentState.GENERATING, f"🔄 重试 {attempt}/{max_retries}...")
                    await asyncio.sleep(2.0 * attempt)
                
                try:
                    if use_streaming:
                        logger.info("[StepExecutor] Using streaming generation for incremental mode")
                        self.agent_core._update_state(AgentState.GENERATING, "🚀 正在生成增量测试代码...")
                        
                        streaming_start_time = time.time()
                        
                        def on_chunk(chunk: str):
                            pass
                        
                        streaming_result = None
                        streaming_task = asyncio.create_task(
                            self.components["streaming_generator"].generate_with_streaming(
                                prompt=prompt,
                                on_chunk=on_chunk,
                                on_progress=lambda p: logger.debug(f"[StepExecutor] Streaming progress: {p:.1%}")
                            )
                        )
                        
                        try:
                            while not streaming_task.done():
                                if self.agent_core._terminated or self.agent_core._stop_requested:
                                    streaming_task.cancel()
                                    raise asyncio.CancelledError("User terminated")
                                
                                try:
                                    streaming_result = await asyncio.wait_for(
                                        asyncio.shield(streaming_task),
                                        timeout=1.0
                                    )
                                    break
                                except asyncio.TimeoutError:
                                    continue
                            
                            if streaming_result and streaming_result.success:
                                test_code = self.agent_core._extract_java_code(streaming_result.complete_code)
                                logger.info(f"[StepExecutor] Streaming incremental generation complete")
                            else:
                                # Fallback to normal generation (OpenAI compatible)
                                logger.info("[StepExecutor] 🔄 Falling back to normal generation for incremental tests")
                                response = await self.agent_core.llm_client.agenerate(prompt, timeout=180.0)
                                test_code = self.agent_core._extract_java_code(response)
                        except asyncio.TimeoutError:
                            logger.warning("[StepExecutor] Streaming timeout, falling back")
                            response = await self.agent_core.llm_client.agenerate(prompt, timeout=180.0)
                            test_code = self.agent_core._extract_java_code(response)
                        except Exception as e:
                            # Catch all other exceptions (404, connection errors, etc.)
                            error_str = str(e).lower()
                            if "404" in error_str or "not found" in error_str:
                                logger.warning(f"[StepExecutor] ❌ Streaming endpoint not found (404), falling back: {e}")
                            else:
                                logger.warning(f"[StepExecutor] ⚠️ Streaming failed ({type(e).__name__}), falling back: {e}")
                            response = await self.agent_core.llm_client.agenerate(prompt, timeout=180.0)
                            test_code = self.agent_core._extract_java_code(response)
                    else:
                        self.agent_core._update_state(AgentState.GENERATING, "🚀 正在调用 LLM 生成增量测试代码...")
                        response = await self.agent_core.llm_client.agenerate(prompt, timeout=180.0)
                        test_code = self.agent_core._extract_java_code(response)
                    
                    if test_code and len(test_code) > 0:
                        logger.info(f"[StepExecutor] ✅ Incremental test code generated successfully")
                        break
                    else:
                        logger.warning(f"[StepExecutor] ⚠️ Generated test code is empty")
                        last_error = Exception("Empty test code generated")
                        
                except asyncio.TimeoutError as e:
                    logger.error(f"[StepExecutor] ⏰ Timeout on attempt {attempt + 1}")
                    last_error = e
                    if attempt < max_retries:
                        continue
                    raise
                except Exception as e:
                    logger.exception(f"[StepExecutor] ❌ Error on attempt {attempt + 1}: {e}")
                    last_error = e
                    if attempt < max_retries:
                        continue
                    raise
            
            if not test_code or len(test_code) == 0:
                logger.error(f"[StepExecutor] ❌ Failed to generate incremental test code")
                raise last_error or Exception("Failed to generate incremental test code")
            
            settings = get_settings()
            test_file_name = f"{class_name}Test.java"
            test_dir = Path(self.agent_core.project_path) / settings.project_paths.src_test_java
            package_path = incremental_context.class_info.get("package", "").replace(".", "/")
            if package_path:
                test_dir = test_dir / package_path
            
            test_dir.mkdir(parents=True, exist_ok=True)
            test_file_path = test_dir / test_file_name
            
            with open(test_file_path, 'w', encoding='utf-8') as f:
                f.write(test_code)
            
            self.agent_core.current_test_file = str(test_file_path.relative_to(self.agent_core.project_path))
            self.agent_core.working_memory.add_generated_test(
                file=self.agent_core.current_test_file,
                method="incremental",
                code=test_code
            )
            
            logger.info(f"[StepExecutor] Incremental test generation complete - "
                       f"TestFile: {self.agent_core.current_test_file}, "
                       f"Preserved: {len(incremental_context.preserved_test_names)}")
            
            return StepResult(
                success=True,
                state=AgentState.GENERATING,
                message=f"Generated incremental tests (preserved {len(incremental_context.preserved_test_names)} tests): {self.agent_core.current_test_file}",
                data={
                    "test_file": self.agent_core.current_test_file,
                    "test_code": test_code,
                    "preserved_count": len(incremental_context.preserved_test_names),
                }
            )
        except asyncio.CancelledError:
            logger.warning("[StepExecutor] ⚠️ Incremental test generation was cancelled")
            return StepResult(
                success=False,
                state=AgentState.PAUSED,
                message="⏹️ 增量测试生成已被用户取消"
            )
        except Exception as e:
            logger.exception(f"[StepExecutor] Failed to generate incremental tests: {e}")
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Error generating incremental tests: {str(e)}"
            )
        finally:
            if hasattr(self.agent_core.llm_client, 'set_progress_callback'):
                self.agent_core.llm_client.set_progress_callback(None)
    
    async def compile_tests(self) -> StepResult:
        """Compile the generated tests asynchronously with validation.
        
        Returns:
            StepResult with compilation results
        """
        logger.info("[StepExecutor] Compiling tests with validation")

        validation_result = self.components["tool_validator"].validate_tool_call(
            "compile_tests",
            (),
            {"test_file": self.agent_core.current_test_file}
        )
        
        if not validation_result.valid:
            logger.warning(f"[StepExecutor] Tool validation failed: {validation_result.warnings}")
            if validation_result.has_errors:
                return StepResult(
                    success=False,
                    state=AgentState.FAILED,
                    message=f"Validation failed: {'; '.join(str(i.message) for i in validation_result.issues if hasattr(i, 'message'))}"
                )

        try:
            logger.debug("[StepExecutor] Getting Maven dependency classpath")
            
            from pyutagent.tools.maven_tools import find_maven_executable
            from pyutagent.tools.java_tools import find_javac_executable
            
            mvn_exe = find_maven_executable()
            if not mvn_exe:
                return StepResult(
                    success=False,
                    state=AgentState.FAILED,
                    message="Maven executable not found. Please configure Maven path in settings."
                )
            
            maven_process = await asyncio.create_subprocess_exec(
                mvn_exe, "dependency:build-classpath", "-Dmdep.outputFile=cp.txt", "-q",
                cwd=self.agent_core.project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await maven_process.communicate()

            classpath = ""
            cp_file = Path(self.agent_core.project_path) / "cp.txt"
            if cp_file.exists():
                classpath = cp_file.read_text(encoding='utf-8').strip()
                logger.debug(f"[StepExecutor] Classpath length: {len(classpath)}")

            settings = get_settings()
            classpath = f"{self.agent_core.project_path}/{settings.project_paths.target_classes};{self.agent_core.project_path}/{settings.project_paths.target_test_classes};{classpath}"

            test_file_path = Path(self.agent_core.project_path) / self.agent_core.current_test_file
            
            javac_exe = find_javac_executable()
            if not javac_exe:
                return StepResult(
                    success=False,
                    state=AgentState.FAILED,
                    message="Java compiler (javac) not found. Please configure JDK path in settings."
                )

            compile_process = await asyncio.create_subprocess_exec(
                javac_exe, "-cp", classpath,
                "-d", str(Path(self.agent_core.project_path) / "target" / "test-classes"),
                str(test_file_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await compile_process.communicate()

            if compile_process.returncode == 0:
                logger.info("[StepExecutor] Compilation successful")
                return StepResult(
                    success=True,
                    state=AgentState.COMPILING,
                    message="Tests compiled successfully"
                )
            else:
                stderr_text = stderr.decode('utf-8', errors='replace') if stderr else ""
                stdout_text = stdout.decode('utf-8', errors='replace') if stdout else ""
                
                errors = []
                if stderr_text.strip():
                    errors = [line.strip() for line in stderr_text.split('\n') if line.strip()]
                elif stdout_text.strip():
                    errors = [line.strip() for line in stdout_text.split('\n') if line.strip()]
                else:
                    errors = ["Unknown compilation error - no output from compiler"]
                
                logger.warning(f"[StepExecutor] Compilation failed - Errors: {len(errors)}, Return code: {compile_process.returncode}")
                
                # P4: Record compilation failure for learning
                if hasattr(self.agent_core, 'feedback_loop') and self.agent_core.feedback_loop:
                    try:
                        from ...core.enhanced_feedback_loop import FeedbackType
                        self.agent_core.feedback_loop.record_feedback(
                            feedback_type=FeedbackType.COMPILATION_FAILURE,
                            context={
                                "test_file": self.agent_core.current_test_file,
                                "class_name": self.agent_core.target_class_info.get("name", "Unknown")
                            },
                            outcome="compilation_failed",
                            details={"errors": errors[:5], "error_count": len(errors)}
                        )
                        logger.debug(f"[StepExecutor] 🧠 P4 Compilation failure recorded for learning")
                    except Exception as e:
                        logger.warning(f"[StepExecutor] Feedback recording failed: {e}")
                
                return StepResult(
                    success=False,
                    state=AgentState.FIXING,
                    message="Compilation failed",
                    data={"errors": errors, "stdout": stdout_text, "stderr": stderr_text}
                )
        except Exception as e:
            logger.exception(f"[StepExecutor] Compilation exception: {e}")
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Error compiling tests: {str(e)}"
            )
    
    async def compile_with_recovery(self, reset_count: int = 0) -> bool:
        """Compile tests with automatic error recovery.
        
        Args:
            reset_count: Current reset count (for preventing infinite recursion)
            
        Returns:
            True if compilation successful
        """
        if self.agent_core.working_memory.skip_verification:
            logger.info("[StepExecutor] Skipping compilation - defer_compilation mode enabled")
            self.agent_core._update_state(AgentState.GENERATING, "⏭️ 跳过编译检查 (批量生成模式)")
            return True
        
        attempt = 0
        
        if self.retry_config.should_stop_reset(reset_count):
            logger.error(f"[StepExecutor] ❌ Exceeded maximum reset count in compilation - ResetCount: {reset_count}/{self.retry_config.max_reset_count}")
            self.agent_core._update_state(
                AgentState.FAILED,
                f"❌ Exceeded max reset count ({self.retry_config.max_reset_count})."
            )
            return False
        
        logger.info(f"[StepExecutor] 🔨 Starting test compilation (with recovery) - Max attempts: {self.agent_core.max_compilation_attempts}, ResetCount: {reset_count}/{self.retry_config.max_reset_count}")
        
        while not self.agent_core._stop_requested and not self.agent_core._terminated:
            attempt += 1
            
            if attempt > self.agent_core.max_compilation_attempts:
                logger.error(f"[StepExecutor] ❌ Exceeded maximum compilation attempts ({self.agent_core.max_compilation_attempts})")
                self.agent_core._update_state(
                    AgentState.FAILED,
                    f"❌ Exceeded max compilation attempts ({self.agent_core.max_compilation_attempts}). Manual intervention required."
                )
                return False
            
            self.agent_core._update_state(AgentState.COMPILING, f"🔨 Attempt {attempt}/{self.agent_core.max_compilation_attempts}: Compiling tests...")
            
            logger.info(f"[StepExecutor] 🔨 Compilation attempt {attempt}/{self.agent_core.max_compilation_attempts} - Running Maven compile...")
            
            try:
                result = await self.compile_tests()
                
                if result.success:
                    logger.info(f"[StepExecutor] ✅ Compilation successful - Attempt: {attempt}")
                    self.agent_core._update_state(AgentState.COMPILING, "✅ Compilation successful")
                    return True
                else:
                    errors = result.data.get("errors", [])
                    
                    if not errors or len(errors) == 0:
                        logger.info("[StepExecutor] ✅ No compilation errors detected, proceeding...")
                        self.agent_core._update_state(AgentState.COMPILING, "✅ 编译通过")
                        return True
                    
                    self.agent_core._update_state(
                        AgentState.FIXING,
                        f"❌ Compilation failed with {len(errors)} error(s). Analyzing..."
                    )
                    
                    logger.warning(f"[StepExecutor] ❌ Compilation failed - Errors: {len(errors)}, calling LLM to fix...")
                    
                    error = Exception("Compilation failed: " + "\n".join(errors[:3]))
                    recovery_result = await self._try_recover(
                        error,
                        {"step": "compilation", "attempt": attempt, "compiler_output": "\n".join(errors), "reset_count": reset_count}
                    )
                    
                    if not recovery_result.get("should_continue", True):
                        logger.error("[StepExecutor] Compilation error recovery failed")
                        self.agent_core._update_state(AgentState.FAILED, "Recovery failed, cannot fix compilation errors")
                        return False
                    
                    action = recovery_result.get("action", "retry")
                    logger.info(f"[StepExecutor] 🔧 Compilation recovery action - Action: {action}")
                    
                    if action == "fix":
                        fixed_code = recovery_result.get("fixed_code")
                        if fixed_code:
                            await self._write_test_file(fixed_code)
                            self.agent_core._update_state(AgentState.FIXING, "🔧 Applied fix, retrying compilation...")
                            logger.info("[StepExecutor] 🔧 Applied LLM fix, retrying compilation...")
                    elif action == "reset":
                        if self.retry_config.can_reset(reset_count):
                            self.agent_core._update_state(AgentState.FIXING, f"🔄 Resetting and regenerating (Reset {reset_count + 1}/{self.retry_config.max_reset_count})...")
                            logger.info(f"[StepExecutor] 🔄 Resetting and regenerating tests - ResetCount: {reset_count + 1}/{self.retry_config.max_reset_count}")
                            reset_result = await self.execute_with_recovery(
                                self.generate_initial_tests,
                                step_name="regenerating after compilation failure",
                                reset_count=reset_count + 1
                            )
                            if not reset_result.success:
                                return False
                        else:
                            logger.error(f"[StepExecutor] Reset denied in compilation - max reset count reached ({self.retry_config.max_reset_count})")
                            self.agent_core._update_state(AgentState.FAILED, f"Cannot reset: max reset count ({self.retry_config.max_reset_count}) reached")
                            return False
                    elif action == "fallback":
                        self.agent_core._update_state(AgentState.FIXING, "🔄 Trying alternative approach...")
                        logger.info("[StepExecutor] 🔄 Trying alternative approach...")
                    
                    continue
                    
            except Exception as e:
                logger.exception(f"[StepExecutor] ❌ Compilation exception: {e}")
                self.agent_core._update_state(AgentState.FIXING, f"❌ Compilation error: {str(e)}")
                
                recovery_result = await self._try_recover(
                    e,
                    {"step": "compilation", "attempt": attempt, "reset_count": reset_count}
                )
                
                if not recovery_result.get("should_continue", True):
                    logger.error("[StepExecutor] ❌ Compilation recovery failed, cannot continue")
                    return False
                
                continue
        
        if self.agent_core._terminated:
            logger.info("[StepExecutor] ⏹️ Compilation terminated")
        else:
            logger.info("[StepExecutor] ⏹️ Compilation stopped (user request)")
        return False
    
    async def run_tests(self) -> StepResult:
        """Run the generated tests.
        
        Returns:
            StepResult with test results
        """
        logger.info("[StepExecutor] Running tests")
        
        try:
            success = self.components["maven_runner"].run_tests()
            
            if success:
                logger.info("[StepExecutor] All tests passed")
                return StepResult(
                    success=True,
                    state=AgentState.TESTING,
                    message="All tests passed"
                )
            else:
                failures = self._parse_test_failures()
                logger.warning(f"[StepExecutor] Tests failed - Failures: {len(failures)}")
                
                # P4: Record test failure for learning
                if hasattr(self.agent_core, 'feedback_loop') and self.agent_core.feedback_loop:
                    try:
                        from ...core.enhanced_feedback_loop import FeedbackType
                        self.agent_core.feedback_loop.record_feedback(
                            feedback_type=FeedbackType.TEST_FAILURE,
                            context={
                                "test_file": self.agent_core.current_test_file,
                                "class_name": self.agent_core.target_class_info.get("name", "Unknown")
                            },
                            outcome="test_failed",
                            details={"failures": failures[:3], "failure_count": len(failures)}
                        )
                        logger.debug(f"[StepExecutor] 🧠 P4 Test failure recorded for learning")
                    except Exception as e:
                        logger.warning(f"[StepExecutor] Feedback recording failed: {e}")
                
                return StepResult(
                    success=False,
                    state=AgentState.FIXING,
                    message="Some tests failed",
                    data={"failures": failures}
                )
        except Exception as e:
            logger.exception(f"[StepExecutor] Test execution exception: {e}")
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Error running tests: {str(e)}"
            )
    
    async def run_tests_with_recovery(self, reset_count: int = 0) -> bool:
        """Run tests with automatic error recovery and partial success handling.
        
        Args:
            reset_count: Current reset count (for preventing infinite recursion)
            
        Returns:
            True if tests pass
        """
        if self.agent_core.working_memory.skip_verification:
            logger.info("[StepExecutor] Skipping test execution - defer_compilation mode enabled")
            self.agent_core._update_state(AgentState.GENERATING, "⏭️ 跳过测试执行 (批量生成模式)")
            return True
        
        attempt = 0
        
        if self.retry_config.should_stop_reset(reset_count):
            logger.error(f"[StepExecutor] ❌ Exceeded maximum reset count in test execution - ResetCount: {reset_count}/{self.retry_config.max_reset_count}")
            self.agent_core._update_state(
                AgentState.FAILED,
                f"❌ Exceeded max reset count ({self.retry_config.max_reset_count})."
            )
            return False
        
        logger.info(f"[StepExecutor] 🧪 Starting test execution (with recovery and partial success handling) - Max attempts: {self.agent_core.max_test_attempts}, ResetCount: {reset_count}/{self.retry_config.max_reset_count}")
        
        while not self.agent_core._stop_requested and not self.agent_core._terminated:
            attempt += 1
            
            if attempt > self.agent_core.max_test_attempts:
                logger.error(f"[StepExecutor] ❌ Exceeded maximum test attempts ({self.agent_core.max_test_attempts})")
                self.agent_core._update_state(
                    AgentState.FAILED,
                    f"❌ Exceeded max test attempts ({self.agent_core.max_test_attempts}). Manual intervention required."
                )
                return False
            
            self.agent_core._update_state(AgentState.TESTING, f"🧪 Attempt {attempt}/{self.agent_core.max_test_attempts}: Running tests...")
            
            logger.info(f"[StepExecutor] 🧪 Test run attempt {attempt}/{self.agent_core.max_test_attempts} - Running Maven test...")
            
            try:
                result = await self.run_tests()
                
                if result.success:
                    logger.info(f"[StepExecutor] ✅ All tests passed - Attempt: {attempt}")
                    self.agent_core._update_state(AgentState.TESTING, "✅ All tests passed")
                    return True
                else:
                    failures = result.data.get("failures", []) if result.data else []
                    
                    if not failures or len(failures) == 0:
                        logger.info("[StepExecutor] ✅ No test failures detected, proceeding...")
                        self.agent_core._update_state(AgentState.TESTING, "✅ 测试通过")
                        return True
                    
                    test_output = result.data.get("stdout", "") if result.data else ""
                    settings = get_settings()
                    surefire_dir = Path(self.agent_core.project_path) / settings.project_paths.target_surefire_reports
                    
                    partial_result = self.components["partial_success_handler"].analyze_test_results(
                        test_output=test_output,
                        surefire_reports_dir=surefire_dir if surefire_dir.exists() else None
                    )
                    
                    if partial_result.has_partial_success:
                        logger.info(f"[StepExecutor] 🔄 Partial success detected - "
                                   f"Passed: {partial_result.passed_tests}, Failed: {partial_result.failed_tests}")
                        
                        if self.components["partial_success_handler"].should_attempt_incremental_fix(partial_result):
                            logger.info("[StepExecutor] 🔄 Attempting incremental fix for failed tests only")
                            
                            incremental_success = await self._handle_incremental_fix(partial_result)
                            if incremental_success:
                                logger.info("[StepExecutor] ✅ Incremental fix successful")
                                return True
                            else:
                                logger.warning("[StepExecutor] ⚠️ Incremental fix failed, falling back to full fix")
                    
                    self.agent_core._update_state(
                        AgentState.FIXING,
                        f"❌ {len(failures)} test(s) failed. Analyzing..."
                    )
                    
                    logger.warning(f"[StepExecutor] ❌ Tests failed - Failures: {len(failures)}, calling LLM to fix...")
                    
                    error = Exception(f"Test failures: {len(failures)} tests failed")
                    recovery_result = await self._try_recover(
                        error,
                        {"step": "test_execution", "attempt": attempt, "failures": failures, "reset_count": reset_count}
                    )
                    
                    if not recovery_result.get("should_continue", True):
                        logger.error("[StepExecutor] Test failure recovery failed")
                        self.agent_core._update_state(AgentState.FAILED, "Recovery failed, cannot fix test failures")
                        return False
                    
                    action = recovery_result.get("action", "retry")
                    logger.info(f"[StepExecutor] 🔧 Test recovery action - Action: {action}")
                    
                    if action == "fix":
                        fixed_code = recovery_result.get("fixed_code")
                        if fixed_code:
                            await self._write_test_file(fixed_code)
                            self.agent_core._update_state(AgentState.FIXING, "🔧 Applied fix, retrying tests...")
                            logger.info("[StepExecutor] 🔧 Applied LLM fix, retrying tests...")
                    elif action == "reset":
                        if self.retry_config.can_reset(reset_count):
                            self.agent_core._update_state(AgentState.FIXING, f"🔄 Resetting and regenerating (Reset {reset_count + 1}/{self.retry_config.max_reset_count})...")
                            logger.info(f"[StepExecutor] 🔄 Resetting and regenerating tests - ResetCount: {reset_count + 1}/{self.retry_config.max_reset_count}")
                            reset_result = await self.execute_with_recovery(
                                self.generate_initial_tests,
                                step_name="regenerating after test failure",
                                reset_count=reset_count + 1
                            )
                            if not reset_result.success:
                                return False
                        else:
                            logger.error(f"[StepExecutor] Reset denied in test execution - max reset count reached ({self.retry_config.max_reset_count})")
                            self.agent_core._update_state(AgentState.FAILED, f"Cannot reset: max reset count ({self.retry_config.max_reset_count}) reached")
                            return False
                    
                    continue
                    
            except Exception as e:
                logger.exception(f"[StepExecutor] ❌ Test execution exception: {e}")
                self.agent_core._update_state(AgentState.FIXING, f"❌ Test execution error: {str(e)}")
                
                recovery_result = await self._try_recover(
                    e,
                    {"step": "test_execution", "attempt": attempt, "reset_count": reset_count}
                )
                
                if not recovery_result.get("should_continue", True):
                    logger.error("[StepExecutor] ❌ Test recovery failed, cannot continue")
                    return False
                
                continue
        
        if self.agent_core._terminated:
            logger.info("[StepExecutor] ⏹️ Test execution terminated")
        else:
            logger.info("[StepExecutor] ⏹️ Test execution stopped (user request)")
        return False
    
    async def analyze_coverage(self) -> StepResult:
        """Analyze test coverage with enhanced error handling and diagnostics.
        
        First attempts to use JaCoCo for precise coverage.
        Falls back to LLM estimation if JaCoCo is not available.
        
        Returns:
            StepResult with coverage analysis results
        """
        logger.info("[StepExecutor] Analyzing coverage with enhanced diagnostics")
        
        try:
            logger.debug("[StepExecutor] Generating coverage report")
            coverage_success = self.components["maven_runner"].generate_coverage()
            
            if not coverage_success:
                logger.warning("[StepExecutor] Maven coverage generation returned false, but continuing to parse")
            
            report = self.components["coverage_analyzer"].parse_report()
            
            if report:
                logger.info(f"[StepExecutor] Coverage analysis complete - Line: {report.line_coverage:.1%}, Branch: {report.branch_coverage:.1%}, Method: {report.method_coverage:.1%}")
                
                coverage_data = {
                    "line_coverage": report.line_coverage,
                    "branch_coverage": report.branch_coverage,
                    "method_coverage": report.method_coverage,
                    "report": report,
                    "source": "jacoco"
                }
                
                if report.line_coverage < 0.3:
                    logger.warning(f"[StepExecutor] Low coverage detected: {report.line_coverage:.1%}")
                    coverage_data["low_coverage_warning"] = True
                
                return StepResult(
                    success=True,
                    state=AgentState.ANALYZING,
                    message=f"Coverage: {report.line_coverage:.1%}",
                    data=coverage_data
                )
            else:
                logger.warning("[StepExecutor] JaCoCo report not found, falling back to LLM estimation")
                return await self._fallback_to_llm_coverage_estimation()
                
        except Exception as e:
            logger.warning(f"[StepExecutor] JaCoCo analysis failed: {e}, falling back to LLM estimation")
            return await self._fallback_to_llm_coverage_estimation()
    
    async def _fallback_to_llm_coverage_estimation(self) -> StepResult:
        """Fall back to LLM-based coverage estimation when JaCoCo is not available.
        
        Returns:
            StepResult with estimated coverage
        """
        logger.info("[StepExecutor] Using LLM for coverage estimation")
        
        source_code = None
        test_code = None
        class_info = None
        
        if self.agent_core.target_class_info:
            source_code = self.agent_core.target_class_info.get("source", "")
            class_info = self.agent_core.target_class_info
        
        if self.agent_core.current_test_file:
            try:
                test_file_path = Path(self.agent_core.project_path) / self.agent_core.current_test_file
                if test_file_path.exists():
                    test_code = test_file_path.read_text(encoding='utf-8')
            except Exception as e:
                logger.warning(f"[StepExecutor] Failed to read test file: {e}")
        
        if not source_code or not test_code:
            logger.warning("[StepExecutor] Insufficient data for LLM estimation, using quick heuristic")
            return self._quick_estimate_fallback(source_code, test_code, class_info)
        
        try:
            from ..llm_coverage_evaluator import LLMCoverageEvaluator, CoverageSource
            
            llm_client = getattr(self.agent_core, 'llm_client', None)
            if not llm_client:
                logger.warning("[StepExecutor] No LLM client available, using quick heuristic")
                return self._quick_estimate_fallback(source_code, test_code, class_info)
            
            evaluator = LLMCoverageEvaluator(llm_client)
            llm_report = await evaluator.evaluate_coverage(
                source_code,
                test_code,
                class_info
            )
            
            logger.info(
                f"[StepExecutor] LLM coverage estimation complete - "
                f"Line: {llm_report.line_coverage:.1%}, "
                f"Branch: {llm_report.branch_coverage:.1%}, "
                f"Method: {llm_report.method_coverage:.1%}, "
                f"Confidence: {llm_report.confidence:.1%}"
            )
            
            return StepResult(
                success=True,
                state=AgentState.ANALYZING,
                message=f"Coverage (LLM estimated): {llm_report.line_coverage:.1%}",
                data={
                    "line_coverage": llm_report.line_coverage,
                    "branch_coverage": llm_report.branch_coverage,
                    "method_coverage": llm_report.method_coverage,
                    "report": llm_report,
                    "source": CoverageSource.LLM_ESTIMATED.value,
                    "confidence": llm_report.confidence,
                    "uncovered_methods": llm_report.uncovered_methods,
                    "recommendations": llm_report.recommendations
                }
            )
        except Exception as e:
            logger.exception(f"[StepExecutor] LLM estimation failed: {e}")
            return self._quick_estimate_fallback(source_code, test_code, class_info)
    
    def _quick_estimate_fallback(
        self,
        source_code: Optional[str],
        test_code: Optional[str],
        class_info: Optional[Dict[str, Any]]
    ) -> StepResult:
        """Quick heuristic-based fallback when LLM is not available.
        
        Args:
            source_code: Source code being tested
            test_code: Test code
            class_info: Class information from parsing
            
        Returns:
            StepResult with heuristic coverage estimate
        """
        logger.info("[StepExecutor] Using quick heuristic for coverage estimation")
        
        if not source_code or not test_code:
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message="Coverage analysis failed: No JaCoCo report and insufficient data for estimation"
            )
        
        from ..llm_coverage_evaluator import LLMCoverageEvaluator, CoverageSource
        
        evaluator = LLMCoverageEvaluator(None)
        llm_report = evaluator.quick_estimate(
            source_code,
            test_code,
            class_info
        )
        
        return StepResult(
            success=True,
            state=AgentState.ANALYZING,
            message=f"Coverage (estimated): {llm_report.line_coverage:.1%}",
            data={
                "line_coverage": llm_report.line_coverage,
                "branch_coverage": llm_report.branch_coverage,
                "method_coverage": llm_report.method_coverage,
                "report": llm_report,
                "source": CoverageSource.LLM_ESTIMATED.value,
                "confidence": llm_report.confidence
            }
        )
    
    async def generate_additional_tests(self, coverage_data: Dict[str, Any]) -> StepResult:
        """Generate additional tests for uncovered code.
        
        Enhanced with P4 intelligent features:
        - Boundary analysis for edge cases
        - Pattern library for test templates
        - Knowledge graph for code relationships
        
        Args:
            coverage_data: Coverage analysis data
            
        Returns:
            StepResult with generation results
        """
        logger.info("[StepExecutor] Generating additional tests with P0 and P4 enhancements")
        
        try:
            report = coverage_data.get("report")
            uncovered_info = self._get_uncovered_info(report)
            
            # P4: Use boundary analyzer to identify edge cases
            boundary_suggestions = []
            if hasattr(self.agent_core, 'boundary_analyzer') and self.agent_core.boundary_analyzer:
                try:
                    source_code = self.agent_core.target_class_info.get("source", "")
                    class_analysis = self.agent_core.boundary_analyzer.analyze_class(source_code)
                    for method_name, analysis in class_analysis.items():
                        if analysis.parameters:
                            boundary_suggestions.append({
                                "method": method_name,
                                "parameters": [
                                    {"name": p.parameter_name, "boundaries": [str(b.value) for b in p.boundaries[:3]]}
                                    for p in analysis.parameters[:2]
                                ]
                            })
                    logger.info(f"[StepExecutor] 🧠 P4 Boundary analysis found {len(boundary_suggestions)} method suggestions")
                except Exception as e:
                    logger.warning(f"[StepExecutor] Boundary analysis failed: {e}")
            
            # P4: Get test patterns for additional tests
            test_patterns = []
            if hasattr(self.agent_core, 'pattern_library') and self.agent_core.pattern_library:
                try:
                    from ...memory.pattern_library import PatternCategory
                    patterns = self.agent_core.pattern_library.find_patterns(
                        category=PatternCategory.BOUNDARY,
                        min_confidence=0.6
                    )
                    test_patterns = [
                        {"name": p.name, "template": p.template[:200]}
                        for p in patterns[:3]
                    ]
                    logger.info(f"[StepExecutor] 🧠 P4 Found {len(test_patterns)} applicable patterns")
                except Exception as e:
                    logger.warning(f"[StepExecutor] Pattern lookup failed: {e}")
            
            logger.debug(f"[StepExecutor] Uncovered info - Lines: {len(uncovered_info.get('lines', []))}")
            
            test_file_path = Path(self.agent_core.project_path) / self.agent_core.current_test_file
            with open(test_file_path, 'r', encoding='utf-8') as f:
                current_test_code = f.read()
            
            # Add P4 insights to prompt context
            enhanced_uncovered_info = {
                **uncovered_info,
                "boundary_suggestions": boundary_suggestions,
                "test_patterns": test_patterns
            }
            
            prompt = self.components["prompt_builder"].build_additional_tests_prompt(
                class_info=self.agent_core.target_class_info,
                existing_tests=current_test_code,
                uncovered_info=enhanced_uncovered_info,
                current_coverage=coverage_data.get("line_coverage", 0.0)
            )
            
            logger.debug(f"[StepExecutor] Additional tests prompt - Length: {len(prompt)}")
            
            response = await self.agent_core.llm_client.agenerate(prompt)
            additional_tests = self.agent_core._extract_java_code(response)
            
            logger.debug(f"[StepExecutor] Extracted additional test code - Length: {len(additional_tests)}")
            
            await self._append_tests_to_file(test_file_path, additional_tests)
            
            # P4: Record coverage improvement attempt
            if hasattr(self.agent_core, 'feedback_loop') and self.agent_core.feedback_loop:
                try:
                    from ...core.enhanced_feedback_loop import FeedbackType
                    self.agent_core.feedback_loop.record_feedback(
                        feedback_type=FeedbackType.COVERAGE_IMPROVEMENT,
                        context={
                            "test_file": self.agent_core.current_test_file,
                            "class_name": self.agent_core.target_class_info.get("name", "Unknown")
                        },
                        outcome="additional_tests_generated",
                        details={
                            "current_coverage": coverage_data.get("line_coverage", 0.0),
                            "uncovered_lines": len(uncovered_info.get("lines", [])),
                            "boundary_suggestions": len(boundary_suggestions)
                        }
                    )
                except Exception as e:
                    logger.warning(f"[StepExecutor] Feedback recording failed: {e}")
            
            logger.info("[StepExecutor] Additional test generation complete")
            
            return StepResult(
                success=True,
                state=AgentState.OPTIMIZING,
                message="Generated additional tests for uncovered code",
                data={"additional_tests": additional_tests, "boundary_suggestions": boundary_suggestions}
            )
        except Exception as e:
            logger.exception(f"[StepExecutor] Failed to generate additional tests: {e}")
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Error generating additional tests: {str(e)}"
            )
    
    async def _write_test_file(self, code: str):
        """Write test code to file.
        
        Args:
            code: Test code to write
        """
        if not self.agent_core.current_test_file:
            logger.warning("[StepExecutor] Cannot write test file - current_test_file is empty")
            return
        
        try:
            test_file_path = Path(self.agent_core.project_path) / self.agent_core.current_test_file
            test_file_path.write_text(code, encoding='utf-8')
            logger.info(f"[StepExecutor] Wrote test file - Path: {test_file_path}, Length: {len(code)}")
        except PermissionError as e:
            logger.error(f"[StepExecutor] Permission denied writing test file: {e}")
            self.agent_core._update_state(AgentState.FAILED, f"Permission denied: {e}")
        except OSError as e:
            logger.error(f"[StepExecutor] OS error writing test file: {e}")
            self.agent_core._update_state(AgentState.FAILED, f"File system error: {e}")
        except Exception as e:
            logger.exception(f"[StepExecutor] Failed to write test file: {e}")
            self.agent_core._update_state(AgentState.FAILED, f"Failed to write test file: {e}")
    
    def _parse_test_failures(self) -> List[Dict[str, Any]]:
        """Parse test failures from Maven output.
        
        Returns:
            List of test failures
        """
        failures = []
        settings = get_settings()
        surefire_dir = Path(self.agent_core.project_path) / settings.project_paths.target_surefire_reports
        
        if surefire_dir.exists():
            for report_file in surefire_dir.glob("*.txt"):
                content = report_file.read_text()
                if "FAILURE" in content or "ERROR" in content:
                    failures.append({
                        "test_name": report_file.stem,
                        "error": content[:500]
                    })
        
        logger.debug(f"[StepExecutor] Parsed test failures - Failures: {len(failures)}")
        return failures
    
    def _get_uncovered_info(self, report) -> Dict[str, Any]:
        """Get information about uncovered code.
        
        Args:
            report: Coverage report
            
        Returns:
            Dictionary with uncovered methods, lines, and branches
        """
        uncovered_info = {
            "methods": [],
            "lines": [],
            "branches": []
        }
        
        if report and report.files:
            for file_coverage in report.files:
                for line_num, is_covered in file_coverage.lines:
                    if not is_covered:
                        uncovered_info["lines"].append(line_num)
        
        logger.debug(f"[StepExecutor] Uncovered info - Lines: {len(uncovered_info['lines'])}")
        return uncovered_info
    
    async def _append_tests_to_file(self, test_file_path: Path, additional_tests: str) -> bool:
        """Append additional tests to existing test file using smart editor.
        
        Args:
            test_file_path: Path to test file
            additional_tests: Additional test code to append
            
        Returns:
            True if successful
        """
        try:
            with open(test_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            last_brace = content.rfind('}')
            if last_brace > 0:
                search_pattern = content[last_brace-50:last_brace+1] if last_brace > 50 else content[:last_brace+1]
                
                edit_result = await self.components["smart_editor"].apply_search_replace(
                    code=content,
                    search=search_pattern,
                    replace=search_pattern.rstrip('}') + "\n" + additional_tests + "\n}",
                    fuzzy=True
                )
                
                if edit_result.success:
                    with open(test_file_path, 'w', encoding='utf-8') as f:
                        f.write(edit_result.modified_code)
                    logger.debug(f"[StepExecutor] Smart appended tests - Path: {test_file_path}")
                    return True
            
            new_content = content[:last_brace] + "\n" + additional_tests + "\n" + content[last_brace:] if last_brace > 0 else content + "\n" + additional_tests
            
            with open(test_file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            logger.debug(f"[StepExecutor] Appended tests to file - Path: {test_file_path}, AddedLength: {len(additional_tests)}")
            return True
            
        except Exception as e:
            logger.error(f"[StepExecutor] Failed to append tests: {e}")
            return False
    
    async def _handle_incremental_fix(self, partial_result: Any) -> bool:
        """Handle incremental fix for partial test success.
        
        Args:
            partial_result: Partial test results with passed/failed tests
            
        Returns:
            True if incremental fix was successful
        """
        try:
            test_file_path = Path(self.agent_core.project_path) / self.agent_core.current_test_file
            with open(test_file_path, 'r', encoding='utf-8') as f:
                current_test_code = f.read()
            
            fix_prompt = self.components["partial_success_handler"].create_incremental_fix_prompt(
                test_code=current_test_code,
                partial_result=partial_result,
                target_class_info=self.agent_core.target_class_info
            )
            
            logger.debug(f"[StepExecutor] Incremental fix prompt - Length: {len(fix_prompt)}")
            
            response = await self.agent_core.llm_client.agenerate(fix_prompt)
            fixed_code = self.agent_core._extract_java_code(response)
            
            merge_result = self.components["partial_success_handler"].merge_incremental_fix(
                original_code=current_test_code,
                fixed_code=fixed_code,
                partial_result=partial_result
            )
            
            if merge_result.success and merge_result.new_test_code:
                await self._write_test_file(merge_result.new_test_code)
                
                logger.info(f"[StepExecutor] Incremental fix applied - "
                           f"Preserved: {len(merge_result.preserved_tests)}, "
                           f"Fixed: {len(merge_result.fixed_tests)}")
                
                verify_result = await self.run_tests()
                return verify_result.success
            else:
                logger.error(f"[StepExecutor] Incremental fix merge failed: {merge_result.error_message}")
                return False
                
        except Exception as e:
            logger.exception(f"[StepExecutor] Incremental fix failed: {e}")
            return False
    
    async def _try_recover(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Try to recover from an error with learning and optimization.
        
        Uses unified error classification service for consistent categorization.
        Includes smart detection of dependency issues for targeted recovery.
        Now includes smart LLM analysis for compilation and test failures.
        
        Args:
            error: The error that occurred
            context: Error context
            
        Returns:
            Recovery result
        """
        import time
        start_time = time.time()
        
        if self.error_classifier.should_skip_recovery(error):
            logger.info(f"[StepExecutor] Detected false positive error, skipping recovery")
            return {
                "should_continue": True,
                "action": "skip",
                "reason": "No actual error detected"
            }
        
        logger.info(f"[StepExecutor] Attempting recovery - Error: {error}, Context: {context}")
        
        detailed_error_info = self.error_classifier.get_detailed_error_info(error, context)
        error_category = self.error_classifier.classify(error, context)
        
        thinking_result = None
        
        if self._enable_thinking and self._thinking_engine and self._thinking_session:
            try:
                thinking_context = {
                    "error": str(error),
                    "error_category": error_category.name if hasattr(error_category, 'name') else str(error_category),
                    "step": context.get("step", "unknown"),
                    "attempt": context.get("attempt", 1),
                    "detailed_error": detailed_error_info
                }
                thinking_result = await self._thinking_engine.think_about_error(
                    error=error,
                    context=thinking_context,
                    session=self._thinking_session
                )
                logger.info(f"[StepExecutor] 🤖 Thinking engine result - Confidence: {thinking_result.confidence:.2f}, Conclusions: {len(thinking_result.conclusions)}")
                if thinking_result.recovery_strategy:
                    logger.info(f"[StepExecutor] 🤖 Recovery strategy suggestion: {thinking_result.recovery_strategy}")
            except Exception as e:
                logger.warning(f"[StepExecutor] Thinking engine error (non-fatal): {e}")
        
        step_name = context.get("step", "")
        
        if detailed_error_info.get("needs_dependency_resolution"):
            logger.info("[StepExecutor] Detected dependency issue, using INSTALL_DEPENDENCIES strategy")
            return await self._recover_from_dependency_issue(error, context, detailed_error_info)
        
        if detailed_error_info.get("is_environment_issue"):
            logger.info("[StepExecutor] Detected environment issue, using FIX_ENVIRONMENT strategy")
            return await self._recover_from_environment_issue(error, context, detailed_error_info)
        
        suggested_strategy = None
        
        if thinking_result and thinking_result.recovery_strategy:
            recovery_str = thinking_result.recovery_strategy.lower()
            if "fix" in recovery_str or "analyze" in recovery_str:
                from pyutagent.core.error_recovery import RecoveryStrategy
                suggested_strategy = (RecoveryStrategy.ANALYZE_AND_FIX, thinking_result.confidence)
                logger.info(f"[StepExecutor] 🤖 Using thinking engine strategy: {suggested_strategy[0].name}")
            elif "reset" in recovery_str or "regenerate" in recovery_str:
                from pyutagent.core.error_recovery import RecoveryStrategy
                suggested_strategy = (RecoveryStrategy.RESET_AND_REGENERATE, thinking_result.confidence)
                logger.info(f"[StepExecutor] 🤖 Using thinking engine strategy: {suggested_strategy[0].name}")
            elif "retry" in recovery_str:
                from pyutagent.core.error_recovery import RecoveryStrategy
                suggested_strategy = (RecoveryStrategy.RETRY_IMMEDIATE, thinking_result.confidence)
                logger.info(f"[StepExecutor] 🤖 Using thinking engine strategy: {suggested_strategy[0].name}")
        
        if step_name in ("compilation", "test_execution") and self.retry_config.enable_smart_retry:
            smart_result = await self._smart_recover_with_llm_analysis(error, context, step_name)
            if smart_result:
                return smart_result
        
        if suggested_strategy is None:
            suggested_strategy = self.components["error_learner"].suggest_strategy(error, error_category, context)
            if suggested_strategy:
                strategy, confidence = suggested_strategy
                logger.info(f"[StepExecutor] Error learner suggests {strategy.name} with confidence {confidence:.2f}")
                
                optimization = self.components["strategy_optimizer"].optimize_strategy_selection(error_category, context)
                logger.info(f"[StepExecutor] Strategy optimizer recommends {optimization.recommended_strategy.name}")
        
        current_test_code = None
        if self.agent_core.current_test_file:
            try:
                test_file_path = Path(self.agent_core.project_path) / self.agent_core.current_test_file
                if test_file_path.exists():
                    current_test_code = test_file_path.read_text(encoding='utf-8')
                    logger.debug(f"[StepExecutor] Read current test code - Length: {len(current_test_code)}")
            except Exception as e:
                logger.warning(f"[StepExecutor] Failed to read test code: {e}")
        
        from pyutagent.core.error_recovery import RecoveryStrategy
        
        if "intelligence_enhanced_cot" in self.components and current_test_code:
            try:
                self.agent_core._update_state(AgentState.FIXING, "🧠 正在进行根因分析...")
                
                error_message = str(error)
                source_code = self.agent_core.target_class_info.get("source", "")
                test_method = context.get("step", "unknown")
                
                enhanced_fix_prompt = self.components["intelligence_enhanced_cot"].generate_enhanced_fix_prompt(
                    error_message=error_message,
                    test_code=current_test_code,
                    source_code=source_code,
                    test_method=test_method
                )
                
                context = {**context, "enhanced_prompt": enhanced_fix_prompt}
                logger.info("[StepExecutor] ✅ Phase 2: Enhanced error analysis complete")
            except Exception as e:
                logger.warning(f"[StepExecutor] Phase 2: Failed to generate enhanced fix prompt, falling back: {e}")
        
        recovery_result = await self.components["error_recovery"].recover(
            error,
            error_context=context,
            current_test_code=current_test_code,
            target_class_info=self.agent_core.target_class_info
        )
        
        elapsed_time = time.time() - start_time
        success = recovery_result.get("action") not in ("abort", "fail")
        strategy_used = RecoveryStrategy.ANALYZE_AND_FIX
        if recovery_result.get("action") == "fix":
            strategy_used = RecoveryStrategy.ANALYZE_AND_FIX
        elif recovery_result.get("action") == "reset":
            strategy_used = RecoveryStrategy.RESET_AND_REGENERATE
        elif recovery_result.get("action") == "retry":
            strategy_used = RecoveryStrategy.RETRY_IMMEDIATE
        elif recovery_result.get("action") == "install_dependencies":
            strategy_used = RecoveryStrategy.INSTALL_DEPENDENCIES
        elif recovery_result.get("action") == "resolve_dependencies":
            strategy_used = RecoveryStrategy.RESOLVE_DEPENDENCIES
        
        self.components["error_learner"].learn_from_recovery(
            error=error,
            error_category=error_category,
            strategy=strategy_used,
            success=success,
            context=context,
            time_to_recover=elapsed_time,
            attempts_needed=context.get("attempt", 1)
        )
        
        self.components["strategy_optimizer"].record_result(
            error_category=error_category,
            strategy=strategy_used,
            success=success,
            time_taken=elapsed_time,
            attempts=context.get("attempt", 1)
        )
        
        logger.info(f"[StepExecutor] Recovery result - Action: {recovery_result.get('action')}, ShouldContinue: {recovery_result.get('should_continue')}")
        
        return recovery_result
    
    async def _smart_recover_with_llm_analysis(
        self,
        error: Exception,
        context: Dict[str, Any],
        step_name: str
    ) -> Optional[Dict[str, Any]]:
        """使用智能 LLM 分析进行恢复。
        
        这是增强版的恢复流程，会：
        1. 收集完整的编译/测试输出
        2. 调用 LLM 进行深度分析
        3. 解析 LLM 给出的行动方案
        4. 执行行动方案
        
        Args:
            error: 发生的异常
            context: 错误上下文
            step_name: 步骤名称
            
        Returns:
            恢复结果，如果无法智能恢复则返回 None
        """
        try:
            error_recovery = self.components.get("error_recovery")
            if not error_recovery or not hasattr(error_recovery, 'analyze_with_smart_context'):
                return None
            
            self.agent_core._update_state(
                AgentState.FIXING,
                f"🧠 正在使用 LLM 智能分析 {step_name} 错误..."
            )
            
            logger.info(f"[StepExecutor] 🧠 Starting smart LLM analysis for {step_name}")
            
            error_context = {
                **context,
                "source_file": self.agent_core.working_memory.current_file,
                "test_file": self.agent_core.current_test_file,
            }
            
            if step_name == "compilation":
                compiler_output = context.get("compiler_output", str(error))
                error_context["compiler_output"] = compiler_output
                error_type = "compilation"
            else:
                test_output = context.get("test_output", str(error))
                error_context["test_output"] = test_output
                error_type = "test_failure"
            
            analysis_result = await error_recovery.analyze_with_smart_context(
                error=error,
                error_context=error_context,
                error_type=error_type
            )
            
            if not analysis_result.get("success"):
                logger.warning(f"[StepExecutor] Smart analysis failed: {analysis_result.get('message')}")
                return None
            
            action_plan = analysis_result.get("action_plan", [])
            confidence = analysis_result.get("confidence", 0.5)
            
            logger.info(
                f"[StepExecutor] 🧠 Smart analysis complete - "
                f"Root cause: {analysis_result.get('root_cause', 'Unknown')[:100]}, "
                f"Actions: {len(action_plan)}, "
                f"Confidence: {confidence:.2f}"
            )
            
            if confidence < 0.5:
                logger.warning(f"[StepExecutor] ⚠️ Low confidence ({confidence:.2f}), falling back to retry instead of executing actions")
                return {
                    "should_continue": True,
                    "action": "retry",
                    "reason": f"Low confidence analysis ({confidence:.2f}), not executing potentially invalid actions",
                    "confidence": confidence
                }
            
            if not action_plan:
                logger.warning("[StepExecutor] No action plan from LLM analysis")
                return None
            
            for action in action_plan:
                action_type = action.get("action", "unknown")
                logger.info(f"[StepExecutor] 📋 LLM recommended action: {action_type}")
                
                if action_type in ("regenerate_test", "regenerate"):
                    logger.info("[StepExecutor] 🔄 LLM recommends test regeneration")
                    return {
                        "should_continue": True,
                        "action": "reset",
                        "reason": "LLM analysis recommends regeneration",
                        "confidence": confidence
                    }
                
                if action_type == "skip_test":
                    logger.info("[StepExecutor] ⏭️ LLM recommends skipping test")
                    return {
                        "should_continue": True,
                        "action": "skip",
                        "reason": "LLM analysis recommends skipping",
                        "confidence": confidence
                    }
            
            execution_result = await error_recovery.execute_action_plan(
                action_plan,
                {
                    **error_context,
                    "confidence": confidence,
                    "test_file": self.agent_core.current_test_file,
                    "project_path": self.agent_core.project_path
                }
            )
            
            results = execution_result.get("results", [])
            success_count = sum(1 for r in results if r.get("success", False))
            failed_count = len(results) - success_count
            
            unknown_failures = sum(
                1 for r in results 
                if not r.get("success", False) and 
                r.get("action", "").lower() in ("unknown", "unknown action")
            )
            
            if unknown_failures > 0:
                logger.warning(f"[StepExecutor] ⚠️ Detected {unknown_failures} unknown action failures out of {len(results)} total")
                
                if unknown_failures >= len(results) * 0.5 or unknown_failures >= 3:
                    logger.error(f"[StepExecutor] ❌ Too many unknown actions ({unknown_failures}/{len(results)}), LLM likely returned invalid actions. Stopping retry cycle.")
                    return {
                        "should_continue": False,
                        "action": "escalate",
                        "reason": f"LLM returned mostly invalid actions ({unknown_failures} unknown out of {len(results)}). Need manual intervention.",
                        "confidence": confidence,
                        "error_type": "invalid_llm_response"
                    }
            
            if execution_result.get("success") and success_count > 0:
                modified_files = execution_result.get("modified_files", [])
                logger.info(f"[StepExecutor] ✅ Smart recovery executed - Success: {success_count}/{len(results)}, Modified: {modified_files}")
                
                return {
                    "should_continue": True,
                    "action": "fix",
                    "message": f"Applied {success_count} fixes out of {len(results)} actions",
                    "modified_files": modified_files,
                    "confidence": confidence,
                    "analysis": analysis_result.get("analysis", "")
                }
            else:
                logger.warning(f"[StepExecutor] Smart recovery execution failed: {execution_result.get('message')}, Success: {success_count}/{len(results)}")
                
                if failed_count > 0 and success_count == 0:
                    logger.error(f"[StepExecutor] ❌ All actions failed ({failed_count}), escalating instead of retrying")
                    return {
                        "should_continue": False,
                        "action": "escalate",
                        "reason": f"All {failed_count} recovery actions failed. No more automatic retries.",
                        "confidence": confidence
                    }
                
                return {
                    "should_continue": True,
                    "action": "retry",
                    "reason": f"Smart recovery partially failed ({success_count}/{len(results)} succeeded), will retry",
                    "confidence": confidence
                }
                
        except Exception as e:
            logger.exception(f"[StepExecutor] Smart recovery failed: {e}")
            return None
    
    async def _recover_from_dependency_issue(
        self, 
        error: Exception, 
        context: Dict[str, Any],
        error_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """从依赖问题中恢复
        
        Args:
            error: 错误对象
            context: 错误上下文
            error_info: 详细错误信息
            
        Returns:
            恢复结果
        """
        from pyutagent.core.error_recovery import DependencyRecoveryHandler, RecoveryStrategy
        
        logger.info("[StepExecutor] 🔧 Starting dependency recovery...")
        
        dependency_info = error_info.get("dependency_info", {})
        missing_packages = dependency_info.get("missing_packages", [])
        is_test_dependency = dependency_info.get("is_test_dependency", False)
        
        if not missing_packages:
            compiler_output = context.get("compiler_output", str(error))
            from pyutagent.core.error_classification import detect_missing_dependencies
            dependency_info = detect_missing_dependencies(compiler_output)
            missing_packages = dependency_info.get("missing_packages", [])
            is_test_dependency = dependency_info.get("is_test_dependency", False)
        
        def _adapt_progress_callback(state: str, message: str):
            if self.agent_core.progress_callback:
                self.agent_core.progress_callback({
                    "state": state,
                    "message": message
                })
        
        compiler_output = context.get("compiler_output", str(error))
        
        handler = DependencyRecoveryHandler(
            project_path=self.agent_core.project_path,
            maven_runner=self.components.get("maven_runner"),
            llm_client=self.agent_core.llm_client,
            prompt_builder=self.components.get("prompt_builder"),
            progress_callback=_adapt_progress_callback
        )
        
        result = await handler.install_missing_dependencies_enhanced(compiler_output)
        
        if result.success:
            installed_deps = result.details.get("installed_dependencies", []) if result.details else []
            dep_list = [f"{d.get('group_id')}:{d.get('artifact_id')}" for d in installed_deps]
            logger.info(f"[StepExecutor] ✅ Dependencies installed successfully: {dep_list}")
            return {
                "success": True,
                "action": "retry",
                "message": f"Dependencies installed: {dep_list}",
                "should_continue": True,
                "strategy": "install_dependencies",
                "installed_packages": installed_deps,
                "analysis": result.details.get("analysis", "") if result.details else "",
                "confidence": result.details.get("confidence", 0.0) if result.details else 0.0
            }
        else:
            if missing_packages:
                suggestions = handler.suggest_pom_additions(missing_packages)
            else:
                suggestions = []
            logger.warning(f"[StepExecutor] ❌ Failed to install dependencies: {result.error_message}")
            
            if suggestions:
                suggestion_text = "\n".join(suggestions)
                logger.info(f"[StepExecutor] 💡 Suggested pom.xml additions:\n{suggestion_text}")
            
            return {
                "success": False,
                "action": "escalate",
                "message": f"Failed to install dependencies. Please add the following to pom.xml:\n{suggestion_text}" if suggestions else f"Failed to install dependencies: {result.error_message}",
                "should_continue": False,
                "strategy": "install_dependencies",
                "suggested_pom_additions": suggestions,
                "missing_packages": missing_packages
            }
    
    async def _recover_from_environment_issue(
        self,
        error: Exception,
        context: Dict[str, Any],
        error_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """从环境问题中恢复
        
        Args:
            error: 错误对象
            context: 错误上下文
            error_info: 详细错误信息
            
        Returns:
            恢复结果
        """
        from pyutagent.core.error_classification import ErrorSubCategory
        
        logger.info("[StepExecutor] 🔧 Starting environment recovery...")
        
        sub_category_str = error_info.get("sub_category", "")
        try:
            sub_category = ErrorSubCategory[sub_category_str] if sub_category_str else ErrorSubCategory.UNKNOWN
        except KeyError:
            sub_category = ErrorSubCategory.UNKNOWN
        
        if sub_category in (ErrorSubCategory.MISSING_DEPENDENCY, ErrorSubCategory.MAVEN_DEPENDENCY_ERROR):
            return await self._recover_from_dependency_issue(error, context, error_info)
        
        return {
            "success": False,
            "action": "escalate",
            "message": "Environment issue cannot be fixed automatically. Please check your environment configuration.",
            "should_continue": False,
            "strategy": "fix_environment"
        }
    
    def _optimize_prompt(self, base_prompt: str, task_type: str) -> str:
        """Optimize prompt for the configured model.
        
        Args:
            base_prompt: Original prompt
            task_type: Type of task (test_generation, error_fix, etc.)
            
        Returns:
            Optimized prompt
        """
        try:
            if self.agent_core.ab_test_id and hasattr(self.agent_core, 'prompt_optimizer'):
                variant_id, prompt = self.agent_core.prompt_optimizer.get_prompt_for_test(
                    self.agent_core.ab_test_id,
                    class_name=self.agent_core.target_class_info.get('name', 'Unknown') if self.agent_core.target_class_info else 'Unknown',
                    package=self.agent_core.target_class_info.get('package', '') if self.agent_core.target_class_info else '',
                    methods=', '.join([m.get('name', '') for m in self.agent_core.target_class_info.get('methods', [])]) if self.agent_core.target_class_info else '',
                    source_code=self.agent_core.target_class_info.get('source', '') if self.agent_core.target_class_info else ''
                )
                
                if variant_id and prompt:
                    self.agent_core.current_ab_variant_id = variant_id
                    logger.debug(f"[StepExecutor] Using A/B test variant: {variant_id}")
                    return prompt
            
            if hasattr(self.agent_core, 'prompt_optimizer'):
                optimized = self.agent_core.prompt_optimizer.optimize_for_model(
                    base_prompt=base_prompt,
                    model_name=self.agent_core.model_name,
                    task_type=task_type
                )
                logger.debug(f"[StepExecutor] Prompt optimized for {self.agent_core.model_name}")
                return optimized
            
            from pyutagent.agent.prompt_optimizer import optimize_prompt
            return optimize_prompt(base_prompt, self.agent_core.model_name, task_type)
            
        except Exception as e:
            logger.warning(f"[StepExecutor] Prompt optimization failed: {e}, using original prompt")
            return base_prompt
    
    def _record_generation_result(self, success: bool, response_time_ms: int = 0):
        """Record generation result for A/B testing.
        
        Args:
            success: Whether generation was successful
            response_time_ms: Response time in milliseconds
        """
        if not self.agent_core.ab_test_id or not hasattr(self.agent_core, 'current_ab_variant_id'):
            return
        
        try:
            if hasattr(self.agent_core, 'prompt_optimizer'):
                self.agent_core.prompt_optimizer.record_ab_test_result(
                    test_id=self.agent_core.ab_test_id,
                    variant_id=self.agent_core.current_ab_variant_id,
                    success=success,
                    response_time_ms=response_time_ms
                )
                logger.debug(f"[StepExecutor] Recorded A/B test result: {success}")
        except Exception as e:
            logger.warning(f"[StepExecutor] Failed to record A/B test result: {e}")
