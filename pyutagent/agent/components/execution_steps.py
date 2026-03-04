"""Step Executor - Individual execution steps for the feedback loop."""

import logging
import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pyutagent.agent.base_agent import StepResult
from pyutagent.core.protocols import AgentState
from pyutagent.core.config import get_settings
from pyutagent.core.retry_config import RetryConfig, DEFAULT_RETRY_CONFIG
from pyutagent.core.error_classification import get_error_classification_service

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
        self.retry_config = retry_config or DEFAULT_RETRY_CONFIG
        self.error_classifier = get_error_classification_service()
        
        logger.debug("[StepExecutor] Initialized")
    
    async def execute_with_recovery(
        self,
        operation,
        *args,
        step_name: str = "operation",
        **kwargs
    ) -> StepResult:
        """Execute an operation with automatic error recovery.
        
        Now includes unified maximum attempt limits to prevent infinite loops.
        
        Args:
            operation: The operation to execute
            *args: Positional arguments
            step_name: Name of the step for logging
            **kwargs: Keyword arguments
            
        Returns:
            StepResult
        """
        attempt = 0
        max_attempts = self.retry_config.get_max_attempts(step_name)
        
        logger.info(f"[StepExecutor] Starting step execution - Step: {step_name}, MaxAttempts: {max_attempts}")
        
        while not self.agent_core._stop_requested and not self.agent_core._terminated:
            attempt += 1
            
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
                        {"step": step_name, "attempt": attempt, "result": result}
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
                        logger.info("[StepExecutor] Resetting and regenerating")
                        return await self.execute_with_recovery(
                            self.generate_initial_tests,
                            step_name="regenerating tests"
                        )
                    
                    delay = self.retry_config.get_delay(attempt)
                    if delay > 0:
                        logger.debug(f"[StepExecutor] Waiting {delay:.1f}s before retry")
                        await asyncio.sleep(delay)
                    
                    continue
                    
            except Exception as e:
                logger.exception(f"[StepExecutor] Step execution exception - Step: {step_name}, Attempt: {attempt}, Error: {e}")
                
                recovery_result = await self._try_recover(
                    e,
                    {"step": step_name, "attempt": attempt}
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
                        
                        try:
                            # Wait with periodic termination checks
                            streaming_result = None
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
                        except asyncio.TimeoutError:
                            logger.warning(f"[StepExecutor] ⏰ Streaming generation timeout ({streaming_timeout/60:.0f} minutes), falling back to normal generation")
                            self.agent_core._update_state(AgentState.GENERATING, f"⏰ 流式生成超时 (>{streaming_timeout/60:.0f}分钟),切换到普通模式...")
                            streaming_result = None
                        
                        if streaming_result and streaming_result.success:
                            test_code = self.agent_core._extract_java_code(streaming_result.complete_code)
                            self.agent_core._update_state(AgentState.GENERATING, f"✅ 流式生成完成 - {len(test_code)} 字符")
                            logger.info(f"[StepExecutor] Streaming generation complete - "
                                       f"Tokens: {streaming_result.total_tokens}, "
                                       f"Time: {streaming_result.total_time:.2f}s")
                        else:
                            if streaming_result:
                                logger.warning(f"[StepExecutor] Streaming generation failed: {streaming_result.state}")
                            self.agent_core._update_state(AgentState.GENERATING, "⚠️ 切换到普通生成模式...")
                            response = await self.agent_core.llm_client.agenerate(prompt)
                            test_code = self.agent_core._extract_java_code(response)
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

            maven_process = await asyncio.create_subprocess_exec(
                "mvn", "dependency:build-classpath", "-Dmdep.outputFile=cp.txt", "-q",
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

            compile_process = await asyncio.create_subprocess_exec(
                "javac", "-cp", classpath,
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
    
    async def compile_with_recovery(self) -> bool:
        """Compile tests with automatic error recovery.
        
        Returns:
            True if compilation successful
        """
        if self.agent_core.working_memory.skip_verification:
            logger.info("[StepExecutor] Skipping compilation - defer_compilation mode enabled")
            self.agent_core._update_state(AgentState.GENERATING, "⏭️ 跳过编译检查 (批量生成模式)")
            return True
        
        attempt = 0
        
        logger.info(f"[StepExecutor] 🔨 Starting test compilation (with recovery) - Max attempts: {self.agent_core.max_compilation_attempts}")
        
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
                        {"step": "compilation", "attempt": attempt, "compiler_output": "\n".join(errors)}
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
                        self.agent_core._update_state(AgentState.FIXING, "🔄 Resetting and regenerating...")
                        logger.info("[StepExecutor] 🔄 Resetting and regenerating tests...")
                        reset_result = await self.execute_with_recovery(
                            self.generate_initial_tests,
                            step_name="regenerating after compilation failure"
                        )
                        if not reset_result.success:
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
                    {"step": "compilation", "attempt": attempt}
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
    
    async def run_tests_with_recovery(self) -> bool:
        """Run tests with automatic error recovery and partial success handling.
        
        Returns:
            True if tests pass
        """
        if self.agent_core.working_memory.skip_verification:
            logger.info("[StepExecutor] Skipping test execution - defer_compilation mode enabled")
            self.agent_core._update_state(AgentState.GENERATING, "⏭️ 跳过测试执行 (批量生成模式)")
            return True
        
        attempt = 0
        
        logger.info(f"[StepExecutor] 🧪 Starting test execution (with recovery and partial success handling) - Max attempts: {self.agent_core.max_test_attempts}")
        
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
                        {"step": "test_execution", "attempt": attempt, "failures": failures}
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
                        self.agent_core._update_state(AgentState.FIXING, "🔄 Resetting and regenerating...")
                        logger.info("[StepExecutor] 🔄 Resetting and regenerating tests...")
                        reset_result = await self.execute_with_recovery(
                            self.generate_initial_tests,
                            step_name="regenerating after test failure"
                        )
                        if not reset_result.success:
                            return False
                    
                    continue
                    
            except Exception as e:
                logger.exception(f"[StepExecutor] ❌ Test execution exception: {e}")
                self.agent_core._update_state(AgentState.FIXING, f"❌ Test execution error: {str(e)}")
                
                recovery_result = await self._try_recover(
                    e,
                    {"step": "test_execution", "attempt": attempt}
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
                    "report": report
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
                logger.warning("[StepExecutor] Failed to parse coverage report - report not found or invalid")
                return StepResult(
                    success=False,
                    state=AgentState.FAILED,
                    message="Failed to parse coverage report - please ensure JaCoCo is configured and tests have run"
                )
        except Exception as e:
            logger.exception(f"[StepExecutor] Coverage analysis exception: {e}")
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Error analyzing coverage: {str(e)}"
            )
    
    async def generate_additional_tests(self, coverage_data: Dict[str, Any]) -> StepResult:
        """Generate additional tests for uncovered code.
        
        Args:
            coverage_data: Coverage analysis data
            
        Returns:
            StepResult with generation results
        """
        logger.info("[StepExecutor] Generating additional tests with P0 enhancements")
        
        try:
            report = coverage_data.get("report")
            uncovered_info = self._get_uncovered_info(report)
            
            logger.debug(f"[StepExecutor] Uncovered info - Lines: {len(uncovered_info.get('lines', []))}")
            
            test_file_path = Path(self.agent_core.project_path) / self.agent_core.current_test_file
            with open(test_file_path, 'r', encoding='utf-8') as f:
                current_test_code = f.read()
            
            prompt = self.components["prompt_builder"].build_additional_tests_prompt(
                class_info=self.agent_core.target_class_info,
                existing_tests=current_test_code,
                uncovered_info=uncovered_info,
                current_coverage=coverage_data.get("line_coverage", 0.0)
            )
            
            logger.debug(f"[StepExecutor] Additional tests prompt - Length: {len(prompt)}")
            
            response = await self.agent_core.llm_client.agenerate(prompt)
            additional_tests = self.agent_core._extract_java_code(response)
            
            logger.debug(f"[StepExecutor] Extracted additional test code - Length: {len(additional_tests)}")
            
            self._append_tests_to_file(test_file_path, additional_tests)
            
            logger.info("[StepExecutor] Additional test generation complete")
            
            return StepResult(
                success=True,
                state=AgentState.OPTIMIZING,
                message="Generated additional tests for uncovered code",
                data={"additional_tests": additional_tests}
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
    
    def _append_tests_to_file(self, test_file_path: Path, additional_tests: str) -> bool:
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
                
                edit_result = self.components["smart_editor"].apply_search_replace(
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
        
        error_category = self.error_classifier.classify(error, context)
        
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
