"""Tests for TerminationChecker."""

import pytest
import time
from pyutagent.core.termination import (
    TerminationChecker,
    TerminationReason,
    TerminationState,
    create_termination_checker,
)


class TestTerminationReason:
    """Tests for TerminationReason enum."""
    
    def test_termination_reasons_exist(self):
        assert TerminationReason.MAX_ITERATIONS
        assert TerminationReason.TARGET_COVERAGE_REACHED
        assert TerminationReason.USER_STOPPED
        assert TerminationReason.USER_TERMINATED
        assert TerminationReason.MAX_ATTEMPTS_EXCEEDED
        assert TerminationReason.ERROR_THRESHOLD_EXCEEDED
        assert TerminationReason.TIMEOUT_EXCEEDED
        assert TerminationReason.NO_UNCOVERED_LINES


class TestTerminationState:
    """Tests for TerminationState."""
    
    def test_default_values(self):
        state = TerminationState()
        
        assert state.should_stop is False
        assert state.reason is None
        assert state.message == ""
        assert state.details == {}
    
    def test_with_values(self):
        state = TerminationState(
            should_stop=True,
            reason=TerminationReason.MAX_ITERATIONS,
            message="Max iterations reached",
            details={"iterations": 10}
        )
        
        assert state.should_stop is True
        assert state.reason == TerminationReason.MAX_ITERATIONS
        assert state.message == "Max iterations reached"
        assert state.details["iterations"] == 10


class TestTerminationChecker:
    """Tests for TerminationChecker class."""
    
    def test_default_values(self):
        checker = TerminationChecker()
        
        assert checker.max_iterations == 2
        assert checker.target_coverage == 0.8
        assert checker.max_total_attempts == 50
        assert checker.max_error_count == 10
        assert checker.timeout_seconds is None
    
    def test_start(self):
        checker = TerminationChecker()
        checker.start()
        
        assert checker._start_time is not None
        assert checker.elapsed_time >= 0
    
    def test_record_error(self):
        checker = TerminationChecker()
        
        checker.record_error()
        assert checker.error_count == 1
        
        checker.record_error(5)
        assert checker.error_count == 6
    
    def test_record_attempt(self):
        checker = TerminationChecker()
        
        checker.record_attempt()
        assert checker.total_attempts == 1
        
        checker.record_attempt(3)
        assert checker.total_attempts == 4
    
    def test_check_user_terminated(self):
        checker = TerminationChecker()
        
        state = checker.check(
            current_iteration=1,
            current_coverage=0.5,
            is_terminated=True
        )
        
        assert state.should_stop is True
        assert state.reason == TerminationReason.USER_TERMINATED
    
    def test_check_user_stopped(self):
        checker = TerminationChecker()
        
        state = checker.check(
            current_iteration=1,
            current_coverage=0.5,
            is_stopped=True
        )
        
        assert state.should_stop is True
        assert state.reason == TerminationReason.USER_STOPPED
    
    def test_check_max_iterations(self):
        checker = TerminationChecker(max_iterations=5)
        
        state = checker.check(
            current_iteration=6,
            current_coverage=0.5
        )
        
        assert state.should_stop is True
        assert state.reason == TerminationReason.MAX_ITERATIONS
    
    def test_check_target_coverage_reached(self):
        checker = TerminationChecker(target_coverage=0.8)
        
        state = checker.check(
            current_iteration=1,
            current_coverage=0.85
        )
        
        assert state.should_stop is True
        assert state.reason == TerminationReason.TARGET_COVERAGE_REACHED
    
    def test_check_no_uncovered_lines(self):
        checker = TerminationChecker()
        
        state = checker.check(
            current_iteration=1,
            current_coverage=0.7,
            has_uncovered_lines=False
        )
        
        assert state.should_stop is True
        assert state.reason == TerminationReason.NO_UNCOVERED_LINES
    
    def test_check_max_attempts_exceeded(self):
        checker = TerminationChecker(max_total_attempts=5)
        
        checker.record_attempt(5)
        
        state = checker.check(
            current_iteration=1,
            current_coverage=0.5
        )
        
        assert state.should_stop is True
        assert state.reason == TerminationReason.MAX_ATTEMPTS_EXCEEDED
    
    def test_check_error_threshold_exceeded(self):
        checker = TerminationChecker(max_error_count=3)
        
        checker.record_error(3)
        
        state = checker.check(
            current_iteration=1,
            current_coverage=0.5
        )
        
        assert state.should_stop is True
        assert state.reason == TerminationReason.ERROR_THRESHOLD_EXCEEDED
    
    def test_check_timeout_exceeded(self):
        checker = TerminationChecker(timeout_seconds=0.1)
        checker.start()
        
        time.sleep(0.15)
        
        state = checker.check(
            current_iteration=1,
            current_coverage=0.5
        )
        
        assert state.should_stop is True
        assert state.reason == TerminationReason.TIMEOUT_EXCEEDED
    
    def test_check_should_continue(self):
        checker = TerminationChecker()
        
        state = checker.check(
            current_iteration=1,
            current_coverage=0.5
        )
        
        assert state.should_stop is False
    
    def test_check_iteration(self):
        checker = TerminationChecker(max_iterations=5)
        
        assert checker.check_iteration(3, 0.5) is False
        assert checker.check_iteration(6, 0.5) is True
    
    def test_callback(self):
        checker = TerminationChecker(max_iterations=2)
        callback_called = []
        
        def callback(state):
            callback_called.append(state)
        
        checker.register_callback(callback)
        
        checker.check(current_iteration=3, current_coverage=0.5)
        
        assert len(callback_called) == 1
        assert callback_called[0].reason == TerminationReason.MAX_ITERATIONS
    
    def test_reset(self):
        checker = TerminationChecker()
        checker.start()
        checker.record_error(5)
        checker.record_attempt(10)
        
        checker.reset()
        
        assert checker._start_time is None
        assert checker.error_count == 0
        assert checker.total_attempts == 0
    
    def test_get_summary(self):
        checker = TerminationChecker(max_iterations=5, target_coverage=0.9)
        checker.start()
        checker.record_error(2)
        checker.record_attempt(3)
        
        summary = checker.get_summary()
        
        assert summary["max_iterations"] == 5
        assert summary["target_coverage"] == 0.9
        assert summary["current_error_count"] == 2
        assert summary["current_total_attempts"] == 3
        assert "elapsed_time" in summary


class TestCreateTerminationChecker:
    """Tests for create_termination_checker function."""
    
    def test_create_with_defaults(self):
        checker = create_termination_checker()
        
        assert checker.max_iterations == 2
        assert checker.target_coverage == 0.8
    
    def test_create_with_custom_values(self):
        checker = create_termination_checker(
            max_iterations=20,
            target_coverage=0.95,
            max_error_count=5
        )
        
        assert checker.max_iterations == 20
        assert checker.target_coverage == 0.95
        assert checker.max_error_count == 5
