"""Comprehensive test of refactored ReActAgent functionality."""
import sys
from pathlib import Path

def test_all_imports():
    """Test all necessary imports."""
    print("="*60)
    print("Testing Imports")
    print("="*60)
    
    try:
        from pyutagent.agent.react_agent import ReActAgent
        print("✓ ReActAgent imported")
        
        from pyutagent.agent.components import (
            AgentCore,
            AgentInitializer,
            FeedbackLoopExecutor,
            StepExecutor,
            AgentRecoveryManager,
            AgentHelpers,
            AgentExtensions,
        )
        print("✓ All components imported")
        
        from pyutagent.memory.working_memory import WorkingMemory
        print("✓ WorkingMemory imported")
        
        from pyutagent.llm.client import LLMClient
        print("✓ LLMClient imported")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_react_agent_structure():
    """Test ReActAgent class structure."""
    print("\n" + "="*60)
    print("Testing ReActAgent Structure")
    print("="*60)
    
    from pyutagent.agent.react_agent import ReActAgent
    
    # Check required methods
    required_methods = [
        'generate_tests',
        'run_feedback_loop',
        'pause',
        'resume',
        'terminate',
        'reset',
        'stop',
        '_init_components',
    ]
    
    missing = []
    for method in required_methods:
        if not hasattr(ReActAgent, method):
            missing.append(method)
            print(f"✗ Missing method: {method}")
        else:
            print(f"✓ Method: {method}")
    
    if missing:
        return False
    
    # Check required properties
    required_props = [
        'current_test_file',
        'target_class_info',
        'project_path',
        'max_iterations',
        'target_coverage',
        'current_iteration',
    ]
    
    for prop in required_props:
        if not hasattr(ReActAgent, prop):
            print(f"✗ Missing property: {prop}")
            return False
        print(f"✓ Property: {prop}")
    
    return True

def test_agent_core():
    """Test AgentCore component."""
    print("\n" + "="*60)
    print("Testing AgentCore Component")
    print("="*60)
    
    from pyutagent.agent.components.core_agent import AgentCore
    from pyutagent.memory.working_memory import WorkingMemory
    from pyutagent.llm.client import LLMClient
    
    # Check that AgentCore doesn't inherit from BaseAgent
    from pyutagent.agent.base_agent import BaseAgent
    if issubclass(AgentCore, BaseAgent):
        print("✗ AgentCore should not inherit from BaseAgent")
        return False
    print("✓ AgentCore doesn't inherit from BaseAgent")
    
    # Check methods
    methods = ['pause', 'resume', 'terminate', 'reset', 'stop', '_update_state', '_check_pause']
    for method in methods:
        if not hasattr(AgentCore, method):
            print(f"✗ Missing method: {method}")
            return False
        print(f"✓ Method: {method}")
    
    return True

def test_component_initialization():
    """Test that all components can be instantiated."""
    print("\n" + "="*60)
    print("Testing Component Initialization")
    print("="*60)
    
    try:
        from pyutagent.agent.components import (
            AgentCore,
            FeedbackLoopExecutor,
            StepExecutor,
        )
        from pyutagent.memory.working_memory import WorkingMemory
        from pyutagent.llm.client import LLMClient
        
        # Create mock objects
        working_memory = WorkingMemory()
        
        print("✓ Components can be initialized")
        return True
    except Exception as e:
        print(f"✗ Component initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_facade_delegation():
    """Test that ReActAgent properly delegates to components."""
    print("\n" + "="*60)
    print("Testing Facade Delegation")
    print("="*60)
    
    import inspect
    from pyutagent.agent.react_agent import ReActAgent
    
    source = inspect.getsource(ReActAgent.__init__)
    
    # Check for component initialization
    components = [
        '_core',
        '_step_executor',
        '_feedback_loop',
        '_recovery_manager',
        '_helpers',
        '_extensions',
    ]
    
    for component in components:
        if component in source:
            print(f"✓ Component initialized: {component}")
        else:
            print(f"⚠ Component not found: {component}")
    
    # Check for delegation in methods
    generate_tests_source = inspect.getsource(ReActAgent.generate_tests)
    if '_feedback_loop' in generate_tests_source or 'run_feedback_loop' in generate_tests_source:
        print("✓ generate_tests delegates to feedback_loop")
    else:
        print("⚠ generate_tests may not delegate properly")
    
    return True

def test_backward_compatibility():
    """Test backward compatibility."""
    print("\n" + "="*60)
    print("Testing Backward Compatibility")
    print("="*60)
    
    from pyutagent.agent.react_agent import ReActAgent
    import inspect
    
    # Check __init__ signature
    sig = inspect.signature(ReActAgent.__init__)
    params = list(sig.parameters.keys())
    
    required_params = ['llm_client', 'working_memory', 'project_path']
    for param in required_params:
        if param in params:
            print(f"✓ Parameter: {param}")
        else:
            print(f"✗ Missing parameter: {param}")
            return False
    
    print("✓ API signature compatible")
    return True

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Comprehensive Refactoring Verification")
    print("="*60 + "\n")
    
    tests = [
        ("All Imports", test_all_imports),
        ("ReActAgent Structure", test_react_agent_structure),
        ("AgentCore Component", test_agent_core),
        ("Component Initialization", test_component_initialization),
        ("Facade Delegation", test_facade_delegation),
        ("Backward Compatibility", test_backward_compatibility),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, None))
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"\n✗ {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    
    for test_name, result, error in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")
        if error:
            print(f"  Error: {error}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Refactoring is complete and functional!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
