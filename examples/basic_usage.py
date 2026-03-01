"""Basic usage example for PyUT Agent.

This example demonstrates how to use PyUT Agent to generate
unit tests for a Java class.
"""

import asyncio
from pathlib import Path

from pyutagent.agent.react_agent import ReActAgent
from pyutagent.llm.client import LLMClient
from pyutagent.core.config import LLMConfig, LLMProvider


async def generate_tests_example():
    """Example: Generate tests for a Java class."""
    
    # Configure LLM
    llm_config = LLMConfig(
        name="OpenAI GPT-4",
        provider=LLMProvider.OPENAI,
        model="gpt-4",
        api_key="your-api-key-here",  # Replace with your API key
    )
    
    # Create LLM client
    llm_client = LLMClient.from_config(llm_config)
    
    # Initialize agent
    agent = ReActAgent(
        project_path="/path/to/your/maven/project",
        llm_client=llm_client
    )
    
    # Generate tests for a Java class
    result = await agent.run_feedback_loop(
        target_file="src/main/java/com/example/UserService.java"
    )
    
    # Check result
    if result.success:
        print(f"✅ Tests generated successfully!")
        print(f"   Coverage: {result.coverage:.1%}")
        print(f"   Test file: {result.test_file}")
    else:
        print(f"❌ Failed to generate tests: {result.message}")
    
    return result


async def batch_generation_example():
    """Example: Generate tests for multiple files."""
    from pyutagent.services.batch_generator import BatchGenerator
    
    # Configure LLM
    llm_config = LLMConfig(
        name="OpenAI GPT-4",
        provider=LLMProvider.OPENAI,
        model="gpt-4",
        api_key="your-api-key-here",
    )
    
    # Create batch generator
    generator = BatchGenerator(
        project_path="/path/to/your/maven/project",
        llm_config=llm_config
    )
    
    # Generate tests for all Java files
    result = await generator.generate_all(
        target_files=[
            "src/main/java/com/example/UserService.java",
            "src/main/java/com/example/OrderService.java",
            "src/main/java/com/example/ProductService.java",
        ]
    )
    
    # Print summary
    print(f"\n📊 Batch Generation Summary:")
    print(f"   Total files: {result.total}")
    print(f"   Successful: {result.successful}")
    print(f"   Failed: {result.failed}")
    print(f"   Success rate: {result.success_rate:.1%}")
    
    return result


async def custom_config_example():
    """Example: Use custom configuration."""
    from pyutagent.core.config import get_settings
    
    # Get settings
    settings = get_settings()
    
    # Customize settings
    settings.generation.target_coverage = 0.85  # 85% coverage target
    settings.generation.max_iterations = 15
    settings.generation.use_aider = True
    
    # Use settings with agent
    llm_config = LLMConfig(
        name="Custom Config",
        provider=LLMProvider.OPENAI,
        model="gpt-4",
        api_key="your-api-key",
    )
    
    agent = ReActAgent(
        project_path="/path/to/project",
        llm_client=LLMClient.from_config(llm_config)
    )
    
    result = await agent.run_feedback_loop(
        target_file="src/main/java/com/example/Service.java"
    )
    
    return result


async def pause_resume_example():
    """Example: Pause and resume generation."""
    from pyutagent.agent.test_generator import TestGeneratorAgent
    
    # Create generator
    generator = TestGeneratorAgent(
        project_path="/path/to/project",
        llm_config=LLMConfig(
            name="OpenAI",
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key="your-api-key",
        )
    )
    
    # Start generation in background
    task = asyncio.create_task(
        generator.generate_tests("src/main/java/com/example/Service.java")
    )
    
    # Wait a bit then pause
    await asyncio.sleep(5)
    print("⏸️ Pausing generation...")
    generator.pause()
    
    # Wait a bit then resume
    await asyncio.sleep(2)
    print("▶️ Resuming generation...")
    generator.resume()
    
    # Wait for completion
    result = await task
    return result


async def progress_callback_example():
    """Example: Monitor progress with callbacks."""
    from pyutagent.agent.base_agent import AgentState
    
    def on_progress(state: AgentState, message: str):
        """Callback for progress updates."""
        state_icons = {
            AgentState.IDLE: "⏳",
            AgentState.PARSING: "📄",
            AgentState.GENERATING: "✨",
            AgentState.COMPILING: "🔨",
            AgentState.TESTING: "🧪",
            AgentState.FIXING: "🔧",
            AgentState.ANALYZING: "📊",
            AgentState.COMPLETED: "✅",
            AgentState.FAILED: "❌",
        }
        icon = state_icons.get(state, "🔄")
        print(f"{icon} [{state.name}] {message}")
    
    # Create agent with progress callback
    agent = ReActAgent(
        project_path="/path/to/project",
        llm_client=LLMClient.from_config(
            LLMConfig(
                name="OpenAI",
                provider=LLMProvider.OPENAI,
                model="gpt-4",
                api_key="your-api-key",
            )
        ),
        progress_callback=on_progress
    )
    
    # Run with progress monitoring
    result = await agent.run_feedback_loop(
        target_file="src/main/java/com/example/Service.java"
    )
    
    return result


if __name__ == "__main__":
    # Run examples
    print("=" * 60)
    print("PyUT Agent - Basic Usage Examples")
    print("=" * 60)
    
    # Example 1: Basic generation
    print("\n1. Basic Test Generation")
    print("-" * 40)
    # asyncio.run(generate_tests_example())  # Uncomment to run
    
    # Example 2: Batch generation
    print("\n2. Batch Test Generation")
    print("-" * 40)
    # asyncio.run(batch_generation_example())  # Uncomment to run
    
    # Example 3: Custom config
    print("\n3. Custom Configuration")
    print("-" * 40)
    # asyncio.run(custom_config_example())  # Uncomment to run
    
    # Example 4: Pause/Resume
    print("\n4. Pause and Resume")
    print("-" * 40)
    # asyncio.run(pause_resume_example())  # Uncomment to run
    
    # Example 5: Progress callback
    print("\n5. Progress Monitoring")
    print("-" * 40)
    # asyncio.run(progress_callback_example())  # Uncomment to run
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print("\nTo run an example, uncomment the corresponding asyncio.run() line.")
