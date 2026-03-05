"""PyUT Agent CLI entry point.

Usage:
    python -m pyutagent                    # Interactive mode
    python -m pyutagent "task description" # Execute task
    python -m pyutagent --voice            # Voice mode
    python -m pyutagent --ide vscode       # IDE integration mode
    python -m pyutagent --config config.yaml # Use config file
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from pyutagent.agent import (
    SkillRegistry,
    SkillLoader,
    EnhancedSkillExecutor,
)
from pyutagent.agent.voice import create_voice_handlers, VoiceConfig
from pyutagent.agent.safety import create_default_policy
from pyutagent.agent.acp_client import create_acp_client

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PyUTAgentCLI:
    """PyUT Agent command-line interface."""

    def __init__(self, config_path: str = None):
        """Initialize CLI.

        Args:
            config_path: Optional path to config file
        """
        self.config_path = config_path
        self.skill_registry = None
        self.skill_executor = None
        self.safety_policy = None

    async def initialize(self):
        """Initialize the agent and components."""
        logger.info("Initializing PyUT Agent components...")

        self.skill_registry = SkillRegistry()
        skill_loader = SkillLoader(self.skill_registry)
        skill_loader.load_builtin_skills()

        self.skill_executor = EnhancedSkillExecutor(self.skill_registry)
        self.safety_policy = create_default_policy()

        logger.info("PyUT Agent components initialized successfully")
        logger.info(f"Loaded {len(self.skill_registry.list_skills())} skills")

    async def run_interactive(self):
        """Run in interactive mode."""
        print("=" * 50)
        print("  PyUT Agent - Interactive Mode")
        print("=" * 50)
        print("Type 'help' for commands, 'exit' to quit")
        print()

        while True:
            try:
                user_input = input("PyUT Agent> ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("Goodbye!")
                    break

                if user_input.lower() == 'help':
                    self._print_help()
                    continue

                if user_input.lower() == 'skills':
                    self._list_skills()
                    continue

                result = await self._execute_task(user_input)
                print(f"\nResult: {result}")
                print()

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                print(f"Error: {e}")

    async def _execute_task(self, task: str):
        """Execute a task using skills.

        Args:
            task: Task description

        Returns:
            Execution result
        """
        logger.info(f"Executing task: {task}")

        matching_skills = self.skill_registry.find_by_trigger(task)

        if matching_skills:
            skill_name = matching_skills[0]
            logger.info(f"Matched skill: {skill_name}")

            result = await self.skill_executor.execute_skill(
                skill_name,
                {"task": task}
            )

            return result

        return {"success": False, "message": "No matching skill found"}

    async def run_task(self, task: str):
        """Run a specific task.

        Args:
            task: Task description
        """
        result = await self._execute_task(task)

        print("\n" + "=" * 50)
        print("  Task Result")
        print("=" * 50)
        print(f"Success: {result.success if hasattr(result, 'success') else result.get('success', False)}")
        print("=" * 50)

    async def run_voice_mode(self):
        """Run in voice interaction mode."""
        print("Initializing voice mode...")

        voice_config = VoiceConfig()
        voice_input, voice_output = await create_voice_handlers(voice_config)

        print("Voice mode ready. Type your request or press Ctrl+C to exit.")

        try:
            while True:
                text = input("\nEnter your request (or 'exit' to quit): ").strip()

                if text.lower() == 'exit':
                    break

                if text:
                    result = await self._execute_task(text)
                    await voice_output.speak(f"Task completed: {result.success if hasattr(result, 'success') else False}")
                else:
                    print("Please type your request")

        except KeyboardInterrupt:
            print("\nVoice mode ended")

    async def run_ide_mode(self, ide: str):
        """Run in IDE integration mode.

        Args:
            ide: IDE name (vscode, idea)
        """
        logger.info(f"Starting IDE integration mode: {ide}")

        if ide == "vscode":
            endpoint = "ws://localhost:8080"
        elif ide == "idea":
            endpoint = "ws://localhost:9876"
        else:
            logger.error(f"Unsupported IDE: {ide}")
            return

        client = await create_acp_client(endpoint)

        if not client.is_connected:
            logger.error(f"Failed to connect to {ide}")
            print(f"Error: Could not connect to {ide}")
            print(f"Make sure PyUT Agent server is running at {endpoint}")
            return

        logger.info(f"Connected to {ide}")

        print(f"PyUT Agent connected to {ide}")
        print("IDE integration ready")
        print("Press Ctrl+C to disconnect")

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\nDisconnecting...")
            await client.disconnect()
            print("Disconnected")

    def _print_help(self):
        """Print help information."""
        help_text = """
Available commands:
  help              - Show this help message
  skills            - List available skills
  exit/quit/q       - Exit the program

Example tasks:
  - 为 UserService 生成单元测试
  - 修复编译错误
  - 分析代码质量
  - 重构这段代码
"""
        print(help_text)

    def _list_skills(self):
        """List available skills."""
        if not self.skill_registry:
            print("Agent not initialized")
            return

        skills = self.skill_registry.list_skills()

        print("\nAvailable Skills:")
        print("-" * 40)
        for skill_name in skills:
            info = self.skill_registry.get_skill_info(skill_name)
            if info:
                print(f"  - {skill_name}")
                print(f"    {info.get('description', '')}")
        print()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PyUT Agent - AI-powered Java unit test generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m pyutagent
  python -m pyutagent "为 UserService 生成单元测试"
  python -m pyutagent --voice
  python -m pyutagent --ide vscode
        """
    )

    parser.add_argument(
        "task",
        nargs="?",
        help="Task description to execute"
    )

    parser.add_argument(
        "--voice", "-v",
        action="store_true",
        help="Enable voice interaction mode"
    )

    parser.add_argument(
        "--ide",
        choices=["vscode", "idea"],
        help="Enable IDE integration mode"
    )

    parser.add_argument(
        "--config", "-c",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    cli = PyUTAgentCLI(config_path=args.config)
    await cli.initialize()

    if args.voice:
        await cli.run_voice_mode()
    elif args.ide:
        await cli.run_ide_mode(args.ide)
    elif args.task:
        await cli.run_task(args.task)
    else:
        await cli.run_interactive()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
