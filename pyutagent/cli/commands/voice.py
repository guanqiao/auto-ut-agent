"""Voice interaction CLI commands."""

import asyncio
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel

console = Console()


@click.group(name='voice')
def voice_group():
    """Voice interaction commands for speech input/output.
    
    Provides voice input and text-to-speech capabilities.
    """
    pass


@voice_group.command(name='listen')
@click.option('--timeout', type=int, default=10, help='Listening timeout in seconds')
def voice_listen(timeout: int):
    """Listen for voice input and display recognized text."""
    from pyutagent.agent.voice import VoiceInputHandler, VoiceConfig, VoiceProvider

    console.print("[cyan]Initializing voice input...[/cyan]")

    config = VoiceConfig(provider=VoiceProvider.LOCAL)
    handler = VoiceInputHandler(config=config)

    async def listen():
        await handler.initialize()
        console.print(f"[green]Listening for up to {timeout} seconds...[/green]")
        console.print("[yellow]Speak now (press Enter to stop early)...[/yellow]")

        await handler.start_listening()

        try:
            await asyncio.wait_for(asyncio.sleep(timeout), timeout=timeout + 1)
        except asyncio.TimeoutError:
            pass

        text = await handler.stop_listening()

        if text:
            console.print(Panel(
                f"[green]Recognized:[/green]\n\n{text}",
                title="Voice Input Result"
            ))
        else:
            console.print("[yellow]No text recognized[/yellow]")

    try:
        asyncio.run(listen())
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@voice_group.command(name='speak')
@click.argument('text')
@click.option('--provider', type=click.Choice(['gtts', 'edge_tts', 'pyttsx3']), default='gtts', help='TTS provider')
def voice_speak(text: str, provider: str):
    """Speak the given text using TTS."""
    from pyutagent.agent.voice import VoiceOutputHandler, VoiceConfig, TTSProvider

    console.print(f"[cyan]Speaking: {text}[/cyan]")

    tts_provider = TTSProvider[provider.upper()]

    config = VoiceConfig(tts_provider=tts_provider)
    handler = VoiceOutputHandler(config=config)

    async def speak():
        await handler.initialize()
        await handler.speak(text)

    try:
        asyncio.run(speak())
        console.print("[green]Speech completed[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@voice_group.command(name='config')
@click.option('--provider', type=click.Choice(['whisper', 'local']), default='local', help='Speech recognition provider')
@click.option('--tts', type=click.Choice(['gtts', 'edge_tts', 'pyttsx3']), default='gtts', help='TTS provider')
@click.option('--language', type=str, default='zh-CN', help='Language code')
def voice_config(provider: str, tts: str, language: str):
    """Show or set voice configuration."""
    from pyutagent.agent.voice import VoiceConfig, VoiceProvider, TTSProvider

    config = VoiceConfig(
        provider=VoiceProvider[provider.upper()],
        tts_provider=TTSProvider[tts.upper()],
        language=language
    )

    console.print(Panel(
        f"[bold]Voice Configuration[/bold]\n\n"
        f"Speech Recognition Provider: [cyan]{config.provider.value}[/cyan]\n"
        f"TTS Provider: [cyan]{config.tts_provider.value}[/cyan]\n"
        f"Language: [cyan]{config.language}[/cyan]",
        title="Voice Settings"
    ))


@voice_group.command(name='test')
def voice_test():
    """Test voice input and output capabilities."""
    from pyutagent.agent.voice import VoiceInputHandler, VoiceOutputHandler, VoiceConfig, VoiceProvider, TTSProvider

    console.print("[cyan]Testing voice capabilities...[/cyan]\n")

    input_config = VoiceConfig(provider=VoiceProvider.LOCAL)
    output_config = VoiceConfig(tts_provider=TTSProvider.GTTS)

    input_handler = VoiceInputHandler(config=input_config)
    output_handler = VoiceOutputHandler(config=output_config)

    async def test():
        console.print("[yellow]1. Testing TTS output...[/yellow]")
        try:
            await output_handler.initialize()
            await output_handler.speak("Voice test successful")
            console.print("[green]   TTS output: OK[/green]")
        except Exception as e:
            console.print(f"[red]   TTS output: FAILED - {e}[/red]")

        console.print("\n[yellow]2. Testing speech recognition...[/yellow]")
        try:
            await input_handler.initialize()
            console.print("[green]   Speech recognition: OK (ready to listen)[/green]")
        except Exception as e:
            console.print(f"[red]   Speech recognition: FAILED - {e}[/red]")

    asyncio.run(test())

    console.print("\n[green]Voice test completed[/green]")
