# PyUT Agent IntelliJ IDEA Plugin

Intelligent Java Unit Test Generation Plugin for IntelliJ IDEA.

## Features

- **AI-Powered Test Generation**: Generate comprehensive unit tests using LLM
- **Smart Code Analysis**: Analyze Java classes and suggest test cases
- **One-Click Testing**: Generate tests with a single click
- **Coverage Integration**: View test coverage directly in IDE
- **Multi-LLM Support**: Support for OpenAI, Anthropic, DeepSeek, and local models

## Requirements

- IntelliJ IDEA 2023.1 or later
- JDK 17 or later
- Python 3.9+ with PyUT Agent installed

## Installation

1. Download the plugin from JetBrains Marketplace
2. Go to Settings → Plugins → Install Plugin from Disk
3. Restart IntelliJ IDEA

## Configuration

1. Go to Settings → Tools → PyUT Agent
2. Configure your LLM provider and API key
3. Set the Python interpreter path
4. Configure test generation preferences

## Usage

### Generate Tests for Current File

1. Open a Java file
2. Right-click in the editor
3. Select "Generate Unit Tests"
4. Review and apply the generated tests

### Generate Tests for Project

1. Right-click on a package or directory
2. Select "Generate Unit Tests for Package"
3. Configure batch generation options
4. Monitor progress in the tool window

## Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Generate Tests | Ctrl+Shift+T |
| Show Chat Panel | Ctrl+Shift+P |
| Open Settings | Ctrl+Alt+S |

## License

MIT License
