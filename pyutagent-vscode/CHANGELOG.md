# Change Log

All notable changes to the PyUT Agent VS Code extension will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Support for TestNG and Spock test frameworks
- Code coverage analysis integration
- Test case optimization suggestions
- Batch test generation
- Support for Python and JavaScript

## [0.1.0] - 2026-03-31

### Added
- **AI-Powered Test Generation**
  - Generate JUnit tests with AI understanding of code logic
  - Context-aware test generation considering project structure
  - Support for boundary condition testing
  
- **Enhanced Chat Panel**
  - Markdown rendering support with Marked.js
  - Real-time streaming output with typing animations
  - Code syntax highlighting
  - Modern, responsive UI design
  
- **Diff Preview Component**
  - Monaco Editor integration for side-by-side comparison
  - Accept/Reject functionality
  - Line change statistics
  - Theme-aware styling
  
- **Terminal Integration**
  - Built-in terminal for command execution
  - Real-time output display
  - Error highlighting (red text)
  - Timeout control and error handling
  
- **Configuration Management**
  - Webview-based configuration panel
  - 5 core settings (API URL, mode, timeout, max retries, auto-approve)
  - Form validation with user-friendly error messages
  - One-click reset to defaults
  
- **Testing Infrastructure**
  - Integration tests for core functionality
  - Unit tests for API client
  - Test runner configuration
  
- **Documentation**
  - Comprehensive README with usage examples
  - API documentation
  - Development guide
  - Quick reference documentation

### Changed
- Improved error handling with unified error management
- Enhanced UI/UX with modern design and smooth animations
- Optimized performance for streaming output
- Better resource management with proper cleanup

### Fixed
- Fixed connection error handling in API client
- Fixed terminal output encoding issues
- Fixed Webview message passing edge cases

### Technical Details
- **Total Lines of Code**: ~2,060 lines
- **Test Coverage**: Core modules tested
- **Dependencies**: 
  - TypeScript 5.3+
  - Monaco Editor 0.45.0
  - Marked.js (CDN)
  - Axios 1.6.2
  - React 18.2.0

### Contributors
- Initial development by PyUT Team

---

## Version History

### Version Format
- **Major**: Breaking changes
- **Minor**: New features (backwards compatible)
- **Patch**: Bug fixes (backwards compatible)

### Release Schedule
- Major releases: Quarterly
- Minor releases: Monthly
- Patch releases: As needed

---

For more information about PyUT Agent, visit our [GitHub repository](https://github.com/coding-agent/pyutagent-vscode).
