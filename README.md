[![Dependabot Updates](https://github.com/navicore/gamecode-cli/actions/workflows/dependabot/dependabot-updates/badge.svg)](https://github.com/navicore/gamecode-cli/actions/workflows/dependabot/dependabot-updates)
[![Rust CI](https://github.com/navicore/gamecode-cli/actions/workflows/rust-ci.yml/badge.svg)](https://github.com/navicore/gamecode-cli/actions/workflows/rust-ci.yml)

# GameCode CLI

A CLI tool for interacting with AWS Bedrock Anthropic Claude models (Claude 3.7 Sonnet and Claude 3.5 Haiku) with agentic capabilities through the gamecode-tools library.

## Overview

GameCode CLI enables you to interact with Anthropic Claude models on AWS Bedrock with full tool-use capabilities. It allows Claude to:

- Read and write files
- Search file contents
- List directories
- Create directories
- Execute shell commands
- And more, via the gamecode-tools integration

## Installation

### Prerequisites

- Rust (2024 edition)
- AWS account with Bedrock access
- Anthropic Claude models enabled in your AWS account

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/navicore/gamecode-cli.git
   cd gamecode-cli
   ```

2. Build the project:
   ```
   cargo build --release
   ```

3. Configure AWS credentials:
   ```
   aws configure
   ```

## Usage

```bash
gamecode-cli [OPTIONS] <PROMPT>...
```

### Options

- `--model <MODEL>`: Specify the Claude model to use (default: `anthropic.claude-3-5-sonnet-20240620-v1:0`)
- `--region <REGION>`: Specify the AWS region (default: `us-east-1`)
- `-v, --verbose`: Enable verbose logging
- `-h, --help`: Show help

### Examples

Ask Claude to list files in the current directory:
```bash
gamecode-cli "List all files in the current directory"
```

Ask Claude to read and summarize a file:
```bash
gamecode-cli "Read the file src/main.rs and explain what it does"
```

Create a new project structure:
```bash
gamecode-cli "Create a basic project structure for a Rust CLI application with error handling and configuration"
```

## Tool Support

GameCode CLI supports tools through two mechanisms:

### 1. Built-in Tools (Default)
Uses the integrated gamecode-tools library for basic file and shell operations.

### 2. MCP (Model Context Protocol) Servers (Recommended)
Use external MCP servers for more flexible and extensible tool support.

#### Setting up MCP Tools

1. **Install an MCP server** (e.g., gamecode-mcp):
   ```bash
   cargo install gamecode-mcp
   ```

2. **Configure the MCP server**:
   ```bash
   gamecode mcp add gamecode gamecode-mcp
   ```

3. **Create a tools.yaml** in your working directory to define available tools

4. **Use MCP mode**:
   ```bash
   gamecode --use-mcp "Your prompt here"
   ```

#### Managing MCP Servers

- List configured servers: `gamecode mcp list`
- Add a server: `gamecode mcp add <name> <command> [args...]`
- Remove a server: `gamecode mcp remove <name>`
- Test a server: `gamecode mcp test <name>`

## Architecture

GameCode CLI uses AWS Bedrock's streaming API with the Converse protocol to maintain an interactive session with Claude. The tool pipeline works as follows:

1. User prompt is sent to Claude via AWS Bedrock
2. Claude may respond with text or request to use a tool
3. Tool requests are executed via gamecode-tools
4. Tool results are sent back to Claude
5. Claude continues processing with tool results

## Configuration

Claude models and behavior can be adjusted through command-line options or by modifying the code directly.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Feel free to submit issues and pull requests.
