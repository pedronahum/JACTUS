# JACTUS MCP Server

**Model Context Protocol (MCP) server for JACTUS** - Enables AI assistants like Claude Code to directly access JACTUS capabilities, providing accurate code generation, validation, and discovery without hallucination.

## What is MCP?

[Model Context Protocol (MCP)](https://modelcontextprotocol.io) is a protocol created by Anthropic that allows AI assistants to interact with external tools and data sources. Think of it as giving Claude "superpowers" specific to JACTUS.

## Features

The JACTUS MCP server provides these capabilities to AI assistants:

### 1. **Contract Discovery**
- List all 18 implemented ACTUS contract types
- Get detailed information about each contract
- Understand contract categories (principal, derivative, exotic)

### 2. **Schema Introspection**
- Get required and optional parameters for any contract type
- Understand field types and validation rules
- See example usage for each contract

### 3. **Example Retrieval**
- Access all Python examples and Jupyter notebooks
- Get working, tested code examples
- Never hallucinate example code

### 4. **Validation**
- Validate contract attributes before creating contracts
- Get clear error messages for invalid parameters
- Receive warnings for common issues

### 5. **Documentation Search**
- Search across all JACTUS documentation
- Find relevant sections in ARCHITECTURE.md, PAM.md, etc.
- Get topic-specific guides

### 6. **Event Types**
- List all ACTUS event types (IED, IP, MD, etc.)
- Understand event descriptions and usage

### 7. **System Diagnostics**
- Health check to verify installation and configuration
- Version information for compatibility checking
- Troubleshooting recommendations

## Installation

### Prerequisites

1. **Install JACTUS** (from parent directory):
```bash
cd /path/to/JACTUS
pip install .
```

2. **Install MCP server**:
```bash
cd tools/mcp-server
pip install .
```

### For Development

```bash
cd tools/mcp-server
pip install -e ".[dev]"
```

## Configuration

### Claude Desktop

Add to your Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "jactus": {
      "command": "python",
      "args": ["-m", "jactus_mcp.server"],
      "cwd": "/path/to/JACTUS"
    }
  }
}
```

### Claude Code (VSCode Extension)

Add to your VSCode settings (`.vscode/settings.json`):

```json
{
  "mcp.servers": {
    "jactus": {
      "command": "python",
      "args": ["-m", "jactus_mcp.server"],
      "cwd": "${workspaceFolder}"
    }
  }
}
```

### Verify Installation

After configuration, restart Claude Desktop or VSCode. You should see the JACTUS MCP server connected.

Test with Claude:
```
User: "List all JACTUS contract types"
Claude: *calls jactus_list_contracts tool*
        "JACTUS implements 18 ACTUS contract types..."
```

## Available Tools

The MCP server provides these tools to AI assistants:

### Contract Tools

| Tool | Description |
|------|-------------|
| `jactus_list_contracts` | List all 18 contract types by category |
| `jactus_get_contract_info` | Get detailed info about a contract type |
| `jactus_get_contract_schema` | Get required/optional parameters |
| `jactus_get_event_types` | List all ACTUS event types |

### Example Tools

| Tool | Description |
|------|-------------|
| `jactus_list_examples` | List all Python scripts and notebooks |
| `jactus_get_example` | Retrieve a specific example's code |

### Validation Tools

| Tool | Description |
|------|-------------|
| `jactus_validate_attributes` | Validate contract attributes |

### Documentation Tools

| Tool | Description |
|------|-------------|
| `jactus_search_docs` | Search all JACTUS documentation |

### System Tools

| Tool | Description |
|------|-------------|
| `jactus_health_check` | Verify MCP server and JACTUS installation |
| `jactus_get_version_info` | Get version information and compatibility status |

## MCP Resources

The server provides direct access to JACTUS documentation through MCP resources:

| Resource URI | Description |
|--------------|-------------|
| `jactus://docs/architecture` | Complete system architecture guide |
| `jactus://docs/pam` | PAM contract implementation walkthrough |
| `jactus://docs/derivatives` | All 8 derivative contract types |
| `jactus://docs/readme` | Project overview and quick start |

**Benefits**: AI assistants can read full documentation without summarization or token limits.

## MCP Prompts

Pre-defined prompt templates for common tasks:

| Prompt | Description | Arguments |
|--------|-------------|-----------|
| `create_contract` | Guide to create a new contract | `contract_type` |
| `troubleshoot_error` | Help troubleshoot errors | `error_message` |
| `understand_contract` | Explain how a contract works | `contract_type` |
| `compare_contracts` | Compare two contract types | `contract_type_1`, `contract_type_2` |

**Benefits**: Quick access to context-aware assistance with relevant JACTUS information pre-loaded.

## Usage Examples

### Example 1: Discovering Contracts

**User**: "What derivative contracts does JACTUS support?"

**Claude** (using MCP):
1. Calls `jactus_list_contracts`
2. Filters for derivative category
3. Responds: "JACTUS supports 8 derivative contracts: FXOUT, OPTNS, FUTUR, SWPPV, SWAPS, CAPFL, CEG, CEC"

### Example 2: Creating a Contract

**User**: "Help me create an interest rate swap"

**Claude** (using MCP):
1. Calls `jactus_get_contract_info("SWPPV")`
2. Calls `jactus_get_contract_schema("SWPPV")`
3. Calls `jactus_get_example("interest_rate_swap_example")`
4. Provides accurate code with correct parameters

### Example 3: Validating Parameters

**User**: "Is this OPTNS contract valid?" [pastes code]

**Claude** (using MCP):
1. Calls `jactus_validate_attributes({...})`
2. Returns: "Missing required field: contract_structure"
3. Explains what's needed

### Example 4: Finding Documentation

**User**: "How do day count conventions work in JACTUS?"

**Claude** (using MCP):
1. Calls `jactus_search_docs("day count convention")`
2. Returns relevant sections from ARCHITECTURE.md
3. Explains based on actual documentation

## Benefits for Developers

### Without MCP Server
```
Developer → Google/docs → Trial & error → Stack Overflow → Maybe works
Time: 2-4 hours for complex contract
```

### With MCP Server
```
Developer → Ask Claude → Claude uses MCP → Exact API → Working code
Time: 5-10 minutes for complex contract
```

### Key Advantages

1. **No Hallucination**: Claude uses actual JACTUS code/docs, not guesses
2. **Always Up-to-Date**: MCP server reads current codebase
3. **Validated Examples**: Returns only tested, working code
4. **Interactive Debugging**: Validates on the fly
5. **Discovery**: Explores actual capabilities, not assumed ones

## Development

### Running Tests

```bash
cd tools/mcp-server
pytest tests/ -v
```

**Test Results**: ✅ 34/34 tests passing (100%)

- 6 contract discovery tests
- 7 documentation search tests
- 6 example retrieval tests
- 5 validation tests
- 6 system diagnostic tests
- 4 MCP features tests (resources & prompts)

All MCP tools, resources, and prompts are fully tested and verified working.

### Adding New Tools

1. Create tool function in `src/jactus_mcp/tools/`
2. Add tool definition in `server.py` `list_tools()`
3. Add tool handler in `server.py` `call_tool()`
4. Add tests in `tests/`

Example:

```python
# In tools/my_tool.py
def my_new_tool(param: str) -> dict:
    """Do something useful."""
    return {"result": f"Processed {param}"}

# In server.py
@app.list_tools()
async def list_tools():
    return [
        # ... existing tools ...
        {
            "name": "jactus_my_tool",
            "description": "My new tool",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "param": {"type": "string", "description": "Input parameter"}
                },
                "required": ["param"],
            },
        },
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    # ... existing handlers ...
    elif name == "jactus_my_tool":
        result = my_tool.my_new_tool(arguments["param"])
```

## Troubleshooting

### MCP Server Not Connecting

1. **Check Python path**: Ensure `python` command works
2. **Verify installation**: `python -m jactus_mcp.server --help`
3. **Check logs**: Look for errors in Claude Desktop/VSCode console
4. **Restart**: Restart Claude Desktop or reload VSCode window

### Tools Not Working

1. **Check JACTUS installation**: `python -c "import jactus; print(jactus.__version__)"`
2. **Verify paths**: Ensure `cwd` in config points to JACTUS root
3. **Test manually**: Run server directly: `python -m jactus_mcp.server`

### Import Errors

```bash
# Ensure JACTUS is installed first
cd /path/to/JACTUS
pip install .

# Then install MCP server
cd tools/mcp-server
pip install .
```

## Architecture

```
jactus-mcp/
├── src/jactus_mcp/
│   ├── server.py              # Main MCP server
│   ├── tools/
│   │   ├── contracts.py       # Contract discovery & schema
│   │   ├── examples.py        # Example retrieval
│   │   ├── validation.py      # Attribute validation
│   │   └── documentation.py   # Documentation search
│   └── __init__.py
├── tests/                     # Test suite
├── pyproject.toml            # Package configuration
└── README.md                 # This file
```

## Version Compatibility

| JACTUS Version | jactus-mcp Version |
|----------------|-------------------|
| 0.1.0          | 0.1.0             |

## Contributing

Contributions to the MCP server are welcome! Please:

1. Add tests for new tools
2. Update this README with new tools/features
3. Follow existing code style (black, ruff)
4. Ensure all tests pass

## License

Apache License 2.0 - Same as JACTUS

## Links

- [JACTUS Repository](https://github.com/pedronahum/JACTUS)
- [Model Context Protocol](https://modelcontextprotocol.io)
- [Claude Desktop](https://claude.ai/download)
- [Claude Code for VSCode](https://marketplace.visualstudio.com/items?itemName=Anthropic.claude-code)

## Support

- **Issues**: [GitHub Issues](https://github.com/pedronahum/JACTUS/issues)
- **Discussions**: [GitHub Discussions](https://github.com/pedronahum/JACTUS/discussions)
- **Email**: pnrodriguezh@gmail.com

---

**Built with ❤️ for the JACTUS community**
