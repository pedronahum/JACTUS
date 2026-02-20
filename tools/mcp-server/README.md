# JACTUS MCP Server

**Model Context Protocol (MCP) server for JACTUS** - Enables AI assistants like Claude Code to directly access JACTUS capabilities, providing accurate code generation, validation, and discovery without hallucination.

## What is MCP?

[Model Context Protocol (MCP)](https://modelcontextprotocol.io) is a protocol created by Anthropic that allows AI assistants to interact with external tools and data sources. Think of it as giving Claude "superpowers" specific to JACTUS.

## Features

The JACTUS MCP server provides these capabilities to AI assistants:

### 1. **Contract Discovery**
- List all 18 implemented ACTUS contract types
- Get detailed information about each contract
- Complete schema coverage for all 18 types with required/optional fields
- Understand contract categories (principal, derivative, exotic)

### 2. **Contract Simulation**
- Simulate any ACTUS contract type end-to-end
- Get structured cash flow events with payoffs and timing
- Support for constant and dict-based risk factor observers
- Summary statistics: total inflows, outflows, net cashflow

### 3. **Schema Introspection**
- Get required and optional parameters for any contract type
- Understand field types and validation rules
- See example usage for each contract

### 4. **Example Retrieval & Execution**
- Access all Python examples and Jupyter notebooks
- Get working, tested code examples
- Execute examples and see their output

### 5. **Validation**
- Validate contract attributes before creating contracts
- Get clear error messages for invalid parameters
- Receive warnings for common issues

### 6. **Documentation Search**
- Search across all JACTUS documentation
- Browse documentation structure and section headers
- Get topic-specific guides (contracts, JAX, events, attributes)

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

### With uv (recommended)

```bash
cd /path/to/JACTUS
uv pip install .
cd tools/mcp-server
uv pip install .
```

### For Development

```bash
cd tools/mcp-server
pip install -e ".[dev]"
```

## Configuration

### Claude Code Auto-Discovery

The JACTUS project root includes a `.mcp.json` file for automatic discovery. When you open the JACTUS project in Claude Code, the MCP server is available automatically.

### Claude Desktop

Add to your Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "jactus": {
      "command": "python",
      "args": ["-m", "jactus_mcp"],
      "cwd": "/path/to/JACTUS"
    }
  }
}
```

### Transport Options

The server supports two transport modes:

```bash
# Default: stdio transport (for local clients)
python -m jactus_mcp

# Streamable HTTP transport (for remote clients)
python -m jactus_mcp --transport streamable-http
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

The MCP server provides 15 tools to AI assistants:

### Contract Tools

| Tool | Description |
|------|-------------|
| `jactus_list_contracts` | List all 18 contract types by category |
| `jactus_get_contract_info` | Get detailed info about a contract type |
| `jactus_get_contract_schema` | Get required/optional parameters for all 18 types |
| `jactus_get_event_types` | List all ACTUS event types |
| `jactus_simulate_contract` | **Simulate a contract and get structured cash flows** |

### Example Tools

| Tool | Description |
|------|-------------|
| `jactus_list_examples` | List all Python scripts and notebooks |
| `jactus_get_example` | Retrieve a specific example's code |
| `jactus_get_quick_start` | Get a quick start PAM example |
| `jactus_run_example` | Execute an example and return its output |

### Validation Tools

| Tool | Description |
|------|-------------|
| `jactus_validate_attributes` | Validate contract attributes |

### Documentation Tools

| Tool | Description |
|------|-------------|
| `jactus_search_docs` | Search all JACTUS documentation |
| `jactus_get_doc_structure` | Browse documentation files and section headers |
| `jactus_get_topic_guide` | Get guides for: contracts, jax, events, attributes |

### System Tools

| Tool | Description |
|------|-------------|
| `jactus_health_check` | Verify MCP server and JACTUS installation |
| `jactus_get_version_info` | Get version information and compatibility status |

## Simulation Tool

The `jactus_simulate_contract` tool is the most powerful capability. It creates and simulates a complete ACTUS contract:

```json
{
  "attributes": {
    "contract_type": "PAM",
    "contract_id": "LOAN-001",
    "contract_role": "RPA",
    "status_date": "2024-01-01",
    "initial_exchange_date": "2024-01-15",
    "maturity_date": "2025-01-15",
    "notional_principal": 100000.0,
    "nominal_interest_rate": 0.05,
    "day_count_convention": "30E360"
  }
}
```

Returns structured events, summary statistics, and optionally contract states at each event.

You can also provide risk factors for market-data-dependent contracts:

```json
{
  "attributes": { ... },
  "risk_factors": {"LIBOR-3M": 0.05, "USD/EUR": 1.18}
}
```

## MCP Resources

The server provides both static and dynamic resources:

### Static Documentation Resources

| Resource URI | Description |
|--------------|-------------|
| `jactus://docs/architecture` | Complete system architecture guide |
| `jactus://docs/pam` | PAM contract implementation walkthrough |
| `jactus://docs/derivatives` | All 8 derivative contract types |
| `jactus://docs/readme` | Project overview and quick start |

### Dynamic Contract Resources

| Resource URI Template | Description |
|----------------------|-------------|
| `jactus://contract/{type}` | Contract info + schema for any type (e.g., `jactus://contract/PAM`) |

## MCP Prompts

Pre-defined prompt templates for common tasks:

| Prompt | Description | Arguments |
|--------|-------------|-----------|
| `create_contract` | Guide to create a new contract | `contract_type` |
| `troubleshoot_error` | Help troubleshoot errors | `error_message` |
| `understand_contract` | Explain how a contract works | `contract_type` |
| `compare_contracts` | Compare two contract types | `contract_type_1`, `contract_type_2` |

## Usage Examples

### Example 1: Simulating a Contract

**User**: "Simulate a $100k PAM loan at 5% interest"

**Claude** (using MCP):
1. Calls `jactus_simulate_contract` with attributes
2. Returns structured events: IED ($-100,000), IP ($2,500), MD ($102,500)
3. Summarizes: "The loan generates $5,000 in total interest over 1 year"

### Example 2: Discovering Contracts

**User**: "What derivative contracts does JACTUS support?"

**Claude** (using MCP):
1. Calls `jactus_list_contracts`
2. Filters for derivative category
3. Responds: "JACTUS supports 8 derivative contracts: FXOUT, OPTNS, FUTUR, SWPPV, SWAPS, CAPFL, CEG, CEC"

### Example 3: Creating a Contract

**User**: "Help me create an interest rate swap"

**Claude** (using MCP):
1. Calls `jactus_get_contract_info("SWPPV")`
2. Calls `jactus_get_contract_schema("SWPPV")`
3. Calls `jactus_get_example("interest_rate_swap_example")`
4. Provides accurate code with correct parameters

### Example 4: Validating Parameters

**User**: "Is this OPTNS contract valid?" [pastes code]

**Claude** (using MCP):
1. Calls `jactus_validate_attributes({...})`
2. Returns: "Missing required field: contract_structure"
3. Explains what's needed

## Development

### Running Tests

```bash
cd tools/mcp-server

# Unit tests (tools modules)
pytest tests/ -v --ignore=tests/test_mcp_integration.py

# Protocol-level integration tests
pytest tests/test_mcp_integration.py -v

# All tests
pytest tests/ -v
```

### Adding New Tools

With FastMCP, adding a new tool is simple:

1. Create tool function in `src/jactus_mcp/tools/`
2. Register it in `server.py` with `@mcp.tool()`
3. Add tests in `tests/`

```python
# In tools/my_tool.py
def my_new_tool(param: str) -> dict:
    """Do something useful."""
    return {"result": f"Processed {param}"}

# In server.py
@mcp.tool()
@log_tool_call
def jactus_my_tool(param: str) -> dict[str, Any]:
    """My new tool description."""
    return my_tool.my_new_tool(param)
```

FastMCP automatically generates the JSON schema from type annotations.

## Architecture

```
jactus-mcp/
├── src/jactus_mcp/
│   ├── server.py              # FastMCP server (tools, resources, prompts)
│   ├── models.py              # Pydantic response models
│   ├── tools/
│   │   ├── _utils.py          # Shared utilities (get_jactus_root)
│   │   ├── contracts.py       # Contract discovery & schema (18 types)
│   │   ├── simulate.py        # Contract simulation
│   │   ├── examples.py        # Example retrieval & execution
│   │   ├── validation.py      # Attribute validation
│   │   ├── documentation.py   # Documentation search & guides
│   │   └── system.py          # Health checks & version info
│   └── __init__.py
├── tests/
│   ├── test_contracts.py      # Contract discovery tests
│   ├── test_simulate.py       # Simulation tests
│   ├── test_examples.py       # Example retrieval tests
│   ├── test_validation.py     # Validation tests
│   ├── test_documentation.py  # Documentation search tests
│   ├── test_system.py         # System diagnostic tests
│   ├── test_mcp_features.py   # Resource & prompt tests
│   └── test_mcp_integration.py # Protocol-level integration tests
├── pyproject.toml             # Package configuration (mcp>=1.0,<2.0)
└── README.md                  # This file
```

## Version Compatibility

| JACTUS Version | jactus-mcp Version | MCP SDK |
|----------------|-------------------|---------|
| 0.1.0          | 0.1.0             | >=1.0,<2.0 |

## Troubleshooting

### MCP Server Not Connecting

1. **Check Python path**: Ensure `python` command works
2. **Verify installation**: `python -c "from jactus_mcp.server import mcp; print('OK')"`
3. **Check logs**: Look for errors in Claude Desktop/VSCode console
4. **Restart**: Restart Claude Desktop or reload VSCode window

### Tools Not Working

1. **Check JACTUS installation**: `python -c "import jactus; print(jactus.__version__)"`
2. **Verify paths**: Ensure `cwd` in config points to JACTUS root
3. **Test manually**: Run server directly: `python -m jactus_mcp`

### Import Errors

```bash
# Ensure JACTUS is installed first
cd /path/to/JACTUS
pip install .

# Then install MCP server
cd tools/mcp-server
pip install .
```

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
