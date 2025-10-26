"""JACTUS MCP Server.

This MCP (Model Context Protocol) server provides AI assistants like Claude Code
with direct access to JACTUS capabilities, enabling accurate code generation,
validation, and discovery without hallucination.
"""

import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server

from .tools import contracts, examples, validation, documentation, system

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create MCP server
app = Server("jactus-mcp")


# Contract Discovery Tools
@app.list_tools()
async def list_tools() -> list[dict[str, Any]]:
    """List all available MCP tools for JACTUS."""
    return [
        {
            "name": "jactus_list_contracts",
            "description": "List all available ACTUS contract types in JACTUS",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "jactus_get_contract_info",
            "description": "Get detailed information about a specific contract type",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "contract_type": {
                        "type": "string",
                        "description": "ACTUS contract type code (e.g., PAM, LAM, SWPPV)",
                    }
                },
                "required": ["contract_type"],
            },
        },
        {
            "name": "jactus_get_contract_schema",
            "description": "Get required and optional parameters for a contract type",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "contract_type": {
                        "type": "string",
                        "description": "ACTUS contract type code (e.g., PAM, LAM, SWPPV)",
                    }
                },
                "required": ["contract_type"],
            },
        },
        {
            "name": "jactus_list_examples",
            "description": "List all available code examples in JACTUS",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "jactus_get_example",
            "description": "Retrieve a specific code example",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "example_name": {
                        "type": "string",
                        "description": "Name of the example (e.g., pam_example, interest_rate_swap_example)",
                    }
                },
                "required": ["example_name"],
            },
        },
        {
            "name": "jactus_validate_attributes",
            "description": "Validate contract attributes for correctness",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "attributes": {
                        "type": "object",
                        "description": "Contract attributes dictionary to validate",
                    }
                },
                "required": ["attributes"],
            },
        },
        {
            "name": "jactus_search_docs",
            "description": "Search JACTUS documentation for specific topics",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'day count convention', 'state transition')",
                    }
                },
                "required": ["query"],
            },
        },
        {
            "name": "jactus_get_event_types",
            "description": "List all ACTUS event types",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "jactus_health_check",
            "description": "Verify MCP server and JACTUS are working correctly",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "jactus_get_version_info",
            "description": "Get JACTUS and MCP server version information",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[dict[str, Any]]:
    """Handle tool calls from MCP clients."""
    try:
        if name == "jactus_list_contracts":
            result = contracts.list_contracts()
        elif name == "jactus_get_contract_info":
            result = contracts.get_contract_info(arguments["contract_type"])
        elif name == "jactus_get_contract_schema":
            result = contracts.get_contract_schema(arguments["contract_type"])
        elif name == "jactus_list_examples":
            result = examples.list_examples()
        elif name == "jactus_get_example":
            result = examples.get_example(arguments["example_name"])
        elif name == "jactus_validate_attributes":
            result = validation.validate_attributes(arguments["attributes"])
        elif name == "jactus_search_docs":
            result = documentation.search_docs(arguments["query"])
        elif name == "jactus_get_event_types":
            result = contracts.get_event_types()
        elif name == "jactus_health_check":
            result = system.health_check()
        elif name == "jactus_get_version_info":
            result = system.get_version_info()
        else:
            return [{"type": "text", "text": f"Unknown tool: {name}"}]

        return [{"type": "text", "text": str(result)}]

    except Exception as e:
        logger.error(f"Error calling tool {name}: {e}", exc_info=True)
        return [{"type": "text", "text": f"Error: {str(e)}"}]


@app.list_resources()
async def list_resources() -> list[dict[str, Any]]:
    """List available documentation resources."""
    return [
        {
            "uri": "jactus://docs/architecture",
            "name": "JACTUS Architecture Guide",
            "description": "Complete system architecture, design patterns, and implementation details",
            "mimeType": "text/markdown",
        },
        {
            "uri": "jactus://docs/pam",
            "name": "PAM Contract Walkthrough",
            "description": "Deep dive into JACTUS internals using Principal at Maturity contract",
            "mimeType": "text/markdown",
        },
        {
            "uri": "jactus://docs/derivatives",
            "name": "Derivative Contracts Guide",
            "description": "Complete guide to all 8 derivative contract types",
            "mimeType": "text/markdown",
        },
        {
            "uri": "jactus://docs/readme",
            "name": "JACTUS README",
            "description": "Project overview, quick start, and installation",
            "mimeType": "text/markdown",
        },
    ]


@app.read_resource()
async def read_resource(uri: str) -> str:
    """Read a documentation resource."""
    try:
        root = documentation.get_jactus_root()

        resource_map = {
            "jactus://docs/architecture": root / "docs" / "ARCHITECTURE.md",
            "jactus://docs/pam": root / "docs" / "PAM.md",
            "jactus://docs/derivatives": root / "docs" / "derivatives.md",
            "jactus://docs/readme": root / "README.md",
        }

        if uri not in resource_map:
            return f"Unknown resource: {uri}"

        file_path = resource_map[uri]
        if not file_path.exists():
            return f"Resource file not found: {file_path}"

        return file_path.read_text()

    except Exception as e:
        logger.error(f"Error reading resource {uri}: {e}", exc_info=True)
        return f"Error reading resource: {str(e)}"


@app.list_prompts()
async def list_prompts() -> list[dict[str, Any]]:
    """List available prompt templates."""
    return [
        {
            "name": "create_contract",
            "description": "Guide to create a new JACTUS contract",
            "arguments": [
                {
                    "name": "contract_type",
                    "description": "Type of contract (PAM, LAM, ANN, SWPPV, etc.)",
                    "required": True,
                }
            ],
        },
        {
            "name": "troubleshoot_error",
            "description": "Help troubleshoot a JACTUS error",
            "arguments": [
                {
                    "name": "error_message",
                    "description": "The error message you're seeing",
                    "required": True,
                }
            ],
        },
        {
            "name": "understand_contract",
            "description": "Explain how a specific contract type works",
            "arguments": [
                {
                    "name": "contract_type",
                    "description": "Contract type to explain (PAM, SWPPV, etc.)",
                    "required": True,
                }
            ],
        },
        {
            "name": "compare_contracts",
            "description": "Compare two contract types",
            "arguments": [
                {
                    "name": "contract_type_1",
                    "description": "First contract type",
                    "required": True,
                },
                {
                    "name": "contract_type_2",
                    "description": "Second contract type",
                    "required": True,
                }
            ],
        },
    ]


@app.get_prompt()
async def get_prompt(name: str, arguments: dict[str, str] | None = None) -> dict[str, Any]:
    """Get a specific prompt template."""
    if arguments is None:
        arguments = {}

    if name == "create_contract":
        contract_type = arguments.get("contract_type", "PAM")
        schema = contracts.get_contract_schema(contract_type)
        example_result = examples.get_example("pam_example") if contract_type == "PAM" else {}

        return {
            "description": f"Create a {contract_type} contract",
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": f"""I want to create a {contract_type} contract in JACTUS.

Required fields: {list(schema.get('required_fields', {}).keys())}
Optional fields: {list(schema.get('optional_fields', {}).keys())}

Please help me create a complete, working example with proper validation.
"""
                    }
                }
            ]
        }

    elif name == "troubleshoot_error":
        error_message = arguments.get("error_message", "")
        health = system.health_check()

        return {
            "description": "Troubleshoot JACTUS error",
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": f"""I'm getting this error in JACTUS:
{error_message}

System health check:
{health}

Please help me understand what's wrong and how to fix it.
"""
                    }
                }
            ]
        }

    elif name == "understand_contract":
        contract_type = arguments.get("contract_type", "PAM")
        info = contracts.get_contract_info(contract_type)

        return {
            "description": f"Explain {contract_type} contract",
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": f"""Please explain the {contract_type} contract type in JACTUS.

Contract info: {info}

I want to understand:
1. What is this contract used for?
2. What are the key parameters?
3. How do cash flows work?
4. What are common use cases?
"""
                    }
                }
            ]
        }

    elif name == "compare_contracts":
        type1 = arguments.get("contract_type_1", "PAM")
        type2 = arguments.get("contract_type_2", "LAM")
        info1 = contracts.get_contract_info(type1)
        info2 = contracts.get_contract_info(type2)

        return {
            "description": f"Compare {type1} vs {type2}",
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": f"""Please compare {type1} and {type2} contracts:

{type1}: {info1}
{type2}: {info2}

What are the key differences? When should I use each one?
"""
                    }
                }
            ]
        }

    else:
        return {
            "description": "Unknown prompt",
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": f"Unknown prompt: {name}"
                    }
                }
            ]
        }


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
