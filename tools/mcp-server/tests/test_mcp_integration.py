"""Protocol-level MCP integration tests.

Tests tools, resources, and prompts through the actual MCP protocol
using the mcp.client module with pytest-asyncio.
"""

import json

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@pytest.fixture
def server_params():
    """Server parameters for spawning the JACTUS MCP server."""
    return StdioServerParameters(
        command="python",
        args=["-m", "jactus_mcp"],
    )


# ---- Tool Discovery ----


@pytest.mark.asyncio
async def test_list_tools(server_params):
    """Verify all expected tools are listed."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]

            expected = [
                "jactus_list_contracts",
                "jactus_get_contract_info",
                "jactus_get_contract_schema",
                "jactus_get_event_types",
                "jactus_list_examples",
                "jactus_get_example",
                "jactus_get_quick_start",
                "jactus_run_example",
                "jactus_validate_attributes",
                "jactus_search_docs",
                "jactus_get_doc_structure",
                "jactus_get_topic_guide",
                "jactus_health_check",
                "jactus_get_version_info",
                "jactus_simulate_contract",
            ]
            for name in expected:
                assert name in tool_names, f"Missing tool: {name}"


# ---- Tool Calls ----


@pytest.mark.asyncio
async def test_call_list_contracts(server_params):
    """Call jactus_list_contracts through protocol and verify JSON response."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("jactus_list_contracts", {})

            # Verify the result is valid JSON, not a Python repr string
            content = result.content[0].text
            data = json.loads(content)
            assert data["total_contracts"] == 18
            assert "categories" in data
            assert "PAM" in data["all_contracts"]


@pytest.mark.asyncio
async def test_call_get_contract_info(server_params):
    """Call jactus_get_contract_info for PAM."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "jactus_get_contract_info",
                {"contract_type": "PAM"},
            )
            content = result.content[0].text
            data = json.loads(content)
            assert data["contract_type"] == "PAM"
            assert data["implemented"] is True


@pytest.mark.asyncio
async def test_call_simulate_contract(server_params):
    """Call jactus_simulate_contract through protocol."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "jactus_simulate_contract",
                {
                    "attributes": {
                        "contract_type": "PAM",
                        "contract_id": "MCP-TEST-001",
                        "contract_role": "RPA",
                        "status_date": "2024-01-01",
                        "initial_exchange_date": "2024-01-15",
                        "maturity_date": "2025-01-15",
                        "notional_principal": 100000.0,
                        "nominal_interest_rate": 0.05,
                        "day_count_convention": "30E360",
                    },
                },
            )

            content = result.content[0].text
            data = json.loads(content)
            assert data["success"] is True
            assert data["num_events"] > 0
            assert isinstance(data["events"], list)
            assert "summary" in data
            assert "total_cashflows" in data["summary"]


@pytest.mark.asyncio
async def test_call_validate_attributes(server_params):
    """Call jactus_validate_attributes through protocol."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "jactus_validate_attributes",
                {
                    "attributes": {
                        "contract_type": "PAM",
                        "contract_id": "TEST",
                        "contract_role": "RPA",
                        "status_date": "2024-01-01",
                        "initial_exchange_date": "2024-01-15",
                        "maturity_date": "2025-01-15",
                        "notional_principal": 100000.0,
                        "nominal_interest_rate": 0.05,
                        "day_count_convention": "30E360",
                    },
                },
            )
            content = result.content[0].text
            data = json.loads(content)
            assert data["valid"] is True


@pytest.mark.asyncio
async def test_call_health_check(server_params):
    """Call jactus_health_check through protocol."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("jactus_health_check", {})
            content = result.content[0].text
            data = json.loads(content)
            assert data["status"] in ("healthy", "degraded")
            assert "versions" in data


# ---- Resources ----


@pytest.mark.asyncio
async def test_list_resources(server_params):
    """Verify resources are listed."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            resources = await session.list_resources()
            uris = [str(r.uri) for r in resources.resources]

            expected_uris = [
                "jactus://docs/architecture",
                "jactus://docs/pam",
                "jactus://docs/derivatives",
                "jactus://docs/readme",
            ]
            for uri in expected_uris:
                assert uri in uris, f"Missing resource: {uri}"


# ---- Prompts ----


@pytest.mark.asyncio
async def test_list_prompts(server_params):
    """Verify prompts are listed."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            prompts = await session.list_prompts()
            names = [p.name for p in prompts.prompts]

            expected = [
                "create_contract",
                "troubleshoot_error",
                "understand_contract",
                "compare_contracts",
            ]
            for name in expected:
                assert name in names, f"Missing prompt: {name}"
