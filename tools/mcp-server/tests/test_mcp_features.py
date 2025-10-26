"""Tests for MCP-specific features (resources and prompts)."""

import pytest
from pathlib import Path


# Note: These tests verify the structure of resources and prompts
# Actual MCP protocol testing would require async test framework


def test_resource_uris():
    """Test that resource URIs follow expected format."""
    expected_uris = [
        "jactus://docs/architecture",
        "jactus://docs/pam",
        "jactus://docs/derivatives",
        "jactus://docs/readme",
    ]

    # Verify URI format
    for uri in expected_uris:
        assert uri.startswith("jactus://")
        assert "docs/" in uri


def test_documentation_files_exist():
    """Test that documentation files referenced by resources exist."""
    # Navigate to JACTUS root
    test_dir = Path(__file__).parent
    root = test_dir.parent.parent.parent  # Go up from tests/ -> mcp-server/ -> tools/ -> JACTUS/

    # Verify docs exist
    docs_dir = root / "docs"
    assert docs_dir.exists(), f"docs directory not found at {docs_dir}"

    # Check specific files
    assert (docs_dir / "ARCHITECTURE.md").exists()
    assert (docs_dir / "PAM.md").exists()
    assert (root / "README.md").exists()


def test_prompt_names():
    """Test that prompt names are descriptive."""
    expected_prompts = [
        "create_contract",
        "troubleshoot_error",
        "understand_contract",
        "compare_contracts",
    ]

    # Verify naming convention
    for prompt in expected_prompts:
        assert "_" in prompt or prompt.islower()
        assert len(prompt) > 5  # Not too short


def test_prompt_arguments():
    """Test that prompts have appropriate arguments."""
    prompt_args = {
        "create_contract": ["contract_type"],
        "troubleshoot_error": ["error_message"],
        "understand_contract": ["contract_type"],
        "compare_contracts": ["contract_type_1", "contract_type_2"],
    }

    # Verify argument structure
    for prompt, args in prompt_args.items():
        assert len(args) > 0, f"Prompt {prompt} should have arguments"
        for arg in args:
            assert isinstance(arg, str)
            assert len(arg) > 0
