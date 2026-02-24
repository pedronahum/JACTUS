"""Tests for example retrieval tools."""

from jactus_mcp.tools import examples


def test_list_examples():
    """Test listing all examples."""
    result = examples.list_examples()

    assert "python_scripts" in result
    assert "jupyter_notebooks" in result
    assert "total" in result

    # Should have some examples
    assert len(result["python_scripts"]) > 0


def test_list_examples_includes_pam():
    """Test that PAM example is listed."""
    result = examples.list_examples()

    script_names = [ex["name"] for ex in result["python_scripts"]]
    assert "pam_example" in script_names


def test_get_example_pam():
    """Test retrieving PAM example."""
    result = examples.get_example("pam_example")

    assert "error" not in result
    assert result["name"] == "pam_example"
    assert "code" in result
    assert len(result["code"]) > 0
    assert "import jactus" in result["code"] or "from jactus" in result["code"]


def test_get_example_with_py_extension():
    """Test retrieving example with .py extension."""
    result = examples.get_example("pam_example.py")

    assert "error" not in result
    assert "code" in result


def test_get_example_not_found():
    """Test retrieving non-existent example."""
    result = examples.get_example("nonexistent_example")

    assert "error" in result
    assert "available_examples" in result


def test_get_quick_start_example():
    """Test quick start example."""
    code = examples.get_quick_start_example()

    assert isinstance(code, str)
    assert len(code) > 0
    assert "from jactus" in code
    assert "ContractType.PAM" in code
    assert "create_contract" in code
