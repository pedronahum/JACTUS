"""Tests for documentation search tools."""

from jactus_mcp.tools import documentation


def test_search_docs_found():
    """Test searching for a term that exists."""
    result = documentation.search_docs("contract")

    assert result["found"] is True
    assert "results" in result
    assert len(result["results"]) > 0


def test_search_docs_architecture():
    """Test searching in ARCHITECTURE.md."""
    result = documentation.search_docs("JAX")

    assert result["found"] is True
    # Should find JAX in documentation
    file_names = [r["file"] for r in result["results"]]
    assert any("ARCHITECTURE" in f or "README" in f for f in file_names)


def test_search_docs_not_found():
    """Test searching for non-existent term."""
    result = documentation.search_docs("xyznonexistentterm123")

    assert result["found"] is False
    assert "available_docs" in result


def test_get_doc_structure():
    """Test getting documentation structure."""
    result = documentation.get_doc_structure()

    assert "documentation_files" in result
    assert result["documentation_files"] > 0
    assert "structure" in result

    # Should have README
    assert "README.md" in result["structure"]


def test_get_topic_guide_contracts():
    """Test getting contracts topic guide."""
    result = documentation.get_topic_guide("contracts")

    assert "title" in result
    assert "content" in result
    assert "Contract Types" in result["content"]


def test_get_topic_guide_jax():
    """Test getting JAX topic guide."""
    result = documentation.get_topic_guide("jax")

    assert "title" in result
    assert "JAX" in result["title"]
    assert "content" in result


def test_get_topic_guide_invalid():
    """Test getting invalid topic guide."""
    result = documentation.get_topic_guide("nonexistent_topic")

    assert "error" in result
    assert "available_topics" in result
