"""``jactus docs`` subcommands: search."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from jactus.cli.output import OutputFormat, print_json, print_table

docs_app = typer.Typer(no_args_is_help=True)


def _get_docs_dir() -> Path:
    """Locate the JACTUS docs/ directory relative to the package."""
    # Walk up from src/jactus/cli/docs.py to find project root
    current = Path(__file__).parent
    for _ in range(10):
        if (current / "docs").is_dir() and (current / "src" / "jactus").is_dir():
            return current / "docs"
        if (current / "docs").is_dir() and (current / "pyproject.toml").is_file():
            return current / "docs"
        current = current.parent
    # Fallback: 4 levels up from cli/
    return Path(__file__).parent.parent.parent.parent / "docs"


def _get_project_root() -> Path:
    """Locate the JACTUS project root."""
    current = Path(__file__).parent
    for _ in range(10):
        if (current / "src" / "jactus").is_dir():
            return current
        current = current.parent
    return Path(__file__).parent.parent.parent.parent


@docs_app.command("search")
def search(
    query: str = typer.Argument(help="Search query"),
    limit: int = typer.Option(5, "--limit", help="Max results per file"),
    source: str | None = typer.Option(None, "--source", help="Filter: api, guide, example"),  # noqa: UP007
) -> None:
    """Search JACTUS documentation for a topic."""
    from jactus.cli import get_state

    state = get_state()
    project_root = _get_project_root()
    docs_dir = project_root / "docs"

    doc_files = [
        docs_dir / "ARCHITECTURE.md",
        docs_dir / "PAM.md",
        docs_dir / "ARRAY_MODE.md",
        docs_dir / "derivatives.md",
        project_root / "README.md",
    ]

    query_lower = query.lower()
    query_words = query_lower.split()
    results: list[dict[str, Any]] = []

    for doc_file in doc_files:
        if not doc_file.exists():
            continue
        try:
            content = doc_file.read_text()
            lines = content.split("\n")
            matches = []
            for i, line in enumerate(lines):
                if any(word in line.lower() for word in query_words):
                    start = max(0, i - 2)
                    end = min(len(lines), i + 3)
                    context = "\n".join(lines[start:end])
                    matches.append(
                        {
                            "line_number": i + 1,
                            "line": line.strip(),
                            "context": context,
                        }
                    )
            if matches:
                results.append(
                    {
                        "file": doc_file.name,
                        "path": str(doc_file.relative_to(project_root)),
                        "matches": matches[:limit],
                        "total_matches": len(matches),
                    }
                )
        except Exception:
            continue

    if state.output == OutputFormat.JSON:
        print_json(
            {
                "query": query,
                "found": bool(results),
                "total_files": len(results),
                "results": results,
            },
            state.pretty,
        )
    else:
        if not results:
            from jactus.cli.output import console

            console.print(f"No results for '{query}'")
            return

        rows = []
        for r in results:
            for m in r["matches"]:
                rows.append([r["file"], str(m["line_number"]), m["line"][:80]])
        print_table(f"Search: '{query}'", ["File", "Line", "Match"], rows, state.no_color)
