"""Shared output formatting for the JACTUS CLI.

All formatting decisions (JSON vs rich tables vs CSV) are centralized here.
No formatting logic should exist in individual command modules.
"""

import csv
import io
import json
import sys
from enum import Enum
from typing import Any

from rich.console import Console
from rich.table import Table

# Consoles: stdout for results, stderr for errors/logs
console = Console()
err_console = Console(stderr=True)


class OutputFormat(str, Enum):
    """Output format for CLI commands."""

    TEXT = "text"
    JSON = "json"
    CSV = "csv"


def auto_output_format() -> OutputFormat:
    """Auto-detect output format based on whether stdout is a TTY."""
    if sys.stdout.isatty():
        return OutputFormat.TEXT
    return OutputFormat.JSON


def resolve_format(fmt: OutputFormat | None) -> OutputFormat:
    """Resolve the output format, using auto-detection if None."""
    if fmt is None:
        return auto_output_format()
    return fmt


def print_json(data: Any, pretty: bool = True) -> None:
    """Print data as JSON to stdout."""
    indent = 2 if pretty else None
    sys.stdout.write(json.dumps(data, indent=indent, default=str) + "\n")
    sys.stdout.flush()


def print_table(
    title: str, columns: list[str], rows: list[list[str]], no_color: bool = False
) -> None:
    """Print a rich table to stdout."""
    out = Console(no_color=no_color)
    table = Table(title=title, show_header=True, header_style="bold")
    for col in columns:
        table.add_column(col)
    for row in rows:
        table.add_row(*[str(cell) for cell in row])
    out.print(table)


def print_csv_output(columns: list[str], rows: list[list[Any]]) -> None:
    """Print data as CSV to stdout."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(columns)
    writer.writerows(rows)
    sys.stdout.write(buf.getvalue())
    sys.stdout.flush()


def print_error(message: str) -> None:
    """Print an error message to stderr."""
    err_console.print(f"[red]Error:[/red] {message}")


def format_currency(value: float) -> str:
    """Format a float as a currency string."""
    if value < 0:
        return f"-${abs(value):,.2f}"
    return f"${value:,.2f}"
