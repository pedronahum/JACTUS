"""``jactus observer`` subcommands: list, describe."""

from __future__ import annotations

import typer

from jactus.cli.output import OutputFormat, print_error, print_json, print_table

observer_app = typer.Typer(no_args_is_help=True)

# ---------------------------------------------------------------------------
# Observer metadata (mirrors MCP list_risk_factor_observers)
# ---------------------------------------------------------------------------

OBSERVERS: dict[str, dict[str, str | bool | dict[str, str]]] = {
    "constant": {
        "class": "ConstantRiskFactorObserver",
        "description": "Flat rate environment. All risk factors held constant.",
        "use_case": "Fixed-rate contracts, testing, simple scenarios",
        "cli_flag": "--observer constant --observer-params '{\"rate\": 0.05}'",
        "python_only": False,
    },
    "market": {
        "class": "DictRiskFactorObserver",
        "description": "Maps risk factor identifiers to fixed values.",
        "use_case": "Contracts needing different values per risk factor identifier",
        "cli_flag": '--observer market --observer-params \'{"risk_factors": {"LIBOR-3M": 0.05}}\'',
        "python_only": False,
    },
    "time_series": {
        "class": "TimeSeriesRiskFactorObserver",
        "description": "Time-varying rates with step or linear interpolation.",
        "use_case": "Floating-rate contracts with rate resets",
        "cli_flag": '--observer market --observer-params \'{"time_series": {"LIBOR-3M": [["2024-01-01", 0.04], ["2024-07-01", 0.045]]}}\'',
        "python_only": False,
    },
    "curve": {
        "class": "CurveRiskFactorObserver",
        "description": "Yield/rate curves keyed by tenor for term structure modeling.",
        "use_case": "Term structure modeling, yield curve pricing",
        "python_only": True,
    },
    "composite": {
        "class": "CompositeRiskFactorObserver",
        "description": "Chains multiple observers with fallback behavior.",
        "use_case": "Complex scenarios with different data sources",
        "python_only": True,
    },
    "callback": {
        "class": "CallbackRiskFactorObserver",
        "description": "Delegates to user-provided Python callables.",
        "use_case": "Custom pricing models, external data integration",
        "python_only": True,
    },
    "jax": {
        "class": "JaxRiskFactorObserver",
        "description": "Integer-indexed, fully JAX-compatible for jit/grad/vmap.",
        "use_case": "Autodiff, sensitivity analysis, batch scenarios",
        "python_only": True,
    },
    "prepayment": {
        "class": "PrepaymentSurfaceObserver",
        "description": "Behavioral: 2D surface-based prepayment model (spread x age).",
        "use_case": "MBS, CLO modeling, loan portfolio prepayment",
        "python_only": True,
        "behavioral": True,
    },
    "deposit": {
        "class": "DepositTransactionObserver",
        "description": "Behavioral: deposit inflows/outflows for UMP contracts.",
        "use_case": "Bank balance sheet modeling, deposit behavior",
        "python_only": True,
        "behavioral": True,
    },
}


@observer_app.command("list")
def list_observers() -> None:
    """List available risk factor observer types."""
    from jactus.cli import get_state

    state = get_state()

    if state.output == OutputFormat.JSON:
        data = [
            {
                "type": key,
                "class": obs["class"],
                "description": obs["description"],
                "python_only": obs.get("python_only", False),
                "behavioral": obs.get("behavioral", False),
            }
            for key, obs in OBSERVERS.items()
        ]
        print_json(data, state.pretty)
    else:
        rows = []
        for key, obs in OBSERVERS.items():
            flags = []
            if obs.get("python_only"):
                flags.append("Python-only")
            if obs.get("behavioral"):
                flags.append("Behavioral")
            rows.append([key, str(obs["description"]), ", ".join(flags) or "CLI + MCP"])
        print_table(
            "RISK FACTOR OBSERVERS", ["Type", "Description", "Availability"], rows, state.no_color
        )


@observer_app.command("describe")
def describe_observer(
    observer_type: str = typer.Option(
        ..., "--type", help="Observer type (e.g., constant, market, time_series)"
    ),
) -> None:
    """Show parameters and usage for a specific observer type."""
    from jactus.cli import get_state

    state = get_state()
    key = observer_type.lower()

    if key not in OBSERVERS:
        print_error(f"Unknown observer type: {key}. Available: {', '.join(OBSERVERS)}")
        raise typer.Exit(code=1)

    obs = OBSERVERS[key]
    if state.output == OutputFormat.JSON:
        print_json(dict(obs, type=key), state.pretty)
    else:
        from jactus.cli.output import console

        console.print(f"[bold]{obs['class']}[/bold] ({key})")
        console.print(f"  {obs['description']}")
        console.print(f"  Use case: {obs['use_case']}")
        if "cli_flag" in obs:
            console.print(f"  CLI usage: {obs['cli_flag']}")
        if obs.get("python_only"):
            console.print("  [yellow]Python API only[/yellow]")
