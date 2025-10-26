"""Basic example demonstrating JACTUS usage.

This example will be expanded in Phase 1+ to show:
- Creating contracts
- Generating payment schedules
- Computing cash flows
- Calculating risk metrics
"""

import jactus
from jactus.logging_config import configure_logging, get_logger

# Configure logging
configure_logging(level="INFO")

# Get a logger
logger = get_logger(__name__)


def main() -> None:
    """Run basic example."""
    logger.info(f"JACTUS version: {jactus.__version__}")
    logger.info("JACTUS is ready!")

    # Example placeholder for Phase 1+
    # contract = create_pam_contract(...)
    # schedule = generate_schedule(contract)
    # cashflows = compute_cashflows(contract, schedule)

    logger.info("This example will be expanded in Phase 1+")


if __name__ == "__main__":
    main()
