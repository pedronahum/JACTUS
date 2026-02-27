"""Contract type implementations for various ACTUS contract types.

This module provides:
- Base contract infrastructure (BaseContract, SimulationHistory)
- Concrete contract implementations (CSH, PAM, STK, COM)
- Contract factory pattern for dynamic instantiation
- Type registration system for extensibility

Example:
    >>> from jactus.contracts import create_contract
    >>> from jactus.core import ContractAttributes, ContractType
    >>>
    >>> # Create contract using factory
    >>> attrs = ContractAttributes(
    ...     contract_id="CSH-001",
    ...     contract_type=ContractType.CSH,
    ...     notional_principal=100000.0,
    ...     currency="USD",
    ... )
    >>> contract = create_contract(attrs, risk_factor_observer)
    >>> result = contract.simulate()
"""

from jactus.contracts.ann import AnnuityContract
from jactus.contracts.base import (
    BaseContract,
    SimulationHistory,
    merge_scheduled_and_observed_events,
    sort_events_by_sequence,
)
from jactus.contracts.capfl import CapFloorContract
from jactus.contracts.cec import CreditEnhancementCollateralContract
from jactus.contracts.ceg import CreditEnhancementGuaranteeContract
from jactus.contracts.clm import CallMoneyContract
from jactus.contracts.com import CommodityContract
from jactus.contracts.csh import CashContract
from jactus.contracts.futur import FutureContract
from jactus.contracts.fxout import FXOutrightContract
from jactus.contracts.lam import LinearAmortizerContract
from jactus.contracts.lax import ExoticLinearAmortizerContract
from jactus.contracts.nam import NegativeAmortizerContract
from jactus.contracts.optns import OptionContract
from jactus.contracts.pam import PrincipalAtMaturityContract
from jactus.contracts.pam_array import (
    PAMArrayParams,
    PAMArrayState,
    batch_simulate_pam,
    prepare_pam_batch,
    precompute_pam_arrays,
    simulate_pam_array,
    simulate_pam_array_jit,
    simulate_pam_portfolio,
)
from jactus.contracts.stk import StockContract
from jactus.contracts.swaps import GenericSwapContract
from jactus.contracts.swppv import PlainVanillaSwapContract
from jactus.contracts.ump import UndefinedMaturityProfileContract
from jactus.core import ContractAttributes, ContractType
from jactus.observers import ChildContractObserver, RiskFactorObserver

# Contract Registry
# Maps ContractType enum values to their implementation classes
CONTRACT_REGISTRY: dict[ContractType, type[BaseContract]] = {
    ContractType.CSH: CashContract,
    ContractType.PAM: PrincipalAtMaturityContract,
    ContractType.LAM: LinearAmortizerContract,
    ContractType.NAM: NegativeAmortizerContract,
    ContractType.ANN: AnnuityContract,
    ContractType.LAX: ExoticLinearAmortizerContract,
    ContractType.CLM: CallMoneyContract,
    ContractType.UMP: UndefinedMaturityProfileContract,
    ContractType.STK: StockContract,
    ContractType.COM: CommodityContract,
    ContractType.FXOUT: FXOutrightContract,
    ContractType.OPTNS: OptionContract,
    ContractType.FUTUR: FutureContract,
    ContractType.SWPPV: PlainVanillaSwapContract,
    ContractType.SWAPS: GenericSwapContract,
    ContractType.CAPFL: CapFloorContract,
    ContractType.CEG: CreditEnhancementGuaranteeContract,
    ContractType.CEC: CreditEnhancementCollateralContract,
}


def register_contract_type(contract_type: ContractType, contract_class: type[BaseContract]) -> None:
    """Register a new contract type in the factory registry.

    This allows for dynamic extensibility - users can register custom
    contract implementations without modifying the core library.

    Args:
        contract_type: The ContractType enum value
        contract_class: The contract implementation class (must extend BaseContract)

    Raises:
        TypeError: If contract_class doesn't extend BaseContract
        ValueError: If contract_type is already registered

    Example:
        >>> class MyCustomContract(BaseContract):
        ...     # Custom implementation
        ...     pass
        >>>
        >>> register_contract_type(ContractType.CUSTOM, MyCustomContract)
    """
    # Validate that contract_class extends BaseContract
    if not issubclass(contract_class, BaseContract):
        raise TypeError(f"Contract class must extend BaseContract, got {contract_class.__name__}")

    # Check if already registered
    if contract_type in CONTRACT_REGISTRY:
        raise ValueError(
            f"Contract type {contract_type.value} is already registered "
            f"with {CONTRACT_REGISTRY[contract_type].__name__}"
        )

    # Register the contract type
    CONTRACT_REGISTRY[contract_type] = contract_class


def create_contract(
    attributes: ContractAttributes,
    risk_factor_observer: RiskFactorObserver,
    child_contract_observer: ChildContractObserver | None = None,
) -> BaseContract:
    """Create a contract instance using the factory pattern.

    Dynamically instantiates the correct contract class based on the
    contract_type attribute. This is the recommended way to create
    contracts when the type is determined at runtime.

    Args:
        attributes: Contract attributes (must include contract_type)
        risk_factor_observer: Observer for market data and risk factors
        child_contract_observer: Optional observer for child contracts

    Returns:
        Instance of the appropriate contract class

    Raises:
        ValueError: If contract_type is not registered
        AttributeError: If attributes doesn't have contract_type

    Example:
        >>> from jactus.contracts import create_contract
        >>> from jactus.core import ContractAttributes, ContractType
        >>> from jactus.observers import ConstantRiskFactorObserver
        >>>
        >>> attrs = ContractAttributes(
        ...     contract_id="PAM-001",
        ...     contract_type=ContractType.PAM,
        ...     notional_principal=100000.0,
        ...     currency="USD",
        ... )
        >>> rf_obs = ConstantRiskFactorObserver(0.05)
        >>> contract = create_contract(attrs, rf_obs)
        >>> isinstance(contract, PrincipalAtMaturityContract)
        True
    """
    # Extract contract type from attributes
    contract_type = attributes.contract_type

    # Look up contract class in registry
    if contract_type not in CONTRACT_REGISTRY:
        available_types = ", ".join(ct.value for ct in CONTRACT_REGISTRY)
        raise ValueError(
            f"Unknown contract type: {contract_type.value}. Available types: {available_types}"
        )

    # Get the contract class
    contract_class = CONTRACT_REGISTRY[contract_type]

    # Instantiate and return
    return contract_class(
        attributes=attributes,
        risk_factor_observer=risk_factor_observer,
        child_contract_observer=child_contract_observer,
    )


def get_available_contract_types() -> list[ContractType]:
    """Get list of all registered contract types.

    Returns:
        List of ContractType enum values that are currently registered

    Example:
        >>> types = get_available_contract_types()
        >>> ContractType.CSH in types
        True
        >>> ContractType.PAM in types
        True
    """
    return list(CONTRACT_REGISTRY.keys())


__all__ = [
    # Base classes
    "BaseContract",
    "SimulationHistory",
    "sort_events_by_sequence",
    "merge_scheduled_and_observed_events",
    # Contract implementations
    "AnnuityContract",
    "CallMoneyContract",
    "CapFloorContract",
    "CashContract",
    "CommodityContract",
    "CreditEnhancementCollateralContract",
    "CreditEnhancementGuaranteeContract",
    "ExoticLinearAmortizerContract",
    "FutureContract",
    "FXOutrightContract",
    "GenericSwapContract",
    "LinearAmortizerContract",
    "NegativeAmortizerContract",
    "OptionContract",
    "PlainVanillaSwapContract",
    "PrincipalAtMaturityContract",
    "StockContract",
    "UndefinedMaturityProfileContract",
    # Factory pattern
    "CONTRACT_REGISTRY",
    "create_contract",
    "register_contract_type",
    "get_available_contract_types",
    # Array-mode PAM simulation
    "PAMArrayState",
    "PAMArrayParams",
    "simulate_pam_array",
    "simulate_pam_array_jit",
    "batch_simulate_pam",
    "precompute_pam_arrays",
    "prepare_pam_batch",
    "simulate_pam_portfolio",
]
