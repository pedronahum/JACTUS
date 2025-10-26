"""Tests for contract factory pattern and type registration.

This module tests the factory pattern implementation for creating
contracts dynamically based on ContractType.
"""

import pytest

from jactus.contracts import (
    CONTRACT_REGISTRY,
    CashContract,
    CommodityContract,
    PrincipalAtMaturityContract,
    StockContract,
    create_contract,
    get_available_contract_types,
    register_contract_type,
)
from jactus.contracts.base import BaseContract
from jactus.core import ActusDateTime, ContractAttributes, ContractRole, ContractType
from jactus.observers import ConstantRiskFactorObserver


class TestContractRegistry:
    """Test CONTRACT_REGISTRY dictionary."""

    def test_registry_contains_all_implemented_types(self):
        """Test that registry contains all 4 implemented contract types."""
        assert ContractType.CSH in CONTRACT_REGISTRY
        assert ContractType.PAM in CONTRACT_REGISTRY
        assert ContractType.STK in CONTRACT_REGISTRY
        assert ContractType.COM in CONTRACT_REGISTRY

    def test_registry_maps_to_correct_classes(self):
        """Test that registry maps to the correct contract classes."""
        assert CONTRACT_REGISTRY[ContractType.CSH] == CashContract
        assert CONTRACT_REGISTRY[ContractType.PAM] == PrincipalAtMaturityContract
        assert CONTRACT_REGISTRY[ContractType.STK] == StockContract
        assert CONTRACT_REGISTRY[ContractType.COM] == CommodityContract

    def test_registry_classes_extend_base_contract(self):
        """Test that all registered classes extend BaseContract."""
        for contract_class in CONTRACT_REGISTRY.values():
            assert issubclass(contract_class, BaseContract)


class TestCreateContract:
    """Test create_contract factory function."""

    @pytest.fixture
    def rf_obs(self):
        """Mock risk factor observer."""
        return ConstantRiskFactorObserver(constant_value=0.05)

    def test_create_cash_contract(self, rf_obs):
        """Test factory creates CashContract for CSH type."""
        attrs = ContractAttributes(
            contract_id="CSH-001",
            contract_type=ContractType.CSH,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
        )

        contract = create_contract(attrs, rf_obs)

        assert isinstance(contract, CashContract)
        assert contract.attributes.contract_id == "CSH-001"

    def test_create_pam_contract(self, rf_obs):
        """Test factory creates PrincipalAtMaturityContract for PAM type."""
        attrs = ContractAttributes(
            contract_id="PAM-001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
        )

        contract = create_contract(attrs, rf_obs)

        assert isinstance(contract, PrincipalAtMaturityContract)
        assert contract.attributes.contract_id == "PAM-001"

    def test_create_stock_contract(self, rf_obs):
        """Test factory creates StockContract for STK type."""
        attrs = ContractAttributes(
            contract_id="STK-001",
            contract_type=ContractType.STK,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
        )

        contract = create_contract(attrs, rf_obs)

        assert isinstance(contract, StockContract)
        assert contract.attributes.contract_id == "STK-001"

    def test_create_commodity_contract(self, rf_obs):
        """Test factory creates CommodityContract for COM type."""
        attrs = ContractAttributes(
            contract_id="COM-001",
            contract_type=ContractType.COM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
        )

        contract = create_contract(attrs, rf_obs)

        assert isinstance(contract, CommodityContract)
        assert contract.attributes.contract_id == "COM-001"

    def test_create_contract_raises_for_unknown_type(self, rf_obs):
        """Test factory raises ValueError for unregistered contract type."""
        # Since all contract types are now implemented, we need to test
        # with a contract type that doesn't exist in the registry
        # We'll temporarily remove one and test
        from jactus.contracts import CONTRACT_REGISTRY

        # Save original
        original_class = CONTRACT_REGISTRY.pop(ContractType.FXOUT)

        try:
            attrs = ContractAttributes(
                contract_id="FXOUT-001",
                contract_type=ContractType.FXOUT,
                contract_role=ContractRole.RPA,
                status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                currency="USD",
            )

            with pytest.raises(ValueError, match="Unknown contract type: FXOUT"):
                create_contract(attrs, rf_obs)
        finally:
            # Restore
            CONTRACT_REGISTRY[ContractType.FXOUT] = original_class

    def test_create_contract_error_message_lists_available_types(self, rf_obs):
        """Test error message lists available contract types."""
        from jactus.contracts import CONTRACT_REGISTRY

        # Temporarily remove a contract type to test error message
        original_class = CONTRACT_REGISTRY.pop(ContractType.FXOUT)

        try:
            attrs = ContractAttributes(
                contract_id="FXOUT-001",
                contract_type=ContractType.FXOUT,
                contract_role=ContractRole.RPA,
                status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                currency="USD",
            )

            with pytest.raises(ValueError) as exc_info:
                create_contract(attrs, rf_obs)

            error_message = str(exc_info.value)
            assert "Available types:" in error_message
            assert "CSH" in error_message
            assert "PAM" in error_message
            assert "STK" in error_message
            assert "COM" in error_message
        finally:
            # Restore
            CONTRACT_REGISTRY[ContractType.FXOUT] = original_class

    def test_create_contract_with_child_observer(self, rf_obs):
        """Test factory passes child_contract_observer correctly."""

        # Create a mock implementation of ChildContractObserver protocol
        class MockChildObserver:
            def observe_child_contract(self, contract_id, time):
                return None

        attrs = ContractAttributes(
            contract_id="CSH-001",
            contract_type=ContractType.CSH,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
        )

        child_obs = MockChildObserver()
        contract = create_contract(attrs, rf_obs, child_contract_observer=child_obs)

        assert isinstance(contract, CashContract)
        assert contract.child_contract_observer is child_obs


class TestRegisterContractType:
    """Test register_contract_type function."""

    @pytest.fixture
    def rf_obs(self):
        """Mock risk factor observer."""
        return ConstantRiskFactorObserver(constant_value=0.05)

    def test_register_new_contract_type(self, rf_obs):
        """Test registering a new contract type."""

        # Create a mock contract class
        class MockContract(BaseContract):
            def generate_event_schedule(self):
                return None

            def initialize_state(self):
                return None

            def get_payoff_function(self, event_type):
                return None

            def get_state_transition_function(self, event_type):
                return None

        # Since all contract types are implemented, we need to temporarily
        # remove one to test re-registration
        original_class = CONTRACT_REGISTRY.pop(ContractType.FXOUT)
        initial_size = len(CONTRACT_REGISTRY)

        try:
            # Register new type
            register_contract_type(ContractType.FXOUT, MockContract)

            # Verify registration
            assert ContractType.FXOUT in CONTRACT_REGISTRY
            assert CONTRACT_REGISTRY[ContractType.FXOUT] == MockContract
            assert len(CONTRACT_REGISTRY) == initial_size + 1

            # Verify factory can create it
            attrs = ContractAttributes(
                contract_id="FXOUT-001",
                contract_type=ContractType.FXOUT,
                contract_role=ContractRole.RPA,
                status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                currency="USD",
            )
            contract = create_contract(attrs, rf_obs)
            assert isinstance(contract, MockContract)

        finally:
            # Cleanup: restore original
            if ContractType.FXOUT in CONTRACT_REGISTRY:
                del CONTRACT_REGISTRY[ContractType.FXOUT]
            CONTRACT_REGISTRY[ContractType.FXOUT] = original_class

    def test_register_raises_for_non_base_contract_class(self):
        """Test registration raises TypeError for non-BaseContract class."""

        class NotAContract:
            pass

        with pytest.raises(TypeError, match="must extend BaseContract"):
            register_contract_type(ContractType.LAM, NotAContract)

    def test_register_raises_for_already_registered_type(self):
        """Test registration raises ValueError for already registered type."""
        with pytest.raises(ValueError, match="already registered"):
            register_contract_type(ContractType.CSH, CashContract)


class TestGetAvailableContractTypes:
    """Test get_available_contract_types helper."""

    def test_returns_list_of_contract_types(self):
        """Test function returns list of ContractType enum values."""
        types = get_available_contract_types()

        assert isinstance(types, list)
        assert len(types) >= 4  # At least CSH, PAM, STK, COM
        assert all(isinstance(t, ContractType) for t in types)

    def test_contains_all_implemented_types(self):
        """Test returned list contains all implemented types."""
        types = get_available_contract_types()

        assert ContractType.CSH in types
        assert ContractType.PAM in types
        assert ContractType.STK in types
        assert ContractType.COM in types

    def test_contains_derivative_types(self):
        """Test returned list contains derivative contract types."""
        types = get_available_contract_types()

        # These derivative types should all be implemented
        derivatives = [ContractType.FXOUT, ContractType.SWPPV, ContractType.SWAPS,
                      ContractType.OPTNS, ContractType.FUTUR, ContractType.CAPFL,
                      ContractType.CEG, ContractType.CEC]

        for contract_type in derivatives:
            assert contract_type in types


class TestFactoryPatternIntegration:
    """Integration tests for factory pattern."""

    @pytest.fixture
    def rf_obs(self):
        """Mock risk factor observer."""
        return ConstantRiskFactorObserver(constant_value=0.05)

    def test_factory_created_contracts_are_functional(self, rf_obs):
        """Test that factory-created contracts are fully functional."""
        # Create CSH contract
        attrs = ContractAttributes(
            contract_id="CSH-001",
            contract_type=ContractType.CSH,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
        )

        contract = create_contract(attrs, rf_obs)

        # Should be able to simulate
        result = contract.simulate()
        assert result is not None
        assert len(result.events) >= 1

    def test_factory_preserves_type_safety(self, rf_obs):
        """Test that factory maintains type safety."""
        attrs_csh = ContractAttributes(
            contract_id="CSH-001",
            contract_type=ContractType.CSH,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
        )

        attrs_pam = ContractAttributes(
            contract_id="PAM-001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
        )

        csh = create_contract(attrs_csh, rf_obs)
        pam = create_contract(attrs_pam, rf_obs)

        # Type checking should work
        assert isinstance(csh, CashContract)
        assert not isinstance(csh, PrincipalAtMaturityContract)
        assert isinstance(pam, PrincipalAtMaturityContract)
        assert not isinstance(pam, CashContract)

    def test_all_registered_types_can_be_created(self, rf_obs):
        """Test that all registered types can be successfully created."""
        test_configs = {
            ContractType.CSH: {
                "contract_id": "CSH-001",
                "notional_principal": 100000.0,
            },
            ContractType.PAM: {
                "contract_id": "PAM-001",
                "initial_exchange_date": ActusDateTime(2024, 1, 15, 0, 0, 0),
                "maturity_date": ActusDateTime(2029, 1, 15, 0, 0, 0),
                "notional_principal": 100000.0,
            },
            ContractType.STK: {
                "contract_id": "STK-001",
            },
            ContractType.COM: {
                "contract_id": "COM-001",
            },
        }

        for contract_type, extra_attrs in test_configs.items():
            attrs = ContractAttributes(
                contract_type=contract_type,
                contract_role=ContractRole.RPA,
                status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                currency="USD",
                **extra_attrs,
            )

            contract = create_contract(attrs, rf_obs)
            assert isinstance(contract, BaseContract)
            assert contract.attributes.contract_type == contract_type
