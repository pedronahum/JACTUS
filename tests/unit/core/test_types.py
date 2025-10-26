"""Unit tests for type definitions and enumerations.

Test ID: T1.1
"""

import json

from jactus.core.types import (
    BusinessDayConvention,
    Calendar,
    ContractPerformance,
    ContractRole,
    ContractType,
    CyclePointOfInterestPayment,
    DayCountConvention,
    EndOfMonthConvention,
    EventType,
    FeeBasis,
    InterestCalculationBase,
    PrepaymentEffect,
    ScalingEffect,
)


class TestEventType:
    """Test EventType enumeration."""

    def test_all_event_types_instantiable(self):
        """Test that all event types can be instantiated."""
        event_types = [
            EventType.AD,
            EventType.IED,
            EventType.MD,
            EventType.PR,
            EventType.PI,
            EventType.PP,
            EventType.PY,
            EventType.PRF,
            EventType.FP,
            EventType.PRD,
            EventType.TD,
            EventType.IP,
            EventType.IPCI,
            EventType.IPCB,
            EventType.RR,
            EventType.RRF,
            EventType.DV,
            EventType.DVF,
            EventType.SC,
            EventType.STD,
            EventType.XD,
            EventType.CE,
        ]
        assert len(event_types) == 22

    def test_event_type_string_value(self):
        """Test that event types have correct string values."""
        assert EventType.IED == "IED"
        assert EventType.IP == "IP"
        assert EventType.MD == "MD"

    def test_event_type_comparison(self):
        """Test event type comparison works correctly."""
        assert EventType.IED == EventType.IED
        assert EventType.IED != EventType.IP
        assert EventType.IED == "IED"  # Can compare with string

    def test_event_type_json_serializable(self):
        """Test that event types are JSON serializable."""
        data = {"event": EventType.IED}
        json_str = json.dumps(data, default=str)
        assert "IED" in json_str


class TestContractType:
    """Test ContractType enumeration."""

    def test_all_contract_types_instantiable(self):
        """Test that all contract types can be instantiated."""
        contract_types = [
            ContractType.PAM,
            ContractType.LAM,
            ContractType.LAX,
            ContractType.NAM,
            ContractType.ANN,
            ContractType.CLM,
            ContractType.UMP,
            ContractType.CSH,
            ContractType.STK,
            ContractType.COM,
            ContractType.FXOUT,
            ContractType.SWPPV,
            ContractType.SWAPS,
            ContractType.CAPFL,
            ContractType.OPTNS,
            ContractType.FUTUR,
            ContractType.CEG,
            ContractType.CEC,
        ]
        assert len(contract_types) == 18

    def test_contract_type_string_value(self):
        """Test that contract types have correct string values."""
        assert ContractType.PAM == "PAM"
        assert ContractType.ANN == "ANN"
        assert ContractType.SWPPV == "SWPPV"


class TestContractRole:
    """Test ContractRole enumeration."""

    def test_all_contract_roles_instantiable(self):
        """Test that all contract roles can be instantiated."""
        roles = [
            ContractRole.RPA,
            ContractRole.RPL,
            ContractRole.LG,
            ContractRole.ST,
            ContractRole.BUY,
            ContractRole.SEL,
            ContractRole.RFL,
            ContractRole.PFL,
            ContractRole.COL,
            ContractRole.CNO,
            ContractRole.GUA,
            ContractRole.OBL,
            ContractRole.UDL,
            ContractRole.UDLP,
            ContractRole.UDLM,
        ]
        assert len(roles) == 15

    def test_contract_role_get_sign_assets(self):
        """Test that asset roles return +1."""
        assert ContractRole.RPA.get_sign() == 1
        assert ContractRole.LG.get_sign() == 1
        assert ContractRole.BUY.get_sign() == 1
        assert ContractRole.RFL.get_sign() == 1
        assert ContractRole.COL.get_sign() == 1
        assert ContractRole.CNO.get_sign() == 1
        assert ContractRole.OBL.get_sign() == 1
        assert ContractRole.UDL.get_sign() == 1
        assert ContractRole.UDLP.get_sign() == 1

    def test_contract_role_get_sign_liabilities(self):
        """Test that liability roles return -1."""
        assert ContractRole.RPL.get_sign() == -1
        assert ContractRole.ST.get_sign() == -1
        assert ContractRole.SEL.get_sign() == -1
        assert ContractRole.PFL.get_sign() == -1
        assert ContractRole.GUA.get_sign() == -1
        assert ContractRole.UDLM.get_sign() == -1

    def test_contract_role_sign_all_roles_covered(self):
        """Test that all roles have a sign defined."""
        for role in ContractRole:
            sign = role.get_sign()
            assert sign in (1, -1), f"Role {role} must have sign +1 or -1"


class TestDayCountConvention:
    """Test DayCountConvention enumeration."""

    def test_all_day_count_conventions_instantiable(self):
        """Test that all day count conventions can be instantiated."""
        conventions = [
            DayCountConvention.AA,
            DayCountConvention.A360,
            DayCountConvention.A365,
            DayCountConvention.E30360ISDA,
            DayCountConvention.E30360,
            DayCountConvention.B30360,
            DayCountConvention.BUS252,
        ]
        assert len(conventions) == 7

    def test_day_count_convention_values(self):
        """Test day count convention string values."""
        assert DayCountConvention.AA == "AA"
        assert DayCountConvention.A360 == "A360"
        assert DayCountConvention.E30360ISDA == "30E360ISDA"


class TestBusinessDayConvention:
    """Test BusinessDayConvention enumeration."""

    def test_all_business_day_conventions_instantiable(self):
        """Test that all business day conventions can be instantiated."""
        conventions = [
            BusinessDayConvention.NULL,
            BusinessDayConvention.SCF,
            BusinessDayConvention.SCMF,
            BusinessDayConvention.CSF,
            BusinessDayConvention.CSMF,
            BusinessDayConvention.SCP,
            BusinessDayConvention.SCMP,
            BusinessDayConvention.CSP,
            BusinessDayConvention.CSMP,
        ]
        assert len(conventions) == 9

    def test_business_day_convention_values(self):
        """Test business day convention string values."""
        assert BusinessDayConvention.NULL == "NULL"
        assert BusinessDayConvention.SCMF == "SCMF"


class TestEndOfMonthConvention:
    """Test EndOfMonthConvention enumeration."""

    def test_all_end_of_month_conventions_instantiable(self):
        """Test that all end of month conventions can be instantiated."""
        conventions = [
            EndOfMonthConvention.EOM,
            EndOfMonthConvention.SD,
        ]
        assert len(conventions) == 2

    def test_end_of_month_convention_values(self):
        """Test end of month convention string values."""
        assert EndOfMonthConvention.EOM == "EOM"
        assert EndOfMonthConvention.SD == "SD"


class TestCalendar:
    """Test Calendar enumeration."""

    def test_all_calendars_instantiable(self):
        """Test that all calendars can be instantiated."""
        calendars = [
            Calendar.NO_CALENDAR,
            Calendar.MONDAY_TO_FRIDAY,
            Calendar.TARGET,
            Calendar.US_NYSE,
            Calendar.UK_SETTLEMENT,
            Calendar.CUSTOM,
        ]
        assert len(calendars) == 6

    def test_calendar_values(self):
        """Test calendar string values."""
        assert Calendar.NO_CALENDAR == "NO_CALENDAR"
        assert Calendar.MONDAY_TO_FRIDAY == "MONDAY_TO_FRIDAY"


class TestContractPerformance:
    """Test ContractPerformance enumeration."""

    def test_all_contract_performances_instantiable(self):
        """Test that all contract performances can be instantiated."""
        performances = [
            ContractPerformance.PF,
            ContractPerformance.DL,
            ContractPerformance.DQ,
            ContractPerformance.DF,
        ]
        assert len(performances) == 4

    def test_contract_performance_values(self):
        """Test contract performance string values."""
        assert ContractPerformance.PF == "PF"
        assert ContractPerformance.DF == "DF"


class TestFeeBasis:
    """Test FeeBasis enumeration."""

    def test_all_fee_bases_instantiable(self):
        """Test that all fee bases can be instantiated."""
        bases = [
            FeeBasis.A,
            FeeBasis.N,
        ]
        assert len(bases) == 2

    def test_fee_basis_values(self):
        """Test fee basis string values."""
        assert FeeBasis.A == "A"
        assert FeeBasis.N == "N"


class TestInterestCalculationBase:
    """Test InterestCalculationBase enumeration."""

    def test_all_interest_calculation_bases_instantiable(self):
        """Test that all interest calculation bases can be instantiated."""
        bases = [
            InterestCalculationBase.NT,
            InterestCalculationBase.NTIED,
            InterestCalculationBase.NTL,
        ]
        assert len(bases) == 3

    def test_interest_calculation_base_values(self):
        """Test interest calculation base string values."""
        assert InterestCalculationBase.NT == "NT"
        assert InterestCalculationBase.NTIED == "NTIED"


class TestCyclePointOfInterestPayment:
    """Test CyclePointOfInterestPayment enumeration."""

    def test_all_cycle_points_instantiable(self):
        """Test that all cycle points can be instantiated."""
        points = [
            CyclePointOfInterestPayment.B,
            CyclePointOfInterestPayment.E,
        ]
        assert len(points) == 2

    def test_cycle_point_values(self):
        """Test cycle point string values."""
        assert CyclePointOfInterestPayment.B == "B"
        assert CyclePointOfInterestPayment.E == "E"


class TestPrepaymentEffect:
    """Test PrepaymentEffect enumeration."""

    def test_all_prepayment_effects_instantiable(self):
        """Test that all prepayment effects can be instantiated."""
        effects = [
            PrepaymentEffect.N,
            PrepaymentEffect.A,
            PrepaymentEffect.M,
        ]
        assert len(effects) == 3

    def test_prepayment_effect_values(self):
        """Test prepayment effect string values."""
        assert PrepaymentEffect.N == "N"
        assert PrepaymentEffect.A == "A"


class TestScalingEffect:
    """Test ScalingEffect enumeration."""

    def test_all_scaling_effects_instantiable(self):
        """Test that all scaling effects can be instantiated."""
        effects = [
            ScalingEffect.S000,
            ScalingEffect.I00,
            ScalingEffect.S0N0,
            ScalingEffect.IN0,
            ScalingEffect.S00M,
            ScalingEffect.I0M,
            ScalingEffect.S0NM,
            ScalingEffect.INM,
        ]
        assert len(effects) == 8

    def test_scaling_effect_values(self):
        """Test scaling effect string values."""
        assert ScalingEffect.S000 == "000"
        assert ScalingEffect.I00 == "I00"
        assert ScalingEffect.IN0 == "IN0"
        assert ScalingEffect.INM == "INM"


class TestEnumJSONSerialization:
    """Test that all enums are JSON serializable."""

    def test_all_enums_json_serializable(self):
        """Test that all enum types can be serialized to JSON."""
        test_data = {
            "event_type": EventType.IED,
            "contract_type": ContractType.PAM,
            "contract_role": ContractRole.RPA,
            "day_count": DayCountConvention.AA,
            "business_day": BusinessDayConvention.SCMF,
            "eom": EndOfMonthConvention.EOM,
            "calendar": Calendar.NO_CALENDAR,
            "performance": ContractPerformance.PF,
            "fee_basis": FeeBasis.A,
            "interest_base": InterestCalculationBase.NT,
            "cycle_point": CyclePointOfInterestPayment.E,
            "prepayment": PrepaymentEffect.N,
            "scaling": ScalingEffect.INM,
        }

        # Convert to JSON and back
        json_str = json.dumps(test_data, default=str)
        loaded = json.loads(json_str)

        # Verify all values preserved
        assert loaded["event_type"] == "IED"
        assert loaded["contract_type"] == "PAM"
        assert loaded["contract_role"] == "RPA"


class TestEnumStringComparison:
    """Test that enums can be compared with strings."""

    def test_enum_string_equality(self):
        """Test that enums can be compared with their string values."""
        assert EventType.IED == "IED"
        assert ContractType.PAM == "PAM"
        assert ContractRole.RPA == "RPA"
        assert DayCountConvention.AA == "AA"

    def test_enum_inequality(self):
        """Test enum inequality comparisons."""
        assert EventType.IED != "MD"
        assert ContractType.PAM != "ANN"
        assert EventType.IED != EventType.MD
