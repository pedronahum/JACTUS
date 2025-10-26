"""Utility functions for schedules, conventions, and mathematical operations."""

from jactus.utilities.calendars import (
    CustomCalendar,
    HolidayCalendar,
    MondayToFridayCalendar,
    NoHolidayCalendar,
    get_calendar,
    is_weekend,
)
from jactus.utilities.conventions import days_between_30_360_methods, year_fraction
from jactus.utilities.math import (
    annuity_amount,
    annuity_amount_vectorized,
    calculate_actus_annuity,
    calculate_actus_annuity_jax,
    compound_factor,
    compound_factor_vectorized,
    contract_role_sign,
    contract_role_sign_vectorized,
    discount_factor,
    discount_factor_vectorized,
    present_value,
    present_value_vectorized,
)
from jactus.utilities.schedules import (
    apply_business_day_convention,
    apply_end_of_month_convention,
    expand_period_to_months,
    generate_array_schedule,
    generate_schedule,
)

__all__ = [
    # Schedule generation
    "generate_schedule",
    "generate_array_schedule",
    "apply_end_of_month_convention",
    "apply_business_day_convention",
    "expand_period_to_months",
    # Day count conventions
    "year_fraction",
    "days_between_30_360_methods",
    # Calendars
    "HolidayCalendar",
    "NoHolidayCalendar",
    "MondayToFridayCalendar",
    "CustomCalendar",
    "get_calendar",
    "is_weekend",
    # Financial math
    "contract_role_sign",
    "contract_role_sign_vectorized",
    "annuity_amount",
    "annuity_amount_vectorized",
    "calculate_actus_annuity",
    "calculate_actus_annuity_jax",
    "discount_factor",
    "discount_factor_vectorized",
    "compound_factor",
    "compound_factor_vectorized",
    "present_value",
    "present_value_vectorized",
]
