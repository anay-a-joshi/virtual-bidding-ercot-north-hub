# ==============================================================================
# Name    : Anay Abhijit Joshi
# Date    : November 22, 2026
# Course  : ISyE 6767 - Interim Project 2 (Virtual Bidding at ERCOT HB_NORTH)
# File    : lookahead_guard.py
# Purpose : The look-ahead test of Lab 2, Task 5. Every price feature of the
#           project must pass it (Data Quality Check 6 of project2.pdf).
# ==============================================================================

import numpy as np
import pandas as pd

# So, these three days are the ones of the Lab 2 notebook: a summer day, a spring-forward
# day (23 hours) and a day after fall-back, so DST edge cases are always tested.
LAB_POISON_DAYS = ("2023-07-15", "2024-03-10", "2025-11-03")
# In my opinion three days are not enough to call a feature safe, so the project uses a superset:
# every split boundary (lags that cross from validation into test), month/leap-day edges, every
# 23- and 25-hour day, the days right after them, and the very last delivery day.
EXTENDED_POISON_DAYS = LAB_POISON_DAYS + (
    "2023-01-01", "2023-01-02", "2024-01-01", "2025-01-01", "2025-01-02",
    "2023-03-12", "2023-03-13", "2023-11-05", "2023-11-06", "2024-11-03", "2024-11-04",
    "2025-03-09", "2025-11-02", "2024-02-29", "2024-03-01", "2025-06-30", "2025-12-31")
DEFAULT_POISON_DAYS = EXTENDED_POISON_DAYS
POISONED_OUTCOME_COLUMNS = ("da_price", "rt_price", "spread")


def assert_no_lookahead(feature_builder_fn, hourly_frame, poison_days=DEFAULT_POISON_DAYS, be_quiet=False):
    # feature_builder_fn(frame) -> DataFrame of features with the same index as frame.
    # Let's do now the test: overwrite every price from D-1 on with garbage, rebuild the
    # features, and check that the features of day D did not change at all.
    clean_feature_frame = feature_builder_fn(hourly_frame)
    for poison_day in pd.to_datetime(list(poison_days)):
        rows_of_day = hourly_frame["delivery_date"] == poison_day
        if not rows_of_day.any():
            raise ValueError(f"Oops! Poison day {poison_day.date()} is not in the data.")
        # So, poison twice: once with huge POSITIVE garbage and once with huge NEGATIVE garbage. A feature
        # that only reads the SIGN of a D-1 spread cannot hide behind "yesterday was positive anyway".
        for garbage_centre in (1e4, -1e4):
            poisoned_frame = hourly_frame.copy()
            is_too_late = poisoned_frame["delivery_date"] >= poison_day - pd.Timedelta(days=1)
            garbage_generator = np.random.default_rng(0)
            for outcome_column in POISONED_OUTCOME_COLUMNS:
                poisoned_frame.loc[is_too_late, outcome_column] = garbage_generator.normal(
                    garbage_centre, 1e3, int(is_too_late.sum()))
            poisoned_frame["y"] = (poisoned_frame["spread"] > 0).astype(int)
            dirty_feature_frame = feature_builder_fn(poisoned_frame)
            if not clean_feature_frame[rows_of_day].equals(dirty_feature_frame[rows_of_day]):
                leaking_columns = [c for c in clean_feature_frame
                                   if not clean_feature_frame.loc[rows_of_day, c].equals(dirty_feature_frame.loc[rows_of_day, c])]
                raise AssertionError(f"Oops! look-ahead on {poison_day.date()} in {leaking_columns}")
    if not be_quiet:
        print("  [PASS] No look-ahead in", list(clean_feature_frame.columns))
    return True


def passes_lookahead(feature_builder_fn, hourly_frame, poison_days=DEFAULT_POISON_DAYS):
    # So, a boolean version, handy for unit tests and for the "wrong features" demonstration.
    try:
        assert_no_lookahead(feature_builder_fn, hourly_frame, poison_days, be_quiet=True)
        return True, ""
    except AssertionError as caught_error:
        return False, str(caught_error)


def wrong_features_for_demo(hourly_frame):
    # Finally, the two classic mistakes from Lab 2, kept on purpose to show the test catches them.
    return pd.DataFrame({"spread_shift24": hourly_frame["spread"].shift(24),
                         "spread_roll7d": hourly_frame["spread"].rolling(24 * 7).mean()},
                        index=hourly_frame.index)
