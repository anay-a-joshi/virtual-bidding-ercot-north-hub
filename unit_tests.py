# ==============================================================================
# Name    : Anay Abhijit Joshi
# Date    : November 22, 2026
# Course  : ISyE 6767 - Interim Project 2 (Virtual Bidding at ERCOT HB_NORTH)
# File    : unit_tests.py
# Purpose : Unit tests for the data pipeline. They reproduce every check cell of
#           the Lab 2 notebook plus the DST edge cases.  Run:  python unit_tests.py
# ==============================================================================

import sys

import numpy as np
import pandas as pd

import config
from data_pipeline import ErcotDatasetBuilder, hour_start_utc, load_hourly_dataset
from lookahead_guard import EXTENDED_POISON_DAYS, passes_lookahead, wrong_features_for_demo

TESTS_PASSED = 0
TESTS_FAILED = 0


def check_true(test_name, condition_value, detail_text=""):
    global TESTS_PASSED, TESTS_FAILED
    if bool(condition_value):
        TESTS_PASSED += 1
        print(f"  [PASS] {test_name}")
    else:
        TESTS_FAILED += 1
        print(f"  [FAIL] {test_name}  {detail_text}")


def check_close(test_name, computed_value, expected_value, tolerance=1e-9):
    global TESTS_PASSED, TESTS_FAILED
    if abs(float(computed_value) - float(expected_value)) <= tolerance:
        TESTS_PASSED += 1
        print(f"  [PASS] {test_name}: computed = {computed_value:.6f}, expected = {expected_value:.6f}")
    else:
        TESTS_FAILED += 1
        print(f"  [FAIL] {test_name}: computed = {computed_value:.6f}, expected = {expected_value:.6f}")


def utc_stamp(text_value):
    return pd.Timestamp(text_value, tz="UTC")


def test_hour_keys_on_dst_days():
    print("\n******************** TEST GROUP 1: Hour-Ending -> UTC Key On DST Days ********************")
    # So, fall back 2023-11-05: HE2 twice. First copy is CDT (UTC-5), repeated copy is CST (UTC-6).
    fall_back_keys = hour_start_utc(["2023-11-05"] * 3, [2, 2, 3], [False, True, False])
    check_true("Fall-back HE2 first copy -> 06:00 UTC", fall_back_keys[0] == utc_stamp("2023-11-05 06:00"))
    check_true("Fall-back HE2 repeated copy -> 07:00 UTC", fall_back_keys[1] == utc_stamp("2023-11-05 07:00"))
    check_true("Fall-back HE3 -> 08:00 UTC", fall_back_keys[2] == utc_stamp("2023-11-05 08:00"))
    # Let's do now spring forward 2023-03-12: 02:00-03:00 never happens, so HE3 cannot exist.
    spring_keys = hour_start_utc(["2023-03-12"] * 3, [2, 3, 4], [False, False, False])
    check_true("Spring-forward HE2 -> 07:00 UTC (CST)", spring_keys[0] == utc_stamp("2023-03-12 07:00"))
    check_true("Spring-forward HE3 -> NaT (impossible hour)", pd.isna(spring_keys[1]))
    check_true("Spring-forward HE4 -> 08:00 UTC (CDT)", spring_keys[2] == utc_stamp("2023-03-12 08:00"))
    check_true("HE24 of a summer day (23:00 CDT) -> 04:00 UTC next day",
               hour_start_utc(["2024-07-01"], [24], [False])[0] == utc_stamp("2024-07-02 04:00"))


def test_price_tables(dataset_builder):
    print("\n******************** TEST GROUP 2: Lab 2 Check 1 And Check 2 (Prices) ********************")
    dam_frame = dataset_builder.load_day_ahead()
    check_true("DA has 35,064 hours in 2022-2025", len(dam_frame) == 35064, f"got {len(dam_frame)}")
    check_true("DA day sizes {24: 1453, 23: 4, 25: 4}",
               dam_frame.groupby("delivery_date").size().value_counts().to_dict() == {24: 1453, 23: 4, 25: 4})
    check_true("DA keys complete and unique",
               dam_frame["interval_start_utc"].notna().all() and not dam_frame["interval_start_utc"].duplicated().any())
    da_series = dam_frame.set_index("interval_start_utc")["da_price"]
    check_close("DA 2023-11-05 HE2 first copy (CDT)", da_series[utc_stamp("2023-11-05 06:00")], 21.68)
    check_close("DA 2023-11-05 HE2 repeated copy (CST)", da_series[utc_stamp("2023-11-05 07:00")], 24.98)

    rtm_frame = dataset_builder.load_real_time()
    check_true("RT has 35,064 hours, each with 4 intervals",
               len(rtm_frame) == 35064 and (rtm_frame["rt_n_intervals"] == 4).all())
    check_true("RT keys == DA keys", set(rtm_frame["interval_start_utc"]) == set(dam_frame["interval_start_utc"]))
    rt_series = rtm_frame.set_index("interval_start_utc")["rt_price"]
    check_close("RT 2023-11-05 HE2 first copy (CDT)", rt_series[utc_stamp("2023-11-05 06:00")], 22.0125)
    check_close("RT 2023-11-05 HE2 repeated copy (CST)", rt_series[utc_stamp("2023-11-05 07:00")], 23.8775)


def test_forecasts(dataset_builder):
    print("\n******************** TEST GROUP 3: Lab 2 Check 3 (Forecasts) ********************")
    vintage_summary = dataset_builder.verify_forecast_vintage()
    check_true("No forecast file posted after 9:00 AM on D-1", (vintage_summary["posted_late"] == 0).all())
    forecast_frames = dataset_builder.load_all_forecasts()
    check_true("Load rows = 27,048", len(forecast_frames["load_forecast"]) == 27048)
    check_true("Wind rows = 27,048", len(forecast_frames["wind_forecast"]) == 27048)
    check_true("Solar rows = 27,045 (3 impossible HE03 rows dropped)", len(forecast_frames["solar_forecast"]) == 27045)
    no_actuals = all(not any(c.startswith(("ACTUAL", "COP_HSL")) for c in f.columns) for f in forecast_frames.values())
    check_true("No ACTUAL / COP_HSL column enters the dataset", no_actuals)


def test_final_table(dataset_builder):
    print("\n******************** TEST GROUP 4: Lab 2 Check 4 (One Table) ********************")
    built_frame = dataset_builder.build_hourly_table()
    saved_frame = load_hourly_dataset()
    check_true("Shape is (27048, 14)", saved_frame.shape == (27048, 14), f"got {saved_frame.shape}")
    check_true("No missing values", saved_frame.isna().sum().sum() == 0)
    check_true("UTC key strictly increasing", saved_frame["interval_start_utc"].is_monotonic_increasing
               and not saved_frame["interval_start_utc"].duplicated().any())
    check_true("1,127 delivery days from 2022-12-01 to 2025-12-31",
               saved_frame["delivery_date"].nunique() == 1127
               and saved_frame["delivery_date"].min() == pd.Timestamp(config.FIRST_DELIVERY_DAY)
               and saved_frame["delivery_date"].max() == pd.Timestamp(config.LAST_DELIVERY_DAY))
    check_close("max |spread - (DA - RT)|", np.max(np.abs(saved_frame["spread"]
                                                          - (saved_frame["da_price"] - saved_frame["rt_price"]))), 0.0)
    check_true("y == 1{spread > 0}", (saved_frame["y"] == (saved_frame["spread"] > 0).astype(int)).all())
    filled_log = dataset_builder.get_filled_value_log()
    check_true("Exactly 3 filled values, all solar, all 0 MW",
               len(filled_log) == 3 and (filled_log["column"] == "solar_fcst_mw").all()
               and (filled_log["filled_value"] == 0).all())
    check_true("CSV round trip keeps the 3 short and 3 long DST days",
               saved_frame.groupby("delivery_date").size().value_counts().to_dict() == {24: 1121, 23: 3, 25: 3})
    check_true("Saved CSV equals the in-memory table (prices)",
               np.allclose(built_frame["spread"].values, saved_frame["spread"].values))


def correct_spread_lag_two_days(hourly_frame):
    # So, the Lab 2 Task 5 recipe: first copy of each hour, move its date forward 2 days, merge back.
    lag_source = hourly_frame.loc[~hourly_frame["repeated"], ["delivery_date", "hour_ending", "spread"]].copy()
    lag_source["delivery_date"] = lag_source["delivery_date"] + pd.Timedelta(days=2)
    merged_lag = hourly_frame[["delivery_date", "hour_ending"]].merge(
        lag_source.rename(columns={"spread": "spread_lag2"}), on=["delivery_date", "hour_ending"], how="left")
    return pd.DataFrame({"spread_lag2": merged_lag["spread_lag2"].values}, index=hourly_frame.index)


def test_lookahead_guard():
    print("\n******************** TEST GROUP 5: Lab 2 Task 5 (Look-Ahead Guard) ********************")
    hourly_frame = load_hourly_dataset()
    did_pass_wrong, _ = passes_lookahead(wrong_features_for_demo, hourly_frame)
    check_true("Guard CATCHES shift(24) and unshifted rolling(24*7)", not did_pass_wrong)
    did_pass_right, failure_text = passes_lookahead(correct_spread_lag_two_days, hourly_frame)
    check_true("Guard PASSES the D-2 same-hour merge lag", did_pass_right, failure_text)
    lag_frame = correct_spread_lag_two_days(hourly_frame)
    row_i = hourly_frame.index[hourly_frame["interval_start_utc"] == utc_stamp("2024-07-03 20:00")][0]
    row_j = hourly_frame.index[hourly_frame["interval_start_utc"] == utc_stamp("2024-07-01 20:00")][0]
    check_close("spread_lag2 on 2024-07-03 20:00 UTC == spread on 2024-07-01 20:00 UTC",
                lag_frame.loc[row_i, "spread_lag2"], hourly_frame.loc[row_j, "spread"])


def same_hour_feature_from_day_minus_one(hourly_frame, sign_only):
    # So, a DELIBERATELY leaking feature: the spread (or only its sign) at the same hour on D-1.
    lag_source = hourly_frame.loc[~hourly_frame["repeated"], ["delivery_date", "hour_ending", "spread"]].copy()
    lag_source["delivery_date"] = lag_source["delivery_date"] + pd.Timedelta(days=1)
    if sign_only:
        lag_source["spread"] = np.sign(lag_source["spread"])
    merged_lag = hourly_frame[["delivery_date", "hour_ending"]].merge(lag_source, on=["delivery_date", "hour_ending"],
                                                                     how="left")
    return pd.DataFrame({"leaky_d1": merged_lag["spread"].values}, index=hourly_frame.index)


def test_edge_cases_and_hardening():
    print("\n******************** TEST GROUP 6: Edge Cases And Hardening ********************")
    hourly_frame = load_hourly_dataset()
    # Step 1: the guard must catch a D-1 lag, even one that only keeps the SIGN.
    did_pass, _ = passes_lookahead(lambda f: same_hour_feature_from_day_minus_one(f, sign_only=False), hourly_frame)
    check_true("Guard CATCHES a same-hour D-1 spread lag", not did_pass)
    did_pass, _ = passes_lookahead(lambda f: same_hour_feature_from_day_minus_one(f, sign_only=True), hourly_frame)
    check_true("Guard CATCHES a sign-only D-1 lag (positive AND negative poison)", not did_pass)
    # Step 2: features that never touch prices (calendar, forecasts) must pass untouched.
    did_pass, failure_text = passes_lookahead(
        lambda f: f[["hour_ending", "load_fcst_system_mw", "wind_fcst_mw", "solar_fcst_mw"]].copy(), hourly_frame)
    check_true("Guard PASSES calendar and forecast columns", did_pass, failure_text)
    check_true("Every extended poison day exists in the data",
               set(pd.to_datetime(list(EXTENDED_POISON_DAYS))).issubset(set(hourly_frame["delivery_date"])))

    # Step 3: the forecast fill may never borrow from the NEXT day's file (that file is posted too late).
    tiny_days = ["2024-07-01"] * 24 + ["2024-07-02"] * 24
    tiny_hours = list(range(1, 25)) * 2
    tiny_frame = pd.DataFrame({"delivery_date": pd.to_datetime(tiny_days), "hour_ending": tiny_hours})
    tiny_frame["interval_start_utc"] = hour_start_utc(tiny_frame["delivery_date"], tiny_frame["hour_ending"],
                                                      [False] * 48)
    tiny_frame["interval_start_local"] = tiny_frame["interval_start_utc"].dt.tz_convert(config.ERCOT_LOCAL_TZ)
    for forecast_name in ("load_fcst_system_mw", "load_fcst_north_mw", "wind_fcst_mw", "solar_fcst_mw"):
        tiny_frame[forecast_name] = 100.0
    tiny_frame.loc[22, "load_fcst_system_mw"] = 40000.0       # day 1, HE23
    tiny_frame.loc[23, "load_fcst_system_mw"] = np.nan        # day 1, HE24 is missing
    tiny_frame.loc[24, "load_fcst_system_mw"] = 99999.0       # day 2, HE01 (next day's file)
    tiny_frame.loc[30, "solar_fcst_mw"] = 0.0
    tiny_frame.loc[32, "solar_fcst_mw"] = 0.0
    tiny_frame.loc[31, "solar_fcst_mw"] = np.nan              # night hour between two zeros
    tiny_builder = ErcotDatasetBuilder(use_cache=True, be_verbose=False)
    filled_tiny = tiny_builder._fill_missing_forecasts(tiny_frame.copy())
    check_close("Missing HE24 filled from same-day HE23 only (not next-day HE01)",
                filled_tiny.loc[23, "load_fcst_system_mw"], 40000.0)
    check_close("Missing night solar hour between two zeros filled with 0", filled_tiny.loc[31, "solar_fcst_mw"], 0.0)
    lonely_frame = tiny_frame.iloc[23:24].copy()               # a day with a single, missing hour
    try:
        tiny_builder._fill_missing_forecasts(lonely_frame)
        check_true("Fill with no same-day neighbour raises an error", False)
    except ValueError:
        check_true("Fill with no same-day neighbour raises an error", True)

    # Step 4: no feature-like column sneaks in; the 14 columns are exactly the agreed ones.
    expected_columns = ["delivery_date", "hour_ending", "repeated", "interval_start_utc", "interval_start_local",
                        "da_price", "rt_price", "rt_n_intervals", "load_fcst_system_mw", "load_fcst_north_mw",
                        "wind_fcst_mw", "solar_fcst_mw", "spread", "y"]
    check_true("Dataset columns are exactly the 14 agreed columns", list(hourly_frame.columns) == expected_columns)
    check_true("Extreme prices are kept (max |spread| > $3,900, not clipped)", hourly_frame["spread"].abs().max() > 3900)
    check_true("Negative RT prices are kept (> 500 hours in 2023-2025)",
               (hourly_frame.loc[hourly_frame["delivery_date"].dt.year >= 2023, "rt_price"] < 0).sum() > 500)


def test_feature_factory():
    print("\n******************** TEST GROUP 7: Feature Factory (Hand-Computed Values) ********************")
    from feature_factory import (ALL_FEATURE_COLUMNS, FEATURE_DESCRIPTIONS, FeatureFactory, ForecastRevisionReader,
                                 nerc_holidays)
    from lookahead_guard import LAB_POISON_DAYS
    holiday_set = nerc_holidays(2022, 2025)
    check_true("NERC: Christmas 2022 (Sunday) observed Monday 2022-12-26", pd.Timestamp("2022-12-26") in holiday_set)
    check_true("NERC: New Year 2023 (Sunday) observed Monday 2023-01-02, not 2023-01-01",
               pd.Timestamp("2023-01-02") in holiday_set and pd.Timestamp("2023-01-01") not in holiday_set)
    check_true("NERC: Thanksgiving 2024-11-28, Memorial Day 2025-05-26, Labor Day 2024-09-02",
               {pd.Timestamp("2024-11-28"), pd.Timestamp("2025-05-26"), pd.Timestamp("2024-09-02")} <= holiday_set)
    check_true("NERC: 6 holidays per year", len(nerc_holidays(2024, 2024)) == 6)
    check_true("Every feature has a description", set(ALL_FEATURE_COLUMNS) == set(FEATURE_DESCRIPTIONS))

    hourly_frame = load_hourly_dataset()
    feature_factory = FeatureFactory(ForecastRevisionReader(be_verbose=False).get_previous_vintage(), be_verbose=False)
    feature_frame = feature_factory.build_feature_frame(hourly_frame)
    # So, the reference hour: 2024-07-03 HE16 (20:00 UTC). Let's recompute every price feature by hand.
    target_row = hourly_frame.index[hourly_frame["interval_start_utc"] == utc_stamp("2024-07-03 20:00")][0]
    he16 = hourly_frame[hourly_frame["hour_ending"] == 16].set_index("delivery_date")["spread"]
    check_close("spread_lag2 == spread 2024-07-01 HE16", feature_frame.loc[target_row, "spread_lag2"],
                he16[pd.Timestamp("2024-07-01")])
    check_close("spread_lag7 == spread 2024-06-26 HE16", feature_frame.loc[target_row, "spread_lag7"],
                he16[pd.Timestamp("2024-06-26")])
    window7 = he16[pd.Timestamp("2024-06-25"):pd.Timestamp("2024-07-01")]
    window28 = he16[pd.Timestamp("2024-06-04"):pd.Timestamp("2024-07-01")]
    check_close("spread_mean7_d2 == mean HE16 spread 06-25..07-01", feature_frame.loc[target_row, "spread_mean7_d2"],
                window7.mean(), 1e-9)
    check_close("spread_std28_d2 == std HE16 spread 06-04..07-01", feature_frame.loc[target_row, "spread_std28_d2"],
                window28.std(), 1e-9)
    check_close("spread_fracpos28_d2 == share S>0 06-04..07-01", feature_frame.loc[target_row, "spread_fracpos28_d2"],
                (window28 > 0).mean(), 1e-12)
    day_two_before = hourly_frame[hourly_frame["delivery_date"] == pd.Timestamp("2024-07-01")]
    check_close("spread_daymean_lag2 == mean spread of 2024-07-01", feature_frame.loc[target_row, "spread_daymean_lag2"],
                day_two_before["spread"].mean(), 1e-9)
    check_close("net_load = load - wind - solar", feature_frame.loc[target_row, "net_load_mw"],
                hourly_frame.loc[target_row, ["load_fcst_system_mw"]].iloc[0] - hourly_frame.loc[target_row, "wind_fcst_mw"]
                - hourly_frame.loc[target_row, "solar_fcst_mw"], 1e-6)
    # Let's do now the revision from the raw files: file 20240703 minus file 20240702, rows of 07/03/2024 HE16.
    raw_now = pd.read_csv(config.RAW_ERCOT_DIR / "load_forecast" / "20240703.csv")
    raw_before = pd.read_csv(config.RAW_ERCOT_DIR / "load_forecast" / "20240702.csv")
    pick_row = lambda raw: raw[(raw["DeliveryDate"] == "07/03/2024") & (raw["HourEnding"] == "16:00")]["SystemTotal"].iloc[0]
    check_close("load_fcst_revision_mw == file D minus file D-1 (raw)", feature_frame.loc[target_row, "load_fcst_revision_mw"],
                pick_row(raw_now) - pick_row(raw_before), 1e-6)
    fall_back_rows = hourly_frame.index[(hourly_frame["delivery_date"] == pd.Timestamp("2024-11-03"))
                                        & (hourly_frame["hour_ending"] == 2)]
    check_true("Both copies of the fall-back HE2 get the same D-2 lag",
               len(fall_back_rows) == 2 and feature_frame.loc[fall_back_rows, "spread_lag2"].nunique() == 1)
    model_frame = feature_factory.build_model_dataset(hourly_frame)
    check_true("Model rows: 8,760 train / 8,784 validation / 8,760 test",
               model_frame["split"].value_counts().to_dict() == {"train": 8760, "validation": 8784, "test": 8760})
    check_true("No missing feature value in 2023-2025", model_frame[ALL_FEATURE_COLUMNS].isna().sum().sum() == 0)
    check_true("No outcome column is used as a feature",
               not set(ALL_FEATURE_COLUMNS) & {"da_price", "rt_price", "spread", "y", "rt_n_intervals"})
    did_pass, failure_text = passes_lookahead(feature_factory.build_feature_frame, hourly_frame, LAB_POISON_DAYS)
    check_true("Whole feature builder passes the look-ahead test (Lab days; main.py runs all 20)", did_pass, failure_text)


def test_backtest_engine():
    print("\n******************** TEST GROUP 8: Backtest Engine (Hand-Computed) ********************")
    from backtest_engine import StrategyBook, annualized_sharpe, hourly_profit, max_drawdown_dollars
    check_close("INC, v=2, S=5, c=1 -> 2*(5-1) = 8", hourly_profit([1], [2], [5], 1.0)[0], 8.0)
    check_close("DEC, v=1, S=5, c=1 -> 1*(-5-1) = -6", hourly_profit([-1], [1], [5], 1.0)[0], -6.0)
    check_close("No trade pays no cost -> 0", hourly_profit([0], [3], [5], 1.0)[0], 0.0)
    check_close("Max drawdown of +1, -3, +2, -1 is 3", max_drawdown_dollars([1, -3, 2, -1]), 3.0)
    check_close("Max drawdown counts a loss from day one (-2 -> 2)", max_drawdown_dollars([-2, 1]), 2.0)
    daily_example = np.array([1.0, 2.0, 3.0])
    check_close("Sharpe = mean/std * sqrt(365)", annualized_sharpe(daily_example), 2.0 / 1.0 * np.sqrt(365))
    try:
        hourly_profit([2], [1], [1], 0.0)
        check_true("An action of 2 is rejected", False)
    except ValueError:
        check_true("An action of 2 is rejected", True)
    tiny_rows = pd.DataFrame({"delivery_date": pd.to_datetime(["2025-01-01"] * 2 + ["2025-01-02"] * 2),
                              "interval_start_utc": pd.date_range("2025-01-01 06:00", periods=4, freq="h", tz="UTC"),
                              "hour_ending": [1, 2, 1, 2], "spread": [10.0, -4.0, 2.0, -1.0]})
    tiny_rows["interval_start_local"] = tiny_rows["interval_start_utc"].dt.tz_convert(config.ERCOT_LOCAL_TZ)
    tiny_book = StrategyBook("tiny", tiny_rows, [1, 1, -1, 0], [1, 2, 1, 1])
    tiny_metrics = tiny_book.metrics(1.0, capital_dollars=100.0)
    # So, hour pnl = 1*(10-1)=9, 2*(-4-1)=-10, 1*(-2-1)=-3, 0 -> total -4; MWh = 4; trades = 3; wins = 1/3.
    check_close("Tiny book total profit = -4", tiny_metrics["total_profit"], -4.0)
    check_close("Tiny book MWh traded = 4 (no-trade hour excluded)", tiny_metrics["mwh_traded"], 4.0)
    check_close("Tiny book win rate = 1/3", tiny_metrics["win_rate"], 1.0 / 3.0)
    check_close("Tiny book annualized return = (-4/2)*365/100", tiny_metrics["annualized_return"], -4.0 / 2 * 365 / 100)
    check_close("Tiny book max drawdown = 4 (day1 -1, day2 -3)", tiny_metrics["max_drawdown"], 4.0)


def test_part2_rules():
    print("\n******************** TEST GROUP 9: Part 2 Trading Rules ********************")
    from magnitude_models import classifier_view_a_edge, scale_to_total_volume, tau_rule_actions
    rule_actions = tau_rule_actions([3.0, -3.0, 1.5, -12.0, 0.0], cost_per_mwh=1.0, tau_dollars=1.0)
    check_true("tau rule: +1 if mu > c+tau, -1 if mu < -(c+tau), else 0",
               list(rule_actions) == [1.0, -1.0, 0.0, -1.0, 0.0])
    scaled = scale_to_total_volume([1.0, 3.0, 0.0, -2.0], 8.0)
    check_close("Sizes scaled to the same total volume (8 MWh)", scaled.sum(), 8.0)
    check_true("Negative raw sizes become 0, ratios kept", scaled[3] == 0 and abs(scaled[1] / scaled[0] - 3) < 1e-12)
    check_close("(2p-1)|S| edge: p=0.75, |S|=10, c=1 -> 4", classifier_view_a_edge([0.75], [10.0], 1.0)[0], 4.0)
    check_close("Capital base is positive", float(capital_base_dollars_for_test() > 0), 1.0)


def capital_base_dollars_for_test():
    from backtest_engine import capital_base_dollars
    from feature_factory import load_model_dataset
    model_frame = load_model_dataset()
    return capital_base_dollars(model_frame[model_frame["split"] == "train"])


def test_allocators():
    print("\n******************** TEST GROUP 10: Part 3 Allocators (Exact Checks) ********************")
    import itertools
    from budget_allocator import (bounded_knapsack_dp, cvar_limited_allocation, mean_variance_allocation,
                                  mean_variance_objective, project_onto_budget_box, ratio_greedy_knapsack,
                                  unit_knapsack_greedy, water_fill_allocation)
    check_true("Equal weights on 24 hours -> 1 MWh each", np.allclose(water_fill_allocation(np.ones(24)), 1.0))
    check_true("Equal weights on a 23-hour day -> 24/23 each", np.allclose(water_fill_allocation(np.ones(23)), 24 / 23))
    random_generator = np.random.default_rng(config.GLOBAL_RANDOM_SEED)
    skewed = water_fill_allocation(random_generator.exponential(1.0, 24) ** 3)
    check_true("Skewed weights: sum = 24, every hour <= 4", abs(skewed.sum() - 24) < 1e-6 and skewed.max() <= 4 + 1e-9)
    check_true("Only 3 hours with weight -> 4 each, budget not spent",
               list(water_fill_allocation(np.r_[np.ones(3), np.zeros(21)])[:4]) == [4, 4, 4, 0])
    all_equal = True
    for _ in range(30):
        unit_values = random_generator.normal(0, 3, 24)
        greedy_value = float(np.dot(np.clip(unit_values, 0, None), unit_knapsack_greedy(unit_values)))
        _, dp_value = bounded_knapsack_dp(unit_values, np.ones(24, dtype=int), 24)
        all_equal &= abs(greedy_value - dp_value) < 1e-9
    check_true("Unit weights: greedy == dynamic programme on 30 random days", all_equal)
    small_values, small_weights = np.array([5.0, 4.0, 3.0]), np.array([4, 3, 2])
    brute_best = max(sum(u * v for u, v in zip(units, small_values))
                     for units in itertools.product(range(3), repeat=3)
                     if sum(units) <= 4 and sum(u * w for u, w in zip(units, small_weights)) <= 9)
    _, dp_small = bounded_knapsack_dp(small_values, small_weights, 9, mwh_budget=4, cap_mwh=2)
    check_close("Collateral DP == brute force (3 hours, cap 2, 4 MWh, $9)", dp_small, brute_best)
    greedy_small = ratio_greedy_knapsack(small_values, small_weights, 9, mwh_budget=4, cap_mwh=2)
    check_true("Ratio greedy is feasible and never beats the DP",
               np.dot(greedy_small, small_weights) <= 9 and np.dot(greedy_small, small_values) <= dp_small + 1e-9)
    projected = project_onto_budget_box(random_generator.normal(3, 3, 24))
    check_true("Projection lands in {0 <= v <= 4, sum v <= 24}",
               projected.min() >= -1e-12 and projected.max() <= 4 + 1e-12 and projected.sum() <= 24 + 1e-9)
    edges = random_generator.normal(1.0, 2.0, 24)
    directions = np.where(random_generator.random(24) > 0.4, 1.0, -1.0)
    factor_matrix = random_generator.normal(0, 5, (24, 24))
    covariance = factor_matrix @ factor_matrix.T / 24 + np.eye(24)
    ours = mean_variance_allocation(edges, directions, covariance, 0.01)
    try:
        import cvxpy as cp
        volume_variable = cp.Variable(24)
        signed_cov = np.outer(directions, directions) * covariance
        problem = cp.Problem(cp.Maximize(edges @ volume_variable - 0.005 * cp.quad_form(volume_variable, cp.psd_wrap(signed_cov))),
                             [volume_variable >= 0, volume_variable <= 4, cp.sum(volume_variable) <= 24])
        problem.solve()
        check_close("Mean-variance: our solver == cvxpy objective",
                    mean_variance_objective(ours, edges, directions, covariance, 0.01), problem.value, 1e-3)
    except ImportError:
        check_true("cvxpy not installed, mean-variance cross-check skipped", True)
    scenario_spreads = random_generator.normal(2, 20, (56, 24))
    loose = cvar_limited_allocation(np.abs(edges), np.ones(24), scenario_spreads, 1.0, 1e9)
    check_close("CVaR with no binding limit spends the knapsack amount (24 MWh)", loose.sum(),
                unit_knapsack_greedy(np.abs(edges)).sum(), 1e-6)
    tight = cvar_limited_allocation(np.abs(edges), np.ones(24), scenario_spreads, 1.0, 1.0)
    check_true("A tighter CVaR limit trades less", tight.sum() < loose.sum())


def main():
    # Finally, every group in order, then the summary line.
    test_hour_keys_on_dst_days()
    dataset_builder = ErcotDatasetBuilder(use_cache=True, be_verbose=False)
    test_price_tables(dataset_builder)
    test_forecasts(dataset_builder)
    test_final_table(dataset_builder)
    test_lookahead_guard()
    test_edge_cases_and_hardening()
    test_feature_factory()
    test_backtest_engine()
    test_part2_rules()
    test_allocators()
    print(f"\nUnit Tests Passed: {TESTS_PASSED}, Failed: {TESTS_FAILED}")
    sys.exit(0 if TESTS_FAILED == 0 else 1)


if __name__ == "__main__":
    main()
