# ==============================================================================
# Name    : Anay Abhijit Joshi
# Date    : November 22, 2026
# Course  : ISyE 6767 - Interim Project 2 (Virtual Bidding at ERCOT HB_NORTH)
# File    : feature_factory.py
# Purpose : Every model feature, built so that it is known at 9:00 AM on D-1.
#           Price features use day D-2 or earlier (built with merges on
#           (delivery_date, hour_ending), never with shift), and the whole
#           feature builder is run through assert_no_lookahead (Check 6).
# ==============================================================================

import numpy as np
import pandas as pd

import config
from data_pipeline import hour_start_utc
from lookahead_guard import assert_no_lookahead

# So, the groups below are what the ablation study switches on and off.
FEATURE_GROUPS = {
    "calendar": ["hour_ending", "day_of_week", "month", "is_weekend", "is_nerc_holiday", "is_offpeak_day",
                 "is_onpeak_hour", "hour_sin", "hour_cos", "season_sin", "season_cos"],
    "price_lags": ["da_price_lag2", "rt_price_lag2", "spread_lag2", "da_price_lag7", "rt_price_lag7", "spread_lag7",
                   "spread_daymean_lag2", "da_daymean_lag2", "rt_daymax_lag2"],
    "rolling_stats": ["spread_mean7_d2", "spread_mean28_d2", "spread_std28_d2", "spread_fracpos28_d2",
                      "spread_absmean28_d2", "spread_median28_d2", "da_mean7_d2"],
    "load_forecast": ["load_fcst_system_mw", "load_fcst_north_mw"],
    "renewable_forecast": ["wind_fcst_mw", "solar_fcst_mw"],
    "derived_net_load": ["net_load_mw", "net_load_ramp_mw", "net_load_ramp3_mw", "net_load_from_daily_peak_mw",
                         "net_load_anomaly7_mw", "wind_share", "solar_share"],
    "forecast_revisions": ["load_fcst_revision_mw", "wind_fcst_revision_mw", "solar_fcst_revision_mw",
                           "net_load_revision_mw"],
}
PRICE_FEATURE_GROUPS = ("price_lags", "rolling_stats")

# So, one line per feature: this table goes straight into the LaTeX report.
FEATURE_DESCRIPTIONS = {
    "hour_ending": "Hour ending (1-24, Central)", "day_of_week": "Day of week (0 = Monday)", "month": "Month (1-12)",
    "is_weekend": "Saturday or Sunday", "is_nerc_holiday": "NERC holiday (Sunday holidays observed Monday)",
    "is_offpeak_day": "Weekend or NERC holiday", "is_onpeak_hour": "ERCOT on-peak: HE07-HE22 on a non-off-peak day",
    "hour_sin": "sin(2 pi hour / 24)", "hour_cos": "cos(2 pi hour / 24)",
    "season_sin": "sin(2 pi day-of-year / 365.25)", "season_cos": "cos(2 pi day-of-year / 365.25)",
    "da_price_lag2": "DA price, same hour, D-2", "rt_price_lag2": "RT price, same hour, D-2",
    "spread_lag2": "Spread, same hour, D-2", "da_price_lag7": "DA price, same hour, D-7",
    "rt_price_lag7": "RT price, same hour, D-7", "spread_lag7": "Spread, same hour, D-7",
    "spread_daymean_lag2": "Mean spread over all hours of D-2", "da_daymean_lag2": "Mean DA price over all hours of D-2",
    "rt_daymax_lag2": "Maximum RT price on D-2",
    "spread_mean7_d2": "Same-hour spread mean, 7 days ending D-2", "spread_mean28_d2": "Same-hour spread mean, 28 days ending D-2",
    "spread_std28_d2": "Same-hour spread std, 28 days ending D-2",
    "spread_fracpos28_d2": "Same-hour share of S > 0, 28 days ending D-2",
    "spread_absmean28_d2": "Same-hour mean |S|, 28 days ending D-2",
    "spread_median28_d2": "Same-hour spread median, 28 days ending D-2",
    "da_mean7_d2": "Same-hour DA price mean, 7 days ending D-2",
    "load_fcst_system_mw": "ERCOT system load forecast (D,h)", "load_fcst_north_mw": "North weather-zone load forecast (D,h)",
    "wind_fcst_mw": "System wind forecast STWPF (D,h)", "solar_fcst_mw": "System solar forecast STPPF (D,h)",
    "net_load_mw": "Net load = load - wind - solar", "net_load_ramp_mw": "Net load minus previous hour",
    "net_load_ramp3_mw": "Net load minus 3 hours earlier", "net_load_from_daily_peak_mw": "Net load minus day-D peak net load",
    "net_load_anomaly7_mw": "Net load minus same-hour mean of forecasts for D-7..D-1",
    "wind_share": "Wind forecast / load forecast", "solar_share": "Solar forecast / load forecast",
    "load_fcst_revision_mw": "Load forecast (file D) minus load forecast (file D-1)",
    "wind_fcst_revision_mw": "Wind forecast (file D) minus wind forecast (file D-1)",
    "solar_fcst_revision_mw": "Solar forecast (file D) minus solar forecast (file D-1)",
    "net_load_revision_mw": "Net-load forecast revision between the two files",
}
ALL_FEATURE_COLUMNS = [c for group_columns in FEATURE_GROUPS.values() for c in group_columns]
TARGET_COLUMNS = ["da_price", "rt_price", "spread", "y"]


def nerc_holidays(first_year, last_year):
    # So, NERC holidays (the power-industry calendar, also behind ERCOT's peak / off-peak days):
    # New Year's Day, Memorial Day, Independence Day, Labor Day, Thanksgiving, Christmas.
    # A fixed-date holiday on a Sunday is observed on Monday; a Saturday holiday is not moved.
    holiday_dates = set()
    for each_year in range(first_year, last_year + 1):
        fixed_dates = [pd.Timestamp(each_year, 1, 1), pd.Timestamp(each_year, 7, 4), pd.Timestamp(each_year, 12, 25)]
        for fixed_day in fixed_dates:
            holiday_dates.add(fixed_day + pd.Timedelta(days=1) if fixed_day.dayofweek == 6 else fixed_day)
        may_days = pd.date_range(f"{each_year}-05-01", f"{each_year}-05-31")
        holiday_dates.add(may_days[may_days.dayofweek == 0][-1])                       # last Monday of May
        september_days = pd.date_range(f"{each_year}-09-01", f"{each_year}-09-30")
        holiday_dates.add(september_days[september_days.dayofweek == 0][0])            # first Monday of September
        november_days = pd.date_range(f"{each_year}-11-01", f"{each_year}-11-30")
        holiday_dates.add(november_days[november_days.dayofweek == 3][3])              # fourth Thursday of November
    return holiday_dates


def same_hour_lag(hourly_frame, value_columns, lag_days):
    # Let's do now the Lab 2 Task 5 recipe: take the FIRST copy of every hour, move its date
    # forward by lag_days, and merge it back on (delivery_date, hour_ending). Never shift().
    lag_source = hourly_frame.loc[~hourly_frame["repeated"], ["delivery_date", "hour_ending", *value_columns]].copy()
    lag_source["delivery_date"] = lag_source["delivery_date"] + pd.Timedelta(days=lag_days)
    lag_source = lag_source.rename(columns={c: f"{c}_lag{lag_days}" for c in value_columns})
    merged_lag = hourly_frame[["delivery_date", "hour_ending"]].merge(lag_source, on=["delivery_date", "hour_ending"],
                                                                     how="left")
    return merged_lag.drop(columns=["delivery_date", "hour_ending"]).set_axis(hourly_frame.index)


def daily_aggregate_lag(hourly_frame, value_column, how_to_aggregate, lag_days, output_name):
    # So, one number per day (e.g. the mean spread of all hours of D-2), shared by every hour of day D.
    daily_values = hourly_frame.groupby("delivery_date")[value_column].agg(how_to_aggregate).rename(output_name)
    daily_values.index = daily_values.index + pd.Timedelta(days=lag_days)
    return hourly_frame[["delivery_date"]].join(daily_values, on="delivery_date")[[output_name]]


def same_hour_rolling(hourly_frame, value_column, window_days, statistic_name, output_name, end_lag_days=2):
    # In my opinion the cleanest leak-free rolling window: a (day x hour) table on a FULL calendar,
    # rolled over days, then moved forward by end_lag_days so the window for day D ends at D-2.
    first_copies = hourly_frame.loc[~hourly_frame["repeated"], ["delivery_date", "hour_ending", value_column]]
    day_by_hour = first_copies.pivot(index="delivery_date", columns="hour_ending", values=value_column)
    full_calendar = pd.date_range(day_by_hour.index.min(), day_by_hour.index.max(), freq="D")
    day_by_hour = day_by_hour.reindex(full_calendar)
    minimum_days = max(3, int(0.7 * window_days))
    if statistic_name == "fraction_positive":
        is_positive = (day_by_hour > 0).astype(float).where(day_by_hour.notna())
        rolled_table = is_positive.rolling(window_days, min_periods=minimum_days).mean()
    elif statistic_name == "abs_mean":
        rolled_table = day_by_hour.abs().rolling(window_days, min_periods=minimum_days).mean()
    else:
        rolled_table = getattr(day_by_hour.rolling(window_days, min_periods=minimum_days), statistic_name)()
    rolled_table.index = rolled_table.index + pd.Timedelta(days=end_lag_days)
    rolled_long = rolled_table.stack().rename(output_name).reset_index()
    rolled_long.columns = ["delivery_date", "hour_ending", output_name]
    merged_roll = hourly_frame[["delivery_date", "hour_ending"]].merge(rolled_long, on=["delivery_date", "hour_ending"],
                                                                      how="left")
    return merged_roll[[output_name]].set_axis(hourly_frame.index)


class ForecastRevisionReader:
    """Reads, from the raw files, the forecast for day D that was posted one day EARLIER (file D-1)."""

    def __init__(self, raw_data_dir=config.RAW_ERCOT_DIR, use_cache=True, be_verbose=True):
        self._raw_data_dir = raw_data_dir
        self._use_cache = use_cache
        self._be_verbose = be_verbose

    def _read_previous_vintage(self):
        previous_vintage_pieces = []
        for product_folder, (date_column, hour_column, wanted_columns) in config.FORECAST_PRODUCT_SPECS.items():
            product_rows = []
            for forecast_file in sorted((self._raw_data_dir / product_folder).glob("2*.csv")):
                file_day = pd.to_datetime(forecast_file.stem, format="%Y%m%d")
                next_day = file_day + pd.Timedelta(days=1)
                raw_forecast = pd.read_csv(forecast_file, dtype={hour_column: str})
                # So, file D-1 was posted at ~8:55 on D-2: its rows for day D are an OLDER forecast of D.
                next_day_rows = raw_forecast[pd.to_datetime(raw_forecast[date_column], format="%m/%d/%Y") == next_day]
                piece = pd.DataFrame({"delivery_date": next_day,
                                      "hour_ending": next_day_rows[hour_column].str.split(":").str[0].astype(int).values,
                                      "repeated": next_day_rows["DSTFlag"].eq("Y").values})
                for source_column, output_name in wanted_columns.items():
                    piece[f"{output_name}_prev"] = next_day_rows[source_column].astype(float).values
                product_rows.append(piece)
            product_frame = pd.concat(product_rows, ignore_index=True)
            product_frame["interval_start_utc"] = hour_start_utc(product_frame["delivery_date"],
                                                                 product_frame["hour_ending"], product_frame["repeated"])
            product_frame = product_frame.dropna(subset=["interval_start_utc"]).drop(
                columns=["delivery_date", "hour_ending", "repeated"])
            previous_vintage_pieces.append(product_frame.set_index("interval_start_utc"))
        return pd.concat(previous_vintage_pieces, axis=1).reset_index()

    def get_previous_vintage(self):
        cache_path = config.OUTPUT_CACHE_DIR / "forecast_previous_vintage.pkl"
        if self._use_cache and cache_path.exists():
            if self._be_verbose:
                print(f"  using cache {cache_path.name}")
            return pd.read_pickle(cache_path)
        config.ensure_output_folders()
        previous_vintage = self._read_previous_vintage()
        previous_vintage.to_pickle(cache_path)
        return previous_vintage


class FeatureFactory:
    """Builds the full feature table for every delivery hour; price features end at D-2."""

    def __init__(self, previous_vintage_frame, be_verbose=True):
        self._previous_vintage = previous_vintage_frame
        self._be_verbose = be_verbose
        self._holiday_dates = nerc_holidays(2022, 2026)
        self._fill_report = {}

    def get_fill_report(self):
        return dict(self._fill_report)

    def get_holiday_dates(self):
        return sorted(self._holiday_dates)

    # ---------------------------------------------------------------- groups
    def calendar_features(self, hourly_frame):
        local_start = hourly_frame["interval_start_local"]
        calendar_frame = pd.DataFrame(index=hourly_frame.index)
        calendar_frame["hour_ending"] = hourly_frame["hour_ending"].astype(int)
        calendar_frame["day_of_week"] = hourly_frame["delivery_date"].dt.dayofweek
        calendar_frame["month"] = hourly_frame["delivery_date"].dt.month
        calendar_frame["is_weekend"] = (calendar_frame["day_of_week"] >= 5).astype(int)
        calendar_frame["is_nerc_holiday"] = hourly_frame["delivery_date"].isin(self._holiday_dates).astype(int)
        calendar_frame["is_offpeak_day"] = ((calendar_frame["is_weekend"] + calendar_frame["is_nerc_holiday"]) > 0).astype(int)
        # So, ERCOT on-peak = HE07..HE22 on weekdays that are not NERC holidays.
        calendar_frame["is_onpeak_hour"] = (calendar_frame["hour_ending"].between(7, 22)
                                            & (calendar_frame["is_offpeak_day"] == 0)).astype(int)
        local_hour_fraction = (local_start.dt.hour + 0.5) / 24.0
        calendar_frame["hour_sin"] = np.sin(2 * np.pi * local_hour_fraction)
        calendar_frame["hour_cos"] = np.cos(2 * np.pi * local_hour_fraction)
        day_fraction = (hourly_frame["delivery_date"].dt.dayofyear - 1) / 365.25
        calendar_frame["season_sin"] = np.sin(2 * np.pi * day_fraction)
        calendar_frame["season_cos"] = np.cos(2 * np.pi * day_fraction)
        return calendar_frame

    def price_lag_features(self, hourly_frame):
        lag_pieces = [same_hour_lag(hourly_frame, ["da_price", "rt_price", "spread"], config.MIN_PRICE_LAG_DAYS),
                      same_hour_lag(hourly_frame, ["da_price", "rt_price", "spread"], 7),
                      daily_aggregate_lag(hourly_frame, "spread", "mean", 2, "spread_daymean_lag2"),
                      daily_aggregate_lag(hourly_frame, "da_price", "mean", 2, "da_daymean_lag2"),
                      daily_aggregate_lag(hourly_frame, "rt_price", "max", 2, "rt_daymax_lag2")]
        return pd.concat(lag_pieces, axis=1)

    def rolling_features(self, hourly_frame):
        rolling_pieces = [same_hour_rolling(hourly_frame, "spread", 7, "mean", "spread_mean7_d2"),
                          same_hour_rolling(hourly_frame, "spread", 28, "mean", "spread_mean28_d2"),
                          same_hour_rolling(hourly_frame, "spread", 28, "std", "spread_std28_d2"),
                          same_hour_rolling(hourly_frame, "spread", 28, "fraction_positive", "spread_fracpos28_d2"),
                          same_hour_rolling(hourly_frame, "spread", 28, "abs_mean", "spread_absmean28_d2"),
                          same_hour_rolling(hourly_frame, "spread", 28, "median", "spread_median28_d2"),
                          same_hour_rolling(hourly_frame, "da_price", 7, "mean", "da_mean7_d2")]
        return pd.concat(rolling_pieces, axis=1)

    def forecast_features(self, hourly_frame):
        forecast_frame = hourly_frame[["load_fcst_system_mw", "load_fcst_north_mw", "wind_fcst_mw", "solar_fcst_mw"]].copy()
        net_load = (hourly_frame["load_fcst_system_mw"] - hourly_frame["wind_fcst_mw"] - hourly_frame["solar_fcst_mw"])
        forecast_frame["net_load_mw"] = net_load
        # So, the ramp is taken in true time order (UTC), so DST days are handled. For HE01 the previous
        # hour is HE24 of D-1 from file D-1 (posted on D-2), which is known at the decision time.
        time_order = hourly_frame["interval_start_utc"].argsort().to_numpy()
        ordered_net_load = net_load.iloc[time_order]
        forecast_frame["net_load_ramp_mw"] = ordered_net_load.diff().reindex(hourly_frame.index)
        forecast_frame["net_load_ramp3_mw"] = ordered_net_load.diff(3).reindex(hourly_frame.index)
        forecast_frame["net_load_from_daily_peak_mw"] = net_load - net_load.groupby(hourly_frame["delivery_date"]).transform("max")
        # Let's do now the anomaly: day D's net load vs the same-hour mean of the forecasts for D-7..D-1
        # (all of those forecast files were posted before the decision time).
        net_load_frame = hourly_frame[["delivery_date", "hour_ending", "repeated"]].assign(net_load_mw=net_load)
        past_week_mean = same_hour_rolling(net_load_frame, "net_load_mw", 7, "mean", "net_load_mean7", end_lag_days=1)
        forecast_frame["net_load_anomaly7_mw"] = net_load - past_week_mean["net_load_mean7"]
        forecast_frame["wind_share"] = hourly_frame["wind_fcst_mw"] / hourly_frame["load_fcst_system_mw"]
        forecast_frame["solar_share"] = hourly_frame["solar_fcst_mw"] / hourly_frame["load_fcst_system_mw"]
        return forecast_frame

    def forecast_revision_features(self, hourly_frame):
        # So, revision = latest forecast (file D, 8:55 on D-1) minus the older one (file D-1, 8:55 on D-2).
        # Both are known at 9:00 on D-1. The DA market sees them too, but a revision is "new news" that
        # bidders may under-react to: our candidate for information that is not fully priced in.
        merged_vintage = hourly_frame[["interval_start_utc"]].merge(self._previous_vintage, on="interval_start_utc",
                                                                    how="left").set_axis(hourly_frame.index)
        revision_frame = pd.DataFrame(index=hourly_frame.index)
        revision_frame["load_fcst_revision_mw"] = hourly_frame["load_fcst_system_mw"] - merged_vintage["load_fcst_system_mw_prev"]
        revision_frame["wind_fcst_revision_mw"] = hourly_frame["wind_fcst_mw"] - merged_vintage["wind_fcst_mw_prev"]
        revision_frame["solar_fcst_revision_mw"] = hourly_frame["solar_fcst_mw"] - merged_vintage["solar_fcst_mw_prev"]
        revision_frame["net_load_revision_mw"] = (revision_frame["load_fcst_revision_mw"]
                                                  - revision_frame["wind_fcst_revision_mw"]
                                                  - revision_frame["solar_fcst_revision_mw"])
        return revision_frame

    # ---------------------------------------------------------------- assembly
    def _fill_within_day(self, feature_frame, hourly_frame):
        # So, a few hours have no D-2 / D-7 partner (the 2 AM hour that did not exist on a spring-forward
        # lag day, the solar mislabel in the older vintage). Fill from the neighbouring hour of the SAME
        # delivery day and of the SAME feature, which uses only information of the same vintage.
        self._fill_report = {}
        model_rows = hourly_frame["delivery_date"] >= pd.Timestamp(f"{config.TRAIN_YEAR}-01-01")
        for feature_column in feature_frame.columns:
            missing_before = int(feature_frame.loc[model_rows, feature_column].isna().sum())
            if missing_before == 0:
                continue
            filled_column = (feature_frame[feature_column].groupby(hourly_frame["delivery_date"])
                             .transform(lambda day_values: day_values.ffill().bfill()))
            feature_frame[feature_column] = filled_column
            self._fill_report[feature_column] = missing_before
        return feature_frame

    def build_feature_frame(self, hourly_frame):
        # Finally, everything in one table with the same index as hourly_frame.
        feature_frame = pd.concat([self.calendar_features(hourly_frame), self.price_lag_features(hourly_frame),
                                   self.rolling_features(hourly_frame), self.forecast_features(hourly_frame),
                                   self.forecast_revision_features(hourly_frame)], axis=1)
        feature_frame = self._fill_within_day(feature_frame[ALL_FEATURE_COLUMNS], hourly_frame)
        return feature_frame

    def build_model_dataset(self, hourly_frame):
        feature_frame = self.build_feature_frame(hourly_frame)
        model_frame = pd.concat([hourly_frame[["delivery_date", "interval_start_utc", "interval_start_local", "repeated"]],
                                 hourly_frame[TARGET_COLUMNS], feature_frame], axis=1)
        model_frame = model_frame[model_frame["delivery_date"].dt.year.isin(
            [config.TRAIN_YEAR, config.VALIDATION_YEAR, config.TEST_YEAR])].reset_index(drop=True)
        split_names = {config.TRAIN_YEAR: "train", config.VALIDATION_YEAR: "validation", config.TEST_YEAR: "test"}
        model_frame["split"] = model_frame["delivery_date"].dt.year.map(split_names)
        return model_frame


def check_every_feature_for_lookahead(feature_factory, hourly_frame):
    # So, Check 6 of project2.pdf: the WHOLE builder is poisoned and re-run, so every feature,
    # price-based or not, is tested on all 20 extended poison days, with + and - garbage.
    # One run covers every column, because the guard compares each feature column of day D separately.
    assert_no_lookahead(feature_factory.build_feature_frame, hourly_frame, be_quiet=True)
    for group_name, group_columns in FEATURE_GROUPS.items():
        price_note = "price features, D-2 or older" if group_name in PRICE_FEATURE_GROUPS else "no price input"
        print(f"  [PASS] {group_name:19s} {len(group_columns):2d} features  ({price_note})")
    return True


def feature_catalog_table():
    # So, the feature table of the report: name, group, description, whether it reads prices.
    catalog_rows = []
    for group_name, group_columns in FEATURE_GROUPS.items():
        for feature_name in group_columns:
            catalog_rows.append({"feature": feature_name, "group": group_name,
                                 "description": FEATURE_DESCRIPTIONS[feature_name],
                                 "uses_prices": group_name in PRICE_FEATURE_GROUPS,
                                 "lookahead_test": "PASS"})
    return pd.DataFrame(catalog_rows)


def plot_feature_signal(model_frame):
    # Let's do now a quick, honest look at signal: Spearman rank correlation of every feature with S,
    # on train (2023) and validation (2024). Rank correlation, because S is heavy-tailed.
    import neon_plots as neon
    signal_rows = {}
    for split_name in ("train", "validation"):
        split_rows = model_frame[model_frame["split"] == split_name]
        signal_rows[split_name] = split_rows[ALL_FEATURE_COLUMNS].corrwith(split_rows["spread"], method="spearman")
    signal_table = pd.DataFrame(signal_rows)
    signal_table["group"] = [g for g, cols in FEATURE_GROUPS.items() for _ in cols]
    signal_table = signal_table.iloc[::-1]
    neon_figure, signal_axis = neon.new_neon_canvas(1, 1, fig_width=10.5, fig_height=12.5)
    row_positions = np.arange(len(signal_table))
    signal_axis.barh(row_positions + 0.2, signal_table["train"], height=0.38, color=neon.NEON_CYAN, alpha=0.92,
                     label="2023 (train)", zorder=3)
    signal_axis.barh(row_positions - 0.2, signal_table["validation"], height=0.38, color=neon.NEON_MAGENTA, alpha=0.92,
                     label="2024 (validation)", zorder=3)
    signal_axis.set_yticks(row_positions, [f"{name}  [{grp}]" for name, grp in zip(signal_table.index, signal_table["group"])],
                           fontsize=8.5)
    signal_axis.set_ylim(-0.7, len(signal_table) - 0.3)
    neon.zero_reference_line(signal_axis, orientation="v")
    neon.label_axis(signal_axis, "Spearman Correlation Of Each Feature With The Spread S",
                    "Spearman rank correlation with S", "Feature  [group]", legend_place="lower right")
    saved_path = neon.save_neon_figure(neon_figure, "features_spearman_signal")
    signal_table.iloc[::-1].to_csv(config.OUTPUT_TABLE_DIR / "feature_spearman_signal.csv")
    return signal_table.iloc[::-1], saved_path


def run_feature_stage(use_cache=True):
    # Finally, the stage called by main.py: build, test for look-ahead, report fills, save.
    from data_pipeline import load_hourly_dataset
    print("\n******************** FEATURES: Build Every Feature Known At 9:00 AM On D-1 ********************")
    hourly_frame = load_hourly_dataset()
    previous_vintage = ForecastRevisionReader(use_cache=use_cache).get_previous_vintage()
    feature_factory = FeatureFactory(previous_vintage)
    model_frame = feature_factory.build_model_dataset(hourly_frame)
    print(f"Step-Number 1: {len(ALL_FEATURE_COLUMNS)} features in {len(FEATURE_GROUPS)} groups; "
          f"model rows by split: {model_frame['split'].value_counts().to_dict()}")
    print("Step-Number 2: Feature values filled from the same delivery day (no D-2 / D-7 partner hour):")
    for feature_name, n_filled in feature_factory.get_fill_report().items():
        print(f"    {feature_name}: {n_filled} hour(s)")
    n_missing = int(model_frame[ALL_FEATURE_COLUMNS].isna().sum().sum())
    print(f"Step-Number 3: Missing feature values left in 2023-2025: {n_missing}")
    if n_missing > 0:
        raise ValueError("Oops! Some features are still missing after the within-day fill.")
    print("Step-Number 4: Check 6, look-ahead test on the WHOLE feature builder "
          "(20 poison days, positive and negative garbage):")
    check_every_feature_for_lookahead(feature_factory, hourly_frame)
    model_frame.to_csv(config.OUTPUT_DATA_DIR / "model_dataset.csv", index=False)
    feature_catalog_table().to_csv(config.OUTPUT_TABLE_DIR / "feature_catalog.csv", index=False)
    pd.Series(feature_factory.get_fill_report(), name="hours_filled").to_csv(config.OUTPUT_TABLE_DIR / "feature_fills.csv")
    signal_table, figure_path = plot_feature_signal(model_frame)
    strongest = signal_table["validation"].abs().sort_values(ascending=False).head(8)
    print("Step-Number 5: Strongest |Spearman| with S on validation: "
          + ", ".join(f"{n} ({signal_table.loc[n, 'validation']:+.3f})" for n in strongest.index))
    print(f"Saved dataset : outputs/data/model_dataset.csv  ({model_frame.shape[0]:,} rows x {model_frame.shape[1]} columns)")
    print(f"Saved figure  : {figure_path.name}")
    print("  [PASS] Check 6 (every feature passes assert_no_lookahead)")
    return model_frame


def load_model_dataset(dataset_path=config.OUTPUT_DATA_DIR / "model_dataset.csv"):
    # So, Parts 1-3 all read the model table through this one function.
    model_frame = pd.read_csv(dataset_path, parse_dates=["delivery_date"])
    model_frame["interval_start_utc"] = pd.to_datetime(model_frame["interval_start_utc"], utc=True)
    model_frame["interval_start_local"] = model_frame["interval_start_utc"].dt.tz_convert(config.ERCOT_LOCAL_TZ)
    model_frame["repeated"] = model_frame["repeated"].astype(bool)
    return model_frame
