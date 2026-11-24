# ==============================================================================
# Name    : Anay Abhijit Joshi
# Date    : November 22, 2026
# Course  : ISyE 6767 - Interim Project 2 (Virtual Bidding at ERCOT HB_NORTH)
# File    : data_pipeline.py
# Purpose : Rebuild ercot_hb_north_hourly.csv from the RAW ERCOT files and run
#           (and report) the data quality checks 1-5 of project2.pdf.
#           Check 6 (no look-ahead) lives in lookahead_guard.py / feature_factory.py.
# ==============================================================================

import numpy as np
import pandas as pd

import config


def hour_start_utc(delivery_dates, hour_endings, repeated_flags, local_tz=config.ERCOT_LOCAL_TZ):
    # So, ERCOT labels an hour by its hour ENDING in local time: HE h covers local [h-1, h).
    # Let's do now the key: local start = date + (HE - 1) hours, then localize.
    # `ambiguous` = True means daylight time, i.e. the FIRST copy of the fall-back hour;
    # the repeated copy (flag Y) is standard time. A local hour that cannot exist becomes NaT.
    local_start = pd.to_datetime(delivery_dates) + pd.to_timedelta(np.asarray(hour_endings) - 1, unit="h")
    return (pd.DatetimeIndex(local_start)
            .tz_localize(local_tz, ambiguous=~np.asarray(repeated_flags, dtype=bool), nonexistent="NaT")
            .tz_convert("UTC"))


def load_hourly_dataset(dataset_path=config.HOURLY_DATASET_CSV):
    # Finally, the one reader every later stage uses, so the dtypes are always the same.
    hourly_frame = pd.read_csv(dataset_path, parse_dates=["delivery_date"])
    hourly_frame["interval_start_utc"] = pd.to_datetime(hourly_frame["interval_start_utc"], utc=True)
    hourly_frame["interval_start_local"] = hourly_frame["interval_start_utc"].dt.tz_convert(config.ERCOT_LOCAL_TZ)
    hourly_frame["repeated"] = hourly_frame["repeated"].astype(bool)
    return hourly_frame


class ErcotDatasetBuilder:
    """Turns the raw ERCOT workbooks and forecast CSVs into one clean hourly table."""

    def __init__(self, raw_data_dir=config.RAW_ERCOT_DIR, hub_name=config.TARGET_HUB_NAME,
                 use_cache=True, be_verbose=True):
        if not raw_data_dir.exists():
            raise FileNotFoundError(f"Oops! Raw data folder not found: {raw_data_dir}")
        self._raw_data_dir = raw_data_dir
        self._hub_name = hub_name
        self._use_cache = use_cache
        self._be_verbose = be_verbose
        self._quality_lines = []          # So, every check writes here; the report is this list.
        self._filled_value_log = []       # one dict per filled cell (Check 4)
        self._day_ahead_frame = None
        self._real_time_frame = None
        self._forecast_frames = {}
        self._hourly_frame = None
        config.ensure_output_folders()

    # ------------------------------------------------------------------ getters
    def get_hourly_frame(self):
        return self._hourly_frame

    def get_day_ahead_frame(self):
        return self._day_ahead_frame

    def get_real_time_frame(self):
        return self._real_time_frame

    def get_quality_lines(self):
        return list(self._quality_lines)

    def get_filled_value_log(self):
        return pd.DataFrame(self._filled_value_log)

    # ------------------------------------------------------------------ helpers
    def _report(self, report_line=""):
        self._quality_lines.append(report_line)
        if self._be_verbose:
            print(report_line)

    def _banner(self, banner_text):
        self._report("")
        self._report(f"******************** {banner_text} ********************")

    def _read_every_sheet(self, file_pattern):
        # So, pd.read_excel(sheet_name=None) gives a dict {month: DataFrame}; let's stack them all.
        # The calamine engine reads the ~800k-row RT workbooks about 10x faster than openpyxl.
        matching_files = sorted((self._raw_data_dir / "prices").glob(file_pattern))
        # So, every year from the first to the last delivery year must be present, otherwise we stop loudly.
        years_needed = set(range(pd.Timestamp(config.FIRST_DELIVERY_DAY).year, pd.Timestamp(config.LAST_DELIVERY_DAY).year + 1))
        years_found = {int(f.stem[-4:]) for f in matching_files if f.stem[-4:].isdigit()}
        if not years_needed.issubset(years_found):
            raise FileNotFoundError(f"Oops! {file_pattern}: missing workbook year(s) {sorted(years_needed - years_found)}")
        stacked_sheets = []
        for each_workbook in matching_files:
            sheet_dictionary = pd.read_excel(each_workbook, sheet_name=None, engine="calamine")
            if len(sheet_dictionary) != 12:
                raise ValueError(f"Oops! {each_workbook.name} has {len(sheet_dictionary)} sheets, expected 12 months.")
            stacked_sheets.extend(sheet_dictionary.values())
            if self._be_verbose:
                print(f"  read {each_workbook.name}: {len(sheet_dictionary)} sheets")
        return pd.concat(stacked_sheets, ignore_index=True)

    def _cached(self, cache_name, builder_method):
        cache_path = config.OUTPUT_CACHE_DIR / f"{cache_name}.pkl"
        if self._use_cache and cache_path.exists():
            if self._be_verbose:
                print(f"  using cache {cache_path.name} (run with --rebuild to re-read the raw workbooks)")
            return pd.read_pickle(cache_path)
        fresh_frame = builder_method()
        fresh_frame.to_pickle(cache_path)
        return fresh_frame

    # ------------------------------------------------------------------ Check 1: day-ahead
    def _build_day_ahead_from_raw(self):
        raw_dam = self._read_every_sheet("*DAMLZHBSPP_*.xlsx")
        raw_dam = raw_dam[raw_dam["Settlement Point"] == self._hub_name]
        dam_frame = pd.DataFrame({
            "delivery_date": pd.to_datetime(raw_dam["Delivery Date"], format="%m/%d/%Y"),
            "hour_ending": raw_dam["Hour Ending"].astype(str).str.split(":").str[0].astype(int),  # "01:00" or "1:00" -> 1
            "repeated": raw_dam["Repeated Hour Flag"].eq("Y"),
            "da_price": raw_dam["Settlement Point Price"].astype(float),
        })
        dam_frame["interval_start_utc"] = hour_start_utc(dam_frame["delivery_date"], dam_frame["hour_ending"],
                                                         dam_frame["repeated"])
        return dam_frame.sort_values("interval_start_utc").reset_index(drop=True)

    def load_day_ahead(self):
        self._banner("TASK 1: Day-Ahead Prices (NP4-180-ER) And One Key Per Hour")
        self._day_ahead_frame = self._cached("dam_hourly", self._build_day_ahead_from_raw)
        dam_frame = self._day_ahead_frame
        hours_per_day = dam_frame.groupby("delivery_date").size().value_counts().sort_index().to_dict()
        n_missing_key = int(dam_frame["interval_start_utc"].isna().sum())
        n_duplicate_key = int(dam_frame["interval_start_utc"].duplicated().sum())
        self._report(f"Step-Number 1: {self._hub_name} day-ahead hours read (2022-2025): {len(dam_frame):,}")
        self._report(f"Step-Number 2: Days by number of hours: {hours_per_day}  (expect 23 and 25 only on DST days)")
        self._report(f"Step-Number 3: Missing UTC keys: {n_missing_key} | Duplicate UTC keys: {n_duplicate_key}")
        odd_days = dam_frame.groupby("delivery_date").size().loc[lambda s: s != 24]
        self._report("Step-Number 4: Non-24-hour days: "
                     + ", ".join(f"{d.date()} ({n}h)" for d, n in odd_days.items()))
        n_blank_price = int(dam_frame["da_price"].isna().sum())
        self._report(f"Step-Number 5: Blank day-ahead prices: {n_blank_price}")
        if n_missing_key > 0 or n_duplicate_key > 0 or n_blank_price > 0 or set(hours_per_day) - {23, 24, 25}:
            raise ValueError("Oops! Check 1 (one key per hour) failed for the day-ahead prices.")
        self._report("  [PASS] Check 1 (one key per hour) on day-ahead prices")
        return dam_frame

    # ------------------------------------------------------------------ Check 2: real-time
    def _build_real_time_from_raw(self):
        raw_rtm = self._read_every_sheet("*RTMLZHBSPP_*.xlsx")
        raw_rtm = raw_rtm[raw_rtm["Settlement Point Name"] == self._hub_name]
        # Let's do now the 15-min -> 1-hour step: the hourly RT price is the MEAN of the 4 intervals,
        # because a 1 MWh position held over the hour settles 0.25 MWh in each 15-minute interval.
        hourly_rtm = (raw_rtm.groupby(["Delivery Date", "Delivery Hour", "Repeated Hour Flag"], as_index=False)
                      ["Settlement Point Price"].agg(rt_price="mean", rt_n_intervals="count"))
        hourly_rtm["interval_start_utc"] = hour_start_utc(
            pd.to_datetime(hourly_rtm["Delivery Date"], format="%m/%d/%Y"),
            hourly_rtm["Delivery Hour"], hourly_rtm["Repeated Hour Flag"].eq("Y"))
        return (hourly_rtm[["interval_start_utc", "rt_price", "rt_n_intervals"]]
                .sort_values("interval_start_utc").reset_index(drop=True))

    def load_real_time(self):
        self._banner("TASK 2: Real-Time Prices (NP6-785-ER), 15 Minutes -> 1 Hour")
        self._real_time_frame = self._cached("rtm_hourly", self._build_real_time_from_raw)
        rtm_frame = self._real_time_frame
        interval_counts = rtm_frame["rt_n_intervals"].value_counts().to_dict()
        keys_match_dam = set(rtm_frame["interval_start_utc"]) == set(self._day_ahead_frame["interval_start_utc"])
        self._report(f"Step-Number 1: Hourly RT rows: {len(rtm_frame):,}")
        self._report(f"Step-Number 2: Hours by number of 15-min intervals: {interval_counts}  (expect only 4)")
        self._report(f"Step-Number 3: RT hour keys identical to DA hour keys: {keys_match_dam}")
        n_blank_price = int(rtm_frame["rt_price"].isna().sum())
        self._report(f"Step-Number 4: Blank real-time prices: {n_blank_price}")
        if (rtm_frame["rt_n_intervals"] != 4).any() or not keys_match_dam or n_blank_price > 0 \
                or rtm_frame["interval_start_utc"].isna().any() or rtm_frame["interval_start_utc"].duplicated().any():
            raise ValueError("Oops! Check 2 (RT aggregation) failed.")
        self._report("  [PASS] Check 2 (every RT hour has four 15-minute intervals)")
        return rtm_frame

    # ------------------------------------------------------------------ Check 3: vintage + columns
    def verify_forecast_vintage(self):
        self._banner("TASK 3a: Forecast Vintage (Posted Before 9:00 AM On D-1)")
        posting_frame = pd.read_csv(self._raw_data_dir / "forecast_posting_times.csv",
                                    parse_dates=["delivery_date", "posted_central"])
        posting_frame["decision_time"] = (posting_frame["delivery_date"]
                                          - pd.Timedelta(days=config.DECISION_LEAD_DAYS)
                                          + pd.Timedelta(hours=config.DECISION_LOCAL_HOUR))
        posting_frame["minutes_before_decision"] = ((posting_frame["decision_time"] - posting_frame["posted_central"])
                                                    .dt.total_seconds() / 60.0)
        vintage_summary = posting_frame.groupby("report").agg(
            files=("delivery_date", "size"),
            first_day=("delivery_date", "min"), last_day=("delivery_date", "max"),
            typical_post_time=("posted_central", lambda s: s.dt.strftime("%H:%M").mode().iloc[0]),
            min_minutes_before=("minutes_before_decision", "min"),
            max_minutes_before=("minutes_before_decision", "max"),
            posted_late=("minutes_before_decision", lambda s: int((s <= 0).sum())))
        self._report(vintage_summary.to_string())
        vintage_summary.to_csv(config.OUTPUT_TABLE_DIR / "forecast_vintage_summary.csv")
        # So, the posting-time table must also cover exactly the files that are on disk.
        for product_folder in config.FORECAST_PRODUCT_SPECS:
            days_on_disk = {pd.to_datetime(f.stem, format="%Y%m%d")
                            for f in (self._raw_data_dir / product_folder).glob("2*.csv")}
            days_listed = set(posting_frame.loc[posting_frame["report"] == product_folder, "delivery_date"])
            if days_on_disk != days_listed:
                raise ValueError(f"Oops! {product_folder}: files on disk and posting times do not match.")
        if (posting_frame["minutes_before_decision"] <= 0).any():
            raise ValueError("Oops! Check 3 failed: a forecast file was posted after the decision time.")
        self._report("  [PASS] Check 3a (every one of the "
                     f"{len(posting_frame):,} forecast files was posted before 9:00 AM on D-1)")
        return vintage_summary

    def load_forecast_product(self, product_folder):
        date_column, hour_column, wanted_columns = config.FORECAST_PRODUCT_SPECS[product_folder]
        kept_pieces, rows_discarded, dropped_rows = [], 0, []
        ignored_columns = None
        for forecast_file in sorted((self._raw_data_dir / product_folder).glob("2*.csv")):
            target_day = pd.to_datetime(forecast_file.stem, format="%Y%m%d")
            raw_forecast = pd.read_csv(forecast_file, dtype={hour_column: str})
            if ignored_columns is None:
                ignored_columns = [c for c in raw_forecast.columns
                                   if c not in wanted_columns and c not in (date_column, hour_column, "DSTFlag")]
            # Let's do now the slice: the file holds the past 48 h and the next 7 days; keep day D only.
            is_day_d = pd.to_datetime(raw_forecast[date_column], format="%m/%d/%Y") == target_day
            rows_discarded += int((~is_day_d).sum())
            day_d_rows = raw_forecast[is_day_d]
            piece = pd.DataFrame({
                "delivery_date": target_day,
                "hour_ending": day_d_rows[hour_column].str.split(":").str[0].astype(int).values,  # "1:00"/"01" -> 1
                "repeated": day_d_rows["DSTFlag"].eq("Y").values,
            })
            for source_column, output_name in wanted_columns.items():
                piece[output_name] = day_d_rows[source_column].astype(float).values
            kept_pieces.append(piece)
        product_frame = pd.concat(kept_pieces, ignore_index=True)
        product_frame["interval_start_utc"] = hour_start_utc(product_frame["delivery_date"],
                                                             product_frame["hour_ending"], product_frame["repeated"])
        impossible_rows = product_frame[product_frame["interval_start_utc"].isna()]
        for _, bad_row in impossible_rows.iterrows():
            dropped_rows.append(f"{bad_row['delivery_date'].date()} HE{bad_row['hour_ending']:02d}")
        clean_product = (product_frame.dropna(subset=["interval_start_utc"])
                         [["interval_start_utc", *wanted_columns.values()]].reset_index(drop=True))
        if clean_product["interval_start_utc"].duplicated().any():
            raise ValueError(f"Oops! {product_folder}: duplicate hour keys after parsing.")
        self._report(f"  {product_folder:15s} files={len(kept_pieces):,}  kept rows (day D)={len(clean_product):,}  "
                     f"discarded rows (other days)={rows_discarded:,}")
        self._report(f"    forecast columns used : {list(wanted_columns)}")
        self._report(f"    columns NOT used      : {ignored_columns}")
        self._report(f"    impossible local hours dropped: {dropped_rows if dropped_rows else 'none'}")
        self._forecast_frames[product_folder] = clean_product
        return clean_product

    def load_all_forecasts(self):
        self._banner("TASK 3b: Forecasts Known At Decision Time (Day D Rows, Forecast Columns Only)")
        for product_folder in config.FORECAST_PRODUCT_SPECS:
            self.load_forecast_product(product_folder)
        self._report("  Note: ACTUAL_* are realized generation, observed after the fact (blank for day D, filled for")
        self._report("  the past 48 h); COP_HSL comes from the resources' current operating plans and *GRPP is a")
        self._report("  probability-of-exceedance production level. None of them is the requested forecast, so none")
        self._report("  of them enters the dataset; the realized ACTUAL values would be look-ahead by construction.")
        self._report("  [PASS] Check 3b (only SystemTotal, North, STWPF_SYSTEM_WIDE, STPPF_SYSTEM_WIDE kept)")
        return self._forecast_frames

    # ------------------------------------------------------------------ Check 4: merge + fill
    def _fill_missing_forecasts(self, merged_frame):
        forecast_columns = [n for spec in config.FORECAST_PRODUCT_SPECS.values() for n in spec[2].values()]
        for forecast_column in forecast_columns:
            missing_positions = np.flatnonzero(merged_frame[forecast_column].isna().to_numpy())
            for row_position in missing_positions:
                # So, neighbours are taken ONLY from the same delivery day, i.e. from the same forecast file.
                # The next day's first hour lives in the NEXT day's file, posted after this day's decision
                # time, so using it would be look-ahead. A missing HE01 or HE24 uses its one same-day neighbour.
                this_day = merged_frame["delivery_date"].iloc[row_position]
                neighbour_values = []
                for neighbour_position in (row_position - 1, row_position + 1):
                    if 0 <= neighbour_position < len(merged_frame) \
                            and merged_frame["delivery_date"].iloc[neighbour_position] == this_day:
                        neighbour_values.append(merged_frame[forecast_column].iloc[neighbour_position])
                neighbour_values = [v for v in neighbour_values if pd.notna(v)]
                if len(neighbour_values) == 0:
                    raise ValueError(f"Oops! {forecast_column} on {this_day.date()}: no same-day neighbour to fill from.")
                if forecast_column == "solar_fcst_mw" and len(neighbour_values) == 2 and max(neighbour_values) == 0:
                    # Let's do now the night case: between two zero-solar hours, zero is the only defensible value.
                    filled_value, fill_rule = 0.0, "zero (night hour, both neighbouring hours are 0 MW)"
                else:
                    # In my opinion the mean of the same-day neighbours is the safest default.
                    filled_value = float(np.mean(neighbour_values))
                    fill_rule = f"mean of {len(neighbour_values)} same-day neighbouring hour(s)"
                merged_frame.iloc[row_position, merged_frame.columns.get_loc(forecast_column)] = filled_value
                local_stamp = merged_frame["interval_start_local"].iloc[row_position]
                self._filled_value_log.append({
                    "column": forecast_column,
                    "delivery_date": merged_frame["delivery_date"].iloc[row_position].date(),
                    "hour_ending": int(merged_frame["hour_ending"].iloc[row_position]),
                    "interval_start_local": local_stamp.strftime("%Y-%m-%d %H:%M %Z"),
                    "filled_value": filled_value, "rule": fill_rule})
        return merged_frame

    def build_hourly_table(self):
        self._banner("TASK 4: One Table (Merge On interval_start_utc) And Completeness")
        delivery_rows = self._day_ahead_frame[self._day_ahead_frame["delivery_date"] >= config.FIRST_DELIVERY_DAY].copy()
        merged_frame = delivery_rows.merge(self._real_time_frame, on="interval_start_utc", how="left")
        for product_folder in config.FORECAST_PRODUCT_SPECS:
            merged_frame = merged_frame.merge(self._forecast_frames[product_folder], on="interval_start_utc", how="left")
        merged_frame["interval_start_local"] = merged_frame["interval_start_utc"].dt.tz_convert(config.ERCOT_LOCAL_TZ)
        missing_before = merged_frame.isna().sum()
        self._report(f"Step-Number 1: Delivery hours {config.FIRST_DELIVERY_DAY} .. {config.LAST_DELIVERY_DAY}: "
                     f"{len(merged_frame):,}")
        self._report("Step-Number 2: Missing values per column after the merge: "
                     + str(missing_before[missing_before > 0].to_dict()))
        merged_frame = self._fill_missing_forecasts(merged_frame)
        self._report("Step-Number 3: Every filled value and how it was filled:")
        for fill_entry in self._filled_value_log:
            self._report(f"    {fill_entry['column']} | {fill_entry['delivery_date']} HE{fill_entry['hour_ending']:02d} "
                         f"({fill_entry['interval_start_local']}) -> {fill_entry['filled_value']:.1f} MW | {fill_entry['rule']}")
        self._report("    Why: on spring-forward days the solar report skips HE02 and instead lists an HE03 that")
        self._report("    cannot exist (02:00-03:00 never happens); load and wind skip HE03 correctly. The impossible")
        self._report("    row was dropped in Task 3 and the real HE02 (01:00-02:00 CST, night) is filled with 0 MW.")
        self.get_filled_value_log().to_csv(config.OUTPUT_TABLE_DIR / "filled_values.csv", index=False)

        merged_frame["spread"] = merged_frame["da_price"] - merged_frame["rt_price"]
        merged_frame["y"] = (merged_frame["spread"] > 0).astype(int)
        ordered_columns = ["delivery_date", "hour_ending", "repeated", "interval_start_utc", "interval_start_local",
                           "da_price", "rt_price", "rt_n_intervals", "load_fcst_system_mw", "load_fcst_north_mw",
                           "wind_fcst_mw", "solar_fcst_mw", "spread", "y"]
        merged_frame = merged_frame[ordered_columns].sort_values("interval_start_utc").reset_index(drop=True)
        missing_after = int(merged_frame.isna().sum().sum())
        self._report(f"Step-Number 4: Final shape {merged_frame.shape}, missing values left: {missing_after}, "
                     f"UTC key strictly increasing: {merged_frame['interval_start_utc'].is_monotonic_increasing}")
        if missing_after > 0 or not merged_frame["interval_start_utc"].is_monotonic_increasing:
            raise ValueError("Oops! Check 4 (completeness) failed.")
        self._report("  [PASS] Check 4 (no missing values, every fill reported)")
        self._hourly_frame = merged_frame
        return merged_frame

    # ------------------------------------------------------------------ Check 5: outliers
    def report_outliers(self):
        self._banner("CHECK 5: Extreme And Negative Prices (Reported, NOT Removed)")
        hourly_frame = self._hourly_frame
        delivery_year = hourly_frame["delivery_date"].dt.year
        yearly_table = hourly_frame.groupby(delivery_year).agg(
            hours=("spread", "size"),
            da_min=("da_price", "min"), da_max=("da_price", "max"),
            rt_min=("rt_price", "min"), rt_max=("rt_price", "max"),
            negative_da_hours=("da_price", lambda s: int((s < 0).sum())),
            negative_rt_hours=("rt_price", lambda s: int((s < 0).sum())),
            spread_abs_over_100=("spread", lambda s: int((s.abs() > 100).sum())),
            spread_abs_over_500=("spread", lambda s: int((s.abs() > 500).sum())))
        yearly_table.index.name = "year"
        self._report(yearly_table.round(2).to_string())
        yearly_table.to_csv(config.OUTPUT_TABLE_DIR / "outliers_by_year.csv")

        spread_quantiles = hourly_frame["spread"].quantile([0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999])
        self._report("Spread quantiles ($/MWh): "
                     + ", ".join(f"q{q*100:g}={v:,.2f}" for q, v in spread_quantiles.items()))
        extreme_rows = hourly_frame.loc[hourly_frame["spread"].abs().nlargest(10).index,
                                        ["interval_start_local", "hour_ending", "da_price", "rt_price", "spread"]]
        extreme_rows = extreme_rows.assign(interval_start_local=extreme_rows["interval_start_local"]
                                           .dt.strftime("%Y-%m-%d %H:%M %Z"))
        self._report("Ten largest |spread| hours:")
        self._report(extreme_rows.round(2).to_string(index=False))
        extreme_rows.to_csv(config.OUTPUT_TABLE_DIR / "top10_abs_spread_hours.csv", index=False)
        self._report("  Decision: these are real market events (scarcity, Winter Storm Elliott, high renewable")
        self._report("  output), so they stay in the dataset untouched. Robustness is handled inside the models")
        self._report("  (Huber / absolute-error losses, rank-based features) and is stated there.")
        self._report("  [PASS] Check 5 (extreme and negative prices reported, none deleted or clipped)")
        return yearly_table

    # ------------------------------------------------------------------ Check 6 demo
    def demonstrate_lookahead_test(self):
        from lookahead_guard import passes_lookahead, wrong_features_for_demo
        self._banner("CHECK 6: The Look-Ahead Test Catches The Two Classic Mistakes")
        did_pass, failure_message = passes_lookahead(wrong_features_for_demo, self._hourly_frame)
        self._report(f"  shift(24) and unshifted rolling(24*7) pass the test? {did_pass}")
        self._report(f"  caught: {failure_message}")
        if did_pass:
            raise ValueError("Oops! The look-ahead test failed to catch a known leaking feature.")
        self._report("  [PASS] The guard works; every project feature is run through it in feature_factory.py")

    # ------------------------------------------------------------------ orchestration
    def save_outputs(self):
        self._hourly_frame.to_csv(config.HOURLY_DATASET_CSV, index=False)
        self._report("")
        self._report(f"Saved dataset : {config.HOURLY_DATASET_CSV.relative_to(config.PROJECT_HOME_DIR)}")
        self._report(f"Saved report  : {config.DATA_QUALITY_REPORT_TXT.relative_to(config.PROJECT_HOME_DIR)}")
        config.DATA_QUALITY_REPORT_TXT.write_text("\n".join(self._quality_lines) + "\n")

    def run_full_pipeline(self):
        # Finally, the six checks in the order of project2.pdf, then save everything.
        self.load_day_ahead()
        self.load_real_time()
        self.verify_forecast_vintage()
        self.load_all_forecasts()
        self.build_hourly_table()
        self.report_outliers()
        self.demonstrate_lookahead_test()
        self.save_outputs()
        return self._hourly_frame
