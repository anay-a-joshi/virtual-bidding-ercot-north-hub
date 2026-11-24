# ==============================================================================
# Name    : Anay Abhijit Joshi
# Date    : November 22, 2026
# Course  : ISyE 6767 - Interim Project 2 (Virtual Bidding at ERCOT HB_NORTH)
# File    : data_sanity.py
# Purpose : Lab 2, Task 6 sanity checks: class balance, prices and spread over
#           time, the most extreme hours, negative prices, and the spread by
#           hour of day (does HB_NORTH carry a persistent DA premium?).
# ==============================================================================

import numpy as np
import pandas as pd

import config
import neon_plots as neon

SPLIT_YEAR_LABELS = {config.TRAIN_YEAR: "2023 (train)", config.VALIDATION_YEAR: "2024 (validation)",
                     config.TEST_YEAR: "2025 (test)"}


class DataSanityInspector:
    """Runs and plots every sanity check that must be looked at before any model is fit."""

    def __init__(self, hourly_frame, be_verbose=True):
        if "spread" not in hourly_frame or "y" not in hourly_frame:
            raise ValueError("Oops! The hourly frame needs the spread and y columns.")
        self._hourly_frame = hourly_frame.copy()
        self._hourly_frame["delivery_year"] = self._hourly_frame["delivery_date"].dt.year
        # So, December 2022 only feeds the lags; the sanity checks are about the three modelling years.
        self._model_years_frame = self._hourly_frame[self._hourly_frame["delivery_year"].isin(SPLIT_YEAR_LABELS)]
        self._be_verbose = be_verbose
        config.ensure_output_folders()

    def _say(self, message=""):
        if self._be_verbose:
            print(message)

    # ---------------------------------------------------------------- 1. class balance
    def class_balance_by_year(self):
        self._say("\n******************** TASK 6.1: Class Balance Per Year (= Accuracy Of Always INC) ********************")
        balance_table = self._model_years_frame.groupby("delivery_year").agg(
            hours=("y", "size"), share_positive=("y", "mean"),
            mean_spread=("spread", "mean"), median_spread=("spread", "median"), std_spread=("spread", "std"))
        balance_table["always_inc_accuracy"] = balance_table["share_positive"]
        balance_table["always_dec_accuracy"] = 1.0 - balance_table["share_positive"]
        balance_table.index.name = "year"
        self._say(balance_table.round(4).to_string())
        balance_table.to_csv(config.OUTPUT_TABLE_DIR / "class_balance_by_year.csv")
        return balance_table

    # ---------------------------------------------------------------- 2. prices and spread over time
    def plot_prices_and_spread_over_time(self, n_extreme_hours=5):
        self._say("\n******************** TASK 6.2: Prices And Spread Over Time, Largest |S| Hours ********************")
        frame = self._model_years_frame
        daily_frame = frame.groupby("delivery_date")[["da_price", "rt_price"]].mean()
        top_rows = frame.loc[frame["spread"].abs().nlargest(n_extreme_hours).index]

        neon_figure, (price_axis, spread_axis) = neon.new_neon_canvas(2, 1, fig_width=12, fig_height=8.2, share_x=True)
        neon.glow_line(price_axis, daily_frame.index, daily_frame["da_price"], neon.NEON_CYAN, "Day-ahead (daily mean)", 1.3)
        neon.glow_line(price_axis, daily_frame.index, daily_frame["rt_price"], neon.NEON_MAGENTA, "Real-time (daily mean)", 1.0,
                       line_alpha=0.85)
        neon.symlog_ticks_dollars(price_axis, linear_threshold=10.0, lowest_value=-30, highest_value=3000,
                                  axis_limits=(-20, 2000))
        neon.label_axis(price_axis, "HB_NORTH Day-Ahead And Real-Time Prices (Daily Mean)", "",
                        "Price ($/MWh, symlog scale)", legend_place="upper right")

        local_time = frame["interval_start_local"].dt.tz_localize(None)
        spread_axis.plot(local_time, frame["spread"], color=neon.NEON_LIME, linewidth=0.5, alpha=0.85,
                         label="Hourly spread S = DA - RT", zorder=3)
        neon.zero_reference_line(spread_axis)
        neon.glow_scatter(spread_axis, top_rows["interval_start_local"].dt.tz_localize(None), top_rows["spread"],
                          neon.NEON_ORANGE, f"{n_extreme_hours} largest |S| hours", dot_size=46)
        neon.symlog_ticks_dollars(spread_axis, linear_threshold=10.0, axis_limits=(-6000, 6000))
        for split_year, split_label in SPLIT_YEAR_LABELS.items():
            for each_axis in (price_axis, spread_axis):
                each_axis.axvspan(pd.Timestamp(f"{split_year}-01-01"), pd.Timestamp(f"{split_year}-12-31 23:00"),
                                  color=neon.neon_color([2023, 2024, 2025].index(split_year)), alpha=0.045, zorder=0)
            spread_axis.text(pd.Timestamp(f"{split_year}-07-01"), 5200, split_label, ha="center", va="top",
                             color=neon.NEON_MUTED_INK, fontsize=10)
        neon.format_date_axis(spread_axis, month_step=3)
        neon.label_axis(spread_axis, "Hourly DART Spread With The Five Largest |S| Hours", "Delivery time (Central)",
                        "Spread ($/MWh, symlog scale)", legend_place="lower right")
        saved_path = neon.save_neon_figure(neon_figure, "sanity_prices_and_spread_over_time")

        # So, "errors or real events?": a data error is usually an isolated single hour; a real event is
        # a cluster of large spreads in neighbouring hours with DA and RT both elevated.
        extreme_table = top_rows[["interval_start_local", "hour_ending", "da_price", "rt_price", "spread"]].copy()
        same_day_big = frame.assign(is_big=frame["spread"].abs() > 100).groupby("delivery_date")["is_big"].sum()
        extreme_table["hours_same_day_with_abs_S_over_100"] = (
            top_rows["delivery_date"].map(same_day_big).astype(int).values)
        extreme_table["interval_start_local"] = extreme_table["interval_start_local"].dt.strftime("%Y-%m-%d %H:%M %Z")
        self._say(extreme_table.round(2).to_string(index=False))
        extreme_table.to_csv(config.OUTPUT_TABLE_DIR / "top5_abs_spread_hours.csv", index=False)
        self._say(f"  figure -> {saved_path.name}")
        return extreme_table

    # ---------------------------------------------------------------- 3. negative prices
    def negative_price_summary(self):
        self._say("\n******************** TASK 6.3: Negative Prices (Kept In The Data) ********************")
        frame = self._model_years_frame
        negative_table = frame.groupby("delivery_year").agg(
            negative_da_hours=("da_price", lambda s: int((s < 0).sum())),
            negative_rt_hours=("rt_price", lambda s: int((s < 0).sum())),
            lowest_da=("da_price", "min"), lowest_rt=("rt_price", "min"))
        negative_table.index.name = "year"
        self._say(negative_table.round(2).to_string())
        negative_rt_by_hour = (frame[frame["rt_price"] < 0].groupby("hour_ending").size()
                               .reindex(range(1, 25), fill_value=0))
        self._say("  Negative RT hours by hour ending: " + ", ".join(f"HE{h}:{n}" for h, n in negative_rt_by_hour.items() if n))
        negative_table.to_csv(config.OUTPUT_TABLE_DIR / "negative_prices_by_year.csv")

        neon_figure, (year_axis, hour_axis) = neon.new_neon_canvas(1, 2, fig_width=12.5, fig_height=4.8)
        year_positions = np.arange(len(negative_table))
        neon.glow_bars(year_axis, year_positions - 0.2, negative_table["negative_da_hours"], neon.NEON_CYAN,
                       "Day-ahead", bar_width=0.38)
        neon.glow_bars(year_axis, year_positions + 0.2, negative_table["negative_rt_hours"], neon.NEON_MAGENTA,
                       "Real-time", bar_width=0.38)
        for x_spot, (da_count, rt_count) in zip(year_positions, negative_table[["negative_da_hours", "negative_rt_hours"]].values):
            year_axis.text(x_spot - 0.2, da_count + 3, f"{da_count}", ha="center", color=neon.NEON_TEXT_INK, fontsize=9)
            year_axis.text(x_spot + 0.2, rt_count + 3, f"{rt_count}", ha="center", color=neon.NEON_TEXT_INK, fontsize=9)
        year_axis.set_xticks(year_positions, [SPLIT_YEAR_LABELS[y] for y in negative_table.index])
        neon.label_axis(year_axis, "Hours With A Negative Price", "Delivery year", "Number of hours")
        neon.glow_bars(hour_axis, negative_rt_by_hour.index, negative_rt_by_hour.values, neon.NEON_MAGENTA,
                       "Real-time, 2023-2025", bar_width=0.75)
        hour_axis.set_xticks(range(1, 25, 2))
        neon.label_axis(hour_axis, "Negative Real-Time Hours By Hour Of Day", "Hour ending (Central)",
                        "Number of hours")
        saved_path = neon.save_neon_figure(neon_figure, "sanity_negative_prices")
        self._say(f"  figure -> {saved_path.name}")
        return negative_table

    # ---------------------------------------------------------------- 4. by hour of day
    def spread_by_hour_of_day(self):
        self._say("\n******************** TASK 6.4: Spread By Hour Of Day (Persistent DA Premium?) ********************")
        frame = self._model_years_frame
        hourly_profile = (frame.groupby(["delivery_year", "hour_ending"])
                          .agg(mean_spread=("spread", "mean"), median_spread=("spread", "median"),
                               share_positive=("y", "mean"))
                          .reset_index())
        hourly_profile.to_csv(config.OUTPUT_TABLE_DIR / "spread_by_hour_of_day.csv", index=False)
        wide_share = hourly_profile.pivot(index="hour_ending", columns="delivery_year", values="share_positive")
        self._say("  Share of positive spreads by hour ending (rows) and year (columns):")
        self._say(wide_share.round(3).T.to_string())
        hours_majority_positive = (wide_share > 0.5).sum()
        self._say("  Hours (of 24) where S > 0 in the majority of days: " + hours_majority_positive.to_string())

        neon_figure, (mean_axis, median_axis, share_axis) = neon.new_neon_canvas(1, 3, fig_width=16, fig_height=5.0)
        for series_position, split_year in enumerate(SPLIT_YEAR_LABELS):
            one_year = hourly_profile[hourly_profile["delivery_year"] == split_year]
            year_color = neon.neon_color(series_position)
            neon.glow_line(mean_axis, one_year["hour_ending"], one_year["mean_spread"], year_color,
                           SPLIT_YEAR_LABELS[split_year], 1.8)
            neon.glow_line(median_axis, one_year["hour_ending"], one_year["median_spread"], year_color,
                           SPLIT_YEAR_LABELS[split_year], 1.8)
            neon.glow_line(share_axis, one_year["hour_ending"], one_year["share_positive"], year_color,
                           SPLIT_YEAR_LABELS[split_year], 1.8)
        for each_axis in (mean_axis, median_axis):
            neon.zero_reference_line(each_axis)
            each_axis.set_xticks(range(1, 25, 2))
        neon.zero_reference_line(share_axis, reference_value=0.5)
        share_axis.set_xticks(range(1, 25, 2))
        share_axis.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        neon.label_axis(mean_axis, "Mean Spread By Hour (Spike-Driven)", "Hour ending (Central)", "Mean spread ($/MWh)")
        neon.label_axis(median_axis, "Median Spread By Hour (Typical Hour)", "Hour ending (Central)",
                        "Median spread ($/MWh)")
        neon.label_axis(share_axis, "Share Of Hours With S > 0", "Hour ending (Central)", "Share positive (P(y = 1))")
        saved_path = neon.save_neon_figure(neon_figure, "sanity_spread_by_hour_of_day")
        self._say(f"  figure -> {saved_path.name}")
        return hourly_profile

    # ---------------------------------------------------------------- 5. distribution (heavy tails)
    def plot_spread_distribution(self):
        self._say("\n******************** TASK 6.5: Spread Distribution (Heavy Tails) ********************")
        frame = self._model_years_frame
        # So, bins are uniform in asinh(S): linear near 0 and logarithmic in the tails, exactly like the
        # symlog axis, so every bin has the same visual width and no artificial spike appears at zero.
        asinh_edges = np.linspace(np.arcsinh(-4500.0), np.arcsinh(4500.0), 121)
        bin_edges = np.sinh(asinh_edges)
        neon_figure, density_axis = neon.new_neon_canvas(1, 1, fig_width=11, fig_height=5.0)
        for series_position, split_year in enumerate(SPLIT_YEAR_LABELS):
            year_spread = frame.loc[frame["delivery_year"] == split_year, "spread"]
            bin_counts, _ = np.histogram(year_spread, bins=bin_edges)
            bin_centres = np.sinh(0.5 * (asinh_edges[:-1] + asinh_edges[1:]))
            neon.glow_line(density_axis, bin_centres, bin_counts / len(year_spread), neon.neon_color(series_position),
                           f"{SPLIT_YEAR_LABELS[split_year]}: std = ${year_spread.std():,.0f}", 1.8)
        density_axis.set_xscale("symlog", linthresh=1.0)
        density_axis.set_yscale("log")
        neon.plain_log_axis(density_axis, "y")
        x_ticks = [-1000, -100, -10, -1, 0, 1, 10, 100, 1000]
        density_axis.set_xticks(x_ticks, [f"{t:,}" for t in x_ticks])
        neon.zero_reference_line(density_axis, orientation="v")
        neon.label_axis(density_axis, "Distribution Of The Hourly Spread By Year", "Spread S ($/MWh, symlog scale)",
                        "Share of hours per bin (log scale)", legend_place="upper right")
        saved_path = neon.save_neon_figure(neon_figure, "sanity_spread_distribution")
        self._say(f"  figure -> {saved_path.name}")

    def run_all_checks(self):
        # Finally, everything Task 6 asks for, in order.
        balance_table = self.class_balance_by_year()
        self.plot_prices_and_spread_over_time()
        self.negative_price_summary()
        self.spread_by_hour_of_day()
        self.plot_spread_distribution()
        return balance_table
