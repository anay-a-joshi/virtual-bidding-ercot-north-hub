# ==============================================================================
# Name    : Anay Abhijit Joshi
# Date    : November 22, 2026
# Course  : ISyE 6767 - Interim Project 2 (Virtual Bidding at ERCOT HB_NORTH)
# File    : run_part3.py
# Purpose : Part 3 (trading budget allocation): every rule at the same budget
#           (24 MWh/day, <= 4 MWh/hour), the knapsack questions, the risk-aware
#           rules (mean-variance and CVaR), and the required metrics: total
#           profit, profit/MWh, average budget used, max drawdown, 5 worst days.
# ==============================================================================

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

import config
import neon_plots as neon
from backtest_engine import StrategyBook, capital_base_dollars, metrics_table
from budget_allocator import (bounded_knapsack_dp, cvar_limited_allocation, mean_variance_allocation,
                              ratio_greedy_knapsack, unit_knapsack_greedy, water_fill_allocation)
from data_pipeline import load_hourly_dataset
from feature_factory import load_model_dataset
from model_suite import PredictionStore

SIGMA_FLOOR_DOLLARS = 1.0
RULE_LABELS = {"A_equal": "A: equal weights", "A_edge": "A: w = |mu|-c", "A_inv_sigma": "A: w = 1/sigma",
               "A_edge_over_var": "A: w = (|mu|-c)/sigma^2", "B_knapsack": "B: knapsack (unit weights)",
               "Q2_collateral_dp": "Q2: collateral knapsack (DP)", "Q2_collateral_greedy": "Q2: collateral, ratio greedy",
               "Q3_mean_variance": "Q3: mean-variance", "Q3_cvar": "Q3: CVaR limit"}


class RiskHistory:
    """Covariance and loss scenarios for day D, built ONLY from the 56 days that end on D-2."""

    def __init__(self, hourly_frame, lookback_days=config.RISK_LOOKBACK_DAYS):
        first_copies = hourly_frame.loc[~hourly_frame["repeated"], ["delivery_date", "hour_ending", "spread"]]
        day_by_hour = first_copies.pivot(index="delivery_date", columns="hour_ending", values="spread")
        day_by_hour = day_by_hour.reindex(columns=range(1, 25))
        # So, the hour that does not exist on spring-forward days takes the previous hour's value.
        self._day_by_hour = day_by_hour.ffill(axis=1).bfill(axis=1)
        self._lookback_days = lookback_days
        self._cache = {}

    def for_day(self, delivery_day):
        if delivery_day not in self._cache:
            window_end = delivery_day - pd.Timedelta(days=config.MIN_PRICE_LAG_DAYS)
            window_start = window_end - pd.Timedelta(days=self._lookback_days - 1)
            history = self._day_by_hour.loc[window_start:window_end].to_numpy()
            if len(history) < self._lookback_days * 0.9:
                raise ValueError(f"Oops! Not enough history before {delivery_day.date()} for the risk model.")
            shrunk_covariance = LedoitWolf().fit(history).covariance_
            self._cache[delivery_day] = (shrunk_covariance, history)
        return self._cache[delivery_day]


def allocate_one_day(rule_name, mu_hat, sigma_hat, hour_endings, cost_value, risk_history, delivery_day,
                     collateral_per_mwh, collateral_budget, rule_parameter=None):
    directions = np.sign(mu_hat)
    net_edges = np.where(directions != 0, np.abs(mu_hat) - cost_value, -cost_value)
    positive_edge = np.clip(net_edges, 0.0, None)
    is_tradable = (directions != 0).astype(float)
    if rule_name == "A_equal":
        volumes = water_fill_allocation(is_tradable)
    elif rule_name == "A_edge":
        volumes = water_fill_allocation(positive_edge)
    elif rule_name == "A_inv_sigma":
        volumes = water_fill_allocation(is_tradable / sigma_hat)
    elif rule_name == "A_edge_over_var":
        volumes = water_fill_allocation(positive_edge / sigma_hat ** 2)
    elif rule_name == "B_knapsack":
        volumes = unit_knapsack_greedy(net_edges)
    elif rule_name == "Q2_collateral_dp":
        volumes, _ = bounded_knapsack_dp(net_edges, collateral_per_mwh, collateral_budget)
    elif rule_name == "Q2_collateral_greedy":
        volumes = ratio_greedy_knapsack(net_edges, collateral_per_mwh, collateral_budget)
    elif rule_name == "Q3_mean_variance":
        covariance, _ = risk_history.for_day(delivery_day)
        column_index = np.asarray(hour_endings) - 1
        volumes = mean_variance_allocation(net_edges, directions, covariance[np.ix_(column_index, column_index)],
                                           rule_parameter)
    elif rule_name == "Q3_cvar":
        _, history = risk_history.for_day(delivery_day)
        volumes = cvar_limited_allocation(net_edges, directions, history[:, np.asarray(hour_endings) - 1], cost_value,
                                          rule_parameter)
    else:
        raise ValueError(f"Oops! Unknown allocation rule {rule_name}.")
    volumes = np.where(directions != 0, volumes, 0.0)
    # Finally, floor to the 0.1 MWh increment (flooring can only lower the total, so the budget still holds).
    increment = config.VOLUME_INCREMENT_MWH
    volumes = np.floor(volumes / increment + 1e-9) * increment
    return directions, volumes


class BudgetAllocationRunner:
    """Runs every allocation rule day by day on a split and turns the result into StrategyBooks."""

    def __init__(self, model_frame, prediction_frame, primary_view_b, risk_history, collateral_budget):
        self._model_frame = model_frame
        self._prediction_frame = prediction_frame
        self._primary_view_b = primary_view_b
        self._risk_history = risk_history
        self._collateral_budget = collateral_budget

    def run_rule(self, split_name, rule_name, cost_value, rule_parameter=None, book_name=None):
        split_mask = (self._model_frame["split"] == split_name).to_numpy()
        split_rows = self._model_frame[split_mask]
        mu_hat_all = self._prediction_frame.loc[split_mask, self._primary_view_b].to_numpy()
        sigma_all = np.maximum(split_rows["spread_std28_d2"].to_numpy(), SIGMA_FLOOR_DOLLARS)
        collateral_all = np.ceil(config.COLLATERAL_SIGMA_MULTIPLE * sigma_all).astype(int)
        actions = np.zeros(len(split_rows))
        volumes = np.zeros(len(split_rows))
        for delivery_day, day_positions in split_rows.groupby("delivery_date").indices.items():
            day_actions, day_volumes = allocate_one_day(
                rule_name, mu_hat_all[day_positions], sigma_all[day_positions],
                split_rows["hour_ending"].to_numpy()[day_positions], cost_value, self._risk_history, delivery_day,
                collateral_all[day_positions], self._collateral_budget, rule_parameter)
            actions[day_positions], volumes[day_positions] = day_actions, day_volumes
        actions = np.where(volumes > 0, actions, 0.0)
        name = book_name or f"part3_{rule_name}_c{int(cost_value)}"
        return StrategyBook(name, split_rows, actions, volumes, mu_hat_all, "mu_hat", part_label="Part 3",
                            cost_scope=cost_value)

    def knapsack_comparison(self, split_name, cost_value):
        # So, Question 2: how often (and by how much) does the ratio greedy miss the DP optimum when the
        # items have DIFFERENT weights? And with unit weights, is greedy always equal to the DP?
        split_mask = (self._model_frame["split"] == split_name).to_numpy()
        split_rows = self._model_frame[split_mask]
        mu_hat_all = self._prediction_frame.loc[split_mask, self._primary_view_b].to_numpy()
        sigma_all = np.maximum(split_rows["spread_std28_d2"].to_numpy(), SIGMA_FLOOR_DOLLARS)
        collateral_all = np.ceil(config.COLLATERAL_SIGMA_MULTIPLE * sigma_all).astype(int)
        comparison_rows, worked_example = [], None
        for delivery_day, day_positions in split_rows.groupby("delivery_date").indices.items():
            mu_hat = mu_hat_all[day_positions]
            net_edges = np.where(mu_hat != 0, np.abs(mu_hat) - cost_value, -cost_value)
            unit_greedy = unit_knapsack_greedy(net_edges)
            _, unit_dp_value = bounded_knapsack_dp(net_edges, np.ones(len(net_edges), dtype=int), config.DAILY_BUDGET_MWH)
            dp_volumes, dp_value = bounded_knapsack_dp(net_edges, collateral_all[day_positions], self._collateral_budget)
            greedy_volumes = ratio_greedy_knapsack(net_edges, collateral_all[day_positions], self._collateral_budget)
            greedy_value = float(np.dot(np.clip(net_edges, 0, None), greedy_volumes))
            comparison_rows.append({"delivery_date": delivery_day,
                                    "unit_greedy_value": float(np.dot(np.clip(net_edges, 0, None), unit_greedy)),
                                    "unit_dp_value": unit_dp_value, "collateral_dp_value": dp_value,
                                    "collateral_greedy_value": greedy_value, "gap": dp_value - greedy_value})
            if worked_example is None and dp_value - greedy_value > 1.0:
                worked_example = pd.DataFrame({
                    "delivery_date": delivery_day.date(), "hour_ending": split_rows["hour_ending"].to_numpy()[day_positions],
                    "mu_hat": mu_hat, "net_edge_per_mwh": net_edges, "collateral_per_mwh": collateral_all[day_positions],
                    "edge_per_collateral_dollar": net_edges / collateral_all[day_positions],
                    "greedy_mwh": greedy_volumes, "dp_mwh": dp_volumes})
        return pd.DataFrame(comparison_rows), worked_example


def plot_part3_figures(test_books_by_cost, tuning_table, knapsack_table, chosen):
    for cost_value, books in test_books_by_cost.items():
        by_rule = {b.get_name().replace("part3_", "").rsplit("_c", 1)[0]: b for b in books}
        panels = [[(RULE_LABELS[r], by_rule[r].daily_profit(cost_value)) for r in ("A_equal", "A_edge", "A_inv_sigma",
                                                                                   "A_edge_over_var")],
                  [(RULE_LABELS[r], by_rule[r].daily_profit(cost_value)) for r in ("Q2_collateral_dp",
                                                                                   "Q2_collateral_greedy")],
                  [(RULE_LABELS[r], by_rule[r].daily_profit(cost_value)) for r in ("Q3_mean_variance", "Q3_cvar")]]
        neon.cumulative_profit_panels(panels, ["Approach A: Scale Weights To B", "Knapsacks", "Risk-Aware (Q3)"],
                                      f"Part 3: Cumulative Profit On 2025, Budget 24 MWh/Day (c = ${cost_value:.0f}/MWh)",
                                      f"part3_cumulative_profit_test_c{int(cost_value)}",
                                      reference_series=by_rule["B_knapsack"].daily_profit(cost_value),
                                      reference_label="B: knapsack (unit weights)")
    # So, where does each rule put its MWh during the day (2025, c = $1)?
    books = {b.get_name().replace("part3_", "").rsplit("_c", 1)[0]: b for b in test_books_by_cost[1.0]}
    neon_figure, (left_axis, right_axis) = neon.new_neon_canvas(1, 2, fig_width=15, fig_height=5.0)
    for panel_axis, rule_names in ((left_axis, ("A_equal", "A_edge", "A_inv_sigma", "A_edge_over_var")),
                                   (right_axis, ("B_knapsack", "Q2_collateral_dp", "Q3_mean_variance", "Q3_cvar"))):
        for series_position, rule_name in enumerate(rule_names):
            trade_frame = books[rule_name].get_trade_frame()
            n_days = trade_frame["delivery_date"].nunique()
            profile = trade_frame.groupby("hour_ending")["volume_mwh"].sum() / n_days
            neon.glow_line(panel_axis, profile.index, profile.values, neon.neon_color(series_position),
                           RULE_LABELS[rule_name], 1.8)
        panel_axis.set_xticks(range(1, 25, 2))
        neon.label_axis(panel_axis, "Average MWh Allocated By Hour (2025, c = $1)", "Hour ending (Central)",
                        "Average MWh per day", legend_place="upper left")
    neon.save_neon_figure(neon_figure, "part3_allocation_by_hour")
    # Let's do now the validation choice of lambda and the CVaR limit (chosen by 2024 Sharpe ratio).
    neon_figure, axis_grid = neon.new_neon_canvas(2, 2, fig_width=14, fig_height=9.0)
    for row_position, (rule_name, parameter_label) in enumerate((("Q3_mean_variance", "Risk aversion lambda (log scale)"),
                                                                 ("Q3_cvar", "CVaR95 daily loss limit L ($, log scale)"))):
        for column_position, (metric_name, metric_label) in enumerate((("sharpe", "Annualized Sharpe ratio"),
                                                                       ("total_profit", "Total profit ($)"))):
            panel_axis = axis_grid[row_position, column_position]
            for series_position, cost_value in enumerate(config.TRANSACTION_COST_GRID):
                curve = tuning_table[(tuning_table["rule"] == rule_name) & (tuning_table["cost"] == cost_value)]
                neon.glow_line(panel_axis, curve["parameter"], curve[metric_name], neon.neon_color(series_position),
                               f"2024, c = ${cost_value:.0f}", 1.8, glow_layers=2)
                panel_axis.scatter(curve["parameter"], curve[metric_name], color=neon.neon_color(series_position), s=18,
                                   zorder=4)
                panel_axis.axvline(chosen[(rule_name, cost_value)], color=neon.neon_color(series_position),
                                   linestyle=":", linewidth=1)
            panel_axis.set_xscale("log")
            neon.plain_log_axis(panel_axis, "x")
            # So, the metric is bound NOW (default argument), not when the figure is drawn.
            panel_axis.yaxis.set_major_formatter(
                lambda v, _, bound_metric=metric_name: f"{v:,.2f}" if bound_metric == "sharpe" else f"{v:,.0f}")
            neon.label_axis(panel_axis, f"{RULE_LABELS[rule_name]}: {metric_label.split(' (')[0]}", parameter_label,
                            metric_label)
    neon.save_neon_figure(neon_figure, "part3_risk_parameter_selection",
                          super_title="Part 3: lambda = Best 2024 Profit Within A 20% Drawdown Budget; L = 5% Of Capital (Dotted = Chosen)")
    # Finally, Question 2: the DP versus the ratio greedy when weights differ.
    neon_figure, gap_axis = neon.new_neon_canvas(1, 1, fig_width=11, fig_height=4.8)
    for series_position, (split_name, split_rows) in enumerate(knapsack_table.groupby("split")):
        gap_values = split_rows["gap"].to_numpy()
        gap_axis.hist(gap_values, bins=np.linspace(0, max(knapsack_table["gap"].max(), 1), 40),
                      color=neon.neon_color(series_position), histtype="step", linewidth=2.2,
                      label=f"{'2024' if split_name == 'validation' else '2025'}: greedy sub-optimal on "
                            f"{(gap_values > 1e-6).mean():.0%} of days")
    gap_axis.set_yscale("log")
    neon.plain_log_axis(gap_axis, "y")
    neon.label_axis(gap_axis, "Q2: Expected-Value Gap, Optimal DP Minus Ratio Greedy (c = $1)",
                    "DP value minus greedy value ($ per day)", "Number of days (log scale)", legend_place="upper right")
    neon.save_neon_figure(neon_figure, "part3_knapsack_dp_vs_greedy")


def run_part3(use_cache=True):
    print("\n******************** PART 3: Trading Budget Allocation (24 MWh/Day, <= 4 MWh/Hour) ********************")
    model_frame = load_model_dataset()
    prediction_frame = PredictionStore(model_frame, use_cache=use_cache, be_verbose=False).get_all_predictions()
    part2_bundle = pd.read_pickle(config.OUTPUT_CACHE_DIR / "part2_books.pkl")
    primary_view_b = part2_bundle["primaries"]["view_b"]
    train_rows = model_frame[model_frame["split"] == "train"]
    capital_dollars = capital_base_dollars(train_rows)
    collateral_budget = int(config.DAILY_BUDGET_MWH * np.median(
        np.ceil(config.COLLATERAL_SIGMA_MULTIPLE * np.maximum(train_rows["spread_std28_d2"], SIGMA_FLOOR_DOLLARS))))
    print(f"Step-Number 1: mu_hat from {primary_view_b} (chosen in Part 2 on 2024); sigma_hat = 28-day same-hour std "
          f"(D-2); collateral = {config.COLLATERAL_SIGMA_MULTIPLE:.0f} x sigma per MWh, daily collateral budget "
          f"= 24 x median(2023) = ${collateral_budget:,}")
    runner = BudgetAllocationRunner(model_frame, prediction_frame, primary_view_b,
                                    RiskHistory(load_hourly_dataset()), collateral_budget)

    # So, lambda (mean-variance) and L (CVaR) are chosen on 2024 by annualized Sharpe ratio, per cost.
    tuning_rows, chosen = [], {}
    for rule_name, parameter_grid in (("Q3_mean_variance", config.MEAN_VARIANCE_LAMBDAS), ("Q3_cvar", config.CVAR_LOSS_LIMITS)):
        for cost_value in config.TRANSACTION_COST_GRID:
            for parameter_value in parameter_grid:
                book = runner.run_rule("validation", rule_name, cost_value, parameter_value)
                tuning_rows.append({"rule": rule_name, "cost": cost_value, "parameter": parameter_value,
                                    **book.metrics(cost_value, capital_dollars)})
            rule_rows = pd.DataFrame([r for r in tuning_rows if r["rule"] == rule_name and r["cost"] == cost_value])
            if rule_name == "Q3_cvar":
                # In my opinion a CVaR limit is a risk-appetite input (no interior Sharpe optimum on 2024).
                chosen[(rule_name, cost_value)] = round(config.CVAR_LIMIT_SHARE_OF_CAPITAL * capital_dollars, 2)
            else:
                # So, the most profitable lambda on 2024 inside the drawdown budget (20% of capital).
                drawdown_budget = config.MEAN_VARIANCE_DRAWDOWN_BUDGET_SHARE * capital_dollars
                inside_budget = rule_rows[rule_rows["max_drawdown"] <= drawdown_budget]
                if inside_budget.empty:
                    inside_budget = rule_rows.nsmallest(1, "max_drawdown")
                chosen[(rule_name, cost_value)] = float(inside_budget.loc[inside_budget["total_profit"].idxmax(), "parameter"])
    tuning_table = pd.DataFrame(tuning_rows)
    tuning_table.to_csv(config.OUTPUT_TABLE_DIR / "part3_risk_parameter_tuning.csv", index=False)
    print(f"Step-Number 2: lambda = most profitable on 2024 with max drawdown <= "
          f"{config.MEAN_VARIANCE_DRAWDOWN_BUDGET_SHARE:.0%} of capital (${config.MEAN_VARIANCE_DRAWDOWN_BUDGET_SHARE * capital_dollars:,.0f}); "
          "L = 5% of capital (policy): "
          + ", ".join(f"{r} c=${c:.0f} -> {v:g}" for (r, c), v in chosen.items()))
    print(tuning_table[["rule", "cost", "parameter", "total_profit", "sharpe", "max_drawdown", "avg_budget_used"]]
          .round(4).to_string(index=False))

    books_by_split = {"validation": [], "test": []}
    for split_name in ("validation", "test"):
        for cost_value in config.TRANSACTION_COST_GRID:
            for rule_name in RULE_LABELS:
                parameter_value = chosen.get((rule_name, cost_value))
                books_by_split[split_name].append(runner.run_rule(split_name, rule_name, cost_value, parameter_value))
    result_frames, worst_day_rows = [], []
    for split_name, books in books_by_split.items():
        result_frames.append(metrics_table(books, capital_dollars).assign(split=split_name))
        for book in books:
            cost_value = book.get_cost_scope()
            for rank_position, (worst_date, worst_value) in enumerate(book.worst_days(cost_value).items(), start=1):
                worst_day_rows.append({"split": split_name, "strategy": book.get_name(), "cost": cost_value,
                                       "rank": rank_position, "delivery_date": worst_date.date(), "profit": worst_value})
    results = pd.concat(result_frames, ignore_index=True)
    results.to_csv(config.OUTPUT_TABLE_DIR / "part3_results.csv", index=False)
    pd.DataFrame(worst_day_rows).to_csv(config.OUTPUT_TABLE_DIR / "part3_five_worst_days.csv", index=False)
    print("Step-Number 3: Every allocation rule at the same budget:")
    print(results[["split", "strategy", "cost", "total_profit", "mwh_traded", "profit_per_mwh", "avg_budget_used",
                   "max_drawdown", "sharpe", "worst_day", "win_rate"]].round(3).to_string(index=False))
    worst_table = pd.DataFrame(worst_day_rows)
    print("Step-Number 4: Five worst days on 2025 at c = $1:")
    print(worst_table[(worst_table["split"] == "test") & (worst_table["cost"] == 1.0)]
          .pivot(index="strategy", columns="rank", values="profit").round(1).to_string())

    knapsack_frames, worked_example = [], None
    for split_name in ("validation", "test"):
        comparison, example = runner.knapsack_comparison(split_name, 1.0)
        knapsack_frames.append(comparison.assign(split=split_name))
        worked_example = worked_example if worked_example is not None else example
    knapsack_table = pd.concat(knapsack_frames, ignore_index=True)
    knapsack_table.to_csv(config.OUTPUT_TABLE_DIR / "part3_knapsack_dp_vs_greedy.csv", index=False)
    unit_mismatch = int((np.abs(knapsack_table["unit_greedy_value"] - knapsack_table["unit_dp_value"]) > 1e-6).sum())
    print(f"Step-Number 5: Unit-weight knapsack, greedy == DP on every day? mismatches = {unit_mismatch} of "
          f"{len(knapsack_table)} days")
    print("  Collateral knapsack, ratio greedy sub-optimal on "
          + ", ".join(f"{s}: {(g['gap'] > 1e-6).mean():.1%} of days (mean gap ${g['gap'].mean():.2f}/day)"
                      for s, g in knapsack_table.groupby("split")))
    if unit_mismatch > 0:
        raise ValueError("Oops! The unit-weight greedy disagreed with the dynamic programme.")
    if worked_example is not None:
        worked_example.to_csv(config.OUTPUT_TABLE_DIR / "part3_knapsack_worked_example.csv", index=False)
        print("  Worked example (first day where the greedy is sub-optimal), hours with any volume:")
        print(worked_example[(worked_example["greedy_mwh"] > 0) | (worked_example["dp_mwh"] > 0)].round(3)
              .to_string(index=False))
    test_books_by_cost = {c: [b for b in books_by_split["test"] if b.get_cost_scope() == c] for c in config.TRANSACTION_COST_GRID}
    plot_part3_figures(test_books_by_cost, tuning_table, knapsack_table, chosen)
    pd.to_pickle({"books": books_by_split, "chosen": chosen, "collateral_budget": collateral_budget},
                 config.OUTPUT_CACHE_DIR / "part3_books.pkl")
    print("\n  [PASS] Part 3 finished: tables -> outputs/tables/part3_*.csv, figures -> outputs/figures/part3_*.png")
    return results


if __name__ == "__main__":
    run_part3()
