# ==============================================================================
# Name    : Anay Abhijit Joshi
# Date    : November 22, 2026
# Course  : ISyE 6767 - Interim Project 2 (Virtual Bidding at ERCOT HB_NORTH)
# File    : final_evaluation.py
# Purpose : "Evaluation and deliverables" of project2.pdf. For EVERY strategy on
#           the 2025 test set: total profit, MWh traded, profit per MWh, number
#           of trades, win rate, cumulative daily profit, annualized return,
#           annualized Sharpe, maximum drawdown; whether profit comes from only
#           a few extreme days; and the CSV of test-set trades.
# ==============================================================================

import json

import numpy as np
import pandas as pd

import config
import neon_plots as neon
from backtest_engine import capital_base_dollars, metrics_table
from feature_factory import load_model_dataset

HEADLINE_COST = 1.0   # So, the realistic cost; c = $0 results are in every table as well.


def load_all_books():
    part1 = pd.read_pickle(config.OUTPUT_CACHE_DIR / "part1_books.pkl")
    part2 = pd.read_pickle(config.OUTPUT_CACHE_DIR / "part2_books.pkl")["books"]
    part3 = pd.read_pickle(config.OUTPUT_CACHE_DIR / "part3_books.pkl")["books"]
    return {split_name: part1[split_name] + part2[split_name] + part3[split_name] for split_name in ("validation", "test")}


def choose_headline_strategies(validation_table):
    # Let's do now the honest choice: the best strategy of each part is picked by 2024 Sharpe at c = $1.
    at_cost = validation_table[validation_table["cost"] == HEADLINE_COST]
    chosen = {"Baseline": "always_inc"}
    for part_label, part_filter in (("Part 1", at_cost["part"] == "Part 1"),
                                    ("Part 2", at_cost["part"].str.startswith("Part 2")),
                                    ("Part 3", at_cost["part"] == "Part 3")):
        part_rows = at_cost[part_filter & (at_cost["strategy"] != "always_inc")]
        chosen[part_label] = part_rows.loc[part_rows["sharpe"].idxmax(), "strategy"]
    return chosen


def profit_after_removing_best_days(book, cost_value, max_days_removed=20):
    daily_pnl = book.daily_profit(cost_value).sort_values(ascending=False).to_numpy()
    return np.array([daily_pnl[k:].sum() for k in range(max_days_removed + 1)])


def build_trades_csv(test_books):
    # So, one long CSV with every real trade of every strategy on 2025, in the PDF's format
    # (time, prediction, action, volume, spread, profit), profit given at the cost(s) the book is valid for.
    trade_frames = []
    for book in test_books:
        cost_scope = book.get_cost_scope()
        trade_rows = book.trade_log(cost_scope if cost_scope is not None else 0.0)
        trade_rows["profit_c0"] = book.hourly_profit(0.0).loc[trade_rows.index] if cost_scope in (None, 0.0) else np.nan
        trade_rows["profit_c1"] = book.hourly_profit(1.0).loc[trade_rows.index] if cost_scope in (None, 1.0) else np.nan
        trade_rows["part"] = book.get_part()
        trade_rows["cost_scope"] = "both" if cost_scope is None else f"c={cost_scope:g}"
        trade_frames.append(trade_rows)
    all_trades = pd.concat(trade_frames, ignore_index=True)
    all_trades["time_local"] = all_trades["interval_start_local"].dt.strftime("%Y-%m-%d %H:%M %Z")
    all_trades["time_utc"] = all_trades["interval_start_utc"].dt.strftime("%Y-%m-%d %H:%M UTC")
    ordered_columns = ["strategy", "part", "cost_scope", "time_local", "time_utc", "delivery_date", "hour_ending",
                       "prediction_kind", "prediction", "action_label", "action", "volume_mwh", "spread",
                       "profit_c0", "profit_c1"]
    return all_trades[ordered_columns].rename(columns={"action_label": "action", "action": "action_sign"})


def run_final_evaluation():
    print("\n******************** FINAL EVALUATION: Every Strategy On The 2025 Test Set ********************")
    model_frame = load_model_dataset()
    capital_dollars = capital_base_dollars(model_frame[model_frame["split"] == "train"])
    books_by_split = load_all_books()
    tables = {split_name: metrics_table(books, capital_dollars).assign(split=split_name)
              for split_name, books in books_by_split.items()}
    all_metrics = pd.concat(tables.values(), ignore_index=True)
    all_metrics.to_csv(config.OUTPUT_TABLE_DIR / "final_all_strategies_metrics.csv", index=False)
    show_columns = ["part", "strategy", "cost", "total_profit", "mwh_traded", "profit_per_mwh", "n_trades", "win_rate",
                    "annualized_return", "sharpe", "max_drawdown", "top5_days_profit", "profit_ex_top5_days"]
    print(f"Step-Number 1: Capital base for annualized return = ${capital_dollars:,.2f} (same for every strategy)")
    print(tables["test"][show_columns].round(3).to_string(index=False))

    headline = choose_headline_strategies(tables["validation"])
    print(f"Step-Number 2: Headline strategy per part, chosen on 2024 by Sharpe at c = ${HEADLINE_COST:.0f}: {headline}")
    test_books = {b.get_name(): b for b in books_by_split["test"]}
    headline_rows = tables["test"][tables["test"]["strategy"].isin(headline.values()) &
                                   (tables["test"]["cost"] == HEADLINE_COST)]
    headline_rows.to_csv(config.OUTPUT_TABLE_DIR / "final_headline_strategies_test.csv", index=False)
    print(headline_rows[show_columns].round(3).to_string(index=False))

    # So, "check whether profit comes from only a few extreme days": profit left after removing the k best days.
    concentration_rows = []
    neon_figure, (cumulative_axis, removal_axis) = neon.new_neon_canvas(1, 2, fig_width=16, fig_height=5.6)
    for series_position, (part_label, strategy_name) in enumerate(headline.items()):
        book = test_books[strategy_name]
        daily_pnl = book.daily_profit(HEADLINE_COST)
        remaining = profit_after_removing_best_days(book, HEADLINE_COST)
        concentration_rows.append({"part": part_label, "strategy": strategy_name, "total_profit": remaining[0],
                                   "ex_best_1_day": remaining[1], "ex_best_5_days": remaining[5],
                                   "ex_best_10_days": remaining[10], "ex_best_20_days": remaining[20],
                                   "share_of_days_positive": float((daily_pnl > 0).mean()),
                                   "best_day": float(daily_pnl.max()), "worst_day": float(daily_pnl.min())})
        neon.glow_line(cumulative_axis, daily_pnl.index, daily_pnl.cumsum(), neon.neon_color(series_position),
                       f"{part_label}: {strategy_name}", 1.7, glow_layers=2)
        neon.glow_line(removal_axis, np.arange(len(remaining)), remaining, neon.neon_color(series_position),
                       f"{part_label}: {strategy_name}", 1.7, glow_layers=2)
        removal_axis.scatter(np.arange(len(remaining)), remaining, color=neon.neon_color(series_position), s=10, zorder=4)
    from matplotlib.ticker import MaxNLocator
    removal_axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    for each_axis in (cumulative_axis, removal_axis):
        neon.zero_reference_line(each_axis)
        each_axis.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    neon.format_date_axis(cumulative_axis, month_step=2)
    neon.label_axis(cumulative_axis, "Cumulative Daily Profit, Headline Strategies (2025, c = $1)", "Delivery date",
                    "Cumulative profit ($)", legend_place="upper left")
    neon.label_axis(removal_axis, "Profit Left After Removing The k Best Days (2025, c = $1)",
                    "Number of best days removed (k)", "Total profit of the remaining days ($)", legend_place="upper right")
    neon.save_neon_figure(neon_figure, "final_headline_and_concentration")
    concentration_table = pd.DataFrame(concentration_rows)
    concentration_table.to_csv(config.OUTPUT_TABLE_DIR / "final_profit_concentration.csv", index=False)
    print("Step-Number 3: Does profit come from a few extreme days? (2025, c = $1)")
    print(concentration_table.round(2).to_string(index=False))

    # Finally, drawdown (underwater) curves of the headline strategies.
    neon_figure, underwater_axis = neon.new_neon_canvas(1, 1, fig_width=13, fig_height=4.8)
    for series_position, (part_label, strategy_name) in enumerate(headline.items()):
        cumulative_pnl = test_books[strategy_name].daily_profit(HEADLINE_COST).cumsum()
        underwater = cumulative_pnl - np.maximum(cumulative_pnl.cummax(), 0.0)
        neon.glow_line(underwater_axis, underwater.index, underwater, neon.neon_color(series_position),
                       f"{part_label}: {strategy_name}", 1.6, glow_layers=2)
    underwater_axis.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    neon.format_date_axis(underwater_axis, month_step=2)
    neon.label_axis(underwater_axis, "Drawdown From The Running Peak (2025, c = $1)", "Delivery date",
                    "Drawdown ($)", legend_place="lower left")
    neon.save_neon_figure(neon_figure, "final_drawdowns")

    # So, "cumulative daily profit for every strategy": the Part 2 tau-rule and filter books are not in any
    # other figure, so let's draw them here, one panel per cost, with Always INC at the same cost.
    reference_book = test_books["always_inc"]
    tau_filter_panels, panel_titles = [], []
    for each_cost in config.TRANSACTION_COST_GRID:
        panel_books = [b for b in books_by_split["test"] if b.get_part() in ("Part 2 tau rule", "Part 2 filter")
                       and b.get_cost_scope() == each_cost]
        tau_filter_panels.append([(b.get_name(), b.daily_profit(each_cost)) for b in panel_books] +
                                 [(f"always_inc (c = ${each_cost:.0f})", reference_book.daily_profit(each_cost))])
        panel_titles.append(f"Part 2 Tau Rule And Filter (c = ${each_cost:.0f}/MWh)")
    neon.cumulative_profit_panels(tau_filter_panels, panel_titles,
                                  "Part 2: Cumulative Daily Profit On 2025, Tau Rule And Classifier Filter (1 MWh/Trade)",
                                  "final_cumulative_tau_and_filter", n_columns=2)

    trades_frame = build_trades_csv(books_by_split["test"])
    trades_frame.to_csv(config.OUTPUT_TRADES_DIR / "test_trades_2025_all_strategies.csv", index=False)
    for part_label, strategy_name in headline.items():
        trades_frame[trades_frame["strategy"] == strategy_name].to_csv(
            config.OUTPUT_TRADES_DIR / f"test_trades_2025_{part_label.lower().replace(' ', '')}_{strategy_name}.csv",
            index=False)
    print(f"Step-Number 4: Test-set trades saved: {len(trades_frame):,} rows, "
          f"{trades_frame['strategy'].nunique()} strategies -> outputs/trades/")

    summary = {"capital_base_dollars": capital_dollars, "headline_strategies": headline,
               "headline_test_metrics": headline_rows.to_dict("records"),
               "profit_concentration": concentration_table.to_dict("records")}
    (config.OUTPUT_TABLE_DIR / "results_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\n  [PASS] Final evaluation finished: outputs/tables/final_*.csv, outputs/trades/*.csv, results_summary.json")
    return all_metrics


if __name__ == "__main__":
    run_final_evaluation()

    