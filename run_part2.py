# ==============================================================================
# Name    : Anay Abhijit Joshi
# Date    : November 22, 2026
# Course  : ISyE 6767 - Interim Project 2 (Virtual Bidding at ERCOT HB_NORTH)
# File    : run_part2.py
# Purpose : Part 2 (spread magnitude): View A |S| and View B S, linear and
#           tree models, squared / absolute / Huber losses and quantiles,
#           MAE and RMSE, the tau rule, sizing at the same total volume, the
#           independence question and regression as a filter for Part 1.
# ==============================================================================

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

import config
import neon_plots as neon
from backtest_engine import StrategyBook, capital_base_dollars, metrics_table
from feature_factory import load_model_dataset
from magnitude_models import (classifier_view_a_edge, edge_calibration_table, quantile_interval_table,
                              regression_accuracy_table, scale_to_total_volume, sign_size_dependence, tau_rule_actions,
                              tau_sweep, view_b_edge)
from model_suite import LOSS_SCORERS, PredictionStore

RISK_FLOOR_DOLLARS = 1.0   # So, a $0 volatility can never produce an infinite size.


def choose_primary_models(prediction_frame):
    # Let's do now every choice on 2024 only.
    validation_rows = prediction_frame[prediction_frame["split"] == "validation"]
    # (1) the Part 1 classifier with the lowest 2024 log loss gives p.
    classifier_losses = {name: float(log_loss(validation_rows["y"], np.clip(validation_rows[name], 1e-6, 1 - 1e-6)))
                         for name in ("p__logistic", "p__lightgbm")}
    primary_classifier = min(classifier_losses, key=classifier_losses.get)
    # (2) the View A model with the lowest 2024 Huber loss gives |S_hat| (robust to the spikes, not a median).
    view_a_losses = {name: LOSS_SCORERS["huber"](validation_rows["spread"].abs(), validation_rows[name])
                     for name in prediction_frame.columns if name.startswith("absA__")}
    primary_view_a = min(view_a_losses, key=view_a_losses.get)
    # (3) the View B model and tau with the highest 2024 profit at c = $1 (the realistic cost).
    sweep_rows = []
    for name in [c for c in prediction_frame.columns if c.startswith("muB__")]:
        sweep_rows.append(tau_sweep(validation_rows, validation_rows[name]).assign(model=name))
    validation_sweep = pd.concat(sweep_rows, ignore_index=True)
    at_cost_one = validation_sweep[validation_sweep["cost"] == 1.0]
    primary_view_b = at_cost_one.loc[at_cost_one["total_profit"].idxmax(), "model"]
    primary_sweep = validation_sweep[validation_sweep["model"] == primary_view_b]
    chosen_tau = {cost: float(primary_sweep[primary_sweep["cost"] == cost].sort_values("total_profit", ascending=False)
                              ["tau"].iloc[0]) for cost in config.TRANSACTION_COST_GRID}
    return {"classifier": primary_classifier, "classifier_losses": classifier_losses, "view_a": primary_view_a,
            "view_a_losses": view_a_losses, "view_b": primary_view_b, "tau_by_cost": chosen_tau,
            "validation_sweep_all_models": validation_sweep}


def plot_loss_comparison(prediction_frame):
    # So, how does the loss change the predictions? Squared loss chases the conditional MEAN (pulled by
    # the 2023 spikes), absolute loss the conditional MEDIAN, Huber sits in between.
    test_rows = prediction_frame[prediction_frame["split"] == "test"]
    neon_figure, (density_axis, hour_axis) = neon.new_neon_canvas(1, 2, fig_width=15, fig_height=5.4)
    bin_edges = np.linspace(-20, 30, 101)
    tree_columns = [("muB__lgbm_squared", "Tree, squared loss"), ("muB__lgbm_huber", "Tree, Huber loss"),
                    ("muB__lgbm_absolute", "Tree, absolute loss"), ("muB__ridge", "Ridge, squared loss"),
                    ("muB__lad_linear", "Linear, absolute loss")]
    actual_counts, _ = np.histogram(test_rows["spread"], bins=bin_edges)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    density_axis.fill_between(bin_centres, actual_counts / len(test_rows), color=neon.NEON_MUTED_INK, alpha=0.25,
                              step="mid", label="Actual S (2025)")
    for series_position, (column_name, series_label) in enumerate(tree_columns):
        predicted_counts, _ = np.histogram(test_rows[column_name], bins=bin_edges)
        neon.glow_line(density_axis, bin_centres, predicted_counts / len(test_rows), neon.neon_color(series_position),
                       f"{series_label} (mean {test_rows[column_name].mean():.1f})", 1.4, glow_layers=2)
    density_axis.set_yscale("log")
    neon.plain_log_axis(density_axis, "y")
    density_axis.set_ylim(1e-4, 1)
    neon.zero_reference_line(density_axis, orientation="v")
    neon.label_axis(density_axis, "Distribution Of Predicted mu_hat By Loss (2025)", "Predicted spread mu_hat ($/MWh)",
                    "Share of hours per $0.50 bin (log scale)", legend_place="upper right")
    hourly_profile = test_rows.groupby("hour_ending")
    hour_axis.plot(hourly_profile["spread"].mean().index, hourly_profile["spread"].mean(), color=neon.NEON_TEXT_INK,
                   linewidth=1.2, linestyle="--", label="Actual mean S")
    hour_axis.plot(hourly_profile["spread"].median().index, hourly_profile["spread"].median(), color=neon.NEON_MUTED_INK,
                   linewidth=1.2, linestyle=":", label="Actual median S")
    for series_position, (column_name, series_label) in enumerate(tree_columns[:3]):
        neon.glow_line(hour_axis, hourly_profile[column_name].mean().index, hourly_profile[column_name].mean(),
                       neon.neon_color(series_position), series_label, 1.8)
    neon.zero_reference_line(hour_axis)
    hour_axis.set_xticks(range(1, 25, 2))
    neon.label_axis(hour_axis, "Average Prediction By Hour (2025)", "Hour ending (Central)", "Spread ($/MWh)",
                    legend_place="upper left")
    neon.save_neon_figure(neon_figure, "part2_loss_comparison")


def plot_regression_errors(accuracy_table):
    # So, a 2 x 2 grid: rows = metric (MAE, RMSE), columns = view (B, A); bars = 2024 versus 2025.
    neon_figure, axis_grid = neon.new_neon_canvas(2, 2, fig_width=16, fig_height=9.5)
    for column_position, (view_name, view_title) in enumerate((("B", "View B: Signed Spread S"), ("A", "View A: |S|"))):
        view_rows = accuracy_table[(accuracy_table["view"] == view_name) & (accuracy_table["split"].isin(["validation", "test"]))]
        model_names = list(dict.fromkeys(view_rows["model"]))
        model_positions = np.arange(len(model_names))
        for row_position, metric_name in enumerate(("MAE", "RMSE")):
            panel_axis = axis_grid[row_position, column_position]
            for series_position, split_name in enumerate(("validation", "test")):
                metric_values = view_rows[view_rows["split"] == split_name].set_index("model").loc[model_names, metric_name]
                neon.glow_bars(panel_axis, model_positions + (series_position - 0.5) * 0.38, metric_values,
                               neon.neon_color(series_position),
                               "2024 (validation)" if split_name == "validation" else "2025 (test)", bar_width=0.36)
                for x_spot, metric_value in zip(model_positions + (series_position - 0.5) * 0.38, metric_values):
                    panel_axis.text(x_spot, metric_value, f"{metric_value:.1f}", ha="center", va="bottom", fontsize=7.5,
                                    color=neon.NEON_TEXT_INK)
            panel_axis.set_xticks(model_positions, [n.split("__")[1].replace("_", "\n") for n in model_names], fontsize=8.5)
            panel_axis.set_ylim(0, view_rows[metric_name].max() * 1.22)
            neon.label_axis(panel_axis, f"{view_title}: {metric_name}", "Model", f"{metric_name} ($/MWh)",
                            show_legend=False)
            panel_axis.legend(loc="upper right", ncol=2, fontsize=8.5)
    neon.save_neon_figure(neon_figure, "part2_regression_errors",
                          super_title="Part 2: MAE And RMSE On 2024 (Validation) And 2025 (Test)")


def plot_tau_selection(primary_sweeps, primary_name, chosen_tau):
    neon_figure, axis_triplet = neon.new_neon_canvas(1, 3, fig_width=17, fig_height=5.0)
    curve_specs = [("validation", 0.0), ("validation", 1.0), ("test", 0.0), ("test", 1.0)]
    for panel_axis, (metric_name, metric_label) in zip(axis_triplet, [("total_profit", "Total profit ($)"),
                                                                      ("n_trades", "Number of trades (1 MWh each)"),
                                                                      ("profit_per_mwh", "Profit per MWh ($)")]):
        for series_position, (split_name, cost_value) in enumerate(curve_specs):
            curve = primary_sweeps[split_name]
            curve = curve[curve["cost"] == cost_value]
            neon.glow_line(panel_axis, curve["tau"], curve[metric_name], neon.neon_color(series_position),
                           f"{'2024' if split_name == 'validation' else '2025'}, c = ${cost_value:.0f}", 1.8, glow_layers=2)
            panel_axis.scatter(curve["tau"], curve[metric_name], color=neon.neon_color(series_position), s=20, zorder=4)
        for cost_value, tau_value in chosen_tau.items():
            panel_axis.axvline(tau_value, color=neon.NEON_MUTED_INK, linestyle=":", linewidth=1)
        neon.zero_reference_line(panel_axis)
        panel_axis.set_xticks(config.TAU_THRESHOLD_GRID)
        neon.label_axis(panel_axis, metric_label.split(" (")[0], "Threshold tau ($/MWh)", metric_label)
    neon.save_neon_figure(neon_figure, "part2_tau_selection",
                          super_title=f"Part 2: Trade Only When |mu_hat| > c + tau  ({primary_name}; dotted = tau chosen on 2024)")


def build_sizing_books(split_rows, prediction_rows, primaries, cost_value):
    # So, every sizing rule is scaled to the SAME total MWh as "1 MWh per View B trade" in this split.
    mu_hat = prediction_rows[primaries["view_b"]].to_numpy()
    probability = prediction_rows[primaries["classifier"]].to_numpy()
    abs_hat = np.clip(prediction_rows[primaries["view_a"]].to_numpy(), 0.0, None)
    rolling_sigma = np.maximum(split_rows["spread_std28_d2"].to_numpy(), RISK_FLOOR_DOLLARS)
    quantile_sigma = np.maximum((prediction_rows["qB__lgbm_q90"] - prediction_rows["qB__lgbm_q10"]).to_numpy() / 2.563,
                                RISK_FLOOR_DOLLARS)
    view_b_actions = tau_rule_actions(mu_hat, cost_value, 0.0)
    view_b_edge_net = np.where(view_b_actions != 0, view_b_edge(mu_hat, cost_value), 0.0)
    target_total = float((view_b_actions != 0).sum())
    class_actions_all = np.where(probability > 0.5, 1.0, -1.0)
    class_edge_net = classifier_view_a_edge(probability, abs_hat, cost_value)
    class_actions = np.where(class_edge_net > 0, class_actions_all, 0.0)
    class_edge_net = np.where(class_actions != 0, class_edge_net, 0.0)
    sizing_specs = [
        ("size_B_unit", "View B, 1 MWh per trade", view_b_actions, (view_b_actions != 0).astype(float)),
        ("size_B_edge", "View B, v ~ |mu|-c", view_b_actions, view_b_edge_net),
        ("size_B_edge_over_var28", "View B, v ~ (|mu|-c)/sigma28^2", view_b_actions, view_b_edge_net / rolling_sigma ** 2),
        ("size_B_edge_over_varQ", "View B, v ~ (|mu|-c)/sigmaQ^2", view_b_actions, view_b_edge_net / quantile_sigma ** 2),
        ("size_A_unit", "(2p-1)|S|, 1 MWh per trade", class_actions, (class_actions != 0).astype(float)),
        ("size_A_edge", "v ~ |2p-1||S|-c", class_actions, class_edge_net),
        ("size_A_edge_over_var28", "v ~ (|2p-1||S|-c)/sigma28^2", class_actions, class_edge_net / rolling_sigma ** 2),
    ]
    books = []
    for strategy_name, description, actions, raw_sizes in sizing_specs:
        volumes = scale_to_total_volume(np.where(actions != 0, raw_sizes, 0.0), target_total)
        books.append(StrategyBook(f"{strategy_name}_c{int(cost_value)}", split_rows, actions, volumes,
                                  prediction_values=mu_hat if "_B_" in strategy_name else class_edge_net,
                                  prediction_kind=description, part_label="Part 2 sizing", cost_scope=cost_value))
    return books, target_total


def plot_dependence_and_calibration(dependence_table, calibration_by_split):
    neon_figure, axis_grid = neon.new_neon_canvas(1, 3, fig_width=18, fig_height=5.2)
    share_axis, validation_axis, test_axis = axis_grid
    for series_position, split_name in enumerate(("train", "validation", "test")):
        split_rows = dependence_table[dependence_table["split"] == split_name]
        neon.glow_line(share_axis, split_rows["abs_spread_decile"], split_rows["share_positive"],
                       neon.neon_color(series_position),
                       {"train": "2023 (train)", "validation": "2024 (validation)", "test": "2025 (test)"}[split_name], 1.8)
        share_axis.scatter(split_rows["abs_spread_decile"], split_rows["share_positive"],
                           color=neon.neon_color(series_position), s=18, zorder=4)
    neon.zero_reference_line(share_axis, reference_value=0.5)
    share_axis.set_xticks(range(1, 11))
    share_axis.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    neon.label_axis(share_axis, "P(S > 0) Inside Each |S| Decile", "|S| decile (1 = smallest, 10 = largest)",
                    "Share of hours with S > 0", legend_place="upper left")
    for panel_axis, (split_label, calibration_table) in zip((validation_axis, test_axis), calibration_by_split.items()):
        for series_position, edge_name in enumerate(calibration_table["edge_estimate"].unique()):
            edge_rows = calibration_table[calibration_table["edge_estimate"] == edge_name]
            neon.glow_line(panel_axis, edge_rows["mean_predicted_edge"], edge_rows["mean_realised_profit"],
                           neon.neon_color(series_position), edge_name, 1.8)
            panel_axis.scatter(edge_rows["mean_predicted_edge"], edge_rows["mean_realised_profit"],
                               color=neon.neon_color(series_position), s=18, zorder=4)
        limit_high = max(calibration_table["mean_predicted_edge"].max(), 1.0)
        panel_axis.plot([0, limit_high], [0, limit_high], color=neon.NEON_MUTED_INK, linestyle="--", linewidth=1,
                        label="Realised = predicted")
        neon.zero_reference_line(panel_axis)
        neon.label_axis(panel_axis, f"Predicted Vs Realised Edge ({split_label})", "Mean predicted gross edge ($/MWh, decile)",
                        "Mean realised profit per MWh ($)", legend_place="upper left")
    neon.save_neon_figure(neon_figure, "part2_sign_size_dependence_and_edge_calibration")


def run_filter_study(prediction_frame, primaries):
    # So, does regression help the classifier? Keep the classifier's direction, but only trade the hours
    # where the View A model expects a big enough move. The cut-off k is a 2024 quantile, chosen on 2024.
    validation_rows = prediction_frame[prediction_frame["split"] == "validation"]
    cut_offs = {f"q{int(q * 100)}": float(validation_rows[primaries["view_a"]].quantile(q)) if q > 0 else -np.inf
                for q in (0.0, 0.25, 0.5, 0.75, 0.9)}
    filter_rows = []
    for split_name in ("validation", "test"):
        split_rows = prediction_frame[prediction_frame["split"] == split_name]
        direction = np.where(split_rows[primaries["classifier"]] > 0.5, 1.0, -1.0)
        for cut_name, cut_value in cut_offs.items():
            actions = np.where(split_rows[primaries["view_a"]] > cut_value, direction, 0.0)
            for cost_value in config.TRANSACTION_COST_GRID:
                hourly_pnl = np.where(actions != 0, actions * split_rows["spread"] - cost_value, 0.0)
                n_trades = int((actions != 0).sum())
                filter_rows.append({"split": split_name, "cut_name": cut_name, "abs_s_hat_cut": cut_value, "cost": cost_value,
                                    "n_trades": n_trades, "total_profit": float(hourly_pnl.sum()),
                                    "profit_per_mwh": float(hourly_pnl.sum() / n_trades) if n_trades else np.nan})
    filter_table = pd.DataFrame(filter_rows)
    chosen_cut = {}
    for cost_value in config.TRANSACTION_COST_GRID:
        validation_choice = filter_table[(filter_table["split"] == "validation") & (filter_table["cost"] == cost_value)]
        chosen_cut[cost_value] = validation_choice.loc[validation_choice["total_profit"].idxmax(), "cut_name"]
    neon_figure, axis_pair = neon.new_neon_canvas(1, 2, fig_width=14, fig_height=5.0)
    cut_names = list(cut_offs)
    for panel_axis, metric_name, metric_label in zip(axis_pair, ("total_profit", "profit_per_mwh"),
                                                     ("Total profit ($)", "Profit per MWh ($)")):
        for series_position, (split_name, cost_value) in enumerate([("validation", 0.0), ("validation", 1.0),
                                                                    ("test", 0.0), ("test", 1.0)]):
            curve = filter_table[(filter_table["split"] == split_name) & (filter_table["cost"] == cost_value)]
            neon.glow_line(panel_axis, np.arange(len(cut_names)), curve[metric_name], neon.neon_color(series_position),
                           f"{'2024' if split_name == 'validation' else '2025'}, c = ${cost_value:.0f}", 1.8, glow_layers=2)
            panel_axis.scatter(np.arange(len(cut_names)), curve[metric_name], color=neon.neon_color(series_position), s=18,
                               zorder=4)
        panel_axis.set_xticks(np.arange(len(cut_names)),
                              ["no filter" if n == "q0" else f"|S_hat| > {n}\n(${cut_offs[n]:.1f})" for n in cut_names])
        neon.zero_reference_line(panel_axis)
        neon.label_axis(panel_axis, f"Classifier + View A Filter: {metric_label.split(' (')[0]}",
                        "Trade only if predicted |S| above this 2024 quantile", metric_label)
    neon.save_neon_figure(neon_figure, "part2_regression_filter_for_classifier")
    filter_table.to_csv(config.OUTPUT_TABLE_DIR / "part2_regression_filter.csv", index=False)
    return filter_table, chosen_cut, cut_offs


def run_part2(use_cache=True):
    print("\n******************** PART 2: Spread Magnitude Forecast (Regression) ********************")
    model_frame = load_model_dataset()
    prediction_frame = PredictionStore(model_frame, use_cache=use_cache).get_all_predictions()
    capital_dollars = capital_base_dollars(model_frame[model_frame["split"] == "train"])

    accuracy_table = regression_accuracy_table(prediction_frame)
    accuracy_table.to_csv(config.OUTPUT_TABLE_DIR / "part2_regression_accuracy.csv", index=False)
    print("Step-Number 1: MAE / RMSE of every regressor (2024 validation and 2025 test):")
    print(accuracy_table[accuracy_table["split"] != "train"][["model", "split", "loss", "MAE", "RMSE", "huber_loss",
                                                             "mean_prediction", "mean_target", "sign_accuracy", "spearman"]]
          .round(3).to_string(index=False))
    interval_table = quantile_interval_table(prediction_frame)
    interval_table.to_csv(config.OUTPUT_TABLE_DIR / "part2_quantile_interval_coverage.csv", index=False)
    print("Step-Number 2: Quantile regression interval [q10, q90] (nominal coverage 80%):")
    print(interval_table.round(4).to_string(index=False))
    plot_loss_comparison(prediction_frame)
    plot_regression_errors(accuracy_table)

    primaries = choose_primary_models(prediction_frame)
    primaries["validation_sweep_all_models"].to_csv(config.OUTPUT_TABLE_DIR / "part2_tau_sweep_all_models_validation.csv",
                                                    index=False)
    print(f"Step-Number 3: Choices made on 2024 -> classifier p: {primaries['classifier']}, View A |S_hat|: "
          f"{primaries['view_a']}, View B mu_hat: {primaries['view_b']}, tau by cost: {primaries['tau_by_cost']}")
    primary_sweeps = {}
    for split_name in ("validation", "test"):
        split_rows = prediction_frame[prediction_frame["split"] == split_name]
        primary_sweeps[split_name] = tau_sweep(split_rows, split_rows[primaries["view_b"]]).assign(split=split_name)
    tau_table = pd.concat(primary_sweeps.values(), ignore_index=True)
    tau_table.to_csv(config.OUTPUT_TABLE_DIR / "part2_tau_sweep_primary.csv", index=False)
    print("Step-Number 4: Tau sweep for the primary View B model:")
    print(tau_table.round(4).to_string(index=False))
    plot_tau_selection(primary_sweeps, primaries["view_b"], primaries["tau_by_cost"])

    books_by_split, sizing_rows = {"validation": [], "test": []}, []
    for split_name in ("validation", "test"):
        split_rows = model_frame[model_frame["split"] == split_name]
        prediction_rows = prediction_frame[prediction_frame["split"] == split_name]
        for cost_value in config.TRANSACTION_COST_GRID:
            tau_value = primaries["tau_by_cost"][cost_value]
            tau_book = StrategyBook(f"viewB_tau{tau_value:g}_c{int(cost_value)}", split_rows,
                                    tau_rule_actions(prediction_rows[primaries["view_b"]], cost_value, tau_value),
                                    np.ones(len(split_rows)), prediction_rows[primaries["view_b"]].to_numpy(), "mu_hat",
                                    part_label="Part 2 tau rule", cost_scope=cost_value)
            sizing_books, target_total = build_sizing_books(split_rows, prediction_rows, primaries, cost_value)
            books_by_split[split_name].extend([tau_book] + sizing_books)
            sizing_rows.append(metrics_table([tau_book] + sizing_books, capital_dollars)
                               .assign(split=split_name, target_total_mwh=target_total))
    sizing_table = pd.concat(sizing_rows, ignore_index=True)
    sizing_table.to_csv(config.OUTPUT_TABLE_DIR / "part2_tau_and_sizing_results.csv", index=False)
    print("Step-Number 5: Tau rule and sizing rules at the SAME total volume:")
    print(sizing_table[["split", "strategy", "cost", "mwh_traded", "n_trades", "max_hourly_mwh", "total_profit",
                        "profit_per_mwh", "sharpe", "max_drawdown", "win_rate"]].round(3).to_string(index=False))
    test_books = {b.get_name(): b for b in books_by_split["test"]}
    for cost_value in config.TRANSACTION_COST_GRID:
        suffix = f"_c{int(cost_value)}"
        neon.cumulative_profit_panels(
            [[(n.replace(suffix, ""), test_books[n + suffix].daily_profit(cost_value))
              for n in ("size_B_unit", "size_B_edge", "size_B_edge_over_var28", "size_B_edge_over_varQ")],
             [(n.replace(suffix, ""), test_books[n + suffix].daily_profit(cost_value))
              for n in ("size_A_unit", "size_A_edge", "size_A_edge_over_var28")]],
            ["View B Edge: |mu_hat| - c", "Classifier + View A Edge: |2p-1||S_hat| - c"],
            f"Part 2: Sizing Rules At The Same Total Volume On 2025 (c = ${cost_value:.0f}/MWh)",
            f"part2_sizing_cumulative_test_c{int(cost_value)}", n_columns=2)

    dependence_table = sign_size_dependence(prediction_frame)
    dependence_table.to_csv(config.OUTPUT_TABLE_DIR / "part2_sign_size_dependence.csv", index=False)
    print("Step-Number 6: Sign versus size, P(S > 0) by |S| decile:")
    print(dependence_table.pivot(index="abs_spread_decile", columns="split", values="share_positive").round(3).to_string())
    calibration_by_split = {}
    for split_name, split_label in (("validation", "2024"), ("test", "2025")):
        split_rows = prediction_frame[prediction_frame["split"] == split_name]
        mu_hat = split_rows[primaries["view_b"]].to_numpy()
        probability = split_rows[primaries["classifier"]].to_numpy()
        calibration_by_split[split_label] = edge_calibration_table(split_rows, {
            "View B: |mu_hat|": (np.abs(mu_hat), np.sign(mu_hat)),
            "(2p-1)|S_hat|": (np.abs(2 * probability - 1) * np.clip(split_rows[primaries["view_a"]].to_numpy(), 0, None),
                              np.where(probability > 0.5, 1.0, -1.0))})
        calibration_by_split[split_label].assign(split=split_label).to_csv(
            config.OUTPUT_TABLE_DIR / f"part2_edge_calibration_{split_label}.csv", index=False)
    plot_dependence_and_calibration(dependence_table, calibration_by_split)
    filter_table, chosen_cut, cut_offs = run_filter_study(prediction_frame, primaries)
    print(f"Step-Number 7: Classifier + View A filter, cut-off chosen on 2024 by cost: {chosen_cut}")
    print(filter_table.round(3).to_string(index=False))
    for split_name in ("validation", "test"):
        split_rows = model_frame[model_frame["split"] == split_name]
        prediction_rows = prediction_frame[prediction_frame["split"] == split_name]
        direction = np.where(prediction_rows[primaries["classifier"]] > 0.5, 1.0, -1.0)
        for cost_value, cut_name in chosen_cut.items():
            actions = np.where(prediction_rows[primaries["view_a"]] > cut_offs[cut_name], direction, 0.0)
            books_by_split[split_name].append(StrategyBook(
                f"classifier_filtered_{cut_name}_c{int(cost_value)}", split_rows, actions, np.ones(len(split_rows)),
                prediction_rows[primaries["classifier"]].to_numpy(), "P(S>0)", part_label="Part 2 filter",
                cost_scope=cost_value))
    summary = {k: v for k, v in primaries.items() if k != "validation_sweep_all_models"}
    pd.to_pickle({"books": books_by_split, "primaries": summary}, config.OUTPUT_CACHE_DIR / "part2_books.pkl")
    print("\n  [PASS] Part 2 finished: tables -> outputs/tables/part2_*.csv, figures -> outputs/figures/part2_*.png")
    return summary


if __name__ == "__main__":
    run_part2()
