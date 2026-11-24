# ==============================================================================
# Name    : Anay Abhijit Joshi
# Date    : November 22, 2026
# Course  : ISyE 6767 - Interim Project 2 (Virtual Bidding at ERCOT HB_NORTH)
# File    : run_part1.py
# Purpose : Part 1 (direction prediction): baselines, classifiers, framings,
#           every metric the PDF lists, the plots, the feature ablation and a
#           robustness refit. Every choice is made on 2024; 2025 is scored once.
# ==============================================================================

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve

import config
import neon_plots as neon
from backtest_engine import capital_base_dollars, classification_scores, metrics_table
from direction_models import FRAMING_ORDER, DirectionStrategyFactory
from feature_factory import ALL_FEATURE_COLUMNS, FEATURE_GROUPS, load_model_dataset
from model_suite import (PredictionStore, direction_label, fit_lightgbm_classifier, fit_logistic)

ABS_SPREAD_BUCKETS = [0, 1, 5, 20, 100, np.inf]
ABS_SPREAD_BUCKET_LABELS = ["< $1", "$1-5", "$5-20", "$20-100", "> $100"]


def score_signals(signals, split_rows, split_name, capital_dollars):
    # So, one row per (strategy, cost): classification metrics + every trading metric.
    result_rows, books = [], []
    for each_signal in signals:
        book = each_signal.to_book(split_rows)
        books.append(book)
        class_scores = classification_scores(split_rows["y"], each_signal.probability_values,
                                             predicted_labels=(each_signal.action_values > 0).astype(int))
        if each_signal.probability_values is None and each_signal.score_values is not None:
            class_scores["roc_auc"] = roc_auc_score(split_rows["y"], each_signal.score_values)
        for metric_row in metrics_table([book], capital_dollars).to_dict("records"):
            result_rows.append({**metric_row, **class_scores, "split": split_name,
                                "framing": each_signal.framing_label, "description": each_signal.description})
    return pd.DataFrame(result_rows), books


def run_feature_ablation(model_frame, chosen_logistic_params, capital_dollars):
    # Let's do now the ablation: refit the chosen logistic (same C, same transform) without one group.
    print("\n******************** PART 1: Feature Ablation (Logistic, Drop One Group) ********************")
    train_rows = model_frame[model_frame["split"] == "train"]
    validation_rows = model_frame[model_frame["split"] == "validation"]
    test_rows = model_frame[model_frame["split"] == "test"]
    ablation_rows = []
    for dropped_group in ["none"] + list(FEATURE_GROUPS):
        kept_columns = [c for c in ALL_FEATURE_COLUMNS if dropped_group == "none" or c not in FEATURE_GROUPS[dropped_group]]
        fitted = fit_logistic(train_rows, validation_rows, kept_columns, direction_label, model_name="ablation",
                              fixed_params=chosen_logistic_params)
        for split_name, split_rows in (("validation", validation_rows), ("test", test_rows)):
            probability = fitted.predict(split_rows)
            scores = classification_scores(split_rows["y"], probability)
            actions = np.where(probability > 0.5, 1.0, -1.0)
            ablation_rows.append({"dropped_group": dropped_group, "n_features": len(kept_columns), "split": split_name,
                                  **scores, "profit_c0": float((actions * split_rows["spread"]).sum())})
    ablation_table = pd.DataFrame(ablation_rows)
    full_model = ablation_table[ablation_table["dropped_group"] == "none"].set_index("split")
    for metric_name in ("log_loss", "roc_auc", "accuracy", "profit_c0"):
        ablation_table[f"delta_{metric_name}"] = ablation_table.apply(
            lambda r: r[metric_name] - full_model.loc[r["split"], metric_name], axis=1)
    ablation_table.to_csv(config.OUTPUT_TABLE_DIR / "part1_feature_ablation.csv", index=False)
    print(ablation_table[ablation_table["split"] == "validation"]
          [["dropped_group", "n_features", "log_loss", "delta_log_loss", "roc_auc", "delta_roc_auc", "profit_c0"]]
          .round(5).to_string(index=False))
    return ablation_table


def run_robustness_refit(model_frame, fitted_models, capital_dollars):
    # So, a robustness row, NOT a selection: the same hyperparameters refitted on 2023 + 2024
    # (2023 is an extreme year; does more recent data help?). Scored once on 2025.
    print("\n******************** PART 1: Robustness, Same Settings Refitted On 2023-2024 ********************")
    train_plus_validation = model_frame[model_frame["split"].isin(["train", "validation"])]
    test_rows = model_frame[model_frame["split"] == "test"]
    logistic_params = fitted_models["logistic"].get_params()
    refit_logistic = fit_logistic(train_plus_validation, train_plus_validation, ALL_FEATURE_COLUMNS, direction_label,
                                  model_name="logistic_refit_2023_24", fixed_params=logistic_params)
    refit_lightgbm = fit_lightgbm_classifier(train_plus_validation, train_plus_validation, ALL_FEATURE_COLUMNS,
                                             model_name="lightgbm_refit_2023_24",
                                             fixed_params=fitted_models["lightgbm_classifier"].get_params())
    robustness_rows = []
    for fitted in (refit_logistic, refit_lightgbm):
        probability = fitted.predict(test_rows)
        actions = np.where(probability > 0.5, 1.0, -1.0)
        robustness_rows.append({"model": fitted.get_name(), **classification_scores(test_rows["y"], probability),
                                "profit_c0": float((actions * test_rows["spread"]).sum()),
                                "profit_c1": float((actions * test_rows["spread"] - 1.0).sum())})
    robustness_table = pd.DataFrame(robustness_rows)
    robustness_table.to_csv(config.OUTPUT_TABLE_DIR / "part1_robustness_refit.csv", index=False)
    print(robustness_table.round(4).to_string(index=False))
    return robustness_table


def plot_part1_figures(test_results, test_books, test_rows, test_signals):
    book_by_name = {b.get_name(): b for b in test_books}
    # So, figure 1 and 2: cumulative daily profit on 2025, one panel per framing, at c = $0 and c = $1.
    for cost_value in config.TRANSACTION_COST_GRID:
        panel_series, panel_titles = [], []
        for framing_name in FRAMING_ORDER:
            # So, Always INC is the dashed reference in every panel, so it is not drawn twice.
            members = [s.signal_name for s in test_signals if s.framing_label == framing_name and s.signal_name != "always_inc"]
            panel_series.append([(name, book_by_name[name].daily_profit(cost_value)) for name in members])
            panel_titles.append(framing_name.title() if framing_name == "baseline" else framing_name)
        neon.cumulative_profit_panels(panel_series, panel_titles,
                                      f"Part 1: Cumulative Daily Profit On 2025 (1 MWh/Hour, c = ${cost_value:.0f}/MWh)",
                                      f"part1_cumulative_profit_test_c{int(cost_value)}",
                                      reference_series=book_by_name["always_inc"].daily_profit(cost_value))

    # Let's do now figure 3: does higher sign accuracy mean higher profit? Full view + zoom on the cluster.
    c0_rows = test_results[test_results["cost"] == 0.0]
    rank_correlation = c0_rows[["accuracy", "total_profit"]].corr(method="spearman").iloc[0, 1]
    crowded = c0_rows["accuracy"] >= 0.60
    zoom_x = (c0_rows.loc[crowded, "accuracy"].min() - 0.004, c0_rows.loc[crowded, "accuracy"].max() + 0.004)
    zoom_y = (c0_rows.loc[crowded, "total_profit"].min() - 400, c0_rows.loc[crowded, "total_profit"].max() + 400)
    neon_figure, (full_axis, zoom_axis) = neon.new_neon_canvas(1, 2, fig_width=17, fig_height=6.6)
    for panel_axis, is_zoom in ((full_axis, False), (zoom_axis, True)):
        for framing_position, framing_name in enumerate(FRAMING_ORDER):
            framing_rows = c0_rows[c0_rows["framing"] == framing_name]
            neon.glow_scatter(panel_axis, framing_rows["accuracy"], framing_rows["total_profit"],
                              neon.neon_color(framing_position), framing_name, dot_size=70)
            for _, point in framing_rows.iterrows():
                if is_zoom != bool(point["accuracy"] >= 0.60):
                    continue
                place_left = is_zoom and point["accuracy"] > sum(zoom_x) / 2
                panel_axis.annotate(point["strategy"], (point["accuracy"], point["total_profit"]),
                                    xytext=(-8 if place_left else 8, 5), textcoords="offset points", fontsize=9,
                                    ha="right" if place_left else "left", color=neon.NEON_TEXT_INK)
        neon.zero_reference_line(panel_axis)
        panel_axis.xaxis.set_major_formatter(lambda v, _: f"{v:.1%}")
        panel_axis.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    full_axis.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
        (zoom_x[0], zoom_y[0]), zoom_x[1] - zoom_x[0], zoom_y[1] - zoom_y[0], fill=False,
        edgecolor=neon.NEON_MUTED_INK, linestyle="--", linewidth=1))
    zoom_axis.set_xlim(*zoom_x)
    zoom_axis.set_ylim(*zoom_y)
    neon.neon_note(full_axis, f"Spearman(accuracy, profit) = {rank_correlation:+.2f}  (2025, c = $0, n = {len(c0_rows)})")
    neon.label_axis(full_axis, "All Strategies (Dashed Box = Zoom)", "Sign accuracy", "Total profit at c = $0 (dollars)",
                    legend_place="lower right")
    neon.label_axis(zoom_axis, "Zoom: The Accuracy >= 60% Cluster", "Sign accuracy", "Total profit at c = $0 (dollars)",
                    show_legend=False)
    neon.save_neon_figure(neon_figure, "part1_accuracy_vs_profit", super_title="Part 1: Sign Accuracy Versus Profit On 2025")

    # So, figure 4: ROC curves and calibration for the probability models.
    probability_signals = [s for s in test_signals if s.probability_values is not None]
    neon_figure, (roc_axis, calibration_axis) = neon.new_neon_canvas(1, 2, fig_width=13, fig_height=5.6)
    for series_position, each_signal in enumerate(probability_signals):
        false_positive, true_positive, _ = roc_curve(test_rows["y"], each_signal.probability_values)
        auc_value = roc_auc_score(test_rows["y"], each_signal.probability_values)
        neon.glow_line(roc_axis, false_positive, true_positive, neon.neon_color(series_position),
                       f"{each_signal.signal_name} (AUC {auc_value:.3f})", 1.5, glow_layers=2)
        observed_share, mean_predicted = calibration_curve(test_rows["y"], each_signal.probability_values, n_bins=10,
                                                           strategy="quantile")
        neon.glow_line(calibration_axis, mean_predicted, observed_share, neon.neon_color(series_position),
                       each_signal.signal_name, 1.5, glow_layers=2)
        calibration_axis.scatter(mean_predicted, observed_share, color=neon.neon_color(series_position), s=14, zorder=4)
    roc_axis.plot([0, 1], [0, 1], color=neon.NEON_MUTED_INK, linestyle="--", linewidth=1, label="No skill")
    calibration_axis.plot([0, 1], [0, 1], color=neon.NEON_MUTED_INK, linestyle="--", linewidth=1, label="Perfect calibration")
    calibration_axis.set_xlim(0.2, 1.0)
    calibration_axis.set_ylim(0.2, 1.0)
    neon.label_axis(roc_axis, "ROC Curves On 2025", "False positive rate", "True positive rate", legend_place="lower right")
    neon.label_axis(calibration_axis, "Calibration On 2025 (Deciles)", "Mean predicted P(S > 0)",
                    "Observed share of S > 0", legend_place="upper left")
    neon.save_neon_figure(neon_figure, "part1_roc_and_calibration")

    # Finally, figure 5: the $0.10 versus $200 question, accuracy and $ at stake by |S| bucket.
    bucket_of_hour = pd.cut(test_rows["spread"].abs(), ABS_SPREAD_BUCKETS, labels=ABS_SPREAD_BUCKET_LABELS, right=False)
    bucket_rows = []
    for name in ("always_inc", "logistic", "weighted_logistic", "regress_sign_tree"):
        hourly_correct = (book_by_name[name].get_trade_frame()["action"].to_numpy() > 0).astype(int) == test_rows["y"].to_numpy()
        hourly_pnl = book_by_name[name].hourly_profit(0.0).to_numpy()
        for bucket_label in ABS_SPREAD_BUCKET_LABELS:
            in_bucket = (bucket_of_hour == bucket_label).to_numpy()
            bucket_rows.append({"strategy": name, "bucket": bucket_label, "hours": int(in_bucket.sum()),
                                "accuracy": float(hourly_correct[in_bucket].mean()),
                                "profit_c0": float(hourly_pnl[in_bucket].sum()),
                                "abs_spread_at_stake": float(test_rows["spread"].abs().to_numpy()[in_bucket].sum())})
    bucket_table = pd.DataFrame(bucket_rows)
    bucket_table.to_csv(config.OUTPUT_TABLE_DIR / "part1_accuracy_by_spread_size.csv", index=False)
    neon_figure, (accuracy_axis, stake_axis) = neon.new_neon_canvas(1, 2, fig_width=14, fig_height=5.2)
    bucket_positions = np.arange(len(ABS_SPREAD_BUCKET_LABELS))
    strategy_names = bucket_table["strategy"].unique()
    bar_width = 0.8 / len(strategy_names)
    for series_position, name in enumerate(strategy_names):
        strategy_rows = bucket_table[bucket_table["strategy"] == name]
        neon.glow_bars(accuracy_axis, bucket_positions + (series_position - 1.5) * bar_width, strategy_rows["accuracy"],
                       neon.neon_color(series_position), name, bar_width=bar_width * 0.95)
        neon.glow_bars(stake_axis, bucket_positions + (series_position - 1.5) * bar_width, strategy_rows["profit_c0"],
                       neon.neon_color(series_position), name, bar_width=bar_width * 0.95)
    hours_per_bucket = bucket_table[bucket_table["strategy"] == "always_inc"]["hours"].to_numpy()
    tick_labels = [f"{label}\n({n:,} h)" for label, n in zip(ABS_SPREAD_BUCKET_LABELS, hours_per_bucket)]
    accuracy_axis.set_xticks(bucket_positions, tick_labels)
    stake_axis.set_xticks(bucket_positions, tick_labels)
    neon.zero_reference_line(accuracy_axis, reference_value=0.5)
    neon.zero_reference_line(stake_axis)
    accuracy_axis.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    accuracy_axis.set_ylim(0, 0.88)
    stake_axis.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    neon.label_axis(accuracy_axis, "Sign Accuracy By Spread Size (2025)", "|S| bucket (hours in 2025)", "Sign accuracy",
                    show_legend=False)
    accuracy_axis.legend(loc="upper center", ncol=4, fontsize=8.5)
    neon.label_axis(stake_axis, "Profit By Spread Size (2025, c = $0)", "|S| bucket (hours in 2025)", "Profit ($)",
                    legend_place="lower left")
    neon.save_neon_figure(neon_figure, "part1_accuracy_and_profit_by_spread_size")
    return bucket_table


def plot_ablation(ablation_table):
    validation_rows = ablation_table[(ablation_table["split"] == "validation") & (ablation_table["dropped_group"] != "none")]
    neon_figure, (loss_axis, profit_axis) = neon.new_neon_canvas(1, 2, fig_width=14, fig_height=5.0)
    group_positions = np.arange(len(validation_rows))
    neon.glow_bars(loss_axis, group_positions, validation_rows["delta_log_loss"], neon.NEON_CYAN, "Change in log loss")
    neon.glow_bars(profit_axis, group_positions, validation_rows["delta_profit_c0"], neon.NEON_MAGENTA, "Change in profit")
    for each_axis in (loss_axis, profit_axis):
        each_axis.set_xticks(group_positions, validation_rows["dropped_group"].str.replace("_", "\n"), fontsize=9)
        neon.zero_reference_line(each_axis)
    neon.label_axis(loss_axis, "Validation Log Loss When A Group Is Dropped", "Dropped feature group",
                    "Log loss minus full model (> 0 = group helps)")
    neon.label_axis(profit_axis, "Validation Profit When A Group Is Dropped", "Dropped feature group",
                    "Profit minus full model, c = $0 ($)")
    neon.save_neon_figure(neon_figure, "part1_feature_ablation", super_title="Part 1: Feature Ablation (Logistic, 2024)")


def run_part1(use_cache=True):
    print("\n******************** PART 1: Direction Prediction (Classification) ********************")
    model_frame = load_model_dataset()
    prediction_store = PredictionStore(model_frame, use_cache=use_cache)
    prediction_frame = prediction_store.get_all_predictions()
    capital_dollars = capital_base_dollars(model_frame[model_frame["split"] == "train"])
    print(f"Step-Number 1: Capital base for annualized returns = 24 MWh x q99(|S|, 2023) = ${capital_dollars:,.2f}")
    strategy_factory = DirectionStrategyFactory(model_frame, prediction_frame)
    tuning_notes = strategy_factory.get_tuning_notes()
    tuning_notes["ramp_rule"].to_csv(config.OUTPUT_TABLE_DIR / "part1_ramp_rule_tuning.csv", index=False)
    tuning_notes["regress_then_sign"].to_csv(config.OUTPUT_TABLE_DIR / "part1_regress_then_sign_tuning.csv", index=False)
    print(f"Step-Number 2: Ramp rule threshold (chosen on 2024) = {tuning_notes['ramp_threshold_mw']:,.0f} MW; "
          f"regress-then-sign models (chosen on 2024) = {tuning_notes['regress_models']}")

    all_results, books_by_split, signals_by_split = [], {}, {}
    for split_name in ("validation", "test"):
        split_rows = model_frame[model_frame["split"] == split_name]
        signals = strategy_factory.build_signals(split_name)
        result_table, books = score_signals(signals, split_rows, split_name, capital_dollars)
        all_results.append(result_table)
        books_by_split[split_name], signals_by_split[split_name] = books, signals
    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(config.OUTPUT_TABLE_DIR / "part1_results.csv", index=False)
    show_columns = ["strategy", "framing", "accuracy", "balanced_accuracy", "roc_auc", "log_loss", "share_inc", "total_profit",
                    "profit_per_mwh", "sharpe", "max_drawdown"]
    for split_name in ("validation", "test"):
        for cost_value in config.TRANSACTION_COST_GRID:
            print(f"\n  {split_name.upper()} ({'2024' if split_name == 'validation' else '2025'}), c = ${cost_value:.0f}/MWh:")
            view = results[(results["split"] == split_name) & (results["cost"] == cost_value)][show_columns]
            print(view.round(4).to_string(index=False))

    test_rows = model_frame[model_frame["split"] == "test"]
    test_results = results[results["split"] == "test"]
    bucket_table = plot_part1_figures(test_results, books_by_split["test"], test_rows, signals_by_split["test"])
    print("\n  Accuracy and profit by |S| bucket (2025, c = $0):")
    print(bucket_table.round(3).to_string(index=False))
    fitted_models = prediction_store.get_fitted_models()
    ablation_table = run_feature_ablation(model_frame, fitted_models["logistic"].get_params(), capital_dollars)
    plot_ablation(ablation_table)
    run_robustness_refit(model_frame, fitted_models, capital_dollars)
    pd.to_pickle(books_by_split, config.OUTPUT_CACHE_DIR / "part1_books.pkl")
    print("\n  [PASS] Part 1 finished: tables -> outputs/tables/part1_*.csv, figures -> outputs/figures/part1_*.png")
    return results


if __name__ == "__main__":
    run_part1()
