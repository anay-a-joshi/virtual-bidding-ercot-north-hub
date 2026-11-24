# ==============================================================================
# Name    : Anay Abhijit Joshi
# Date    : November 22, 2026
# Course  : ISyE 6767 - Interim Project 2 (Virtual Bidding at ERCOT HB_NORTH)
# File    : magnitude_models.py
# Purpose : Part 2 trading logic on top of the regressors of model_suite.py:
#           the tau trade-selection rule, sizing rules compared at the SAME
#           total volume, the two edge estimates (View B: |mu|-c, View A +
#           classifier: (2p-1)|S_hat|), and the "regression as a filter" rule.
# ==============================================================================

import warnings

import numpy as np
import pandas as pd

import config
from model_suite import LOSS_SCORERS


def safe_spearman(predicted, target):
    # So, a constant prediction (the lasso that shrinks to its intercept) has no rank correlation: NaN, quietly.
    if np.nanstd(np.asarray(predicted, dtype=float)) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(pd.Series(np.asarray(predicted)).corr(pd.Series(np.asarray(target)), method="spearman"))


def regression_accuracy_table(prediction_frame, split_names=("train", "validation", "test")):
    # So, MAE / RMSE (asked by the PDF) plus Huber loss, bias, and sign accuracy for every regressor.
    metric_rows = []
    for column_name in [c for c in prediction_frame.columns if c.startswith(("muB__", "absA__", "qB__lgbm_q50"))]:
        view_name = "A" if column_name.startswith("absA__") else "B"
        for split_name in split_names:
            split_rows = prediction_frame[prediction_frame["split"] == split_name]
            target = split_rows["spread"].abs() if view_name == "A" else split_rows["spread"]
            predicted = split_rows[column_name]
            metric_rows.append({
                "model": column_name, "view": view_name, "split": split_name,
                "family": "tree" if "lgbm" in column_name else "linear",
                "loss": ("quantile" if "q50" in column_name else "absolute" if ("absolute" in column_name or "lad" in column_name)
                         else "huber" if "huber" in column_name else "squared"),
                "MAE": LOSS_SCORERS["absolute"](target, predicted),
                "RMSE": float(np.sqrt(LOSS_SCORERS["squared"](target, predicted))),
                "huber_loss": LOSS_SCORERS["huber"](target, predicted),
                "mean_prediction": float(predicted.mean()), "mean_target": float(target.mean()),
                "median_target": float(target.median()),
                "sign_accuracy": float(((predicted > 0) == (split_rows["spread"] > 0)).mean()) if view_name == "B" else np.nan,
                "spearman": safe_spearman(predicted, target)})
    return pd.DataFrame(metric_rows)


def quantile_interval_table(prediction_frame):
    rows = []
    for split_name in ("validation", "test"):
        split_rows = prediction_frame[prediction_frame["split"] == split_name]
        inside = (split_rows["spread"] >= split_rows["qB__lgbm_q10"]) & (split_rows["spread"] <= split_rows["qB__lgbm_q90"])
        rows.append({"split": split_name, "coverage_q10_q90": float(inside.mean()), "nominal": 0.8,
                     "below_q10": float((split_rows["spread"] < split_rows["qB__lgbm_q10"]).mean()),
                     "above_q90": float((split_rows["spread"] > split_rows["qB__lgbm_q90"]).mean()),
                     "mean_width": float((split_rows["qB__lgbm_q90"] - split_rows["qB__lgbm_q10"]).mean())})
    return pd.DataFrame(rows)


def tau_rule_actions(mu_hat, cost_per_mwh, tau_dollars):
    # So, the PDF rule: +1 if mu > c + tau, -1 if mu < -(c + tau), 0 otherwise.
    mu_hat = np.asarray(mu_hat, dtype=float)
    hurdle = cost_per_mwh + tau_dollars
    return np.where(mu_hat > hurdle, 1.0, np.where(mu_hat < -hurdle, -1.0, 0.0))


def tau_sweep(split_rows, mu_hat, cost_grid=config.TRANSACTION_COST_GRID, tau_grid=config.TAU_THRESHOLD_GRID):
    sweep_rows = []
    spread_values = split_rows["spread"].to_numpy()
    for cost_value in cost_grid:
        for tau_value in tau_grid:
            actions = tau_rule_actions(mu_hat, cost_value, tau_value)
            hourly_pnl = np.where(actions != 0, actions * spread_values - cost_value, 0.0)
            n_trades = int((actions != 0).sum())
            sweep_rows.append({"cost": cost_value, "tau": tau_value, "n_trades": n_trades,
                               "total_profit": float(hourly_pnl.sum()),
                               "profit_per_mwh": float(hourly_pnl.sum() / n_trades) if n_trades else np.nan,
                               "win_rate": float((hourly_pnl[actions != 0] > 0).mean()) if n_trades else np.nan})
    return pd.DataFrame(sweep_rows)


def scale_to_total_volume(raw_sizes, target_total_mwh):
    # Let's do now the "same total volume" normalisation: multiply every size by one constant k so that
    # sum(v) = target. Sizes are non-negative; if nothing is tradable the result is all zeros.
    raw_sizes = np.clip(np.nan_to_num(np.asarray(raw_sizes, dtype=float), nan=0.0, posinf=0.0), 0.0, None)
    raw_total = raw_sizes.sum()
    if raw_total <= 0 or target_total_mwh <= 0:
        return np.zeros_like(raw_sizes)
    return raw_sizes * (target_total_mwh / raw_total)


def view_b_edge(mu_hat, cost_per_mwh):
    # So, View B: expected profit per MWh after cost of trading in the direction of mu_hat.
    return np.abs(np.asarray(mu_hat, dtype=float)) - cost_per_mwh


def classifier_view_a_edge(positive_probability, predicted_abs_spread, cost_per_mwh):
    # So, the PDF's (2p - 1)|S_hat| uses p for direction confidence and View A for size. With
    # direction a = sign(2p - 1) the expected gross profit per MWh is |2p - 1| |S_hat|, IF sign and
    # size are independent (checked by sign_size_dependence below), minus the cost c.
    confidence = np.abs(2.0 * np.asarray(positive_probability, dtype=float) - 1.0)
    return confidence * np.asarray(predicted_abs_spread, dtype=float) - cost_per_mwh


def sign_size_dependence(prediction_frame, n_buckets=10):
    # In my opinion the cleanest test of independence: P(S > 0) inside each |S| decile. Under
    # independence every decile would show the same share.
    dependence_rows = []
    for split_name in ("train", "validation", "test"):
        split_rows = prediction_frame[prediction_frame["split"] == split_name]
        size_decile = pd.qcut(split_rows["spread"].abs(), n_buckets, labels=False, duplicates="drop")
        grouped = split_rows.groupby(size_decile)
        for decile_value, decile_rows in grouped:
            dependence_rows.append({"split": split_name, "abs_spread_decile": int(decile_value) + 1,
                                    "abs_spread_low": float(decile_rows["spread"].abs().min()),
                                    "abs_spread_high": float(decile_rows["spread"].abs().max()),
                                    "share_positive": float((decile_rows["spread"] > 0).mean()),
                                    "mean_spread": float(decile_rows["spread"].mean()), "hours": len(decile_rows)})
    return pd.DataFrame(dependence_rows)


def edge_calibration_table(split_rows, edge_estimates, n_buckets=10):
    # So, is an edge estimate a valid estimate of expected profit? Bucket hours by predicted gross edge
    # and compare with the REALISED profit per MWh of trading that direction.
    calibration_rows = []
    for edge_name, (gross_edge, direction) in edge_estimates.items():
        realised = np.asarray(direction) * split_rows["spread"].to_numpy()
        edge_bucket = pd.qcut(pd.Series(gross_edge).rank(method="first"), n_buckets, labels=False)
        for bucket_value in range(n_buckets):
            in_bucket = (edge_bucket == bucket_value).to_numpy()
            calibration_rows.append({"edge_estimate": edge_name, "bucket": bucket_value + 1,
                                     "mean_predicted_edge": float(np.mean(np.asarray(gross_edge)[in_bucket])),
                                     "mean_realised_profit": float(realised[in_bucket].mean()),
                                     "median_realised_profit": float(np.median(realised[in_bucket]))})
    return pd.DataFrame(calibration_rows)
