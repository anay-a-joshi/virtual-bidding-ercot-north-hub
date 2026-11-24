# ==============================================================================
# Name    : Anay Abhijit Joshi
# Date    : November 22, 2026
# Course  : ISyE 6767 - Interim Project 2 (Virtual Bidding at ERCOT HB_NORTH)
# File    : direction_models.py
# Purpose : Part 1. Every way of turning information into a DIRECTION
#           (INC = +1, DEC = -1), 1 MWh per hour:
#             Step 1 baselines      : Always INC/DEC, persistence D-2 / D-7, seasonal majority
#             (a) classifiers       : logistic, LightGBM, profit-weighted logistic
#             (b) heuristic rules   : 28-day same-hour mean sign, net-load-ramp rule
#             (c) regress then sign : best linear and best tree View B regressor
#             (d) different labels  : three classes, threshold label (|S| > $5 only)
# ==============================================================================

import numpy as np
import pandas as pd

import config
from backtest_engine import StrategyBook

FRAMING_ORDER = ["baseline", "(a) classifier", "(b) heuristic", "(c) regress-then-sign", "(d) different label"]


class DirectionSignal:
    """One direction strategy: actions for every hour, plus an optional probability and score."""

    def __init__(self, signal_name, framing_label, action_values, probability_values=None, score_values=None,
                 description=""):
        action_values = np.asarray(action_values, dtype=float)
        if not np.all(np.isin(action_values, (-1.0, 1.0))):
            raise ValueError(f"Oops! {signal_name}: a direction strategy must say INC (+1) or DEC (-1) every hour.")
        self.signal_name = signal_name
        self.framing_label = framing_label
        self.action_values = action_values
        self.probability_values = None if probability_values is None else np.asarray(probability_values, dtype=float)
        # So, the score is what ranks hours for ROC-AUC (a probability, a predicted $ spread, ...).
        self.score_values = (self.probability_values if score_values is None and probability_values is not None
                             else (None if score_values is None else np.asarray(score_values, dtype=float)))
        self.description = description

    def to_book(self, split_rows):
        prediction = self.probability_values if self.probability_values is not None else self.score_values
        kind = "P(S>0)" if self.probability_values is not None else ("score" if self.score_values is not None else "")
        return StrategyBook(self.signal_name, split_rows, self.action_values, np.ones(len(split_rows)), prediction,
                            kind, part_label="Part 1")


def sign_to_action(values):
    # So, a predicted y = 1{value > 0}: INC when strictly positive, DEC otherwise (S = 0 is "not > 0").
    return np.where(np.asarray(values, dtype=float) > 0, 1.0, -1.0)


class DirectionStrategyFactory:
    """Builds every Part 1 strategy for a given split; rules that need a choice learn it on 2023/2024 only."""

    def __init__(self, model_frame, prediction_frame):
        if not model_frame.index.equals(prediction_frame.index):
            raise ValueError("Oops! The model table and the prediction table must share the same index.")
        self._model_frame = model_frame
        self._prediction_frame = prediction_frame
        train_rows = model_frame[model_frame["split"] == "train"]
        # Let's do now the seasonal majority: the more frequent sign in 2023 for each hour of the week.
        hour_of_week = train_rows["day_of_week"] * 24 + train_rows["hour_ending"] - 1
        self._seasonal_majority = train_rows["y"].groupby(hour_of_week).mean().gt(0.5).astype(int)
        self._ramp_threshold, self._ramp_tuning = self._tune_ramp_rule()
        self._regress_choice, self._regress_tuning = self._tune_regress_then_sign()

    def get_tuning_notes(self):
        return {"ramp_rule": self._ramp_tuning, "regress_then_sign": self._regress_tuning,
                "ramp_threshold_mw": self._ramp_threshold, "regress_models": self._regress_choice}

    def _rows(self, split_name):
        return self._model_frame["split"] == split_name

    def _tune_ramp_rule(self):
        # So, (b) "go DEC when forecast net load ramps sharply": a sharp ramp = 3-hour net-load ramp above
        # a 2023 quantile; WHICH quantile is chosen on 2024 by profit (1 MWh/hour, cost does not change ranks).
        train_ramps = self._model_frame.loc[self._rows("train"), "net_load_ramp3_mw"]
        validation_rows = self._model_frame[self._rows("validation")]
        tuning_rows = []
        for ramp_quantile in config.RAMP_RULE_QUANTILES:
            ramp_threshold = float(train_ramps.quantile(ramp_quantile))
            actions = np.where(validation_rows["net_load_ramp3_mw"] > ramp_threshold, -1.0, 1.0)
            tuning_rows.append({"quantile": ramp_quantile, "threshold_mw": ramp_threshold,
                                "validation_profit_c0": float((actions * validation_rows["spread"]).sum())})
        tuning_table = pd.DataFrame(tuning_rows)
        best_row = tuning_table.loc[tuning_table["validation_profit_c0"].idxmax()]
        return float(best_row["threshold_mw"]), tuning_table

    def _tune_regress_then_sign(self):
        # So, (c) uses the View B regressors of Part 2; the best LINEAR and best TREE model are chosen
        # by 2024 profit of trading sign(mu_hat).
        validation_mask = self._rows("validation")
        validation_spread = self._model_frame.loc[validation_mask, "spread"].to_numpy()
        tuning_rows = []
        for column_name in [c for c in self._prediction_frame.columns if c.startswith("muB__")]:
            actions = sign_to_action(self._prediction_frame.loc[validation_mask, column_name])
            tuning_rows.append({"model": column_name, "family": "tree" if "lgbm" in column_name else "linear",
                                "validation_profit_c0": float((actions * validation_spread).sum())})
        tuning_table = pd.DataFrame(tuning_rows)
        chosen = {family: tuning_table[tuning_table["family"] == family].sort_values(
            "validation_profit_c0", ascending=False)["model"].iloc[0] for family in ("linear", "tree")}
        return chosen, tuning_table

    def build_signals(self, split_name):
        split_mask = self._rows(split_name)
        rows = self._model_frame[split_mask]
        predictions = self._prediction_frame[split_mask]
        hour_of_week = rows["day_of_week"] * 24 + rows["hour_ending"] - 1
        signals = [
            DirectionSignal("always_inc", "baseline", np.ones(len(rows)), description="INC every hour"),
            DirectionSignal("always_dec", "baseline", -np.ones(len(rows)), description="DEC every hour"),
            DirectionSignal("persistence_d2", "baseline", sign_to_action(rows["spread_lag2"]),
                            score_values=rows["spread_lag2"], description="sign of S, same hour, D-2"),
            DirectionSignal("persistence_d7", "baseline", sign_to_action(rows["spread_lag7"]),
                            score_values=rows["spread_lag7"], description="sign of S, same hour, D-7"),
            DirectionSignal("seasonal_majority", "baseline",
                            np.where(hour_of_week.map(self._seasonal_majority).to_numpy() == 1, 1.0, -1.0),
                            description="majority sign in 2023 for the hour of the week"),
            DirectionSignal("logistic", "(a) classifier", sign_to_action(predictions["p__logistic"] - 0.5),
                            probability_values=predictions["p__logistic"], description="L2 logistic regression"),
            DirectionSignal("lightgbm_classifier", "(a) classifier", sign_to_action(predictions["p__lightgbm"] - 0.5),
                            probability_values=predictions["p__lightgbm"], description="gradient-boosted trees"),
            DirectionSignal("weighted_logistic", "(a) classifier",
                            sign_to_action(predictions["p__weighted_logistic"] - 0.5),
                            probability_values=predictions["p__weighted_logistic"],
                            description="logistic, hours weighted by capped |S|"),
            DirectionSignal("heuristic_mean28_sign", "(b) heuristic", sign_to_action(rows["spread_mean28_d2"]),
                            score_values=rows["spread_mean28_d2"], description="sign of 28-day same-hour mean (D-2)"),
            DirectionSignal("heuristic_netload_ramp", "(b) heuristic",
                            np.where(rows["net_load_ramp3_mw"] > self._ramp_threshold, -1.0, 1.0),
                            score_values=-rows["net_load_ramp3_mw"],
                            description=f"DEC if 3-h net-load ramp > {self._ramp_threshold:,.0f} MW, else INC"),
            DirectionSignal("regress_sign_linear", "(c) regress-then-sign",
                            sign_to_action(predictions[self._regress_choice["linear"]]),
                            score_values=predictions[self._regress_choice["linear"]],
                            description=f"sign of {self._regress_choice['linear']}"),
            DirectionSignal("regress_sign_tree", "(c) regress-then-sign",
                            sign_to_action(predictions[self._regress_choice["tree"]]),
                            score_values=predictions[self._regress_choice["tree"]],
                            description=f"sign of {self._regress_choice['tree']}"),
            DirectionSignal("three_class_logistic", "(d) different label",
                            np.where(predictions["p3__large_positive"] >= predictions["p3__large_negative"], 1.0, -1.0),
                            score_values=predictions["p3__large_positive"] - predictions["p3__large_negative"],
                            description="INC if P(S > $10) >= P(S < -$10)"),
            DirectionSignal("threshold_label_logistic", "(d) different label",
                            sign_to_action(predictions["p__threshold_logistic"] - 0.5),
                            probability_values=predictions["p__threshold_logistic"],
                            description="logistic trained only on hours with |S| > $5"),
        ]
        return signals
