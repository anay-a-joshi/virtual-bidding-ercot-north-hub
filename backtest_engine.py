# ==============================================================================
# Name    : Anay Abhijit Joshi
# Date    : November 22, 2026
# Course  : ISyE 6767 - Interim Project 2 (Virtual Bidding at ERCOT HB_NORTH)
# File    : backtest_engine.py
# Purpose : ONE engine that scores every strategy of Parts 1-3 the same way:
#           profit = v * (a * S - c), and every metric the PDF asks for
#           (total profit, MWh, profit/MWh, trades, win rate, cumulative daily
#           profit, annualized return, annualized Sharpe, max drawdown, five
#           worst days, budget used, profit concentration in extreme days).
# ==============================================================================

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score

import config

TRADE_FRAME_COLUMNS = ["delivery_date", "interval_start_utc", "interval_start_local", "hour_ending", "spread"]


def hourly_profit(action_values, volume_values, spread_values, cost_per_mwh):
    # So, pi = v (a S - c). A "no trade" (a = 0 or v = 0) earns exactly zero and pays no cost.
    action_values = np.asarray(action_values, dtype=float)
    volume_values = np.asarray(volume_values, dtype=float)
    spread_values = np.asarray(spread_values, dtype=float)
    if np.any(volume_values < 0):
        raise ValueError("Oops! Volumes must be non-negative.")
    if not np.all(np.isin(action_values, (-1.0, 0.0, 1.0))):
        raise ValueError("Oops! Actions must be +1 (INC), -1 (DEC) or 0 (no trade).")
    is_trading = (action_values != 0) & (volume_values > 0)
    return np.where(is_trading, volume_values * (action_values * spread_values - cost_per_mwh), 0.0)


def capital_base_dollars(train_frame):
    # Let's do now the capital for annualized returns, fixed once from 2023 and shared by everyone.
    abs_spread_quantile = train_frame["spread"].abs().quantile(config.CAPITAL_ABS_SPREAD_QUANTILE)
    return float(config.DAILY_BUDGET_MWH * abs_spread_quantile)


def max_drawdown_dollars(daily_profit_values):
    # So, the deepest peak-to-trough fall of the cumulative daily P&L, starting from zero.
    cumulative_pnl = np.cumsum(np.asarray(daily_profit_values, dtype=float))
    running_peak = np.maximum.accumulate(np.concatenate([[0.0], cumulative_pnl]))[1:]
    return float(np.max(running_peak - cumulative_pnl)) if len(cumulative_pnl) else 0.0


def annualized_sharpe(daily_profit_values):
    daily_profit_values = np.asarray(daily_profit_values, dtype=float)
    daily_std = daily_profit_values.std(ddof=1) if len(daily_profit_values) > 1 else 0.0
    if daily_std <= 1e-12:
        return float("nan")
    return float(daily_profit_values.mean() / daily_std * np.sqrt(config.DAYS_PER_YEAR))


def classification_scores(true_labels, positive_probability=None, predicted_labels=None):
    # In my opinion probability metrics only make sense for real probabilities, so they are NaN otherwise.
    true_labels = np.asarray(true_labels, dtype=int)
    if predicted_labels is None:
        predicted_labels = (np.asarray(positive_probability) > 0.5).astype(int)
    score_table = {"accuracy": accuracy_score(true_labels, predicted_labels),
                   "balanced_accuracy": balanced_accuracy_score(true_labels, predicted_labels),
                   "roc_auc": float("nan"), "log_loss": float("nan")}
    if positive_probability is not None:
        clipped_probability = np.clip(np.asarray(positive_probability, dtype=float), 1e-6, 1 - 1e-6)
        score_table["roc_auc"] = roc_auc_score(true_labels, clipped_probability)
        score_table["log_loss"] = log_loss(true_labels, clipped_probability, labels=[0, 1])
    return score_table


class StrategyBook:
    """All the hourly decisions of one strategy on one split, and every metric computed from them."""

    def __init__(self, strategy_name, split_rows, action_values, volume_values, prediction_values=None,
                 prediction_kind="", part_label="", cost_scope=None):
        if len(split_rows) != len(action_values) or len(split_rows) != len(volume_values):
            raise ValueError(f"Oops! {strategy_name}: rows, actions and volumes have different lengths.")
        self._strategy_name = strategy_name
        self._part_label = part_label
        self._prediction_kind = prediction_kind
        # So, a strategy whose DECISIONS depend on c (e.g. trade only if |mu| > c + tau) is built once per
        # cost and is only ever scored at that cost; cost_scope = None means "same decisions for every c".
        self._cost_scope = cost_scope
        self._trade_frame = split_rows[TRADE_FRAME_COLUMNS].copy().reset_index(drop=True)
        self._trade_frame["action"] = np.asarray(action_values, dtype=float)
        self._trade_frame["volume_mwh"] = np.asarray(volume_values, dtype=float)
        self._trade_frame["prediction"] = (np.asarray(prediction_values, dtype=float)
                                           if prediction_values is not None else np.nan)
        # So, fail loudly if a strategy ever produced a non-finite volume or an invalid action.
        if not np.all(np.isfinite(self._trade_frame["volume_mwh"])):
            raise ValueError(f"Oops! {strategy_name}: non-finite volume.")
        hourly_profit(self._trade_frame["action"], self._trade_frame["volume_mwh"], self._trade_frame["spread"], 0.0)

    # ---------------------------------------------------------------- getters
    def get_name(self):
        return self._strategy_name

    def get_part(self):
        return self._part_label

    def get_cost_scope(self):
        return self._cost_scope

    def get_trade_frame(self):
        return self._trade_frame.copy()

    # ---------------------------------------------------------------- profit views
    def hourly_profit(self, cost_per_mwh):
        return pd.Series(hourly_profit(self._trade_frame["action"], self._trade_frame["volume_mwh"],
                                       self._trade_frame["spread"], cost_per_mwh), index=self._trade_frame.index)

    def daily_profit(self, cost_per_mwh):
        # Let's do now one number per delivery day (days with no trade count as $0, they are real days).
        return self.hourly_profit(cost_per_mwh).groupby(self._trade_frame["delivery_date"]).sum()

    def daily_volume(self):
        is_trading = self._trade_frame["action"] != 0
        return (self._trade_frame["volume_mwh"].where(is_trading, 0.0)
                .groupby(self._trade_frame["delivery_date"]).sum())

    def worst_days(self, cost_per_mwh, how_many=5):
        return self.daily_profit(cost_per_mwh).nsmallest(how_many)

    def metrics(self, cost_per_mwh, capital_dollars):
        hourly_pnl = self.hourly_profit(cost_per_mwh)
        daily_pnl = self.daily_profit(cost_per_mwh)
        is_trade = (self._trade_frame["action"] != 0) & (self._trade_frame["volume_mwh"] > 0)
        mwh_traded = float(self._trade_frame.loc[is_trade, "volume_mwh"].sum())
        total_profit = float(hourly_pnl.sum())
        n_days = len(daily_pnl)
        top_five_days = daily_pnl.nlargest(5).sum()
        return {
            "strategy": self._strategy_name, "part": self._part_label, "cost": cost_per_mwh,
            "total_profit": total_profit,
            "mwh_traded": mwh_traded,
            "profit_per_mwh": total_profit / mwh_traded if mwh_traded > 0 else float("nan"),
            "n_trades": int(is_trade.sum()),
            "share_inc": float((self._trade_frame.loc[is_trade, "action"] > 0).mean()) if is_trade.any() else float("nan"),
            "win_rate": float((hourly_pnl[is_trade] > 0).mean()) if is_trade.any() else float("nan"),
            "annualized_return": total_profit / n_days * config.DAYS_PER_YEAR / capital_dollars,
            "sharpe": annualized_sharpe(daily_pnl.values),
            "max_drawdown": max_drawdown_dollars(daily_pnl.values),
            "max_hourly_mwh": float(self._trade_frame.loc[is_trade, "volume_mwh"].max()) if is_trade.any() else 0.0,
            "avg_budget_used": float(self.daily_volume().mean() / config.DAILY_BUDGET_MWH),
            "top5_days_profit": float(top_five_days),
            "profit_ex_top5_days": float(total_profit - top_five_days),
            "worst_day": float(daily_pnl.min()),
            "n_days": n_days,
        }

    def trade_log(self, cost_per_mwh):
        # Finally, the rows that are real trades, in the format the PDF asks for.
        trade_rows = self._trade_frame.copy()
        trade_rows["profit"] = self.hourly_profit(cost_per_mwh).values
        trade_rows["strategy"] = self._strategy_name
        trade_rows["prediction_kind"] = self._prediction_kind
        trade_rows["action_label"] = trade_rows["action"].map({1.0: "INC", -1.0: "DEC", 0.0: "NONE"})
        return trade_rows[(trade_rows["action"] != 0) & (trade_rows["volume_mwh"] > 0)]


def metrics_table(strategy_books, capital_dollars, cost_grid=config.TRANSACTION_COST_GRID):
    # So, one row per (strategy, cost): the comparison table used everywhere in the report.
    return pd.DataFrame([book.metrics(each_cost, capital_dollars) for book in strategy_books for each_cost in cost_grid
                         if book.get_cost_scope() is None or book.get_cost_scope() == each_cost])
