# ==============================================================================
# Name    : Anay Abhijit Joshi
# Date    : November 22, 2026
# Course  : ISyE 6767 - Interim Project 2 (Virtual Bidding at ERCOT HB_NORTH)
# File    : budget_allocator.py
# Purpose : Part 3. Split a daily budget of B = 24 MWh (at most 4 MWh per hour)
#           across the hours of day D:
#             Approach A : v_h = min(4, alpha * w_h), alpha so that sum v = B
#             Approach B : knapsack with unit weights (dynamic programme + greedy)
#             Question 2 : knapsack with UNEQUAL weights (dollar collateral)
#             Question 3 : mean-variance  max v'(mu - c) - lambda/2 x' Sigma x
#                          CVaR limit     max v'(mu - c)  s.t.  CVaR_95(loss) <= L
#           All functions work on any number of hours (23, 24 or 25 on DST days).
# ==============================================================================

import numpy as np
from scipy.optimize import linprog

import config


# ---------------------------------------------------------------------- Approach A
def water_fill_allocation(hour_weights, budget_mwh=config.DAILY_BUDGET_MWH, cap_mwh=config.MAX_HOURLY_MWH):
    # So, v_h = min(cap, alpha * w_h) with alpha found by bisection so that sum v = B. If the hours with a
    # positive weight cannot absorb B even at the cap, every one of them gets the cap (budget not spent).
    hour_weights = np.clip(np.nan_to_num(np.asarray(hour_weights, dtype=float)), 0.0, None)
    if not np.any(hour_weights > 0):
        return np.zeros_like(hour_weights)
    if cap_mwh * np.count_nonzero(hour_weights) <= budget_mwh:
        return np.where(hour_weights > 0, cap_mwh, 0.0)
    alpha_low, alpha_high = 0.0, budget_mwh / hour_weights[hour_weights > 0].min() + 1.0
    for _ in range(200):
        alpha_mid = 0.5 * (alpha_low + alpha_high)
        if np.minimum(cap_mwh, alpha_mid * hour_weights).sum() > budget_mwh:
            alpha_high = alpha_mid
        else:
            alpha_low = alpha_mid
    return np.minimum(cap_mwh, alpha_low * hour_weights)


# ---------------------------------------------------------------------- Approach B
def unit_knapsack_greedy(unit_values, budget_mwh=config.DAILY_BUDGET_MWH, cap_mwh=config.MAX_HOURLY_MWH):
    # So, with equal weights (1 MWh) the knapsack is solved by sorting: fill the best hours to the cap,
    # stop when the budget is used or the next hour has non-positive value.
    unit_values = np.asarray(unit_values, dtype=float)
    volumes = np.zeros_like(unit_values)
    remaining = int(budget_mwh)
    for hour_position in np.argsort(-unit_values, kind="stable"):
        if remaining <= 0 or unit_values[hour_position] <= 0:
            break
        volumes[hour_position] = min(int(cap_mwh), remaining)
        remaining -= int(volumes[hour_position])
    return volumes


def bounded_knapsack_dp(unit_values, unit_weights, weight_budget, mwh_budget=int(config.DAILY_BUDGET_MWH),
                        cap_mwh=int(config.MAX_HOURLY_MWH)):
    # Let's do now the exact solver: a dynamic programme over (MWh used, weight used). Each hour picks
    # u in {0..cap} units, value u * e_h, weight u * k_h (integers). Works for unit weights too, which
    # is how the unit test proves the greedy answer above is optimal.
    unit_values = np.asarray(unit_values, dtype=float)
    unit_weights = np.asarray(unit_weights, dtype=int)
    if np.any(unit_weights < 0):
        raise ValueError("Oops! Knapsack weights must be non-negative integers.")
    weight_budget = int(weight_budget)
    best_value = np.zeros((mwh_budget + 1, weight_budget + 1))
    choice_record = np.zeros((len(unit_values), mwh_budget + 1, weight_budget + 1), dtype=np.int8)
    for hour_position, (value_per_unit, weight_per_unit) in enumerate(zip(unit_values, unit_weights)):
        if value_per_unit <= 0:
            continue    # So, an hour with no positive expected value is never chosen.
        new_value = best_value.copy()
        for units in range(1, cap_mwh + 1):
            mwh_shift, weight_shift = units, units * weight_per_unit
            if mwh_shift > mwh_budget or weight_shift > weight_budget:
                break
            candidate = np.full_like(best_value, -np.inf)
            candidate[mwh_shift:, weight_shift:] = best_value[:mwh_budget + 1 - mwh_shift,
                                                              :weight_budget + 1 - weight_shift] + units * value_per_unit
            improves = candidate > new_value
            new_value = np.where(improves, candidate, new_value)
            choice_record[hour_position][improves] = units
        best_value = new_value
    volumes = np.zeros(len(unit_values))
    mwh_left, weight_left = mwh_budget, weight_budget
    for hour_position in range(len(unit_values) - 1, -1, -1):
        units = int(choice_record[hour_position, mwh_left, weight_left])
        volumes[hour_position] = units
        mwh_left -= units
        weight_left -= units * int(unit_weights[hour_position])
    return volumes, float(best_value[mwh_budget, weight_budget])


def ratio_greedy_knapsack(unit_values, unit_weights, weight_budget, mwh_budget=config.DAILY_BUDGET_MWH,
                          cap_mwh=config.MAX_HOURLY_MWH):
    # So, the natural heuristic for unequal weights: best value per $ of collateral first. Not optimal.
    unit_values = np.asarray(unit_values, dtype=float)
    unit_weights = np.asarray(unit_weights, dtype=float)
    ratio = np.where(unit_weights > 0, unit_values / np.maximum(unit_weights, 1e-12), np.inf)
    volumes = np.zeros_like(unit_values)
    mwh_left, weight_left = mwh_budget, float(weight_budget)
    for hour_position in np.argsort(-ratio, kind="stable"):
        if unit_values[hour_position] <= 0:
            continue
        affordable = int(min(cap_mwh, mwh_left, np.floor(weight_left / max(unit_weights[hour_position], 1e-12))))
        if affordable > 0:
            volumes[hour_position] = affordable
            mwh_left -= affordable
            weight_left -= affordable * unit_weights[hour_position]
    return volumes


# ---------------------------------------------------------------------- Question 3: risk-aware
def project_onto_budget_box(raw_volumes, budget_mwh=config.DAILY_BUDGET_MWH, cap_mwh=config.MAX_HOURLY_MWH):
    # So, Euclidean projection onto {0 <= v <= cap, sum v <= B}: clip; if the sum is still too large, shift
    # every coordinate down by the same theta (found by bisection) and clip again.
    clipped = np.clip(raw_volumes, 0.0, cap_mwh)
    if clipped.sum() <= budget_mwh + 1e-12:
        return clipped
    theta_low, theta_high = 0.0, float(np.max(raw_volumes))
    for _ in range(100):
        theta_mid = 0.5 * (theta_low + theta_high)
        if np.clip(raw_volumes - theta_mid, 0.0, cap_mwh).sum() > budget_mwh:
            theta_low = theta_mid
        else:
            theta_high = theta_mid
    return np.clip(raw_volumes - theta_high, 0.0, cap_mwh)


def mean_variance_allocation(net_edges, directions, spread_covariance, risk_aversion, budget_mwh=config.DAILY_BUDGET_MWH,
                             cap_mwh=config.MAX_HOURLY_MWH, n_iterations=600):
    # Let's do now   max_v  e'v - (lambda/2) x' Sigma x,  x = a * v,  0 <= v <= 4,  sum v <= B
    # with accelerated projected gradient ascent (concave quadratic, step 1/L, L = lambda * max eigenvalue).
    net_edges = np.asarray(net_edges, dtype=float)
    signed_covariance = np.outer(directions, directions) * spread_covariance
    lipschitz = risk_aversion * max(float(np.linalg.eigvalsh(signed_covariance).max()), 1e-9)
    step_size = 1.0 / lipschitz
    current = project_onto_budget_box(np.where(net_edges > 0, 1.0, 0.0), budget_mwh, cap_mwh)
    momentum_point, momentum_weight = current.copy(), 1.0
    for _ in range(n_iterations):
        gradient = net_edges - risk_aversion * signed_covariance @ momentum_point
        next_point = project_onto_budget_box(momentum_point + step_size * gradient, budget_mwh, cap_mwh)
        next_weight = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum_weight ** 2))
        momentum_point = next_point + ((momentum_weight - 1.0) / next_weight) * (next_point - current)
        if np.max(np.abs(next_point - current)) < 1e-7:
            current = next_point
            break
        current, momentum_weight = next_point, next_weight
    return current


def mean_variance_objective(volumes, net_edges, directions, spread_covariance, risk_aversion):
    signed_position = np.asarray(directions) * np.asarray(volumes)
    return float(np.dot(net_edges, volumes) - 0.5 * risk_aversion * signed_position @ spread_covariance @ signed_position)


def cvar_limited_allocation(net_edges, directions, scenario_spreads, cost_per_mwh, loss_limit,
                            confidence=config.CVAR_CONFIDENCE, budget_mwh=config.DAILY_BUDGET_MWH,
                            cap_mwh=config.MAX_HOURLY_MWH):
    # So, the Rockafellar-Uryasev LP: maximise expected profit subject to CVaR_beta(daily loss) <= L, where
    # the loss scenarios are the last 56 days of hourly spreads (all ending D-2, so no look-ahead).
    # Variables: [v_1..v_n, eta, z_1..z_S].
    n_hours = len(net_edges)
    n_scenarios = scenario_spreads.shape[0]
    scenario_loss_per_mwh = cost_per_mwh - scenario_spreads * np.asarray(directions)[None, :]   # S x n
    objective = np.concatenate([-np.asarray(net_edges, dtype=float), [0.0], np.zeros(n_scenarios)])
    tail_rows = np.hstack([scenario_loss_per_mwh, -np.ones((n_scenarios, 1)), -np.eye(n_scenarios)])
    cvar_row = np.concatenate([np.zeros(n_hours), [1.0], np.full(n_scenarios, 1.0 / ((1.0 - confidence) * n_scenarios))])
    budget_row = np.concatenate([np.ones(n_hours), [0.0], np.zeros(n_scenarios)])
    inequality_matrix = np.vstack([tail_rows, cvar_row, budget_row])
    inequality_bounds = np.concatenate([np.zeros(n_scenarios), [loss_limit], [budget_mwh]])
    variable_bounds = [(0.0, cap_mwh)] * n_hours + [(None, None)] + [(0.0, None)] * n_scenarios
    solution = linprog(objective, A_ub=inequality_matrix, b_ub=inequality_bounds, bounds=variable_bounds, method="highs")
    if not solution.success:
        return np.zeros(n_hours)     # In my opinion "do not trade" is the only safe fallback.
    return np.clip(solution.x[:n_hours], 0.0, cap_mwh)
