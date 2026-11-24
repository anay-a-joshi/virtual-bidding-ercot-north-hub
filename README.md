# Interim Project II: Virtual Bidding Strategy in Power Markets: A Case with ERCOT North Hub

**Name:** Anay Abhijit Joshi
**Course:** ISyE/MATH 6767, Design and Implementation of Systems to Support Computational Finance (Georgia Institute of Technology, Fall 2026)
**Date:** November 2026

We predict the hourly DART spread at ERCOT's North Hub, S = P_DA − P_RT, and turn the predictions into
virtual INC/DEC trades. Every trade for delivery day D is decided at **9:00 AM on D−1**, using only
information that existed at that moment. Train = 2023, validation = 2024 (every modeling choice),
test = 2025 (scored once, at the end). The full write-up is in `Project_2_Anay_Joshi.pdf`.

---

## 1. Setup (Python virtual environment)

```bash
cd ~/Desktop/Interim_Project_2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Python 3.11 or newer is required (pandas 3); the results in the report were produced with Python 3.12 on macOS.
On macOS, if LightGBM fails to import, run `brew install libomp` (LightGBM needs OpenMP).
The raw ERCOT files must be unzipped in `lab2_data/` next to the code (folder layout of `lab2_data.zip`).

## 2. Run

```bash
python main.py --stage all --rebuild    # everything, from the raw files
python main.py --stage all              # same, reusing cached price tables and models (179 s)
python unit_tests.py                    # 89 unit tests, prints "Unit Tests Passed: 89, Failed: 0"
```

Single stages: `--stage data | features | part1 | part2 | part3 | evaluate`.
Without `--rebuild` the price tables and fitted models are reused from `outputs/cache/`.
The hourly dataset `ercot_hb_north_hourly.csv` is regenerated from the raw files on every run.

## 3. Files

| File | What it does |
|---|---|
| `config.py` | Every constant: paths, time zone, decision time, data split, costs, τ grid, budget, model grids |
| `data_pipeline.py` | `ErcotDatasetBuilder`: raw DA/RT workbooks + forecast CSVs → `ercot_hb_north_hourly.csv`; data quality checks 1–5 |
| `lookahead_guard.py` | `assert_no_lookahead` (Lab 2, Task 5), hardened: 20 poison days, positive **and** negative garbage |
| `data_sanity.py` | `DataSanityInspector`: Lab 2 Task 6 (class balance, extremes, negative prices, spread by hour, tails) |
| `feature_factory.py` | `FeatureFactory`: 42 features in 7 groups, every one tested for look-ahead (check 6) |
| `model_suite.py` | `PredictionStore`: every classifier and regressor, fitted on 2023, tuned on 2024, predictions cached |
| `backtest_engine.py` | `StrategyBook`: π = v(aS − c) and every metric the PDF asks for |
| `direction_models.py` | Part 1 strategies: baselines, classifiers, heuristics, regress-then-sign, label variants |
| `magnitude_models.py` | Part 2 rules: τ selection, sizing at equal volume, edge estimates, sign/size dependence |
| `budget_allocator.py` | Part 3 algorithms: water-filling, knapsack (DP + greedy), collateral knapsack, mean-variance QP, CVaR LP |
| `run_part1.py`, `run_part2.py`, `run_part3.py` | Run each part: tables, neon figures, ablation, robustness |
| `final_evaluation.py` | All strategies on 2025, profit-concentration check, test-set trades CSV, `results_summary.json` |
| `neon_plots.py` | One shared neon figure style (colour-blind-checked palette, every axis labelled) |
| `main.py` | Single entry point |
| `unit_tests.py` | 89 tests: DST keys, every Lab 2 check value, hand-computed features and metrics, solvers vs brute force / cvxpy |
| `requirements.txt` | Python packages for the virtual environment |
| `lab2_student.ipynb` | Lab 2 notebook (Tasks 0–6, with outputs and written answers) |
| `Project_2_Anay_Joshi.pdf` | The report: answers to every Question, figures, baseline comparison, feature ablation, limitations |

## 4. Outputs

| Folder | Content |
|---|---|
| `outputs/data/` | `ercot_hb_north_hourly.csv` (27,048 × 14), `model_dataset.csv` (26,304 rows, 42 features) |
| `outputs/tables/` | `data_quality_report.txt`, every result table of Parts 1–3, `final_*.csv`, `results_summary.json` |
| `outputs/figures/` | 26 neon PNG figures (300 dpi) used by the report |
| `outputs/trades/` | `test_trades_2025_all_strategies.csv` (337,531 trades of 50 strategies: time, prediction, action, volume, spread, profit) + one CSV per headline strategy |
| `outputs/predictions/` | Every model's prediction for every hour of 2023–2025 |

## 5. Data pipeline and no look-ahead

* **One key per hour:** hour-ending → hour start in UTC (`ambiguous` = first copy is CDT, repeated copy is CST;
  spring-forward HE03 → NaT). 35,064 hours in 2022–2025, four 23-hour and four 25-hour days, no duplicates.
* **RT aggregation:** mean of the four 15-minute prices; every hour has exactly 4 intervals.
* **Forecast vintage:** all 3,381 files posted before 9:00 on D−1 (load 08:30, wind/solar 08:55). Only
  `SystemTotal`, `North`, `STWPF_SYSTEM_WIDE`, `STPPF_SYSTEM_WIDE` are read; ACTUAL/COP_HSL/GRPP never.
* **Completeness:** 3 filled values in total, the solar spring-forward night hours (ERCOT mislabels them), set to 0 MW.
* **Outliers:** reported, never deleted or clipped (max |S| = $3,923 on 2023-09-06); robustness lives in the models.
* **No look-ahead:** price features are D−2 or older and built with merges, never `shift`; the whole feature builder
  passes `assert_no_lookahead` on 20 days (DST days, split boundaries, leap day) with ± garbage.

## 6. Headline results on the 2025 test set (c = $1/MWh)

Each part's headline strategy is chosen on **2024** by annualized Sharpe ratio at c = $1, never on 2025.

| Part | Strategy | Profit | $/MWh | Sharpe | Max drawdown | Profit without 5 best days |
|---|---|---:|---:|---:|---:|---:|
| Baseline | `always_inc` | −$647 | −0.07 | −0.09 | $5,998 | −$6,114 |
| Part 1 | `three_class_logistic` | −$38 | 0.00 | −0.01 | $5,128 | −$5,500 |
| Part 2 | `classifier_filtered_q90_c1` | $2,388 | 2.19 | 0.48 | $3,750 | −$2,686 |
| Part 3 | `part3_Q3_mean_variance_c1` | $406 | 0.19 | 0.13 | $2,608 | −$2,019 |

Every headline strategy beats Always INC on 2025, but each one is negative once its five best days are removed,
so the report does not claim a reliable trading edge. The run prints these rows under
`FINAL EVALUATION, Step-Number 2`; they are saved in `outputs/tables/final_headline_strategies_test.csv`, and every
strategy, both costs and every metric are in `outputs/tables/final_all_strategies_metrics.csv`.

**Reproducibility note.** LightGBM runs with `deterministic=True` and a fixed seed (6767). Some 2024 choices are
near-ties (for example, the chosen View B model and τ beat the runner-up by $11 on 2024), so a different machine or
library version could select a different model. The numbers in the report come from the macOS run whose output is
in the report's Appendix B.

## 7. Requirements checklist (project2.pdf)

| Requirement | Where |
|---|---|
| Spread, RT 15-min averaging, c = 0 and c = 1 | `data_pipeline.py`, `backtest_engine.py`, every table |
| Decision time 9:00 on D−1, price lags ≥ D−2, `assert_no_lookahead` | `lookahead_guard.py`, `feature_factory.py` |
| Train 2023 / validate 2024 / test 2025 once | `model_suite.py`, every `run_part*.py` |
| All required features (+ extras) | `feature_factory.py`, `feature_catalog.csv` |
| Data checks 1–6 performed and reported | `data_quality_report.txt`, stage logs |
| Part 1 baselines, logistic + LightGBM, ≥ 2 framings (all four done), metrics, profits | `run_part1.py` |
| Part 2 Views A and B, linear + tree, 3 losses, quantile, MAE/RMSE, τ grid, sizing at same volume | `run_part2.py` |
| Part 3 Approach A (4 weights), knapsack, unequal-weight knapsack, risk term (mean-variance and CVaR) | `run_part3.py` |
| Every strategy: profit, MWh, $/MWh, trades, win rate, cumulative P&L, annualized return, Sharpe, drawdown | `final_evaluation.py` |
| Profit from a few extreme days? | `final_profit_concentration.csv`, `final_headline_and_concentration.png` |
| CSV of test-set trades | `outputs/trades/` |
| Answers to every Question, baseline comparison, feature ablation, limitations | `Project_2_Anay_Joshi.pdf`; `part1_feature_ablation.csv` |

## 8. Stated assumptions

* **Capital base** for annualized return (not defined in the PDF): 24 MWh × 99th percentile of |S| in 2023 = $16,962, the same for every strategy.
* **Mean-variance λ**: the Sharpe ratio keeps rising as λ grows (positions shrink) and profit keeps rising as λ → 0, so λ is the most profitable value on 2024 whose 2024 maximum drawdown stays within 20% of capital ($3,392). Chosen: λ = 0.01.
* **CVaR limit**: the 2024 Sharpe ratio has no interior optimum in L (at c = $1 it mostly rises with L; at c = $0 it is highest for the tightest L, where the rule hardly trades), so L is a risk-appetite policy: daily CVaR₉₅ ≤ 5% of capital ($848).
* **Volume increment**: Part 3 volumes are floored to 0.1 MWh (assumed minimum bid size).
* **Collateral example (Part 3, Q2)**: collateral per MWh = 2 × the 28-day same-hour spread std; daily budget = 24 × its 2023 median ($504).

> **Academic integrity:** This repository is my original work for ISyE/MATH 6767 at Georgia Tech, shared for reference only; copying or submitting any part of it as your own work is plagiarism, for which I bear no responsibility.

### Thank You!
![Profile Views](https://komarev.com/ghpvc/?username=anay-a-joshi&color=green)  
