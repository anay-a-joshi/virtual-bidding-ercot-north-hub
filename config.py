# ==============================================================================
# Name    : Anay Abhijit Joshi
# Date    : November 22, 2026
# Course  : ISyE 6767 - Interim Project 2 (Virtual Bidding at ERCOT HB_NORTH)
# File    : config.py
# Purpose : One single place for every constant of the project, so that no
#           other file ever hard-codes a path, a date, a cost or a budget.
# ==============================================================================

from pathlib import Path

# So, everything is located relative to this file, so the project runs from any folder.
PROJECT_HOME_DIR = Path(__file__).resolve().parent
RAW_ERCOT_DIR = PROJECT_HOME_DIR / "lab2_data"
PRICE_WORKBOOK_DIR = RAW_ERCOT_DIR / "prices"
POSTING_TIMES_CSV = RAW_ERCOT_DIR / "forecast_posting_times.csv"

# Let's do now the output folders (they are created on demand by ensure_output_folders).
OUTPUT_HOME_DIR = PROJECT_HOME_DIR / "outputs"
OUTPUT_DATA_DIR = OUTPUT_HOME_DIR / "data"
OUTPUT_TABLE_DIR = OUTPUT_HOME_DIR / "tables"
OUTPUT_FIGURE_DIR = OUTPUT_HOME_DIR / "figures"
OUTPUT_CACHE_DIR = OUTPUT_HOME_DIR / "cache"
HOURLY_DATASET_CSV = OUTPUT_DATA_DIR / "ercot_hb_north_hourly.csv"
DATA_QUALITY_REPORT_TXT = OUTPUT_TABLE_DIR / "data_quality_report.txt"

# ERCOT market conventions.
TARGET_HUB_NAME = "HB_NORTH"
ERCOT_LOCAL_TZ = "US/Central"          # Central Prevailing Time (CST in winter, CDT in summer)

# Decision time: all trades for day D are decided at 9:00 AM local on day D-1.
DECISION_LEAD_DAYS = 1
DECISION_LOCAL_HOUR = 9
# In my opinion, the cleanest way to say "price information must be D-2 or older".
MIN_PRICE_LAG_DAYS = 2

# Delivery window shipped in lab2_data (December 2022 only feeds the lags).
FIRST_DELIVERY_DAY = "2022-12-01"
LAST_DELIVERY_DAY = "2025-12-31"

# Train / validate / test split (by delivery year). Every choice is made on 2024.
TRAIN_YEAR = 2023
VALIDATION_YEAR = 2024
TEST_YEAR = 2025

# Forecast products: folder -> (date column, hour-ending column, {source column: output name}).
# Finally, only FORECAST columns are listed here; the ACTUAL / COP_HSL / GRPP columns are
# realized or non-forecast values and are never read into the dataset.
FORECAST_PRODUCT_SPECS = {
    "load_forecast": ("DeliveryDate", "HourEnding",
                      {"SystemTotal": "load_fcst_system_mw", "North": "load_fcst_north_mw"}),
    "wind_forecast": ("DELIVERY_DATE", "HOUR_ENDING",
                      {"STWPF_SYSTEM_WIDE": "wind_fcst_mw"}),
    "solar_forecast": ("DELIVERY_DATE", "HOUR_ENDING",
                       {"STPPF_SYSTEM_WIDE": "solar_fcst_mw"}),
}

# Trading parameters used in Parts 1-3.
TRANSACTION_COST_GRID = (0.0, 1.0)            # $/MWh, report both
TAU_THRESHOLD_GRID = (0.0, 1.0, 2.0, 5.0, 10.0)
DAILY_BUDGET_MWH = 24.0
MAX_HOURLY_MWH = 4.0

# Capital base for the annualized rate of return (the PDF does not define one).
# So, one fixed number for EVERY strategy: the dollars needed to survive a full day of the
# maximum budget (24 MWh) at the training-year 99th percentile of |S|. Set from 2023 only.
CAPITAL_ABS_SPREAD_QUANTILE = 0.99
DAYS_PER_YEAR = 365.0

# Part 1 label variants.
THREE_CLASS_LARGE_SPREAD = 10.0       # |S| > $10 is a "large" move (about the top quarter of hours)
THRESHOLD_LABEL_MIN_ABS_SPREAD = 5.0  # "only label hours with |S| above a threshold" (about half the hours)

# Part 2 / Part 3 model and rule grids (every choice made on validation = 2024).
HUBER_DELTA_DOLLARS = 10.0
QUANTILE_LEVELS = (0.1, 0.5, 0.9)
RAMP_RULE_QUANTILES = (0.80, 0.90, 0.95)
MEAN_VARIANCE_LAMBDAS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
# So, lambda is a risk preference: the Sharpe ratio is almost scale-free in it (it keeps rising as the
# positions shrink) and profit keeps rising as lambda -> 0. We therefore pick the MOST PROFITABLE lambda on
# 2024 whose 2024 maximum drawdown stays within this share of the capital base (a risk budget).
MEAN_VARIANCE_DRAWDOWN_BUDGET_SHARE = 0.20
CVAR_LOSS_LIMITS = (5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0)
CVAR_CONFIDENCE = 0.95
# So, on 2024 the Sharpe ratio is monotone in the CVaR limit L (no interior optimum), so L is set as a
# risk-appetite policy instead: daily CVaR95 loss <= 5% of the capital base. The whole grid above is
# still evaluated on 2024 and reported as the risk-return frontier.
CVAR_LIMIT_SHARE_OF_CAPITAL = 0.05
RISK_LOOKBACK_DAYS = 56               # history ending D-2 for covariance / CVaR scenarios
COLLATERAL_SIGMA_MULTIPLE = 2.0       # collateral per MWh = 2 x (28-day same-hour spread std)

# So, Part 3 volumes are floored to a minimum tradable increment (an assumption, stated in the report);
# without it the optimisers produce thousands of 0.001 MWh "trades" that inflate trade counts.
VOLUME_INCREMENT_MWH = 0.1

# Extra output folders for Parts 1-3.
OUTPUT_PREDICTION_DIR = OUTPUT_HOME_DIR / "predictions"
OUTPUT_TRADES_DIR = OUTPUT_HOME_DIR / "trades"

# Reproducibility.
GLOBAL_RANDOM_SEED = 6767


def ensure_output_folders():
    # So, let's create every output folder once, quietly, if it does not exist yet.
    for each_folder in (OUTPUT_DATA_DIR, OUTPUT_TABLE_DIR, OUTPUT_FIGURE_DIR, OUTPUT_CACHE_DIR,
                        OUTPUT_HOME_DIR / "predictions", OUTPUT_HOME_DIR / "trades"):
        each_folder.mkdir(parents=True, exist_ok=True)
