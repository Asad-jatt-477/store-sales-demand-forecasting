#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retail Demand Forecasting System
Phase 1: Business Framing, Data Audit & Integrated Dataset
==========================================================

Client context : Corporacion Favorita, a large grocery retailer in Ecuador
                 with 54 stores and 33 product families.

Business problem
    Store replenishment is planned from forecasts of daily unit demand.
    Over-ordering wastes perishables (produce, meat, bakery); under-ordering
    leaves empty shelves, lost sales and lost customers.

What the system must produce
    A forecast of daily unit sales for every store x product family,
    16 days ahead. The planning team uses this to place orders.

    Forecast unit : store x product family x day (1,782 time series)
    Horizon       : 16 days (matches the replenishment planning window)
    Target        : `sales` = units sold (kg for weighed items)
    Primary users : Supply-chain planners, store operations, category managers
    Metrics (P4)  : RMSLE (benchmark), WAPE (business-facing), Forecast Bias

Project roadmap
    1 (this script) : Business framing, raw data audit, data quality gate,
                      integrated modelling table
    2               : Exploratory analysis and business insights
    3               : Feature engineering (calendar, lags, rolling, promotions)
    4               : Baselines, ML models, time-based backtesting
    5               : Business evaluation (WAPE by family/store, bias, inventory)
    6               : Deployment (API + dashboard)

Phase 1 rule: every cleaning decision below is based on a check run against
the data, never on an assumption.

Outputs (written to ./processed/):
    retail_demand_master.parquet     integrated modelling table
    store_day_transactions.parquet   footfall table for EDA only
    data_dictionary.csv              column meaning + forecast-time availability
    data_quality_report.csv          all PASS/FAIL checks
    phase1_audit_summary.json        key numbers

Usage:
    python 01_data_preparation.py
(run it from the folder that contains train.csv, or from a folder that has
 data/raw/train.csv)
"""

# =============================================================================
# 1. SETUP & CONFIGURATION
# =============================================================================
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 80)

SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path(".")

# Raw files: the folder that contains train.csv.
CANDIDATES = [Path("."), Path("data/raw"), Path("../data/raw"),
              SCRIPT_DIR, SCRIPT_DIR / "data" / "raw", SCRIPT_DIR.parent / "data" / "raw"]
RAW_DIR = next((p for p in CANDIDATES if (p / "train.csv").exists()), None)
assert RAW_DIR is not None, "train.csv not found - set RAW_DIR to your raw data folder"

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(exist_ok=True)

HORIZON = 16  # forecast horizon in days (business requirement)


def section(title):
    """Print a visible section header (replaces notebook markdown headings)."""
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def show(obj, title=None):
    """Script replacement for notebook display(): print frames in full."""
    if title:
        print(title)
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        print(obj.to_string())
    else:
        print(obj)


section("1. SETUP & CONFIGURATION")
print("Raw data folder :", RAW_DIR)
print("Output folder   :", OUT_DIR)

# =============================================================================
# 2. LOAD RAW FILES
# Every raw file is loaded as-is and never overwritten. All outputs go to
# processed/. sample_submission.csv is NOT loaded: it is only the Kaggle upload
# format (same ids as test.csv, all sales 0) and has no business value.
# =============================================================================
section("2. LOAD RAW FILES")

train = pd.read_csv(RAW_DIR / "train.csv", parse_dates=["date"],
                    dtype={"store_nbr": "int16", "onpromotion": "int32", "sales": "float64"})
test = pd.read_csv(RAW_DIR / "test.csv", parse_dates=["date"],
                   dtype={"store_nbr": "int16", "onpromotion": "int32"})
stores = pd.read_csv(RAW_DIR / "stores.csv", dtype={"store_nbr": "int16"})
oil = pd.read_csv(RAW_DIR / "oil.csv", parse_dates=["date"])
holidays = pd.read_csv(RAW_DIR / "holidays_events.csv", parse_dates=["date"])
transactions = pd.read_csv(RAW_DIR / "transactions.csv", parse_dates=["date"],
                           dtype={"store_nbr": "int16"})

raw = {"train": train, "test": test, "stores": stores, "oil": oil,
       "holidays_events": holidays, "transactions": transactions}


def describe_table(name, df):
    d = {"table": name, "rows": len(df), "columns": df.shape[1],
         "null_cells": int(df.isna().sum().sum()), "duplicate_rows": int(df.duplicated().sum())}
    if "date" in df:
        d["date_min"], d["date_max"] = df["date"].min().date(), df["date"].max().date()
    return d


inventory = pd.DataFrame([describe_table(n, d) for n, d in raw.items()])
show(inventory, "\nTable inventory:")

for name, df_ in raw.items():
    print(f"\n=== {name}  {df_.shape}")
    show(df_.head(3))

# =============================================================================
# 3. DATA QUALITY GATE (DATA CONTRACT)
# In production a forecasting pipeline must STOP when incoming data breaks its
# contract (duplicate keys, missing store, broken date range). Otherwise it
# silently produces bad orders. Each check records PASS or FAIL; the script
# stops if any check fails.
# =============================================================================
section("3. DATA QUALITY GATE (DATA CONTRACT)")

checks = []


def check(name, passed, detail=""):
    checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})


KEY = ["date", "store_nbr", "family"]
n_stores, n_fam = stores["store_nbr"].nunique(), train["family"].nunique()

check("train: key (date, store, family) unique", not train.duplicated(KEY).any())
check("test: key (date, store, family) unique", not test.duplicated(KEY).any())
check("train/test: no missing values", train.isna().sum().sum() == 0 and test.isna().sum().sum() == 0)
check("train: sales >= 0", (train["sales"] >= 0).all(), f"min={train['sales'].min()}")
check("train/test: onpromotion >= 0", (train["onpromotion"] >= 0).all() and (test["onpromotion"] >= 0).all())
check("stores: store_nbr unique", stores["store_nbr"].is_unique, f"{n_stores} stores")
check("train stores == stores.csv", set(train["store_nbr"]) == set(stores["store_nbr"]))
check("test stores/families == train", set(test["store_nbr"]) == set(train["store_nbr"])
      and set(test["family"]) == set(train["family"]), f"{n_fam} families")
check("test starts the day after train ends",
      test["date"].min() == train["date"].max() + pd.Timedelta(days=1),
      f"train end {train['date'].max().date()}, test start {test['date'].min().date()}")
check("test covers exactly the horizon", test["date"].nunique() == HORIZON, f"{test['date'].nunique()} days")
check("test is a complete store x family grid", len(test) == HORIZON * n_stores * n_fam, f"{len(test):,} rows")
check("test ids continue train ids", test["id"].min() == train["id"].max() + 1)
check("oil: one row per date", oil["date"].is_unique)
check("oil covers the test window", oil["date"].max() >= test["date"].max(), f"oil ends {oil['date'].max().date()}")
check("transactions: key (date, store) unique", not transactions.duplicated(["date", "store_nbr"]).any())
check("holidays: Local names all match a store city",
      set(holidays.query("locale == 'Local'")["locale_name"]) <= set(stores["city"]))
check("holidays: Regional names all match a store state",
      set(holidays.query("locale == 'Regional'")["locale_name"]) <= set(stores["state"]))
check("holidays: every transferred holiday has a Transfer row",
      holidays["transferred"].sum() == (holidays["type"] == "Transfer").sum(),
      f"{holidays['transferred'].sum()} transferred / {(holidays['type'] == 'Transfer').sum()} Transfer")

gate = pd.DataFrame(checks)
show(gate)
assert (gate["status"] == "PASS").all(), "Data quality gate failed - investigate before continuing"
print("All", len(gate), "data quality checks passed.")

# =============================================================================
# 4. BUSINESS-FOCUSED DATA AUDIT
# These findings decide how the modelling table is built. Each one would quietly
# distort a demand forecast if ignored.
#
# Reference period rule (no look-ahead): business reference facts (store opening
# dates, first-sale dates, never-sold series, promotion tracking start) are
# learned ONLY from data before the validation window. This mirrors production:
# on the day a forecast is made, the system only knows the past. The validation
# window is then a true out-of-time test.
# =============================================================================
section("4. BUSINESS-FOCUSED DATA AUDIT")

TRAIN_END = train["date"].max()
VALID_START = TRAIN_END - pd.Timedelta(days=HORIZON - 1)      # last 16 days of history = validation
ref = train[train["date"] < VALID_START]                        # reference data: strictly before validation
print("Validation window :", VALID_START.date(), "to", TRAIN_END.date())
print("Reference period  :", ref["date"].min().date(), "to", ref["date"].max().date(), f"({len(ref):,} rows)")

# -----------------------------------------------------------------------------
# 4.1 Missing calendar days
# Finding : the only missing dates are 25 December of each year (stores closed).
# Decision: add these dates back with sales = 0 and flag them (is_imputed_row).
#           Lag features such as "sales 7 days ago" need a continuous daily
#           calendar, otherwise they shift.
# -----------------------------------------------------------------------------
section("4.1 Missing calendar days")

full_train_days = pd.date_range(train["date"].min(), train["date"].max(), freq="D")
missing_days = full_train_days.difference(pd.DatetimeIndex(train["date"].unique()))
print("Calendar days in train period:", len(full_train_days))
print("Days present in train         :", train["date"].nunique())
print("Missing days                  :", [d.date() for d in missing_days])

# -----------------------------------------------------------------------------
# 4.2 Store opening dates
# Finding : several stores opened during the data period; before opening their
#           sales are recorded as 0.
# Decision: flag pre-opening days (is_pre_opening) and exclude them from
#           training. Those zeros mean "store did not exist", not "no demand".
# -----------------------------------------------------------------------------
section("4.2 Store opening dates")

store_day = train.groupby(["date", "store_nbr"], as_index=False)["sales"].sum()
store_open_date = (store_day[(store_day["sales"] > 0) & (store_day["date"] < VALID_START)]
                   .groupby("store_nbr")["date"].min()
                   .rename("store_open_date"))
late_openers = store_open_date[store_open_date > train["date"].min() + pd.Timedelta(days=1)]
print(f"Stores opened after the data starts: {len(late_openers)}")
show(late_openers.sort_values().to_frame())

# -----------------------------------------------------------------------------
# 4.3 Store closure days (after opening)
# Finding : a whole store selling zero units across all 33 families in a day
#           means the store was closed (1 January almost every year, plus
#           several long temporary shutdowns).
# Decision: flag these store-days (is_store_closed) and exclude from training.
#           A closure is not a drop in demand.
# -----------------------------------------------------------------------------
section("4.3 Store closure days (after opening)")

sd = store_day.merge(store_open_date, on="store_nbr")
closed = sd[(sd["sales"] == 0) & (sd["date"] >= sd["store_open_date"])].copy()
print("Store-days with zero total sales after opening:", len(closed))
print("\nMost common calendar days:")
show(closed["date"].dt.strftime("%m-%d").value_counts().head(5).to_frame("store_days"))

# Group consecutive closed days into closure episodes
closed = closed.sort_values(["store_nbr", "date"])
closed["episode"] = (closed.groupby("store_nbr")["date"].diff().dt.days != 1).cumsum()
episodes = (closed.groupby(["store_nbr", "episode"])["date"].agg(start="min", end="max", days="count")
            .reset_index(drop=False).drop(columns="episode"))
print("Closure episodes of 3+ days (renovation / temporary shutdown):")
show(episodes[episodes["days"] >= 3].sort_values("days", ascending=False))

# -----------------------------------------------------------------------------
# 4.3b Calendar closures: 25 December and 1 January (known in advance)
# Finding : no store reports sales on 25 December in any year. On 1 January
#           every store is closed except store 25 (Salinas, a beach town), which
#           opens every year. Store 36 opened once, in 2014, then was closed.
# Decision: is_closure_calendar_day = 25 Dec for every store, and 1 Jan for
#           stores closed on most past New Year's Days. These days are known in
#           advance, so this is a FORECAST-TIME RULE ("expect 0"), NOT a model
#           feature (closed days are excluded from training, so the model would
#           see the flag on only a handful of rows and learn the opposite).
#           A blanket "1 Jan = 0" rule would be wrong for store 25.
# -----------------------------------------------------------------------------
section("4.3b Calendar closures: 25 December and 1 January")

jan1 = (store_day[(store_day["date"].dt.strftime("%m-%d") == "01-01") & (store_day["date"] < VALID_START)]
        .merge(store_open_date.reset_index(), on="store_nbr"))
jan1 = jan1[jan1["date"] >= jan1["store_open_date"]]
print(jan1.assign(year=jan1["date"].dt.year, closed=jan1["sales"].eq(0))
      .groupby("year")["closed"].agg(stores="size", closed="sum").to_string())
print("\nStores that sold on 1 January:")
show(jan1[jan1["sales"] > 0].merge(stores[["store_nbr", "city"]], on="store_nbr")[["date", "store_nbr", "city", "sales"]])

# Per-store rule learned from history: closed on 1 January in most past years -> expected closed
jan1_closed_rate = jan1.groupby("store_nbr")["sales"].apply(lambda s: (s == 0).mean()).rename("jan1_closed_rate")
print(f"Stores closed on every past 1 January: {(jan1_closed_rate == 1).sum()} of {len(jan1_closed_rate)}")
show(jan1_closed_rate[jan1_closed_rate < 1].to_frame())

# -----------------------------------------------------------------------------
# 4.4 Product families that never sold / launched late
# Finding : 53 store x family combinations never sold a single unit (family not
#           stocked in that store). Several families only start selling in 2014
#           or later across ALL stores at once -> company-wide assortment or
#           recording change.
# -----------------------------------------------------------------------------
section("4.4 Product families that never sold / launched late")

series_stats = (ref.groupby(["store_nbr", "family"])["sales"].sum().rename("total_sales").to_frame()
                .join(ref[ref["sales"] > 0].groupby(["store_nbr", "family"])["date"]
                      .agg(first_sale_date="min", last_sale_date="max"))
                .reset_index())
dead = series_stats[series_stats["total_sales"] == 0]
print(f"Total series: {len(series_stats):,}")
print(f"Series that NEVER sold in the reference period (not in that store's assortment): {len(dead)}")
sold_in_valid = train[(train["date"] >= VALID_START) & (train["sales"] > 0)][["store_nbr", "family"]].drop_duplicates()
print("Of those, series that then sold in the validation window:",
      len(dead.merge(sold_in_valid, on=["store_nbr", "family"])))
show(dead["family"].value_counts().to_frame("dead_series"))

fam_launch = ref[ref["sales"] > 0].groupby("family")["date"].min()
print("Families whose first sale is after the data start:")
show(fam_launch[fam_launch > train["date"].min() + pd.Timedelta(days=1)].sort_values().to_frame("first_sale_date"))

# -----------------------------------------------------------------------------
# 4.4b Series that start late in one store: new listing or slow seller?
# A late first sale in one store means either the product was LISTED late (the
# earlier zeros = "not on the shelf") or it sells RARELY (earlier zeros = real
# zero demand). The behaviour AFTER the first sale tells them apart.
# Finding : most late-starting series stay at zero on most days even after the
#           first sale -> slow sellers, whose early zeros are real demand.
#           Only a small group sells regularly once it starts (new listings).
# Decision:
#   is_dead_series -> business rule: forecast = 0, not modelled.
#   is_pre_launch  -> excluded from training ONLY for (a) days before the
#                     family's company-wide launch and (b) days before a *new
#                     listing* started. Zeros of slow sellers are KEPT.
# -----------------------------------------------------------------------------
section("4.4b Late-start series: new listing or slow seller?")

series_stats = (series_stats
                .merge(store_open_date.reset_index(), on="store_nbr", how="left")
                .merge(fam_launch.rename("family_launch_date").reset_index(), on="family", how="left"))
after = ref.merge(series_stats[["store_nbr", "family", "first_sale_date"]], on=["store_nbr", "family"])
after = after[after["date"] >= after["first_sale_date"]]
zero_after = ((after["sales"] == 0).groupby([after["store_nbr"], after["family"]]).mean() * 100
              ).rename("zero_pct_after_first_sale").reset_index()
series_stats = series_stats.merge(zero_after, on=["store_nbr", "family"], how="left")

expected_start = series_stats[["store_open_date", "family_launch_date"]].max(axis=1)
series_stats["start_gap_days"] = (series_stats["first_sale_date"] - expected_start).dt.days

REGULAR_SELLER_MAX_ZERO_PCT = 50   # after its first sale, the item sells on most days
late = series_stats[series_stats["start_gap_days"] > 7]
normal = series_stats[series_stats["start_gap_days"] <= 7]
print("Median % zero-sales days AFTER first sale")
print(f"   series that started on time : {normal['zero_pct_after_first_sale'].median():.1f}%  ({len(normal)} series)")
print(f"   series that started late    : {late['zero_pct_after_first_sale'].median():.1f}%  ({len(late)} series)")

series_stats["is_new_listing"] = ((series_stats["start_gap_days"] > 7)
                                  & (series_stats["zero_pct_after_first_sale"] < REGULAR_SELLER_MAX_ZERO_PCT))
summary_late = (late.assign(kind=np.where(late["zero_pct_after_first_sale"] < REGULAR_SELLER_MAX_ZERO_PCT,
                                          "new listing (sells most days after start)",
                                          "slow seller (mostly zero even after start)"))
                .groupby("kind").agg(series=("family", "size"), zero_days_before_first_sale=("start_gap_days", "sum")))
show(summary_late)

# -----------------------------------------------------------------------------
# 4.4c Whole product families at zero across almost all stores
#      (data gaps & legal sales bans)
# If one family sells nothing in >= 90 % of the open stores on the same day,
# while it normally sells in most stores, that is not a demand signal: it is a
# data-recording gap or a legal ban on selling.
# Findings:
#  - 8 families share exactly the same empty months (all 2013, Feb 2014,
#    Apr-Jun 2014, Aug 2014, early 2015): sales were NOT RECORDED.
#  - LIQUOR, WINE, BEER is zero on almost every Sunday until May 2016 (Ecuador's
#    Sunday alcohol restriction, relaxed in May 2016); other zero days fall
#    around carnival/elections ("ley seca").
#  - PRODUCE is zero everywhere until mid-March 2013 (covered by is_pre_launch).
# Decision: is_family_unavailable, excluded from training. Thresholds are learned
#           from the reference period. Built from sales -> a row filter, never
#           a feature. Future legal bans belong to forecast-time rules.
# -----------------------------------------------------------------------------
section("4.4c Whole families at zero (data gaps & legal sales bans)")

open_store_days = store_day[store_day["sales"] > 0][["date", "store_nbr"]]            # store traded that day
live = series_stats.loc[series_stats["total_sales"] > 0, ["store_nbr", "family"]]   # series that ever sold
fam_day = (train.merge(open_store_days, on=["date", "store_nbr"])
                .merge(live, on=["store_nbr", "family"])
                .groupby(["family", "date"])["sales"]
                .agg(stores="size", zero_stores=lambda s: (s == 0).sum()).reset_index())
fam_day["zero_share"] = fam_day["zero_stores"] / fam_day["stores"]

# "Normally sold in most stores" is learned from the reference period only
typical_zero_share = fam_day[fam_day["date"] < VALID_START].groupby("family")["zero_share"].median()
NORMALLY_SOLD = typical_zero_share[typical_zero_share < 0.5].index
FAMILY_GAP_SHARE = 0.9
family_gap_days = fam_day[(fam_day["zero_share"] >= FAMILY_GAP_SHARE) & fam_day["family"].isin(NORMALLY_SOLD)][["family", "date"]]


def periods(dates):
    dates = dates.sort_values()
    run = (dates.diff().dt.days != 1).cumsum()
    return dates.groupby(run.values).agg(["min", "max", "count"])


report = []
for fam, g in family_gap_days.groupby("family"):
    long_runs = periods(g["date"]).query("count >= 3")
    report.append({"family": fam, "gap_days": len(g),
                   "% Sundays": round((g["date"].dt.dayofweek == 6).mean() * 100),
                   "periods of 3+ days": "; ".join(f"{a.date()}..{b.date()}" for a, b, _ in long_runs.values) or "-"})
pd.set_option("display.max_colwidth", 200)
show(pd.DataFrame(report).sort_values("gap_days", ascending=False))
pd.set_option("display.max_colwidth", 80)

liq_gap = family_gap_days.query("family == 'LIQUOR,WINE,BEER'")["date"].sort_values()
sundays = liq_gap[liq_gap.dt.dayofweek == 6]
weekly = sundays[sundays.diff().dt.days.eq(7) | sundays.diff(-1).dt.days.eq(-7)]   # part of a week-after-week run
last_weekly = weekly.max()
print("\nWeekly zero-liquor Sundays run until:", last_weekly.date(), f"({len(weekly)} Sundays)")
liq_open = train[train["family"] == "LIQUOR,WINE,BEER"].merge(open_store_days, on=["date", "store_nbr"])
liq_sun = liq_open[liq_open["date"].dt.dayofweek == 6]
print("Store-Sundays with zero liquor sales  - up to that date:",
      f"{(liq_sun.loc[liq_sun['date'] <= last_weekly, 'sales'] == 0).mean():.1%}",
      "| after:", f"{(liq_sun.loc[liq_sun['date'] > last_weekly, 'sales'] == 0).mean():.1%}")
print("Other zero-liquor days (not part of the weekly Sunday run):")
print([d.strftime("%Y-%m-%d (%a)") for d in liq_gap if d not in set(weekly)])

# -----------------------------------------------------------------------------
# 4.5 When did promotion tracking start?
# Finding : no promotions are recorded before this date; the share then rises
#           steadily every year. A large retailer running zero promotions for 15
#           months is unrealistic -> promotions were not tracked yet.
# Decision: is_promo_tracked. In Phase 3/4 either train after this date or treat
#           onpromotion = 0 before it as "unknown", not "no promotion".
# -----------------------------------------------------------------------------
section("4.5 When did promotion tracking start?")

promo_start = ref.loc[ref["onpromotion"] > 0, "date"].min()
print("First date with any promotion recorded:", promo_start.date())
promo_share = (train.assign(year=train["date"].dt.year)
               .groupby("year")["onpromotion"].apply(lambda s: (s > 0).mean() * 100).round(1))
show(promo_share.to_frame("% rows with promotion"))

# -----------------------------------------------------------------------------
# 4.6 Transactions: useful signal, but a leakage risk
# Finding : transactions (footfall) correlate strongly with sales but are NOT
#           known for future days.
# Decision: NOT joined into the modelling table (same-day transactions would be
#           target leakage). Saved as a separate store-day table for Phase 2 EDA;
#           may only be used later as lagged features (lag >= 16 days).
# -----------------------------------------------------------------------------
section("4.6 Transactions: useful signal, but a leakage risk")

sd_tx = store_day.merge(transactions, on=["date", "store_nbr"], how="left")
print("Store-days in train                     :", len(sd_tx))
print("Store-days with no transactions record  :", sd_tx["transactions"].isna().sum())
print("   ...of which store sold nothing (closed):", (sd_tx["transactions"].isna() & (sd_tx["sales"] == 0)).sum())
print("Correlation(transactions, store sales)  :", round(sd_tx[["transactions", "sales"]].corr().iloc[0, 1], 3))
print("Transactions available for the forecast window?", "transactions" in test.columns)

# -----------------------------------------------------------------------------
# 4.7 Oil price gaps
# Finding : oil is a trading-day series (Mon-Fri) with missing values; a naive
#           join leaves a third of the dates without a price.
# Decision: reindex to the full calendar and forward-fill (last known price on
#           weekends); the first day is back-filled.
# Cautions:
#  1. The strong LEVEL correlation almost disappears on CHANGES: oil fell while
#     sales grew, so it is mostly a shared trend, not cause and effect. Oil must
#     prove its value in Phase 4 (with vs without).
#  2. A future oil price is unknown at forecast time: only lagged oil is allowed.
# -----------------------------------------------------------------------------
section("4.7 Oil price gaps")

full_calendar = pd.date_range(train["date"].min(), test["date"].max(), freq="D")
print("Calendar days (train + test):", len(full_calendar))
print("Oil rows                     :", len(oil), "| missing values:", oil["dcoilwtico"].isna().sum())
print("Oil weekdays present (0=Mon) :", sorted(int(d) for d in oil["date"].dt.dayofweek.unique()))
naive = pd.DataFrame({"date": full_train_days}).merge(oil, on="date", how="left")
print("Train dates with NO oil price after a naive merge:", naive["dcoilwtico"].isna().sum())

# Is the oil-sales relationship real or just two trends moving in opposite directions?
oil_ff = oil.set_index("date")["dcoilwtico"].reindex(full_train_days).ffill().bfill()
weekly = pd.DataFrame({"sales": train.groupby("date")["sales"].sum(), "oil": oil_ff}).resample("W").mean()
print("Weekly correlation, levels              :", round(weekly["sales"].corr(weekly["oil"]), 3))
print("Weekly correlation, 4-week % changes    :", round(weekly["sales"].pct_change(4).corr(weekly["oil"].pct_change(4)), 3))

# -----------------------------------------------------------------------------
# 4.8 Holidays: why a direct join is wrong
# Business rules (dataset's documented meaning of each type):
#   Holiday with transferred=True -> officially moved, was a NORMAL day: dropped
#   Transfer   -> the day the holiday was actually observed: holiday
#   Bridge     -> extra day off for a long weekend: holiday
#   Additional -> days around a holiday (21-26 Dec, eves...). NOT days off, they
#                 are peak shopping days -> is_holiday_season_day (separate flag,
#                 its sales effect is the opposite of a closure day)
#   Work Day   -> Saturday made a working day to compensate a Bridge
#                 -> is_workday_override
#   Event      -> Mother's Day, Black Friday, World Cup, 2016 earthquake
#                 -> is_national_event (+ earthquake flag)
# National holidays apply to every store, Regional to stores in that STATE,
# Local to stores in that CITY. The table is aggregated to ONE row per
# (date, store) BEFORE joining, so no rows are duplicated.
# -----------------------------------------------------------------------------
section("4.8 Holidays: why a direct join is wrong")

hol_per_date = holidays.groupby("date").size()
rows_per_date = train.groupby("date").size()
extra_rows = int(((hol_per_date - 1).clip(lower=0) * rows_per_date).dropna().sum())
print("Dates with more than one holiday/event:", int((hol_per_date > 1).sum()))
print(f"Naive train.merge(holidays, on='date') would create {extra_rows:,} DUPLICATE rows")
print("\nType x locale:")
show(pd.crosstab(holidays["type"], holidays["locale"]))
print("Holidays inside the forecast window:")
show(holidays[holidays["date"].between(test["date"].min(), test["date"].max())])

# -----------------------------------------------------------------------------
# 4.8b Is every holiday the same? (test, not assumption)
# Finding : national holidays push sales in OPPOSITE directions (shopping rush vs
#           quiet vs closed), so one yes/no is_national_holiday flag averages the
#           effects and is a weak signal. Almost every holiday name occurs <= 5
#           times, so raw national_desc as a category would let a model memorise
#           3-5 examples.
# Decision: national_desc stays in the master table; in Phase 3 it becomes a
#           holiday-specific HISTORICAL EFFECT feature computed from past years
#           only (known in advance, leakage-free).
# -----------------------------------------------------------------------------
section("4.8b Is every holiday the same?")

# Effect of each national holiday/event on total company sales vs the same weekday 2-3 weeks before/after
daily = ref.groupby("date")["sales"].sum()
nat_names = (holidays[(holidays["locale"] == "National") & ~holidays["transferred"]]
             .groupby("date")["description"].agg(lambda s: " | ".join(sorted(s))))
rows = []
for d, name in nat_names.items():
    if d < pd.Timestamp("2014-01-01") or d not in daily.index:
        continue
    ref_days = [d + pd.Timedelta(weeks=k) for k in (-3, -2, 2, 3)]
    ref_days = [x for x in ref_days if x in daily.index and x not in nat_names.index]
    if ref_days:
        rows.append({"date": d, "holiday": name, "effect_%": (daily[d] / daily[ref_days].mean() - 1) * 100})
effects = pd.DataFrame(rows)
summary_h = (effects.groupby("holiday")["effect_%"].agg(times="size", avg_effect="mean").round(1)
             .sort_values("avg_effect"))
print(f"Distinct national holiday/event names: {len(summary_h)} | seen 5 times or fewer: {(summary_h['times'] <= 5).sum()}")
keep = ["Primer dia del ano", "Dia de la Madre", "Batalla de Pichincha", "Viernes Santo", "Carnaval",
        "Navidad-4", "Navidad-2", "Independencia de Cuenca", "Dia del Trabajo"]
show(summary_h.reindex([k for k in keep if k in summary_h.index]))

# -----------------------------------------------------------------------------
# 4.8c Did the 2016 earthquake really change demand?
# Finding : a real, measurable demand shock (company-wide rise after the quake,
#           much larger in the affected province Manabi).
# Decision: is_earthquake_period comes straight from the holidays file
#           ("Terremoto Manabi" events). It is 0 for every future day, so it is
#           SAFE as a feature: the model can learn "this spike was the
#           earthquake" instead of treating it as normal April demand.
# -----------------------------------------------------------------------------
section("4.8c Did the 2016 earthquake really change demand?")

quake_window = pd.date_range("2016-04-16", "2016-05-16")
base = pd.Series({d: daily[[d - pd.Timedelta(weeks=k) for k in (5, 6, 7, 8)]].mean() for d in quake_window})
lift = (daily[quake_window] / base - 1) * 100
print(f"Company-wide sales, 16-30 Apr 2016 vs same weekdays 5-8 weeks earlier: {lift[:'2016-04-30'].mean():+.1f}%")
print(f"Company-wide sales,  1-16 May 2016                                  : {lift['2016-05-01':].mean():+.1f}%")
manabi = store_day[store_day["store_nbr"].isin(stores.loc[stores["state"] == "Manabi", "store_nbr"])]
for s_nbr, grp in manabi.groupby("store_nbr"):
    g2 = grp.set_index("date")["sales"]
    pre, post = g2["2016-03-16":"2016-04-15"].mean(), g2["2016-04-16":"2016-05-16"].mean()
    if pre > 0:
        print(f"Store {s_nbr} (Manabi, the affected province): {pre:,.0f} -> {post:,.0f} units/day ({(post / pre - 1) * 100:+.0f}%)")

# =============================================================================
# 5. BUILD THE INTEGRATED MODELLING TABLE
# =============================================================================
section("5. BUILD THE INTEGRATED MODELLING TABLE")

# -----------------------------------------------------------------------------
# 5.1 Complete calendar spine
# Every store x family x day from the first training day to the last forecast
# day, including the missing 25 December dates.
# -----------------------------------------------------------------------------
section("5.1 Complete calendar spine")

families = sorted(train["family"].unique())
store_list = sorted(stores["store_nbr"].unique())
spine = pd.MultiIndex.from_product([full_calendar, store_list, families], names=KEY).to_frame(index=False)
spine["store_nbr"] = spine["store_nbr"].astype("int16")
EXPECTED_ROWS = len(full_calendar) * len(store_list) * len(families)
print(f"Spine rows: {len(spine):,} = {len(full_calendar)} days x {len(store_list)} stores x {len(families)} families")
assert len(spine) == EXPECTED_ROWS

# -----------------------------------------------------------------------------
# 5.2 Stack train + test (concat) and attach to the spine (left merge)
# -----------------------------------------------------------------------------
section("5.2 Stack train + test and attach to the spine")

history = pd.concat([train.assign(source="train"), test.assign(source="test")], ignore_index=True)
assert len(history) == len(train) + len(test)

df = spine.merge(history, on=KEY, how="left", validate="one_to_one")
assert len(df) == EXPECTED_ROWS

# Rows that exist only in the spine = the missing 25-Dec days (stores closed)
df["is_imputed_row"] = df["source"].isna().astype("int8")
imputed_dates = df.loc[df["is_imputed_row"] == 1, "date"].dt.date.unique()
assert set(imputed_dates) == set(missing_days.date), "Unexpected imputed dates"
df.loc[df["is_imputed_row"] == 1, ["sales", "onpromotion"]] = 0
df["source"] = df["source"].fillna("train")
df["id"] = df["id"].astype("Int64")
df["onpromotion"] = df["onpromotion"].astype("int32")
print("Imputed rows:", df["is_imputed_row"].sum(), "on dates", sorted(imputed_dates))

# -----------------------------------------------------------------------------
# 5.3 Time-based split
# The last 16 days of history become the validation window. It mirrors the real
# 16-day forecast exactly. A random split would leak future information.
# -----------------------------------------------------------------------------
section("5.3 Time-based split")

train_end, valid_start = TRAIN_END, VALID_START   # defined in section 4
df["split"] = np.select(
    [df["source"].eq("test"), df["date"] >= valid_start],
    ["test", "valid"], default="train")
df = df.drop(columns="source")
summary = df.groupby("split")["date"].agg(["min", "max", "nunique", "count"])
show(summary)

# -----------------------------------------------------------------------------
# 5.4 Store attributes (left merge on store_nbr)
# -----------------------------------------------------------------------------
section("5.4 Store attributes")

store_attr = stores.rename(columns={"type": "store_type", "cluster": "store_cluster"})
before = len(df)
df = df.merge(store_attr, on="store_nbr", how="left", validate="many_to_one")
assert len(df) == before and df[["city", "state", "store_type", "store_cluster"]].notna().all().all()
print("Store attributes attached - rows unchanged:", f"{len(df):,}")

# -----------------------------------------------------------------------------
# 5.5 Oil price (full calendar, forward-filled)
# -----------------------------------------------------------------------------
section("5.5 Oil price (full calendar, forward-filled)")

oil_daily = oil.set_index("date")["dcoilwtico"].reindex(full_calendar)
oil_was_missing = oil_daily.isna()
oil_daily = oil_daily.ffill().bfill()
oil_table = pd.DataFrame({"date": full_calendar,
                          "oil_price": oil_daily.values,
                          "is_oil_imputed": oil_was_missing.values.astype("int8")})
assert oil_table["oil_price"].notna().all()

before = len(df)
df = df.merge(oil_table, on="date", how="left", validate="many_to_one")
assert len(df) == before and df["oil_price"].notna().all()
print(f"Oil attached. Imputed oil days: {oil_table['is_oil_imputed'].sum()} of {len(oil_table)}")

# -----------------------------------------------------------------------------
# 5.6 Holidays & events (aggregated to one row per date / store, then merged)
# -----------------------------------------------------------------------------
section("5.6 Holidays & events")

HOLIDAY_TYPES = {"Holiday", "Transfer", "Bridge"}   # real days off
SEASON_TYPES = {"Additional"}                        # shopping days around a holiday - NOT days off

h = holidays[~holidays["transferred"]].copy()           # moved-away holidays were normal days
h["is_day_off"] = h["type"].isin(HOLIDAY_TYPES)
h["is_season"] = h["type"].isin(SEASON_TYPES)
h["is_workday"] = h["type"].eq("Work Day")
h["is_event"] = h["type"].eq("Event")
h["is_earthquake"] = h["description"].str.startswith("Terremoto")

# National level: applies to all stores -> one row per date
nat = h[h["locale"] == "National"]
nat_daily = nat.groupby("date").agg(
    is_national_holiday=("is_day_off", "max"),
    is_national_season=("is_season", "max"),
    is_workday_override=("is_workday", "max"),
    is_national_event=("is_event", "max"),
    is_earthquake_period=("is_earthquake", "max"),
    national_desc=("description", lambda s: " | ".join(sorted(s))),
).reset_index()

# Regional (state) and Local (city): map to the stores they apply to -> one row per (date, store)
reg = (h[(h["locale"] == "Regional") & h["is_day_off"]]
       .merge(stores[["store_nbr", "state"]], left_on="locale_name", right_on="state"))
loc = (h[(h["locale"] == "Local") & h["is_day_off"]]
       .merge(stores[["store_nbr", "city"]], left_on="locale_name", right_on="city"))
reg_daily = reg.groupby(["date", "store_nbr"]).size().gt(0).rename("is_regional_holiday").reset_index()
loc_daily = loc.groupby(["date", "store_nbr"]).size().gt(0).rename("is_local_holiday").reset_index()
# Regional/local "Additional" days (e.g. the eve of a city's foundation day)
sub = h[h["locale"].isin(["Regional", "Local"]) & h["is_season"]]
season_daily = (pd.concat([
        sub[sub["locale"] == "Regional"].merge(stores[["store_nbr", "state"]], left_on="locale_name", right_on="state"),
        sub[sub["locale"] == "Local"].merge(stores[["store_nbr", "city"]], left_on="locale_name", right_on="city")])
    [["date", "store_nbr"]].drop_duplicates().assign(is_subnational_season=True))
for t in (reg_daily, loc_daily, season_daily):
    t["store_nbr"] = t["store_nbr"].astype("int16")

for t in (nat_daily, reg_daily, loc_daily, season_daily):
    assert not t.duplicated([c for c in ("date", "store_nbr") if c in t]).any()

before = len(df)
df = (df.merge(nat_daily, on="date", how="left", validate="many_to_one")
        .merge(reg_daily, on=["date", "store_nbr"], how="left", validate="many_to_one")
        .merge(loc_daily, on=["date", "store_nbr"], how="left", validate="many_to_one")
        .merge(season_daily, on=["date", "store_nbr"], how="left", validate="many_to_one"))
assert len(df) == before, "Holiday merge changed the row count!"

flag_cols = ["is_national_holiday", "is_workday_override", "is_national_event",
             "is_earthquake_period", "is_regional_holiday", "is_local_holiday"]
df[flag_cols + ["is_national_season", "is_subnational_season"]] = (
    df[flag_cols + ["is_national_season", "is_subnational_season"]].fillna(False).astype("int8"))
df["is_holiday_season_day"] = df[["is_national_season", "is_subnational_season"]].max(axis=1).astype("int8")
df = df.drop(columns=["is_national_season", "is_subnational_season"])
flag_cols.append("is_holiday_season_day")
df["national_desc"] = df["national_desc"].fillna("")
df["is_holiday"] = df[["is_national_holiday", "is_regional_holiday", "is_local_holiday"]].max(axis=1).astype("int8")
print("Rows unchanged after holiday merge:", f"{len(df):,}")
show(df.groupby("split", observed=True)[flag_cols + ["is_holiday"]].sum())

# -----------------------------------------------------------------------------
# 5.7 Business status flags
# These flags say whether a zero in `sales` means NO DEMAND or NO OPPORTUNITY to
# sell. Training a model on the second kind teaches it the wrong lesson.
# -----------------------------------------------------------------------------
section("5.7 Business status flags")

# Store opening
df = df.merge(store_open_date.reset_index(), on="store_nbr", how="left", validate="many_to_one")
df["is_pre_opening"] = (df["date"] < df["store_open_date"]).astype("int8")

# Store closed (whole store sold nothing) - only observable in history
hist = df["split"] != "test"
store_total = df[hist].groupby(["date", "store_nbr"])["sales"].transform("sum")
df["is_store_closed"] = 0
df.loc[hist, "is_store_closed"] = ((store_total == 0) & (df.loc[hist, "is_pre_opening"] == 0)).astype("int8")
df["is_store_closed"] = df["is_store_closed"].astype("int8")

# Series assortment status
df = df.merge(series_stats[["store_nbr", "family", "first_sale_date"]],
              on=["store_nbr", "family"], how="left", validate="many_to_one")
df["is_dead_series"] = df["first_sale_date"].isna().astype("int8")
df = (df.merge(fam_launch.rename("family_launch_date").reset_index(), on="family", how="left", validate="many_to_one")
        .merge(series_stats[["store_nbr", "family", "is_new_listing"]], on=["store_nbr", "family"],
               how="left", validate="many_to_one"))
before_family_launch = df["date"] < df["family_launch_date"]
before_new_listing = df["is_new_listing"].fillna(False).astype(bool) & (df["date"] < df["first_sale_date"])
df["is_pre_launch"] = ((before_family_launch | before_new_listing)
                       & (df["is_pre_opening"] == 0) & (df["is_dead_series"] == 0)).astype("int8")
df = df.drop(columns=["family_launch_date", "is_new_listing"])

# Promotion tracking
df["is_promo_tracked"] = (df["date"] >= promo_start).astype("int8")

# Calendar closures known in advance (section 4.3b)
expected_jan1_closed = set(jan1_closed_rate[jan1_closed_rate >= 0.5].index)
is_dec25 = (df["date"].dt.month == 12) & (df["date"].dt.day == 25)
is_jan1 = (df["date"].dt.month == 1) & (df["date"].dt.day == 1)
df["is_closure_calendar_day"] = (is_dec25 | (is_jan1 & df["store_nbr"].isin(expected_jan1_closed))).astype("int8")

# Family-wide data gaps / legal sales bans (section 4.4c) - history only
df = df.merge(family_gap_days.assign(is_family_unavailable=1), on=["family", "date"], how="left", validate="many_to_one")
df["is_family_unavailable"] = df["is_family_unavailable"].fillna(0).astype("int8")
df.loc[df["split"] == "test", "is_family_unavailable"] = 0

# Rows a model may learn from (historical, real trading opportunity)
df["is_trainable"] = ((df["split"] != "test") & (df["is_pre_opening"] == 0) & (df["is_store_closed"] == 0)
                      & (df["is_dead_series"] == 0) & (df["is_pre_launch"] == 0)
                      & (df["is_family_unavailable"] == 0)).astype("int8")

status = df[hist][["is_imputed_row", "is_pre_opening", "is_store_closed", "is_dead_series",
                   "is_pre_launch", "is_family_unavailable", "is_trainable"]].sum().to_frame("rows")
status["% of history"] = (status["rows"] / hist.sum() * 100).round(2)
show(status)

# Evidence for separating is_holiday_season_day: average store-day sales on
# normal days, holiday-season days and real days off (trading days only).
print("\nEvidence for separating is_holiday_season_day:")
sd_effect = (df[df["is_trainable"] == 1]
             .groupby(["date", "store_nbr"], observed=True)
             .agg(store_sales=("sales", "sum"), season=("is_holiday_season_day", "max"), day_off=("is_holiday", "max"))
             .reset_index())
sd_effect["day_kind"] = np.select([sd_effect["season"] == 1, sd_effect["day_off"] == 1],
                                  ["holiday-season day (Additional)", "day-off holiday"], default="normal day")
effect = sd_effect.groupby("day_kind")["store_sales"].agg(["mean", "count"]).round(0)
effect["vs normal %"] = ((effect["mean"] / effect.loc["normal day", "mean"] - 1) * 100).round(1)
show(effect)

# -----------------------------------------------------------------------------
# 5.8 Transactions -> separate store-day table (EDA only)
# -----------------------------------------------------------------------------
section("5.8 Transactions -> separate store-day table (EDA only)")

store_day_tx = (df[hist].groupby(["date", "store_nbr"], as_index=False)
                .agg(store_sales=("sales", "sum"), is_store_closed=("is_store_closed", "max"),
                     is_pre_opening=("is_pre_opening", "max"))
                .merge(transactions, on=["date", "store_nbr"], how="left", validate="one_to_one"))
print("Store-day rows:", len(store_day_tx), "| missing transactions:", store_day_tx["transactions"].isna().sum())
print("Missing transactions on open store-days with sales:",
      int((store_day_tx["transactions"].isna() & (store_day_tx["store_sales"] > 0)).sum()))

# =============================================================================
# 6. FINAL VALIDATION GATE
# Before the dataset is handed to Phase 2, totals are reconciled back to the raw
# files. Like a financial reconciliation, nothing may be lost or double-counted.
# =============================================================================
section("6. FINAL VALIDATION GATE")

final_checks = []


def fcheck(name, passed, detail=""):
    final_checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})


fcheck("row count = full calendar grid", len(df) == EXPECTED_ROWS, f"{len(df):,}")
fcheck("key (date, store, family) unique", not df.duplicated(KEY).any())
fcheck("total sales reconcile to raw train",
       np.isclose(df.loc[hist, "sales"].sum(), train["sales"].sum()), f"{train['sales'].sum():,.2f}")
fcheck("total promotions reconcile to raw train+test",
       df["onpromotion"].sum() == train["onpromotion"].sum() + test["onpromotion"].sum())
fcheck("sales missing ONLY in test", df["sales"].isna().equals(df["split"].eq("test")))
fcheck("test rows = raw test", (df["split"] == "test").sum() == len(test))
fcheck("test ids preserved", set(df.loc[df["split"] == "test", "id"]) == set(test["id"]))
non_null = [c for c in df.columns if c not in ("sales", "id", "first_sale_date")]
fcheck("no unexpected missing values", df[non_null].notna().all().all())
fcheck("validation window = 16 days", df.loc[df["split"] == "valid", "date"].nunique() == HORIZON)
fcheck("no imputed rows in test", df.loc[df["split"] == "test", "is_imputed_row"].sum() == 0)
fcheck("reference facts use only pre-validation data (no look-ahead)",
       df["store_open_date"].max() < VALID_START and df["first_sale_date"].max() < VALID_START
       and promo_start < VALID_START, f"reference ends {(VALID_START - pd.Timedelta(days=1)).date()}")
christmas_eves = df["national_desc"].str.contains("Navidad-", regex=False)
fcheck("Additional days (e.g. Navidad-1..4) are season days, not days off",
       df.loc[christmas_eves, "is_national_holiday"].eq(0).all()
       and df.loc[christmas_eves, "is_holiday_season_day"].eq(1).all())

final_gate = pd.DataFrame(final_checks)
show(final_gate)
assert (final_gate["status"] == "PASS").all(), "Final validation failed"
print("Final dataset validated.")

# =============================================================================
# 7. OPTIMISE TYPES, DOCUMENT & SAVE
# =============================================================================
section("7. OPTIMISE TYPES, DOCUMENT & SAVE")

col_order = ["id", "date", "store_nbr", "family", "split", "sales", "onpromotion",
             "city", "state", "store_type", "store_cluster",
             "oil_price", "is_oil_imputed",
             "is_holiday", "is_national_holiday", "is_regional_holiday", "is_local_holiday",
             "is_holiday_season_day", "is_workday_override", "is_national_event", "is_earthquake_period", "national_desc",
             "store_open_date", "first_sale_date",
             "is_imputed_row", "is_pre_opening", "is_store_closed", "is_dead_series",
             "is_pre_launch", "is_promo_tracked", "is_closure_calendar_day", "is_family_unavailable", "is_trainable"]
assert set(col_order) == set(df.columns), set(df.columns) ^ set(col_order)
df = df[col_order].sort_values(KEY).reset_index(drop=True)

for c in ["family", "split", "city", "state", "store_type", "national_desc"]:
    df[c] = df[c].astype("category")
df["store_cluster"] = df["store_cluster"].astype("int8")
df["oil_price"] = df["oil_price"].astype("float32")

print(f"Final shape: {df.shape} | memory: {df.memory_usage(deep=True).sum() / 1e6:,.0f} MB")
df.info()

# Data dictionary - 'known_at_forecast_time' is the leakage guard for Phase 3
dictionary = [
    ("id", "Kaggle row id (missing for imputed 25-Dec rows)", "train/test", "yes"),
    ("date", "Calendar day", "spine", "yes"),
    ("store_nbr", "Store identifier", "train/test", "yes"),
    ("family", "Product family", "train/test", "yes"),
    ("split", "train / valid (last 16 days of history) / test (forecast window)", "derived", "yes"),
    ("sales", "TARGET: units sold (kg for weighed families); empty in test", "train", "NO - target"),
    ("onpromotion", "Number of items in the family on promotion", "train/test", "yes (promo plan is set in advance)"),
    ("city", "Store city", "stores", "yes"),
    ("state", "Store state/province", "stores", "yes"),
    ("store_type", "Store format A-E", "stores", "yes"),
    ("store_cluster", "Group of similar stores", "stores", "yes"),
    ("oil_price", "WTI oil price, forward-filled on non-trading days (level correlation is mostly shared trend)", "oil", "NO for future days - only lagged, and only if it helps in Phase 4"),
    ("is_oil_imputed", "1 if oil price was filled (weekend/market holiday/missing)", "derived", "yes"),
    ("is_holiday", "Any day-off holiday applying to this store", "holidays", "yes (calendar)"),
    ("is_national_holiday", "National day off (Holiday/Transfer/Bridge). Weak alone: holidays push sales in opposite directions (see 4.8b)", "holidays", "yes"),
    ("is_regional_holiday", "Day off in the store's state", "holidays", "yes"),
    ("is_local_holiday", "Day off in the store's city", "holidays", "yes"),
    ("is_holiday_season_day", "Shopping day around a holiday (type Additional) - not a day off", "holidays", "yes (calendar)"),
    ("is_workday_override", "Weekend day made a working day", "holidays", "yes"),
    ("is_national_event", "National event (Mother's Day, Black Friday, World Cup, earthquake...)", "holidays", "partly - earthquake was not foreseeable"),
    ("is_earthquake_period", "16-Apr-2016 earthquake and the following 30 days", "holidays", "yes - always 0 in the future; usable as a feature so the shock is not learned as normal demand"),
    ("national_desc", "Holiday identity. Phase 3: turn into a past-years holiday-effect feature; raw names are too rare to use directly", "holidays", "yes"),
    ("store_open_date", "First day the store recorded sales (learned from reference period only)", "derived", "yes"),
    ("first_sale_date", "First day this store x family sold in the reference period (empty = never sold)", "derived", "yes"),
    ("is_imputed_row", "Row added for a missing 25-Dec date", "derived", "yes"),
    ("is_pre_opening", "Date before the store opened", "derived", "yes"),
    ("is_store_closed", "Store sold nothing in any family that day. Built FROM sales -> filter only, never a feature", "derived", "NO - derived from the target"),
    ("is_dead_series", "Store x family never sold - forecast 0 by rule", "derived", "yes"),
    ("is_pre_launch", "Before the family's company-wide launch, or before a new listing started in this store", "derived", "yes"),
    ("is_promo_tracked", "Promotions are recorded from this date onward", "derived", "yes"),
    ("is_closure_calendar_day", "25 Dec for all stores; 1 Jan for stores historically closed that day (not store 25)", "derived", "yes - known calendar; forecast-time rule only, NOT a feature (almost never 1 in training rows)"),
    ("is_family_unavailable", "Family at zero in >=90% of open stores that day (data gap or legal sales ban, e.g. liquor on Sundays before May 2016)", "derived", "NO - derived from the target; filter only"),
    ("is_trainable", "Real selling opportunity in history - row filter for training. Built FROM sales -> never a feature", "derived", "NO - derived from the target"),
]
data_dict = pd.DataFrame(dictionary, columns=["column", "description", "source", "known_at_forecast_time"])
data_dict["dtype"] = data_dict["column"].map(df.dtypes.astype(str))
assert data_dict["column"].tolist() == df.columns.tolist()
show(data_dict)

try:
    df.to_parquet(OUT_DIR / "retail_demand_master.parquet", index=False)
    store_day_tx.to_parquet(OUT_DIR / "store_day_transactions.parquet", index=False)
    fmt = "parquet"
except ImportError:
    print("pyarrow not installed (pip install pyarrow) - saving as pickle instead")
    df.to_pickle(OUT_DIR / "retail_demand_master.pkl")
    store_day_tx.to_pickle(OUT_DIR / "store_day_transactions.pkl")
    fmt = "pickle"

hist = df["split"] != "test"   # recomputed after re-ordering
data_dict.to_csv(OUT_DIR / "data_dictionary.csv", index=False)
pd.concat([gate.assign(stage="raw_input"), final_gate.assign(stage="final_output")]) \
  .to_csv(OUT_DIR / "data_quality_report.csv", index=False)

audit_summary = {
    "rows_final": int(len(df)),
    "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
    "n_stores": int(df["store_nbr"].nunique()), "n_families": int(df["family"].nunique()),
    "n_series": int(df[["store_nbr", "family"]].drop_duplicates().shape[0]),
    "train_period": [str(df.loc[df.split == "train", "date"].min().date()), str(df.loc[df.split == "train", "date"].max().date())],
    "valid_period": [str(valid_start.date()), str(train_end.date())],
    "test_period": [str(test["date"].min().date()), str(test["date"].max().date())],
    "missing_dates_imputed": [str(d.date()) for d in missing_days],
    "late_opening_stores": {int(k): str(v.date()) for k, v in late_openers.items()},
    "dead_series": int(len(dead)),
    "promo_tracking_start": str(promo_start.date()),
    "reference_period_end": str((VALID_START - pd.Timedelta(days=1)).date()),
    "pct_history_trainable": round(float(df.loc[hist, "is_trainable"].mean() * 100), 2),
    "pct_zero_sales_trainable": round(float((df.loc[df.is_trainable == 1, "sales"] == 0).mean() * 100), 2),
    "output_format": fmt,
}
with open(OUT_DIR / "phase1_audit_summary.json", "w") as f:
    json.dump(audit_summary, f, indent=2)

print("\nFiles written to", OUT_DIR.resolve())
for p in sorted(OUT_DIR.iterdir()):
    print(f"{p.name:40s} {p.stat().st_size / 1e6:8.2f} MB")
print(json.dumps(audit_summary, indent=2))

# =============================================================================
# 8. PHASE 1 DECISION LOG
# -----------------------------------------------------------------------------
#  #  Finding                                   Decision
#  1  25 Dec missing every year                 Re-inserted with sales=0, flagged
#                                               (continuous calendar for lags)
#  2  Stores opened mid-history                 is_pre_opening, excluded from training
#  3  Whole-store zero days (1 Jan, renovation) is_store_closed, excluded from training
#  4  53 store x family series never sold       is_dead_series -> forecast 0 by rule
#  5  Families launched after 2013; late        is_pre_launch (family launch + NEW
#     listings in a store                       listings only); slow sellers' zeros
#                                               are kept (real demand)
#  6  No promotions before tracking start       is_promo_tracked (0 = "unknown")
#  7  Transactions unknown for future days      Kept out of model table (EDA file only)
#  8  Oil has weekend/holiday gaps              Forward-filled + flagged; lag it in P3
#  9  Holidays: multiple rows per date,         Rule-based processing -> one row per
#     transferred days                          date/store; no duplicated rows
# 10  Real forecast window = 16 days            Validation = last 16 days of history
# 11  "Additional" days = shopping peaks        Separate is_holiday_season_day flag
# 12  Reference facts could peek into valid.    Learned only before validation window
# 13  Oil-sales level corr. = shared trend      Oil kept as an UNPROVEN feature
# 14  Store 25 opens on 1 Jan                   is_closure_calendar_day learned per store
# 15  National holidays have opposite effects   Keep national_desc; past-years holiday-
#                                               effect feature in Phase 3
# 16  8 families share identical empty months;  is_family_unavailable, excluded from
#     liquor zero on Sundays until May 2016     training
#
# Forecast-time business rules (used in Phase 4-6)
#   Store not opened yet (date < store_open_date)  -> 0
#   Series never sold (is_dead_series)             -> 0
#   is_closure_calendar_day                        -> 0
#   Store closed for renovation (store ops info)   -> 0
#   Everything else                                -> model forecast
#
# Next: Phase 2, exploratory analysis and business insights.
# =============================================================================
print("\nPhase 1 complete. Next: Phase 2 - exploratory analysis and business insights.")
