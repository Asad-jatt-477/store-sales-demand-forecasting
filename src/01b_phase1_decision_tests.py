#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retail Demand Forecasting System
Phase 1b: Testing the Phase 1 Decisions
=======================================

Phase 1 made two decisions that shape every later phase. This script TESTS them
on the data instead of trusting them:

 1. No leakage : sales-history features for the validation window must not use
                 any validation-window sales.
 2. `is_trainable` filter : should a demand model learn from the zeros of closed
                 stores, not-yet-opened stores, never-sold series and
                 not-yet-launched products?

Method
    A real forecasting model (LightGBM) is trained twice with identical
    settings, once on ALL history rows (A) and once on `is_trainable == 1` rows
    only (B). Both are then scored on 16 days they never saw (out-of-time
    backtests).

Further tests in this script
    - Feature-group ablation (what really carries the accuracy)
    - Does `is_closure_calendar_day` match what really happened (in-sample and
      out-of-time on 1 January 2017)?
    - Is the holiday NAME real information, and is a yes/no flag enough?

Requirements
    pip install lightgbm pandas numpy pyarrow
    Input : processed/retail_demand_master.parquet   (from 01_data_preparation.py)
            holidays_events.csv                      (raw data folder)
    Runtime: roughly 5-15 minutes on a laptop (more on slow machines).

NOTE: the features here are a deliberately simple set built only to test these
decisions. Proper feature engineering is Phase 3.

Usage:
    python 01b_phase1_decision_tests.py
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as exc:  # clear message instead of a bare traceback
    raise SystemExit("lightgbm is required for this script: pip install lightgbm") from exc

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 200)


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


# =============================================================================
# 0. LOAD THE PHASE 1 MASTER TABLE
# =============================================================================
section("0. LOAD THE PHASE 1 MASTER TABLE")

MASTER_PATH = Path("data/processed/retail_demand_master.parquet")
if not MASTER_PATH.exists():
    raise SystemExit(f"{MASTER_PATH} not found - run 01_data_preparation.py first")

m = pd.read_parquet(MASTER_PATH)
m = m.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
print(m.shape)

# =============================================================================
# 1. FEATURES THAT ARE KNOWN 16 DAYS AHEAD
# On the forecast day (15 Aug) the system must predict 16 Aug to 31 Aug. For the
# last day (31 Aug), the most recent known sales day is 15 Aug, which is 16 days
# earlier. So every sales-history feature uses data that is AT LEAST 16 days old
# (lag_16, lag_21, ...). A "7 days ago" feature would be unknown for most of the
# 16 forecast days, and using it would be leakage.
# =============================================================================
section("1. FEATURES THAT ARE KNOWN 16 DAYS AHEAD")


def build_features(df):
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    g = df.groupby(["store_nbr", "family"], observed=True)["sales"]
    for lag in (16, 21, 28, 35, 42, 364):
        df[f"lag_{lag}"] = g.shift(lag)
    s16 = g.shift(16)
    keys = [df["store_nbr"], df["family"]]
    df["roll7_lag16"] = s16.groupby(keys, observed=True).transform(lambda x: x.rolling(7, min_periods=1).mean())
    df["roll28_lag16"] = s16.groupby(keys, observed=True).transform(lambda x: x.rolling(28, min_periods=1).mean())
    df["same_weekday_mean4"] = df[["lag_21", "lag_28", "lag_35", "lag_42"]].mean(axis=1)
    return df


HISTORY_FEATURES = ["lag_16", "lag_21", "lag_28", "lag_35", "lag_42", "lag_364",
                    "roll7_lag16", "roll28_lag16", "same_weekday_mean4"]
m = build_features(m)
print("History features built:", HISTORY_FEATURES)

# =============================================================================
# 2. LEAKAGE TEST
# Test: delete every sales value in the validation window, rebuild the features,
# and compare. If any validation-row feature changes, it was secretly using
# validation sales.
# =============================================================================
section("2. LEAKAGE TEST")

blind = m.drop(columns=HISTORY_FEATURES).copy()
blind.loc[blind["split"] == "valid", "sales"] = np.nan
blind = build_features(blind)
v = m["split"] == "valid"
same = all(np.allclose(m.loc[v, c].values, blind.loc[v, c].values, equal_nan=True) for c in HISTORY_FEATURES)
print("Validation-row features unchanged after deleting validation sales:", same)
assert same, "LEAKAGE: a feature uses validation-window sales"

example = m[(m["store_nbr"] == 1) & (m["family"] == "GROCERY I")].set_index("date")
show(example.loc[["2017-07-30", "2017-07-31", "2017-08-15", "2017-08-16"], ["split", "sales", "lag_16"]])

# Reading the example: the lag_16 value of 16 Aug (a TEST row) is the sales of
# 31 Jul (a VALID row), and lag_16 of 15 Aug (valid) is the sales of 30 Jul
# (train). The three splits are ONE continuous timeline. That is why train,
# valid and test sit in one file: split them into separate files and the first
# test days would lose their history features.

# =============================================================================
# 3. MODEL A vs MODEL B: OUT-OF-TIME BACKTESTS
# =============================================================================
section("3. MODEL A vs MODEL B: OUT-OF-TIME BACKTESTS")

for c in ["family", "store_type", "city"]:
    m[c] = m[c].astype("category")
m["store_cat"] = m["store_nbr"].astype("category")
m["dow"], m["dom"], m["month"] = m["date"].dt.dayofweek, m["date"].dt.day, m["date"].dt.month

FEATURES = (["store_cat", "family", "store_type", "store_cluster", "city", "onpromotion", "dow", "dom", "month",
             "is_holiday", "is_national_holiday", "is_local_holiday", "is_regional_holiday", "is_holiday_season_day",
             "is_workday_override", "is_national_event", "is_earthquake_period"]
            + HISTORY_FEATURES)
assert not {"is_store_closed", "is_trainable", "sales", "id"} & set(FEATURES)   # target-derived columns never used


def wape(y, p):
    return np.abs(y - p).sum() / y.sum() * 100


def rmsle(y, p):
    return np.sqrt(np.mean((np.log1p(np.clip(p, 0, None)) - np.log1p(y)) ** 2))


def backtest(train_end, eval_start, focus_stores=()):
    """Train model A (all rows) and model B (is_trainable rows) up to train_end
    and score both on the 16 days starting at eval_start."""
    train_end, eval_start = pd.Timestamp(train_end), pd.Timestamp(eval_start)
    ev = m[(m["date"] >= eval_start) & (m["date"] <= eval_start + pd.Timedelta(days=15))].copy()
    hist = m[(m["date"] >= "2014-06-01") & (m["date"] <= train_end)]
    for name, rows in [("A", hist), ("B", hist[hist["is_trainable"] == 1])]:
        model = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=127, min_child_samples=50,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
        model.fit(rows[FEATURES], np.log1p(rows["sales"]))
        ev["pred_" + name] = np.expm1(model.predict(ev[FEATURES])).clip(0)
    ev["pred_B"] = np.where(ev["is_dead_series"] == 1, 0, ev["pred_B"])                       # rule: never sold -> 0
    ev["pred_B_rules"] = np.where((ev["is_pre_opening"] == 1) | (ev["is_closure_calendar_day"] == 1), 0, ev["pred_B"])
    ev["baseline"] = ev["roll28_lag16"].fillna(0)

    open_rows = (ev["is_pre_opening"] == 0) & (ev["is_store_closed"] == 0)
    res = []
    for col, label in [("baseline", "Baseline: average of last 28 known days"),
                       ("pred_A", "A: trained on ALL rows"),
                       ("pred_B", "B: trained on is_trainable rows"),
                       ("pred_B_rules", "B + known-calendar rules")]:
        res.append({"model": label,
                    "RMSLE all rows": rmsle(ev["sales"], ev[col]),
                    "RMSLE open stores": rmsle(ev.loc[open_rows, "sales"], ev.loc[open_rows, col]),
                    "WAPE % open stores": wape(ev.loc[open_rows, "sales"], ev.loc[open_rows, col])})
    print(f"Train until {train_end.date()} | evaluate {eval_start.date()} + 15 days | "
          f"rows where the store was closed or not yet open: {(~open_rows).sum():,}")
    show(pd.DataFrame(res).round(4))
    for s in focus_stores:
        x = ev[ev["store_nbr"] == s]
        print(f"Store {s}: actual {x['sales'].sum():,.0f} units | A forecast {x['pred_A'].sum():,.0f} | "
              f"B forecast {x['pred_B'].sum():,.0f}")
    return model


# -----------------------------------------------------------------------------
# Backtest 1: the real validation window (31 Jul - 15 Aug 2017)
# -----------------------------------------------------------------------------
section("Backtest 1: the real validation window (31 Jul - 15 Aug 2017)")
t = time.time()
model_b = backtest("2017-07-30", "2017-07-31")
print(f"({time.time() - t:.0f}s)")

# -----------------------------------------------------------------------------
# Backtest 2: the 16 days after store 25 reopened
#             (store 18 still closed, store 52 not yet open)
# -----------------------------------------------------------------------------
section("Backtest 2: 16 days after store 25 reopened (store 18 closed, store 52 not yet open)")
backtest("2016-10-26", "2016-10-27", focus_stores=[25])

# -----------------------------------------------------------------------------
# Backtest 3: the 16 days after store 18 reopened (store 52 not yet open)
# -----------------------------------------------------------------------------
section("Backtest 3: 16 days after store 18 reopened (store 52 not yet open)")
backtest("2016-12-02", "2016-12-03", focus_stores=[18])

# =============================================================================
# 4. WHAT DOES THE MODEL RELY ON?
# =============================================================================
section("4. WHAT DOES THE MODEL RELY ON?")

imp = pd.Series(model_b.booster_.feature_importance("gain"), index=FEATURES)
show((imp / imp.sum() * 100).sort_values(ascending=False).round(2).to_frame("% of total gain").head(15))

# -----------------------------------------------------------------------------
# 4b. Importance numbers can mislead: remove each feature group and measure the
#     damage.
# "Gain" importance says how often a feature was used to split, not how much
# accuracy would be lost without it. The honest test is ABLATION: train the same
# model without one group of features and score it on the validation window. If
# the error rises, that group carries real information.
# -----------------------------------------------------------------------------
section("4b. Ablation: remove each feature group and measure the damage")

check_closure = m.loc[m["is_trainable"] == 1, "is_closure_calendar_day"].sum()
print(f"Training rows with is_closure_calendar_day = 1: {check_closure} (so it is a forecast rule, not a feature)")

CAL = ["dow", "dom", "month"]
WHO = ["store_cat", "family", "store_type", "store_cluster", "city"]
PROMO = ["onpromotion"]
HOL = ["is_holiday", "is_national_holiday", "is_local_holiday", "is_regional_holiday", "is_holiday_season_day",
       "is_workday_override", "is_national_event", "is_earthquake_period"]
ALL = HISTORY_FEATURES + CAL + WHO + PROMO + HOL
groups = {"All features": ALL,
          "Only sales history": HISTORY_FEATURES,
          "No sales history": CAL + WHO + PROMO + HOL,
          "All minus calendar": [f for f in ALL if f not in CAL],
          "All minus store/family": [f for f in ALL if f not in WHO],
          "All minus promotion": [f for f in ALL if f not in PROMO]}
ev = m[m["split"] == "valid"]
tr = m[(m["date"] >= "2014-06-01") & (m["split"] == "train") & (m["is_trainable"] == 1)]
abl = []
for label, feats in groups.items():
    mdl = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.07, num_leaves=127, min_child_samples=50,
                            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
    mdl.fit(tr[feats], np.log1p(tr["sales"]))
    p = np.where(ev["is_dead_series"] == 1, 0, np.expm1(mdl.predict(ev[feats])).clip(0))
    abl.append({"features": label, "WAPE %": wape(ev["sales"], p), "RMSLE": rmsle(ev["sales"], p)})
show(pd.DataFrame(abl).round(4))

# =============================================================================
# 5. TEST: DOES `is_closure_calendar_day` MATCH WHAT REALLY HAPPENED?
# In-sample (all history) and out-of-time: the 1 January rule is learned from
# 2013-2016 only and then checked on 1 January 2017.
# =============================================================================
section("5. Does is_closure_calendar_day match what really happened?")

sd = (m[(m["split"] != "test") & (m["is_pre_opening"] == 0)]
      .groupby(["date", "store_nbr"], observed=True)
      .agg(sales=("sales", "sum"), flag=("is_closure_calendar_day", "max")).reset_index())
cal = sd[sd["date"].dt.strftime("%m-%d").isin(["12-25", "01-01"])]
print(f"25-Dec / 1-Jan store-days: {len(cal)} | actually closed: {(cal['sales'] == 0).sum()}")
print(f"  closed and flagged: {((cal['sales'] == 0) & (cal['flag'] == 1)).sum()} | "
      f"closed but NOT flagged: {((cal['sales'] == 0) & (cal['flag'] == 0)).sum()} | "
      f"open but flagged: {((cal['sales'] > 0) & (cal['flag'] == 1)).sum()}")

jan1 = sd[sd["date"].dt.strftime("%m-%d") == "01-01"]
rate = jan1[jan1["date"] < "2017-01-01"].groupby("store_nbr")["sales"].apply(lambda s: (s == 0).mean())
j17 = jan1[jan1["date"] == "2017-01-01"].copy()
j17["predicted_closed"] = j17["store_nbr"].isin(rate[rate >= 0.5].index)
j17["actually_closed"] = j17["sales"] == 0
print(f"Out-of-time, 1 Jan 2017: {len(j17)} stores | correct {(j17['predicted_closed'] == j17['actually_closed']).sum()} "
      f"| wrong {(j17['predicted_closed'] != j17['actually_closed']).sum()}")

# =============================================================================
# 6. TEST: IS THE HOLIDAY *NAME* REAL INFORMATION, AND IS A YES/NO FLAG ENOUGH?
# Step 1, consistency: if a holiday's effect on sales repeats year after year,
# its name carries real, predictable information. If it jumps randomly, it is
# noise.
# =============================================================================
section("6. Is the holiday name real information? (Step 1: consistency)")

RAW_DIR = next((p for p in [Path("."), Path("data/raw"), Path("../data/raw")]
                if (p / "holidays_events.csv").exists()), None)
assert RAW_DIR is not None, "holidays_events.csv not found - set RAW_DIR to your raw data folder"
hol = pd.read_csv(RAW_DIR / "holidays_events.csv", parse_dates=["date"])
nat = hol[(hol["locale"] == "National") & ~hol["transferred"]]
names = nat.groupby("date")["description"].agg(lambda s: " | ".join(sorted(s)))
daily = m[m["split"] != "test"].groupby("date")["sales"].sum()

rows = []
for d, name in names.items():
    if d < pd.Timestamp("2013-02-01") or d not in daily.index:
        continue
    ref = [d + pd.Timedelta(weeks=k) for k in (-3, -2, 2, 3)]
    ref = [x for x in ref if x in daily.index and x not in names.index]
    if ref:
        rows.append({"date": d, "year": d.year, "holiday": name, "effect_%": (daily[d] / daily[ref].mean() - 1) * 100})
eff = pd.DataFrame(rows).sort_values("date")
# For each occurrence: the average effect of the SAME holiday in EARLIER years only (no look-ahead)
eff["effect_prev_years"] = eff.groupby("holiday")["effect_%"].transform(lambda s: s.shift().expanding().mean())

SHOW_HOLIDAYS = ["Navidad-3", "Navidad-2", "Navidad-1", "Dia del Trabajo", "Dia de la Madre", "Black Friday",
                 "Carnaval", "Viernes Santo", "Dia de Difuntos"]
show(eff.pivot_table(index="holiday", columns="year", values="effect_%").round(0).reindex(SHOW_HOLIDAYS))
pairs = eff.dropna(subset=["effect_prev_years"])
print(f"Holiday occurrences with earlier years available: {len(pairs)}")
print(f"Correlation (this year's effect vs earlier years' average): {pairs['effect_%'].corr(pairs['effect_prev_years']):.2f}")
print(f"Average error predicting the effect from earlier years: {(pairs['effect_%'] - pairs['effect_prev_years']).abs().mean():.1f} points"
      f" | assuming every holiday has the same (zero) effect: {pairs['effect_%'].abs().mean():.1f} points")

# -----------------------------------------------------------------------------
# Step 2, model test: three models with identical settings, scored on three
# 16-day windows that contain holidays:
#   M0: no holiday information at all.
#   M1: only the yes/no holiday flags.
#   M2: the flags plus `holiday_effect_prev_years`, the average effect of THIS
#       holiday in earlier years (leakage-free).
# -----------------------------------------------------------------------------
section("6. Step 2: model test M0 / M1 / M2 on three holiday windows")

m = m.merge(eff[["date", "effect_prev_years"]].rename(columns={"effect_prev_years": "holiday_effect_prev_years"}),
            on="date", how="left")
m["has_named_holiday"] = m["date"].isin(eff["date"])
m.loc[~m["has_named_holiday"], "holiday_effect_prev_years"] = 0.0

BASE = ["store_cat", "family", "store_type", "store_cluster", "city", "onpromotion", "dow", "dom", "month"] + HISTORY_FEATURES
FLAGS = ["is_holiday", "is_national_holiday", "is_local_holiday", "is_regional_holiday", "is_holiday_season_day",
         "is_workday_override", "is_national_event", "is_earthquake_period"]
SETS = {"M0 no holiday info": BASE, "M1 yes/no flags": BASE + FLAGS,
        "M2 flags + holiday effect (past years)": BASE + FLAGS + ["holiday_effect_prev_years"]}
results = []
for train_end, eval_start in [("2015-12-14", "2015-12-15"), ("2016-12-14", "2016-12-15"), ("2017-04-27", "2017-04-28")]:
    es = pd.Timestamp(eval_start)
    ev = m[(m["date"] >= es) & (m["date"] <= es + pd.Timedelta(days=15)) & (m["is_trainable"] == 1)]
    tr = m[(m["date"] >= "2014-06-01") & (m["date"] <= train_end) & (m["is_trainable"] == 1)]
    hd = ev["has_named_holiday"].values
    for label, feats in SETS.items():
        mdl = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.07, num_leaves=127, min_child_samples=50,
                                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
        mdl.fit(tr[feats], np.log1p(tr["sales"]))
        p = np.expm1(mdl.predict(ev[feats])).clip(0)
        results.append({"window": f"{eval_start} +15d", "model": label,
                        "WAPE % holiday days": wape(ev["sales"].values[hd], p[hd]),
                        "WAPE % normal days": wape(ev["sales"].values[~hd], p[~hd]),
                        "WAPE % all days": wape(ev["sales"].values, p)})
show(pd.DataFrame(results).round(2))

# =============================================================================
# 7. HOW TO READ THESE RESULTS
# (author's run; small differences on another machine are normal)
# -----------------------------------------------------------------------------
# Filter decision (is_trainable):
#  - Open stores: model B has lower error than model A in all 3 backtests
#    (validation window WAPE ~ 12.8 % vs 13.6 %, against a 21.2 % baseline).
#    Zeros that mean "shut" or "not recorded" teach the model the wrong thing
#    about demand.
#  - Just-reopened stores are hard for every model. Store 18: A forecast ~ 44k
#    vs B ~ 183k, actual ~ 198k. Store 25: A ~ 32k vs B ~ 112k, actual ~ 201k.
#    B is much closer, but still under-forecasts store 25, so a planner should
#    review the first weeks after a reopening.
#  - Closed / not-yet-open stores: B was never shown a closed store, so it
#    forecasts normal sales there. That is handled by forecast-time rules, not
#    by the model. Backtest 3 (store 52 not opened yet): B + rules is best on
#    every metric. Backtest 2: store 18 was closed for renovation for the whole
#    window. This dataset has no renovation schedule, so the rule cannot be
#    applied, and A looks better on all-rows RMSLE only because it forecasts
#    near-zero everywhere for that store. In production the store-operations
#    calendar supplies this. It is a stated dataset limitation.
#
# Calendar closure flag: 397 of 397 real closures on 25 Dec / 1 Jan are
#    flagged, with 1 false flag (store 36 in 2014). Learned only from 2013-2016,
#    it predicts 1 Jan 2017 correctly for all 53 stores.
#
# Holidays:
#  - The holiday NAME carries real information: a holiday's effect this year
#    correlates 0.83 with its average effect in earlier years.
#  - A yes/no flag alone is weak: on holiday days it was WORSE than no holiday
#    information in 2 of 3 windows.
#  - Adding the past-years holiday effect (M2) gave the lowest holiday-day error
#    in all 3 windows. But on normal December days the holiday features slightly
#    increased error, so in the December windows the all-days error did not
#    improve. Phase 3 must keep testing this, not assume it.
#
# What the model relies on: gain importance gives about 90 % to three
#    sales-history features, but the ablation test (4b) shows that this number
#    hides the truth. Removing calendar, store/family or promotion information
#    each makes the forecast clearly worse, and a model WITHOUT any sales
#    history still beats the baseline. Accuracy comes from several groups of
#    information working together.
#
# Known caveat of backtests 2 and 3 (measured, small): the Phase 1 flags
#    (is_dead_series, is_pre_launch, family-gap thresholds) were learned from
#    data up to 30 Jul 2017. For the 2016 backtests a few series that first sold
#    AFTER the 2016 origin are therefore treated as already known: 0.33 %
#    (backtest 2) and 0.20 % (backtest 3) of their training rows. Backtest 1,
#    the real validation window, has no such issue. In Phase 4 every backtest
#    recomputes these flags from data before its own origin.
# =============================================================================
print("\nPhase 1b complete. All decision tests finished.")
