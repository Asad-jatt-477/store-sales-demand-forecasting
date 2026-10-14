# =====================================================================================
# Retail Demand Forecasting System
# Phase 10a: Dashboard export -- project ke nateeje + data -> website ke liye chhoti JSON files
# -------------------------------------------------------------------------------------
# Koi naya model / faisla nahi. Sirf Phase 01-09 ke outputs ko website ke format mein likhna:
#   dashboard/public/data/*.json      -> charts, tables, KPIs (website parhti hai)
#   dashboard/public/data/series/*.json -> har store ki 33 families: history, forecast, backtests
#   dashboard/api/assets/*             -> live forecast API ke liye: final model + test window ke features
#   dashboard/requirements.txt         -> API ke Python packages (AAP KE computer ke versions, taa ke model load ho)
# Jaanch: API ke assets se dobara forecast bana kar Phase 08 ke forecast se milaya jata hai (farq ~0 hona chahiye).
#
# Chalane ka tareeqa (project folder se):   python src/10_export_dashboard.py
# Waqt: ~1-3 minute (koi training nahi; Phase 07/08/09 ke cache aur saved model se).
# =====================================================================================
import gzip
import json
import shutil
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

warnings.filterwarnings("ignore")
T0 = time.time()
PROCESSED, REPORT_DIR, MODEL_DIR = Path("data/processed"), Path("reports"), Path("models")
OUT = Path("dashboard/public/data")
SERIES_DIR = OUT / "series"
API_ASSETS = Path("dashboard/api/assets")
for d in (OUT, SERIES_DIR, API_ASSETS):
    d.mkdir(parents=True, exist_ok=True)
CACHE_07, CACHE_08, CACHE_09 = REPORT_DIR / "07_cache", REPORT_DIR / "08_cache", REPORT_DIR / "09_cache"
KEYS = ["store_nbr", "family"]


def section(t):
    print("\n" + "=" * 90 + f"\n{t}\n" + "=" * 90, flush=True)


def r(x, n=3):
    # JSON ke liye: numpy -> python, NaN -> None, round
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        return [r(v, n) for v in list(x)]
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return round(float(x), n)
    return x


def no_nan(x):
    # browser ka JSON.parse NaN/Infinity nahi parh sakta -> null
    if isinstance(x, dict):
        return {k: no_nan(v) for k, v in x.items()}
    if isinstance(x, list):
        return [no_nan(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


# website English mein hai: purani reports ki Roman Urdu keys/labels ka English
EN = {"sab families": "All families", "kharab hone wali": "Perishable", "baqi": "Non-perishable",
      "asli units": "actual_units", "overstock model": "overstock_model", "overstock naive": "overstock_naive",
      "understock model": "understock_model", "understock naive": "understock_naive",
      "overstock kami %": "overstock_cut_pct", "understock kami %": "understock_cut_pct",
      "forecast/actual": "bias", "growth YoY (28 din)": "growth_yoy", "growth pichle 28 din se": "growth_vs_prev28",
      "WAPE (khule)": "wape", "RMSLE (khule)": "rmsle",
      "sales history (sab lags)": "Sales history (all lags)",
      "is window mein badalta hi nahi -> yahan nap nahi sakte": "Does not change inside the test window, so it cannot be measured here",
      "Zalzale ke baad (May 2016)": "After the earthquake (May 2016)", "Christmas se pehle (Dec 2016)": "Before Christmas (Dec 2016)",
      "Aam window (Jun 2017)": "Ordinary window (Jun 2017)",
      "H1 Same family, bohat se stores": "High: same family in many stores", "H2 Poori company ka khaas din": "High: company-wide special day",
      "H3 Store-wide rush": "High: store-wide rush", "H4 Promotion surge": "High: promotion surge",
      "H5 Isolated spike (bulk order / ghalat entry)": "High: isolated spike (bulk order or entry error)",
      "L0 Store takreeban band (disruption)": "Low: store almost closed (disruption)",
      "L1 Same family, bohat se stores": "Low: same family in many stores", "L2 Poori company ka khaas din": "Low: company-wide special day",
      "L3 Store-wide kami (holiday aadat)": "Low: store-wide dip (holiday habit)",
      "L4 Isolated drop (stock-out / record nahi hua)": "Low: isolated drop (stock-out or not recorded)"}


def en(x):
    if isinstance(x, dict):
        return {EN.get(k, k): en(v) for k, v in x.items()}
    if isinstance(x, list):
        return [en(v) for v in x]
    if isinstance(x, str):
        return EN.get(x, x)
    return x


def dump(obj, name):
    p = OUT / name
    with open(p, "w", encoding="utf-8") as f:
        json.dump(no_nan(en(obj)), f, separators=(",", ":"), ensure_ascii=False, default=str, allow_nan=False)
    print(f"  wrote {p} ({p.stat().st_size / 1024:,.0f} KB)")


def wape(y, p):
    return float(np.abs(y - p).sum() / y.sum() * 100)


def rmsle(y, p):
    return float(np.sqrt(np.mean((np.log1p(np.clip(p, 0, None)) - np.log1p(y)) ** 2)))


# =====================================================================================
section("0. Inputs parhna (config, summaries, data)")
# =====================================================================================
cfg = json.load(open(REPORT_DIR / "07_final_config.json"))
s08 = json.load(open(REPORT_DIR / "08_summary.json"))
s09 = json.load(open(REPORT_DIR / "09_summary.json"))
FEATS, CATS, WINDOWS = cfg["features"], cfg["categorical_features"], cfg["windows"]
VALID, HOLDOUT, TUNE = "Validation (Aug 2017)", cfg["holdout"], cfg["tuning_folds"]
ALL_W = TUNE + [VALID, HOLDOUT]
MEMBERS = cfg["members"]
mb = MEMBERS[0]
print(f"Model: {cfg['choice']} | windows: {ALL_W}")

df = pd.read_parquet(PROCESSED / "retail_demand_features.parquet")
df["family"] = df["family"].astype(str)
df = df.sort_values(KEYS + ["date"]).reset_index(drop=True)
master = pd.read_parquet(PROCESSED / "retail_demand_master_treated.parquet",
                         columns=["date", "store_nbr", "family", "split", "sales", "onpromotion", "city", "state", "store_type",
                                  "store_cluster", "is_national_holiday", "is_earthquake_period", "is_trainable", "first_sale_date",
                                  "is_store_closed", "is_pre_opening"])
master["family"] = master["family"].astype(str)
for c in ("city", "state", "store_type"):
    master[c] = master[c].astype(str)
df = df.merge(master.drop_duplicates(KEYS)[KEYS + ["first_sale_date"]], on=KEYS, how="left")
CAT_LEVELS = {c: df[c].astype("category").cat.categories for c in CATS}
for c in CATS:
    df[c] = df[c].astype("category")
DATES, SALES = df["date"].values, df["sales_clean"].values
HIST, TEST = (df["split"] != "test").values, (df["split"] == "test").values
OPEN = ((df["is_pre_opening"] == 0) & (df["is_store_closed"] == 0)).values
CUTOFF = df.loc[HIST, "date"].max()
CLEAN_START = df["date"].min()
print(f"Features rows {len(df):,} | master rows {len(master):,} | cutoff {CUTOFF.date()}")


def win_masks(w):
    cutoff, start = (pd.Timestamp(x) for x in WINDOWS[w])
    evm = (DATES >= np.datetime64(start)) & (DATES <= np.datetime64(start + pd.Timedelta(days=15)))
    return cutoff, start, evm


def zero_rule(cutoff, rows):
    fs = df.loc[rows, "first_sale_date"]
    return ((fs.isna() | (fs > cutoff)).values | (df.loc[rows, "is_pre_opening"].values == 1)
            | (df.loc[rows, "is_closure_calendar_day"].values == 1))


def backtest_preds(w):
    clean = lambda t: "".join(ch if ch.isalnum() else "_" for ch in t)
    for cp in (CACHE_07 / f"{clean(mb['model'])}__{mb['config']}__{clean(w)}__s0.parquet",
               CACHE_08 / f"{mb['model'].split()[0]}__{w.split()[0]}__s0.parquet",
               CACHE_08 / "reproduce_validation_s0.parquet" if w == VALID else Path("_none_"),
               CACHE_09 / f"{mb['model'].split()[0]}__{w.split()[0]}__s0.parquet"):
        if cp.exists():
            return pd.read_parquet(cp)["pred"].values
    raise SystemExit(f"Backtest forecast nahi mila: {w}. Pehle Phase 07/09 chalayen.")


def seasonal_naive(w, evm):
    cutoff = pd.Timestamp(WINDOWS[w][0])
    p = np.expm1(df.loc[evm, "same_wd_mean4"].fillna(df.loc[evm, "roll28_lag16"]).fillna(0).values)
    return np.where(zero_rule(cutoff, evm), 0, np.clip(p, 0, None))


# =====================================================================================
section("1. Backtests (6 windows) -- evaluation tables")
# =====================================================================================
frames, BT = [], {}
for w in ALL_W:
    cutoff, start, evm = win_masks(w)
    p, nv = backtest_preds(w), seasonal_naive(w, evm)
    BT[w] = (evm, p, nv)
    pre = HIST & (DATES > np.datetime64(cutoff - pd.Timedelta(days=90))) & (DATES <= np.datetime64(cutoff)) & OPEN
    size = df.loc[pre].groupby(KEYS)["sales_clean"].mean().rename("avg90").reset_index()
    f = df.loc[evm, KEYS + ["date", "store_type", "onpromotion", "is_national_holiday", "is_local_holiday", "is_regional_holiday"]].copy()
    f["store_type"] = f["store_type"].astype(str)
    f["y"], f["p"], f["naive"], f["open"] = SALES[evm], p, nv, OPEN[evm]
    f["window"], f["din"] = w, (f["date"] - start).dt.days + 1
    frames.append(f.merge(size, on=KEYS, how="left"))
E = pd.concat(frames, ignore_index=True)
E = E[E["open"]].drop(columns="open")
E["err"], E["err_n"] = (E["y"] - E["p"]).abs(), (E["y"] - E["naive"]).abs()


def agg(g):
    return {"units": float(g["y"].sum()), "wape": wape(g["y"], g["p"]), "wape_naive": wape(g["y"], g["naive"]),
            "bias": float(g["p"].sum() / g["y"].sum()), "rmsle": rmsle(g["y"], g["p"])}


windows = []
for w in ALL_W:
    a = agg(E[E["window"] == w])
    windows.append({"window": w, "role": "tuning fold" if w in TUNE else ("decision" if w == VALID else "holdout (untouched)"),
                    "cutoff": WINDOWS[w][0], "start": WINDOWS[w][1], **{k: r(v) for k, v in a.items()}})
fam_rows = []
for fam, g in E.groupby("family"):
    a = agg(g)
    under = int(sum(g[g["window"] == w]["p"].sum() < g[g["window"] == w]["y"].sum() for w in ALL_W))
    fam_rows.append({"family": fam, **{k: r(v) for k, v in a.items()}, "error_share": r(g["err"].sum() / E["err"].sum() * 100),
                     "under_windows": under})
store_meta = master.drop_duplicates("store_nbr").set_index("store_nbr")[["city", "state", "store_type", "store_cluster"]]
store_rows = []
for st, g in E.groupby("store_nbr"):
    a = agg(g)
    m = store_meta.loc[st]
    store_rows.append({"store": int(st), "city": m["city"], "state": m["state"], "type": m["store_type"], "cluster": int(m["store_cluster"]),
                       **{k: r(v) for k, v in a.items()}})
horizon = [{"day": int(d), **{k: r(v) for k, v in agg(g).items()}} for d, g in E.groupby("din")]
wd_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
E["weekday"] = E["date"].dt.day_name()
weekday = [{"weekday": d, **{k: r(v) for k, v in agg(E[E["weekday"] == d]).items()}} for d in wd_order]
E["size_grp"] = pd.qcut(E["avg90"].rank(method="first"), 5, labels=["Smallest 20%", "2nd", "3rd", "4th", "Largest 20%"])
size = [{"group": str(k), "avg_daily": r(g["avg90"].mean(), 1), "units_share": r(g["y"].sum() / E["y"].sum() * 100),
         **{kk: r(v) for kk, v in agg(g).items()}} for k, g in E.groupby("size_grp", observed=True)]
E["day_kind"] = np.select([E["is_national_holiday"] == 1, (E["is_local_holiday"] + E["is_regional_holiday"]) > 0],
                          ["National holiday", "Local / regional holiday"], default="Regular day")
day_kind = [{"kind": k, "rows": int(len(g)), **{kk: r(v) for kk, v in agg(g).items()}} for k, g in E.groupby("day_kind")]
E["promo"] = np.where(E["onpromotion"] > 0, "On promotion", "No promotion")
promo = [{"kind": k, "rows": int(len(g)), **{kk: r(v) for kk, v in agg(g).items()}} for k, g in E.groupby("promo")]
stype = [{"type": k, **{kk: r(v) for kk, v in agg(g).items()}} for k, g in E.groupby("store_type")]
dump({"windows": windows, "families": fam_rows, "stores": store_rows, "store_types": stype, "horizon": horizon,
      "weekday": weekday, "size_groups": size, "day_kind": day_kind, "promo": promo,
      "growth_vs_bias": s09.get("growth_vs_bias"), "bias_by_year": s09.get("bias_by_year")}, "evaluation.json")

# =====================================================================================
section("2. Business impact + safety buffer + importance")
# =====================================================================================
buf = pd.read_csv(REPORT_DIR / "09_safety_buffer_by_family.csv")
imp = pd.read_csv(REPORT_DIR / "09_permutation_importance.csv")
business = s09.get("business", [])
dump({"business": business, "perishables": s09.get("perishable_families_by_name", []),
      "buffers": [{"family": row["family"], "buffer80": r(row["buffer 80%"], 1), "buffer90": r(row["buffer 90%"], 1),
                   "coverage80": r(row["holdout coverage (buffer 80%)"], 1), "coverage90": r(row["holdout coverage (buffer 90%)"], 1),
                   "units_share": r(row["units %"], 2)} for _, row in buf.iterrows()],
      "importance": [{"name": row["feature / group"], "kind": row["kind"], "delta_wape": r(row["WAPE barha (points)"], 2),
                      "spread": r(row["spread"], 3), "note": (row["note"] if "note" in imp.columns and isinstance(row["note"], str) else "")}
                     for _, row in imp.iterrows()]}, "business.json")

# =====================================================================================
section("3. Model journey: Phase 05 ablation, Phase 06 comparison, Phase 07 tuning")
# =====================================================================================
ms = pd.read_csv(REPORT_DIR / "06_model_scores.csv")
ms.columns = ["model", "window", "wape", "rmsle", "bias"]
cmp06 = (ms.groupby("model")[["wape", "rmsle", "bias"]].mean().sort_values("wape").reset_index())
models = {"phase06_avg": [{k: r(v) for k, v in row.items()} for row in cmp06.to_dict(orient="records")],
          "phase06_by_window": [{k: r(v) for k, v in row.items()} for row in ms.to_dict(orient="records")]}
cs_path = REPORT_DIR / "07_confirmation_scores.csv"
if cs_path.exists():
    cs = pd.read_csv(cs_path).groupby(["model", "config", "window"])[["WAPE", "RMSLE"]].mean().reset_index()
    models["phase07_confirmation"] = [{"model": a, "config": b, "window": c, "wape": r(d), "rmsle": r(e)}
                                      for a, b, c, d, e in cs.itertuples(index=False)]
trials = {}
for name in ("LightGBM", "XGBoost", "HistGB"):
    p = REPORT_DIR / f"07_trials_{name}.csv"
    if p.exists():
        t = pd.read_csv(p).sort_values("trial")
        t["best_so_far"] = t["WAPE"].cummin()
        trials[name] = [{"trial": int(a), "wape": r(b), "best": r(c)} for a, b, c in t[["trial", "WAPE", "best_so_far"]].itertuples(index=False)]
models["phase07_trials"] = trials
models["final"] = {"model": cfg["choice"], "params": mb["params"], "decisions": cfg.get("decisions", []), "scores": cfg.get("scores", [])}
abl = REPORT_DIR / "05_ablation_results_v2.csv"
if abl.exists():
    a = pd.read_csv(abl).groupby("variant")[["WAPE", "RMSLE"]].mean().sort_values("WAPE").reset_index()
    models["phase05_ablation"] = [{"variant": v, "wape": r(x), "rmsle": r(y)} for v, x, y in a.itertuples(index=False)]
dump(models, "models.json")

# =====================================================================================
section("4. Data insights (EDA) -- poori history se")
# =====================================================================================
hist = master[master["split"] != "test"]
daily = hist.groupby("date")["sales"].sum()
clean = hist[(hist["date"] >= CLEAN_START) & (hist["is_trainable"] == 1)]
cd = clean.groupby("date")["sales"].sum()
dow = cd.groupby(cd.index.dayofweek).mean()
dom = cd.groupby(cd.index.day).mean()
mon = cd.groupby(cd.index.month).mean()
fam_units = clean.groupby("family")["sales"].sum().sort_values(ascending=False)
zero_share = clean.groupby("family")["sales"].apply(lambda s: (s == 0).mean() * 100)
pl = clean.assign(p=clean["onpromotion"] > 0).groupby(["family", "p"])["sales"].mean().unstack()
promo_lift = ((pl[True] / pl[False] - 1) * 100).replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
hd = clean.groupby("date").agg(sales=("sales", "sum"), hol=("is_national_holiday", "max"))
hd["dow"] = hd.index.dayofweek
base_dow = hd[hd["hol"] == 0].groupby("dow")["sales"].mean()
hol_eff = float((hd.loc[hd["hol"] == 1, "sales"] / hd.loc[hd["hol"] == 1, "dow"].map(base_dow)).mean() - 1) * 100
quake = daily["2016-03-15":"2016-06-15"]
qflag = hist.groupby("date")["is_earthquake_period"].max()["2016-03-15":"2016-06-15"]
sch = hist[hist["family"] == "SCHOOL AND OFFICE SUPPLIES"].groupby("date")["sales"].sum()
school = {str(y): [{"week": int(i + 1), "sales": r(v, 0)} for i, v in enumerate(sch[f"{y}-07-01":f"{y}-09-30"].resample("W").sum().values)]
          for y in (2015, 2016, 2017)}
yearly = daily.groupby(daily.index.year).agg(["sum", "count"])
store_tot = clean.groupby("store_nbr")["sales"].sum().sort_values(ascending=False)
type_share = clean.groupby("store_type")["sales"].sum()
o3 = json.load(open(REPORT_DIR / "03_outliers_summary.json")) if (REPORT_DIR / "03_outliers_summary.json").exists() else {}
eda = {
    "daily_total": [{"date": d.strftime("%Y-%m-%d"), "sales": r(v, 0)} for d, v in daily.items()],
    "clean_start": CLEAN_START.strftime("%Y-%m-%d"),
    "dow_index": [{"day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][i], "index": r(v / dow.mean() * 100, 1)} for i, v in dow.items()],
    "dom_index": [{"day": int(i), "index": r(v / dom.mean() * 100, 1)} for i, v in dom.items()],
    "month_index": [{"month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][i - 1],
                     "index": r(v / mon.mean() * 100, 1)} for i, v in mon.items()],
    "family_share": [{"family": f, "share": r(v / fam_units.sum() * 100, 2), "zero_days": r(zero_share[f], 1)} for f, v in fam_units.items()],
    "promo_lift": [{"family": f, "lift": r(v, 1)} for f, v in promo_lift.items()],
    "holiday_effect_pct": r(hol_eff, 1),
    "earthquake": [{"date": d.strftime("%Y-%m-%d"), "sales": r(v, 0), "flag": int(qflag.get(d, 0))} for d, v in quake.items()],
    "school_season": school,
    "yearly": [{"year": int(y), "total": r(row["sum"], 0), "per_day": r(row["sum"] / row["count"], 0)} for y, row in yearly.iterrows()],
    "store_totals": [{"store": int(s), "sales": r(v, 0), "type": store_meta.loc[s, "store_type"], "city": store_meta.loc[s, "city"]}
                     for s, v in store_tot.items()],
    "store_type_share": [{"type": t, "share": r(v / type_share.sum() * 100, 1)} for t, v in type_share.items()],
    "outliers": {"rows": o3.get("scope_rows"), "robust_z": o3.get("robust_z_extremes"), "iqr": o3.get("iqr_outliers"),
                 "causes": o3.get("causes", {}), "rows_changed": o3.get("rows_changed"), "tests": o3.get("test_wape", {})},
}
dump(eda, "insights.json")

# =====================================================================================
section("5. Forecast (16-31 Aug 2017) + har store ki series files")
# =====================================================================================
fc = pd.read_parquet(REPORT_DIR / "08_test_forecast.parquet")
fc["family"] = fc["family"].astype(str)
daily_fc = fc.groupby("date")["sales"].sum()
recent = daily[CUTOFF - pd.Timedelta(days=118):CUTOFF]
ly = daily[daily_fc.index.min() - pd.Timedelta(days=364):daily_fc.index.max() - pd.Timedelta(days=364)]
dump({"history": [{"date": d.strftime("%Y-%m-%d"), "sales": r(v, 0)} for d, v in recent.items()],
      "forecast": [{"date": d.strftime("%Y-%m-%d"), "sales": r(v, 0)} for d, v in daily_fc.items()],
      "last_year": [{"date": (d + pd.Timedelta(days=364)).strftime("%Y-%m-%d"), "sales": r(v, 0)} for d, v in ly.items()],
      "by_family": [{"family": f, "per_day": r(v / 16, 1)} for f, v in fc.groupby("family")["sales"].sum().sort_values(ascending=False).items()],
      "by_store": [{"store": int(s), "total": r(v, 0)} for s, v in fc.groupby("store_nbr")["sales"].sum().sort_values(ascending=False).items()],
      "total": r(fc["sales"].sum(), 0), "summary": {k: s08.get(k) for k in ("forecast_vs_last4weeks", "forecast_vs_lastyear_growth",
                                                                              "history_ratio_aug16_31_vs_4weeks", "training_rows", "trained_until")}},
     "forecast.json")

bufmap = buf.set_index("family")[["buffer 80%", "buffer 90%"]].to_dict(orient="index")
hstart = CUTOFF - pd.Timedelta(days=111)
H = df.loc[HIST & (DATES >= np.datetime64(hstart)), KEYS + ["date", "sales_clean", "onpromotion"]]
F = fc.set_index(KEYS + ["date"])["sales"]
for st, gst in df.loc[TEST, KEYS].groupby("store_nbr"):
    out = {"store": int(st), **{k: (store_meta.loc[st, k] if k != "store_cluster" else int(store_meta.loc[st, k]))
                                for k in ("city", "state", "store_type", "store_cluster")}, "families": {}}
    hs = H[H["store_nbr"] == st]
    for fam, hf in hs.groupby("family"):
        f16 = fc[(fc["store_nbr"] == st) & (fc["family"] == fam)].sort_values("date")
        rec = {"history": r(hf["sales_clean"].values, 1), "history_promo": r(hf["onpromotion"].values, 0),
               "forecast": r(f16["sales"].values, 1), "buffer80": r(bufmap[fam]["buffer 80%"], 1), "buffer90": r(bufmap[fam]["buffer 90%"], 1),
               "backtests": {}}
        for w in ALL_W:
            evm, p, nv = BT[w]
            k = (df.loc[evm, "store_nbr"].values == st) & (df.loc[evm, "family"].values == fam)
            rec["backtests"][w] = {"actual": r(SALES[evm][k], 1), "model": r(p[k], 1), "naive": r(nv[k], 1)}
        out["families"][fam] = rec
    with open(SERIES_DIR / f"store_{int(st)}.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, separators=(",", ":"))
print(f"  wrote {len(list(SERIES_DIR.glob('*.json')))} series files -> {SERIES_DIR}")

# =====================================================================================
section("6. Live forecast API ke assets + jaanch")
# =====================================================================================
src_model = MODEL_DIR / f"08_{mb['model'].split()[0]}_s0.joblib"
shutil.copy(src_model, API_ASSETS / "model.joblib")
cols = KEYS + ["date"] + [c for c in FEATS if c not in KEYS]
T = df.loc[TEST, KEYS + ["date"] + FEATS].copy()
T["zero"] = zero_rule(CUTOFF, TEST).astype(int)
for c in CATS:
    T[c] = T[c].astype(str) if T[c].dtype.name == "category" else T[c]
T["date"] = T["date"].dt.strftime("%Y-%m-%d")
with gzip.open(API_ASSETS / "test_features.csv.gz", "wt", encoding="utf-8") as fh:
    T.to_csv(fh, index=False)
api_meta = {"features": FEATS, "categorical": CATS,
            "categories": {c: [x.item() if hasattr(x, "item") else x for x in CAT_LEVELS[c]] for c in CATS},
            "category_dtype": {c: str(CAT_LEVELS[c].dtype) for c in CATS},
            "buffers": {f: {"p80": round(v["buffer 80%"], 2), "p90": round(v["buffer 90%"], 2)} for f, v in bufmap.items()},
            "forecast_window": [str(daily_fc.index.min().date()), str(daily_fc.index.max().date())]}
json.dump(api_meta, open(API_ASSETS / "meta.json", "w"), indent=1)
# Jaanch: assets se dobara forecast == Phase 08 forecast?
model = joblib.load(API_ASSETS / "model.joblib")
X = df.loc[TEST, FEATS]
chk = np.where(T["zero"].values == 1, 0, np.expm1(model.predict(X)).clip(0))
diff = float(np.abs(chk - fc.sort_values(KEYS + ["date"])["sales"].values).max())
print(f"  API assets: model {(API_ASSETS / 'model.joblib').stat().st_size / 1e6:.1f} MB, test features "
      f"{(API_ASSETS / 'test_features.csv.gz').stat().st_size / 1e6:.2f} MB | Phase 08 forecast se sab se bara farq: {diff:.6f} units")
assert diff < 1e-3, "API assets ka forecast Phase 08 se nahi milta"
req = (f"numpy=={np.__version__}\npandas=={pd.__version__}\nscikit-learn=={sklearn.__version__}\njoblib=={joblib.__version__}\n")
Path("dashboard/requirements.txt").write_text(req)
print("  dashboard/requirements.txt (aap ke versions, taa ke model Vercel par load ho):\n    " + req.replace("\n", "\n    "))

# =====================================================================================
section("7. Meta + KPIs")
# =====================================================================================
FEATURE_INFO = {
    "store_cat": ("Store / product", "Which of the 54 stores"),
    "family_cat": ("Store / product", "Which of the 33 product families"),
    "dow": ("Calendar", "Day of week"), "dom": ("Calendar", "Day of month (paydays on the 15th and month end)"),
    "month": ("Calendar", "Month of year"),
    "is_national_holiday": ("Holiday / event", "National day off (holiday, transfer, bridge)"),
    "is_national_event": ("Holiday / event", "National event (Mother's Day, Black Friday, ...)"),
    "is_earthquake_period": ("Holiday / event", "April-May 2016 earthquake period (always 0 in the future)"),
    "onpromotion": ("Promotion", "Number of items in the family on promotion that day (planned in advance)"),
    "lag_16": ("Sales history", "Sales 16 days earlier"), "lag_21": ("Sales history", "Sales 21 days earlier"),
    "lag_364": ("Sales history", "Sales on the same weekday one year earlier"),
    "roll7_lag16": ("Sales history", "Average of the 7 days ending 16 days earlier"),
    "roll28_lag16": ("Sales history", "Average of the 28 days ending 16 days earlier"),
    "same_wd_mean4": ("Sales history", "Average of the same weekday over 4 weeks (21-42 days earlier)"),
}
hold = next(w for w in windows if w["window"] == HOLDOUT)
val = next(w for w in windows if w["window"] == VALID)
biz = {b["group"]: b for b in business}
allb = biz.get("sab families", {})
kpis = {"holdout_wape": hold["wape"], "holdout_naive": hold["wape_naive"], "holdout_rmsle": hold["rmsle"],
        "validation_wape": val["wape"], "validation_naive": val["wape_naive"], "validation_rmsle": val["rmsle"],
        "overstock_reduction": allb.get("overstock kami %"), "understock_reduction": allb.get("understock kami %"),
        "families_beating_naive": int(sum(1 for f in fam_rows if f["wape"] < f["wape_naive"])), "n_families": len(fam_rows),
        "stores_beating_naive": int(sum(1 for s in store_rows if s["wape"] < s["wape_naive"])), "n_stores": len(store_rows),
        "forecast_total": r(fc["sales"].sum(), 0), "kaggle_score": s09.get("kaggle_score")}
meta = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "model": cfg["choice"], "model_config": mb["config"],
        "params": mb["params"], "n_stores": int(df["store_nbr"].nunique()), "n_families": int(df["family"].nunique()),
        "n_series": int(df[KEYS].drop_duplicates().shape[0]), "history_start": str(master["date"].min().date()),
        "clean_start": str(CLEAN_START.date()), "cutoff": str(CUTOFF.date()), "forecast_window": api_meta["forecast_window"],
        "horizon": 16, "training_rows": s08.get("training_rows"), "noise_wape": cfg["noise"]["WAPE"],
        "features": [{"name": f, "group": FEATURE_INFO.get(f, ("Other", ""))[0], "description": FEATURE_INFO.get(f, ("", f))[1]} for f in FEATS],
        "stores": [{"store": int(s), "city": m["city"], "state": m["state"], "type": m["store_type"], "cluster": int(m["store_cluster"])}
                   for s, m in store_meta.sort_index().iterrows()],
        "families": sorted(df["family"].unique().tolist()), "windows": windows, "kpis": kpis,
        "versions": {"scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__}}
dump(meta, "meta.json")
print(f"\nKul waqt: {(time.time() - T0) / 60:.1f} minute | Saved: dashboard/public/data/, dashboard/api/assets/, dashboard/requirements.txt")
