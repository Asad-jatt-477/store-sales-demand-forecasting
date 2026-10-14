# =====================================================================================
# Retail Demand Forecasting System
# Phase 06: Model Comparison (baseline models, BINA tuning ke)
# -------------------------------------------------------------------------------------
# Is script ka maqsad:
#   1. Kai model families ko BARABAR imtihaan dena (wahi features, wahi 5 windows, wahi metrics, wahi rules).
#   2. Har model ko USKI APNI preprocessing dena (Phase 05 ka faisla: encoding/scaling/NaN model-specific hain).
#   3. Horizon strategy test karna: kya forecast ke pehle 7 din ke liye taaza (lag >= 7) features madad karte hain?
#   4. Phase 07 (Optuna tuning) ke liye 1-2 behtareen candidates chunna -- saboot ke sath.
#
# Sections:  0 data | 1 windows/rules/metrics | 2 horizon features + leakage test | 3 models + apni preprocessing
#            4 train + forecast (5 windows) | 4b har model ka seed noise | 5 nateeje + faisla | 5b overfitting check
#            6 horizon strategy test | 7 ghalti family ke hisaab se + bias flags | 7b actual vs forecast charts
#            7c ghalti store ke hisaab se | 8 khulasa (numbers se khud bana hua)
#
# Tuning yahan JAAN BOOJH KAR nahi hai: sab models "samajhdar default" settings par hain. Tuning Phase 07 mein hogi,
# aur us ka cross-validation bhi yahi time-based windows hongi (random K-fold time series mein leakage hai).
#
# Chalane ka tareeqa (data folder se):   python 06_model_comparison.py
# Zaroori libraries:  pandas, numpy, lightgbm, scikit-learn, matplotlib, xgboost
#   Install:  python -m pip install pandas numpy lightgbm scikit-learn matplotlib xgboost pyarrow
#   Agar xgboost install nahi, to woh model SKIP ho jayega aur script chalti rahegi.
#   CatBoost is file se nikal diya gaya hai: candidate nahi tha, bohot dheema tha, aur zaroori nahi.
#
# Resume: har (model, window) ka nateeja reports/06_cache/ mein save hota hai. Laptop so jaye ya script ruk jaye,
# to dobara chalane par sirf baqi kaam hota hai. Sab kuch dobara chalana ho: FORCE_RERUN = True.
# =====================================================================================
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # charts sirf file mein (koi window nahi khulegi)
import matplotlib.pyplot as plt
import matplotlib.ticker
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

FORCE_RERUN = False
PROCESSED, REPORT_DIR = Path("data/processed"), Path("reports")
FIG_DIR = REPORT_DIR / "figures" / "06_model_comparison"
CACHE_DIR = REPORT_DIR / "06_cache"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
KEYS = ["store_nbr", "family"]
CLEAN_START = pd.Timestamp("2015-06-01")
BLUE, ORANGE, AQUA, RED, GRAY, LIGHT_GRAY, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#9a9994", "#d9d8d4", "#52514e"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 10, "axes.titlesize": 12,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e8e7e3", "axes.axisbelow": True,
                     "legend.frameon": False})


def section(title):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100, flush=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [chart saved] {FIG_DIR / (name + '.png')}")


T_START = time.time()

# =====================================================================================
section("0. Data load: Phase 05 ka feature dataset aur final feature list")
# =====================================================================================
info = json.load(open(REPORT_DIR / "05_final_features.json"))
FEATS = info["final_features"]
CATS = info["categorical_features"]
NOISE_W = info["noise"]["WAPE_points"]
NOISE_R = info["noise"]["RMSLE"]
UNDER_FAMS = info["under_recorded_families_lag_mask"]

df = pd.read_parquet(PROCESSED / "retail_demand_features.parquet")
df["family"] = df["family"].astype(str)
df = df.sort_values(KEYS + ["date"]).reset_index(drop=True)
master = pd.read_parquet(PROCESSED / "retail_demand_master_treated.parquet",
                         columns=KEYS + ["date", "sales_clean", "is_trainable", "is_family_unavailable", "first_sale_date"])
master["family"] = master["family"].astype(str)
master = master.sort_values(KEYS + ["date"]).reset_index(drop=True)
df = df.merge(master.drop_duplicates(KEYS)[KEYS + ["first_sale_date"]], on=KEYS, how="left")
for c in CATS:
    df[c] = df[c].astype("category")
print(f"Rows: {len(df):,} | final features ({len(FEATS)}): {FEATS}")
print(f"Phase 05 noise (yeh hamari 'barabar' ki hadd hai): WAPE {NOISE_W:.3f} points | RMSLE {NOISE_R:.4f}")
print("Usool: do models ka farq agar noise se chhota hai, to woh 'barabar' mane jayenge (koi jeeta nahi).")

# =====================================================================================
section("1. Imtihaan ka setup: 5 time-based windows, rules, metrics")
# =====================================================================================
# Har window: train sirf cutoff tak (clean period, is_trainable == 1), eval agle 16 din.
# Yeh wahi windows hain jo Phase 05 mein thin (4 + Aug 2016), taa ke nateeje muqable ke qabil hon.
WINDOWS = {"Validation (Aug 2017)": ("2017-07-30", "2017-07-31"),
           "Aug 2016": ("2016-07-30", "2016-07-31"),
           "May 2016": ("2016-05-17", "2016-05-18"),
           "Dec 2016": ("2016-12-02", "2016-12-03"),
           "Jun 2017": ("2017-06-14", "2017-06-15")}
DATES = df["date"].values
Y = df["target_log1p"].values
SALES = df["sales_clean"].values
TRAINABLE = (df["is_trainable"] == 1).values
OPEN = ((df["is_pre_opening"] == 0) & (df["is_store_closed"] == 0)).values


def masks(cutoff, eval_start):
    te, es = np.datetime64(pd.Timestamp(cutoff)), np.datetime64(pd.Timestamp(eval_start))
    trm = TRAINABLE & (DATES <= te)                       # df pehle hi clean period se shuru hai
    evm = (DATES >= es) & (DATES <= es + np.timedelta64(15, "D"))
    return trm, evm


def zero_rule(cutoff):
    # Rule (feature nahi), Phase 05 ka: cutoff tak kabhi nahi biki / pre-opening / closure-calendar din -> 0
    never = df["first_sale_date"].isna() | (df["first_sale_date"] > pd.Timestamp(cutoff))
    return (never | (df["is_pre_opening"] == 1) | (df["is_closure_calendar_day"] == 1)).values


MASKS = {w: masks(*v) for w, v in WINDOWS.items()}
ZR = {w: zero_rule(v[0]) for w, v in WINDOWS.items()}


def wape(y, p):  return np.abs(y - p).sum() / y.sum() * 100
def rmsle(y, p): return float(np.sqrt(np.mean((np.log1p(np.clip(p, 0, None)) - np.log1p(y)) ** 2)))


for w, (trm, evm) in MASKS.items():
    print(f"  {w:<22} train rows {trm.sum():>9,} | eval rows {evm.sum():>6,} | eval khule store rows {(evm & OPEN).sum():>6,}")
print("Metrics: WAPE (kul units ki ghalti %, business) aur RMSLE (Kaggle; har row barabar). Sirf khule stores par.")

# =====================================================================================
section("2. Horizon features: forecast ke pehle 7 din ke liye taaza history (lag >= 7)")
# =====================================================================================
# Phase 05 ke saare history features >= 16 din purane hain, kyun ke forecast ka AAKHRI din 16 din aage hai.
# Lekin forecast ke din 1-7 ke liye 7+ din purani sales bhi maloom hoti hai. Yahan woh features banaye jate hain.
# Masking rule wahi: band/unavailable din aur 4 kam-likhi families ke partial din = NaN (Phase 04/05).
unav = master[master["is_family_unavailable"] == 1].groupby("date")["family"].nunique()
PARTIAL_DATES = pd.DatetimeIndex(unav[unav >= 5].index)


def build_h7(m):
    ok = ((m["is_trainable"] == 1) & m["sales_clean"].notna()
          & ~(m["date"].isin(PARTIAL_DATES) & m["family"].isin(UNDER_FAMS)))
    ys = pd.Series(np.where(ok, np.log1p(m["sales_clean"]), np.nan), index=m.index)
    keys = [m["store_nbr"], m["family"]]
    g = ys.groupby(keys)
    out = m[KEYS + ["date"]].copy()
    for L in (7, 14, 21, 28):
        out[f"h7_lag_{L}"] = g.shift(L)
    s7 = g.shift(7).groupby(keys)
    out["h7_roll7"] = s7.transform(lambda s: s.rolling(7, min_periods=4).mean())
    out["h7_roll28"] = s7.transform(lambda s: s.rolling(28, min_periods=14).mean())
    out["h7_wd_mean4"] = out[["h7_lag_7", "h7_lag_14", "h7_lag_21", "h7_lag_28"]].mean(axis=1)
    return out


H7_COLS = ["h7_lag_7", "h7_lag_14", "h7_roll7", "h7_roll28", "h7_wd_mean4"]
t0 = time.time()
h7 = build_h7(master)
df = df.merge(h7[KEYS + ["date"] + H7_COLS], on=KEYS + ["date"], how="left")
# Horizon model ke features: static/calendar/promo + lag_364 wahi; history features 7-din wale
REPLACE = {"same_wd_mean4": "h7_wd_mean4", "roll28_lag16": "h7_roll28", "roll7_lag16": "h7_roll7",
           "lag_16": "h7_lag_7", "lag_21": "h7_lag_14"}
FEATS_H7 = [REPLACE.get(f, f) for f in FEATS]
print(f"Horizon features bane ({time.time() - t0:.0f}s): {H7_COLS}")
print(f"Horizon model ke features: {FEATS_H7}")

# --- Leakage test: validation window ki sales mita kar dobara banao; din 1-7 ke features nahi badalne chahiyen ---
blind = master.copy()
blind.loc[blind["date"] >= pd.Timestamp("2017-07-31"), "sales_clean"] = np.nan
hb = build_h7(blind)
chk = h7["date"].between("2017-07-31", "2017-08-06")
same = all(np.allclose(h7.loc[chk, c].values, hb.loc[chk, c].values, equal_nan=True) for c in H7_COLS)
chk2 = h7["date"].between("2017-08-07", "2017-08-15")
leaks_after7 = not all(np.allclose(h7.loc[chk2, c].values, hb.loc[chk2, c].values, equal_nan=True) for c in H7_COLS)
print(f"Leakage test (validation din 1-7): features badle? {'NAHI -> safe' if same else 'HAAN -> LEAKAGE'}")
print(f"Control (din 8-16 par yeh features validation ki sales dekhte hain? {leaks_after7}) -> isi liye yeh SIRF din 1-7 ke liye hain")
assert same, "LEAKAGE in horizon features"
del blind, hb

# =====================================================================================
section("3. Models aur un ki APNI preprocessing")
# =====================================================================================
# Har model ko wahi information, lekin us ki zaroorat ke mutabiq shakal mein:
#   Tree models (LightGBM, XGBoost, HistGB): categories native, NaN native, scaling nahi (Phase 05, 6b).
#   Ridge (linear): one-hot (store, family, aur dow/dom/month bhi, kyun ke un ka asar seedhi lakeer nahi),
#                   NaN -> median + "missing" flag, numeric -> StandardScaler. Sab SIRF training par fit (Pipeline).
# Settings: koi tuning nahi; har model ki "samajhdar default" settings, LightGBM Phase 05 wali.
try:
    import xgboost as xgb
except ImportError:
    xgb = None

CAL_AS_CAT = ["dow", "dom", "month"]


def num_cols(feats):
    return [c for c in feats if c not in CATS and c not in CAL_AS_CAT]


def fit_predict(model, feats, trm, evm, seed=0, extra_idx=None):
    # Model train karo (sirf training rows), eval rows ka forecast do (log scale).
    # extra_idx: training ki kuch rows jin par bhi predict karna hai (overfitting check, Section 5b).
    Xtr, ytr = df.loc[trm, feats], Y[trm]
    tf = lambda X: X                                          # model-specific input shakal
    if model == "LightGBM":
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.07, num_leaves=127, min_child_samples=50,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
        m.fit(Xtr, ytr)
    elif model == "XGBoost":
        m = xgb.XGBRegressor(n_estimators=400, learning_rate=0.07, tree_method="hist", grow_policy="lossguide",
                             max_leaves=127, max_depth=0, min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
                             enable_categorical=True, max_cat_to_onehot=1, random_state=seed, n_jobs=-1)
        m.fit(Xtr, ytr)
    elif model == "HistGB (sklearn)":
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.07, max_leaf_nodes=127, min_samples_leaf=50,
                                          categorical_features="from_dtype", random_state=seed)
        m.fit(Xtr, ytr)
    elif model == "Ridge (linear)":
        nums = num_cols(feats)
        pre = ColumnTransformer([
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=1), CATS + [c for c in CAL_AS_CAT if c in feats]),
            ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), nums)])
        def tf(X):
            X = X.copy()
            for c in CATS:
                X[c] = X[c].astype(str)
            return X
        m = Pipeline([("pre", pre), ("ridge", Ridge(alpha=1.0))])   # preprocessing SIRF training par fit
        m.fit(tf(Xtr), ytr)
    else:
        raise ValueError(model)
    p_ev = m.predict(tf(df.loc[evm, feats]))
    p_ex = m.predict(tf(df.loc[extra_idx, feats])) if extra_idx is not None else None
    return p_ev, p_ex


MODELS = ["LightGBM", "XGBoost", "HistGB (sklearn)", "Ridge (linear)"]
SKIPPED = []
if xgb is None:
    SKIPPED.append("XGBoost")
MODELS = [m for m in MODELS if m not in SKIPPED]
print("Models:", MODELS + ["Baseline: 28-din average", "Baseline: seasonal naive"])
print("  (seasonal naive = usi weekday ki pichle 4 hafton ki average, 21-42 din purani; retail ka aam baseline)")
if SKIPPED:
    print("SKIP (library install nahi):", SKIPPED, "-> python -m pip install xgboost")

# =====================================================================================
section("4. Har model, har window: train + forecast (cache ke sath)")
# =====================================================================================
def cache_path(model, w, tag=""):
    clean = lambda t: t.replace(" ", "_").replace("(", "").replace(")", "").replace(">=", "ge").replace("/", "-")
    return CACHE_DIR / f"{clean(model)}__{clean(w)}{('__' + tag) if tag else ''}.parquet"


def get_preds(model, w, seed=0):
    # Har eval row ka forecast (units mein, rule ke baad) -- cache se ya train kar ke. seed 1 = noise check (4b).
    cp = cache_path(model, w, "" if seed == 0 else f"seed{seed}")
    if cp.exists() and not FORCE_RERUN:
        return pd.read_parquet(cp)["pred"].values, None
    trm, evm = MASKS[w]
    t0 = time.time()
    if model == "Baseline: 28-din average":
        p = np.expm1(df.loc[evm, "roll28_lag16"].fillna(0).values)
    elif model == "Baseline: seasonal naive":
        p = np.expm1(df.loc[evm, "same_wd_mean4"].fillna(df.loc[evm, "roll28_lag16"]).fillna(0).values)
    elif model == "LightGBM horizon (din 1-7 lag>=7)":
        p16 = get_preds("LightGBM", w)[0]
        day = (df.loc[evm, "date"] - df.loc[evm, "date"].min()).dt.days.values + 1
        p7 = np.expm1(fit_predict("LightGBM", FEATS_H7, trm, evm)[0]).clip(0)
        p = np.where(day <= 7, p7, p16)               # din 1-7: taaza model, din 8-16: single model
    else:
        p = np.expm1(fit_predict(model, FEATS, trm, evm, seed)[0]).clip(0)
    p = np.where(ZR[w][evm], 0, np.clip(p, 0, None))
    pd.DataFrame({"pred": p}).to_parquet(cp, index=False)
    secs = time.time() - t0
    print(f"  {model:<38} {w:<22} seed {seed} {secs:6.0f}s", flush=True)
    return p, secs


ALL = ["Baseline: 28-din average", "Baseline: seasonal naive"] + MODELS + ["LightGBM horizon (din 1-7 lag>=7)"]
rows, timing = [], {}
for model in ALL:
    for w, (trm, evm) in MASKS.items():
        p, secs = get_preds(model, w)
        if secs is not None:
            timing.setdefault(model, []).append(secs)
        o = OPEN[evm]
        y = SALES[evm][o]
        rows.append({"model": model, "window": w, "WAPE": wape(y, p[o]), "RMSLE": rmsle(y, p[o]),
                     "forecast / actual": p[o].sum() / y.sum()})
res = pd.DataFrame(rows)
res.to_csv(REPORT_DIR / "06_model_scores.csv", index=False)

# =====================================================================================
section("4b. Har model ka apna noise (seed 0 vs seed 1)")
# =====================================================================================
# Phase 05 ka noise (0.21) sirf LightGBM ka tha. "Barabar" ka faisla har model ke APNE noise par bhi hona chahiye.
# Candidates (LightGBM, XGBoost, HistGB) ko seed 1 se bhi chalaya jata hai. Ridge deterministic hai (noise 0).
SEED_MODELS = [m for m in ["LightGBM", "XGBoost", "HistGB (sklearn)"] if m in MODELS]
noise_rows = []
for model in SEED_MODELS:
    for w, (trm, evm) in MASKS.items():
        o = OPEN[evm]
        y = SALES[evm][o]
        p0, _ = get_preds(model, w, 0)
        p1, _ = get_preds(model, w, 1)
        noise_rows.append({"model": model, "window": w, "|ΔWAPE| seed0-seed1": abs(wape(y, p0[o]) - wape(y, p1[o])),
                           "|ΔRMSLE| seed0-seed1": abs(rmsle(y, p0[o]) - rmsle(y, p1[o]))})
nz = pd.DataFrame(noise_rows)
MODEL_NOISE = nz.groupby("model")[["|ΔWAPE| seed0-seed1", "|ΔRMSLE| seed0-seed1"]].mean()
MODEL_NOISE.loc["Ridge (linear)"] = [0.0, 0.0]
print(MODEL_NOISE.round(4).to_string())
print(f"Phase 05 noise (LightGBM, seed + placebo): WAPE {NOISE_W:.3f} | RMSLE {NOISE_R:.4f}")
print("Usool: model A aur best ka farq 'barabar' tab hai jab woh in teeno se chhota ho: Phase 05 noise, best ka seed noise, A ka seed noise.")


def tie_thr(m, best):
    nw = [NOISE_W] + [MODEL_NOISE.loc[x, "|ΔWAPE| seed0-seed1"] for x in (m, best) if x in MODEL_NOISE.index]
    nr = [NOISE_R] + [MODEL_NOISE.loc[x, "|ΔRMSLE| seed0-seed1"] for x in (m, best) if x in MODEL_NOISE.index]
    return max(nw), max(nr)


# =====================================================================================
section("5. Nateeje: har window aur average")
# =====================================================================================
wt = res.pivot(index="model", columns="window", values="WAPE")[list(WINDOWS)]
rt = res.pivot(index="model", columns="window", values="RMSLE")[list(WINDOWS)]
summary = pd.DataFrame({"avg WAPE": wt.mean(axis=1), "avg RMSLE": rt.mean(axis=1),
                        "avg forecast/actual": res.groupby("model")["forecast / actual"].mean()})
summary = summary.sort_values("avg WAPE")
print("WAPE % (har window):")
print(wt.loc[summary.index].round(3).to_string())
print("\nRMSLE (har window):")
print(rt.loc[summary.index].round(4).to_string())
print("\nAverage (5 windows):")
print(summary.round(4).to_string())

single = [m for m in summary.index if m != "LightGBM horizon (din 1-7 lag>=7)"]
best = [m for m in single if not m.startswith("Baseline")][0]
summary["ΔWAPE vs best"] = summary["avg WAPE"] - summary.loc[best, "avg WAPE"]
summary["ΔRMSLE vs best"] = summary["avg RMSLE"] - summary.loc[best, "avg RMSLE"]
summary["windows mein best se behtar (WAPE)"] = [(wt.loc[m] < wt.loc[best]).sum() for m in summary.index]


def verdict(m):
    if m == best:
        return "BEST (single model)"
    if m.startswith("Baseline"):
        return "baseline"
    if m.startswith("LightGBM horizon"):
        return "horizon test (Section 6)"
    dW, dR = summary.loc[m, "ΔWAPE vs best"], summary.loc[m, "ΔRMSLE vs best"]
    tw, tr = tie_thr(m, best)
    if abs(dW) <= tw and dR <= tr:
        return f"BARABAR (noise hadd {tw:.3f} ke andar) -> Phase 07 candidate"
    return f"peeche (noise hadd {tw:.3f} se zyada)"


summary["faisla"] = [verdict(m) for m in summary.index]
print("\nBest single model ke muqable:")
print(summary[["avg WAPE", "ΔWAPE vs best", "avg RMSLE", "ΔRMSLE vs best", "windows mein best se behtar (WAPE)", "faisla"]].round(4).to_string())
if timing:
    print("\nTraining + forecast waqt (is run mein, seconds per window):",
          {k: round(float(np.mean(v)), 0) for k, v in timing.items()})

# Chart: average WAPE aur RMSLE
s = summary.sort_values("avg WAPE", ascending=False)
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
colors = [RED if m.startswith("Baseline") else (AQUA if m.startswith("LightGBM horizon") else BLUE) for m in s.index]
axes[0].barh(s.index, s["avg WAPE"], color=colors)
lo = summary.loc[best, "avg WAPE"]
axes[0].axvspan(lo - NOISE_W, lo + NOISE_W, color=GRAY, alpha=0.2, label="best ± noise")
axes[0].set_title("Average WAPE % (5 windows, kam = behtar; grey patti = best ± noise)")
axes[1].barh(s.index, s["avg RMSLE"], color=colors)
axes[1].set_title("Average RMSLE (kam = behtar)")
for ax, col in [(axes[0], "avg WAPE"), (axes[1], "avg RMSLE")]:
    for i, v in enumerate(s[col]):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8, color=INK2)
save(fig, "01_model_comparison")

# Chart: har window par WAPE
fig, ax = plt.subplots(figsize=(11, 4.6))
for m in summary.index:
    style = dict(linewidth=2.2) if m == best else dict(linewidth=1.2, alpha=0.8)
    ax.plot(list(WINDOWS), wt.loc[m], marker="o", label=m, **style)
ax.set_ylabel("WAPE %")
ax.set_title("Har window par WAPE (kam = behtar)")
ax.legend(fontsize=8, ncol=2)
save(fig, "02_wape_by_window")

# =====================================================================================
section("5b. Overfitting check: training par ghalti vs nayi window par ghalti")
# =====================================================================================
# Har model (validation window) ko training ke AAKHRI 28 din par bhi score kiya jata hai (yeh din model ne training
# mein dekhe thay), aur phir agle 16 din (validation, jo nahi dekhe). Bara farq = model training data "yaad" kar raha hai.
trm, evm = MASKS["Validation (Aug 2017)"]
cut = pd.Timestamp(WINDOWS["Validation (Aug 2017)"][0])
in_idx = np.flatnonzero(trm & (DATES > np.datetime64(cut - pd.Timedelta(days=28))))
in_open = OPEN[in_idx]
ov = []
for model in MODELS:
    cp = cache_path(model, "Validation (Aug 2017)", "insample28")
    if cp.exists() and not FORCE_RERUN:
        p_in = pd.read_parquet(cp)["pred"].values
    else:
        t0 = time.time()
        _, p_in_log = fit_predict(model, FEATS, trm, evm, 0, extra_idx=in_idx)
        p_in = np.where(ZR["Validation (Aug 2017)"][in_idx], 0, np.expm1(p_in_log).clip(0))
        pd.DataFrame({"pred": p_in}).to_parquet(cp, index=False)
        print(f"  {model:<38} in-sample check {time.time() - t0:6.0f}s", flush=True)
    y_in = SALES[in_idx][in_open]
    w_in, r_in = wape(y_in, p_in[in_open]), rmsle(y_in, p_in[in_open])
    w_out = res[(res["model"] == model) & (res["window"] == "Validation (Aug 2017)")]["WAPE"].iloc[0]
    r_out = res[(res["model"] == model) & (res["window"] == "Validation (Aug 2017)")]["RMSLE"].iloc[0]
    ov.append({"model": model, "WAPE training (aakhri 28 din)": w_in, "WAPE validation (agle 16 din)": w_out,
               "farq (val - train)": w_out - w_in, "RMSLE training": r_in, "RMSLE validation": r_out})
ov = pd.DataFrame(ov).set_index("model").sort_values("WAPE validation (agle 16 din)")
print(ov.round(4).to_string())
print("Padhne ka tareeqa: training par ghalti hamesha kam hoti hai. Sawal yeh hai ke farq kitna bara hai, aur kya kam training")
print("ghalti wala model validation par bhi behtar hai. Agar training bohat kam aur validation zyada -> overfitting ka ishara.")
ov.to_csv(REPORT_DIR / "06_overfitting_check.csv")

# =====================================================================================
section("6. Horizon strategy: din 1-7 ke liye taaza features (lag >= 7) se faida?")
# =====================================================================================
# Faisle ka usool (pehle se tay): horizon tareeqa tabhi apnaya jayega jab DIN 1-7 par WAPE ka average faida
# noise se bara ho, AUR 5 mein se kam se kam 4 windows mein behtar ho.
hrows = []
for w, (trm, evm) in MASKS.items():
    day = (df.loc[evm, "date"] - df.loc[evm, "date"].min()).dt.days.values + 1
    o = OPEN[evm]
    p_s, _ = get_preds("LightGBM", w)
    p_h, _ = get_preds("LightGBM horizon (din 1-7 lag>=7)", w)
    for d in range(1, 17):
        k = o & (day == d)
        hrows.append({"window": w, "din": d, "single": wape(SALES[evm][k], p_s[k]), "horizon": wape(SALES[evm][k], p_h[k])})
    k = o & (day <= 7)
    hrows.append({"window": w, "din": "1-7", "single": wape(SALES[evm][k], p_s[k]), "horizon": wape(SALES[evm][k], p_h[k])})
ht = pd.DataFrame(hrows)
h17 = ht[ht["din"] == "1-7"].set_index("window")[["single", "horizon"]]
h17["farq (horizon - single)"] = h17["horizon"] - h17["single"]
print("Din 1-7 ka WAPE (single LightGBM vs horizon tareeqa):")
print(h17.round(3).to_string())
gain = h17["farq (horizon - single)"].mean()
better = int((h17["farq (horizon - single)"] < 0).sum())
HORIZON_OK = gain < -NOISE_W and better >= 4
print(f"Average farq din 1-7: {gain:+.3f} points | behtar windows: {better}/5 | noise {NOISE_W:.3f} -> horizon tareeqa apnao: {HORIZON_OK}")
ht.to_csv(REPORT_DIR / "06_horizon_test.csv", index=False)

curve = ht[ht["din"] != "1-7"].groupby("din")[["single", "horizon"]].mean()
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(curve.index, curve["single"], marker="o", color=BLUE, label="single model (sab lags >= 16)")
ax.plot(curve.index, curve["horizon"], marker="o", color=AQUA, label="horizon (din 1-7: lags >= 7)")
ax.axvline(7.5, color=GRAY, linestyle="--", linewidth=1)
ax.set_xlabel("forecast ka din (1 = cutoff ke agle din)")
ax.set_ylabel("WAPE % (5 windows ki average)")
ax.set_title("Forecast ka din vs ghalti: kya taaza history pehle hafte mein madad karti hai?")
ax.legend()
save(fig, "03_horizon_curve")

# =====================================================================================
section("7. Ghalti kahan hai? (best single model, validation window)")
# =====================================================================================
trm, evm = MASKS["Validation (Aug 2017)"]
o = OPEN[evm]
p_best, _ = get_preds(best, "Validation (Aug 2017)")
p_base, _ = get_preds("Baseline: seasonal naive", "Validation (Aug 2017)")   # wahi baseline jo store table (7c) mein hai
seg = pd.DataFrame({"family": df.loc[evm, "family"].values[o], "y": SALES[evm][o],
                    "abs_err_model": np.abs(SALES[evm][o] - p_best[o]), "abs_err_base": np.abs(SALES[evm][o] - p_base[o]),
                    "pred": p_best[o]})
fam = seg.groupby("family").agg(units=("y", "sum"), err_model=("abs_err_model", "sum"), err_base=("abs_err_base", "sum"),
                                pred=("pred", "sum"))
fam["WAPE model %"] = fam["err_model"] / fam["units"] * 100
fam["WAPE seasonal naive %"] = fam["err_base"] / fam["units"] * 100
fam["kul ghalti mein hissa %"] = fam["err_model"] / fam["err_model"].sum() * 100
fam["forecast/actual"] = fam["pred"] / fam["units"]
fam = fam.sort_values("kul ghalti mein hissa %", ascending=False)
print(f"{best}: validation par kul ghalti ka sab se bara hissa kin families se:")
print(fam[["units", "WAPE model %", "WAPE seasonal naive %", "kul ghalti mein hissa %", "forecast/actual"]].head(10).round(2).to_string())
top5 = fam["units"].nlargest(5).index
print(f"\nTop-5 families (units ka {fam.loc[top5, 'units'].sum() / fam['units'].sum():.0%}) ka kul ghalti mein hissa: "
      f"{fam.loc[top5, 'err_model'].sum() / fam['err_model'].sum():.0%}")
fam.to_csv(REPORT_DIR / "06_validation_error_by_family.csv")
# Families jahan forecast systematic taur par ghalat simt mein hai: kul GHALTI ka 1%+ hissa, aur forecast/actual 0.7 se kam ya 1.3 se zyada
flag = fam[(fam["kul ghalti mein hissa %"] >= 1) & ((fam["forecast/actual"] < 0.7) | (fam["forecast/actual"] > 1.3))]
BIAS_FLAGS = flag.index.tolist()
print("\nSystematic bias wali families (forecast/actual < 0.7 ya > 1.3, kul ghalti ka 1%+):",
      flag[["units", "forecast/actual", "WAPE model %"]].round(2).to_dict(orient="index") if len(flag) else "koi nahi")

# =====================================================================================
section("7b. Actual vs forecast: tasveer mein")
# =====================================================================================
cand_models = [best] + [m for m in summary.index if summary.loc[m, "faisla"].startswith("BARABAR")]
fig, axes = plt.subplots(2, 1, figsize=(11, 7.5))
for ax, w in zip(axes, ["Validation (Aug 2017)", "Aug 2016"]):
    trm, evm = MASKS[w]
    d = df.loc[evm, ["date"]].copy()
    d["actual"] = SALES[evm]
    for m in cand_models + ["Baseline: seasonal naive"]:
        d[m] = get_preds(m, w)[0]
    tot = d.groupby("date").sum()
    ax.plot(tot.index, tot["actual"], color="black", linewidth=2.4, label="asli sales")
    for m, col in zip(cand_models + ["Baseline: seasonal naive"], [BLUE, AQUA, ORANGE, RED]):
        ax.plot(tot.index, tot[m], linewidth=1.4, color=col, label=m, linestyle="--" if m.startswith("Baseline") else "-")
    ax.set_title(f"{w}: company ki kul daily sales, asli vs forecast")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e3:,.0f}k"))
    ax.legend(fontsize=8, ncol=3)
save(fig, "04_actual_vs_forecast_total")

trm, evm = MASKS["Validation (Aug 2017)"]
d = df.loc[evm, ["date", "family"]].copy()
d["actual"], d["forecast"] = SALES[evm], get_preds(best, "Validation (Aug 2017)")[0]
fams = list(d.groupby("family")["actual"].sum().nlargest(4).index) + BIAS_FLAGS[:2]
fams = list(dict.fromkeys(fams))
fig, axes = plt.subplots(int(np.ceil(len(fams) / 2)), 2, figsize=(12, 3.0 * int(np.ceil(len(fams) / 2))), squeeze=False)
for ax, f in zip(axes.ravel(), fams):
    t = d[d["family"] == f].groupby("date")[["actual", "forecast"]].sum()
    ax.plot(t.index, t["actual"], color="black", linewidth=2, label="asli")
    ax.plot(t.index, t["forecast"], color=BLUE, linewidth=1.5, label=f"forecast ({best})")
    ax.set_title(f + (" (bias flag)" if f in BIAS_FLAGS else ""), fontsize=10)
    ax.tick_params(axis="x", labelsize=7, rotation=30)
    ax.legend(fontsize=7)
for ax in axes.ravel()[len(fams):]:
    ax.axis("off")
save(fig, "05_actual_vs_forecast_families")

# =====================================================================================
section("7c. Store ke hisaab se ghalti (best model, validation)")
# =====================================================================================
o = OPEN[evm]
st = pd.DataFrame({"store_nbr": df.loc[evm, "store_nbr"].values[o], "y": SALES[evm][o],
                   "p": get_preds(best, "Validation (Aug 2017)")[0][o],
                   "pb": get_preds("Baseline: seasonal naive", "Validation (Aug 2017)")[0][o]})
st["e"], st["eb"] = (st["y"] - st["p"]).abs(), (st["y"] - st["pb"]).abs()
sg = st.groupby("store_nbr").agg(units=("y", "sum"), err=("e", "sum"), err_b=("eb", "sum"), pred=("p", "sum"))
sg["WAPE model %"] = sg["err"] / sg["units"] * 100
sg["WAPE seasonal naive %"] = sg["err_b"] / sg["units"] * 100
sg["forecast/actual"] = sg["pred"] / sg["units"]
sg["kul ghalti mein hissa %"] = sg["err"] / sg["err"].sum() * 100
sg = sg.sort_values("WAPE model %", ascending=False)
print("Sab se zyada WAPE wale stores:")
print(sg[["units", "WAPE model %", "WAPE seasonal naive %", "forecast/actual", "kul ghalti mein hissa %"]].head(10).round(2).to_string())
print(f"\nStores jahan model, seasonal naive se BURA: {int((sg['WAPE model %'] > sg['WAPE seasonal naive %']).sum())} / {len(sg)}")
print(f"Store WAPE ki range: {sg['WAPE model %'].min():.1f}% se {sg['WAPE model %'].max():.1f}% | median {sg['WAPE model %'].median():.1f}%")
n_low, n_high = int((sg["forecast/actual"] < 0.85).sum()), int((sg["forecast/actual"] > 1.15).sum())
top10 = sg.head(10)
print(f"Forecast/actual < 0.85 (kam forecast): {n_low} stores | > 1.15 (zyada forecast): {n_high} stores | "
      f"sab se zyada WAPE wale 10 stores mein forecast/actual ki median: {top10['forecast/actual'].median():.2f}")
if 52 in sg.index:
    r52 = sg.loc[52]
    print(f"Store 52 (sab se naya, Apr 2017 mein khula): WAPE {r52['WAPE model %']:.1f}%, forecast/actual {r52['forecast/actual']:.2f}, "
          f"WAPE rank {list(sg.index).index(52) + 1}/{len(sg)}")
sg.to_csv(REPORT_DIR / "06_validation_error_by_store.csv")
fig, ax = plt.subplots(figsize=(12, 4))
sg2 = sg.sort_values("WAPE model %")
ax.bar(sg2.index.astype(str), sg2["WAPE model %"], color=[RED if r > 1.15 or r < 0.85 else BLUE for r in sg2["forecast/actual"]])
ax.set_xlabel("store_nbr (laal = forecast/actual 0.85 se kam ya 1.15 se zyada)")
ax.set_ylabel("WAPE %")
ax.tick_params(axis="x", labelsize=7, rotation=90)
ax.set_title(f"{best}: har store ki WAPE (validation)")
save(fig, "06_wape_by_store")

# =====================================================================================
section("8. Khulasa (numbers se khud bana hua) aur Phase 07 ke liye faisla")
# =====================================================================================
cands = [best] + [m for m in summary.index if summary.loc[m, "faisla"].startswith("BARABAR")]
base_avg = summary.loc["Baseline: 28-din average", "avg WAPE"]
lines = [
    f"- Best single model: {best} | avg WAPE {summary.loc[best, 'avg WAPE']:.3f}% | avg RMSLE {summary.loc[best, 'avg RMSLE']:.4f}",
    f"- Baseline (28-din average) avg WAPE {base_avg:.3f}% -> best model ki ghalti {(1 - summary.loc[best, 'avg WAPE'] / base_avg):.0%} kam",
    f"- Noise ki hadd (Phase 05): WAPE {NOISE_W:.3f} | RMSLE {NOISE_R:.4f}",
]
for m in summary.index:
    if m not in (best,) and not m.startswith("Baseline") and not m.startswith("LightGBM horizon"):
        lines.append(f"- {m}: ΔWAPE {summary.loc[m, 'ΔWAPE vs best']:+.3f}, ΔRMSLE {summary.loc[m, 'ΔRMSLE vs best']:+.4f} -> {summary.loc[m, 'faisla']}")
lines.append(f"- Seed noise (apna): " + ", ".join(f"{m} {MODEL_NOISE.loc[m, '|ΔWAPE| seed0-seed1']:.3f}" for m in MODEL_NOISE.index))
ovb = ov.loc[best]
lines.append(f"- Overfitting ({best}): training (aakhri 28 din) WAPE {ovb['WAPE training (aakhri 28 din)']:.2f}% vs validation "
             f"{ovb['WAPE validation (agle 16 din)']:.2f}% (farq {ovb['farq (val - train)']:+.2f}); sab models ka farq: "
             + ", ".join(f"{m} {ov.loc[m, 'farq (val - train)']:+.2f}" for m in ov.index))
lines.append(f"- Store-wise ({best}, validation): WAPE {sg['WAPE model %'].min():.1f}%-{sg['WAPE model %'].max():.1f}%, "
             f"seasonal naive se bura {int((sg['WAPE model %'] > sg['WAPE seasonal naive %']).sum())}/{len(sg)} stores")
lines.append(f"- Horizon tareeqa (din 1-7): farq {gain:+.3f} points, {better}/5 windows behtar -> {'APNAO' if HORIZON_OK else 'nahi'}")
lines.append(f"- Phase 07 (Optuna tuning) ke candidates: {cands}")
if BIAS_FLAGS:
    lines.append(f"- Systematic bias (validation): {BIAS_FLAGS} -> forecast asli se bohat alag; Phase 07/08 mein khaas tafteesh.")
lgb_gap = summary.loc["LightGBM", "avg WAPE"] - summary.loc[best, "avg WAPE"]
lgb_thr = tie_thr("LightGBM", best)[0]
if best != "LightGBM" and lgb_gap > lgb_thr:
    lines.append(f"- DHYAN: {best} LightGBM se noise se zyada aage ({lgb_gap:.3f}). Phase 05 ke feature faisle LightGBM par hue thay "
                 "-> Phase 07 se pehle halka feature re-check zaroori.")
else:
    lines.append(f"- LightGBM best se sirf {lgb_gap:.3f} points peeche (noise hadd {lgb_thr:.3f} ke andar) -> Phase 05 ke feature faisle "
                 "(LightGBM par bane) is jeet ke liye bhi maqbool hain; feature re-check ki zaroorat nahi.")
ties = [m for m in summary.index if summary.loc[m, "faisla"].startswith("BARABAR")]
if ties:
    lines.append(f"- DHYAN: {best} aur {ties} ka farq noise ke andar hai, is liye '{best} best hai' sirf label hai, saboot nahi. "
                 "Doosre computer/seed par tarteeb badal sakti hai. Isi liye teeno Phase 07 mein tune honge.")
lines.append("- Ridge (linear): baseline se thora behtar, lekin tree models se bohat peeche. Mumkin wazahat (test nahi kiya): "
             "is data mein asar seedhi lakeer nahi (promo 1.5x ke baad flat, store-wise holiday asar ulta-seedha, Phase 04).")
for ln in lines:
    print(ln)
SKIPPED_NOTE = f"- Skip hue models (library nahi): {SKIPPED}" if SKIPPED else ""
if SKIPPED_NOTE:
    print(SKIPPED_NOTE)
out = {"best_single_model": best, "phase07_candidates": cands, "horizon_strategy_adopted": bool(HORIZON_OK),
       "horizon_gain_days1_7_wape": round(float(gain), 4), "noise": {"WAPE": NOISE_W, "RMSLE": NOISE_R},
       "summary": summary.round(4).reset_index().to_dict(orient="records"), "skipped_models": SKIPPED,
       "windows": WINDOWS, "features": FEATS, "features_h7": FEATS_H7, "bias_flag_families_validation": BIAS_FLAGS,
       "model_seed_noise": MODEL_NOISE.round(4).to_dict(orient="index"), "overfitting_check": ov.round(4).to_dict(orient="index")}
json.dump(out, open(REPORT_DIR / "06_model_comparison_summary.json", "w"), indent=2, default=str)
print(f"\nSaved: reports/06_model_scores.csv, 06_horizon_test.csv, 06_validation_error_by_family.csv, 06_overfitting_check.csv, "
      f"06_validation_error_by_store.csv, "
      f"06_model_comparison_summary.json, charts in {FIG_DIR}")
print(f"Kul waqt: {(time.time() - T_START) / 60:.1f} minute")
