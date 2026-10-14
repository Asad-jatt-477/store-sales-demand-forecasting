# =====================================================================================
# Retail Demand Forecasting System
# Phase 08: Final model training + 16-din ka forecast (16-31 Aug 2017) + Kaggle submission
# -------------------------------------------------------------------------------------
# Is script ka maqsad:
#   1. Phase 07 ka chuna hua model (reports/07_final_config.json) aur us ki settings parhna -- koi nayi tuning nahi.
#   2. Test window ke features check karna: sahi din se aaye? leakage? khaali values validation jaisi?
#   3. Pipeline reproduce check: Phase 08 ka code validation window par wahi nateeja deta hai jo Phase 07 ne diya?
#   4. Do faisle DATA se (usool pehle se tay): (a) 2 seeds ki average? (b) kul forecast par bias correction?
#   5. Final training: 15 Aug 2017 tak ki SAARI qabil-e-training history -> 16-31 Aug ka forecast.
#   6. Zero rules (kabhi na biki series, pre-opening, closure-calendar din) -> submission.csv (Kaggle format).
#   7. Forecast ki sanity: pichle 4 hafte aur pichle saal ke muqable; family ke hisaab se.
#   8. Khule masle: holdout ka ~5% kam forecast kahan se aaya? SCHOOL AND OFFICE SUPPLIES ka season.
#   9. Model save (Phase 10 deployment ke liye) + khulasa JSON.
#
# Sections:
#   0 setup + config           | 1 test window ke features ki jaanch   | 2 pipeline reproduce check
#   3 faisla: seeds ki average | 4 faisla: bias correction            | 5 final training + forecast
#   6 submission.csv + checks  | 7 forecast sanity                    | 8 holdout bias kahan se
#   9 SCHOOL family            | 10 model save + khulasa
#
# Chalane ka tareeqa (data folder se):   python 08_final_forecast.py
# Zaroori: Phase 07 mukammal (reports/07_final_config.json). reports/07_cache mojood ho to bohot waqt bachta hai.
# Waqt: ~2 final trainings (~5-10 min har ek) + cache. Resume: har forecast reports/08_cache mein save hota hai.
# =====================================================================================
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import joblib
import lightgbm as lgb
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    import xgboost as xgb
except ImportError:
    xgb = None

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

PROCESSED, REPORT_DIR, MODEL_DIR = Path("data/processed"), Path("reports"), Path("models")
FIG_DIR = REPORT_DIR / "figures" / "08_final_forecast"
CACHE_07, CACHE_08 = REPORT_DIR / "07_cache", REPORT_DIR / "08_cache"
for d in (FIG_DIR, CACHE_08, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)
KEYS = ["store_nbr", "family"]
RAW_CANDIDATES = [Path("."), Path("data/raw"), Path("../data/raw")]
BLUE, ORANGE, AQUA, RED, GRAY, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#9a9994", "#52514e"
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
section("0. Setup: Phase 07 ka faisla (config) aur data")
# =====================================================================================
cfg = json.load(open(REPORT_DIR / "07_final_config.json"))
if cfg.get("smoke"):
    raise SystemExit("07_final_config.json SMOKE run ka hai -> pehle asli Phase 07 chalayen.")
FEATS, CATS = cfg["features"], cfg["categorical_features"]
NOISE = cfg["noise"]
WINDOWS = cfg["windows"]
VALID, HOLDOUT, TUNE = "Validation (Aug 2017)", cfg["holdout"], cfg["tuning_folds"]
SEL = TUNE + [VALID]                                   # Phase 07 ki faisla windows (holdout kabhi nahi)
MEMBERS = cfg["members"]                               # chuna gaya model (ya ensemble ke members)
print(f"Phase 07 ka faisla: {cfg['choice']} | ensemble: {cfg['use_ensemble']}")
for mb in MEMBERS:
    print(f"  {mb['model']} ({mb['config']}): {mb['params']}")
print(f"Features ({len(FEATS)}): {FEATS}")

df = pd.read_parquet(PROCESSED / "retail_demand_features.parquet")
df["family"] = df["family"].astype(str)
df = df.sort_values(KEYS + ["date"]).reset_index(drop=True)
# first_sale_date (poori history se, Phase 01) -- zero rule wahi rahe jo Phase 05-07 mein parkha gaya
fsd = pd.read_parquet(PROCESSED / "retail_demand_master_treated.parquet", columns=KEYS + ["first_sale_date"])
fsd["family"] = fsd["family"].astype(str)
df = df.merge(fsd.drop_duplicates(KEYS), on=KEYS, how="left")
del fsd
for c in CATS:
    df[c] = df[c].astype("category")
DATES = df["date"].values
Y = df["target_log1p"].values
SALES = df["sales_clean"].values
HIST = (df["split"] != "test").values
TEST = (df["split"] == "test").values
TRAINABLE = (df["is_trainable"] == 1).values
OPEN = ((df["is_pre_opening"] == 0) & (df["is_store_closed"] == 0)).values
CUTOFF = pd.Timestamp(df.loc[HIST, "date"].max())           # aakhri din jis ki sales maloom hai
TEST_START, TEST_END = df.loc[TEST, "date"].min(), df.loc[TEST, "date"].max()
print(f"Rows {len(df):,} | history {df.loc[HIST, 'date'].min().date()} .. {CUTOFF.date()} | "
      f"forecast window {TEST_START.date()} .. {TEST_END.date()} ({TEST.sum():,} rows)")
print(f"Versions: lightgbm {lgb.__version__}, sklearn {sklearn.__version__}, xgboost {xgb.__version__ if xgb else 'NAHI'}")


def wape(y, p):
    return np.abs(y - p).sum() / y.sum() * 100


def rmsle(y, p):
    return float(np.sqrt(np.mean((np.log1p(np.clip(p, 0, None)) - np.log1p(y)) ** 2)))


# =====================================================================================
section("1. Test window ke features ki jaanch (forecast se PEHLE)")
# =====================================================================================
# (a) Leakage/alignment: test din d ka lag_16 = (d - 16) ki sales honi chahiye, aur sab se naya source din <= CUTOFF.
# (b) Khaali (NaN) features: test mein validation jaisi hon -> model ne aisi rows training mein dekhi hain.
src = df.loc[TEST, KEYS + ["date", "lag_16"]].copy()
src["src_date"] = src["date"] - pd.Timedelta(days=16)
hist_sales = df.loc[HIST, KEYS + ["date", "sales_clean", "is_trainable"]].rename(columns={"date": "src_date"})
chk = src.merge(hist_sales, on=KEYS + ["src_date"], how="left")
both_nan = chk["lag_16"].isna() & (chk["is_trainable"] != 1)            # masking rule: non-trainable source -> NaN
match = np.isclose(chk["lag_16"], np.log1p(chk["sales_clean"])) | both_nan
print(f"(a) lag_16 = 16 din pehle ki sales: {match.mean():.2%} rows | NaN sirf masked source par: {both_nan.sum():,} rows | "
      f"sab se naya source din: {chk['src_date'].max().date()} (cutoff {CUTOFF.date()})")
assert match.all(), "lag_16 alignment ghalat -- features file check karein"
assert chk["src_date"].max() <= CUTOFF, "LEAKAGE: test feature cutoff ke baad ki sales dekh raha hai"
va_m = (df["split"] == "valid").values
nan_tab = pd.DataFrame({"training rows": df.loc[TRAINABLE, FEATS].isna().mean() * 100,
                        "validation": df.loc[va_m, FEATS].isna().mean() * 100,
                        "test": df.loc[TEST, FEATS].isna().mean() * 100})
nan_tab = nan_tab[nan_tab.max(axis=1) > 0]
print("(b) Khaali values (%) -- jin features mein koi khaali hai:")
print(nan_tab.round(2).to_string())
l364 = df.loc[TEST & df["lag_364"].isna().values, ["store_nbr", "family"]]
print(f"    test mein lag_364 khaali: {len(l364):,} rows; sab se zyada stores: "
      f"{l364['store_nbr'].value_counts().head(5).to_dict()} (pichle saal yeh din non-trainable/band thay)")
print("    Tree models khaali values ko khud sambhalte hain (Phase 05 test); training mein bhi aisi rows thin.")

# =====================================================================================
section("2. Pipeline reproduce check: Phase 08 ka code, validation window par")
# =====================================================================================
# Wahi model, wahi settings, validation cutoff (30 Jul) tak training -> validation forecast. Phase 07 ke forecast se
# milna chahiye. Agar na mile to final training par bharosa nahi (code mein farq hai).
HAS_MAX_FEATURES = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 4)


def make_model(name, p, seed):
    if name == "LightGBM":
        return lgb.LGBMRegressor(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], num_leaves=p["num_leaves"],
                                 min_child_samples=p["min_child_samples"], subsample=p["subsample"], subsample_freq=1,
                                 colsample_bytree=p["colsample_bytree"], reg_lambda=p["reg_lambda"],
                                 random_state=seed, verbose=-1)
    if name == "XGBoost":
        return xgb.XGBRegressor(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], tree_method="hist",
                                grow_policy="lossguide", max_leaves=p["max_leaves"], max_depth=0,
                                min_child_weight=p["min_child_weight"], subsample=p["subsample"],
                                colsample_bytree=p["colsample_bytree"], reg_lambda=p["reg_lambda"],
                                enable_categorical=True, max_cat_to_onehot=1, random_state=seed, n_jobs=-1)
    extra = {"max_features": p["max_features"]} if HAS_MAX_FEATURES else {}
    return HistGradientBoostingRegressor(max_iter=p["n_estimators"], learning_rate=p["learning_rate"],
                                         max_leaf_nodes=p["max_leaf_nodes"], min_samples_leaf=p["min_samples_leaf"],
                                         l2_regularization=p["l2_regularization"], categorical_features="from_dtype",
                                         early_stopping=False, random_state=seed, **extra)


def win_masks(w):
    cutoff, start = (pd.Timestamp(x) for x in WINDOWS[w])
    trm = TRAINABLE & (DATES <= np.datetime64(cutoff))
    evm = (DATES >= np.datetime64(start)) & (DATES <= np.datetime64(start + pd.Timedelta(days=15)))
    return cutoff, trm, evm


def never_sold(cutoff, rows):
    # Phase 05-07 ka rule: first_sale_date (Phase 01, poori history, 30 Jul 2017 tak) khaali ya cutoff ke baad.
    # Final forecast ke liye cutoff 15 Aug hai: is liye 31 Jul-15 Aug ki sales bhi dekhi jati hain (koi nayi series
    # in dinon mein pehli dafa biki ho to woh "kabhi nahi biki" na gini jaye).
    fs = df.loc[rows, "first_sale_date"]
    never = (fs.isna() | (fs > cutoff)).values
    recent = df.loc[HIST & (DATES <= np.datetime64(cutoff)) & (SALES > 0), KEYS].drop_duplicates().assign(_s=1)
    sold_recent = df.loc[rows, KEYS].merge(recent, on=KEYS, how="left")["_s"].eq(1).values
    return never & ~sold_recent


def zero_rule(cutoff, rows):
    # Rule (feature nahi): kabhi nahi biki / pre-opening / closure-calendar din -> forecast 0
    return never_sold(cutoff, rows) | (df.loc[rows, "is_pre_opening"].values == 1) | (df.loc[rows, "is_closure_calendar_day"].values == 1)


def cpath07(name, cfg_name, w, seed):
    clean = lambda t: "".join(ch if ch.isalnum() else "_" for ch in t)
    return CACHE_07 / f"{clean(name)}__{cfg_name}__{clean(w)}__s{seed}.parquet"


def fit(name, p, trm, seed):
    return make_model(name, p, seed).fit(df.loc[trm, FEATS], Y[trm])


def forecast(model, rows, zr):
    return np.where(zr, 0, np.expm1(model.predict(df.loc[rows, FEATS])).clip(0))


def window_preds(mb, w, seed):
    # Phase 07 cache -> warna Phase 08 cache -> warna train. (Sirf backtest windows ke liye.)
    for cp in (cpath07(mb["model"], mb["config"], w, seed), CACHE_08 / f"{mb['model'].split()[0]}__{w.split()[0]}__s{seed}.parquet"):
        if cp.exists():
            return pd.read_parquet(cp)["pred"].values
    cutoff, trm, evm = win_masks(w)
    t0 = time.time()
    p = forecast(fit(mb["model"], mb["params"], trm, seed), evm, zero_rule(cutoff, evm))
    pd.DataFrame({"pred": p}).to_parquet(CACHE_08 / f"{mb['model'].split()[0]}__{w.split()[0]}__s{seed}.parquet", index=False)
    print(f"    {mb['model']} {w} seed {seed}: train + forecast {time.time() - t0:.0f}s", flush=True)
    return p


def choice_window_preds(w, seed):
    return np.mean([window_preds(mb, w, seed) for mb in MEMBERS], axis=0)


cutoff_v, trm_v, evm_v = win_masks(VALID)
rep_path = CACHE_08 / "reproduce_validation_s0.parquet"
if rep_path.exists():
    p_rep = pd.read_parquet(rep_path)["pred"].values
else:
    t0 = time.time()
    p_rep = np.mean([forecast(fit(mb["model"], mb["params"], trm_v, 0), evm_v, zero_rule(cutoff_v, evm_v)) for mb in MEMBERS], axis=0)
    pd.DataFrame({"pred": p_rep}).to_parquet(rep_path, index=False)
    print(f"  validation par dobara training: {time.time() - t0:.0f}s")
o_v = OPEN[evm_v]
w_rep = wape(SALES[evm_v][o_v], p_rep[o_v])
have07 = all(cpath07(mb["model"], mb["config"], VALID, 0).exists() for mb in MEMBERS)
if have07:
    p07 = np.mean([pd.read_parquet(cpath07(mb["model"], mb["config"], VALID, 0))["pred"].values for mb in MEMBERS], axis=0)
    w07 = wape(SALES[evm_v][o_v], p07[o_v])
    max_diff = float(np.abs(p_rep - p07).max())
    print(f"  Phase 08 code: validation WAPE {w_rep:.3f}% | Phase 07 (seed 0): {w07:.3f}% | sab se bara farq ek row par: {max_diff:.4f} units")
    REPRODUCED = abs(w_rep - w07) < 0.01
    print("  -> " + ("MATCH: Phase 08 ka code wahi model banata hai jo Phase 07 ne parkha." if REPRODUCED else
                     "FARQ: library version / code alag -> neeche ke nateeje dhyan se dekhein."))
else:
    w07, REPRODUCED = None, None
    print(f"  Phase 08 code: validation WAPE {w_rep:.3f}% (Phase 07 cache nahi mila, is liye muqabla config ke score se): "
          f"Phase 07 report {next(r['WAPE (khule)'] for r in cfg['scores'] if r['window'] == VALID):.3f}% (2 seeds avg)")

# =====================================================================================
section("3. Faisla (a): final model 1 seed ya 2 seeds ki average?")
# =====================================================================================
# 2 seeds ki average forecast "ittefaq" ko kam karti hai, lekin waqt 2 guna. Usool (pehle se tay, Phase 07 jaisa):
# average tabhi jab Phase 07 ki faisla windows (4 folds + validation) par faida noise se zyada ho AUR 5 mein se
# kam az kam 4 windows mein behtar. Data: Phase 07 ke seed 0 aur seed 1 forecasts (cache). Holdout sirf report.
seed_rows = []
have_seed_cache = all(cpath07(mb["model"], mb["config"], w, s).exists() for mb in MEMBERS for w in SEL + [HOLDOUT] for s in (0, 1))
if have_seed_cache:
    for w in SEL + [HOLDOUT]:
        _, _, evm = win_masks(w)
        o = OPEN[evm]
        p0, p1 = choice_window_preds(w, 0), choice_window_preds(w, 1)
        y = SALES[evm][o]
        seed_rows.append({"window": w, "seed 0": wape(y, p0[o]), "seed 1": wape(y, p1[o]), "2 seeds avg": wape(y, (p0[o] + p1[o]) / 2)})
    st = pd.DataFrame(seed_rows).set_index("window")
    st["faida (single avg - 2seed)"] = (st["seed 0"] + st["seed 1"]) / 2 - st["2 seeds avg"]
    print(st.round(3).to_string())
    g = float(st.loc[SEL, "faida (single avg - 2seed)"].mean())
    wins = int((st.loc[SEL, "faida (single avg - 2seed)"] > 0).sum())
    USE_2SEEDS = g > NOISE["WAPE"] and wins >= 4
    print(f"Faida (faisla windows) {g:+.3f}, behtar {wins}/5, noise {NOISE['WAPE']:.3f} -> "
          f"{'2 SEEDS ki average' if USE_2SEEDS else '1 seed (faida noise se chhota, waqt bachao)'}")
else:
    USE_2SEEDS, g, wins = False, None, None
    print("Phase 07 ke seed 0/1 forecasts (reports/07_cache) nahi mile -> test nahi ho sakta -> 1 seed (Phase 07 jaisa).")
SEEDS_FINAL = [0, 1] if USE_2SEEDS else [0]

# =====================================================================================
section("4. Faisla (b): kul forecast par bias correction (multiplier) lagana hai?")
# =====================================================================================
# Holdout par forecast/actual 0.945 tha (~5% kam). Kya yeh har window mein hota hai (systematic) ya ek window ka
# ittefaq? Usool (pehle se tay): correction SIRF tab jab Phase 07 ki SAARI 6 windows mein forecast ek hi taraf ho
# (sab kam ya sab zyada). Agar kabhi kam kabhi zyada -> koi systematic bias nahi -> correction nahi (warna noise fit).
fa = [{"window": r["window"], "forecast/actual": r["forecast/actual"]} for r in cfg["scores"]]
fa = pd.DataFrame(fa).set_index("window")
print(fa.round(3).to_string())
under, over = int((fa["forecast/actual"] < 1).sum()), int((fa["forecast/actual"] > 1).sum())
APPLY_BIAS = under == len(fa) or over == len(fa)
BIAS_MULT = float(1 / fa["forecast/actual"].mean()) if APPLY_BIAS else 1.0
print(f"Kam forecast: {under} windows | zyada: {over} windows | average {fa['forecast/actual'].mean():.3f} -> "
      + (f"systematic -> correction x{BIAS_MULT:.3f}" if APPLY_BIAS else "ek taraf nahi -> correction NAHI"))

# =====================================================================================
section("5. Final training: 15 Aug 2017 tak ki saari history -> 16-31 Aug forecast")
# =====================================================================================
# Training rows: is_trainable == 1 aur date <= CUTOFF (validation ke 16 din bhi shamil, kyun ke ab woh history hai).
# Trees ki tadaad Phase 07 wali: validation fit 1.31M rows par tha, ab ~2% zyada rows -> wahi tadaad maqool.
trm_final = TRAINABLE & (DATES <= np.datetime64(CUTOFF))
print(f"Training rows: {trm_final.sum():,} (Phase 07 validation fit: {trm_v.sum():,}; farq {trm_final.sum() / trm_v.sum() - 1:+.1%})")
zr_test = zero_rule(CUTOFF, TEST)
ns = never_sold(CUTOFF, TEST)
print(f"Zero rules test rows par: {zr_test.sum():,} rows ({zr_test.mean():.1%}) -> forecast 0 | kabhi na biki: {ns.sum():,} rows "
      f"({ns.sum() // 16} series) | pre-opening {int(df.loc[TEST, 'is_pre_opening'].sum())} | "
      f"closure-calendar {int(df.loc[TEST, 'is_closure_calendar_day'].sum())}")
preds, models = [], {}
for mb in MEMBERS:
    for s in SEEDS_FINAL:
        mp = MODEL_DIR / f"08_{mb['model'].split()[0]}_s{s}.joblib"
        if mp.exists():
            model = joblib.load(mp)
            print(f"  {mb['model']} seed {s}: saved model load kiya ({mp})")
        else:
            t0 = time.time()
            model = fit(mb["model"], mb["params"], trm_final, s)
            joblib.dump(model, mp, compress=3)
            print(f"  {mb['model']} seed {s}: final training {time.time() - t0:.0f}s -> saved {mp}", flush=True)
        models[(mb["model"], s)] = mp
        preds.append(forecast(model, TEST, zr_test))
P_TEST = np.mean(preds, axis=0) * BIAS_MULT
P_TEST = np.where(zr_test, 0, P_TEST)
print(f"Forecast bana: {len(P_TEST):,} rows | kul {P_TEST.sum():,.0f} units | 0 wali rows {int((P_TEST == 0).sum()):,} | "
      f"min {P_TEST.min():.2f}, max {P_TEST.max():,.0f}")

# =====================================================================================
section("6. submission.csv (Kaggle format) + checks")
# =====================================================================================
out = df.loc[TEST, ["id", "date", "store_nbr", "family"]].copy()
out["sales"] = P_TEST
out["id"] = out["id"].astype("int64")
sub = out[["id", "sales"]].sort_values("id").reset_index(drop=True)
raw_dir = next((p for p in RAW_CANDIDATES if (p / "test.csv").exists()), None)
checks = {"rows = 28,512": len(sub) == 28512, "id unique": sub["id"].is_unique,
          "koi khaali (NaN) nahi": bool(sub["sales"].notna().all()), "sab >= 0": bool((sub["sales"] >= 0).all())}
if raw_dir is not None:
    test_ids = pd.read_csv(raw_dir / "test.csv", usecols=["id"])["id"]
    checks["ids = test.csv ke ids"] = set(sub["id"]) == set(test_ids)
    if (raw_dir / "sample_submission.csv").exists():
        ss = pd.read_csv(raw_dir / "sample_submission.csv")
        checks["columns = sample_submission"] = list(ss.columns) == list(sub.columns)
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
assert all(checks.values()), "submission check fail"
Path("submissions").mkdir(exist_ok=True); sub.to_csv("submissions/submission.csv", index=False)
out.to_parquet(REPORT_DIR / "08_test_forecast.parquet", index=False)
print("Saved: submission.csv (Kaggle par upload), reports/08_test_forecast.parquet (store/family/date ke sath)")

# =====================================================================================
section("7. Forecast ki sanity: pichle 4 hafte aur pichle saal ke muqable")
# =====================================================================================
# Yeh FAISLA nahi, sirf jaanch: kya forecast ka level data ki haqeeqat ke qareeb hai?
daily_hist = df.loc[HIST].groupby("date")["sales_clean"].sum()
daily_fc = out.groupby("date")["sales"].sum()
last28 = daily_hist[CUTOFF - pd.Timedelta(days=27):CUTOFF]
dow_avg = last28.groupby(last28.index.dayofweek).mean()                        # pichle 4 hafton ka har weekday
ref_dow = np.array([dow_avg[d.dayofweek] for d in daily_fc.index])
ly = daily_hist[TEST_START - pd.Timedelta(days=364):TEST_END - pd.Timedelta(days=364)]   # pichle saal wahi weekdays
growth = last28.sum() / daily_hist[CUTOFF - pd.Timedelta(days=27 + 364):CUTOFF - pd.Timedelta(days=364)].sum()
san = pd.DataFrame({"forecast": daily_fc.values, "pichle 4 hafte (same weekday avg)": ref_dow,
                    "pichla saal x growth": ly.values * growth}, index=daily_fc.index.date)
print(san.round(0).to_string())
print(f"Kul 16 din: forecast {daily_fc.sum():,.0f} | pichle 4 hafte jaisa {ref_dow.sum():,.0f} "
      f"({daily_fc.sum() / ref_dow.sum():.3f}x) | pichla saal x growth ({growth:.3f}) {ly.sum() * growth:,.0f} "
      f"({daily_fc.sum() / (ly.sum() * growth):.3f}x)")
# "Normal" kya hai? Pichle saalon mein 16-31 Aug ki ASLI sales apne pichle 4 hafton (same weekday) ka kitna guna thi.
# Sirf clean period (Jun 2015 se) ke saal: 2015, 2016 (2014 kam-recording daur mein hai, Phase 02/04).
hist_ratio = {}
for yr in range(CUTOFF.year - 2, CUTOFF.year):
    c = pd.Timestamp(f"{yr}-{CUTOFF:%m-%d}")
    if c - pd.Timedelta(days=27) < daily_hist.index.min():
        continue
    l = daily_hist[c - pd.Timedelta(days=27):c]
    dw = l.groupby(l.index.dayofweek).mean()
    ts = pd.date_range(c + pd.Timedelta(days=1), periods=16)
    hist_ratio[yr] = float(daily_hist[ts].sum() / sum(dw[t.dayofweek] for t in ts))
fc_ratio = daily_fc.sum() / ref_dow.sum()
print("Pichle saalon mein 16-31 Aug asli / pichle 4 hafte: " + ", ".join(f"{k}: {v:.3f}" for k, v in hist_ratio.items())
      + f" | hamara forecast: {fc_ratio:.3f}")
if hist_ratio:
    gap = fc_ratio / np.mean(list(hist_ratio.values())) - 1
    print(f"-> forecast pichle saalon ke pattern se {gap:+.1%}. Phase 07 backtests ki hadd (forecast/actual "
          f"{fa['forecast/actual'].min():.3f}-{fa['forecast/actual'].max():.3f}) ke andar"
          + (" hai." if fa['forecast/actual'].min() - 1 <= gap <= fa['forecast/actual'].max() - 1 else " NAHI -> dhyan."))
    print("   Correction NAHI (Section 4 ka usool): yeh sirf 2 saal ka ishara hai; Kaggle score asli jawab dega.")

fam_recent = df.loc[HIST & (DATES > np.datetime64(CUTOFF - pd.Timedelta(days=14)))].groupby("family")["sales_clean"].sum() / 14
fam_fc = out.groupby("family")["sales"].sum() / 16
fam = pd.DataFrame({"forecast / din": fam_fc, "pichle 14 din / din": fam_recent})
fam["ratio"] = fam["forecast / din"] / fam["pichle 14 din / din"]
fam = fam.sort_values("ratio")
print("\nFamilies jin ka forecast pichle 14 din se sab se zyada mukhtalif (ratio = forecast / haal ki sales):")
print(pd.concat([fam.head(4), fam.tail(4)]).round(2).to_string())

fig, ax = plt.subplots(figsize=(11, 4.2))
h = daily_hist[CUTOFF - pd.Timedelta(days=60):]
ax.plot(h.index, h.values, color="black", linewidth=1.6, label="asli sales (history)")
ax.plot(daily_fc.index, daily_fc.values, color=BLUE, linewidth=2.2, marker="o", markersize=3, label="forecast 16-31 Aug")
ax.plot(daily_fc.index, ly.values * growth, color=GRAY, linestyle="--", label="pichla saal x growth (sirf muqabla)")
ax.axvline(CUTOFF, color=GRAY, linewidth=1)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e3:,.0f}k"))
ax.set_title("Company ki kul rozana sales: aakhri 60 din + 16 din ka forecast")
ax.legend(fontsize=8)
save(fig, "01_history_and_forecast")

# =====================================================================================
section("8. Holdout ka ~5% kam forecast: kahan se aaya? (Phase 07 ka khula sawal)")
# =====================================================================================
_, _, evm_h = win_masks(HOLDOUT)
o_h = OPEN[evm_h]
p_h = choice_window_preds(HOLDOUT, 0)
hb = pd.DataFrame({"family": df.loc[evm_h, "family"].values[o_h], "y": SALES[evm_h][o_h], "p": p_h[o_h]})
hb = hb.groupby("family")[["y", "p"]].sum()
# Kya yahi families DOOSRI windows mein bhi kam forecast hoti hain? (systematic ya sirf holdout ka ittefaq)
fam_fa = {}
avail = {VALID: p_rep}
for w in TUNE:                                          # Phase 07 cache ho to purani windows bhi (seed 0)
    if all(cpath07(mb["model"], mb["config"], w, 0).exists() for mb in MEMBERS):
        avail[w] = choice_window_preds(w, 0)
for w, pw in avail.items():
    _, _, evm = win_masks(w)
    o = OPEN[evm]
    t = pd.DataFrame({"family": df.loc[evm, "family"].values[o], "y": SALES[evm][o], "p": pw[o]}).groupby("family")[["y", "p"]].sum()
    fam_fa[w.split(" (")[0]] = t["p"] / t["y"]
hb["farq (forecast - asli)"] = hb["p"] - hb["y"]
hb["forecast/actual"] = hb["p"] / hb["y"]
hb["kul kami mein hissa %"] = hb["farq (forecast - asli)"] / hb["farq (forecast - asli)"].sum() * 100
hb = hb.sort_values("farq (forecast - asli)")
print(f"Holdout kul: forecast/actual {hb['p'].sum() / hb['y'].sum():.3f} | kami {hb['p'].sum() - hb['y'].sum():,.0f} units")
print(hb.head(6).round(3).to_string())
top3 = hb.head(3)
print(f"Sab se zyada kami wali 3 families kul kami ka {top3['kul kami mein hissa %'].sum():.0f}% hain.")
ff = pd.DataFrame(fam_fa).loc[top3.index]
ff["Holdout"] = top3["forecast/actual"]
ff["kam forecast (windows)"] = (ff < 1).sum(axis=1).astype(str) + "/" + str(ff.shape[1])
print("\nYahi families doosri windows mein (forecast/actual; 1 se kam = kam forecast):")
print(ff.round(3).to_string())
print("Padhne ka tareeqa: agar har window mein 1 se kam -> systematic (Phase 09 mein tafteesh); warna holdout ka ittefaq.\n"
      "Correction yahan NAHI: holdout kisi faisle ke liye nahi (Phase 07 ka usool).")

# =====================================================================================
section("9. SCHOOL AND OFFICE SUPPLIES: season ka data, aur test window mein kya badla")
# =====================================================================================
# Phase 06/07 mein validation (31 Jul-15 Aug) par forecast asli ka ~20% tha. Wajah data se: model ke features
# kam az kam 16 din purane hain. Neeche dekhein ke har saal season KAB shuru hua, aur test window ke features mein
# season pehle se dikh raha hai ya nahi.
SCH = "SCHOOL AND OFFICE SUPPLIES"
sch = df.loc[HIST & (df["family"] == SCH).values].groupby("date")["sales_clean"].sum()
wk = []
for yr in sorted(set(sch.index.year)):
    s = sch[f"{yr}-07-01":f"{yr}-09-30"]
    if len(s) < 30:
        continue
    weekly = s.resample("W").sum()
    peak = weekly.idxmax()
    start = weekly[weekly > 3 * weekly.iloc[:3].mean()].index.min()      # sirf bayan ke liye: Jul ke pehle 3 hafton ka 3 guna
    wk.append({"saal": yr, "season shuru (hafta)": start.date() if pd.notna(start) else None,
               "peak hafta": peak.date(), "peak hafte ki sales": weekly.max(),
               "data": "15 Aug tak (season jari)" if s.index.max() < pd.Timestamp(f"{yr}-09-30") else "poora"})
print(pd.DataFrame(wk).to_string(index=False))
lv = []
for name, m in [("validation (31 Jul-15 Aug)", va_m), ("test (16-31 Aug)", TEST)]:
    k = m & (df["family"] == SCH).values
    lv.append({"window": name, "lag_16 avg (units)": np.expm1(df.loc[k, "lag_16"]).mean(),
               "roll7_lag16 avg": np.expm1(df.loc[k, "roll7_lag16"]).mean(), "lag_364 avg": np.expm1(df.loc[k, "lag_364"]).mean()})
lv = pd.DataFrame(lv).set_index("window")
print("\nModel ko kya dikh raha hai (SCHOOL, per store-day, units):")
print(lv.round(1).to_string())
sch_fc = out[out["family"] == SCH].groupby("date")["sales"].sum()
sch_recent = sch[CUTOFF - pd.Timedelta(days=13):CUTOFF].mean()
print(f"SCHOOL forecast: {sch_fc.mean():,.0f}/din | pichle 14 din asli: {sch_recent:,.0f}/din | "
      f"pichle saal 16-31 Aug: {sch[TEST_START - pd.Timedelta(days=364):TEST_END - pd.Timedelta(days=364)].mean():,.0f}/din")
print("Faisla (data se): SCHOOL ke liye koi alag rule NAHI. Wajah: (1) har saal season ka waqt alag (upar ki table), aur\n"
      "  data mein sirf 2 poore season (2015, 2016) + 2017 ka shuru -> koi rule pehle test nahi ho sakta (fit hi hoga);\n"
      "  (2) test window ke features mein season pehle se shamil hai (lag_16 validation se kai guna) -> validation wali\n"
      "  kami seedhi test par nahi aati. Yeh family kul units ka chhota hissa hai; Kaggle score aur Phase 09 mein dekhenge.")

fig, ax = plt.subplots(figsize=(11, 4))
for yr, col in zip(sorted(set(sch.index.year))[-3:], [GRAY, ORANGE, BLUE]):
    s = sch[f"{yr}-07-01":f"{yr}-09-30"]
    ax.plot(s.index.dayofyear, s.values, color=col, linewidth=1.4, label=str(yr))
ax.plot(sch_fc.index.dayofyear, sch_fc.values, color=BLUE, linestyle="--", linewidth=2, label="2017 forecast")
ax.set_xlabel("saal ka din (182 = 1 Jul, 213 = 1 Aug, 244 = 1 Sep)")
ax.set_ylabel("rozana units (sab stores)")
ax.set_title("SCHOOL AND OFFICE SUPPLIES: har saal season ka waqt alag")
ax.legend(fontsize=8)
save(fig, "02_school_season")

# =====================================================================================
section("10. Model save + khulasa")
# =====================================================================================
meta = {"phase": "08", "choice": cfg["choice"], "members": MEMBERS, "seeds": SEEDS_FINAL, "bias_multiplier": BIAS_MULT,
        "features": FEATS, "categorical_features": CATS, "target": cfg["target"], "trained_until": str(CUTOFF.date()),
        "training_rows": int(trm_final.sum()), "model_files": [str(v) for v in models.values()],
        "zero_rules": ["kabhi na biki series (cutoff tak sales history)", "is_pre_opening", "is_closure_calendar_day"],
        "forecast_window": [str(TEST_START.date()), str(TEST_END.date())],
        "reproduce_check": {"phase08_validation_wape": round(w_rep, 4), "phase07_validation_wape_s0": None if w07 is None else round(w07, 4),
                            "match": REPRODUCED},
        "seed_average_test": {"gain": g, "windows_better": wins, "adopted": USE_2SEEDS},
        "bias_test": {"under_windows": under, "over_windows": over, "applied": APPLY_BIAS},
        "forecast_total_units": float(P_TEST.sum()),
        "forecast_vs_last4weeks": float(daily_fc.sum() / ref_dow.sum()),
        "forecast_vs_lastyear_growth": float(daily_fc.sum() / (ly.sum() * growth)),
        "history_ratio_aug16_31_vs_4weeks": hist_ratio}
json.dump(meta, open(REPORT_DIR / "08_summary.json", "w"), indent=2, default=str)
lines = [
    f"- Model: {cfg['choice']} ({', '.join(mb['config'] for mb in MEMBERS)} settings), {len(SEEDS_FINAL)} seed, "
    f"training {trm_final.sum():,} rows (15 Aug 2017 tak).",
    f"- Features ki jaanch: lag_16 alignment 100%, sab se naya source din {chk['src_date'].max().date()} -> leakage nahi.",
    f"- Reproduce: validation WAPE {w_rep:.3f}%" + (f" vs Phase 07 {w07:.3f}% -> {'MATCH' if REPRODUCED else 'FARQ'}" if w07 is not None else ""),
    f"- Seeds average: {'apnayi' if USE_2SEEDS else 'nahi'}" + (f" (faida {g:+.3f}, noise {NOISE['WAPE']:.3f})" if g is not None else " (cache nahi)"),
    f"- Bias correction: {'lagayi' if APPLY_BIAS else 'nahi'} (backtests: {under} kam, {over} zyada).",
    f"- Forecast 16 din: {P_TEST.sum():,.0f} units; pichle 4 hafte ka {daily_fc.sum() / ref_dow.sum():.3f}x, "
    f"pichla saal x growth ka {daily_fc.sum() / (ly.sum() * growth):.3f}x; pichle saalon ka normal (4 hafte): "
    + ", ".join(f"{k} {v:.3f}" for k, v in hist_ratio.items()) + ".",
    f"- Holdout kami: sab se zyada {', '.join(top3.index)} ({top3['kul kami mein hissa %'].sum():.0f}% hissa).",
    "- SCHOOL: koi alag rule nahi (season ka waqt har saal alag; test features mein season pehle se dikh raha hai).",
    "- Agla qadam: submission.csv Kaggle par upload -> score (RMSLE) note karein. Phase 07 validation RMSLE "
    f"{next(r['RMSLE (khule)'] for r in cfg['scores'] if r['window'] == VALID):.3f} se muqabla.",
]
for ln in lines:
    print(ln)
print(f"\nSaved: submission.csv, reports/08_test_forecast.parquet, reports/08_summary.json, models/08_*.joblib, charts in {FIG_DIR}")
print(f"Kul waqt: {(time.time() - T_START) / 60:.1f} minute")
