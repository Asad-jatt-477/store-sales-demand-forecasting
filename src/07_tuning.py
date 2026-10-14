# =====================================================================================
# Retail Demand Forecasting System
# Phase 07: Hyperparameter tuning (Optuna) + time-series cross-validation + confirmation
# -------------------------------------------------------------------------------------
# Is script ka maqsad:
#   1. Phase 06 ke teeno barabar candidates (LightGBM, XGBoost, HistGB) ki settings Optuna se tune karna.
#   2. Tuning SIRF 4 purani time windows (folds) par. Aug 2017 validation tuning se bilkul bahar (lock).
#   3. Ek NAYI "untouched" window (holdout) jo ab tak kisi faisle mein istemal nahi hui. Yeh sirf aakhir mein
#      report hoti hai, kisi faisle mein nahi -> is se pata chalta hai ke tuning ka faida asli hai ya nahi.
#   4. Tuned settings ko 2 seeds par default settings se muqabla (noise-aware, faisle ka usool pehle se tay).
#   5. Ensemble (teeno ki average) test, scores khule stores AUR saari rows dono par, aur khule masle
#      (SCHOOL AND OFFICE SUPPLIES, Store 53) ka haal.
#   6. Phase 08 ke liye final model + settings JSON mein save.
#
# Sections:
#   0 setup + data           | 1 windows (4 tuning folds, validation lock, holdout data se chuna)
#   2 default settings       | 3 search space (har range ki wajah)     | 4 Optuna tuning (resume ke sath)
#   5 tuning ke nateeje      | 6 confirmation: default vs tuned (2 seeds)  | 6b tuning ka optimism check
#   6c overfitting check     | 7 ensemble test         | 8 final faisla + saari rows par score + holdout
#   9 khule masle (SCHOOL family, Store 53)            | 10 khulasa + Phase 08 ke liye config
#
# Main metric: WAPE (business: kul units ki ghalti). RMSLE (Kaggle) "guard" hai: tuned settings RMSLE ko noise se
# zyada kharab nahi kar sakti. Badalna ho to PRIMARY = "RMSLE" kar dein (guard khud ulta ho jayega).
#
# Chalane ka tareeqa (data folder se):   python 07_tuning.py
# Zaroori libraries:  python -m pip install optuna pandas numpy lightgbm xgboost scikit-learn matplotlib pyarrow
#
# Waqt: tuning lambi hai. Har model ka budget: N_TRIALS trials YA TIMEOUT_MIN minute (jo pehle ho).
# Resume: Optuna ke trials reports/07_optuna.db (SQLite) mein, aur har forecast reports/07_cache/ mein save hota hai.
# Script ruk jaye / laptop so jaye -> dobara chalayein, wahin se shuru hogi. Sab naye sire se: FRESH_START = True.
# =====================================================================================
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # charts sirf file mein
import matplotlib.pyplot as plt
import lightgbm as lgb
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    import optuna
except ImportError as exc:
    raise SystemExit("optuna install nahi: python -m pip install optuna") from exc
try:
    import xgboost as xgb
except ImportError:
    xgb = None

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

# ------------------------------ settings (yahan badlein) ------------------------------
PRIMARY = "WAPE"            # tuning ka main metric: "WAPE" ya "RMSLE"
N_TRIALS = 0               # har model ke liye trials (resume par pehle wale bhi gine jate hain)
TIMEOUT_MIN = 180           # har model ke liye IS RUN mein tuning ka zyada se zyada waqt (minute); resume par dobara milta hai
SEEDS = [0, 1]              # confirmation ke seeds
FRESH_START = False         # True = purani tuning + cache mita kar naye sire se
SMOKE = os.environ.get("RDF_SMOKE") == "1"   # SIRF code test ke liye (5% rows, 3 trials). Asli run mein OFF.
# ---------------------------------------------------------------------------------------

PROCESSED, REPORT_DIR = Path("data/processed"), Path("reports")
TAG = "07_smoke" if SMOKE else "07"
FIG_DIR = REPORT_DIR / "figures" / ("07_smoke" if SMOKE else "07_tuning")
CACHE_DIR = REPORT_DIR / f"{TAG}_cache"
CACHE_06 = REPORT_DIR / "06_cache"
DB_PATH = REPORT_DIR / f"{TAG}_optuna.db"
if FRESH_START:
    import shutil
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    DB_PATH.unlink(missing_ok=True)
for d in (FIG_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)
if SMOKE:
    N_TRIALS, TIMEOUT_MIN = 3, 10

KEYS = ["store_nbr", "family"]
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
section("0. Setup: Phase 05 ka feature dataset, final features, noise")
# =====================================================================================
if SMOKE:
    print("!!! SMOKE MODE: sirf code test (5% training rows, 3 trials). Yeh nateeje faislon ke liye NAHI hain. !!!")
info = json.load(open(REPORT_DIR / "05_final_features.json"))
FEATS = info["final_features"]
CATS = info["categorical_features"]
NOISE = {"WAPE": info["noise"]["WAPE_points"], "RMSLE": info["noise"]["RMSLE"]}
P_I = 0 if PRIMARY == "WAPE" else 1          # metric tuple (WAPE, RMSLE) mein main metric ki jagah
G_I = 1 - P_I                                  # guard metric
GUARD = "RMSLE" if PRIMARY == "WAPE" else "WAPE"

df = pd.read_parquet(PROCESSED / "retail_demand_features.parquet")
df["family"] = df["family"].astype(str)
df = df.sort_values(KEYS + ["date"]).reset_index(drop=True)
master = pd.read_parquet(PROCESSED / "retail_demand_master_treated.parquet", columns=KEYS + ["date", "sales_clean", "first_sale_date"])
master["family"] = master["family"].astype(str)
df = df.merge(master.drop_duplicates(KEYS)[KEYS + ["first_sale_date"]], on=KEYS, how="left")
df = df.merge(master[KEYS + ["date", "sales_clean"]], on=KEYS + ["date"], how="left") if "sales_clean" not in df else df
del master
for c in CATS:
    df[c] = df[c].astype("category")

DATES = df["date"].values
Y = df["target_log1p"].values
SALES = df["sales_clean"].values
TRAINABLE = (df["is_trainable"] == 1).values
OPEN = ((df["is_pre_opening"] == 0) & (df["is_store_closed"] == 0)).values
SMOKE_ROWS = (np.random.default_rng(0).random(len(df)) < 0.05) if SMOKE else np.ones(len(df), bool)
print(f"Rows: {len(df):,} | features ({len(FEATS)}): {FEATS}")
print(f"Main metric: {PRIMARY} | guard: {GUARD} | Phase 05 noise: WAPE {NOISE['WAPE']:.3f} points, RMSLE {NOISE['RMSLE']:.4f}")
print(f"Versions: optuna {optuna.__version__}, lightgbm {lgb.__version__}, sklearn {sklearn.__version__}, "
      f"xgboost {xgb.__version__ if xgb else 'NAHI'}")

# =====================================================================================
section("1. Windows: 4 tuning folds | validation (lock) | untouched holdout (data se chuna)")
# =====================================================================================
# Tuning folds = Phase 06 ki 4 purani windows. Validation (Aug 2017) Phase 05 ke feature chunne mein istemal ho chuki
# hai, is liye tuning se bahar (warna score asal se acha dikhega). Lekin yeh 5 windows SAB Phase 05 mein dekhi ja
# chuki hain -> koi bhi "bilkul anchhui" nahi. Is liye ek nayi holdout window data ke rule se chuni jati hai:
#   Rule (pehle se tay): validation se pehle ki SAB SE TAAZA 16-din window jo (a) kisi purani window se overlap na kare,
#   (b) us mein koi closure-calendar din (25 Dec / 1 Jan) na ho, (c) earthquake period na ho.
#   Taaza is liye: Kaggle test (Aug 2017) ke sab se qareeb hai. Yeh window kisi FAISLE mein istemal nahi hogi.
TUNE_WINDOWS = {"Jun 2017": ("2017-06-14", "2017-06-15"),      # tarteeb: taaza pehle (pruning pehle 2 folds dekhti hai)
                "Aug 2016": ("2016-07-30", "2016-07-31"),
                "Dec 2016": ("2016-12-02", "2016-12-03"),
                "May 2016": ("2016-05-17", "2016-05-18")}
VALID = "Validation (Aug 2017)"
VALID_WINDOW = {VALID: ("2017-07-30", "2017-07-31")}


def choose_holdout():
    used = [pd.Timestamp(v[1]) for v in {**TUNE_WINDOWS, **VALID_WINDOW}.values()]
    used = [(a, a + pd.Timedelta(days=15)) for a in used]
    day = df.groupby("date").agg(cal=("is_closure_calendar_day", "max"), quake=("is_earthquake_period", "max"))
    first_ok = df["date"].min() + pd.Timedelta(days=365)              # lag_364 ke liye pura saal history
    last_start = pd.Timestamp(VALID_WINDOW[VALID][1]) - pd.Timedelta(days=16)
    for es in pd.date_range(first_ok, last_start)[::-1]:              # taaza se purani taraf
        ee = es + pd.Timedelta(days=15)
        if any(not (ee < a or es > b) for a, b in used):
            continue
        d = day.loc[es:ee]
        if d["cal"].max() == 0 and d["quake"].max() == 0:
            return (es - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), es.strftime("%Y-%m-%d")
    raise RuntimeError("koi holdout window nahi mili")


h_cut, h_start = choose_holdout()
HOLDOUT = f"Holdout ({pd.Timestamp(h_start):%d %b}-{pd.Timestamp(h_start) + pd.Timedelta(days=15):%d %b %Y})"
WINDOWS = {**TUNE_WINDOWS, **VALID_WINDOW, HOLDOUT: (h_cut, h_start)}
TUNE = list(TUNE_WINDOWS)
SEL = TUNE + [VALID]                       # faislon wali windows (holdout kabhi nahi)
ROLE = {**{w: "tuning fold" for w in TUNE}, VALID: "faisla (tuning se bahar)", HOLDOUT: "SIRF report (koi faisla nahi)"}


def masks(cutoff, eval_start):
    te, es = np.datetime64(pd.Timestamp(cutoff)), np.datetime64(pd.Timestamp(eval_start))
    return TRAINABLE & (DATES <= te), (DATES >= es) & (DATES <= es + np.timedelta64(15, "D"))


def zero_rule(cutoff):
    # Phase 05/06 ka rule (feature nahi): cutoff tak kabhi nahi biki / pre-opening / closure-calendar din -> 0
    never = df["first_sale_date"].isna() | (df["first_sale_date"] > pd.Timestamp(cutoff))
    return (never | (df["is_pre_opening"] == 1) | (df["is_closure_calendar_day"] == 1)).values


MASKS = {w: masks(*v) for w, v in WINDOWS.items()}
ZR = {w: zero_rule(v[0]) for w, v in WINDOWS.items()}
for w, (trm, evm) in MASKS.items():
    print(f"  {w:<30} cutoff {WINDOWS[w][0]} | train rows {trm.sum():>9,} | eval rows {evm.sum():>6,} | "
          f"khule store rows {(evm & OPEN).sum():>6,} | {ROLE[w]}")


def wape(y, p):
    return np.abs(y - p).sum() / y.sum() * 100


def rmsle(y, p):
    return float(np.sqrt(np.mean((np.log1p(np.clip(p, 0, None)) - np.log1p(y)) ** 2)))


def metrics(w, p, rows="open"):
    # (WAPE, RMSLE). rows="open": sirf khule stores (Phase 06 jaisa). rows="all": saari eval rows.
    evm = MASKS[w][1]
    y = SALES[evm]
    k = OPEN[evm] if rows == "open" else np.ones(evm.sum(), bool)
    return wape(y[k], p[k]), rmsle(y[k], p[k])


# =====================================================================================
section("2. Default settings (Phase 06) -- muqable ka paimana")
# =====================================================================================
# LightGBM/XGBoost: bilkul Phase 06 wali settings -> un ke Phase 06 forecasts (reports/06_cache) dobara istemal ho
# sakte hain (agar file mojood ho), warna dobara train.
# HistGB: Phase 06 mein early_stopping "auto" ON tha (1M+ rows par sklearn khud 10% random rows alag kar leta hai)
# -> HistGB ne 90% rows par seekha. Yeh leakage nahi tha, lekin barabar muqabla nahi. Ab early_stopping=False
# (sab models pura training data, trees ki tadaad tuning chunti hai), aur HistGB default dobara train hota hai.
HAS_MAX_FEATURES = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 4)
DEFAULTS = {
    "LightGBM": dict(learning_rate=0.07, num_leaves=127, min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                     reg_lambda=0.0, n_estimators=400),
    "XGBoost": dict(learning_rate=0.07, max_leaves=127, min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=1.0, n_estimators=400),
    "HistGB (sklearn)": dict(learning_rate=0.07, max_leaf_nodes=127, min_samples_leaf=50, l2_regularization=0.0,
                             max_features=1.0, n_estimators=400),
}
MODELS = [m for m in DEFAULTS if not (m == "XGBoost" and xgb is None)]
if "XGBoost" not in MODELS:
    print("XGBoost install nahi -> skip (python -m pip install xgboost)")
for m in MODELS:
    print(f"  {m:<18} {DEFAULTS[m]}")


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


def predict_at(name, model, X, ks):
    # Ek hi trained model se kai "trees ki tadaad" par forecast (dobara train kiye baghair).
    if name == "LightGBM":
        return {k: model.predict(X, num_iteration=k) for k in ks}
    if name == "XGBoost":
        return {k: model.predict(X, iteration_range=(0, k)) for k in ks}
    want, out = set(ks), {}
    for i, p in enumerate(model.staged_predict(X), start=1):
        if i in want:
            out[i] = p
    return out


def fit_forecast(name, p, w, seed, ks=None, extra_idx=None):
    # Train (sirf window ki training rows) -> forecast units mein, zero rule ke baad. ks = trees ki checkpoints.
    trm, evm = MASKS[w]
    trm = trm & SMOKE_ROWS
    model = make_model(name, p, seed).fit(df.loc[trm, FEATS], Y[trm])
    ks = ks or [p["n_estimators"]]
    raw = predict_at(name, model, df.loc[evm, FEATS], ks)
    out = {k: np.where(ZR[w][evm], 0, np.expm1(v).clip(0)) for k, v in raw.items()}
    p_ex = None
    if extra_idx is not None:
        p_ex = np.where(ZR[w][extra_idx], 0, np.expm1(model.predict(df.loc[extra_idx, FEATS])).clip(0))
    return out, p_ex


# =====================================================================================
section("3. Search space: har setting ki range aur us ki wajah")
# =====================================================================================
# Wajah (dataset se): 1,782 series, har series ki ~400-800 training din; bohot si choti families mein zero zyada
#   (Phase 02/04). Is liye "leaf mein kam se kam rows" ki range 20 se 1000 tak: bari value = choti/shor wali series
#   doosri series ke sath mil kar seekhti hain; choti value = bari series ki baareek details. Data batayega kaun sa.
# num_leaves 31-511: Phase 06 default 127 beech mein hai; dono taraf 4x tak jagah.
# learning_rate 0.03-0.2: 0.03 par hi ~2000+ trees chahiyen; is se neeche laptop par bohot dheema (amli wajah, andaza nahi).
# Trees ki tadaad: alag se tune NAHI hoti. Har trial cap tak train hota hai aur har 50 trees par forecast check hota
#   hai; behtareen tadaad data chunta hai. cap = 80 / learning_rate (300-2000 ke beech).
#   Yeh 80 data se aaya: pehle 40/lr rakha tha; asli data par test run (har model 2 trials) mein HAR trial ki
#   behtareen tadaad cap par aayi aur ghalti abhi gir rahi thi (misal: XGBoost 400 trees 13.57 -> 650 trees 13.46).
#   Is liye budget 2x kiya. TEST ab bhi chalta hai: behtareen tadaad baar baar cap par aaye -> Section 5 batayega.
# subsample/colsample 0.5-1.0 aur L2 regularization 0.001-100: aam boosting ranges (bahar ki maloomat); sirf 15
#   features hain, is liye colsample 0.5 se neeche nahi.
# HistGB mein row-subsample ka option nahi hai (sklearn), is liye woh setting us ke liye nahi.
LR_RANGE, CAP_BUDGET, CAP_MIN, CAP_MAX, STEP = (0.03, 0.2), 80, 300, 2000, 50
if SMOKE:
    CAP_MAX = 300


def trees_cap(lr):
    return int(np.clip(round(CAP_BUDGET / lr / STEP) * STEP, CAP_MIN, CAP_MAX))


def suggest(trial, name):
    p = {"learning_rate": trial.suggest_float("learning_rate", *LR_RANGE, log=True)}
    if name == "LightGBM":
        p["num_leaves"] = trial.suggest_int("num_leaves", 31, 511, log=True)
        p["min_child_samples"] = trial.suggest_int("min_child_samples", 20, 1000, log=True)
        p["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
        p["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        p["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-3, 100, log=True)
    elif name == "XGBoost":
        p["max_leaves"] = trial.suggest_int("max_leaves", 31, 511, log=True)
        p["min_child_weight"] = trial.suggest_int("min_child_weight", 20, 1000, log=True)   # squared error: weight = rows
        p["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
        p["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        p["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-3, 100, log=True)
    else:
        p["max_leaf_nodes"] = trial.suggest_int("max_leaf_nodes", 31, 511, log=True)
        p["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 20, 1000, log=True)
        p["l2_regularization"] = trial.suggest_float("l2_regularization", 1e-3, 100, log=True)
        p["max_features"] = trial.suggest_float("max_features", 0.5, 1.0) if HAS_MAX_FEATURES else 1.0
    p["n_estimators"] = trees_cap(p["learning_rate"])
    return p


def default_as_trial(name):
    # Pehla trial = Phase 06 default (taa ke tuning ka muqabla usi study ke andar ho). L2 = 0 range mein nahi,
    # is liye 0.001 (takreeban 0).
    d = {k: v for k, v in DEFAULTS[name].items() if k != "n_estimators"}
    for k in ("reg_lambda", "l2_regularization"):
        if k in d:
            d[k] = max(d[k], 1e-3)
    if name == "HistGB (sklearn)" and not HAS_MAX_FEATURES:
        d.pop("max_features")
    return d


for m in MODELS:
    print(f"  {m:<18} learning_rate {LR_RANGE}, leaves 31-511, min rows/leaf 20-1000, L2 0.001-100"
          + (", subsample/colsample 0.5-1.0" if m != "HistGB (sklearn)" else ", max_features 0.5-1.0"))
print(f"  Trees: cap = {CAP_BUDGET}/learning_rate ({CAP_MIN}-{CAP_MAX}), har {STEP} trees par check")

# =====================================================================================
section("4. Optuna tuning: 4 folds par, validation aur holdout bahar")
# =====================================================================================
# Har trial: settings -> 4 folds par train (seed 0) -> har checkpoint par 4 folds ki average WAPE -> behtareen
# checkpoint ki value Optuna ko. Pruning: 8 trials ke baad, jo trial 2 folds ke baad median se bura ho, ruk jata hai
# (waqt bachata hai). Sampler: TPE (seed 42 + pehle se trials) -> naya run dobara chalane par wahi sequence.
STORAGE = f"sqlite:///{DB_PATH.as_posix()}"
# TPE pehle N_STARTUP trials RANDOM chunta hai, phir pichle trials se seekh kar chunta hai (Optuna default 10).
# Hamare trials mehnge hain (~10-15 min); is liye 5 random trials ke baad seekhna shuru (pehle run mein har model
# ke 6-10 random trials ho chuke hain, to resume par TPE foran seekhne wala mode mein hoga).
N_STARTUP = 5


def is_default_trial(t, name):
    # Kya yeh trial Phase 06 default settings wala hai? (params match, chhoti float tolerance ke sath)
    d = default_as_trial(name)
    params = t.params or t.system_attrs.get("fixed_params", {})      # WAITING (queue) trial ke params yahan hote hain
    return set(params) == set(d) and all(abs(float(params[k]) - float(v)) < 1e-9 for k, v in d.items())


def cleanup_stale(study):
    # Pichla run beech mein ruka (Ctrl+C / band) to us waqt chal raha trial DB mein "RUNNING" reh jata hai,
    # bina kisi nateeje ke. Usay FAIL mark karo taa ke ginti aur faislon mein shamil na ho.
    stale = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
    if stale:
        st = optuna.storages.RDBStorage(STORAGE)
        for t in stale:
            st.set_trial_state_values(t._trial_id, state=optuna.trial.TrialState.FAIL)
        print(f"  {len(stale)} adhoora trial (pichla run beech mein ruka tha) FAIL mark kiya: {[t.number for t in stale]}")
    return stale


def objective(trial, name):
    p = suggest(trial, name)
    ks = list(range(STEP, p["n_estimators"] + 1, STEP))
    if p["n_estimators"] not in ks:
        ks.append(p["n_estimators"])
    per_fold = {}
    for i, w in enumerate(TUNE):
        preds, _ = fit_forecast(name, p, w, seed=0, ks=ks)
        per_fold[w] = {k: metrics(w, preds[k]) for k in ks}
        mean_p = {k: np.mean([per_fold[f][k][P_I] for f in per_fold]) for k in ks}
        best_k = min(mean_p, key=mean_p.get)
        trial.report(mean_p[best_k], i)
        if trial.should_prune():
            raise optuna.TrialPruned()
    mean_g = {k: float(np.mean([per_fold[f][k][G_I] for f in TUNE])) for k in ks}
    trial.set_user_attr("ks", ks)
    trial.set_user_attr("mean_primary_by_k", [float(mean_p[k]) for k in ks])
    trial.set_user_attr("mean_guard_by_k", [mean_g[k] for k in ks])
    trial.set_user_attr("best_k", int(best_k))
    trial.set_user_attr("cap", int(p["n_estimators"]))
    trial.set_user_attr("fold_primary", {f: float(per_fold[f][best_k][P_I]) for f in TUNE})
    return float(mean_p[best_k])


def log_trial(study, trial):
    st = trial.state.name
    v = f"{trial.value:.3f}" if trial.value is not None else "-"
    best = f"{study.best_value:.3f}" if any(t.state.name == "COMPLETE" for t in study.trials) else "-"
    extra = f" | trees {trial.user_attrs.get('best_k')}/{trial.user_attrs.get('cap')}" if st == "COMPLETE" else ""
    print(f"    trial {trial.number:>3} {st:<8} {PRIMARY} {v} | best ab tak {best}{extra}", flush=True)


STUDIES = {}
for name in MODELS:
    optuna.create_study(study_name=f"{TAG}_{name}", storage=STORAGE, direction="minimize", load_if_exists=True)
    n_prev = len(optuna.load_study(study_name=f"{TAG}_{name}", storage=STORAGE).trials)
    # Seed = 42 + pehle se trials: naya run wahi sequence (dobara chalane par same), resume par purane trials ki
    # settings dobara nahi aazmata (warna wahi random settings repeat hoti hain -- smoke test mein dekha).
    study = optuna.load_study(study_name=f"{TAG}_{name}", storage=STORAGE,
                              sampler=optuna.samplers.TPESampler(seed=42 + n_prev, multivariate=True, n_startup_trials=N_STARTUP),
                              pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    cleanup_stale(study)
    # Default (Phase 06) trial muqable ka paimana hai: agar COMPLETE nahi (misal: adhoora reh gaya), dobara queue karo
    has_def = any(is_default_trial(t, name) and t.state.name in ("COMPLETE", "WAITING") for t in study.trials)
    if not has_def:
        study.enqueue_trial(default_as_trial(name))
        print(f"  {name}: default (Phase 06) trial abhi COMPLETE nahi -> queue kiya (pehle yahi chalega)")
    done = sum(t.state.name in ("COMPLETE", "PRUNED") for t in study.trials)
    left = max(0, N_TRIALS - done)
    print(f"\n  {name}: pehle se {done} trials | ab {left} aur (ya {TIMEOUT_MIN} min)", flush=True)
    if left:
        t0 = time.time()
        study.optimize(lambda tr: objective(tr, name), n_trials=left, timeout=TIMEOUT_MIN * 60,
                       callbacks=[log_trial], catch=(ValueError,))
        print(f"  {name}: tuning waqt {(time.time() - t0) / 60:.1f} min", flush=True)
    STUDIES[name] = study

# =====================================================================================
section("5. Tuning ke nateeje: behtareen settings (guard ke sath)")
# =====================================================================================
# Chunne ka usool (pehle se tay): COMPLETE trials mein se sab se kam main-metric wala, BASHARTE us ka guard metric
# default trial (Phase 06 settings, 400 trees) se noise se zyada bura na ho.
BEST, trial_tables, cap_hits = {}, {}, {}
for name, study in STUDIES.items():
    comp = [t for t in study.trials if t.state.name == "COMPLETE"]
    t0 = next((t for t in comp if is_default_trial(t, name)), None)
    if t0 is None:
        raise SystemExit(f"{name}: default (Phase 06) trial COMPLETE nahi hua -> script dobara chalayen (woh pehle chalega).")
    ks0 = t0.user_attrs["ks"]
    k400 = ks0.index(400) if 400 in ks0 else int(np.argmin([abs(k - 400) for k in ks0]))
    ref_p, ref_g = t0.user_attrs["mean_primary_by_k"][k400], t0.user_attrs["mean_guard_by_k"][k400]
    rows = []
    for t in comp:
        i = t.user_attrs["ks"].index(t.user_attrs["best_k"])
        g = t.user_attrs["mean_guard_by_k"][i]
        rows.append({"trial": t.number, PRIMARY: t.value, GUARD: g, "guard ok": g <= ref_g + NOISE[GUARD],
                     "trees": t.user_attrs["best_k"], "cap": t.user_attrs["cap"], **t.params})
    tt = pd.DataFrame(rows).sort_values(PRIMARY)
    tt.to_csv(REPORT_DIR / f"{TAG}_trials_{name.split()[0]}.csv", index=False)
    trial_tables[name] = tt
    ok = tt[tt["guard ok"]]
    if ok.empty:                                             # koi trial guard paas nahi -> default hi
        print(f"\n{name}: koi trial {GUARD} guard paas nahi kar saka -> default settings")
        BEST[name] = dict(DEFAULTS[name])
        trial_tables[name], cap_hits[name] = tt, 0
        continue
    bt = ok.iloc[0]
    params = {k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in study.trials[int(bt["trial"])].params.items()}
    params = {**DEFAULTS[name], **params, "n_estimators": int(bt["trees"])}
    if name == "HistGB (sklearn)" and not HAS_MAX_FEATURES:
        params["max_features"] = 1.0
    BEST[name] = params
    top = tt.head(5)
    cap_hits[name] = float((top["trees"] == top["cap"]).mean())        # top trials (5 ya kam) mein cap par aane ka hissa
    n_pruned = sum(t.state.name == "PRUNED" for t in study.trials)
    print(f"\n{name}: {len(comp)} complete, {n_pruned} pruned trials")
    print(f"  default (trial {t0.number}, Phase 06 settings, 400 trees) 4 folds: {PRIMARY} {ref_p:.3f} | {GUARD} {ref_g:.4f}")
    print(f"  behtareen (trial {int(bt['trial'])}): {PRIMARY} {bt[PRIMARY]:.3f} | {GUARD} {bt[GUARD]:.4f} | "
          f"farq vs default {bt[PRIMARY] - ref_p:+.3f} (YEH tuning folds par hai -> optimistic; asli faisla Section 6)")
    print(f"  settings: {params}")
    print(f"  Top-{len(top)} trials mein behtareen trees cap par: {int(round(cap_hits[name] * len(top)))}/{len(top)} "
          + ("-> cap shayad chhota; zyada trees se faida ho sakta hai (dhyan)" if cap_hits[name] >= 0.6 else "-> cap kaafi hai"))
    print(top.round(4).to_string(index=False))

# Chart: best-so-far har trial ke baad
fig, ax = plt.subplots(figsize=(10, 4))
for (name, study), col in zip(STUDIES.items(), [BLUE, ORANGE, AQUA]):
    vals = [(t.number, t.value) for t in study.trials if t.state.name == "COMPLETE"]
    if vals:
        n, v = zip(*sorted(vals))
        ax.plot(n, np.minimum.accumulate(v), marker="o", markersize=3, color=col, label=f"{name} (best ab tak)")
        t0v = [t.value for t in study.trials if t.state.name == "COMPLETE" and is_default_trial(t, name)]
        if t0v:
            ax.axhline(t0v[0], color=col, linestyle=":", linewidth=1)
ax.set_xlabel("trial number")
ax.set_ylabel(f"4 folds ki average {PRIMARY}")
ax.set_title(f"Optuna: har trial ke baad behtareen {PRIMARY} (dotted = default settings, behtareen trees par)")
ax.legend(fontsize=8)
save(fig, "01_optuna_history")

# Chart + table: kaun si setting sab se ahem (fANOVA). Kam trials par yeh andaza kamzor hota hai.
imps = {}
for name, study in STUDIES.items():
    try:
        imps[name] = optuna.importance.get_param_importances(study)
    except Exception as e:                                   # bohot kam complete trials
        print(f"  {name}: importance nahi nikli ({type(e).__name__})")
if imps:
    fig, axes = plt.subplots(1, len(imps), figsize=(4.2 * len(imps), 3.4), squeeze=False)
    for ax, (name, imp) in zip(axes[0], imps.items()):
        s = pd.Series(imp).sort_values()
        ax.barh(s.index, s.values, color=BLUE)
        ax.set_title(name, fontsize=10)
        print(f"  {name} setting importance: " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(imp.items(), key=lambda x: -x[1])))
    fig.suptitle("Kaun si setting WAPE par sab se zyada asar daalti hai (fANOVA)", x=0.01, ha="left", fontweight="bold")
    save(fig, "02_param_importance")

# =====================================================================================
section("6. Confirmation: default vs tuned, 2 seeds, 6 windows")
# =====================================================================================
# Tuning ne seed 0 par 4 folds dekhe; is liye wahan ka faida "optimistic" hai. Ab dono settings ko 2 seeds par saari
# windows par chalaya jata hai. Faisle ka usool (pehle se tay), har model ke liye:
#   tuned APNAO agar: (1) tuning folds par faida > noise hadd, (2) validation par noise se zyada bura NAHI,
#                     (3) tuning folds par guard metric noise se zyada bura nahi.  Warna default rakho.
# Holdout is faisle mein shamil NAHI.
ALL_W = list(WINDOWS)
in_idx = np.flatnonzero(MASKS[VALID][0] & (DATES > np.datetime64(pd.Timestamp(WINDOWS[VALID][0]) - pd.Timedelta(days=28))))


def cpath(name, cfg, w, seed, tag=""):
    clean = lambda t: "".join(ch if ch.isalnum() else "_" for ch in t)
    return CACHE_DIR / f"{clean(name)}__{cfg}__{clean(w)}__s{seed}{tag}.parquet"


def get_preds(name, cfg, w, seed):
    cp = cpath(name, cfg, w, seed)
    if cp.exists():
        return pd.read_parquet(cp)["pred"].values
    # Phase 06 ke forecasts: LightGBM/XGBoost default bilkul wahi settings -> dobara istemal (agar mojood)
    if cfg == "default" and name in ("LightGBM", "XGBoost") and not SMOKE:
        c06 = CACHE_06 / f"{name}__{w.replace(' ', '_').replace('(', '').replace(')', '')}{'' if seed == 0 else f'__seed{seed}'}.parquet"
        if c06.exists():
            p = pd.read_parquet(c06)["pred"].values
            pd.DataFrame({"pred": p}).to_parquet(cp, index=False)
            return p
    params = DEFAULTS[name] if cfg == "default" else BEST[name]
    t0 = time.time()
    extra = in_idx if (w == VALID and seed == 0) else None
    preds, p_ex = fit_forecast(name, params, w, seed, extra_idx=extra)
    p = preds[params["n_estimators"]]
    pd.DataFrame({"pred": p}).to_parquet(cp, index=False)
    if p_ex is not None:
        pd.DataFrame({"pred": p_ex}).to_parquet(cpath(name, cfg, w, seed, "_insample28"), index=False)
    print(f"    {name:<18} {cfg:<8} {w:<30} seed {seed} {time.time() - t0:5.0f}s", flush=True)
    return p


crow = []
for name in MODELS:
    for cfg in ("default", "tuned"):
        for w in ALL_W:
            for s in SEEDS:
                p = get_preds(name, cfg, w, s)
                (wo, ro), (wa, ra) = metrics(w, p, "open"), metrics(w, p, "all")
                evm = MASKS[w][1]
                crow.append({"model": name, "config": cfg, "window": w, "seed": s, "WAPE": wo, "RMSLE": ro,
                             "WAPE all rows": wa, "RMSLE all rows": ra,
                             "forecast/actual": p[OPEN[evm]].sum() / SALES[evm][OPEN[evm]].sum()})
conf = pd.DataFrame(crow)
conf.to_csv(REPORT_DIR / f"{TAG}_confirmation_scores.csv", index=False)
cm = conf.groupby(["model", "config", "window"])[["WAPE", "RMSLE", "WAPE all rows", "RMSLE all rows", "forecast/actual"]].mean()

# Seed noise (har model + config): |seed0 - seed1| ki average saari windows par
sn = conf.pivot_table(index=["model", "config", "window"], columns="seed", values=["WAPE", "RMSLE"])
SEED_NOISE = {}
for (name, cfg), g in sn.groupby(level=[0, 1]):
    SEED_NOISE[(name, cfg)] = {m: float((g[(m, SEEDS[0])] - g[(m, SEEDS[1])]).abs().mean()) for m in ("WAPE", "RMSLE")}


def thr(metric, *keys):
    return max([NOISE[metric]] + [SEED_NOISE[k][metric] for k in keys if k in SEED_NOISE])


print(f"\n{PRIMARY} (2 seeds ki average, khule stores):")
tab = cm[PRIMARY].unstack("window")[ALL_W]
print(tab.round(3).to_string())

FINAL_CFG, decisions = {}, []
for name in MODELS:
    d, t = cm.loc[(name, "default")], cm.loc[(name, "tuned")]
    g_tune = float((d.loc[TUNE, PRIMARY] - t.loc[TUNE, PRIMARY]).mean())
    g_val = float(d.loc[VALID, PRIMARY] - t.loc[VALID, PRIMARY])
    g_hold = float(d.loc[HOLDOUT, PRIMARY] - t.loc[HOLDOUT, PRIMARY])
    guard_worse = float((t.loc[TUNE, GUARD] - d.loc[TUNE, GUARD]).mean())
    th = thr(PRIMARY, (name, "default"), (name, "tuned"))
    adopt = g_tune > th and g_val > -th and guard_worse <= thr(GUARD, (name, "default"), (name, "tuned"))
    FINAL_CFG[name] = "tuned" if adopt else "default"
    folds_better = int((t.loc[TUNE, PRIMARY] < d.loc[TUNE, PRIMARY]).sum())
    decisions.append({"model": name, f"faida tuning folds ({PRIMARY})": g_tune, "folds behtar": f"{folds_better}/4",
                      "faida validation": g_val, "faida holdout (sirf report)": g_hold,
                      f"{GUARD} badla (tuning folds)": guard_worse, "noise hadd": th,
                      "faisla": "TUNED apnao" if adopt else "DEFAULT rakho"})
dec = pd.DataFrame(decisions).set_index("model")
print("\nFaida = default ghalti - tuned ghalti (musbat = tuned behtar):")
print(dec.round(4).to_string())

fig, axes = plt.subplots(1, len(MODELS), figsize=(5.2 * len(MODELS), 4.2), sharey=True, squeeze=False)
x = np.arange(len(ALL_W))
for ax, name in zip(axes[0], MODELS):
    ax.bar(x - 0.2, tab.loc[(name, "default")], 0.4, color=GRAY, label="default")
    ax.bar(x + 0.2, tab.loc[(name, "tuned")], 0.4, color=BLUE, label="tuned")
    ax.axvspan(len(ALL_W) - 1.5, len(ALL_W) - 0.5, color=AQUA, alpha=0.12)
    ax.set_xticks(x, [w.split(" (")[0] for w in ALL_W], rotation=35, fontsize=8)
    ax.set_title(f"{name}: {FINAL_CFG[name]}", fontsize=10)
    ax.legend(fontsize=8)
axes[0][0].set_ylabel(f"{PRIMARY} % (2 seeds avg)")
fig.suptitle("Default vs tuned har window par (hari patti = holdout, kisi faisle mein nahi)", x=0.01, ha="left", fontweight="bold")
save(fig, "03_default_vs_tuned")

# =====================================================================================
section("6b. Tuning ka optimism: jo windows tuning ne dekhi vs jo nahi dekhi")
# =====================================================================================
# Agar tuning folds par faida bara aur validation/holdout par chhota (ya ulta) -> settings folds ke mutabiq "fit" ho
# gayi (selection optimism). Yeh number sirf samajhne ke liye; faisla Section 6 ka hai.
for name in MODELS:
    r = dec.loc[name]
    print(f"  {name:<18} faida: tuning folds {r[f'faida tuning folds ({PRIMARY})']:+.3f} | validation {r['faida validation']:+.3f} | "
          f"holdout {r['faida holdout (sirf report)']:+.3f}")

# =====================================================================================
section("6c. Overfitting check (validation window, seed 0): training ke aakhri 28 din vs agle 16 din")
# =====================================================================================
ovr = []
for name in MODELS:
    for cfg in ("default", "tuned"):
        cp = cpath(name, cfg, VALID, 0, "_insample28")
        if not cp.exists():                                        # Phase 06 cache se aaya tha -> in-sample alag se
            params = DEFAULTS[name] if cfg == "default" else BEST[name]
            _, p_in = fit_forecast(name, params, VALID, 0, extra_idx=in_idx)
            pd.DataFrame({"pred": p_in}).to_parquet(cp, index=False)
        p_in = pd.read_parquet(cp)["pred"].values
        o = OPEN[in_idx]
        w_in = wape(SALES[in_idx][o], p_in[o])
        w_out = conf[(conf["model"] == name) & (conf["config"] == cfg) & (conf["window"] == VALID) & (conf["seed"] == 0)]["WAPE"].iloc[0]
        ovr.append({"model": name, "config": cfg, "WAPE training (aakhri 28 din)": w_in, "WAPE validation": w_out, "farq": w_out - w_in})
ovr = pd.DataFrame(ovr).set_index(["model", "config"])
print(ovr.round(3).to_string())
print("Padhne ka tareeqa: tuned ka farq default se bohot bara ho (jaise 2x) -> tuned settings training data 'yaad' kar rahi hain.")

# =====================================================================================
section("7. Ensemble test: teeno models (final settings) ki seedhi average")
# =====================================================================================
# Sirf EK ensemble test hota hai (barabar weights) -- kai combinations/weights aazmana khud selection optimism hai.
# Pehle check: kya models ki ghaltiyan alag hain? Agar ghaltiyon ka correlation ~1 ho to average se faida nahi hota.
names_final = [(n, FINAL_CFG[n]) for n in MODELS]
evm = MASKS[VALID][1]
o = OPEN[evm]
err = pd.DataFrame({n: SALES[evm][o] - get_preds(n, c, VALID, 0)[o] for n, c in names_final})
print("Validation par ghaltiyon (asli - forecast) ka correlation:")
print(err.corr().round(3).to_string())

ens_rows = []
for w in ALL_W:
    for s in SEEDS:
        p = np.mean([get_preds(n, c, w, s) for n, c in names_final], axis=0)
        (wo, ro), (wa, ra) = metrics(w, p, "open"), metrics(w, p, "all")
        evm_w = MASKS[w][1]
        ens_rows.append({"model": "Ensemble (average)", "config": "final", "window": w, "seed": s, "WAPE": wo, "RMSLE": ro,
                         "WAPE all rows": wa, "RMSLE all rows": ra,
                         "forecast/actual": p[OPEN[evm_w]].sum() / SALES[evm_w][OPEN[evm_w]].sum()})
ens = pd.DataFrame(ens_rows)
final_rows = pd.concat([conf[conf.apply(lambda r: FINAL_CFG.get(r["model"]) == r["config"], axis=1)], ens])
fm = final_rows.groupby(["model", "window"])[["WAPE", "RMSLE", "WAPE all rows", "RMSLE all rows", "forecast/actual"]].mean()
cands = list(dict.fromkeys(final_rows["model"]))
sel_mean = {m: float(fm.loc[m].loc[SEL, PRIMARY].mean()) for m in cands}
singles = [m for m in cands if m != "Ensemble (average)"]
best_single = min(singles, key=sel_mean.get)
th_e = thr(PRIMARY, (best_single, FINAL_CFG[best_single]))
ens_gain = sel_mean[best_single] - sel_mean["Ensemble (average)"]
ens_wins = int((fm.loc["Ensemble (average)"].loc[SEL, PRIMARY] < fm.loc[best_single].loc[SEL, PRIMARY]).sum())
USE_ENS = ens_gain > th_e and ens_wins >= 3
print(f"\nFaisla windows ({len(SEL)}: 4 folds + validation) par average {PRIMARY}:")
for m in sorted(cands, key=sel_mean.get):
    print(f"  {m:<22} {sel_mean[m]:.3f}")
print(f"Usool (pehle se tay): ensemble tabhi jab best single ({best_single}) se noise ({th_e:.3f}) zyada behtar AUR 5 mein "
      f"se kam az kam 3 windows mein behtar. Faida {ens_gain:+.3f}, behtar windows {ens_wins}/5 -> "
      f"{'ENSEMBLE apnao' if USE_ENS else 'single model'}")

# =====================================================================================
section("8. Final faisla + scores (khule stores AUR saari rows) + holdout")
# =====================================================================================
CHOICE = "Ensemble (average)" if USE_ENS else best_single
ties = [m for m in singles if m != best_single and abs(sel_mean[m] - sel_mean[best_single]) <= thr(PRIMARY, (m, FINAL_CFG[m]), (best_single, FINAL_CFG[best_single]))]
print(f"Phase 08 ke liye: {CHOICE}" + (f" (settings: {FINAL_CFG[CHOICE]})" if CHOICE in FINAL_CFG else f" ({names_final})"))
if ties and not USE_ENS:
    print(f"DHYAN: {best_single} aur {ties} ka farq noise ke andar -> '{best_single} best' sirf label hai, saboot nahi.")

# Seasonal naive baseline har window par (Phase 06 jaisa), muqable ke liye
def seasonal_naive(w):
    evm_w = MASKS[w][1]
    p = np.expm1(df.loc[evm_w, "same_wd_mean4"].fillna(df.loc[evm_w, "roll28_lag16"]).fillna(0).values)
    return np.where(ZR[w][evm_w], 0, np.clip(p, 0, None))


rep = []
for w in ALL_W:
    sn_o, sn_a = metrics(w, seasonal_naive(w), "open"), metrics(w, seasonal_naive(w), "all")
    r = fm.loc[CHOICE].loc[w]
    lg = cm.loc[("LightGBM", "default", w)]
    rep.append({"window": w, "role": ROLE[w], "WAPE (khule)": r["WAPE"], "RMSLE (khule)": r["RMSLE"],
                "WAPE (saari rows)": r["WAPE all rows"], "RMSLE (saari rows)": r["RMSLE all rows"],
                "forecast/actual": r["forecast/actual"], "LightGBM default (Phase 06) WAPE": lg["WAPE"],
                "seasonal naive WAPE": sn_o[0]})
rep = pd.DataFrame(rep).set_index("window")
print(rep.round(3).to_string())
print("Saari rows: band stores ke din bhi shamil (model ko band hone ka pata nahi; wahan sirf zero rules kaam karte hain).")
print(f"{HOLDOUT} kisi faisle mein shamil NAHI tha -> yeh number sab se imaandar andaza hai "
      f"(ek hi window, is liye ~±{NOISE['WAPE']:.2f} noise).")

fig, ax = plt.subplots(figsize=(10, 4.2))
xs = np.arange(len(ALL_W))
for m, col in zip(cands, [BLUE, ORANGE, AQUA, INK2]):
    ax.plot(xs, fm.loc[m].loc[ALL_W, PRIMARY], marker="o", color=col, label=m, linewidth=2.4 if m == CHOICE else 1.2)
ax.plot(xs, [rep.loc[w, "seasonal naive WAPE"] for w in ALL_W], color=RED, linestyle="--", label="seasonal naive")
ax.axvspan(len(ALL_W) - 1.5, len(ALL_W) - 0.5, color=AQUA, alpha=0.12)
ax.set_xticks(xs, [w.split(" (")[0] for w in ALL_W], rotation=25, fontsize=8)
ax.set_ylabel(f"{PRIMARY} % (khule stores, 2 seeds avg)")
ax.set_title(f"Final settings har window par (mota = chuna gaya: {CHOICE}; hari patti = holdout)")
ax.legend(fontsize=8, ncol=2)
save(fig, "04_final_candidates")

# =====================================================================================
section("9. Khule masle: SCHOOL AND OFFICE SUPPLIES aur Store 53 (chune gaye model ke sath)")
# =====================================================================================
# Yeh tuning se hal hone ki umeed nahi (andaza); sirf track kiya jata hai. Asli tafteesh Phase 08 mein.


def choice_preds(w, s=0):
    if CHOICE == "Ensemble (average)":
        return np.mean([get_preds(n, c, w, s) for n, c in names_final], axis=0)
    return get_preds(CHOICE, FINAL_CFG[CHOICE], w, s)


iss = []
for w in [VALID, HOLDOUT]:
    evm_w = MASKS[w][1]
    o = OPEN[evm_w]
    p, sn_p, y = choice_preds(w)[o], seasonal_naive(w)[o], SALES[evm_w][o]
    fam = df.loc[evm_w, "family"].values[o]
    st = df.loc[evm_w, "store_nbr"].values[o]
    k = fam == "SCHOOL AND OFFICE SUPPLIES"
    j = st == 53
    iss.append({"window": w, "SCHOOL forecast/actual": p[k].sum() / y[k].sum(), "SCHOOL WAPE": wape(y[k], p[k]),
                "Store 53 WAPE": wape(y[j], p[j]), "Store 53 seasonal naive WAPE": wape(y[j], sn_p[j]),
                "Store 53 forecast/actual": p[j].sum() / y[j].sum()})
iss = pd.DataFrame(iss).set_index("window")
print(iss.round(3).to_string())

# =====================================================================================
section("10. Khulasa (numbers se khud bana hua) + Phase 08 ke liye config")
# =====================================================================================
lines = [f"- Main metric {PRIMARY}, guard {GUARD}. Tuning: 4 folds ({', '.join(TUNE)}); validation aur holdout tuning se bahar."]
for name in MODELS:
    r = dec.loc[name]
    lines.append(f"- {name}: tuning folds faida {r[f'faida tuning folds ({PRIMARY})']:+.3f} ({r['folds behtar']} folds), "
                 f"validation {r['faida validation']:+.3f}, holdout {r['faida holdout (sirf report)']:+.3f}, noise {r['noise hadd']:.3f} "
                 f"-> {r['faisla']}")
lines.append(f"- Ensemble: faida {ens_gain:+.3f} vs {best_single}, {ens_wins}/5 windows -> {'apnaya' if USE_ENS else 'nahi'}")
lines.append(f"- Phase 08 model: {CHOICE}. Validation WAPE {rep.loc[VALID, 'WAPE (khule)']:.3f}% "
             f"(saari rows {rep.loc[VALID, 'WAPE (saari rows)']:.3f}%), holdout WAPE {rep.loc[HOLDOUT, 'WAPE (khule)']:.3f}% "
             f"(seasonal naive {rep.loc[HOLDOUT, 'seasonal naive WAPE']:.3f}%).")
for name in MODELS:
    if cap_hits.get(name, 0) >= 0.6:
        lines.append(f"- DHYAN {name}: top trials ke trees cap par -> zyada trees aazmane ki gunjaish.")
lines.append(f"- SCHOOL AND OFFICE SUPPLIES forecast/actual: validation {iss.loc[VALID, 'SCHOOL forecast/actual']:.2f}, "
             f"holdout {iss.loc[HOLDOUT, 'SCHOOL forecast/actual']:.2f} -> Phase 08 tafteesh.")
lines.append(f"- Store 53 WAPE validation {iss.loc[VALID, 'Store 53 WAPE']:.1f}% vs seasonal naive "
             f"{iss.loc[VALID, 'Store 53 seasonal naive WAPE']:.1f}%.")
lines.append("- Limitation: 5 purani windows Phase 05 ke feature chunne mein dekhi gayi thin; holdout hi anchhui window hai, "
             "aur ek window ka noise ~0.2 point hai. Asli aakhri imtihaan Kaggle test (Phase 08).")
if SMOKE:
    lines.insert(0, "- !!! SMOKE MODE: yeh nateeje sirf code test hain, faislon ke liye NAHI !!!")
for ln in lines:
    print(ln)

final_config = {
    "phase": "07", "smoke": SMOKE, "primary_metric": PRIMARY, "guard_metric": GUARD,
    "choice": CHOICE, "use_ensemble": bool(USE_ENS),
    "members": [{"model": n, "config": c, "params": DEFAULTS[n] if c == "default" else BEST[n]}
                for n, c in (names_final if USE_ENS else [(CHOICE, FINAL_CFG[CHOICE])])],
    "all_models": {n: {"final_config": FINAL_CFG[n], "default": DEFAULTS[n], "tuned": BEST[n]} for n in MODELS},
    "histgb_note": "early_stopping=False (Phase 06 mein auto ON tha)",
    "features": FEATS, "categorical_features": CATS, "target": info["target"], "train_filter": info["train_filter"],
    "zero_rules": info["zero_rules"], "windows": WINDOWS, "tuning_folds": TUNE, "holdout": HOLDOUT,
    "noise": NOISE, "seeds": SEEDS,
    "scores": rep.round(4).reset_index().to_dict(orient="records"),
    "decisions": dec.round(4).reset_index().to_dict(orient="records"),
    "open_issues": iss.round(4).reset_index().to_dict(orient="records"),
}
json.dump(final_config, open(REPORT_DIR / f"{TAG}_final_config.json", "w"), indent=2, default=str)
print(f"\nSaved: reports/{TAG}_final_config.json (Phase 08 isi ko parhega), {TAG}_trials_*.csv, "
      f"{TAG}_confirmation_scores.csv, {DB_PATH.name}, charts in {FIG_DIR}")
print(f"Kul waqt: {(time.time() - T_START) / 60:.1f} minute")
