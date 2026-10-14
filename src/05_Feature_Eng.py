# # Retail Demand Forecasting System
# ## Phase 05: Feature Engineering (banana, leakage test, aur ablation se final karna)

# | Phase | Notebook | Nateeja |
# |---|---|---|
# | 01 / 01b | Data preparation, decision tests | Master dataset, flags, rules, leakage usool (history ≥ 16 din) |
# | 02 | Pre-EDA | Data ki nature, clean period (Jun 2015 se) |
# | 03 | Outliers | Test ke baad koi value nahi badli; seed noise ~0.10 WAPE points |
# | 04 | Post-EDA | 37 features ka plan: 15 Keep, 16 Ablation, 4 Drop |
# | **05** | **Feature Engineering (yeh notebook)** | Features banana, leakage test, aur **ablation se final set** |

# **Is phase ka maqsad:** Phase 04 ne har feature ka **rishta** naapa tha (correlation). Lekin rishta hona aur model ko faida dena **alag baatein** hain (Phase 03 ka sabaq). Yahan har "Ablation" feature ko **asli model ke sath** test kiya jayega: model us feature ke sath aur us ke baghair, 4 alag waqton par.

# **Kahani ki misaal:** ek cricket team mein naya khiladi lena hai. Practice mein woh acha lagta hai (correlation). Asal sawal yeh hai ke **team mein aane se team jeet-ti zyada hai ya nahi?** Is liye usay 4 matches khilao, aur nateeja us farq se compare karo jo **sirf qismat** se aata hai (seed noise aur placebo). Agar farq qismat se bara hai, tabhi khiladi team mein rahega.

# **Usool (pichle phases se):**
# 1. Har sales-history feature **≥ 16 din purana** (forecast ka aakhri din 16 din aage hai).
# 2. Lag source ke band/unavailable din aur 4 kam-likhi families ke partial din = **NaN** (Phase 04, Section 3a/5).
# 3. Target = `log1p(sales_clean)` (Phase 02 test + Phase 03: `sales_clean == sales`).
# 4. Training sirf `is_trainable == 1` aur clean period (Jun 2015 se).
# 5. Rules (feature nahi): dead series, pre-opening, closure-calendar din → 0.
# 6. Imtihaan **asli, bina chhuyi** sales par.
# 7. **Koi faisla tab tak nahi jab tak woh noise se bara na ho.**

# | Section | Kaam |
# |---|---|
# | 0 | Setup |
# | 1 | Feature plan load karna (Phase 04) |
# | 2 | Lag source ka rule dobara sabit karna |
# | 3 | Saare features banana (ek function) |
# | 4 | **Leakage tests** |
# | 5 | Features ki sanity check (missing, range, forecast window) |
# | 6 | Reference model, encoding/scaling ka faisla, windows, metrics, baseline |
# | 6b | Test: scaling tree model ko badalti hai? |
# | 7 | Base model ("Keep" features) + **noise ka andaza** (seed + placebo) |
# | 8 | **Ablation:** har group alag |
# | 9 | Faisla table |
# | 10 | Final feature set ki tasdeeq (forward) |
# | 10b | **Backward check:** permutation importance + drop-one |
# | 10c | `month` ka khaas August test |
# | 10d | Final feature set ka report card |
# | 10e | May 2016 RMSLE ki tafteesh |
# | 11 | Model kin features par chalta hai (gain + SHAP, ehtiyat ke sath) |
# | 12 | Files save karna |
# | 13 | Khulasa |

# **Runtime note:** Section 7–10d mein ~90 model train hote hain. Aap ke computer par is mein **1.5–2.5 ghante** lag sakte hain. Results `reports/05_ablation_results_v2.csv` mein save hote hain (v2: naya zero-rule, purani cache file use nahi hogi). Dobara chalane par (`FORCE_RERUN = False`) purane results load ho jate hain, aur model dobara train nahi hote.



# ------------------------------------------------------------
## 0. Setup

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import lightgbm as lgb

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*obj.round has no effect.*")
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 220)

PROCESSED = Path("data/processed")
REPORT_DIR = Path("reports")
FIG_DIR = REPORT_DIR / "figures" / "05_feature_engineering"
FIG_DIR.mkdir(parents=True, exist_ok=True)
FORCE_RERUN = False            # True = ablation dobara chalao, chahe purane results maujood hon

BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
GRAY, LIGHT_GRAY = "#9a9994", "#d9d8d4"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e8e7e3"
plt.rcParams.update({
    "figure.figsize": (10, 4.2), "figure.dpi": 110, "savefig.dpi": 150,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 10, "axes.labelsize": 10, "axes.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c8c7c2", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "text.color": INK, "legend.frameon": False, "legend.fontsize": 9,
})

def display(obj):
    """Notebook ke display() ki jagah: script mein table print karta hai."""
    print(obj.to_string() if hasattr(obj, "to_string") else obj)
    print()

def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)      # script mein popup/blocking nahi; figure disk par save ho chuki hai

KEYS = ["store_nbr", "family"]
CLEAN_START = pd.Timestamp("2015-06-01")
master = pd.read_parquet(PROCESSED / "retail_demand_master_treated.parquet")
master["family"] = master["family"].astype(str)
master = master.sort_values(KEYS + ["date"]).reset_index(drop=True)
TRAIN_END = master.loc[master["split"] == "train", "date"].max()
VALID_END = master.loc[master["split"] == "valid", "date"].max()
assert (master.groupby(KEYS)["date"].diff().dropna() == pd.Timedelta(days=1)).all()   # lag = row shift sahi
assert np.array_equal(master["sales_clean"].fillna(-1).values, master["sales"].fillna(-1).values)
print(f"Rows: {len(master):,} | train end {TRAIN_END.date()} | valid end {VALID_END.date()} | "
      f"test: {master.loc[master['split'] == 'test', 'date'].min().date()} -> {master['date'].max().date()}")


# -------------------------------------------------------------

## 1. Feature plan load karna (Phase 04)
# Phase 04 ki table hi is notebook ka **naqsha** hai. Neeche har plan-row ko asli columns se joda gaya hai. Ablation wale features **groups** mein test honge. Ek group = ek idea, jaise "promo apne normal ke muqable". Ek idea ke do roop (misaal: `roll14` aur `ewm28`, dono "taaza level") ek sath test hote hain.

plan = pd.read_csv(REPORT_DIR / "04_feature_plan.csv")
print(plan["faisla"].value_counts().to_string())

KEEP = ["store_cat", "family_cat", "store_type", "store_cluster", "city",               # static
        "dow", "dom",                                                                   # calendar
        "is_national_holiday", "is_holiday_season_day", "is_national_event", "is_earthquake_period",
        "onpromotion",                                                                  # promotion
        "same_wd_mean4", "roll28_lag16", "lag_364", "roll7_lag16", "lag_16", "lag_21"]  # history

ABLATION_GROUPS = {
    "state":                      ["state"],
    "days_to_month_end":          ["days_to_month_end"],
    "month":                      ["month"],
    "local + regional holiday":   ["is_local_holiday", "is_regional_holiday"],
    "workday override":           ["is_workday_override"],
    "store_holiday_effect":       ["store_holiday_effect"],
    "day before holiday":         ["is_day_before_holiday"],
    "carnaval sunday":            ["is_carnaval_sunday"],
    "promo_ratio28":              ["promo_ratio28"],
    "roll14 + ewm28":             ["roll14_lag16", "ewm28_lag16"],
    "lag_28":                     ["lag_28"],
    "lag_35 + lag_42":            ["lag_35", "lag_42"],
    "own_trend + fam_trend":      ["own_trend", "fam_trend"],
    "fam_roll28":                 ["fam_roll28_lag16"],
    "zero_share + std":           ["zero_share28_lag16", "std28_lag16"],
    "days_since_open":            ["days_since_open"],
}
DROPPED = ["roll56_lag16", "store_roll28_lag16", "oil_price", "transactions"]
NEVER = ["is_trainable", "is_store_closed", "outlier_cause", "outlier_robust_z", "outlier_action",
         "is_outlier_treated", "id", "sales", "sales_clean"]
n_abl = sum(len(v) for v in ABLATION_GROUPS.values())
print(f"\nKeep columns: {len(KEEP)} | Ablation groups: {len(ABLATION_GROUPS)} ({n_abl} columns) | Drop: {len(DROPPED)}")


# ----------------------------------------------------------------

## 2. Lag source ka rule dobara sabit karna
# Phase 04 (Section 3a) ne dikhaya tha ke 2014–15 ke partial-recording dinon mein sirf 4 families **kam likhi** gayi thin. Yahan wahi test dobara chalta hai, taa ke yeh notebook **khud apna saboot** rakhe aur kisi purani file ke number par na tike.

unav = master[master["is_family_unavailable"] == 1].groupby("date")["family"].nunique()
PARTIAL_DATES = pd.DatetimeIndex(unav[unav >= 5].index)
tr = master["split"] == "train"
y_raw = np.where((master["is_trainable"] == 1) & tr, np.log1p(master["sales_clean"]), np.nan)
g = pd.Series(y_raw).groupby([master["store_nbr"], master["family"]])
t = pd.DataFrame({"family": master["family"], "lag364": g.shift(364).values,
                  "src_partial": master.groupby(KEYS)["date"].shift(364).isin(PARTIAL_DATES).values,
                  "y": y_raw, "use": (master["date"] >= CLEAN_START) & tr & (master["is_trainable"] == 1)})
t = t[t["use"]].dropna(subset=["lag364", "y"])
gap = (t.assign(d=t["lag364"] - t["y"]).groupby(["family", "src_partial"])["d"].mean().unstack())
gap["partial - normal"] = gap[True] - gap[False]
GAP_THR = -0.30
UNDER_FAMS = sorted(gap.index[gap["partial - normal"] < GAP_THR])
print("Kam-likhi families (partial dinon mein lag source = NaN):", UNDER_FAMS)
print(f"Baqi families ka farq: {gap.drop(UNDER_FAMS)['partial - normal'].min():.2f} se "
      f"{gap.drop(UNDER_FAMS)['partial - normal'].max():.2f}")

# ------------------------------------------------------------------

## 3. Saare features banana (ek function)
# Ek hi function `build_features(df)` saare features banata hai. Is ke do faide hain:
# 1. Training, validation aur test ke liye **bilkul ek jaisa** hisaab.
# 2. Leakage test (Section 4) isi function ko dobara chala kar hota hai.

# | Feature | Kaise bana | Leakage-safe kyun |
# |---|---|---|
# | `lag_L` (16, 21, 28, 35, 42, 364) | `y_src` ko L din peechhe khiskana | Sab ≥ 16 din purane |
# | `roll7/14/28_lag16`, `ewm28_lag16` | 16 din peechhe khiskaye `y_src` ki rolling average / EWM | Sab se naya din bhi 16 din purana |
# | `same_wd_mean4` | lag 21, 28, 35, 42 ki average | ≥ 21 din |
# | `own_trend` | `roll7_lag16 − roll28_lag16` | ≥ 16 din |
# | `fam_roll28_lag16`, `fam_trend` | Family ka saare stores mein average `y_src`, phir 16 din khiskana | ≥ 16 din |
# | `zero_share28_lag16`, `std28_lag16` | Pichle 28 (16 din peechhe) dinon mein zero ka hissa / std | ≥ 16 din |
# | `promo_ratio28` | `(onpromotion+1) / (pichle 28 din ki median + 1)` | Promo plan pehle se maloom (test file mein hai) |
# | `store_holiday_effect` | Har store ka **pichle** national holidays par average asar | Sirf woh holidays jo ≥ 16 din pehle guzar chuke, aur asar sirf **peechhe** ke normal se |
# | `is_day_before_holiday`, `is_carnaval_sunday`, `days_to_month_end`, `dow`, `dom`, `month` | Calendar se | Calendar pehle se maloom |
# | `days_since_open` | Store khulne ke baad din, **56 par cap** | Opening date maloom; cap kyun ke Phase 04 mein ramp hafte 3–4 tak khatam |

CAT_COLS = ["store_cat", "family_cat", "store_type", "city", "state"]

def build_features(df, mask_sources=True):
    # df: master jaisa (sorted by store, family, date; har series ke din lagataar). sales jahan NaN, woh "maloom nahi".
    out = df[KEYS + ["date", "split", "onpromotion", "store_type", "store_cluster", "city", "state", "store_open_date",
                     "is_national_holiday", "is_regional_holiday", "is_local_holiday", "is_holiday_season_day",
                     "is_workday_override", "is_national_event", "is_earthquake_period", "national_desc"]].copy()
    # --- Lag source: sirf bharose wale din ---
    if mask_sources:
        ok = ((df["is_trainable"] == 1) & df["sales_clean"].notna()
              & ~(df["date"].isin(PARTIAL_DATES) & df["family"].isin(UNDER_FAMS)))
    else:                                   # Section 7b ke test ke liye: band din = 0, partial din jaise likhe gaye
        ok = df["sales_clean"].notna()
    y_src = pd.Series(np.where(ok, np.log1p(df["sales_clean"]), np.nan), index=df.index)
    keys = [df["store_nbr"], df["family"]]
    gs = y_src.groupby(keys)
    for L in (16, 21, 28, 35, 42, 364):
        out[f"lag_{L}"] = gs.shift(L)
    s16 = gs.shift(16)
    g16 = s16.groupby(keys)
    for w in (7, 14, 28):
        out[f"roll{w}_lag16"] = g16.transform(lambda s: s.rolling(w, min_periods=max(2, w // 2)).mean())
    out["ewm28_lag16"] = g16.transform(lambda s: s.ewm(halflife=14, min_periods=7, ignore_na=True).mean())
    out["std28_lag16"] = g16.transform(lambda s: s.rolling(28, min_periods=14).std())
    out["same_wd_mean4"] = out[["lag_21", "lag_28", "lag_35", "lag_42"]].mean(axis=1)
    out["own_trend"] = out["roll7_lag16"] - out["roll28_lag16"]
    zero_src = pd.Series(np.where(ok, (df["sales_clean"] == 0).astype(float), np.nan), index=df.index)
    out["zero_share28_lag16"] = zero_src.groupby(keys).shift(16).groupby(keys).transform(
        lambda s: s.rolling(28, min_periods=14).mean())
    # --- Family-level history (saare stores) ---
    fam_day = y_src.groupby([df["family"], df["date"]]).mean().rename("fy").reset_index().sort_values(["family", "date"])
    fs = fam_day.groupby("family")["fy"].shift(16)
    fam_day["fam_roll28_lag16"] = fs.groupby(fam_day["family"]).transform(lambda s: s.rolling(28, min_periods=14).mean())
    fam_day["fam_roll7_lag16"] = fs.groupby(fam_day["family"]).transform(lambda s: s.rolling(7, min_periods=4).mean())
    fam_day["fam_trend"] = fam_day["fam_roll7_lag16"] - fam_day["fam_roll28_lag16"]
    out = out.merge(fam_day[["family", "date", "fam_roll28_lag16", "fam_trend"]], on=["family", "date"], how="left")
    # --- Promotion ---
    out["promo_ratio28"] = (out["onpromotion"] + 1) / (
        out.groupby(KEYS)["onpromotion"].transform(lambda s: s.shift(1).rolling(28, min_periods=7).median()) + 1)
    # --- Store ka holiday-rawaiya: sirf PICHLE holidays (>= 16 din pehle), asar sirf PEECHHE ke normal se ---
    dow = df["date"].dt.dayofweek
    trail = y_src.groupby([df["store_nbr"], df["family"], dow]).transform(
        lambda s: s.shift(1).rolling(4, min_periods=3).median())                 # usi weekday ke pichle 4 hafte
    hol = (df["is_national_holiday"] == 1) & (df["is_workday_override"] == 0)
    r = pd.DataFrame({"store_nbr": df["store_nbr"], "date": df["date"], "r": y_src - trail})[hol.values].dropna()
    sh = r.groupby(["store_nbr", "date"])["r"].mean().reset_index().sort_values(["store_nbr", "date"])
    sh["cum_mean"] = sh.groupby("store_nbr")["r"].transform(lambda s: s.expanding().mean())
    sh["usable_from"] = (sh["date"] + pd.Timedelta(days=16)).astype(out["date"].dtype)   # dono columns ki date-resolution ek jaisi
    tmp = out[["store_nbr", "date"]].drop_duplicates().sort_values("date")
    sh = sh.sort_values("usable_from")
    tmp = pd.merge_asof(tmp, sh[["store_nbr", "usable_from", "cum_mean"]], left_on="date", right_on="usable_from",
                        by="store_nbr", direction="backward")
    out = out.merge(tmp[["store_nbr", "date", "cum_mean"]].rename(columns={"cum_mean": "store_holiday_effect"}),
                    on=["store_nbr", "date"], how="left")
    # --- Calendar ---
    out["dow"] = out["date"].dt.dayofweek
    out["dom"] = out["date"].dt.day
    out["month"] = out["date"].dt.month
    out["days_to_month_end"] = out["date"].dt.days_in_month - out["dom"]
    nat_days = pd.DatetimeIndex(df.loc[hol, "date"].unique())
    out["is_day_before_holiday"] = ((out["date"] + pd.Timedelta(days=1)).isin(nat_days) & ~out["date"].isin(nat_days)).astype("int8")
    carn = df.loc[df["national_desc"].astype(str).str.contains("Carnaval"), "date"].drop_duplicates()
    out["is_carnaval_sunday"] = out["date"].isin(carn[carn.dt.dayofweek == 0] - pd.Timedelta(days=1)).astype("int8")
    out["days_since_open"] = (out["date"] - out["store_open_date"]).dt.days.clip(lower=0, upper=56)
    # --- Categories ---
    out["store_cat"] = out["store_nbr"].astype("category")
    out["family_cat"] = out["family"].astype("category")
    for c in ["store_type", "city", "state"]:
        out[c] = out[c].astype("category")
    return out.drop(columns=["national_desc", "store_open_date"])

t0 = time.time()
F = build_features(master)
assert len(F) == len(master) and (F[KEYS + ["date"]].values == master[KEYS + ["date"]].values).all()
ALL_FEATURES = KEEP + [c for cols in ABLATION_GROUPS.values() for c in cols]
missing_cols = [c for c in ALL_FEATURES if c not in F.columns]
assert not missing_cols, missing_cols
assert not set(NEVER) & set(ALL_FEATURES) and not set(DROPPED) & set(ALL_FEATURES)
print(f"Features bane: {len(ALL_FEATURES)} columns ({time.time() - t0:.0f}s) | NEVER/DROP list ka koi column shaamil nahi: True")


# --------------------------------------------------------------------

## 4. Leakage tests
# **Test 1 (validation):** validation ki **saari sales mita do** (NaN), features dobara banao. Agar validation rows ka koi bhi feature badla, to woh chupke se validation ki sales use kar raha tha.

# **Test 2 (backtest window):** wahi test ek purani window (18 May 2016 se) par: us din ke **baad** ki saari sales mita do. Yeh zaroori hai kyun ke Section 8 ke backtests mein features ek hi baar bante hain. Yeh test sabit karta hai ke kisi window ke features us window ki ya us ke baad ki sales nahi dekhte.

# `store_holiday_effect` aur `promo_ratio28` jaise naye features ke liye yeh test khaas taur par zaroori hai.

def leakage_test(cut_start, check_start, check_end):
    blind = master.copy()
    blind.loc[blind["date"] >= cut_start, "sales_clean"] = np.nan
    Fb = build_features(blind)
    m = F["date"].between(check_start, check_end).values
    bad = []
    for c in ALL_FEATURES:
        a, b = F.loc[m, c], Fb.loc[m, c]
        if isinstance(a.dtype, pd.CategoricalDtype):
            same = (a.astype(str).values == b.astype(str).values).all()
        else:
            same = np.allclose(a.astype(float).values, b.astype(float).values, equal_nan=True)
        if not same:
            bad.append(c)
    return bad

t0 = time.time()
bad1 = leakage_test(pd.Timestamp("2017-07-31"), pd.Timestamp("2017-07-31"), VALID_END)
print(f"Test 1 (validation): badalne wale features = {bad1 if bad1 else 'koi nahi'}")
bad2 = leakage_test(pd.Timestamp("2016-05-18"), pd.Timestamp("2016-05-18"), pd.Timestamp("2016-06-02"))
print(f"Test 2 (18 May 2016 window): badalne wale features = {bad2 if bad2 else 'koi nahi'}  ({time.time() - t0:.0f}s)")
assert not bad1 and not bad2, "LEAKAGE"

# ------------------------------------------------------------------

## 5. Features ki sanity check
# Har feature ke liye: training mein kitna **missing** hai, kya range **samajh mein aati** hai, aur kya yeh **forecast window (test)** mein maujood hai. Jo binary feature test window mein **ek din bhi active nahi**, woh is forecast par asar nahi daal sakta. Yeh Section 8 ke faisle mein kaam aayega.

train_rows = (F["split"] == "train") & (F["date"] >= CLEAN_START) & (master["is_trainable"] == 1).values
test_rows = F["split"] == "test"
rows = []
for c in ALL_FEATURES:
    s, st = F.loc[train_rows, c], F.loc[test_rows, c]
    num = not isinstance(s.dtype, pd.CategoricalDtype)
    rows.append({"feature": c, "group": "KEEP" if c in KEEP else next(k for k, v in ABLATION_GROUPS.items() if c in v),
                 "train missing %": s.isna().mean() * 100,
                 "train min": s.min() if num else np.nan, "train max": s.max() if num else np.nan,
                 "test missing %": st.isna().mean() * 100,
                 "test mein active (binary)": int((st == 1).sum()) if num and set(s.dropna().unique()) <= {0, 1} else np.nan})
sanity = pd.DataFrame(rows).set_index("feature")
display(sanity.round(3))

# -------------------------------------------------------------------

## 6. Model aur imtihaan ka setup
# **Pehle ek zaroori wazahat: "reference model".** Features ko test karne ke liye **koi ek** model chahiye jis se score naapa jaye. Yeh **final model ka chunaav nahi** hai. Final model Phase 06 mein kai models ke muqable se chuna jayega. Yahan LightGBM ko **reference** liya gaya hai, aur yeh is data ki nature ki wajah se hai:
# 1. **Missing values khud sambhalta hai.** `lag_364` mein ~12% NaN hai, aur yeh NaN jaan boojh kar hain (Section 2).
# 2. **Categories khud sambhalta hai** (54 stores, 33 families).
# 3. **Seedhi lakeer ke baghair rishte pakarta hai.** Promo ka asar 1.5x ke baad flat hai, aur holiday ka asar har store par ulta-seedha (Phase 04).
# 4. **Ek hi model 1,782 series ke liye kaafi tez hai.** Phase 01b mein yeh simple baseline se bohat behtar tha.

# **Is ka asar (imaandari se):** is notebook ke feature faisle **is reference model ke hisaab se** hain. Agar Phase 06 mein koi aur model jeeta, to us ke sath feature faisle ka halka test dobara hoga.

# **Encoding, scaling aur NaN ka faisla (model-specific):**

# | Model qisam | Encoding | Scaling | NaN |
# |---|---|---|---|
# | Tree (LightGBM / XGBoost / CatBoost) | Native categorical (pandas `category`) | **Zaroorat nahi** (Section 6b mein test) | Khud sambhalte hain |
# | Linear (Ridge) / Neural net | One-hot / embeddings | **Zaroori** (StandardScaler) | Impute + "missing" flag |

# Is liye `retail_demand_features.parquet` mein features **raw** rakhe gaye hain (koi scaling/one-hot nahi). Har model Phase 06 mein apni preprocessing **apne pipeline mein**, sirf **training data par fit** kar ke karega, taa ke validation ka data scaler/encoder mein na jaye (leakage).

# - **Reference model ki settings:** Phase 01b/03 jaisi, `log1p` target par.
# - **Rules:** jo series **cutoff tak kabhi nahi biki**, pre-opening aur closure-calendar din → 0. Final forecast (cutoff 15 Aug) ke liye "cutoff tak kabhi nahi biki" bilkul Phase 01 ka **dead series** rule hai. Backtests mein yeh un series ko bhi pakarta hai jo us waqt **launch hi nahi hui thin** (misaal: BOOKS, jo Oct 2016 mein launch hui). *Pichle version mein backtests mein sirf dead-series rule tha. Is ki wajah se May 2016 mein BOOKS ke liye ghalat forecast bane (Section 10e).*
# - **Metrics:** **WAPE** (business: kul units ki ghalti %) aur **RMSLE** (Kaggle ka metric; har row barabar, chhoti series bhi aham). Sirf khule stores ki rows par.
# - **4 windows** (Phase 03 wale):

# | Window | Train ka aakhri din | Eval (16 din) | Khaasiyat |
# |---|---|---|---|
# | Validation | 30 Jul 2017 | 31 Jul – 15 Aug 2017 | Asli mock exam, 1 national holiday |
# | May 2016 | 17 May 2016 | 18 May – 2 Jun 2016 | Zalzale ke baad, 1 national holiday |
# | Dec 2016 | 2 Dec 2016 | 3 – 18 Dec 2016 | Christmas se pehle |
# | Jun 2017 | 14 Jun 2017 | 15 – 30 Jun 2017 | Aam window |

# - **Baseline:** pichle 28 maloom dinon ki average (`roll28_lag16`), taa ke pata chale ke model waqai kuch seekh raha hai.


WINDOWS = {"Validation (Aug 2017)": ("2017-07-30", "2017-07-31"),
           "May 2016": ("2016-05-17", "2016-05-18"),
           "Dec 2016": ("2016-12-02", "2016-12-03"),
           "Jun 2017": ("2017-06-14", "2017-06-15")}
PARAMS = dict(n_estimators=400, learning_rate=0.07, num_leaves=127, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)

Y = np.log1p(master["sales_clean"].values)
TRAINABLE = (master["is_trainable"] == 1).values
def zero_rule(cutoff):
    # Rule (feature nahi): series jo CUTOFF tak kabhi nahi biki, pre-opening, aur closure-calendar din -> 0.
    # "Cutoff tak kabhi nahi biki" sirf cutoff tak ki info use karta hai (leakage-free). Final forecast (cutoff 15 Aug)
    # ke liye yeh bilkul Phase 01 ka dead-series rule hai; backtests mein yeh un series ko bhi pakarta hai jo
    # us waqt launch hi nahi hui thin (misaal: BOOKS, Oct 2016 mein launch).
    fsd = master["first_sale_date"]
    never_sold = fsd.isna() | (fsd > pd.Timestamp(cutoff))
    return (never_sold | (master["is_pre_opening"] == 1) | (master["is_closure_calendar_day"] == 1)).values
OPEN = ((master["is_pre_opening"] == 0) & (master["is_store_closed"] == 0)).values
SALES = master["sales_clean"].values
DATES = master["date"].values

def wape(y, p):  return np.abs(y - p).sum() / y.sum() * 100
def rmsle(y, p): return np.sqrt(np.mean((np.log1p(np.clip(p, 0, None)) - np.log1p(y)) ** 2))

def window_masks(te, es):
    te, es = np.datetime64(pd.Timestamp(te)), np.datetime64(pd.Timestamp(es))
    trm = TRAINABLE & (DATES >= np.datetime64(CLEAN_START)) & (DATES <= te)
    evm = (DATES >= es) & (DATES <= es + np.timedelta64(15, "D"))
    return trm, evm

MASKS = {w: window_masks(*v) for w, v in WINDOWS.items()}
ZR = {w: zero_rule(v[0]) for w, v in WINDOWS.items()}
ZR["Aug 2016"] = zero_rule("2016-07-30")
ZR_PHASE01 = ((master["is_dead_series"] == 1) | (master["is_pre_opening"] == 1) | (master["is_closure_calendar_day"] == 1)).values
for w, (trm, evm) in MASKS.items():
    extra = int((ZR[w] & ~ZR_PHASE01 & evm).sum())
    print(f"{w}: eval rows jo ab rule se 0 hongi lekin Phase 01 rule se nahi = {extra:,}")

def evaluate(features, seed=0, extra=None, FF=None, masks=None, keep_models=None):
    # Har window par train + eval. extra: placebo column. FF: doosra feature table (7b). masks: doosri windows (10c).
    # keep_models: dict diya jaye to har window ka model aur eval data us mein rakh diya jata hai (10b/10d).
    res = {}
    X = (F if FF is None else FF)[features].copy()
    if extra is not None:
        X[extra[0]] = extra[1]
    for w, (trm, evm) in (MASKS if masks is None else masks).items():
        mdl = lgb.LGBMRegressor(**PARAMS, random_state=seed)
        mdl.fit(X[trm], Y[trm])
        p_raw = np.expm1(mdl.predict(X[evm])).clip(0)
        p = np.where(ZR[w][evm], 0, p_raw)
        o = OPEN[evm]
        res[w] = {"WAPE": wape(SALES[evm][o], p[o]), "RMSLE": rmsle(SALES[evm][o], p[o])}
        if keep_models is not None:
            keep_models[w] = (mdl, X[evm], p, p_raw)
    return res

base_line = {}
for w, (trm, evm) in MASKS.items():
    o = OPEN[evm]
    p = np.where(ZR[w][evm], 0, np.expm1(F.loc[evm, "roll28_lag16"].fillna(0).values))
    base_line[w] = {"WAPE": wape(SALES[evm][o], p[o]), "RMSLE": rmsle(SALES[evm][o], p[o])}
print("Baseline (pichle 28 maloom dinon ki average):")
display(pd.DataFrame(base_line).T.round(4))


# ----------------------------------------------------------------------

# ### 6b. Test: kya scaling tree model ke nateeje badalti hai?
# **Theory:** tree model sirf values ki **tarteeb** dekhta hai. Is liye scaling se splits, aur predictions, **nahi** badalni chahiyen. Isay maanne ki bajaye **test** karte hain: ek hi model teen versions par, wahi seed:
# 1. unscaled
# 2. standardized (mean 0, std 1)
# 3. sirf divide (x / std), jahan 0 apni jagah rehta hai

# Scaler ka mean/std sirf training se liya gaya hai. Waqt bachane ke liye training ka ek random hissa (300,000 rows) aur 100 trees liye gaye hain.


trm, evm = MASKS["Validation (Aug 2017)"]
idx = np.flatnonzero(trm)
idx = np.sort(np.random.default_rng(0).choice(idx, 300_000, replace=False))
num_cols = [c for c in KEEP if c not in CAT_COLS]
Xa_tr, Xa_ev = F.loc[idx, KEEP], F.loc[evm, KEEP]
mu, sd = Xa_tr[num_cols].mean(), Xa_tr[num_cols].std().replace(0, 1)       # sirf TRAINING se fit
Xb_tr, Xb_ev = Xa_tr.copy(), Xa_ev.copy()
Xb_tr[num_cols] = (Xa_tr[num_cols] - mu) / sd
Xb_ev[num_cols] = (Xa_ev[num_cols] - mu) / sd
small = dict(PARAMS, n_estimators=100)
Xc_tr, Xc_ev = Xa_tr.copy(), Xa_ev.copy()                               # sirf divide (0 apni jagah rehta hai)
Xc_tr[num_cols] = Xa_tr[num_cols] / sd
Xc_ev[num_cols] = Xa_ev[num_cols] / sd
o = OPEN[evm]
rows = []
for name, tr_, ev_ in [("unscaled", Xa_tr, Xa_ev), ("standardized (mean 0, std 1)", Xb_tr, Xb_ev), ("sirf divide (x / std)", Xc_tr, Xc_ev)]:
    pr = lgb.LGBMRegressor(**small, random_state=0).fit(tr_, Y[idx]).predict(ev_)
    pp = np.where(ZR["Validation (Aug 2017)"][evm], 0, np.expm1(pr).clip(0))
    rows.append({"version": name, "WAPE": wape(SALES[evm][o], pp[o]), "RMSLE": rmsle(SALES[evm][o], pp[o]),
                 "max |farq| predictions (log) vs unscaled": np.nan if name == "unscaled" else np.abs(pr - pa).max()})
    if name == "unscaled":
        pa = pr
display(pd.DataFrame(rows).set_index("version").round(4))


# -------------------------------------------------------------------

## 7. Base model ("Keep" features) aur noise ka andaza
# **Do qisam ka noise naapa jata hai:**
# 1. **Seed noise:** wahi features, sirf random seed alag (0 vs 1).
# 2. **Placebo:** base mein ek **bilkul random** column jor do, jis mein koi information nahi. Is se model ke faisle thore badalte hain, aur score bhi thora badalta hai. Yeh batata hai ke **kisi bhi nayi column** ko jorne se, bina information ke, kitna farq aa jata hai.

# **Noise = har window mein in dono farqon ka bara, phir 4 windows ki average.** Koi feature tabhi rakha jayega jab us ka faida is noise se **bara** ho.


CACHE = REPORT_DIR / "05_ablation_results_v2.csv"      # v2: cutoff-based zero rule (purane results is rule se nahi)
cached = pd.read_csv(CACHE) if (CACHE.exists() and not FORCE_RERUN) else None

def run(name, features, seed=0, extra=None, FF=None, masks=None):
    if cached is not None and name in set(cached["variant"]):
        r = cached[cached["variant"] == name]
        return {row["window"]: {"WAPE": row["WAPE"], "RMSLE": row["RMSLE"]} for _, row in r.iterrows()}
    t0 = time.time()
    out = evaluate(features, seed, extra, FF, masks)
    print(f"  {name:<32} {time.time() - t0:5.0f}s")
    return out

results = {}
rng = np.random.default_rng(42)
placebo = rng.normal(size=len(F))
results["BASE (seed 0)"] = run("BASE (seed 0)", KEEP, 0)
results["BASE (seed 1)"] = run("BASE (seed 1)", KEEP, 1)
results["PLACEBO (random column)"] = run("PLACEBO (random column)", KEEP, 0, ("placebo", placebo))

def table(res):
    return pd.concat({k: pd.DataFrame(v).T for k, v in res.items()}, names=["variant", "window"])

tb = table(results)
base = tb.loc["BASE (seed 0)"]
noise_w = pd.concat([(tb.loc["BASE (seed 1)"]["WAPE"] - base["WAPE"]).abs(),
                     (tb.loc["PLACEBO (random column)"]["WAPE"] - base["WAPE"]).abs()], axis=1).max(axis=1)
noise_r = pd.concat([(tb.loc["BASE (seed 1)"]["RMSLE"] - base["RMSLE"]).abs(),
                     (tb.loc["PLACEBO (random column)"]["RMSLE"] - base["RMSLE"]).abs()], axis=1).max(axis=1)
NOISE_W, NOISE_R = noise_w.mean(), noise_r.mean()
print(f"Noise (4 windows ki average): WAPE {NOISE_W:.3f} points | RMSLE {NOISE_R:.4f}")
display(tb.unstack("window").round(4))
cmp = pd.DataFrame({"Baseline WAPE": pd.DataFrame(base_line).T["WAPE"], "Base model WAPE": base["WAPE"],
                    "Baseline RMSLE": pd.DataFrame(base_line).T["RMSLE"], "Base model RMSLE": base["RMSLE"]})
display(cmp.round(4))


# ----------------------------------------------------------------------

### 7b. Lag source ka rule: model par test
# Phase 04 ne faisla kiya tha ke lag ka source din agar band/unavailable ho (ya 4 kam-likhi families ka partial din), to value **NaN** rakhi jaye, 0 nahi. Woh faisla **correlation** par tha. Yahan wahi faisla **model** par test hota hai.

# Yeh "feature jorna ya na jorna" nahi, balkay **do tareeqon mein se ek chunna** hai. Is liye faisle ka usool (results se pehle tay):
# - Jo tareeqa **zyada window × metric muqablon** (4 windows × 2 metrics = 8) mein behtar ho, woh jeetega.
# - Barabar ho to **WAPE** (business metric: kul units ki ghalti) faisla karega.


F_unmasked = build_features(master, mask_sources=False)
results["BASE, lag source unmasked (0)"] = run("BASE, lag source unmasked (0)", KEEP, 0, FF=F_unmasked)
tb = table(results)
um = tb.loc["BASE, lag source unmasked (0)"]
src_cmp = pd.DataFrame({"WAPE masked (NaN)": base["WAPE"], "WAPE unmasked (0)": um["WAPE"],
                        "RMSLE masked (NaN)": base["RMSLE"], "RMSLE unmasked (0)": um["RMSLE"]})
wins_masked = int((base["WAPE"] < um["WAPE"]).sum() + (base["RMSLE"] < um["RMSLE"]).sum())
print(f"Masked (NaN) jeeta: {wins_masked}/8 muqable | WAPE average farq (masked - unmasked): "
      f"{(base['WAPE'] - um['WAPE']).mean():+.3f} | RMSLE average farq: {(base['RMSLE'] - um['RMSLE']).mean():+.4f}")
if wins_masked < 4 or (wins_masked == 4 and (base["WAPE"] - um["WAPE"]).mean() > 0):
    F = F_unmasked
    print("Faisla: UNMASKED (0) -> aage ke saare tests is se.")
else:
    print("Faisla: MASKED (NaN) rehta hai (Phase 04 ka faisla model par bhi sahi).")
del F_unmasked
display(src_cmp.round(4))

# ---------------------------------------------------------------------------

## 8. Ablation: har group alag
# Har group ko **base ke upar** jora jata hai (base + group), aur 4 windows par score kiya jata hai. Farq = (base + group) − base. **Manfi farq = behtar.**

# **Pehle: kya group test bhi ho sakta hai?** Agar koi binary feature kisi bhi eval window **aur** test window mein ek din bhi active nahi, to us ka asar **naapa hi nahi ja sakta**, aur woh is forecast par asar bhi nahi daal sakta. Aise group ko train karna sirf waqt zaya karna hai. Yeh faisla data se hota hai (neeche ki table).

def is_binary(c):
    s = F.loc[train_rows, c].dropna()
    return not isinstance(s.dtype, pd.CategoricalDtype) and set(s.unique()) <= {0, 1}

act = {}
for gname, cols in ABLATION_GROUPS.items():
    if all(is_binary(c) for c in cols):
        row = {w: int(F.loc[evm, cols].max(axis=1).sum()) for w, (trm, evm) in MASKS.items()}
        row["test window"] = int(F.loc[test_rows, cols].max(axis=1).sum())
        act[gname] = row
act = pd.DataFrame(act).T
NOT_TESTABLE = act.index[(act.sum(axis=1) == 0)].tolist()
print("Binary groups: kitni rows par active (eval windows + test window)")
display(act)
print("Test nahi ho sakte (har window aur test mein 0 rows):", NOT_TESTABLE if NOT_TESTABLE else "koi nahi")

# ------------------------------------------------------------------------------

t_all = time.time()
for gname, cols in ABLATION_GROUPS.items():
    if gname in NOT_TESTABLE:
        continue
    results[f"+ {gname}"] = run(f"+ {gname}", KEEP + cols, 0)
print(f"Ablation total: {time.time() - t_all:.0f}s")
tb = table(results)
tb.reset_index().to_csv(CACHE, index=False)

# --------------------------------------------------------------------

## 9. Faisla table
# **Faisle ka usool (pehle se tay, results dekhne se pehle):** group **rakha jayega** agar dono shartein poori hon:
# 1. **Ek metric par saaf faida:** average WAPE ka faida noise se bara ho (`ΔWAPE < −noise_W`), **ya** average RMSLE ka faida noise se bara ho (`ΔRMSLE < −noise_R`).
# 2. **Doosre metric par saaf nuqsaan nahi:** doosre metric ka average farq noise se zyada kharab na ho.

# Yani ek metric saaf behtar ho, aur doosra saaf kharab na ho. Warna group **nahi** rakha jayega. Is ki wajah Occam ka usool hai: bina saboot ke feature model ko pecheeda banata hai.

rows = []
for gname, cols in ABLATION_GROUPS.items():
    if gname in NOT_TESTABLE:
        rows.append({"group": gname, "columns": ", ".join(cols), "avg ΔWAPE": np.nan, "avg ΔRMSLE": np.nan,
                     "windows behtar (WAPE)": "-", "faisla": "Nahi (test window mein active nahi, naapa nahi ja sakta)"})
        continue
    r = tb.loc[f"+ {gname}"]
    dW, dR = (r["WAPE"] - base["WAPE"]), (r["RMSLE"] - base["RMSLE"])
    gain_w, gain_r = dW.mean() < -NOISE_W, dR.mean() < -NOISE_R
    hurt_w, hurt_r = dW.mean() > NOISE_W, dR.mean() > NOISE_R
    keep = (gain_w and not hurt_r) or (gain_r and not hurt_w)
    rows.append({"group": gname, "columns": ", ".join(cols), "avg ΔWAPE": dW.mean(), "avg ΔRMSLE": dR.mean(),
                 "windows behtar (WAPE)": f"{(dW < 0).sum()}/4",
                 "faisla": "PASS (tasdeeq Section 10 mein)" if keep else ("Nahi (nuqsaan)" if (hurt_w or hurt_r) else "Nahi (noise se bara faida nahi)")})
decision = pd.DataFrame(rows).set_index("group").sort_values("avg ΔWAPE")
SELECTED = [c for gname, cols in ABLATION_GROUPS.items() if decision.loc[gname, "faisla"].startswith("PASS") for c in cols]
print("Har window par WAPE ka farq (variant - BASE seed 0); PLACEBO aur BASE (seed 1) noise ke liye:")
dw = (tb["WAPE"].unstack("window") - base["WAPE"]).drop(index=["BASE (seed 0)"], errors="ignore")
display(dw[[c for c in WINDOWS]].round(3))
print(f"Noise: WAPE {NOISE_W:.3f} | RMSLE {NOISE_R:.4f}")
print("Section 9 pass groups:", [g for g in decision.index if decision.loc[g, "faisla"].startswith("PASS")])
display(decision.round(4))


# -------------------------------------------------------------

d = decision.dropna(subset=["avg ΔWAPE"]).sort_values("avg ΔWAPE")
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
for ax, col, nz, title in [(axes[0], "avg ΔWAPE", NOISE_W, "WAPE ka farq (points)"),
                           (axes[1], "avg ΔRMSLE", NOISE_R, "RMSLE ka farq")]:
    colors = [BLUE if f.startswith("PASS") else (RED if "nuqsaan" in f else LIGHT_GRAY) for f in d["faisla"]]
    ax.barh(d.index, d[col], color=colors)
    ax.axvspan(-nz, nz, color=GRAY, alpha=0.15, label="noise ki hadd")
    ax.axvline(0, color=GRAY, linewidth=1)
    ax.set_title(title)
    ax.legend(loc="lower right")
axes[0].tick_params(axis="y", labelsize=8)
fig.suptitle("Har group ko base mein jorne ka asar (manfi = behtar; neela = Section 9 pass)", x=0.01, ha="left",
             fontweight="bold", fontsize=12)
save(fig, "01_ablation_decision")

# ------------------------------------------------------------------


## 10. Final feature set ki tasdeeq
# Akele akele faida dene wale groups **ek sath** bhi faida dein, yeh zaroori nahi. Ek seed par mila faida qismat bhi ho sakta hai. Is liye final set (Keep + rakhe gaye groups) ko **dono seeds** par base se compare kiya jata hai, aur **Section 9 wala hi usool** lagta hai: dono seeds ki average par faida noise se bara ho.


FINAL_FEATURES = KEEP + SELECTED
CONFIRMED = None
if SELECTED:
    results["FINAL (seed 0)"] = run("FINAL (seed 0)", FINAL_FEATURES, 0)
    results["FINAL (seed 1)"] = run("FINAL (seed 1)", FINAL_FEATURES, 1)
    tb = table(results)
    tb.reset_index().to_csv(CACHE, index=False)
    f0, f1 = tb.loc["FINAL (seed 0)"], tb.loc["FINAL (seed 1)"]
    b1 = tb.loc["BASE (seed 1)"]
    conf = pd.DataFrame({"ΔWAPE seed 0": f0["WAPE"] - base["WAPE"], "ΔWAPE seed 1": f1["WAPE"] - b1["WAPE"],
                         "ΔRMSLE seed 0": f0["RMSLE"] - base["RMSLE"], "ΔRMSLE seed 1": f1["RMSLE"] - b1["RMSLE"]})
    conf.loc["average"] = conf.mean()
    print(f"Final features: {len(FINAL_FEATURES)} (Keep {len(KEEP)} + {len(SELECTED)})")
    display(conf.round(4))
    avg_dW = conf.loc["average", ["ΔWAPE seed 0", "ΔWAPE seed 1"]].mean()
    avg_dR = conf.loc["average", ["ΔRMSLE seed 0", "ΔRMSLE seed 1"]].mean()
    # Section 9 jaisa hi usool: dono seeds ki average par ek metric ka faida noise se bara, doosre ka nuqsaan noise se zyada nahi
    CONFIRMED = (avg_dW < -NOISE_W and avg_dR <= NOISE_R) or (avg_dR < -NOISE_R and avg_dW <= NOISE_W)
    print(f"Dono seeds ki average: ΔWAPE {avg_dW:+.3f} | ΔRMSLE {avg_dR:+.4f} -> tasdeeq: {CONFIRMED}")
    if not CONFIRMED:
        FINAL_FEATURES = KEEP
        print("Final set base se behtar sabit nahi hua -> sirf KEEP features rakhe gaye.")
else:
    print("Koi ablation group noise se bara faida nahi de saka -> final set = KEEP features.")
print(f"\nFINAL FEATURES ({len(FINAL_FEATURES)}):", FINAL_FEATURES)

# -------------------------------------------------------------


# ## 10b. Backward check: kya "Keep" features mein koi faltu hai?
# Section 8 sirf **forward** tha (base mein naya feature jorna). Mukammal feature selection ke liye **backward** check bhi zaroori hai: kya 18 Keep features mein se koi **hataya** ja sakta hai? Do qadam:

# **Qadam 1: out-of-time permutation importance.** Har window ke base model par ek feature-group ki values ko eval rows ke andar **shuffle** kar do (baqi sab waise hi). Agar WAPE bohat barhe, to model us group par bharosa karta hai. Yeh **eval window** par hota hai (jo model ne nahi dekhi), is liye seedha "forecast par asar" batata hai. Aapas mein jure features (misaal: `store_type`, `cluster`, `city`) ek **group** mein shuffle hote hain, warna ek doosre ki kami poori kar dete hain.

# **Qadam 2: drop-one retrain.** Jin groups ki permutation importance noise se **kam** ho, unhein hata kar model **dobara train** hota hai. Faisle ka usool (results se pehle tay):
# - Agar hatane se nuqsaan noise se zyada **nahi** (`ΔWAPE ≤ noise` aur `ΔRMSLE ≤ noise`), to group **hata** do (sada model behtar).
# - Phir doosre seed par **tasdeeq**.
# - **Jo binary group kisi eval window mein active hi nahi** (misaal: holiday-season), us ka asar naapa nahi ja sakta. Woh **Phase 04 ke saboot** par rakha jata hai. Section 8 ka usool bhi yahi tha: jo naapa na ja sake, us ka faisla test se nahi hota.


PERM_GROUPS = {
    "store_cat": ["store_cat"], "family_cat": ["family_cat"],
    "store_type + cluster + city": ["store_type", "store_cluster", "city"],
    "dow": ["dow"], "dom": ["dom"],
    "is_national_holiday": ["is_national_holiday"], "is_holiday_season_day": ["is_holiday_season_day"],
    "national_event + earthquake": ["is_national_event", "is_earthquake_period"],
    "onpromotion": ["onpromotion"],
    "same_wd_mean4": ["same_wd_mean4"], "roll28_lag16": ["roll28_lag16"], "roll7_lag16": ["roll7_lag16"],
    "lag_364": ["lag_364"], "lag_16 + lag_21": ["lag_16", "lag_21"],
}
assert sorted(c for v in PERM_GROUPS.values() for c in v) == sorted(KEEP)
t0 = time.time()
BASE_MODELS = {}
_ = evaluate(KEEP, 0, keep_models=BASE_MODELS)            # base (seed 0) dobara, models rakhne ke liye
print(f"Base models dobara train ({time.time() - t0:.0f}s)")

rng_p = np.random.default_rng(7)
prow = []
for gname, cols in PERM_GROUPS.items():
    row = {"group": gname}
    for w, (mdl, Xev, p0, _) in BASE_MODELS.items():
        evm = MASKS[w][1]
        o = OPEN[evm]
        Xp = Xev.copy()
        perm = rng_p.permutation(len(Xp))
        for c in cols:
            Xp[c] = Xev[c].values[perm]
        p = np.where(ZR[w][evm], 0, np.expm1(mdl.predict(Xp)).clip(0))
        row[w] = wape(SALES[evm][o], p[o]) - wape(SALES[evm][o], p0[o])
        binary = all(set(F.loc[train_rows, c].dropna().unique()) <= {0, 1} for c in cols)
        row.setdefault("active rows (eval)", 0)
        row["active rows (eval)"] += int(Xev[cols].max(axis=1).sum()) if binary else len(Xev)
    prow.append(row)
perm_tab = pd.DataFrame(prow).set_index("group")
perm_tab["avg ΔWAPE (shuffle)"] = perm_tab[list(WINDOWS)].mean(axis=1)
perm_tab = perm_tab.sort_values("avg ΔWAPE (shuffle)", ascending=False)
display(perm_tab.round(3))

fig, ax = plt.subplots(figsize=(9.5, 5))
pt = perm_tab.sort_values("avg ΔWAPE (shuffle)")
ax.barh(pt.index, pt["avg ΔWAPE (shuffle)"], color=[BLUE if v > NOISE_W else LIGHT_GRAY for v in pt["avg ΔWAPE (shuffle)"]])
ax.axvline(NOISE_W, color=RED, linestyle="--", linewidth=1.2, label=f"noise ({NOISE_W:.2f})")
ax.set_xscale("symlog", linthresh=0.5)
ax.set_xlabel("shuffle karne par WAPE kitna barha (points, 4 windows ki average; symlog scale)")
ax.set_title("Out-of-time permutation importance (base model)")
ax.legend(loc="lower right")
save(fig, "03_permutation_importance")

# ----------------------------------------------------------------


weak = perm_tab.index[(perm_tab["avg ΔWAPE (shuffle)"] < NOISE_W)].tolist()
untestable = perm_tab.index[perm_tab["active rows (eval)"] == 0].tolist()
to_test = [g for g in weak if g not in untestable]
print("Permutation importance noise se kam:", weak)
print("In mein se eval windows mein active hi nahi (Phase 04 saboot par rakhe jayenge):", untestable)
print("Drop-one retrain hoga:", to_test)
drows, DROP_GROUPS = [], []
for g in to_test:
    feats = [c for c in KEEP if c not in PERM_GROUPS[g]]
    results[f"- {g}"] = run(f"- {g}", feats, 0)
    r = table(results).loc[f"- {g}"]
    dW, dR = (r["WAPE"] - base["WAPE"]).mean(), (r["RMSLE"] - base["RMSLE"]).mean()
    drop = dW <= NOISE_W and dR <= NOISE_R
    drows.append({"group hataya": g, "avg ΔWAPE": dW, "avg ΔRMSLE": dR, "pehla faisla": "hata do?" if drop else "rakho"})
    if drop:
        DROP_GROUPS.append(g)
tb = table(results)
tb.reset_index().to_csv(CACHE, index=False)
drop_tab = pd.DataFrame(drows).set_index("group hataya") if drows else pd.DataFrame()
display(drop_tab.round(4))


# --------------------------------------------------------------------


# Tasdeeq: saare "hata do?" groups ek sath hata kar, DONO seeds par
REDUCED = [c for c in KEEP if not any(c in PERM_GROUPS[g] for g in DROP_GROUPS)]
if DROP_GROUPS:
    results["REDUCED (seed 0)"] = run("REDUCED (seed 0)", REDUCED, 0)
    results["REDUCED (seed 1)"] = run("REDUCED (seed 1)", REDUCED, 1)
    tb = table(results)
    tb.reset_index().to_csv(CACHE, index=False)
    b1 = tb.loc["BASE (seed 1)"]
    dW = np.mean([(tb.loc["REDUCED (seed 0)"]["WAPE"] - base["WAPE"]).mean(), (tb.loc["REDUCED (seed 1)"]["WAPE"] - b1["WAPE"]).mean()])
    dR = np.mean([(tb.loc["REDUCED (seed 0)"]["RMSLE"] - base["RMSLE"]).mean(), (tb.loc["REDUCED (seed 1)"]["RMSLE"] - b1["RMSLE"]).mean()])
    REDUCE_OK = dW <= NOISE_W and dR <= NOISE_R
    print(f"Hatane wale groups: {DROP_GROUPS} | dono seeds ki average: ΔWAPE {dW:+.3f}, ΔRMSLE {dR:+.4f} -> tasdeeq: {REDUCE_OK}")
else:
    REDUCE_OK = False
    print("Koi group hatane layak nahi nikla.")
if REDUCE_OK:
    FINAL_FEATURES = REDUCED
print(f"Backward check ke baad features: {len(FINAL_FEATURES)}")


# -----------------------------------------------------------

## 10c. `month` ka khaas test: August window
# Section 8 mein `month` ne validation (Aug 2017) par bara faida diya (−1.38), lekin Dec 2016 par nuqsaan (+0.53). Hamara forecast **August** ka hai, is liye yeh sawal khula nahi chhora ja sakta.

# **Test:** ek aur **August window** (train 30 Jul 2016 tak, eval 31 Jul – 15 Aug 2016), dono seeds, `month` ke sath aur baghair. Is window ki training mein August ki **sirf ek** misaal hai (2015), bilkul Dec 2016 ki tarah. Yani yeh ek **mushkil** imtihaan hai.

# **Faisle ka usool (results se pehle tay):**
# - `month` final set mein **tabhi** aayega jab Aug 2016 mein bhi dono seeds ki average par **WAPE ka faida noise se bara** ho, aur RMSLE ka nuqsaan noise se zyada na ho. Tab **do alag August windows** ek hi baat kahengi.
# - Warna `month` bahar rahega.


AUG16 = {"Aug 2016": window_masks("2016-07-30", "2016-07-31")}
base_feats = list(FINAL_FEATURES)
r = {}
for name, feats, seed in [("AUG16 base (seed 0)", base_feats, 0), ("AUG16 base (seed 1)", base_feats, 1),
                          ("AUG16 + month (seed 0)", base_feats + ["month"], 0), ("AUG16 + month (seed 1)", base_feats + ["month"], 1)]:
    results[name] = run(name, feats, seed, masks=AUG16)
tb = table(results)
tb.reset_index().to_csv(CACHE, index=False)
a = {k: tb.loc[k].iloc[0] for k in ["AUG16 base (seed 0)", "AUG16 base (seed 1)", "AUG16 + month (seed 0)", "AUG16 + month (seed 1)"]}
dW = np.mean([a["AUG16 + month (seed 0)"]["WAPE"] - a["AUG16 base (seed 0)"]["WAPE"],
              a["AUG16 + month (seed 1)"]["WAPE"] - a["AUG16 base (seed 1)"]["WAPE"]])
dR = np.mean([a["AUG16 + month (seed 0)"]["RMSLE"] - a["AUG16 base (seed 0)"]["RMSLE"],
              a["AUG16 + month (seed 1)"]["RMSLE"] - a["AUG16 base (seed 1)"]["RMSLE"]])
seed_gap = abs(a["AUG16 base (seed 0)"]["WAPE"] - a["AUG16 base (seed 1)"]["WAPE"])
MONTH_OK = dW < -NOISE_W and dR <= NOISE_R
print(f"Aug 2016: month ka asar (dono seeds ki average) ΔWAPE {dW:+.3f}, ΔRMSLE {dR:+.4f} | "
      f"is window mein seed farq {seed_gap:.3f} | noise {NOISE_W:.3f} -> month rakho: {MONTH_OK}")
if MONTH_OK:
    FINAL_FEATURES = FINAL_FEATURES + ["month"]
display(pd.DataFrame(a).T.round(4))
print(f"Final features ab: {len(FINAL_FEATURES)}")

# -----------------------------------------------------------

## 10d. Final feature set ka report card (faisla nahi, sirf report)
# Section 10b ne kuch features **hataye** aur 10c ne `month` **jora**. Yeh dono faisle alag alag test hue thay. Ab **final set ek sath** base (18 Keep) se compare hota hai, 5 windows (4 + Aug 2016) aur dono seeds par. Yeh report card hai, naya faisla nahi, taa ke saaf dikhe ke final set kahan behtar aur kahan bura hai.

ALL5 = dict(MASKS, **AUG16)
for seed in (0, 1):
    results[f"FINAL v2 (seed {seed})"] = run(f"FINAL v2 (seed {seed})", FINAL_FEATURES, seed, masks=ALL5)
tb = table(results)
tb.reset_index().to_csv(CACHE, index=False)
rc = []
for seed in (0, 1):
    fv = tb.loc[f"FINAL v2 (seed {seed})"]
    for w in ALL5:
        b = tb.loc[f"BASE (seed {seed})"].loc[w] if w in WINDOWS else tb.loc[f"AUG16 base (seed {seed})"].loc[w]
        rc.append({"seed": seed, "window": w, "ΔWAPE (final - base)": fv.loc[w, "WAPE"] - b["WAPE"],
                   "ΔRMSLE (final - base)": fv.loc[w, "RMSLE"] - b["RMSLE"], "final WAPE": fv.loc[w, "WAPE"]})
rc = pd.DataFrame(rc)
card = rc.groupby("window")[["ΔWAPE (final - base)", "ΔRMSLE (final - base)", "final WAPE"]].mean().reindex(list(ALL5))
card.loc["average (5 windows)"] = card.mean()
print(f"Final set ({len(FINAL_FEATURES)} features) vs base (18 Keep), dono seeds ki average | noise WAPE {NOISE_W:.3f}")
display(card.round(4))


# ----------------------------------------------------------------

# ## 10e. May 2016: model RMSLE par baseline se bura kyun?
# Pichle version mein May 2016 window ka base model WAPE par baseline se bohat behtar tha, lekin **RMSLE par bura** (0.537 vs 0.515). Andaza lagane ki bajaye ghalti ko **hisson mein** bant kar dekha gaya. RMSLE `(log farq)²` ki average hai, is liye har row ka `(log farq)²` jor kar dekha ja sakta hai ke ghalti **kahan se** aati hai.

# **Khoj:** sab se bari wajah **BOOKS** thi. BOOKS Oct 2016 mein launch hui, yani May 2016 mein abhi **launch hi nahi hui thi**. Pichla rule sirf "poori history mein kabhi nahi bika" (dead series) ko 0 deta tha, aur BOOKS dead nahi thi. Is liye model ne us ke liye sales predict kar di, jab ke asal mein 0 thi. RMSLE har row ko barabar ginta hai, is liye in chhoti ghaltiyon ne RMSLE ko bohat bigaara. Isi khoj se Section 6 ka rule badla. Neeche dono rules ka muqabla hai.


mdl, Xev, p_model, p_raw = BASE_MODELS["May 2016"]
evm = MASKS["May 2016"][1]
o = OPEN[evm]
y = SALES[evm][o]
p_old = np.where(ZR_PHASE01[evm], 0, p_raw)[o]                      # pichla rule (sirf dead series)
pm = p_model[o]                                                      # naya rule (cutoff tak kabhi nahi biki)
pb = np.where(ZR["May 2016"][evm], 0, np.expm1(F.loc[evm, "roll28_lag16"].fillna(0).values))[o]
print(f"May 2016 RMSLE: model + pichla rule {rmsle(y, p_old):.4f} | model + naya rule {rmsle(y, pm):.4f} | "
      f"baseline {rmsle(y, pb):.4f}")
fam_old = pd.DataFrame({"family": master.loc[evm, "family"].values[o],
                        "se_old_rule": (np.log1p(p_old) - np.log1p(y)) ** 2, "se_new_rule": (np.log1p(pm) - np.log1p(y)) ** 2})
print("Pichle rule ke sath sab se zyada ghalti wali families (squared log error ka jor):")
display(fam_old.groupby("family").sum().sort_values("se_old_rule", ascending=False).head(4).round(1))
d = pd.DataFrame({"family": master.loc[evm, "family"].values[o], "store_nbr": master.loc[evm, "store_nbr"].values[o],
                  "date": master.loc[evm, "date"].values[o], "y": y,
                  "se_model": (np.log1p(pm) - np.log1p(y)) ** 2, "se_base": (np.log1p(pb) - np.log1p(y)) ** 2})
d["actual zero"] = d["y"] == 0
seg = d.groupby("actual zero").agg(rows=("y", "size"), model_sq_err_sum=("se_model", "sum"), base_sq_err_sum=("se_base", "sum"))
seg["model share %"] = seg["model_sq_err_sum"] / d["se_model"].sum() * 100
print(f"RMSLE model {np.sqrt(d['se_model'].mean()):.4f} | baseline {np.sqrt(d['se_base'].mean()):.4f}")
display(seg.round(2))
fam = d.groupby("family")[["se_model", "se_base"]].sum()
fam["model - baseline"] = fam["se_model"] - fam["se_base"]
print("Families jahan model, baseline se sab se zyada bura (squared log error ka jor):")
display(fam.sort_values("model - baseline", ascending=False).head(6).round(1))
st = d.groupby("store_nbr")[["se_model", "se_base"]].sum()
st["model - baseline"] = st["se_model"] - st["se_base"]
print("Stores jahan model, baseline se sab se zyada bura:")
display(st.sort_values("model - baseline", ascending=False).head(6).round(1))


# ------------------------------------------------------------------------

## 11. Model kin features par chalta hai? (ehtiyat ke sath)
# Phase 01b ka sabaq: **"gain" importance gumraah kar sakti hai.** Yeh batati hai ke feature kitni baar split mein use hua, yeh nahi ke us ke baghair kitna nuqsaan hoga. Asal saboot Section 8 ka ablation hai. Yeh chart sirf **samajhne** ke liye hai (validation window ka final model).

trm, evm = MASKS["Validation (Aug 2017)"]
mdl = lgb.LGBMRegressor(**PARAMS, random_state=0).fit(F.loc[trm, FINAL_FEATURES], Y[trm])
print(f"Final features ({len(FINAL_FEATURES)}):", FINAL_FEATURES)
imp = pd.Series(mdl.booster_.feature_importance("gain"), index=FINAL_FEATURES)
imp = (imp / imp.sum() * 100).sort_values()
fig, ax = plt.subplots(figsize=(9, 0.32 * len(imp) + 1.2))
ax.barh(imp.index, imp.values, color=BLUE)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
ax.set_title("Final model: gain importance (sirf samajhne ke liye)")
ax.tick_params(axis="y", labelsize=8)
save(fig, "02_final_gain_importance")
display(imp.sort_values(ascending=False).round(2).head(10).to_frame("% of gain"))


# ----------------------------------------------------------------------

# SHAP (TreeSHAP, LightGBM ka apna): har feature ne har prediction ko kitna upar/neeche kiya
Xs = F.loc[evm, FINAL_FEATURES].sample(20_000, random_state=0)
contrib = mdl.predict(Xs, pred_contrib=True)[:, :-1]          # aakhri column = base value
shap_mean = pd.Series(np.abs(contrib).mean(axis=0), index=FINAL_FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(9, 0.32 * len(shap_mean) + 1.2))
ax.barh(shap_mean.index, shap_mean.values, color=AQUA)
ax.set_xlabel("mean |SHAP| (log1p sales par asar)")
ax.set_title("Final model: SHAP (validation window, 20,000 rows)")
ax.tick_params(axis="y", labelsize=8)
save(fig, "04_final_shap")
display(shap_mean.sort_values(ascending=False).round(4).head(10).to_frame("mean |SHAP|"))

# -----------------------------------------------------------------

## 12. Files save karna
# - `processed/retail_demand_features.parquet`: clean period se aage ki saari rows (train, valid, test), final features, target aur zaroori flags. Phase 06 (modeling) isi file se shuru hoga.
# - `reports/05_final_features.json`: final feature list, categorical columns, noise, aur har group ka faisla.
# - `reports/05_ablation_results_v2.csv`: har variant ka har window par score (cache bhi).


keep_cols = (KEYS + ["date", "split"] + [c for c in FINAL_FEATURES if c not in KEYS] +
             [c for c in ALL_FEATURES if c not in FINAL_FEATURES])          # ablation columns bhi (Phase 06 mein dobara test ke liye)
keep_cols = list(dict.fromkeys(keep_cols))
out = F.loc[F["date"] >= CLEAN_START, keep_cols].copy()
flags = master.loc[master["date"] >= CLEAN_START, ["id", "sales_clean", "is_trainable", "is_dead_series", "is_pre_opening",
                                                   "is_store_closed", "is_closure_calendar_day"]]
out = pd.concat([out, flags], axis=1)
out["target_log1p"] = np.log1p(out["sales_clean"])
assert out.loc[out["split"] == "test", "target_log1p"].isna().all()
assert out.loc[out["split"] == "valid", "sales_clean"].notna().all()
out.to_parquet(PROCESSED / "retail_demand_features.parquet", index=False)
final_info = {
    "final_features": FINAL_FEATURES,
    "categorical_features": [c for c in FINAL_FEATURES if c in CAT_COLS],
    "target": "target_log1p (= log1p(sales_clean))",
    "train_filter": "split == 'train' & is_trainable == 1 & date >= 2015-06-01",
    "zero_rules": ["is_dead_series", "is_pre_opening", "is_closure_calendar_day"],
    "under_recorded_families_lag_mask": UNDER_FAMS,
    "noise": {"WAPE_points": round(float(NOISE_W), 4), "RMSLE": round(float(NOISE_R), 5)},
    "section9_pass_groups": [g for g in decision.index if decision.loc[g, "faisla"].startswith("PASS")],
    "section10_confirmed": CONFIRMED,
    "lag_source_rule": "masked (NaN) - Section 7b model test",
    "reference_model": "LightGBM (feature testing only; final model chosen in Phase 06)",
    "preprocessing_note": "features raw; encoding/scaling/NaN handling per model inside Phase 06 pipelines, fit on train only",
    "backward_dropped_groups": DROP_GROUPS if REDUCE_OK else [],
    "month_added_after_aug2016_test": bool(MONTH_OK),
    "phase06_todo": ["horizon strategy: single model with lag>=16 vs horizon-specific lags (lag>=h)",
                     "compare model families with own preprocessing", "re-check features if winner is not LightGBM"],
    "ablation_decision": decision[["avg ΔWAPE", "avg ΔRMSLE", "faisla"]].round(4).to_dict(orient="index"),
    "never_features": NEVER, "dropped_in_phase04": DROPPED,
}
json.dump(final_info, open(REPORT_DIR / "05_final_features.json", "w"), indent=2, default=str)
print("Saved:", PROCESSED / "retail_demand_features.parquet", out.shape)
print("Saved:", REPORT_DIR / "05_final_features.json")
print("Saved:", CACHE)

# ------------------------------------------------------------------

## 13. Khulasa
# **Kya bana:** 39 candidate features ek function (`build_features`) se. Dono **leakage tests pass** hue.

# **Feature selection, teeno tareeqon se:**

# | Tareeqa | Kahan | Nateeja |
# |---|---|---|
# | Filter (correlation, partial corr, redundancy) | Phase 04 | 37 features ka plan |
# | Wrapper: forward (base + group) | Section 8–10 | Koi extra group noise se bara aur mustaqil faida nahi deta |
# | Wrapper: backward (drop-one) | Section 10b | `store_type + cluster + city` aur `is_holiday_season_day` hataye |
# | Embedded: permutation importance (out-of-time), gain, SHAP | Section 10b, 11 | Taaza level + usi weekday ki history sab se aham |
# | Khaas test: `month` (do August windows) | Section 10c | August forecast ke liye shaamil |

# **Final feature set: 15 features** (`reports/05_final_features.json`).

# **Is version mein theek ki gayi ghaltiyan:**
# 1. **Backtest zero-rule:** "cutoff tak kabhi nahi biki" → 0. Pichla rule BOOKS jaisi launch-se-pehle wali series ko nahi pakarta tha. Isi ne May 2016 RMSLE aur `month` ke faisle ko bigaara tha.
# 2. **Scaling ka daawa:** test ke mutabiq "bilkul ek jaisa" ghalat tha. Asar noise jitna hai, aur faida nahi.
# 3. **LightGBM** ab saaf taur par **reference model** kaha gaya hai. Encoding/scaling har model ke apne pipeline ka kaam hai (Phase 06).
# 4. **Backward selection** aur **permutation importance** shaamil kiye gaye.

# **Imaandar hadein:**
# - Ablation sirf ~0.2 WAPE point se bare asar pakar sakta hai.
# - Final set **August ke liye** chuna gaya hai. May/Dec 2016 mein yeh base se thora bura hai (Section 10d). Doosre mahinon ke forecast se pehle `month` aur `is_holiday_season_day` dobara test hone chahiyen.
# - Validation feature selection mein use hui, is liye Phase 06 mein us ka score thora optimistic ho sakta hai.

# **Phase 06 (Modeling) ke liye:**
# 1. **Model families ka muqabla**, har ek apni preprocessing ke sath (Ridge: one-hot + scaling + impute; tree models: native). Wahi windows aur noise ki hadd.
# 2. **Horizon strategy:** ek model (sab lags ≥ 16) vs horizon-wise (din h ke liye lag ≥ h).
# 3. Agar jeetne wala model LightGBM nahi, to feature faislon ka halka test dobara.
# 4. Final model ko 15 Aug tak ke data par train kar ke test (16–31 Aug) ka forecast.