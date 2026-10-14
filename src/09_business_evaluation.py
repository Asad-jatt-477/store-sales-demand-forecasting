# =====================================================================================
# Retail Demand Forecasting System
# Phase 09: Business evaluation + explainability (model kahan acha, kahan kamzor, aur kyun)
# -------------------------------------------------------------------------------------
# Is script ka maqsad (koi naya model nahi; Phase 07/08 ka chuna hua model hi parkha jata hai):
#   1. 6 backtest windows (Phase 07) ke forecasts ek jagah: family, store, forecast ka din, weekday, series ka size,
#      holiday/promo -> ghalti kahan hai, aur kya woh har window mein ek hi taraf hai (systematic) ya ittefaq.
#   2. Khula masla #1 (2017 mein halka kam forecast): saal ke hisaab se bias, aur kya yeh haal ki growth se juda hai.
#   3. Explainability: permutation importance (features ko shuffle kar ke ghalti kitni barhi), aur promotion ka asar.
#      (SHAP is model par nahi chalta: HistGB ke categorical store/family columns SHAP parh nahi sakta -- test kiya.)
#   4. Business: zyada forecast = zyada stock (kharab hone wali cheezon mein wastage), kam forecast = khaali shelf.
#      Model vs seasonal naive (asaan tareeqa) units mein.
#   5. Safety buffer: har family ke liye kitna extra stock rakhein taa ke 80% / 90% din maal kam na pare -- purani
#      windows se seekh kar, holdout par TEST.
#
# Sections:
#   0 setup | 1 forecasts (6 windows) | 2 family | 3 store | 4 forecast ka din + weekday | 5 series ka size
#   6 holiday / promo din | 7 saal ke hisaab se bias (masla #1) | 8 permutation importance | 9 promotion ka asar
#   10 business: overstock / understock | 11 safety buffer (holdout par test) | 12 khulasa
#
# Chalane ka tareeqa (data folder se):   python 09_business_evaluation.py
# Zaroori: Phase 07 + 08 mukammal. reports/07_cache se 6 windows ke forecasts aate hain (training nahi).
# Waqt: ~1 training (explainability ke liye validation model, ~7 min) + chand minute. Resume: models/ aur 09_cache.
# KAGGLE_SCORE: Kaggle par submission ka score yahan likh dein (khulase mein aayega), warna None rehne dein.
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

KAGGLE_SCORE = None            # misal: 0.3912 (Kaggle ka RMSLE score); None = abhi upload nahi kiya

PROCESSED, REPORT_DIR, MODEL_DIR = Path("data/processed"), Path("reports"), Path("models")
FIG_DIR = REPORT_DIR / "figures" / "09_business_evaluation"
CACHE_07, CACHE_08, CACHE_09 = REPORT_DIR / "07_cache", REPORT_DIR / "08_cache", REPORT_DIR / "09_cache"
for d in (FIG_DIR, CACHE_09, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)
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


def wape(y, p):
    return np.abs(y - p).sum() / y.sum() * 100


def rmsle(y, p):
    return float(np.sqrt(np.mean((np.log1p(np.clip(p, 0, None)) - np.log1p(y)) ** 2)))


T_START = time.time()

# =====================================================================================
section("0. Setup: Phase 07 ka model, windows aur data")
# =====================================================================================
cfg = json.load(open(REPORT_DIR / "07_final_config.json"))
if cfg.get("smoke"):
    raise SystemExit("07_final_config.json SMOKE run ka hai -> pehle asli Phase 07 chalayen.")
FEATS, CATS, NOISE, WINDOWS = cfg["features"], cfg["categorical_features"], cfg["noise"], cfg["windows"]
VALID, HOLDOUT, TUNE = "Validation (Aug 2017)", cfg["holdout"], cfg["tuning_folds"]
SEL = TUNE + [VALID]
MEMBERS = cfg["members"]
ALL_W = TUNE + [VALID, HOLDOUT]
print(f"Model: {cfg['choice']} ({', '.join(m['config'] for m in MEMBERS)}) | windows: {ALL_W}")

df = pd.read_parquet(PROCESSED / "retail_demand_features.parquet")
df["family"] = df["family"].astype(str)
df = df.sort_values(KEYS + ["date"]).reset_index(drop=True)
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
TRAINABLE = (df["is_trainable"] == 1).values
OPEN = ((df["is_pre_opening"] == 0) & (df["is_store_closed"] == 0)).values
HAS_MAX_FEATURES = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 4)
print(f"Rows {len(df):,} | versions: sklearn {sklearn.__version__}, lightgbm {lgb.__version__}")


def win_masks(w):
    cutoff, start = (pd.Timestamp(x) for x in WINDOWS[w])
    trm = TRAINABLE & (DATES <= np.datetime64(cutoff))
    evm = (DATES >= np.datetime64(start)) & (DATES <= np.datetime64(start + pd.Timedelta(days=15)))
    return cutoff, trm, evm


def zero_rule(cutoff, rows):
    fs = df.loc[rows, "first_sale_date"]
    never = (fs.isna() | (fs > cutoff)).values
    return never | (df.loc[rows, "is_pre_opening"].values == 1) | (df.loc[rows, "is_closure_calendar_day"].values == 1)


def make_model(name, p, seed):
    if name == "LightGBM":
        return lgb.LGBMRegressor(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], num_leaves=p["num_leaves"],
                                 min_child_samples=p["min_child_samples"], subsample=p["subsample"], subsample_freq=1,
                                 colsample_bytree=p["colsample_bytree"], reg_lambda=p["reg_lambda"], random_state=seed, verbose=-1)
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


def cached(mb, w):
    # Phase 07 cache -> Phase 08 cache -> Phase 09 cache. Na mile to train (seed 0) aur 09 cache mein save.
    clean = lambda t: "".join(ch if ch.isalnum() else "_" for ch in t)
    for cp in (CACHE_07 / f"{clean(mb['model'])}__{mb['config']}__{clean(w)}__s0.parquet",
               CACHE_08 / f"{mb['model'].split()[0]}__{w.split()[0]}__s0.parquet",
               CACHE_08 / "reproduce_validation_s0.parquet" if w == VALID and len(MEMBERS) == 1 else Path("_none_"),
               CACHE_09 / f"{mb['model'].split()[0]}__{w.split()[0]}__s0.parquet"):
        if cp.exists():
            return pd.read_parquet(cp)["pred"].values
    cutoff, trm, evm = win_masks(w)
    t0 = time.time()
    model = make_model(mb["model"], mb["params"], 0).fit(df.loc[trm, FEATS], Y[trm])
    p = np.where(zero_rule(cutoff, evm), 0, np.expm1(model.predict(df.loc[evm, FEATS])).clip(0))
    pd.DataFrame({"pred": p}).to_parquet(CACHE_09 / f"{mb['model'].split()[0]}__{w.split()[0]}__s0.parquet", index=False)
    print(f"    (cache nahi mila) {mb['model']} {w}: train + forecast {time.time() - t0:.0f}s", flush=True)
    return p


def seasonal_naive(w, evm):
    cutoff = pd.Timestamp(WINDOWS[w][0])
    p = np.expm1(df.loc[evm, "same_wd_mean4"].fillna(df.loc[evm, "roll28_lag16"]).fillna(0).values)
    return np.where(zero_rule(cutoff, evm), 0, np.clip(p, 0, None))


# =====================================================================================
section("1. 6 windows ke forecasts ek jagah (khule stores; Phase 07 cache, seed 0)")
# =====================================================================================
# Har row: window, store, family, din, asli sales, model forecast, seasonal naive forecast + kuch attributes.
# Series ka "size" cutoff se PEHLE ke 90 din ki average se (forecast ke waqt maloom) -> leakage nahi.
frames = []
for w in ALL_W:
    cutoff, _, evm = win_masks(w)
    p = np.mean([cached(mb, w) for mb in MEMBERS], axis=0)
    pre = HIST & (DATES > np.datetime64(cutoff - pd.Timedelta(days=90))) & (DATES <= np.datetime64(cutoff)) & OPEN
    size = df.loc[pre].groupby(KEYS)["sales_clean"].agg(avg90="mean", zero90=lambda s: (s == 0).mean()).reset_index()
    f = df.loc[evm, KEYS + ["date", "store_type", "store_cluster", "city", "onpromotion", "is_national_holiday",
                            "is_national_event", "is_local_holiday", "is_regional_holiday"]].copy()
    f["y"], f["p"], f["naive"], f["open"] = SALES[evm], p, seasonal_naive(w, evm), OPEN[evm]
    f["window"] = w
    f["year"] = pd.Timestamp(WINDOWS[w][1]).year
    f["din"] = (f["date"] - pd.Timestamp(WINDOWS[w][1])).dt.days + 1
    f = f.merge(size, on=KEYS, how="left")
    frames.append(f)
E = pd.concat(frames, ignore_index=True)
E = E[E["open"]].drop(columns="open")
E["err"], E["err_naive"], E["diff"] = (E["y"] - E["p"]).abs(), (E["y"] - E["naive"]).abs(), E["p"] - E["y"]
summ = E.groupby("window").apply(lambda g: pd.Series({"WAPE model": wape(g.y, g.p), "WAPE naive": wape(g.y, g.naive),
                                                      "RMSLE model": rmsle(g.y, g.p), "forecast/actual": g.p.sum() / g.y.sum()}))
print(summ.loc[ALL_W].round(3).to_string())
print(f"Rows: {len(E):,} (6 windows x khule store rows). In mein se holdout kisi faisle mein shamil nahi tha.")


def seg_table(by, min_units_share=0.0):
    g = E.groupby(by)
    t = pd.DataFrame({"units": g["y"].sum(), "WAPE model %": g["err"].sum() / g["y"].sum() * 100,
                      "WAPE naive %": g["err_naive"].sum() / g["y"].sum() * 100,
                      "forecast/actual": g["p"].sum() / g["y"].sum()})
    t["kul ghalti mein hissa %"] = g["err"].sum() / E["err"].sum() * 100
    # Har window mein bias ki simt: kitni windows mein forecast kam (sum p < sum y)
    sign = E.groupby([by, "window"]).apply(lambda x: x["p"].sum() < x["y"].sum()).unstack("window")
    t["kam forecast (windows)"] = sign.sum(axis=1).astype(int)
    t["model naive se behtar"] = t["WAPE model %"] < t["WAPE naive %"]
    return t


# =====================================================================================
section("2. Family ke hisaab se: ghalti kahan, aur kya bias har window mein ek taraf (systematic)?")
# =====================================================================================
# Systematic ka usool (pehle se tay): 6 mein se 6 windows mein ek hi taraf (sab kam ya sab zyada).
fam = seg_table("family").sort_values("kul ghalti mein hissa %", ascending=False)
print(fam.head(12).round(2).to_string())
sys_under = fam[fam["kam forecast (windows)"] == len(ALL_W)].index.tolist()
sys_over = fam[fam["kam forecast (windows)"] == 0].index.tolist()
print(f"\nHar window mein KAM forecast (systematic): {sys_under or 'koi nahi'}")
print(f"Har window mein ZYADA forecast (systematic): {sys_over or 'koi nahi'}")
print(f"Families jahan model seasonal naive se behtar: {int(fam['model naive se behtar'].sum())}/{len(fam)}")
fam.to_csv(REPORT_DIR / "09_error_by_family.csv")

fig, ax = plt.subplots(figsize=(10, 6.5))
t = fam.sort_values("WAPE model %")
yy = np.arange(len(t))
ax.barh(yy, t["WAPE naive %"], color=GRAY, alpha=0.5, label="seasonal naive")
ax.barh(yy, t["WAPE model %"], height=0.5, color=BLUE, label="model")
ax.set_yticks(yy, t.index, fontsize=7)
ax.set_xlabel("WAPE % (6 windows, khule stores)")
ax.set_title("Family ke hisaab se ghalti: model vs seasonal naive")
ax.legend(fontsize=8)
save(fig, "01_wape_by_family")

# =====================================================================================
section("3. Store ke hisaab se + store ki qisam (type) aur city")
# =====================================================================================
st = seg_table("store_nbr").sort_values("WAPE model %", ascending=False)
print("Sab se zyada ghalti wale 8 stores:")
print(st.head(8).round(2).to_string())
print(f"Stores jahan model naive se BURA: {int((~st['model naive se behtar']).sum())}/{len(st)} -> "
      f"{st.index[~st['model naive se behtar']].tolist()}")
ty = seg_table("store_type").sort_values("WAPE model %")
print("\nStore type ke hisaab se:")
print(ty.round(2).to_string())
st.to_csv(REPORT_DIR / "09_error_by_store.csv")

# =====================================================================================
section("4. Forecast ka din (1-16) aur weekday: kya door ke din zyada ghalat hain?")
# =====================================================================================
# Sab features >= 16 din purane -> har din ke liye maloomat ek jaisi purani. Data batayega ke ghalti din ke sath
# barhti hai ya nahi (Phase 06 ka horizon test isi se juda tha).
dn = E.groupby("din").apply(lambda g: pd.Series({"WAPE model %": wape(g.y, g.p), "WAPE naive %": wape(g.y, g.naive),
                                                  "forecast/actual": g.p.sum() / g.y.sum()}))
print(dn.round(2).T.to_string())
slope = np.polyfit(dn.index, dn["WAPE model %"], 1)[0]
print(f"Har agle din ghalti mein average tabdeeli: {slope:+.3f} points/din (din 1 se 16 tak kul ~{slope * 15:+.2f})")
wd = E.assign(weekday=E["date"].dt.day_name()).groupby("weekday").apply(
    lambda g: pd.Series({"WAPE model %": wape(g.y, g.p), "forecast/actual": g.p.sum() / g.y.sum()}))
print("\nWeekday:")
print(wd.round(3).T.to_string())

fig, ax = plt.subplots(figsize=(9, 3.8))
ax.plot(dn.index, dn["WAPE model %"], marker="o", color=BLUE, label="model")
ax.plot(dn.index, dn["WAPE naive %"], marker="o", color=GRAY, linestyle="--", label="seasonal naive")
ax.set_xlabel("forecast ka din (1 = cutoff ke agle din)")
ax.set_ylabel("WAPE %")
ax.set_title("Forecast ke har din ki ghalti (6 windows)")
ax.legend(fontsize=8)
save(fig, "02_wape_by_horizon_day")

# =====================================================================================
section("5. Series ka size: bari aur choti (kam bikne wali) series par model kaisa hai?")
# =====================================================================================
# Size = cutoff se pehle ke 90 din ki average rozana sales (maloom maloomat). 5 barabar hisson (quintiles) mein.
E["size_grp"] = pd.qcut(E["avg90"].rank(method="first"), 5, labels=["1 sab se choti", "2", "3", "4", "5 sab se bari"])
sz = E.groupby("size_grp", observed=True).apply(lambda g: pd.Series({
    "avg sales/din": g["avg90"].mean(), "zero din % (pichle 90)": g["zero90"].mean() * 100, "units %": g.y.sum() / E.y.sum() * 100,
    "WAPE model %": wape(g.y, g.p), "WAPE naive %": wape(g.y, g.naive), "forecast/actual": g.p.sum() / g.y.sum(),
    "RMSLE model": rmsle(g.y, g.p)}))
print(sz.round(3).to_string())
print("Padhne ka tareeqa: WAPE units ka hisaab hai (bari series haavi); choti series ka asar RMSLE mein dikhta hai.")
print("Choti series par forecast/actual 1 se kam: target log1p hai -> model log mein 'beech' ka andaza deta hai, jo choti/\n"
      "  kabhi-kabhi bikne wali series mein asli average se neeche hota hai (bahar ki maloomat). RMSLE ke liye yahi sahi hai,\n"
      "  units ke liye kam. Yeh series kul units ka bohot chhota hissa hain (upar 'units %').")

# =====================================================================================
section("6. Holiday aur promotion wale din vs aam din")
# =====================================================================================
E["din_qisam"] = np.select([E["is_national_holiday"] == 1, (E["is_local_holiday"] + E["is_regional_holiday"]) > 0,
                            E["is_national_event"] == 1], ["national holiday", "local/regional holiday", "national event"],
                           default="aam din")
E["promo"] = np.where(E["onpromotion"] > 0, "promotion par", "promotion nahi")
for col in ("din_qisam", "promo"):
    t = E.groupby(col).apply(lambda g: pd.Series({"rows": len(g), "units %": g.y.sum() / E.y.sum() * 100,
                                                  "WAPE model %": wape(g.y, g.p), "WAPE naive %": wape(g.y, g.naive),
                                                  "forecast/actual": g.p.sum() / g.y.sum()}))
    print(t.round(3).to_string(), "\n")

# =====================================================================================
section("7. Khula masla #1: 2017 mein halka kam forecast -- saal ke hisaab se aur growth ke sath")
# =====================================================================================
# Sawal: kya bias saal se juda hai? Aur kya woh cutoff se pehle ki growth (haal ki sales / pichle saal usi waqt;
# forecast ke waqt maloom) ke sath chalta hai? 6 windows = kamzor saboot; sirf ishara, faisla nahi.
yr = E.groupby("year").apply(lambda g: pd.Series({"windows": g.window.nunique(), "forecast/actual": g.p.sum() / g.y.sum(),
                                                  "WAPE model %": wape(g.y, g.p)}))
print(yr.round(3).to_string())
daily = df.loc[HIST].groupby("date")["sales_clean"].sum()
gr = []
for w in ALL_W:
    c = pd.Timestamp(WINDOWS[w][0])
    now = daily[c - pd.Timedelta(days=27):c].sum()
    ly = daily[c - pd.Timedelta(days=27 + 364):c - pd.Timedelta(days=364)].sum()
    prev = daily[c - pd.Timedelta(days=55):c - pd.Timedelta(days=28)].sum()
    gr.append({"window": w, "forecast/actual": summ.loc[w, "forecast/actual"],
               "growth YoY (28 din)": now / ly if ly > 0 else np.nan, "growth pichle 28 din se": now / prev})
gr = pd.DataFrame(gr).set_index("window")
print(gr.round(3).to_string())
for col in ["growth YoY (28 din)", "growth pichle 28 din se"]:
    ok = gr[[col, "forecast/actual"]].dropna()
    if len(ok) >= 4:
        print(f"Correlation (forecast/actual vs {col}): {ok.corr().iloc[0, 1]:+.2f} ({len(ok)} windows)")
print("Padhne ka tareeqa: mazboot manfi correlation = jab sales tezi se barhti hai, model peeche reh jata hai (trend\n"
      "  poora nahi pakarta). Sirf 6 points hain -> yeh ishara hai, saboot nahi. Kaggle score ke sath dekhein.")

# =====================================================================================
section("8. Explainability: permutation importance (validation window, out-of-sample)")
# =====================================================================================
# Tareeqa: validation cutoff (30 Jul) tak train hua model; validation ke 16 din par ek feature (ya group) ko rows ke
# beech shuffle karo -> WAPE kitna barha. Zyada barha = model us maloomat par zyada depend karta hai.
# Groups: sales history ke lags aapas mein bohot milte julte hain (Phase 04); akele shuffle karne par doosra lag kami
# poori kar deta hai -> asar kam dikhta. Is liye group bhi shuffle hota hai. 3 dafa shuffle, average.
cutoff_v, trm_v, evm_v = win_masks(VALID)
vm_path = MODEL_DIR / "09_validation_model.joblib"
mb0 = MEMBERS[0]
if vm_path.exists():
    vmodel = joblib.load(vm_path)
    print(f"  validation model load kiya ({vm_path})")
else:
    t0 = time.time()
    vmodel = make_model(mb0["model"], mb0["params"], 0).fit(df.loc[trm_v, FEATS], Y[trm_v])
    joblib.dump(vmodel, vm_path, compress=3)
    print(f"  validation model train: {time.time() - t0:.0f}s -> {vm_path}", flush=True)
Xv = df.loc[evm_v, FEATS].reset_index(drop=True)
zr_v = zero_rule(cutoff_v, evm_v)
o_v = OPEN[evm_v]
y_v = SALES[evm_v][o_v]


def pred_v(X):
    return np.where(zr_v, 0, np.expm1(vmodel.predict(X)).clip(0))[o_v]


base_w = wape(y_v, pred_v(Xv))
print(f"  base validation WAPE (yeh model, seed 0): {base_w:.3f}%"
      + (" -- (ensemble ka sirf pehla member parkha ja raha hai)" if len(MEMBERS) > 1 else ""))
GROUPS = {"sales history (sab lags)": [f for f in FEATS if f.startswith(("lag_", "roll", "same_wd"))],
          "calendar (dow, dom, month)": [f for f in ("dow", "dom", "month") if f in FEATS],
          "holiday/event flags": [f for f in FEATS if f.startswith("is_")],
          "store + family": [f for f in ("store_cat", "family_cat") if f in FEATS]}
items = [(f, [f]) for f in FEATS] + [(g, cols) for g, cols in GROUPS.items() if len(cols) > 1]
rng = np.random.default_rng(0)
imp = []
for name, cols in items:
    ws = []
    for _ in range(3):
        Xp = Xv.copy()
        perm = rng.permutation(len(Xp))
        for c in cols:
            Xp[c] = Xv[c].values[perm]
        if any(c in CATS for c in cols):
            for c in cols:
                if c in CATS:
                    Xp[c] = pd.Categorical(Xp[c], categories=Xv[c].cat.categories)
        ws.append(wape(y_v, pred_v(Xp)))
    imp.append({"feature / group": name, "kind": "group" if len(cols) > 1 else "feature",
                "WAPE barha (points)": np.mean(ws) - base_w, "spread": np.std(ws)})
imp = pd.DataFrame(imp).sort_values("WAPE barha (points)", ascending=False).set_index("feature / group")
const = [f for f in FEATS if Xv[f].nunique(dropna=False) <= 1]
imp["note"] = ["is window mein badalta hi nahi -> yahan nap nahi sakte" if i in const else "" for i in imp.index]
print(imp.round(3).to_string())
imp.to_csv(REPORT_DIR / "09_permutation_importance.csv")
print("Padhne ka tareeqa: 0 ke qareeb = model is ke baghair bhi theek (ya doosra feature wahi maloomat de raha).")

fig, ax = plt.subplots(figsize=(9, 5.2))
t = imp.sort_values("WAPE barha (points)")
ax.barh(t.index, t["WAPE barha (points)"], color=[ORANGE if k == "group" else BLUE for k in t["kind"]])
ax.set_xlabel("WAPE kitna barha jab yeh shuffle kiya (points)")
ax.set_title("Model kis maloomat par depend karta hai (narangi = group)")
save(fig, "03_permutation_importance")

# =====================================================================================
section("9. Promotion ka asar (model ke mutabiq): agar promotion na hoti to forecast kitna hota?")
# =====================================================================================
# Validation ki jin rows par onpromotion > 0 hai, wahan onpromotion = 0 kar ke dobara forecast. Farq = model ke mutabiq
# promotion ka asar. DHYAN: yeh model ka seekha hua rishta hai, sabit shuda sabab (causal) nahi.
promo_rows = Xv["onpromotion"].values > 0
X0 = Xv.copy()
X0.loc[promo_rows, "onpromotion"] = 0
p_with = np.where(zr_v, 0, np.expm1(vmodel.predict(Xv)).clip(0))
p_wo = np.where(zr_v, 0, np.expm1(vmodel.predict(X0)).clip(0))
fam_v = df.loc[evm_v, "family"].values
pr = pd.DataFrame({"family": fam_v, "promo": promo_rows, "with": p_with, "without": p_wo})
pr = pr[pr["promo"]].groupby("family")[["with", "without"]].sum()
pr["promotion se izafa %"] = (pr["with"] / pr["without"] - 1) * 100
pr["promo rows"] = pd.Series(fam_v[promo_rows]).value_counts()
pr = pr[pr["promo rows"] >= 50].sort_values("promotion se izafa %", ascending=False)
print(f"Promotion wali rows: {promo_rows.sum():,} | kul izafa (model ke mutabiq): {(pr['with'].sum() / pr['without'].sum() - 1) * 100:+.1f}%")
print(pd.concat([pr.head(6), pr.tail(4)]).round(1).to_string())
big = pr.index[pr["promotion se izafa %"] > 100].tolist()
if big:
    print(f"DHYAN {big}: izafa 100% se zyada -> aksar promotion aur season/event ek hi waqt (data mein saath saath),\n"
          "  is liye model dono ka asar promotion ko de deta hai. Yeh number promotion ka asal asar NAHI maana jaye.")

# =====================================================================================
section("10. Business: zyada forecast (overstock) vs kam forecast (understock) -- model vs seasonal naive")
# =====================================================================================
# Overstock units = forecast asli se zyada (zyada maal mangwaya). Understock = kam (shelf khaali, sale gayi).
# Kharab hone wali families (naam se, bahar ki maloomat): fresh cheezein jin mein zyada stock = wastage.
PERISH = [f for f in ["PRODUCE", "DAIRY", "MEATS", "POULTRY", "SEAFOOD", "BREAD/BAKERY", "EGGS", "DELI", "PREPARED FOODS"]
          if f in set(E["family"])]
E["over"], E["under"] = (E["p"] - E["y"]).clip(lower=0), (E["y"] - E["p"]).clip(lower=0)
E["over_n"], E["under_n"] = (E["naive"] - E["y"]).clip(lower=0), (E["y"] - E["naive"]).clip(lower=0)
E["perish"] = E["family"].isin(PERISH)
bz = []
for label, m in [("sab families", np.ones(len(E), bool)), ("kharab hone wali", E["perish"].values),
                 ("baqi", ~E["perish"].values)]:
    g = E[m]
    bz.append({"group": label, "asli units": g.y.sum(), "overstock model": g.over.sum(), "overstock naive": g.over_n.sum(),
               "understock model": g.under.sum(), "understock naive": g.under_n.sum()})
bz = pd.DataFrame(bz).set_index("group")
bz["overstock kami %"] = (1 - bz["overstock model"] / bz["overstock naive"]) * 100
bz["understock kami %"] = (1 - bz["understock model"] / bz["understock naive"]) * 100
print(bz.round(0).to_string())
print(f"Kharab hone wali families (naam se): {PERISH}")
print("Paisa nahi: dataset mein qeemat (price) nahi -> sirf units. Production mein qeemat x units = rupaye ka asar.")

# =====================================================================================
section("11. Safety buffer: har family mein kitna extra stock taa ke 80% / 90% din maal kam na pare")
# =====================================================================================
# Seekhna SIRF faisla windows (4 folds + validation) se: har store-family-din par asli/forecast. Family ka 80/90
# percentile = itna buffer (forecast x (1 + b)). TEST: yahi buffer holdout (anchhui) par lagao -> kitne % din maal pura.
TARGETS = [0.8, 0.9]
lrn = E[E["window"].isin(SEL) & (E["p"] > 0)].copy()
tst = E[(E["window"] == HOLDOUT) & (E["p"] > 0)].copy()
lrn["r"], tst["r"] = lrn["y"] / lrn["p"], tst["y"] / tst["p"]
buf = lrn.groupby("family")["r"].quantile(TARGETS).unstack()
buf.columns = [f"buffer {int(q * 100)}%" for q in TARGETS]
buf = (buf - 1) * 100
tst = tst.merge(buf, left_on="family", right_index=True)
cov = {}
for q in TARGETS:
    col = f"buffer {int(q * 100)}%"
    ok = tst["y"] <= tst["p"] * (1 + tst[col] / 100)
    cov[col] = ok.groupby(tst["family"]).mean() * 100
    print(f"Target {int(q * 100)}%: holdout par asal mein maal pura raha {ok.mean() * 100:.1f}% store-family-din "
          f"(families ki range {cov[col].min():.0f}%-{cov[col].max():.0f}%)")
bt = buf.join(pd.DataFrame({f"holdout coverage ({k})": v for k, v in cov.items()}))
units_share = E.groupby("family")["y"].sum() / E["y"].sum() * 100
bt["units %"] = units_share
print(bt.sort_values("units %", ascending=False).head(12).round(1).to_string())
bt.to_csv(REPORT_DIR / "09_safety_buffer_by_family.csv")
print("Padhne ka tareeqa: 'buffer 80% = +25' ka matlab forecast se 25% zyada rakhein to purani windows mein 80% din kaafi tha;\n"
      "  holdout coverage batata hai ke nayi window par yeh waada kitna nibha (80 ke qareeb = buffer bharose ke qabil).")

# =====================================================================================
section("12. Khulasa (numbers se khud bana hua)")
# =====================================================================================
top_imp = imp[imp["kind"] == "feature"].head(3)
lines = [
    f"- 6 windows: model WAPE {summ['WAPE model'].min():.2f}-{summ['WAPE model'].max():.2f}% vs seasonal naive "
    f"{summ['WAPE naive'].min():.2f}-{summ['WAPE naive'].max():.2f}%; holdout {summ.loc[HOLDOUT, 'WAPE model']:.2f}%.",
    f"- Families: model naive se behtar {int(fam['model naive se behtar'].sum())}/{len(fam)}; systematic kam: {sys_under or 'koi nahi'}; "
    f"systematic zyada: {sys_over or 'koi nahi'}.",
    f"- Stores: naive se bura {int((~st['model naive se behtar']).sum())}/{len(st)}.",
    f"- Forecast ka din: ghalti {slope:+.3f} points/din (din 1 se 16 tak ~{slope * 15:+.2f}).",
    f"- Saal: forecast/actual " + ", ".join(f"{int(k)}: {v:.3f}" for k, v in yr["forecast/actual"].items()) + ".",
    f"- Sab se ahem maloomat: " + ", ".join(f"{i} (+{v:.2f})" for i, v in top_imp["WAPE barha (points)"].items())
    + f"; sales history group +{imp.loc['sales history (sab lags)', 'WAPE barha (points)']:.2f}.",
    f"- Business (6 windows): overstock {bz.loc['sab families', 'overstock kami %']:.0f}% kam, understock "
    f"{bz.loc['sab families', 'understock kami %']:.0f}% kam (naive ke muqable); kharab hone wali families mein overstock "
    f"{bz.loc['kharab hone wali', 'overstock kami %']:.0f}% kam.",
    f"- Safety buffer: 80% target holdout par asal coverage "
    f"{(tst['y'] <= tst['p'] * (1 + tst['buffer 80%'] / 100)).mean() * 100:.1f}%, 90% target "
    f"{(tst['y'] <= tst['p'] * (1 + tst['buffer 90%'] / 100)).mean() * 100:.1f}%.",
    f"- Kaggle score: {KAGGLE_SCORE if KAGGLE_SCORE is not None else 'abhi nahi (upload kar ke KAGGLE_SCORE likhein)'}"
    + (f" vs validation RMSLE {summ.loc[VALID, 'RMSLE model']:.3f}" if KAGGLE_SCORE is not None else ""),
    "- Limitations: qeemat nahi (rupaye nahi, sirf units); store renovation calendar nahi; SCHOOL season har saal alag; "
    "permutation importance model ka rishta batata hai, sabab (causal) nahi.",
]
for ln in lines:
    print(ln)
out = {"phase": "09", "kaggle_score": KAGGLE_SCORE, "window_summary": summ.round(4).reset_index().to_dict(orient="records"),
       "systematic_under_families": sys_under, "systematic_over_families": sys_over,
       "stores_worse_than_naive": st.index[~st["model naive se behtar"]].tolist(), "horizon_slope_points_per_day": float(slope),
       "bias_by_year": yr["forecast/actual"].round(4).to_dict(), "growth_vs_bias": gr.round(4).reset_index().to_dict(orient="records"),
       "permutation_importance": imp["WAPE barha (points)"].round(4).to_dict(),
       "business": bz.round(1).reset_index().to_dict(orient="records"), "perishable_families_by_name": PERISH}
json.dump(out, open(REPORT_DIR / "09_summary.json", "w"), indent=2, default=str)
print(f"\nSaved: reports/09_error_by_family.csv, 09_error_by_store.csv, 09_permutation_importance.csv, "
      f"09_safety_buffer_by_family.csv, 09_summary.json, charts in {FIG_DIR}")
print(f"Kul waqt: {(time.time() - T_START) / 60:.1f} minute")
