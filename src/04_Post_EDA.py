# # Retail Demand Forecasting System
# ## Phase 04: Post-EDA (features ka faisla, saboot ke sath)

# | Phase | Notebook | Kaam |
# |---|---|---|
# | 01 | `01_data_preparation` | Master dataset aur flags |
# | 02 | `02_Pre_EDA` | Data ki nature samajhna |
# | 03 | `03_Outliers_Detection_and_Treatment` | Har extreme ki wajah. Test ke baad: **koi value nahi badli** |
# | **04** | **`04_Post_EDA` (yeh notebook)** | Pre-EDA ke patterns ki **mazbooti** check karna, aur har mumkin feature ka **saboot** |
# | 05 | Feature engineering | Is notebook ki feature list ko banana aur **ablation test** se final karna |

# **Post-EDA ka maqsad:** Pre-EDA ne batayi ke data **kaisa hai**. Post-EDA yeh poochta hai: **model ko kya dena chahiye, aur kyun?** Har feature ke liye teen sawal hain:
# 1. **Kya forecast ke waqt yeh maloom hoga?** (leakage nahi hona chahiye)
# 2. **Kya is mein target ke baare mein asli information hai?** (data se naapa jayega)
# 3. **Kya yeh kisi doosre feature ki copy to nahi?** (redundancy)

# **Kahani ki misaal:** ek dukandaar ko kal ke liye maal mangwana hai. Woh apne pichle registers dekhta hai: "Pichle Sunday kitna bika? Pichle mahine ki average kya thi? Kal tankhwah ka din hai? Kal discount hai?" Har sawal ek **feature** hai. Post-EDA yeh check karta hai ke kaun se sawal waqai madad karte hain aur kaun se sirf waqt zaya karte hain.

# **Phase 03 ka nateeja yahan kya matlab rakhta hai:** outlier treatment ke baad koi value nahi badli (`sales_clean == sales`). Is liye "treatment ke baad data kaisa hai" wala aam Post-EDA step yahan chhota hai. Asal kaam yeh hai:
# - (a) Pre-EDA ke patterns **asli events ke baghair bhi** wahi rehte hain ya nahi.
# - (b) Har feature ka saboot.

# **Usool (pichle phases jaise):**
# 1. **Sirf training period** (clean period: 1 Jun 2015 se 30 Jul 2017). Validation ki sales nahi dekhi jayengi. Validation/test ke **features** (promotion, holidays) dekhe ja sakte hain, kyun ke woh forecast ke waqt maloom hote hain.
# 2. Analysis sirf `is_trainable == 1` rows par.
# 3. **Har insight ke numbers upar ke output se** aate hain.
# 4. Yahan features ka **faisla saboot se** hota hai. **Final** keep/drop Phase 05 mein ablation test (seed noise ke muqable) se hoga, jaisa Phase 03 mein seekha.

# | Section | Sawal |
# |---|---|
# | 0 | Setup aur Phase 03 ka check |
# | 1 | Naapne ka tareeqa: "local normal" se farq (residual) |
# | 2 | Kya Pre-EDA ke patterns asli events ke baghair bhi wahi hain? |
# | 3 | Sales history features: kaun se din ki purani sales sab se zyada batati hain? |
# | 4 | History features ki aapas mein overlap (redundancy) |
# | 5 | Lag sources mein band/partial din: 0 rakhein ya khaali (NaN)? |
# | 6 | Calendar: weekday, payday, mahina |
# | 7 | Holidays: kis qisam ka asar, aur kya har store par ek jaisa? |
# | 8 | Nayi calendar khojen: Carnaval Sunday, holiday se pehle/baad ke din |
# | 9 | Promotion: asar, shakal, aur family ke hisaab se farq |
# | 10 | Store aur family ki static information |
# | 11 | Group-level history: family aur store ka kul rujhan |
# | 12 | Oil aur transactions: rakhein ya nahi? |
# | 13 | Volatility aur zero-share: intermittent series ke liye |
# | 14 | Naye stores aur nayi series (store 52) |
# | 15 | Validation/test window mein features ki availability |
# | 16 | Final feature plan (Phase 05 ke liye) |





## 0. Setup aur Phase 03 ka check
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*obj.round has no effect.*")
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)

PROCESSED = Path("data/processed")
REPORT_DIR = Path("reports")
FIG_DIR = REPORT_DIR / "figures" / "04_post_eda"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
GRAY, LIGHT_GRAY = "#9a9994", "#d9d8d4"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e8e7e3"
DIVERGING = LinearSegmentedColormap.from_list("div", [BLUE, "#f0efec", RED])

plt.rcParams.update({
    "figure.figsize": (10, 4.2), "figure.dpi": 110, "savefig.dpi": 150,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 10, "axes.labelsize": 10, "axes.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c8c7c2", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "text.color": INK, "legend.frameon": False, "legend.fontsize": 9,
    "lines.linewidth": 2, "lines.solid_capstyle": "round",
})

def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)      # script mein popup/blocking nahi; figure disk par save ho chuki hai

pctfmt = mtick.PercentFormatter(decimals=0)

def display(obj):
    """Notebook ke display() ki jagah: script mein table print karta hai."""
    print(obj.to_string() if hasattr(obj, "to_string") else obj)
    print()
print("Charts folder:", FIG_DIR)


# -------------------------------------------------------------------

master = pd.read_parquet(PROCESSED / "retail_demand_master_treated.parquet")
master["family"] = master["family"].astype(str)
master = master.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
store_tx = pd.read_parquet(PROCESSED / "store_day_transactions.parquet")

CLEAN_START = pd.Timestamp("2015-06-01")
TRAIN_END = master.loc[master["split"] == "train", "date"].max()
KEYS = ["store_nbr", "family"]

# --- Phase 03 ka nateeja confirm ---
same = np.array_equal(master["sales_clean"].fillna(-1).values, master["sales"].fillna(-1).values)
print("sales_clean == sales (Phase 03 mein koi value nahi badli):", same)
assert same
print("Outlier causes (Phase 03):")
print(master.loc[master["outlier_cause"] != "none", "outlier_cause"].value_counts().to_string())

# --- Har series ki har tareekh mojood hai? (lag = row shift tabhi sahi hai) ---
gap = master.groupby(KEYS)["date"].diff().dropna()
assert (gap == pd.Timedelta(days=1)).all()
print("\nHar series mein din lagataar hain (lag ke liye shift sahi hai): True")

# Partial-recording din (Phase 02 ka rule: >= 5 families ka data gayab)
unavail = master[master["is_family_unavailable"] == 1].groupby("date")["family"].nunique()
PARTIAL_DATES = pd.DatetimeIndex(unavail[unavail >= 5].index)
print(f"Partial-recording din: {len(PARTIAL_DATES)} (aakhri: {PARTIAL_DATES.max().date()})")

# --------------------------------------------------------------------------

# ## 1. Naapne ka tareeqa: "local normal" se farq (residual)
# Kisi cheez (holiday, payday, promotion) ka asar naapne ke liye seedha average lena **dhoka** de sakta hai. Bare stores aur bari families average ko kheench lete hain, aur trend (sales ka waqt ke sath barhna) bhi asar mein mil jata hai.

# **Tareeqa:** har series ki har value ko **usi series ke aas paas ke dinon** se compare karo:
# - `y = log1p(sales)` (Phase 02 ka faisla).
# - **Local level** = usi series ke 28 din (14 peechhe, 14 aage) ki average `y`. Is mein har weekday 4 baar aata hai, is liye weekday ka asar barabar ho jata hai.
# - **Residual** = `y − local level`. Isay % mein badalne ke liye `exp(residual) − 1`.
# - Din-khaas cheezon (holiday, Carnaval) ke liye **weekday normal** use hoga: usi weekday ki 9 hafton ki median (Phase 03 wala). Is se weekday ka asar pehle hi hat jata hai.

# Yeh **sirf naapne** ka tareeqa hai, feature nahi. Is mein "aage ke din" bhi shaamil hain, jo feature mein kabhi nahi ho sakte.


hist = master[master["split"] == "train"].copy()
A = hist[(hist["is_trainable"] == 1) & (hist["date"] >= CLEAN_START)].copy()     # analysis rows
A["y"] = np.log1p(A["sales_clean"])
A["dow"] = A["date"].dt.dayofweek
A["dom"] = A["date"].dt.day
A["month"] = A["date"].dt.month

# Local level: 28-din centered average (series ke andar). Non-trainable din NaN hain, isliye date-index par reindex.
full = hist[hist["date"] >= CLEAN_START - pd.Timedelta(days=60)][KEYS + ["date"]].copy()
full = full.merge(A[KEYS + ["date", "y"]], on=KEYS + ["date"], how="left")
full["local_level"] = (full.groupby(KEYS)["y"].transform(lambda s: s.rolling(29, center=True, min_periods=14).mean()))
full["dow"] = full["date"].dt.dayofweek
full["wd_normal"] = (full.groupby(KEYS + ["dow"])["y"]
                       .transform(lambda s: s.rolling(9, center=True, min_periods=5).median()))
A = A.merge(full[KEYS + ["date", "local_level", "wd_normal"]], on=KEYS + ["date"], how="left")
A["resid"] = A["y"] - A["local_level"]
A["wd_resid"] = A["y"] - A["wd_normal"]
eff = lambda r: (np.exp(np.nanmean(r)) - 1) * 100          # residual -> % asar
print(f"Analysis rows: {len(A):,} | series: {A[KEYS].drop_duplicates().shape[0]:,} | "
      f"residual ban saka: {A['resid'].notna().mean():.1%}")


# --------------------------------------------------------------------------

# ## 2. Kya Pre-EDA ke patterns asli events ke baghair bhi wahi hain?
# Phase 03 ne har extreme value ko ek wajah di thi (`outlier_cause`). Sawal yeh hai ke Pre-EDA ke patterns (weekday, payday, mahina) kahin **chand bare events** ki wajah se to nahi bane. **Do tests:**
# 1. **Extremes hata kar:** patterns ko saari rows par aur Phase 03 ke extremes hata kar naapo. *Yeh test kamzor hai:* extremes sirf ~0.6% rows hain, is liye in ke hatne se average ka zyada badalna mushkil hai. Yeh sirf yeh batata hai ke chand bare din pattern nahi bana rahe.
# 2. **Mazboot test, waqt ke sath:** clean period ko do hisson mein bant kar (Jun 2015–May 2016 aur Jun 2016–Jul 2017) dekho ke pattern **dono saal wahi** hai ya nahi. Agar ek saal ka pattern doosre saal dohrata hai, to model future mein bhi us par bharosa kar sakta hai.


normal = A[A["outlier_cause"] == "none"]
rows = []
for name, col in [("weekday", "dow"), ("mahine ki tareekh", "dom")]:
    a = A.groupby(col)["resid"].apply(eff)
    b = normal.groupby(col)["resid"].apply(eff)
    rows.append({"pattern": name, "corr (saari rows vs extremes ke baghair)": a.corr(b),
                 "max farq (% points)": (a - b).abs().max()})
m_all = A.groupby("month")["wd_resid"].apply(eff)
stab = pd.DataFrame(rows)
print(f"Extremes (Phase 03) ki rows: {(A['outlier_cause'] != 'none').sum():,} ({(A['outlier_cause'] != 'none').mean():.2%})")
display(stab.round(3))

# ------------------------------------------------------------------------------

# Test 2: kya pattern dono periods (saalon) mein wahi hai?
A["period"] = np.where(A["date"] < "2016-06-01", "Jun2015-May2016", "Jun2016-Jul2017")
A["resid_nodow"] = A["resid"] - A.groupby("dow")["resid"].transform("mean")      # weekday ka asar hata kar
p_dow = A.groupby(["period", "dow"])["resid"].apply(eff).unstack(0)
p_dom = A.groupby(["period", "dom"])["resid_nodow"].apply(eff).unstack(0)
print(f"Weekday pattern: period 1 vs period 2 corr = {p_dow.corr().iloc[0, 1]:.3f}")
print(f"Tareekh pattern (weekday hata kar): period 1 vs period 2 corr = {p_dom.corr().iloc[0, 1]:.3f}")
p_dow.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
display(p_dow.round(1).T)

# ----------------------------------------------------------------------------------

dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
a, b = A.groupby("dow")["resid"].apply(eff), normal.groupby("dow")["resid"].apply(eff)
x = np.arange(7)
axes[0].bar(x - 0.2, a, 0.4, color=BLUE, label="saari rows")
axes[0].bar(x + 0.2, b, 0.4, color=LIGHT_GRAY, label="extremes ke baghair")
axes[0].set_xticks(x, dow_names)
axes[0].set_title("Weekday ka asar (local level se farq)")
axes[0].yaxis.set_major_formatter(pctfmt)
axes[0].axhline(0, color=GRAY, linewidth=1)
axes[0].legend()
a, b = A.groupby("dom")["resid"].apply(eff), normal.groupby("dom")["resid"].apply(eff)
axes[1].plot(a.index, a, color=BLUE, marker="o", markersize=3, label="saari rows")
axes[1].plot(b.index, b, color=GRAY, linestyle="--", label="extremes ke baghair")
axes[1].axhline(0, color=GRAY, linewidth=1)
axes[1].set_title("Mahine ki tareekh ka asar")
axes[1].set_xlabel("tareekh")
axes[1].yaxis.set_major_formatter(pctfmt)
axes[1].legend()
save(fig, "01_patterns_with_without_extremes")


# -------------------------------------------------------------------------

## 3. Sales history features: kaun se din ki purani sales sab se zyada batati hain?
# **Leakage ka usool (Phase 01b se):** forecast ke din (15 Aug) system ko 16–31 Aug batana hai. 31 Aug ke liye sab se taaza maloom din 15 Aug hai, yani **16 din** purana. Is liye har sales-history feature **kam se kam 16 din purana** hoga.

# **Candidate features:**

# | Feature | Matlab |
# |---|---|
# | `lag_16` | 16 din pehle ki sales (sab se taaza mumkin) |
# | `lag_21, lag_28, lag_35, lag_42` | 3, 4, 5, 6 hafte pehle, yani **usi weekday** ki sales |
# | `lag_364` | 52 hafte pehle (pichle saal, usi weekday) |
# | `roll7 / roll14 / roll28 / roll56 (lag16)` | 16 din pehle khatam hone wale 7/14/28/56 din ki average |
# | `same_wd_mean4` | lag 21/28/35/42 ki average (usi weekday ka taaza normal) |
# | `ewm28 (lag16)` | Exponentially weighted average (naye din ko zyada wazan) |

# **Naapne ke do tareeqe:**
# - **Global correlation:** saari series ek sath. Yeh zyada tar "bara store bara bechta hai" pakarta hai, jo aasan hai.
# - **Within-series correlation:** har series ka apna average hata kar. Yeh batata hai ke feature **usi series ke andar** din-ba-din ka utaar-charhao kitna samjhata hai. **Asal imtihaan yahi hai.**


# ### 3a. Pehle: lag ka source kaun sa din "bharose ka" hai?
# Band store, pre-opening aur family-unavailable din `is_trainable == 0` hain. Wahan ki value demand nahi hai (Section 5 mein saboot). Lekin 2014–15 ke **partial-recording** din alag masla hain. Un dinon mein kuch families ka data **kam likha gaya** tha. Sawal yeh hai: **kaun si families?** Andaze se saari families ko hata dena sahi data bhi phenk dega.

# **Test (har family alag):** `lag_364` ki value aur asli target ka farq (log) dekho:
# - jab source din partial tha, aur
# - jab source din normal tha.

# Agar family ka data theek likha gaya tha, to dono farq barabar honge. Agar kam likha gaya tha, to partial wala farq bohat manfi hoga.


hist["y_raw"] = np.where(hist["is_trainable"] == 1, np.log1p(hist["sales_clean"]), np.nan)
gr = hist.groupby(KEYS)
t364 = hist[KEYS + ["date"]].assign(lag364_raw=gr["y_raw"].shift(364),
                                    src_partial=gr["date"].shift(364).isin(PARTIAL_DATES))
G = A[KEYS + ["date", "y"]].merge(t364, on=KEYS + ["date"]).dropna(subset=["lag364_raw"])
G["gap"] = G["lag364_raw"] - G["y"]
fam_gap = G.groupby(["family", "src_partial"])["gap"].mean().unstack().rename(
    columns={False: "farq (normal source)", True: "farq (partial source)"})
fam_gap["partial - normal"] = fam_gap["farq (partial source)"] - fam_gap["farq (normal source)"]
GAP_THR = -0.30          # log mein -0.30 = ~26% kam. Baqi families -0.15 se +0.36 ke beech hain (neeche table)
UNDER_FAMS = fam_gap.index[fam_gap["partial - normal"] < GAP_THR].tolist()
print("Partial dinon mein KAM likhi gayi families:", UNDER_FAMS)
print("(NaN wali families: partial dinon mein un ka data tha hi nahi, woh pehle se is_trainable = 0 hain)")
display(fam_gap.sort_values("partial - normal").round(2))


# ------------------------------------------------------------------

# Lag source value: sirf "asli demand" wale din.
# Band store / pre-launch / family-unavailable = NaN; partial-recording din = NaN SIRF un families ke liye jo kam likhi gayin.
src_ok = (hist["is_trainable"] == 1) & ~(hist["date"].isin(PARTIAL_DATES) & hist["family"].isin(UNDER_FAMS))
old_ok = (hist["is_trainable"] == 1) & ~hist["date"].isin(PARTIAL_DATES)          # pichla version: saari families
for name, ok in [("pichla rule (saari families)", old_ok), ("naya rule (sirf kam likhi families)", src_ok)]:
    av = ok.groupby([hist["store_nbr"], hist["family"]]).shift(364).astype("boolean").fillna(False)
    av = hist.loc[av.astype(bool).values, KEYS + ["date"]].merge(A[KEYS + ["date"]], on=KEYS + ["date"])
    print(f"lag_364 availability, {name}: {len(av) / len(A):.1%}")
hist["y_src"] = np.where(src_ok, np.log1p(hist["sales_clean"]), np.nan)
hist["y_src_zero"] = np.log1p(hist["sales_clean"])               # Section 5 ke test ke liye (band din = 0)
g = hist.groupby(KEYS)["y_src"]
for L in (16, 21, 28, 35, 42, 364):
    hist[f"lag_{L}"] = g.shift(L)
s16 = g.shift(16)
grp16 = s16.groupby([hist["store_nbr"], hist["family"]])
for w in (7, 14, 28, 56):
    hist[f"roll{w}_lag16"] = grp16.transform(lambda s: s.rolling(w, min_periods=max(2, w // 2)).mean())
hist["ewm28_lag16"] = grp16.transform(lambda s: s.ewm(halflife=14, min_periods=7, ignore_na=True).mean())
hist["same_wd_mean4"] = hist[["lag_21", "lag_28", "lag_35", "lag_42"]].mean(axis=1)

HISTORY = ["lag_16", "lag_21", "lag_28", "lag_35", "lag_42", "lag_364", "roll7_lag16", "roll14_lag16",
           "roll28_lag16", "roll56_lag16", "ewm28_lag16", "same_wd_mean4"]
A = A.merge(hist[KEYS + ["date"] + HISTORY], on=KEYS + ["date"], how="left")

def demean(df, cols):
    return df[cols] - df.groupby(KEYS)[cols].transform("mean")

def within_corr(df, x, y="y"):
    d = df[KEYS + [x, y]].dropna()
    dm = demean(d, [x, y])
    return dm[x].corr(dm[y])

hc = pd.DataFrame({
    "global corr": [A[[f, "y"]].dropna().corr().iloc[0, 1] for f in HISTORY],
    "within-series corr": [within_corr(A, f) for f in HISTORY],
    "available %": [A[f].notna().mean() * 100 for f in HISTORY]}, index=HISTORY)
display(hc.sort_values("within-series corr", ascending=False).round(3))

# ----------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 4.4))
o = hc.sort_values("within-series corr")
ax.barh(o.index, o["global corr"], color=LIGHT_GRAY, label="global (saari series ek sath)")
ax.barh(o.index, o["within-series corr"], color=BLUE, height=0.5, label="within-series (asal imtihaan)")
ax.set_xlim(0, 1)
ax.set_title("History features ka target se rishta (log scale, correlation)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
save(fig, "02_history_feature_correlation")


# --------------------------------------------------------------------

# ## 4. History features ki aapas mein overlap (redundancy)
# Agar do features taqreeban ek jaise hon, to doosra model ko kuch naya nahi deta. Bas training dheemi hoti hai aur importance do jagah bat jati hai. Neeche **within-series** correlation hai (series ka average hata kar), taa ke "bara store" wala asar na ho.

# Phir **incremental test**: ek mazboot base (`roll28_lag16` + `same_wd_mean4`) ke baad har feature mein **bachi hui** information kitni hai (partial correlation)?

D = A[KEYS + ["y"] + HISTORY].dropna(subset=[f for f in HISTORY if f != "lag_364"])
DM = demean(D, ["y"] + HISTORY)
cm = DM[HISTORY].corr()
fig, ax = plt.subplots(figsize=(8.5, 7))
im = ax.imshow(cm.values, cmap=DIVERGING, vmin=-1, vmax=1)
ax.set_xticks(range(len(HISTORY)), HISTORY, rotation=60, ha="right", fontsize=8)
ax.set_yticks(range(len(HISTORY)), HISTORY, fontsize=8)
for i in range(len(HISTORY)):
    for j in range(len(HISTORY)):
        ax.text(j, i, f"{cm.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                color="white" if abs(cm.values[i, j]) > 0.75 else INK)
ax.grid(False)
ax.set_title("History features ki aapas mein correlation (within-series)")
fig.colorbar(im, ax=ax, shrink=0.7)
save(fig, "03_history_feature_redundancy")


# -------------------------------------------------------

def partial_corr(dm, x, controls, y="y"):
    d = dm[[x, y] + controls].dropna()
    Z = np.column_stack([np.ones(len(d))] + [d[c].values for c in controls])
    rx = d[x].values - Z @ np.linalg.lstsq(Z, d[x].values, rcond=None)[0]
    ry = d[y].values - Z @ np.linalg.lstsq(Z, d[y].values, rcond=None)[0]
    return np.corrcoef(rx, ry)[0, 1]

BASE = ["roll28_lag16", "same_wd_mean4"]
inc = pd.Series({f: partial_corr(DM, f, BASE) for f in HISTORY if f not in BASE}).sort_values(ascending=False)
display(inc.round(3).to_frame("base ke baad bachi information (partial corr)"))

# -----------------------------------------------------------

## 5. Lag sources mein band ya partial din: 0 rakhein ya khaali (NaN)?
# Agar 16 din pehle store band tha (1 Jan, renovation), to us din ki sales **0** hai. Lekin yeh demand nahi thi, store ki majboori thi. Isi tarah 2014–15 ke partial-recording mahinon ki values **kam likhi gayi** thin, aur `lag_364` wahan se value uthata hai.

# **Do options:**
# - **0 rakho:** model ko lagega ke us din demand zero thi.
# - **NaN rakho:** model ko pata hoga ke "yeh din maloom nahi". Tree models (LightGBM) NaN ko khud sambhal lete hain.

# **Test:** jin rows ke lag source mein aisa din tha, wahan dono versions ka target se rishta compare karo.


gz = hist.groupby(KEYS)["y_src_zero"]
tmp = hist[KEYS + ["date"]].copy()
for L in (16, 364):
    tmp[f"lag_{L}_zero"] = gz.shift(L)
    tmp[f"src_bad_{L}"] = (~src_ok).groupby([hist["store_nbr"], hist["family"]]).shift(L).astype("boolean").fillna(False).astype(bool)
Z = A[KEYS + ["date", "y", "lag_16", "lag_364"]].merge(tmp, on=KEYS + ["date"])
rows = []
for L in (16, 364):
    bad = Z[Z[f"src_bad_{L}"]]
    rows.append({"lag": f"lag_{L}", "rows jin ka source band/partial din tha": len(bad),
                 "un mein band/kam-likhi value ki average (log)": bad[f"lag_{L}_zero"].mean(),
                 "un rows ka asli target average (log)": bad["y"].mean(),
                 "baqi rows: lag vs target average farq": (Z.loc[~Z[f"src_bad_{L}"], f"lag_{L}"] - Z.loc[~Z[f"src_bad_{L}"], "y"]).mean()})
display(pd.DataFrame(rows).round(3))


# ----------------------------------------------------------

## 6. Calendar: weekday, payday, mahina
# Calendar features **hamesha pehle se maloom** hote hain, is liye leakage ka koi khatra nahi. Sawal sirf yeh hai ke asar kitna mazboot hai aur kis shakal mein dena chahiye.

dow_eff = A.groupby("dow")["resid"].apply(eff)
# Payday: weekday ka asar hata kar (resid - us weekday ka average resid)
A["resid_nodow"] = A["resid"] - A.groupby("dow")["resid"].transform("mean")
dom_eff = A.groupby("dom")["resid_nodow"].apply(eff)
dom_n = A.drop_duplicates("date").groupby("dom").size()
print("Weekday asar (%):", dow_eff.round(1).to_dict())
print("\nTareekh asar, weekday hata kar (%):")
print(dom_eff.round(1).to_frame("asar %").T.to_string())


# -------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10.5, 3.8))
ax.bar(dom_eff.index, dom_eff, color=[ORANGE if d in (1, 2, 15, 16, 30, 31) else BLUE for d in dom_eff.index])
ax.axhline(0, color=GRAY, linewidth=1)
ax.set_xticks(range(1, 32))
ax.set_xlabel("mahine ki tareekh (narangi = 1, 2, 15, 16, 30, 31)")
ax.yaxis.set_major_formatter(pctfmt)
ax.set_title("Payday ka asar: weekday hata kar, local level se farq")
save(fig, "04_payday_effect")

# -----------------------------------------------------------------

# Mahine ka pattern: kya har saal dohrata hai? Company-level, har mahine ka "pichle mahine ke muqable" farq
# Sirf purane stores (clean period se kafi pehle khule), aur "har khule store ki average" -- taa ke naye stores
# ka khulna ya renovation ke liye band hona "mahine ka asar" na lage.
open_dates = master.drop_duplicates("store_nbr").set_index("store_nbr")["store_open_date"]
OLD_STORES = open_dates[open_dates < CLEAN_START - pd.Timedelta(days=60)].index
sd_ = A[A["store_nbr"].isin(OLD_STORES)].groupby(["store_nbr", "date"])["sales_clean"].sum().reset_index()
print(f"Purane stores: {len(OLD_STORES)} (naye stores bahar: {sorted(set(A['store_nbr']) - set(OLD_STORES))})")
comp = sd_.groupby("date")["sales_clean"].mean()            # us din har khule store ki average sales
mon = comp.groupby(comp.index.to_period("M")).mean()
mon = mon[mon.index >= pd.Period("2015-06", "M")]
chg = (np.log(mon) - np.log(mon.shift(1))).dropna() * 100
tab = pd.DataFrame({"chg": chg.values, "year": chg.index.year, "month": chg.index.month})
piv = tab.pivot(index="month", columns="year", values="chg")
print("Mahine-dar-mahina farq (%), har saal alag:")
display(piv.round(1))
both_15_16 = piv[[2015, 2016]].dropna()
both_16_17 = piv[[2016, 2017]].dropna()
print(f"2015 vs 2016 (mahine {list(both_15_16.index)}): corr = {both_15_16.corr().iloc[0, 1]:.2f}")
print(f"2016 vs 2017 (mahine {list(both_16_17.index)}): corr = {both_16_17.corr().iloc[0, 1]:.2f}")


# -----------------------------------------------------------------------

## 7. Holidays: kis qisam ka asar, aur kya har store par ek jaisa?
# Phase 01 ne holidays ko alag flags mein toda tha. Yahan har flag ka asar **weekday normal** se naapa gaya hai (usi weekday ke aas paas ke hafton ki median), taa ke Monday holiday ko Monday hi se compare kiya jaye.

flags = ["is_national_holiday", "is_regional_holiday", "is_local_holiday", "is_holiday_season_day",
         "is_workday_override", "is_national_event", "is_earthquake_period"]
hrows = []
for f in flags:
    sub = A[A[f] == 1]
    hrows.append({"flag": f, "din": sub["date"].nunique(), "rows": len(sub), "asar %": eff(sub["wd_resid"]),
                  "median asar %": (np.exp(sub["wd_resid"].median()) - 1) * 100})
base_days = A[(A[flags].sum(axis=1) == 0)]
hrows.append({"flag": "(koi flag nahi)", "din": base_days["date"].nunique(), "rows": len(base_days),
              "asar %": eff(base_days["wd_resid"]), "median asar %": (np.exp(base_days["wd_resid"].median()) - 1) * 100})
display(pd.DataFrame(hrows).set_index("flag").round(2))

# ----------------------------------------------------------------------

# National holiday ka asar HAR STORE par alag alag
nh = A[(A["is_national_holiday"] == 1) & (A["is_workday_override"] == 0)]
per_store = nh.groupby("store_nbr")["wd_resid"].apply(eff).sort_values()
meta = master.drop_duplicates("store_nbr").set_index("store_nbr")[["city", "store_type"]]
fig, ax = plt.subplots(figsize=(10.5, 4))
ax.bar(range(len(per_store)), per_store.values, color=[RED if v < 0 else BLUE for v in per_store.values])
ax.set_xticks(range(len(per_store)), per_store.index, fontsize=6.5, rotation=90)
ax.axhline(0, color=GRAY, linewidth=1)
ax.yaxis.set_major_formatter(pctfmt)
ax.set_xlabel("store_nbr")
ax.set_title("National holiday ka asar har store par (weekday normal se farq)")
save(fig, "05_holiday_effect_by_store")
print(f"Stores: {len(per_store)} | asar ki range: {per_store.min():.1f}% se {per_store.max():.1f}% | "
      f"manfi asar wale stores: {(per_store < 0).sum()}")
out = pd.concat([per_store.head(5), per_store.tail(5)]).to_frame("asar %").join(meta)
display(out.round(1))

# ---------------------------------------------------------------------

# Kya store ka holiday-rawaiya har saal wahi rehta hai? (2015-16 vs 2016-17 ke holidays)
nh = nh.assign(period=np.where(nh["date"] < "2016-06-01", "Jun2015-May2016", "Jun2016-Jul2017"))
p = nh.groupby(["store_nbr", "period"])["wd_resid"].apply(eff).unstack().dropna()
print(f"Stores jin ke dono periods mein holidays hain: {len(p)} | "
      f"store holiday-asar ki correlation (period 1 vs period 2): {p.corr().iloc[0, 1]:.2f}")

# -------------------------------------------------------------------------

## 8. Nayi calendar khojen: Carnaval Sunday, holiday se pehle/baad ke din
# Phase 02/03 mein kuch din mile thay jo holidays file mein nahi: **Carnaval se pehle ka Sunday**, aur **holiday se pehle/baad ke Sundays**. Yahan in ka asar company level par naapa gaya hai: har din ke liye saari series ka average `wd_resid`, phir in khaas dinon ka average.

day = A.groupby("date").agg(wd_resid=("wd_resid", "mean"), nat=("is_national_holiday", "max"),
                            wko=("is_workday_override", "max"))
day["dow"] = day.index.dayofweek
nat_days = set(pd.DatetimeIndex(master.loc[(master["is_national_holiday"] == 1) & (master["is_workday_override"] == 0), "date"].unique()))
carn = master.loc[master["national_desc"].astype(str).str.contains("Carnaval"), "date"].drop_duplicates()
carn_sun = set(carn[carn.dt.dayofweek == 0] - pd.Timedelta(days=1))
SEASON_DAYS = pd.DatetimeIndex(master.loc[master["is_holiday_season_day"] == 1, "date"].unique())
day["before_holiday"] = [(d + pd.Timedelta(days=1)) in nat_days and d not in nat_days for d in day.index]
day["after_holiday"] = [(d - pd.Timedelta(days=1)) in nat_days and d not in nat_days for d in day.index]
day["carnaval_sunday"] = day.index.isin(list(carn_sun))
rows = []
for name, mask in [("Carnaval se pehle ka Sunday", day["carnaval_sunday"]),
                   ("Holiday se pehle ka din (sab)", day["before_holiday"] & ~day["carnaval_sunday"]),
                   ("  - jab woh Sunday ho", day["before_holiday"] & ~day["carnaval_sunday"] & (day["dow"] == 6)),
                   ("  - jab woh Sunday na ho", day["before_holiday"] & (day["dow"] != 6)),
                   ("  - Sunday na ho, Christmas-season (24/31 Dec) ke baghair",
                    day["before_holiday"] & (day["dow"] != 6) & ~day.index.isin(SEASON_DAYS)),
                   ("Holiday ke baad ka din", day["after_holiday"]),
                   ("Aam din (in mein se koi nahi, holiday nahi)", ~(day["before_holiday"] | day["after_holiday"] |
                                                                   day["carnaval_sunday"] | (day["nat"] == 1)))]:
    sub = day[mask]
    rows.append({"din ki qisam": name, "kitne din": len(sub), "average asar %": (np.exp(sub["wd_resid"].mean()) - 1) * 100,
                 "kitne din manfi": f"{(sub['wd_resid'] < 0).sum()}/{len(sub)}"})
display(pd.DataFrame(rows).round(1))

# -----------------------------------------------------------------------------

pre_sun = day[day["before_holiday"] & (day["dow"] == 6)].copy()
names = master.drop_duplicates("date").set_index("date")["national_desc"].astype(str)
pre_sun["agle din ka holiday"] = [names.get(d + pd.Timedelta(days=1), "") for d in pre_sun.index]
pre_sun["asar %"] = (np.exp(pre_sun["wd_resid"]) - 1) * 100
display(pre_sun[["agle din ka holiday", "asar %"]].round(1))

# --------------------------------------------------------------------------

## 9. Promotion: asar, shakal, aur family ke hisaab se farq
# `onpromotion` = us din, us store mein, us family ki kitni items discount par thin. Yeh **test file mein bhi maujood** hai (company pehle se plan karti hai), is liye forecast ke waqt maloom hota hai. Leakage nahi.

# **Mushkil:** GROCERY I mein 50 items promo par hona aam baat hai, jab ke BOOKS mein 1 bhi kam hi hota hai. Is liye "kitni items" ke sath "**apne normal ke muqable kitni**" bhi dekhna zaroori hai.

warnings.filterwarnings("ignore", category=RuntimeWarning)     # kuch families mein promo kabhi nahi badalta (corr = NaN)
P = A[KEYS + ["date", "y", "onpromotion", "resid", "wd_resid"]].copy()
P["promo_log"] = np.log1p(P["onpromotion"])
print(f"Within-series corr: log1p(onpromotion) vs y = {within_corr(P, 'promo_log'):.3f}")
# Series ka "normal" promo: PICHLE 28 din ki median (leakage-safe, kyun ke promo pehle se maloom hai)
hp = hist[KEYS + ["date", "onpromotion"]].copy()
hp["promo_med28"] = hp.groupby(KEYS)["onpromotion"].transform(lambda s: s.shift(1).rolling(28, min_periods=7).median())
P = P.merge(hp[KEYS + ["date", "promo_med28"]], on=KEYS + ["date"], how="left")
P["promo_ratio"] = (P["onpromotion"] + 1) / (P["promo_med28"] + 1)
bins = [0, 0.5, 0.9, 1.1, 1.5, 2, 3, np.inf]
labels = ["<0.5x", "0.5-0.9x", "~1x (normal)", "1.1-1.5x", "1.5-2x", "2-3x", ">3x"]
P["ratio_bin"] = pd.cut(P["promo_ratio"], bins, labels=labels, right=False)
curve = P.groupby("ratio_bin", observed=True).agg(rows=("wd_resid", "size"), asar=("wd_resid", eff))
print(f"Within-series corr: log(promo_ratio) vs y = {within_corr(P.assign(lr=np.log(P['promo_ratio'])), 'lr'):.3f}")
display(curve.round(1))

# -------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
axes[0].bar(curve.index.astype(str), curve["asar"], color=[RED if v < 0 else BLUE for v in curve["asar"]])
axes[0].axhline(0, color=GRAY, linewidth=1)
axes[0].set_title("Promo apne normal ke muqable vs sales asar")
axes[0].set_xlabel("aaj ka promo / pichle 28 din ki median")
axes[0].yaxis.set_major_formatter(pctfmt)
axes[0].tick_params(axis="x", rotation=30)
# Family ke hisaab se: within-series corr
Pr = P.dropna(subset=["promo_ratio"]).assign(lr=lambda d: np.log(d["promo_ratio"]))
fam = pd.Series({f: within_corr(d, "lr") for f, d in Pr.groupby("family")}).dropna().sort_values()
axes[1].barh(fam.index, fam.values, color=[RED if v < 0 else BLUE for v in fam.values])
axes[1].axvline(0, color=GRAY, linewidth=1)
axes[1].tick_params(axis="y", labelsize=7)
axes[1].set_title("Family ke hisaab se promo ka rishta (within-series corr)")
save(fig, "06_promotion_effect")
print(f"Families jahan rishta manfi: {list(fam[fam < 0].index)}")
print(f"Sab se mazboot: {fam.tail(3).round(2).to_dict()}")

# ---------------------------------------------------------------------------

## 10. Store aur family ki static information
# `store_type`, `store_cluster`, `city`, `state` sab **store_nbr se tay** hote hain. Har store ki ek hi type, ek hi city hoti hai. To kya yeh faaltu hain?

static = master.drop_duplicates("store_nbr")[["store_nbr", "store_type", "store_cluster", "city", "state", "store_open_date"]]
print("Har store ke liye unique values (1 = store_nbr se poori tarah tay):")
print(master.groupby("store_nbr")[["store_type", "store_cluster", "city", "state"]].nunique().max().to_string())
print(f"\nUnique: types {static['store_type'].nunique()}, clusters {static['store_cluster'].nunique()}, "
      f"cities {static['city'].nunique()}, states {static['state'].nunique()}")
# Naye store ke liye: store 52 jaisa store kitni history rakhta hai?
s52 = static.set_index("store_nbr").loc[52]
print(f"\nStore 52: type {s52['store_type']}, cluster {s52['store_cluster']}, city {s52['city']}, "
      f"open {s52['store_open_date'].date()} -> training mein sirf {(TRAIN_END - s52['store_open_date']).days + 1} din")
# Kya same type/cluster ke stores ki family-mix milti hai? (store 52 ke liye "padosi" se seekhna)
mix = A.groupby(["store_nbr", "family"])["sales_clean"].sum().unstack(fill_value=0)
mix = mix.div(mix.sum(axis=1), axis=0)
cl = static.set_index("store_nbr")["store_cluster"]
same_cl, diff_cl = [], []
st = mix.index.tolist()
for i in range(len(st)):
    for j in range(i + 1, len(st)):
        c = np.corrcoef(mix.loc[st[i]], mix.loc[st[j]])[0, 1]
        (same_cl if cl[st[i]] == cl[st[j]] else diff_cl).append(c)
print(f"Family-mix ki similarity: same cluster {np.mean(same_cl):.3f} vs alag cluster {np.mean(diff_cl):.3f}")

# -------------------------------------------------------------------------------

## 11. Group-level history: family aur store ka kul rujhan
# Pre-EDA mein series ka aapas mein co-movement median ~0.40 tha. Yani ek family saare stores mein ek sath upar-neeche hoti hai. To kya **poori family ki** (saare stores) pichli sales, ek series ki apni pichli sales ke baad bhi kuch naya batati hai?

# Features (sab 16+ din purane):
# - `fam_roll28_lag16`: is family ki saare stores mein average (log).
# - `store_roll28_lag16`: is store ki saari families ka average (log).
# - `fam_trend`: family ka 7-din / 28-din ka ratio (kya family abhi upar ja rahi hai?).

hist["fam_day"] = hist.groupby(["family", "date"])["y_src"].transform("mean")
hist["store_day"] = hist.groupby(["store_nbr", "date"])["y_src"].transform("mean")
fd = hist.drop_duplicates(["family", "date"])[["family", "date", "fam_day"]].sort_values(["family", "date"])
fd["fam_roll28_lag16"] = fd.groupby("family")["fam_day"].transform(lambda s: s.shift(16).rolling(28, min_periods=14).mean())
fd["fam_roll7_lag16"] = fd.groupby("family")["fam_day"].transform(lambda s: s.shift(16).rolling(7, min_periods=4).mean())
fd["fam_trend"] = fd["fam_roll7_lag16"] - fd["fam_roll28_lag16"]
sd = hist.drop_duplicates(["store_nbr", "date"])[["store_nbr", "date", "store_day"]].sort_values(["store_nbr", "date"])
sd["store_roll28_lag16"] = sd.groupby("store_nbr")["store_day"].transform(lambda s: s.shift(16).rolling(28, min_periods=14).mean())
A = (A.merge(fd[["family", "date", "fam_roll28_lag16", "fam_trend"]], on=["family", "date"], how="left")
      .merge(sd[["store_nbr", "date", "store_roll28_lag16"]], on=["store_nbr", "date"], how="left"))
A["own_trend"] = A["roll7_lag16"] - A["roll28_lag16"]
GROUP = ["fam_roll28_lag16", "store_roll28_lag16", "fam_trend", "own_trend"]
D2 = A[KEYS + ["y"] + BASE + GROUP].dropna()
DM2 = demean(D2, ["y"] + BASE + GROUP)
display(pd.DataFrame({"within-series corr": [DM2[f].corr(DM2["y"]) for f in GROUP],
              "base ke baad bachi information": [partial_corr(DM2, f, BASE) for f in GROUP]}, index=GROUP).round(3))


# -----------------------------------------------------------

# ## 12. Oil aur transactions: rakhein ya nahi?
# **Oil:** Pre-EDA mein oil price ka sales se level par correlation −0.77 tha, lekin daily change par sirf 0.02. Matlab yeh rishta shayad sirf **dono ke trend** ki wajah se tha (oil girta raha, sales barhti rahin). Yahan **within-series** aur **detrended** tareeqe se test hai.

# **Transactions:** test file mein transactions **nahi** hain. Forecast ke waqt aaj ke customers maloom nahi hote, is liye same-day transactions feature nahi ban sakte. Sirf purani (16+ din) transactions mumkin hain. Sawal yeh hai ke kya woh store ki purani sales se zyada kuch batati hain.

oil = master.drop_duplicates("date").set_index("date")["oil_price"].sort_index()
oil_d = pd.DataFrame({"oil": np.log(oil)})
oil_d["oil_chg28"] = oil_d["oil"] - oil_d["oil"].shift(28)
oil_d["oil_lag16"] = oil_d["oil"].shift(16)
O = A[KEYS + ["date", "y", "resid"]].merge(oil_d, left_on="date", right_index=True, how="left")
comp_day = A.groupby("date")["resid"].mean()
od = oil_d.join(comp_day.rename("comp_resid"), how="inner")
print(f"Within-series corr: log(oil) vs y            = {within_corr(O, 'oil'):.3f}   (trend shaamil)")
print(f"Corr: oil 28-din change vs company residual  = {od['oil_chg28'].corr(od['comp_resid']):.3f}   (trend hata kar)")
O["t"] = (O["date"] - O["date"].min()).dt.days
DMo = demean(O.dropna(subset=["oil_lag16"]), ["y", "oil_lag16", "t"])
print(f"Partial corr: oil_lag16 vs y, series level aur time trend hata kar = {partial_corr(DMo, 'oil_lag16', ['t']):.3f}")

# -------------------------------------------------------------

tx = store_tx[store_tx["date"] <= TRAIN_END].sort_values(["store_nbr", "date"]).copy()
tx["ltx"] = np.log1p(tx["transactions"])
tx["tx_roll28_lag16"] = tx.groupby("store_nbr")["ltx"].transform(lambda s: s.shift(16).rolling(28, min_periods=14).mean())
T = A[KEYS + ["date", "y"] + BASE + ["store_roll28_lag16"]].merge(tx[["store_nbr", "date", "tx_roll28_lag16"]],
                                                                  on=["store_nbr", "date"], how="left").dropna()
DMt = demean(T, ["y", "tx_roll28_lag16", "store_roll28_lag16"] + BASE)
print(f"Within-series corr (store ki 16+ din purani transactions vs store ki 16+ din purani sales): "
      f"{DMt['tx_roll28_lag16'].corr(DMt['store_roll28_lag16']):.3f}")
print(f"Partial corr: tx_roll28_lag16 vs y (base + store sales ke baad) = "
      f"{partial_corr(DMt, 'tx_roll28_lag16', BASE + ['store_roll28_lag16']):.3f}")
print("Test file mein 'transactions' column hai?", "transactions" in pd.read_csv("data/raw/test.csv", nrows=2).columns)

# --------------------------------------------------------------

## 13. Volatility aur zero-share: intermittent series ke liye
# Pre-EDA: ~16% series intermittent ya lumpy hain (aksar zero). Phase 03: kuch series mein spikes baar baar aate hain. In series ke liye **average** kaafi nahi. Model ko yeh bhi pata hona chahiye ke series kitni **be-tarteeb** hai aur kitni baar zero bechti hai. Dono 16+ din purani history se ban sakte hain.

hist["is_zero_src"] = np.where(src_ok, (hist["sales_clean"] == 0).astype(float), np.nan)
gzs = hist.groupby(KEYS)["is_zero_src"]
hist["zero_share28_lag16"] = gzs.transform(lambda s: s.shift(16).rolling(28, min_periods=14).mean())
hist["std28_lag16"] = grp16.transform(lambda s: s.rolling(28, min_periods=14).std())
A = A.merge(hist[KEYS + ["date", "zero_share28_lag16", "std28_lag16"]], on=KEYS + ["date"], how="left")
A["is_zero"] = (A["sales_clean"] == 0).astype(float)
Dz = A[KEYS + ["y", "is_zero", "zero_share28_lag16", "std28_lag16"] + BASE].dropna()
print(f"Corr: zero_share28_lag16 vs aaj zero hona (global) = {Dz['zero_share28_lag16'].corr(Dz['is_zero']):.3f}")
print(f"Corr: zero_share28_lag16 vs aaj zero hona (within-series, series ka average hata kar) = {within_corr(Dz, 'zero_share28_lag16', 'is_zero'):.3f}")
DMz = demean(Dz, ["y", "zero_share28_lag16", "std28_lag16"] + BASE)
print(f"Partial corr (base ke baad): zero_share28_lag16 vs y = {partial_corr(DMz, 'zero_share28_lag16', BASE):.3f}")
# Kya volatility error ka ishara deti hai? |y - roll28| (naive forecast ki ghalti) vs std28
Dz["naive_err"] = (Dz["y"] - Dz["roll28_lag16"]).abs()
print(f"Corr: std28_lag16 vs naive forecast ki ghalti (global) = {Dz['std28_lag16'].corr(Dz['naive_err']):.3f}")
q = pd.qcut(Dz["zero_share28_lag16"].rank(method="first"), 5, labels=["sab se kam", "2", "3", "4", "sab se zyada"])
display(Dz.groupby(q, observed=True).agg(zero_share=("zero_share28_lag16", "mean"), aaj_zero=("is_zero", "mean"),
                                 naive_err=("naive_err", "mean")).round(3))

# -----------------------------------------------------------------

## 14. Naye stores aur nayi series (store 52)
# Pre-EDA: naye stores ke pehle ~4 hafte baad ke muqable 9–47% zyada thay. Store 52 test ke shuru mein sirf ~118 din purana hoga, aur us ka `lag_364` hoga hi nahi. Sawal yeh hai ke kya "store kitne din purana hai" ka koi pattern hai.

opened = static[(static["store_open_date"] >= CLEAN_START - pd.Timedelta(days=60))]["store_nbr"].tolist()
print("Clean period ke qareeb/andar khulne wale stores:", opened)
st_day = A[A["store_nbr"].isin(opened)].groupby(["store_nbr", "date"])["sales_clean"].sum().reset_index()
st_day = st_day.merge(static[["store_nbr", "store_open_date"]], on="store_nbr")
st_day["age_week"] = ((st_day["date"] - st_day["store_open_date"]).dt.days // 7)
rel = []
for s, d in st_day.groupby("store_nbr"):
    ref = d.loc[d["age_week"].between(8, 15), "sales_clean"].mean()
    if np.isnan(ref):
        continue
    w = d[d["age_week"] < 16].groupby("age_week")["sales_clean"].mean() / ref
    rel.append(w.rename(s))
rel = pd.concat(rel, axis=1)
fig, ax = plt.subplots(figsize=(10, 3.8))
for s in rel.columns:
    ax.plot(rel.index, rel[s], marker="o", markersize=3, linewidth=1.3, label=f"store {s}")
ax.axhline(1, color=GRAY, linewidth=1)
ax.axvspan(-0.5, 3.5, color=ORANGE, alpha=0.08)
ax.set_xlabel("store khulne ke baad hafta")
ax.set_ylabel("sales / (hafta 8-15 ki average)")
ax.set_title("Naye stores: pehle hafton mein sales (hafta 8-15 = 1)")
ax.legend()
save(fig, "07_new_store_ramp")
display(rel.loc[0:5].round(2))

# -------------------------------------------------------------------

## 15. Validation/test window mein features ki availability
# Forecast ke waqt features mojood hone chahiye. Yahan sirf **features** dekhe gaye hain, validation ki sales nahi. History features ke liye sales sirf **15 Aug tak** maloom maani gayi hain (test ke liye), aur validation ke liye **30 Jul tak**.

def avail(cut, start, end):
    h = master[master["date"] <= end][KEYS + ["date", "sales_clean", "is_trainable"]].copy()   # Section 3a wala rule
    ok = (h["is_trainable"] == 1) & (h["date"] <= cut) & ~(h["date"].isin(PARTIAL_DATES) & h["family"].isin(UNDER_FAMS))
    h["ys"] = np.where(ok, np.log1p(h["sales_clean"]), np.nan)
    gg = h.groupby(KEYS)["ys"]
    for L in (16, 21, 364):
        h[f"lag_{L}"] = gg.shift(L)
    h["roll28_lag16"] = gg.shift(16).groupby([h["store_nbr"], h["family"]]).transform(
        lambda s: s.rolling(28, min_periods=14).mean())
    w = h[h["date"].between(start, end)]
    return w[["lag_16", "lag_21", "lag_364", "roll28_lag16"]].notna().mean() * 100

av = pd.DataFrame({"validation (31 Jul-15 Aug)": avail(TRAIN_END, pd.Timestamp("2017-07-31"), pd.Timestamp("2017-08-15")),
                   "test (16-31 Aug)": avail(pd.Timestamp("2017-08-15"), pd.Timestamp("2017-08-16"), pd.Timestamp("2017-08-31"))})
print("History features mojood (% rows):")
display(av.round(1))
win = master[master["split"].isin(["valid", "test"])]
print("Validation/test window mein calendar/promo features:")
print(win.groupby("split", observed=True)[["is_national_holiday", "is_local_holiday", "is_regional_holiday",
                                            "is_national_event", "is_holiday_season_day"]].sum().to_string())
print("onpromotion missing:", win["onpromotion"].isna().sum())

# ----------------------------------------------------------------------

## 16. Final feature plan (Phase 05 ke liye)
# Har feature ka faisla upar ke saboot se hai. **"Ablation"** ka matlab hai Phase 05 mein model ke sath aur baghair test, aur seed noise ke muqable faisla (Phase 03 ka sabaq).

plan = pd.DataFrame([
    ("store_nbr (category)", "static", "Store ka size 10x tak alag; holiday ka asar har store par ulta-seedha (Sec 7)", "Kabhi nahi badalta", "Keep"),
    ("family (category)", "static", "Promo ka asar family par alag (Sec 9); 01b ablation", "Kabhi nahi badalta", "Keep"),
    ("store_type, store_cluster, city", "static", "Store 52 (102 din) ke liye ummeed; family-mix 0.987 vs 0.944 (chhota farq) (Sec 10)", "Kabhi nahi badalta", "Keep"),
    ("state", "static", "City ka mota group; regional holiday ka flag alag se hai (Sec 10)", "Kabhi nahi badalta", "Ablation"),
    ("dow", "calendar", "Sat +23.5%, Sun +21.5%, Thu -12.9%; dono saal wahi (corr 0.997) (Sec 2, 6)", "Calendar pehle se maloom", "Keep"),
    ("dom", "calendar", "1 tareekh +18% (Jan/Dec hata kar bhi), mid-month -5%; dono saal corr 0.85 (Sec 2, 6)", "Calendar pehle se maloom", "Keep"),
    ("days_to_month_end", "calendar", "31 tareekh +6%, 26-28 tareekh -5%: mahine ka aakhir alag din par aata hai (Sec 6)", "Calendar pehle se maloom", "Ablation"),
    ("month", "calendar", "Dec +18%/+16.5%, Jan -13% dono saal (50 purane stores); Aug har saal alag (+1.2% / -2.6%) (Sec 6)", "Calendar pehle se maloom", "Ablation"),
    ("is_national_holiday", "holiday", "Average +16.8%, median +8.7% (Sec 7)", "Holidays file pehle se maloom", "Keep"),
    ("is_holiday_season_day", "holiday", "+28% (Christmas/New Year ke din) (Sec 7)", "Holidays file pehle se maloom", "Keep"),
    ("is_national_event, is_earthquake_period", "holiday", "Event +3%, zalzala +4.9%; April 2016 ko 'aam April' samajhne se bachata hai (Sec 6, 7)", "Holidays file pehle se maloom", "Keep"),
    ("is_local_holiday, is_regional_holiday", "holiday", "Median asar ~0; regional sirf 474 rows (Sec 7)", "Holidays file pehle se maloom", "Ablation"),
    ("is_workday_override", "holiday", "Clean period mein sirf 1 din (Sec 7)", "Holidays file pehle se maloom", "Ablation"),
    ("store_holiday_effect", "holiday (naya)", "Har store ka asar -51% se +94%, aur dono periods mein corr 0.94 (Sec 7)", "Sirf forecast se PEHLE ke holidays se banega", "Ablation"),
    ("is_day_before_holiday", "holiday (naya)", "Christmas ke baghair +5.9% (9 din, 3 manfi): kamzor (Sec 8)", "Holidays file pehle se maloom", "Ablation"),
    ("is_carnaval_sunday", "holiday (naya)", "-21.5% (clean period mein 2/2); forecast window mein nahi (Sec 8)", "Carnaval ki tareekh pehle se maloom", "Ablation"),
    ("onpromotion", "promotion", "Within-series corr 0.150; family ke hisaab se 0.45 tak (Sec 9)", "Test file mein maujood (plan pehle se)", "Keep"),
    ("promo_ratio28", "promotion (naya)", "Normal se kam -2 se -4%, zyada +5 se +7%, 1.5x ke baad flat (Sec 9)", "Pichle 28 din ka promo + aaj ka (dono maloom)", "Ablation"),
    ("same_wd_mean4", "history", "Within-series corr 0.345, sab se mazboot (Sec 3)", "Lag 21-42 (16+ din purane)", "Keep"),
    ("roll28_lag16", "history", "Base level; base ka hissa (Sec 3, 4)", "16+ din purana", "Keep"),
    ("lag_364", "history", "Base ke baad sab se zyada nayi info (partial 0.180); availability 87.4% (Sec 3a, 4)", "364 din purana; band source = NaN, partial din sirf 4 kam-likhi families ke liye NaN", "Keep"),
    ("roll7_lag16", "history", "Base ke baad partial 0.138: taaza level (Sec 4)", "16+ din purana", "Keep"),
    ("lag_16", "history", "Akele kamzor (0.108, alag weekday) lekin base ke baad 0.084 (Sec 3, 4)", "Sab se taaza mumkin (16 din)", "Keep"),
    ("lag_21", "history", "Within 0.290; base ke baad 0.066 (Sec 3, 4)", "16+ din purana", "Keep"),
    ("roll14_lag16, ewm28_lag16", "history", "roll7 jaisi info (partial 0.124 / 0.092) (Sec 4)", "16+ din purana", "Ablation"),
    ("lag_28", "history", "Base ke baad sirf 0.017 (Sec 4)", "16+ din purana", "Ablation"),
    ("lag_35, lag_42", "history", "Base ke baad -0.038 / -0.047: ulti simt ki info (trend), taqat lag_21 jitni (Sec 4)", "16+ din purana", "Ablation"),
    ("roll56_lag16", "history", "Base ke baad -0.016: zero ke qareeb (Sec 4)", "16+ din purana", "Drop"),
    ("own_trend, fam_trend", "history (naya)", "Base ke baad 0.128 / 0.110: rujhan (momentum) (Sec 11)", "16+ din purana", "Ablation"),
    ("fam_roll28_lag16", "history (group)", "Base ke baad -0.072: series ka family ke muqable level (ulti simt) (Sec 11)", "16+ din purana", "Ablation"),
    ("store_roll28_lag16", "history (group)", "Base ke baad 0.017: nayi info nahi (Sec 11)", "16+ din purana", "Drop"),
    ("zero_share28_lag16, std28_lag16", "history (naya)", "Aaj zero se within-series corr 0.202 (global 0.776 zyada tar series ki pehchaan); std vs naive ghalti 0.456 (Sec 13)", "16+ din purana", "Ablation"),
    ("days_since_open", "store", "Naye store pehle 2 hafte 1.2-2.3x; forecast window mein koi store ramp mein nahi (Sec 14)", "Opening date pehle se maloom", "Ablation"),
    ("oil_price (har shakal)", "external", "Trend hata kar corr 0.004-0.009 (Sec 12)", "-", "Drop"),
    ("transactions (har shakal)", "external", "Test mein nahi; purani tx ki base ke baad info -0.002 (Sec 12)", "-", "Drop"),
    ("dead series / pre-opening / closure din = 0", "rule", "Phase 01/01b", "Pehle se maloom", "Rule (feature nahi)"),
    ("is_trainable, is_store_closed, outlier_cause, is_outlier_treated, id, sales", "target se bane", "Sales se bane hain", "LEAKAGE: kabhi feature nahi", "Never"),

], columns=["feature", "group", "saboot (is notebook se)", "leakage-safe kyun", "faisla"])
plan.to_csv(REPORT_DIR / "04_feature_plan.csv", index=False)
print(f"Features: {len(plan)} | " + " | ".join(f"{k}: {v}" for k, v in plan["faisla"].value_counts().items()))
display(plan)


# ---------------------------------------------------------------

summary = {
    "pattern_stability": stab.round(4).to_dict(orient="records"),
    "history_features": hc.round(4).to_dict(orient="index"),
    "incremental_after_base": inc.round(4).to_dict(),
    "dow_effect_pct": dow_eff.round(2).to_dict(),
    "dom_effect_pct": dom_eff.round(2).to_dict(),
    "holiday_effect_by_store_range_pct": [round(float(per_store.min()), 2), round(float(per_store.max()), 2)],
    "promo_curve_pct": curve["asar"].round(2).to_dict(),
    "feature_plan_counts": plan["faisla"].value_counts().to_dict(),
}
json.dump(summary, open(REPORT_DIR / "04_post_eda_summary.json", "w"), indent=2, default=str)
print("Saved:", REPORT_DIR / "04_feature_plan.csv")
print("Saved:", REPORT_DIR / "04_post_eda_summary.json")

# ---------------------------------------------------------------------


# ## Khulasa
# **Phase 03 ke baad data ki halat:** koi value nahi badli. Pre-EDA ke patterns **dono saal dohrate** hain: weekday corr 0.997, payday corr 0.85. Yani yeh future ke liye bharose ke qabil hain.

# **Model ko kya chahiye (saboot ke sath):**
# 1. **Usi weekday ki purani sales** (`same_wd_mean4`, `lag_21`) sab se mazboot hain. Sab se taaza `lag_16` akela kamzor hai (alag weekday).
# 2. **Pichle saal ki sales (`lag_364`)** sab se zyada **nayi** info deti hai.
#    - Band store wale source din par value **NaN**, kyun ke wahan ka 0 jhoot bolta hai.
#    - Partial-recording din par NaN **sirf 4 families** ke liye (PRODUCE, BEVERAGES, SCHOOL AND OFFICE SUPPLIES, MAGAZINES), kyun ke data ne sirf inhein kam-likha dikhaya. Is se availability 72.6% se 87.4% hui.
# 3. **Weekday** (Sat +23%, Thu −13%) aur **mahine ki tareekh** (1 tareekh +18%, mid-month −5%) mazboot hain.
# 4. **Holiday ka asar har store par ulta-seedha** hai (−51% se +94%), aur har saal wahi (corr 0.94). Yeh is phase ki sab se kaam ki khoj hai.
# 5. **Promotion:** normal se zyada promo par +5–7%, phir flat. Asar family par bohat munhasir hai.
# 6. **Drop (saboot ke sath):** oil, transactions, `store_roll28`, `roll56`. **Manfi partial correlation wale features** (`lag_35`, `lag_42`, `fam_roll28`) Drop nahi, **Ablation** hain, kyun ke manfi bhi information hai.

# **Feature plan:** upar ki table aur `reports/04_feature_plan.csv`. "Keep" core features hain. "Ablation" wale Phase 05 mein model ke sath aur baghair test honge, **seed noise ke muqable** (Phase 03 ka sabaq).

# **Is version mein theek ki gayi ghaltiyan (review ke baad):**
# 1. `lag_364` masking saari families par thi, ab sirf data se sabit 4 families par hai.
# 2. Manfi partial correlation ko "kuch nahi" samjha gaya tha.
# 3. Mahine ka analysis naye stores se bigra tha, ab 50 purane stores ki average se hai.
# 4. `zero_share` ka global corr dikha kar ise zyada bara bataya tha, ab within-series (0.202) bhi hai.
# 5. Section 2 ka kamzor test akela tha, ab dono saal ka test bhi hai.
# 6. Kuch wazahatein bina label ke thin, ab "andaza" likha hai.

# **Limitations (imaandari):**
# - Correlations **rishta** batati hain, model ka asli faida nahi. Final faisla ablation karega.
# - Clean period sirf ~2 saal ka hai: har mahine ki 2 misaalen, aur holidays sirf 21 din.
# - Kuch wazahatein (Salinas samundar ka shehar, payday ki wajah, promo ki planning) **andaze** hain aur waise hi label ki gayi hain.
# - Partial-family threshold (−0.30) data ke farq se chuna gaya hai: 4 families −0.62 se neeche hain, baqi −0.15 se upar.

# **Phase 05 (Feature engineering) ke liye:** ek hi function mein saare features banana (leakage test ke sath, jaisa 01b mein). Phir "Keep" set ka base model, aur har "Ablation" group ko seed noise ke muqable test karna.