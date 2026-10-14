#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Retail Demand Forecasting System
Phase 02: Pre-EDA (outlier treatment se pehle data ko samajhna)
=================================================================

Project ka sequence
-------------------
01  01_data_preparation                  Files jorna, data quality checks, master dataset
02  02_pre_eda (yeh script)              Data ki nature samajhna: target, trend, seasonality, families,
                                         stores, promotions, holidays aur outliers ki pehli nazar
03  03_outliers_detection_and_treatment  Outliers ki wajah pehchanna, sirf ghaltiyon ka ilaaj
04  04_post_eda                          Treatment ke baad dobara EDA
05+ Feature engineering, modeling, evaluation, deployment

Maqsad
------
Forecast model banane se pehle data ki asli nature samajhna. Yahan KOI DATA BADLA NAHI JATA:
sirf dekha, naapa aur samjha jata hai. Treatment Phase 03 mein hoga.

EDA ke 3 usool
--------------
1. Sirf training period (2013-01-01 se 2017-07-30). Validation (31 Jul - 15 Aug 2017) aur test
   (16 - 31 Aug 2017) yahan nahi dekhe jate, warna faisle chupke se unpar fit ho jayenge (leakage).
2. Demand ka analysis sirf `is_trainable == 1` rows par. Company ki kul sales wale charts mein saari
   rows use hoti hain, kyunke wahan asli business picture dikhani hai.
3. Har section ke saath insight aur aage ke phases par asar likha gaya hai.

Inputs  : <processed-dir>/retail_demand_master.parquet
          <processed-dir>/store_day_transactions.parquet
          holidays_events.csv  (raw data folder)
Outputs : reports/figures/02_pre_eda/*.png         (numbered charts)
          reports/tables/02_pre_eda/*.csv          (har table ki CSV)
          reports/02_pre_eda_summary.json          (key numbers)
          reports/02_pre_eda_extreme_values.csv    (Phase 03 ke liye extreme rows)
          reports/02_pre_eda_conclusions.md        (business insights aur aage ke faisle)

Usage
-----
    python 02_pre_eda.py
    python 02_pre_eda.py --processed-dir processed --raw-dir data/raw --reports-dir reports
    python 02_pre_eda.py --show        # charts screen par bhi dikhao (default: sirf PNG save)

Sections
--------
0b Data overview | 0c Numeric inputs | 1 Target | 2 Sales trend | 2b Recording regimes |
3 Weekly pattern | 4 Monthly pattern | 5 Payday effect | 6 Families (+6b ADI/CV2) | 7 Stores |
8 Promotions | 9 Holidays, Christmas, earthquake | 10 Oil | 11 Transactions & basket |
12 Autocorrelation | 13 Stationarity | 14 Outliers first look | 15 Forecast window vs history |
16 Local/regional holidays | 17 New stores (store 52) | 18 Co-movement | 19 Stock-out check |
20 Summary & decisions
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)

logger = logging.getLogger("pre_eda")

# --------------------------------------------------------------------------------------
# Project constants
# --------------------------------------------------------------------------------------
CLEAN_START = pd.Timestamp("2015-06-01")      # Section 2b mein saboot ke sath
PARTIAL_MIN_FAMILIES = 5                      # itni families gayab hon to din "partial recording" hai
HORIZON_DAYS = 16                             # forecast horizon
DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# --------------------------------------------------------------------------------------
# Chart style: ek jaisa, saaf aur professional (grid lines bilkul nahi)
# --------------------------------------------------------------------------------------
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
LIGHT_BLUE = "#86b6ef"
GRAY, LIGHT_GRAY = "#9a9994", "#d9d8d4"
INK, INK2 = "#0b0b0b", "#52514e"
DIVERGING = LinearSegmentedColormap.from_list("div", [BLUE, "#f0efec", RED])

thousands = mtick.FuncFormatter(lambda x, _: f"{x:,.0f}")
pct = mtick.PercentFormatter(decimals=0)


def apply_plot_style() -> None:
    """Saare charts ke liye global matplotlib style (grid band, legend frameless)."""
    plt.rcParams.update({
        "figure.figsize": (10, 4.2), "figure.dpi": 110, "savefig.dpi": 150,
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.labelsize": 10, "axes.labelcolor": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#c8c7c2", "axes.linewidth": 0.8,
        "axes.grid": False, "axes.axisbelow": True,                      # NO grid lines anywhere
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "text.color": INK, "legend.frameon": False, "legend.fontsize": 9, "legend.title_fontsize": 9,
        "lines.linewidth": 2, "lines.solid_capstyle": "round",
    })


# --------------------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------------------
class Outputs:
    """Charts, tables aur reports ko save karne wali ek jagah."""

    def __init__(self, reports_dir: Path, show: bool) -> None:
        self.reports_dir = reports_dir
        self.fig_dir = reports_dir / "figures" / "02_pre_eda"
        self.table_dir = reports_dir / "tables" / "02_pre_eda"
        self.show = show
        for p in (self.fig_dir, self.table_dir):
            p.mkdir(parents=True, exist_ok=True)

    def fig(self, fig: plt.Figure, name: str) -> None:
        """Har chart ko numbered PNG mein save karo (aur chahein to dikhao)."""
        fig.tight_layout()
        path = self.fig_dir / f"{name}.png"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        logger.info("Chart saved: %s", path)
        if self.show:
            plt.show()
        plt.close(fig)

    def table(self, df: pd.DataFrame | pd.Series, name: str, title: str) -> None:
        """Table ko log mein dikhao aur CSV mein save karo (notebook ke cell output ki jagah)."""
        if isinstance(df, pd.Series):
            df = df.to_frame(df.name if df.name is not None else "value")
        logger.info("%s\n%s", title, df.to_string())
        df.to_csv(self.table_dir / f"{name}.csv")


@dataclass
class EDAData:
    """Pre-EDA ke shared datasets (sab training period tak mehdood, siwaye `master`)."""
    master: pd.DataFrame          # poora master (train + validation + test), sirf schema/overview/forecast-window ke liye
    store_tx: pd.DataFrame        # store-day transactions (training period tak)
    hist: pd.DataFrame            # training period ki saari rows
    demand: pd.DataFrame          # hist, is_trainable == 1
    clean: pd.DataFrame           # demand, CLEAN_START ke baad
    recent: pd.DataFrame          # demand ke aakhri 364 din (+1 din)
    daily_total: pd.Series        # hist ki rozana company-wide sales
    partial_dates: pd.DatetimeIndex


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------
def load_data(processed_dir: Path) -> EDAData:
    master = pd.read_parquet(processed_dir / "retail_demand_master.parquet")
    store_tx = pd.read_parquet(processed_dir / "store_day_transactions.parquet")
    master["family"] = master["family"].astype(str)

    # EDA sirf training period par (validation/test ko haath nahi lagana)
    hist = master[master["split"] == "train"].copy()
    demand = hist[hist["is_trainable"] == 1].copy()                      # asli demand wali rows
    store_tx = store_tx[store_tx["date"] <= hist["date"].max()]

    # Partial-recording din: jab 5 ya zyada "normally sold" families ka data gayab tha (Phase 1 flag)
    unavail_per_day = hist[hist["is_family_unavailable"] == 1].groupby("date")["family"].nunique()
    partial_dates = pd.DatetimeIndex(unavail_per_day[unavail_per_day >= PARTIAL_MIN_FAMILIES].index)

    clean = demand[demand["date"] >= CLEAN_START].copy()                 # saaf period ki demand
    recent = demand[demand["date"] >= hist["date"].max() - pd.Timedelta(days=364)]   # aakhri 1 saal
    daily_total = hist.groupby("date")["sales"].sum()

    logger.info("Training period : %s -> %s  (%d din)", hist["date"].min().date(), hist["date"].max().date(),
                hist["date"].nunique())
    logger.info("History rows    : %s", f"{len(hist):,}")
    logger.info("Demand rows     : %s  (%.1f%% of history, is_trainable == 1)", f"{len(demand):,}",
                len(demand) / len(hist) * 100)
    logger.info("Clean period    : %s -> %s  (%s demand rows)", CLEAN_START.date(), hist["date"].max().date(),
                f"{len(clean):,}")
    logger.info("Stores / families / series: %d / %d / %s", hist["store_nbr"].nunique(), hist["family"].nunique(),
                f"{hist[['store_nbr', 'family']].drop_duplicates().shape[0]:,}")
    return EDAData(master, store_tx, hist, demand, clean, recent, daily_total, partial_dates)


def partial_runs(partial_dates: pd.DatetimeIndex, min_len: int = 20):
    """Lagatar partial-recording dinon ke (start, end) runs, jo kam az kam `min_len` din lambe hon."""
    s = pd.Series(partial_dates)
    runs = (s.diff().dt.days != 1).cumsum()
    for _, grp in s.groupby(runs.values):
        if len(grp) >= min_len:
            yield grp.min(), grp.max()


# ======================================================================================
# 0b / 0c. Data overview aur numeric inputs
# ======================================================================================
def section_0_overview(d: EDAData, out: Outputs, res: dict[str, Any]) -> None:
    """
    Insight 0 (data overview):
    - Master mein sirf 3 columns mein missing values hain aur teeno ki wajah saabit hai:
        1. `sales`           : sirf test ke 16 din (jo forecast karne hain)
        2. `id`              : sirf 25 December ki rows (Phase 1 mein calendar poora karne ke liye add hui)
        3. `first_sale_date` : sirf woh 53 series jo kabhi nahi bikin (dead series)
    - Kisi aur column mein missing value nahi => Pre-EDA mein koi imputation ki zaroorat nahi.
    - Unique values: 54 stores, 33 families, 22 cities, 16 states, 5 store types, 17 clusters.

    Insight 0c (numeric inputs):
    - `onpromotion` right-skewed hai (aadhi se zyada rows par promotion nahi, kuch par sainkron items).
      Phase 05 mein is ka log ya "series ke normal se kitna zyada" roop behtar rahega.
    - `oil_price` saaf period mein tang range mein raha (Section 10 mein sales se rishta check hoga).
    - `transactions` ka range bohat bara hai (chhote stores ke few hundred se bare stores ke hazaron tak).
    """
    master, clean = d.master, d.clean

    overview = pd.DataFrame({
        "dtype": master.dtypes.astype(str),
        "missing": master.isna().sum(),
        "missing %": (master.isna().mean() * 100).round(2),
        "unique values": master.nunique(),
    })
    logger.info("Master dataset: %s rows x %d columns", f"{master.shape[0]:,}", master.shape[1])
    out.table(overview, "00b_data_overview", "0b. Data overview")

    # Kya har missing value ki wajah samajh mein aati hai? (ek ek kar ke verify)
    checks = pd.DataFrame([
        {"column": "sales", "missing": int(master["sales"].isna().sum()),
         "wajah": "test ke 16 din (yahi forecast karna hai)",
         "verified": bool(master["sales"].isna().equals(master["split"].eq("test")))},
        {"column": "id", "missing": int(master["id"].isna().sum()),
         "wajah": "25 December ki add ki hui rows (Kaggle id nahi thi)",
         "verified": bool(master["id"].isna().equals(master["is_imputed_row"].eq(1)))},
        {"column": "first_sale_date", "missing": int(master["first_sale_date"].isna().sum()),
         "wajah": "woh series jo kabhi nahi biki (dead series)",
         "verified": bool(master["first_sale_date"].isna().equals(master["is_dead_series"].eq(1)))},
    ]).set_index("column")
    out.table(checks, "00b_missing_value_checks", "0b. Missing values ki wajah (verified)")

    # ---- 0c. Numeric inputs ki pehli nazar (saaf period) ----
    train_clean_days = master.loc[(master["date"] >= CLEAN_START) & (master["split"] == "train")]
    num = pd.DataFrame({
        "onpromotion": clean["onpromotion"].describe(),
        "oil_price": train_clean_days.drop_duplicates("date")["oil_price"].describe(),
        "transactions": d.store_tx.loc[(d.store_tx["date"] >= CLEAN_START) & (d.store_tx["transactions"] > 0),
                                       "transactions"].describe(),
    }).round(1)
    out.table(num, "00c_numeric_inputs", "0c. Numeric inputs (saaf period)")

    bins = [-0.5, 0.5, 5.5, 20.5, 100.5, np.inf]
    labels = ["0", "1-5", "6-20", "21-100", "100+"]
    promo_bins = (pd.cut(clean["onpromotion"], bins=bins, labels=labels).value_counts(normalize=True)
                  .reindex(labels) * 100)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    bars = ax.bar(labels, promo_bins.values, color=BLUE, width=0.6, label="Share of rows (%)")
    for b, v in zip(bars, promo_bins.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontsize=9, color=INK2)
    ax.set_title("onpromotion: kitne items promotion par the (saaf period, har store-family-day)")
    ax.set_xlabel("Items on promotion")
    ax.set_ylabel("% of rows")
    ax.yaxis.set_major_formatter(pct)
    ax.set_ylim(0, promo_bins.max() + 10)
    ax.legend(loc="upper right")
    out.fig(fig, "00_onpromotion_distribution")


# ======================================================================================
# 1. Target variable
# ======================================================================================
def section_1_target(d: EDAData, out: Outputs, res: dict[str, Any]) -> None:
    """
    Target = `sales`: ek store mein, ek product family ki, ek din mein biki hui QUANTITY (paisa nahi).
    MEATS, POULTRY, PRODUCE kg mein bikti hain, is liye wahan decimals hain.
    Shakal: 54 stores x 33 families = 1,782 time series, har ek ka agle 16 din (16-31 Aug 2017) ka forecast.
    Yeh time-series REGRESSION problem hai.

    Insight 1:
    - Target bohat right-skewed hai (skewness ~6.7). Median sirf 33 units, top 1% rows 6,800+ units,
      max 1 lakh se upar. ~12% demand rows mein sales zero hai (zyada tar slow-selling products).
    - `log(1 + sales)` se distribution kaafi mutawazin ho jati hai (skewness ~0).
    - Asar (Phase 05/06): model ko `log1p(sales)` par train karna. Metrics: RMSLE (log scale) aur
      WAPE (business volume wali ghalti) dono.

    Insight 2: 1,782 series ek jaisi nahi: kuch roz hazaron units, kuch kabhi kabhi. Ek hi model ko teeno
    qisam sambhalni hongi, is liye Section 6 mein demand pattern classification hai.
    """
    s = d.clean["sales"]                                  # saaf period (Section 2b dekhein)
    res["s"] = s
    target_stats = pd.Series({
        "rows": len(s), "mean": s.mean(), "median": s.median(),
        "p90": s.quantile(0.90), "p99": s.quantile(0.99), "max": s.max(),
        "zero share": (s == 0).mean(),
        "skewness (raw)": s.skew(), "skewness (log1p)": np.log1p(s).skew(),
    }).round(3)
    out.table(target_stats.to_frame("sales (clean demand rows)"), "01_target_stats", "1. Target statistics")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    clip = s.quantile(0.99)
    axes[0].hist(s.clip(upper=clip), bins=60, color=BLUE, edgecolor="white", linewidth=0.6,
                 label="Raw sales (clipped at p99)")
    axes[0].set_title("Raw sales (99th percentile par clip)")
    axes[0].set_xlabel("Units per store-family-day")
    axes[0].set_ylabel("Rows")
    axes[0].yaxis.set_major_formatter(thousands)
    axes[0].xaxis.set_major_formatter(thousands)
    axes[1].hist(np.log1p(s), bins=60, color=AQUA, edgecolor="white", linewidth=0.6, label="log(1 + sales)")
    axes[1].set_title("log(1 + sales)")
    axes[1].set_xlabel("log1p(units)")
    axes[1].set_ylabel("Rows")
    axes[1].yaxis.set_major_formatter(thousands)
    for ax in axes:
        ax.legend(loc="upper right")
    fig.suptitle("Target ki distribution: bohat zyada right-skewed", x=0.01, ha="left", fontsize=13,
                 fontweight="bold")
    out.fig(fig, "01_target_distribution")

    examples = [(44, "GROCERY I", "High volume, smooth"),
                (44, "LIQUOR,WINE,BEER", "Medium volume, spiky"),
                (32, "BABY CARE", "Low volume, intermittent (zyada din zero)")]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for ax, (st, fam, label) in zip(axes, examples):
        x = d.hist[(d.hist["store_nbr"] == st) & (d.hist["family"] == fam) & (d.hist["date"] >= "2016-08-01")]
        ax.plot(x["date"], x["sales"], color=BLUE, linewidth=1.2, label=f"Store {st} | {fam}")
        ax.set_title(label, fontsize=10.5)
        ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=5, integer=True))
        ax.yaxis.set_major_formatter(thousands)
        ax.set_ylabel("Units")
        ax.legend(loc="upper left")
    fig.suptitle("Teen asli series (Aug 2016 - Jul 2017): har series ki shakal alag hai", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    out.fig(fig, "02_example_series")


# ======================================================================================
# 2. Company sales trend
# ======================================================================================
def section_2_trend(d: EDAData, out: Outputs, res: dict[str, Any]) -> None:
    """
    Insight 3 (trend):
    - 2014-2015 ke bare utaar-charhao (orange shaded mahine) demand ke nahi, DATA RECORD hone ke hain
      (Section 2b mein saabit). Is liye 2013-2015 ki "growth" seedha nahi parhi ja sakti.
    - Saaf period mein same-month saal-ba-saal muqabla: sales ab bhi barh rahi hai, lekin dheemi raftaar se.
      "same stores" column asli organic growth hai (naye stores ke baghair).
    - Asar (Phase 05): model ko trend samajhna hoga; pichle hafton ki rolling averages yeh kaam karti hain.
    """
    daily_total, hist = d.daily_total, d.hist
    ma28 = daily_total.rolling(28, center=True).mean()

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(daily_total.index, daily_total.values, color=LIGHT_GRAY, linewidth=0.8, label="Rozana total")
    ax.plot(daily_total.index, ma28, color=BLUE, linewidth=2, label="28-din moving average")
    ax.set_title("Company ki rozana total sales (saare stores aur families)")
    ax.set_ylabel("Units per day")
    ax.yaxis.set_major_formatter(thousands)
    for start, end in partial_runs(d.partial_dates):
        ax.axvspan(start, end, color=ORANGE, alpha=0.10, linewidth=0)
    ax.plot([], [], color=ORANGE, alpha=0.3, linewidth=8, label="Partial-recording mahine (Section 2b)")
    q = pd.Timestamp("2016-04-16")
    ax.annotate("Zalzala\n(16 Apr 2016)", xy=(q, ma28[q]),
                xytext=(q - pd.Timedelta(days=330), daily_total.max() * 0.93), fontsize=9, color=INK2,
                arrowprops=dict(arrowstyle="-", color=GRAY, linewidth=0.8))
    ax.legend(loc="upper left")
    out.fig(fig, "03_daily_total_sales_trend")

    # Imaandar growth: sirf SAAF period ke saal aapas mein milao, same mahine
    def avg_daily(df: pd.DataFrame, start: str, end: str) -> float:
        x = df[(df["date"] >= start) & (df["date"] <= end)]
        return x["sales"].sum() / x["date"].nunique()

    old_stores = d.master.loc[d.master["store_open_date"] <= "2015-05-31", "store_nbr"].unique()   # 2015 se pehle
    same_store = hist[hist["store_nbr"].isin(old_stores)]
    periods = [("2015-06-01", "2015-12-31", "2016-06-01", "2016-12-31"),
               ("2016-01-01", "2016-07-30", "2017-01-01", "2017-07-30")]
    growth = pd.DataFrame({
        "period": ["Jun-Dec 2015 -> Jun-Dec 2016", "Jan-Jul 2016 -> Jan-Jul 2017"],
        "growth % (all stores)": [avg_daily(hist, p[2], p[3]) / avg_daily(hist, p[0], p[1]) - 1 for p in periods],
        "growth % (same stores)": [avg_daily(same_store, p[2], p[3]) / avg_daily(same_store, p[0], p[1]) - 1
                                   for p in periods],
    }).set_index("period")
    growth = (growth * 100).round(1)
    logger.info("Same stores = %d stores jo June 2015 se pehle khul chuke the", len(old_stores))
    out.table(growth, "03_growth_year_over_year", "2. Saal-ba-saal growth (same months)")


# ======================================================================================
# 2b. Data recording regimes
# ======================================================================================
def section_2b_recording_regimes(d: EDAData, out: Outputs, res: dict[str, Any]) -> None:
    """
    Insight (data quality) - EDA ki sab se aham khoj:
    - BEVERAGES (company ki 2nd sab se bari family) ki sales kuch mahinon ki pehli tareekh par aadhi ho jati hai
      aur agle "normal" mahine ki pehli tareekh par wapas dugni (misal: 31 Mar 2014 ~1.4 lakh, 1 Apr ~70 hazar).
      Asli demand kabhi mahine ki pehli tareekh par achanak aadhi nahi hoti => data record hone ka tareeqa badla.
    - Yahi mahine (poora 2013, Feb 2014, Apr-Jun 2014, Aug 2014, Jan-Apr 2015) woh hain jab Phase 1 ke mutabiq
      8 families ka data gayab tha. FROZEN FOODS jaisi families mein bhi sales kam record hui.
    - Aakhri aisa period May 2015 ke shuru mein khatam; 1 June 2015 se data saaf hai.

    Faisla (aage ke phases ke liye bohat aham):
    - Model training data 1 June 2015 se (`CLEAN_START`): ~2 saal 2 mahine saaf data, 2 Christmas shamil.
    - Is EDA ke seasonal analyses bhi isi saaf period par.
    - Purana data (2013-May 2015) sirf lag features ke liye, aur wahan bhi ehtiyat.
    - Phase 1 ka `is_trainable` flag yeh nahi pakar saka (sales zero nahi, aadhi thi); faisla training window se lagu hoga.
    """
    bev = d.hist[d.hist["family"] == "BEVERAGES"].groupby("date")["sales"].sum()["2013-10-01":"2015-09-30"]

    fig, ax = plt.subplots(figsize=(10, 4))
    for start, end in partial_runs(d.partial_dates):
        ax.axvspan(start, end, color=ORANGE, alpha=0.10, linewidth=0)
    ax.plot(bev.index, bev.values, color=LIGHT_GRAY, linewidth=0.8, label="BEVERAGES, rozana total")
    ax.plot(bev.index, bev.rolling(7, center=True).mean(), color=BLUE, linewidth=2, label="BEVERAGES, 7-din average")
    ax.plot([], [], color=ORANGE, alpha=0.3, linewidth=8, label="Mahine jab 8 families ka data gayab tha")
    ax.set_title("BEVERAGES: sales har dafa mahine ki pehli tareekh par aadhi / dugni hoti hai")
    ax.set_ylabel("Units per day")
    ax.yaxis.set_major_formatter(thousands)
    ax.legend(loc="upper left")
    out.fig(fig, "03b_recording_regimes_beverages")

    # Har family: partial-recording mahinon ki average vs baqi mahine (Oct 2013 - May 2015 ke andar)
    win = d.demand[(d.demand["date"] >= "2013-10-01") & (d.demand["date"] < CLEAN_START)].copy()
    win["partial"] = win["date"].isin(d.partial_dates)
    daily_fam = win.groupby(["family", "date", "partial"])["sales"].sum().reset_index()
    ratio = daily_fam.groupby(["family", "partial"])["sales"].mean().unstack()
    ratio = (ratio[True] / ratio[False]).dropna().sort_values()
    out.table(ratio.round(2).head(8).to_frame("ratio"), "03b_partial_vs_normal_ratio",
              "2b. Partial-recording dinon ki sales / normal dinon ki sales (1.00 = koi farq nahi)")


# ======================================================================================
# 3. Weekly seasonality
# ======================================================================================
def section_3_weekly(d: EDAData, out: Outputs, res: dict[str, Any]) -> None:
    """
    Insight 4 (weekly pattern):
    - Saturday aur Sunday sab se zyada bikne wale din, Thursday sab se kamzor.
    - Har family ka pattern alag (heatmap): kuch weekend par bohat barhti hain, kuch (SCHOOL AND OFFICE
      SUPPLIES) mein weekly farq kam.
    - Business: weekend se pehle zyada stock; Thursday/Friday ko shelves bharna.
    - Asar (Phase 05): `day_of_week`, "pichle hafton mein isi weekday ki average" (`same_weekday_mean4`),
      aur family x weekday combination.
    """
    d_daily = d.clean.groupby("date")["sales"].sum()
    d_daily = d_daily / d_daily.rolling(28, center=True, min_periods=14).mean()      # trend hatao -> index
    dow_index = d_daily.groupby(d_daily.index.dayofweek).mean()
    res["d_daily"], res["dow_index"] = d_daily, dow_index

    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars = ax.bar(DOW_NAMES, dow_index.values, color=BLUE, width=0.6, label="Sales index (trend hata kar)")
    ax.axhline(1, color=GRAY, linewidth=1, label="Average din (1.00)")
    for b, v in zip(bars, dow_index.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_ylim(0.7, dow_index.max() + 0.16)
    ax.set_title("Hafte ke har din ki sales, average din ke muqable (1.00 = average)")
    ax.set_ylabel("Sales index")
    ax.legend(loc="upper left", ncol=2)
    out.fig(fig, "04_day_of_week_index")
    out.table(dow_index.rename(lambda i: DOW_NAMES[i]).round(3).to_frame("index"), "04_day_of_week_index",
              "3. Day-of-week index")

    fam_daily = d.clean.groupby(["family", "date"])["sales"].sum().reset_index()
    fam_daily["dow"] = fam_daily["date"].dt.dayofweek
    fam_dow = fam_daily.pivot_table(index="family", columns="dow", values="sales", aggfunc="mean")
    fam_dow = fam_dow.div(fam_dow.mean(axis=1), axis=0)                       # har family apne average se
    fam_dow = fam_dow.loc[fam_dow[[5, 6]].mean(axis=1).sort_values(ascending=False).index]
    fam_dow.columns = DOW_NAMES

    fig, ax = plt.subplots(figsize=(8.5, 9))
    im = ax.imshow(fam_dow.values, cmap=DIVERGING, norm=TwoSlopeNorm(vcenter=1, vmin=0.6, vmax=1.4), aspect="auto")
    ax.set_xticks(range(7), DOW_NAMES)
    ax.set_yticks(range(len(fam_dow)), fam_dow.index, fontsize=8.5)
    for i in range(fam_dow.shape[0]):
        for j in range(7):
            ax.text(j, i, f"{fam_dow.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color=INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Index (1.00 = family ka average din)")
    ax.set_title("Har family ka weekly pattern (upar: weekend par sab se zyada barhne wali)")
    ax.legend(handles=[Patch(facecolor=RED, label="Family ke average din se zyada"),
                       Patch(facecolor=BLUE, label="Family ke average din se kam")],
              loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=2)
    out.fig(fig, "05_family_day_of_week_heatmap")


# ======================================================================================
# 4. Monthly seasonality
# ======================================================================================
def section_4_monthly(d: EDAData, out: Outputs, res: dict[str, Any]) -> None:
    """
    Insight 5 (saal ka pattern):
    - December saal ka sab se bara mahina (average se ~18% zyada, Christmas shopping).
    - August aur September sab se kamzor (average se 6-10% kam). Hamara asli forecast August ka hai.
    - Ehtiyat: April ka index uncha dikhta hai, lekin saaf period mein sirf ek April (2016) poora hai aur woh
      zalzale ka mahina tha (Section 9). Saaf data sirf ~2 saal ka hai, is liye monthly pattern weekly jitna pakka nahi.
    - Business: December ke liye extra inventory/staff; August mein over-stocking se bachna.
    - Asar (Phase 05): `month`, "pichle saal isi din ki sales" (lag 364), aur `is_earthquake_period`.
    """
    # 2016 saaf period ka akela poora saal hai; trend hatane ke liye har din ko 365-din centered average se divide
    cd = d.clean.groupby("date")["sales"].sum()
    trend365 = cd.rolling(365, center=True, min_periods=300).mean()
    mi = (cd / trend365).dropna()
    month_index = mi.groupby(mi.index.month).mean()
    res["month_index"] = month_index
    logger.info("Month index in dinon se bana: %s -> %s", mi.index.min().date(), mi.index.max().date())

    colors = [RED if m == 12 else BLUE for m in month_index.index]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    bars = ax.bar(MONTH_NAMES, month_index.values, color=colors, width=0.6)
    ax.axhline(1, color=GRAY, linewidth=1)
    for b, v in zip(bars, month_index.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_ylim(0.75, month_index.max() + 0.14)
    ax.set_title("Mahine ka sales index (saaf period, trend hata kar; 1.00 = average)")
    ax.set_ylabel("Sales index")
    ax.legend(handles=[Patch(facecolor=BLUE, label="Baqi mahine"),
                       Patch(facecolor=RED, label="December (peak)"),
                       Line2D([0], [0], color=GRAY, linewidth=1, label="Average (1.00)")],
              loc="upper left", ncol=3)
    out.fig(fig, "06_month_of_year_index")


# ======================================================================================
# 5. Payday effect
# ======================================================================================
def section_5_payday(d: EDAData, out: Outputs, res: dict[str, Any]) -> None:
    """
    Ecuador mein sarkari tankhwah mahine ki 15 tareekh aur aakhri din milti hai (Kaggle description).

    Insight 6 (payday):
    - Sab se zyada sales mahine ke pehle 3-5 din (1-3 tareekh par average se ~10-17% zyada).
    - Mahine ke aakhri din (30-31) sales phir upar; 16 tareekh par chhota uchhaal. Tankhwah se pehle ke
      din (10-14, 24-28) sab se kamzor.
    - Mahine ke aakhir wali tankhwah ka asar 15 tareekh wali se kahin bara.
    - Asar (Phase 05): `day_of_month` aur "tankhwah ke baad kitne din guzre" (days since payday).
    """
    dom = res["d_daily"].to_frame("idx")
    dom["day"] = dom.index.day
    normal_days = d.clean.groupby("date")["is_holiday"].max() == 0          # holidays nikaal kar
    dom = dom[normal_days.reindex(dom.index).fillna(False)]
    dom_index = dom.groupby("day")["idx"].mean()

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(dom_index.index, dom_index.values, color=BLUE, marker="o", markersize=4, label="Sales index")
    ax.axhline(1, color=GRAY, linewidth=1, label="Average (1.00)")
    for k, day in enumerate((1, 15, 30)):
        ax.axvline(day, color=GRAY, linewidth=1, linestyle="--", zorder=0,
                   label="Payday ke qareeb tareekhein (1, 15, 30)" if k == 0 else None)
    ax.set_xticks(range(1, 32, 2))
    ax.set_title("Mahine ki har tareekh ka sales index (trend aur holidays hata kar)")
    ax.set_xlabel("Mahine ki tareekh")
    ax.set_ylabel("Sales index")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)       # legend data se door, neeche
    out.fig(fig, "07_day_of_month_index")
    out.table(dom_index.loc[[1, 2, 3, 14, 15, 16, 28, 29, 30, 31]].round(3).to_frame("index"),
              "07_day_of_month_index", "5. Day-of-month index (chuni hui tareekhein)")


# ======================================================================================
# 6 / 6b. Product families aur demand pattern classification
# ======================================================================================
def section_6_families(d: EDAData, out: Outputs, res: dict[str, Any]) -> None:
    """
    Insight 7 (families):
    - Volume chand families mein: GROCERY I, BEVERAGES aur PRODUCE akele ~2/3 units; top 5 families 80%.
      Bohat si families (BOOKS, BABY CARE, HARDWARE...) ka hissa 1% se bhi kam.
    - Business: forecast ghalti ka paisa zyada tar inhi bari families mein; accuracy sab se pehle inhi ki dekhein.
    - Asar (Phase 06): WAPE (volume-weighted) business ke liye sahi metric; RMSLE chhoti families ko barabar
      ahmiyat deta hai. Dono report honge, aur family-wise alag bhi.

    6b. Demand pattern classification (Syntetos-Boylan, ADI / CV2):
      ADI = average demand interval (1 = roz bikta hai), CV2 = bikne wale din quantity ka utaar-charhao.
      Smooth (ADI<1.32, CV2<0.49) | Erratic (ADI<1.32) | Intermittent (CV2<0.49) | Lumpy (baqi, sab se mushkil)

    Insight 8 (forecast kitna mushkil hai):
    - Do-tihai (67%) series Smooth hain aur volume ka ~97% inhi se aata hai (achi khabar).
    - ~16% series Intermittent/Lumpy (BABY CARE, HOME APPLIANCES, SCHOOL AND OFFICE SUPPLIES, HARDWARE, BOOKS):
      ginti mein kaafi, volume mein 0.1% se kam.
    - Asar (Phase 06): evaluation class ke hisaab se bhi report ho; Lumpy ke liye simple baseline se muqabla.
    """
    recent = d.recent
    fam_share = recent.groupby("family")["sales"].sum().sort_values(ascending=False)
    fam_share = fam_share / fam_share.sum() * 100
    cum = fam_share.cumsum()
    res["cum"] = cum

    fig, ax = plt.subplots(figsize=(9.5, 8))
    order = fam_share.index[::-1]
    ax.barh(order, fam_share[order].values, color=BLUE, height=0.6, label="Share of total units (%)")
    for y, fam in enumerate(order):
        ax.text(fam_share[fam] + 0.3, y, f"{fam_share[fam]:.1f}%", va="center", fontsize=8, color=INK2)
    ax.set_title("Har family ka total sales (units) mein hissa: aakhri 12 mahine")
    ax.set_xlabel("% of total units")
    ax.xaxis.set_major_formatter(pct)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_xlim(0, fam_share.max() + 4)
    ax.legend(loc="lower right")
    out.fig(fig, "08_family_sales_share")
    logger.info("Top 3 families ka hissa: %.1f%% | Top 5: %.1f%% | 80%% volume sirf %d families se aata hai",
                cum.iloc[2], cum.iloc[4], int((cum < 80).sum()) + 1)

    # ---- 6b. ADI / CV2 ----
    def classify(g: pd.Series) -> pd.Series:
        nz = g[g > 0]
        if len(nz) < 2:
            return pd.Series({"ADI": np.nan, "CV2": np.nan, "units": g.sum()})
        return pd.Series({"ADI": len(g) / len(nz), "CV2": (nz.std() / nz.mean()) ** 2, "units": g.sum()})

    cls = recent.groupby(["store_nbr", "family"])["sales"].apply(classify).unstack().dropna()
    cls["class"] = np.select(
        [(cls["ADI"] < 1.32) & (cls["CV2"] < 0.49), (cls["ADI"] < 1.32), (cls["CV2"] < 0.49)],
        ["Smooth", "Erratic", "Intermittent"], default="Lumpy")
    summary_cls = cls.groupby("class").agg(series=("units", "size"), units=("units", "sum"))
    summary_cls["% of series"] = summary_cls["series"] / summary_cls["series"].sum() * 100
    summary_cls["% of units"] = summary_cls["units"] / summary_cls["units"].sum() * 100
    summary_cls = summary_cls.reindex(["Smooth", "Erratic", "Intermittent", "Lumpy"])
    res["summary_cls"] = summary_cls

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    panels = [(axes[0], "% of series", "Kitni series", BLUE), (axes[1], "% of units", "Kitna volume (units)", AQUA)]
    for ax, col, title, color in panels:
        ax.barh(summary_cls.index[::-1], summary_cls[col][::-1], color=color, height=0.55, label=col)
        for y, v in enumerate(summary_cls[col][::-1]):
            ax.text(v + 1, y, f"{v:.1f}%" if v >= 0.1 else "<0.1%", va="center", fontsize=9, color=INK2)
        ax.set_title(title)
        ax.xaxis.set_major_formatter(pct)
        ax.set_xlim(0, 110)
        ax.legend(loc="lower right")
    fig.suptitle("Demand pattern classes (aakhri 12 mahine, ADI / CV²)", x=0.01, ha="left", fontsize=13,
                 fontweight="bold")
    out.fig(fig, "09_demand_pattern_classes")
    out.table(summary_cls.round(1), "09_demand_pattern_summary", "6b. Demand pattern classes")

    # Har family ki series kis class mein hain
    fam_cls = pd.crosstab(pd.Series(cls.index.get_level_values("family"), name="family", index=cls.index),
                          cls["class"])
    fam_cls = fam_cls.reindex(columns=["Smooth", "Erratic", "Intermittent", "Lumpy"], fill_value=0)
    fam_cls = fam_cls.loc[fam_cls.sort_values(["Smooth", "Erratic"], ascending=False).index]
    out.table(fam_cls, "09_family_demand_classes", "6b. Family-wise demand classes (series count)")


# ======================================================================================
# 7. Stores
# ======================================================================================
def section_7_stores(d: EDAData, out: Outputs, res: dict[str, Any]) -> None:
    """
    Insight 9 (stores):
    - Stores ke size mein bohat bara farq: sab se bara store (44) sab se chhote (32) se ~10 guna zyada bechta hai.
    - Type A sab se bare (~32 hazar units/din), Type C sab se chhote (~8 hazar): type asli size ka ishara hai.
    - Quito akela company ke ~50% units (18 stores), Guayaquil ~11%.
    - Asar (Phase 05): `store_nbr`, `store_type`, `city`, `cluster` zaroori (Phase 1b ablation mein store/family
      hatane se ghalti barhi thi). Har store ka "normal level" rolling averages se bhi aata hai.
    """
    store_attrs = d.master.drop_duplicates("store_nbr").set_index("store_nbr")[["store_type", "city", "store_cluster"]]
    store_daily = (d.recent.groupby(["store_nbr", "date"])["sales"].sum().groupby("store_nbr").mean()
                   .rename("avg_daily_units").to_frame().join(store_attrs).sort_values("avg_daily_units"))

    fig, ax = plt.subplots(figsize=(9.5, 9))
    ax.barh([f"Store {i}" for i in store_daily.index], store_daily["avg_daily_units"], color=BLUE, height=0.65,
            label="Average daily units")
    for y, (v, t) in enumerate(zip(store_daily["avg_daily_units"], store_daily["store_type"])):
        ax.text(v + 150, y, f"type {t}", va="center", fontsize=7.5, color=INK2)
    ax.set_title("Har store ki average rozana sales (aakhri 12 mahine, khule din)")
    ax.set_xlabel("Units per day")
    ax.xaxis.set_major_formatter(thousands)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.legend(loc="lower right")
    out.fig(fig, "10_store_size_ranking")

    by_type = store_daily.groupby("store_type")["avg_daily_units"].agg(stores="size", avg_units_per_store="mean").round(0)
    by_type["% of company units"] = (store_daily.groupby("store_type")["avg_daily_units"].sum()
                                     / store_daily["avg_daily_units"].sum() * 100).round(1)
    by_city = (store_daily.groupby("city")["avg_daily_units"].agg(stores="size", total="sum")
               .sort_values("total", ascending=False))
    by_city["% of company units"] = (by_city["total"] / by_city["total"].sum() * 100).round(1)
    logger.info("Sab se bara store, sab se chhote se %.1f guna bara hai",
                store_daily["avg_daily_units"].max() / store_daily["avg_daily_units"].min())
    out.table(by_type.sort_values("avg_units_per_store", ascending=False), "10_stores_by_type", "7. Stores by type")
    out.table(by_city.head(6).drop(columns="total"), "10_stores_by_city", "7. Top cities (company units share)")


# ======================================================================================
# 8. Promotions
# ======================================================================================
def section_8_promotions(d: EDAData, out: Outputs, res: dict[str, Any]) -> None:
    """
    Insight 10 (promotions):
    - Promotion 2014 se record hone lagi aur har saal barhi; 2017 mein ~aadhi rows par koi item promotion par.
    - Lagbhag har family mein promotion wale dinon par sales zyada; aam series mein ~+14%. Bari families mein
      farq chhota (GROCERY I ~11%, DAIRY ~10%, BEVERAGES ~6%), chhoti mein bara (LADIESWEAR ~68%, BEAUTY ~52%).
    - Ehtiyat: yeh CORRELATION hai, causation nahi. SCHOOL AND OFFICE SUPPLIES ka bara number is ki misaal hai:
      promotion school season mein lagti hai jab demand waise hi kai guna hoti hai.
    - Asar (Phase 05): `onpromotion` future mein maloom feature hai (test.csv mein diya hua) => bohat qeemti.
      Features: promotion level, "series ke normal se kitna zyada", aur 2014 se pehle ke 0 ko "unknown"
      (`is_promo_tracked`).
    """
    master, demand = d.master, d.demand
    promo_start = master.loc[master["is_promo_tracked"] == 1, "date"].min()
    monthly_promo = demand.groupby(demand["date"].dt.to_period("M"))["onpromotion"].apply(lambda x: (x > 0).mean() * 100)
    monthly_promo.index = monthly_promo.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(monthly_promo.index, monthly_promo.values, color=BLUE, label="Rows with promotion (% per month)")
    ax.axvline(promo_start, color=GRAY, linewidth=1, label=f"Promotion tracking shuru: {promo_start.date()}")
    ax.set_title("Kitni rows par promotion thi (har mahine, %)")
    ax.set_ylabel("% rows with onpromotion > 0")
    ax.yaxis.set_major_formatter(pct)
    ax.legend(loc="upper left")
    out.fig(fig, "11_promotion_coverage_over_time")

    # Ek hi series ke andar: promotion wale din vs bina promotion ke din (saaf period)
    p = d.clean
    g = p.groupby(["store_nbr", "family", p["onpromotion"] > 0])["sales"].agg(["mean", "size"]).unstack()
    ok = (g[("size", True)] >= 30) & (g[("size", False)] >= 30) & (g[("mean", False)] > 0)
    lift = (g.loc[ok, ("mean", True)] / g.loc[ok, ("mean", False)] - 1) * 100
    res["lift"] = lift
    fam_lift = lift.groupby(level="family").median().sort_values()

    cap = 200                                   # bohat bari values chart bigaarti hain -> clip, label asli
    shown_lift = fam_lift.clip(upper=cap)
    fig, ax = plt.subplots(figsize=(9.5, 8))
    ax.barh(fam_lift.index, shown_lift.values, color=[RED if v < 0 else BLUE for v in fam_lift.values], height=0.6)
    ax.axvline(0, color=GRAY, linewidth=1)
    for y, (v, sv) in enumerate(zip(fam_lift.values, shown_lift.values)):
        label = f"{v:+.0f}%" + (" (axis se bahar)" if v > cap else "")
        ax.text(sv + (3 if v >= 0 else -3), y, label, va="center", ha="left" if v >= 0 else "right",
                fontsize=8, color=INK2)
    ax.set_xlim(min(-10, fam_lift.min() - 15), cap + 70)
    ax.set_title("Promotion wale dinon mein sales kitni zyada thi (har family, median series)")
    ax.set_xlabel("% difference vs bina promotion ke din (same store, same family)")
    ax.tick_params(axis="y", labelsize=8.5)
    ax.legend(handles=[Patch(facecolor=BLUE, label="Promotion par sales zyada"),
                       Patch(facecolor=RED, label="Promotion par sales kam")], loc="lower right")
    out.fig(fig, "12_promotion_lift_by_family")
    logger.info("Series compared: %s | median lift across all: %+.0f%%", f"{ok.sum():,}", lift.median())


# ======================================================================================
# 9. Holidays, Christmas aur zalzala
# ======================================================================================
def find_raw_dir(raw_dir: Path | None) -> Path:
    candidates = [raw_dir] if raw_dir else [Path("."), Path("data/raw"), Path("../data/raw")]
    for p in candidates:
        if (p / "holidays_events.csv").exists():
            return p
    raise FileNotFoundError(f"holidays_events.csv nahi mili. Dhoondi gayi jagahein: {[str(c) for c in candidates]}. "
                            "--raw-dir se sahi folder batayein.")


def section_9_holidays(d: EDAData, out: Outputs, res: dict[str, Any], raw_dir: Path | None) -> None:
    """
    Insight 11 (holidays aur events):
    - Holidays ka asar ek jaisa nahi: kuch par rush (Christmas se pehle, Dia del Trabajo, Independencia de
      Cuenca), kuch par sales girti hain (1 January ~zero, Dia de la Madre, Batalla de Pichincha). Phase 1b
      mein test hua tha ke yeh asar har saal dohraya jata hai.
    - Christmas: 25 Dec se ~10 din pehle sales barhti hain, aakhri 5 dinon mein November se ~50% zyada.
      25 Dec aur 1 Jan ko stores band; 2 January ko bara uchhaal (band dinon ke baad khareedari).
    - Zalzala (April 2016): baad ke 2 hafton mein sales normal se zyada (relief khareedari), phir normal.
    - Business: Christmas se 2 hafte pehle stock; har holiday ki planning uske apne pichle asar se.
    - Asar (Phase 05): "isi holiday ka pichle saalon mein asar", "Christmas se kitne din pehle",
      `is_earthquake_period`.
    """
    hol = pd.read_csv(find_raw_dir(raw_dir) / "holidays_events.csv", parse_dates=["date"])
    nat = hol[(hol["locale"] == "National") & ~hol["transferred"]]
    names = nat.groupby("date")["description"].agg(lambda x: " | ".join(sorted(x)))
    company = d.daily_total
    partial_dates = d.partial_dates

    rows = []
    for dt, name in names.items():
        if dt < pd.Timestamp("2014-01-01") or dt not in company.index:
            continue
        ref = [dt + pd.Timedelta(weeks=k) for k in (-3, -2, 2, 3)]
        ref = [x for x in ref if x in company.index and x not in names.index]
        same_regime = all((x in partial_dates) == (dt in partial_dates) for x in ref)     # recording regime ek jaisa
        if ref and same_regime:
            rows.append({"holiday": name, "effect": (company[dt] / company[ref].mean() - 1) * 100})
    eff = pd.DataFrame(rows).groupby("holiday")["effect"].agg(["mean", "size"])
    eff = eff[eff["size"] >= 2].sort_values("mean")
    eff = pd.concat([eff.head(8), eff.tail(10)]).drop_duplicates()

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.barh(eff.index, eff["mean"], color=[RED if v < 0 else BLUE for v in eff["mean"]], height=0.6)
    ax.axvline(0, color=GRAY, linewidth=1)
    for y, v in enumerate(eff["mean"]):
        ax.text(v + (2 if v >= 0 else -2), y, f"{v:+.0f}%", va="center", ha="left" if v >= 0 else "right",
                fontsize=8, color=INK2)
    ax.set_xlim(eff["mean"].min() - 18, eff["mean"].max() + 18)
    ax.set_title("National holidays ka company sales par average asar (2014-2017)")
    ax.set_xlabel("% vs isi weekday ke 2-3 hafte pehle/baad ke normal din")
    ax.tick_params(axis="y", labelsize=8.5)
    ax.legend(handles=[Patch(facecolor=BLUE, label="Sales barhi"), Patch(facecolor=RED, label="Sales giri")],
              loc="lower right")
    out.fig(fig, "13_holiday_effects")

    # Christmas season
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for year, color in [(2014, GRAY), (2015, ORANGE), (2016, BLUE)]:
        w = company[f"{year}-12-10":f"{year + 1}-01-06"]
        base = company[f"{year}-11-01":f"{year}-11-30"].mean()
        days = (w.index - pd.Timestamp(f"{year}-12-25")).days
        ax.plot(days, w.values / base, color=color, linewidth=1.8, label=f"Christmas {year}")
    ax.axvline(0, color=LIGHT_GRAY, linewidth=1, label="25 December")
    ax.axhline(1, color=GRAY, linewidth=1, label="November average (1.00)")
    ax.set_title("Christmas season: rozana sales, November ki average ke muqable")
    ax.set_xlabel("Din, 25 December se pehle (-) aur baad (+)")
    ax.set_ylabel("Index (1.00 = Nov average)")
    ax.legend(loc="upper left")
    out.fig(fig, "14_christmas_season")

    # 2016 ka zalzala
    w16 = company["2016-03-01":"2016-06-15"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(w16.index, w16.values, color=BLUE, linewidth=1.5, label="Company rozana sales (2016)")
    ax.axvspan(pd.Timestamp("2016-04-16"), pd.Timestamp("2016-05-16"), color=ORANGE, alpha=0.12, linewidth=0,
               label="Zalzala + 30 din (is_earthquake_period)")
    ax.set_title("2016 ka zalzala: company ki rozana sales (Mar - Jun 2016)")
    ax.set_ylabel("Units per day")
    ax.yaxis.set_major_formatter(thousands)
    ax.legend(loc="upper left")
    out.fig(fig, "15_earthquake_2016")


# ======================================================================================
# 10. Oil price
# ======================================================================================
def section_10_oil(d: EDAData, out: Outputs, res: dict[str, Any]) -> None:
    """
    Dono ko ek hi scale par dikhane ke liye har ek ka pehla hafta = 100 (do alag y-axis wala chart gumrah karta hai).

    Insight 12 (oil):
    - Levels mein mazboot ULTA rishta: oil 2014-2015 mein gira aur sales barhti rahi.
    - Lekin tabdeeliyon (4-hafte ki % change) mein rishta ~zero. Levels wala rishta sirf is liye ke dono ka apna
      apna trend tha => SPURIOUS CORRELATION.
    - Asar (Phase 05/06): oil feature banana ho to sirf LAGGED, aur woh bhi tab jab test mein ghalti kam kare.
    """
    oil_daily = d.master.drop_duplicates("date").set_index("date")["oil_price"]
    weekly = pd.DataFrame({"sales": d.daily_total, "oil": oil_daily.reindex(d.daily_total.index)}).resample("W").mean().dropna()
    idx = weekly / weekly.iloc[0] * 100
    chg = weekly.pct_change(4).dropna() * 100
    res["weekly"], res["chg"] = weekly, chg

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
    axes[0].plot(idx.index, idx["sales"], color=BLUE, label="Sales")
    axes[0].plot(idx.index, idx["oil"], color=ORANGE, label="Oil price")
    axes[0].set_title("Levels (pehla hafta = 100)")
    axes[0].set_ylabel("Index")
    axes[0].legend(loc="upper left")
    axes[1].scatter(chg["oil"], chg["sales"], s=14, color=BLUE, alpha=0.6, edgecolor="white", linewidth=0.4,
                    label="4-hafte ki tabdeeli (har point = 1 hafta)")
    axes[1].axhline(0, color=GRAY, linewidth=1)
    axes[1].axvline(0, color=GRAY, linewidth=1)
    axes[1].set_title("4-hafte ki tabdeeli (%)")
    axes[1].set_xlabel("Oil price change %")
    axes[1].set_ylabel("Sales change %")
    axes[1].legend(loc="upper left")
    fig.suptitle("Oil aur sales: levels ek doosre ke ulat chalte hain, lekin tabdeeliyon mein koi rishta nahi",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    out.fig(fig, "16_oil_vs_sales")
    logger.info("Correlation of levels: %.2f | correlation of 4-week changes: %.2f",
                weekly["sales"].corr(weekly["oil"]), chg["sales"].corr(chg["oil"]))


# ======================================================================================
# 11. Transactions aur basket size
# ======================================================================================
def section_11_transactions(d: EDAData, out: Outputs, res: dict[str, Any]) -> None:
    """
    `transactions` = ek store mein ek din ki receipts. SIRF EDA ke liye: aane wale dinon ke customers forecast
    ke waqt maloom nahi hote, is liye model feature nahi ban sakta (leakage).

    Insight 13a (customers aur basket):
    - Customers aur sales ka rishta bohat mazboot (correlation ~0.9).
    - Saaf period mein basket size ~9-10 units par stable, dheere barh raha (Jan-Jul 2016: 9.15 -> 2017: 9.74).
      Pehle ke saalon mein basket "dugna" dikhta tha, woh Section 2b ke recording gaps ki wajah se tha.
    - Business: growth zyada tar zyada customers (footfall) aur thori bari baskets se aa rahi hai.
    - Asar (Phase 05): transactions ko same-day feature KABHI nahi banana (leakage); sirf 16+ din purani, aur
      test se saabit hone par.
    """
    st = d.store_tx
    tx = st[(st["is_store_closed"] == 0) & (st["is_pre_opening"] == 0)].dropna(subset=["transactions"])
    tx = tx[tx["transactions"] > 0].copy()
    tx_clean = tx[tx["date"] >= CLEAN_START]
    basket_m = tx_clean.groupby(tx_clean["date"].dt.to_period("M")).apply(
        lambda g: g["store_sales"].sum() / g["transactions"].sum())
    basket_m.index = basket_m.index.to_timestamp()

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
    hb = axes[0].hexbin(tx_clean["transactions"], tx_clean["store_sales"], gridsize=45, cmap="Blues", mincnt=1,
                        bins="log", linewidths=0)
    fig.colorbar(hb, ax=axes[0], label="Store-days (log scale)")
    axes[0].set_title("Customers vs units (har store-din)")
    axes[0].set_xlabel("Transactions")
    axes[0].set_ylabel("Units sold")
    axes[0].xaxis.set_major_formatter(thousands)
    axes[0].yaxis.set_major_formatter(thousands)
    axes[0].legend(handles=[Patch(facecolor=BLUE, label="Store-day density")], loc="upper left")
    axes[1].plot(basket_m.index, basket_m.values, color=BLUE, label="Average basket (monthly)")
    axes[1].set_title("Average basket: units per transaction (mahana)")
    axes[1].set_ylabel("Units per transaction")
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[1].set_ylim(basket_m.min() * 0.9, basket_m.max() * 1.08)
    axes[1].legend(loc="upper left")
    fig.suptitle("Customers aur sales (saaf period, Jun 2015 - Jul 2017)", x=0.01, ha="left", fontsize=12.5,
                 fontweight="bold")
    out.fig(fig, "17_transactions_and_basket")

    half = tx_clean.groupby([tx_clean["date"].dt.year.rename("year"),
                             (tx_clean["date"].dt.month <= 7).rename("jan_to_jul")]).apply(
        lambda g: g["store_sales"].sum() / g["transactions"].sum()).round(2)
    logger.info("Correlation (transactions vs units): %.2f", tx_clean["transactions"].corr(tx_clean["store_sales"]))
    logger.info("Basket Jan-Jul 2016: %s | Jan-Jul 2017: %s", half[(2016, True)], half[(2017, True)])
    out.table(basket_m.round(2).tail(6).to_frame("units per transaction"), "17_basket_last_months",
              "11. Average basket (aakhri 6 mahine)")


# ======================================================================================
# 12. Autocorrelation
# ======================================================================================
def section_12_autocorrelation(d: EDAData, out: Outputs, res: dict[str, Any]) -> None:
    """
    Har series ke andar: aaj ki sales ka k din pehle ki sales se rishta. Har series ka apna average pehle
    ghata diya (bari/chhoti series ka farq rishta na bane). Forecast 16 din aage ka hai => lag 1-15 forecast
    ke waqt maloom nahi (chart mein grey).

    Insight 14a (time-series structure):
    - Har 7 din baad (7, 14, 21, 28...) rishta upar jata hai: weekly pattern. Lag 7 (~0.39) lag 1 (~0.37) se bhi zyada.
    - Allowed lags mein lag 16 khud kamzor (~0.11), kyunke 16 din pehle hafte ka koi aur din tha. 21, 28, 35, 42
      (isi weekday) ka rishta ~0.25-0.30, takreeban 3 guna. Lag 364 (pichla saal, isi weekday) ~0.23.
    - Asar (Phase 05): history features sirf lag >= 16 aur weekday-aligned: lag 21, 28, 35, 42, 364,
      "pichle 4-8 hafton mein isi weekday ki average", aur 16 din pehle khatam hone wali rolling averages.
    - Phase 06 option: forecast din h ke liye lag >= h allowed => direct multi-horizon model; ek model wale
      tareeqe se compare karenge.
    """
    z = d.hist[["store_nbr", "family", "date", "sales", "is_trainable"]].sort_values(["store_nbr", "family", "date"]).copy()
    z["y"] = np.where((z["is_trainable"] == 1) & (z["date"] >= CLEAN_START), np.log1p(z["sales"]), np.nan)
    z["y"] = z["y"] - z.groupby(["store_nbr", "family"])["y"].transform("mean")
    gz = z.groupby(["store_nbr", "family"])["y"]
    lags = list(range(1, 57)) + [364]
    acf = pd.Series({k: z["y"].corr(gz.shift(k)) for k in lags})
    res["acf"] = acf

    shown = acf.drop(364)
    colors = [LIGHT_GRAY if k < HORIZON_DAYS else (BLUE if k % 7 == 0 else LIGHT_BLUE) for k in shown.index]
    fig, ax = plt.subplots(figsize=(10.5, 4))
    ax.bar(shown.index, shown.values, color=colors, width=0.7)
    ax.axvspan(0.4, HORIZON_DAYS - 0.4, color=LIGHT_GRAY, alpha=0.25, linewidth=0)
    ax.set_title(f"Aaj ki sales ka k din pehle ki sales se rishta (lag 364 = pichla saal: {acf[364]:.2f})")
    ax.set_xlabel("Lag k (din)")
    ax.set_ylabel("Correlation (series ke andar)")
    ax.set_xticks([1, 7, 14, 16, 21, 28, 35, 42, 49, 56])
    ax.set_ylim(0, shown.max() * 1.15)
    ax.legend(handles=[Patch(facecolor=LIGHT_GRAY, label="Lag 1-15: forecast ke waqt maloom nahi (horizon = 16)"),
                       Patch(facecolor=BLUE, label="Lag 7 ke multiples (isi weekday)"),
                       Patch(facecolor=LIGHT_BLUE, label="Baqi allowed lags (>= 16)")], loc="upper right")
    out.fig(fig, "18_autocorrelation_by_lag")
    out.table(acf.loc[[1, 7, 14, 16, 21, 28, 35, 42, 364]].round(3).to_frame("correlation"),
              "18_autocorrelation_key_lags", "12. Autocorrelation (chune hue lags)")


# ======================================================================================
# 13. Stationarity
# ======================================================================================
def section_13_stationarity(d: EDAData, out: Outputs, res: dict[str, Any]) -> None:
    """
    Stationary = average aur utaar-charhao (variance) waqt ke sath na badlein.

    Insight 13b (stationarity):
    - Sales ka average waqt ke sath barhta hai (trend) => raw series stationary NAHI.
    - Raw sales ka std bohat badalta hai (index ~60 se 160), aur average ke sath upar-neeche (correlation ~0.6).
    - `log1p` ke baad std tang range (~0.13-0.27) mein; sab se bara uchhaal April-May 2016 (zalzala).
    - Asar (Phase 05/06): tree-based models (LightGBM) ko stationarity ki zaroorat nahi, lekin trend ki
      information features se deni hogi => rolling averages aur relative features zaroori. `log1p` target
      utaar-charhao bhi mutawazin karta hai (Section 1 ki doosri daleel).
    """
    cd_clean = d.hist[d.hist["date"] >= CLEAN_START].groupby("date")["sales"].sum()
    cd_clean = cd_clean[~cd_clean.index.strftime("%m-%d").isin(["12-25", "01-01"])]   # company-wide band din nikaal kar
    roll_mean = cd_clean.rolling(28).mean()
    roll_std = cd_clean.rolling(28).std()
    log_cv = np.log1p(cd_clean).rolling(28).std()

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    axes[0].plot(roll_mean.index, roll_mean / roll_mean.dropna().iloc[0] * 100, color=BLUE, label="Rolling mean")
    axes[0].plot(roll_std.index, roll_std / roll_std.dropna().iloc[0] * 100, color=ORANGE, label="Rolling std")
    axes[0].set_title("Raw sales: mean aur std (pehla point = 100)")
    axes[0].set_ylabel("Index")
    axes[1].plot(log_cv.index, log_cv.values, color=BLUE, label="Rolling std of log1p(sales)")
    axes[1].set_title("log1p(sales) ka rolling std")
    axes[1].set_ylabel("Std of log sales")
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.legend(loc="upper left")
    fig.suptitle("Stationarity check: company ki rozana sales (saaf period, 28-din window)", x=0.01, ha="left",
                 fontsize=12.5, fontweight="bold")
    out.fig(fig, "19_stationarity_rolling_stats")
    first, last = roll_mean.dropna().iloc[0], roll_mean.dropna().iloc[-1]
    logger.info("Rolling mean: %s -> %s (%+.1f%%)", f"{first:,.0f}", f"{last:,.0f}", (last / first - 1) * 100)
    logger.info("Corr(rolling mean, rolling std) raw: %.2f", roll_mean.corr(roll_std))


# ======================================================================================
# 14. Outliers ki pehli nazar
# ======================================================================================
def section_14_outliers(d: EDAData, out: Outputs, res: dict[str, Any]) -> None:
    """
    Sirf dekhna, treatment Phase 03 mein. IQR jaisa aam tareeqa yahan kaam nahi karta (bari families ki har value
    "outlier" lagegi, aur har Sunday bhi). Is liye har value ko usi series ke apne normal se compare kiya:
      1. `log1p(sales)`
      2. Normal = usi weekday ki pichle 4 aur agle 4 hafton ki median (weekly pattern aur trend dono hat jate hain)
      3. Farq / series ka MAD => robust z-score
      4. |z| > 5 = "extreme" (sakht hadd, sirf waqai ghair-mamooli values)

    Insight 14b (outliers, pehli nazar):
    - Saaf period ki ~11.5 lakh rows check hui; sirf ~0.7% (8,267) extreme: ~68% bohat zyada, ~32% bohat kam.
    - Sab se zyada extreme din jaane pehchane events: 22-24 Dec, 31 Dec, 2 Jan, Dia del Trabajo,
      Independencia de Cuenca, zalzale ke agle din.
    - Wajah: ~34% Christmas/New Year season, ~15% holiday/event, ~13% promotion, ~8% zalzala, ~4% Carnaval se
      pehle ka Sunday => ~3/4 extremes ki asli wajah maloom; baqi ~26% Phase 03 mein.
    - Nayi khoj: Carnaval se pehle wala Sunday aas paas ke Sundays se 25-34% kam (5 mein se 4 saal). Holidays
      file mein nahi, lekin har saal dohraya jata hai => Phase 05 ka naya feature.
    - FROZEN FOODS mein sab se zyada extremes, ~93% December mein.
    - High extremes mein ~71% din promotion thi (aam taur par ~37%) => promotion bohat se outliers ki wajah.
    - ~1.5 lakh rows ka z-score nahi bana (series zyada tar zero, MAD = 0): unka tareeqa alag (Phase 03).
    - Natija: zyada tar outliers ASLI DEMAND hain; inhein hata dena sab se bari ghalti hogi.

    Phase 03 plan (yeh script kuch nahi badalta):
      1. Har extreme ki wajah pehchanna (holiday/event, promotion, store open/close, data ki ghalti)
      2. Asli events ko nahi hatana, flag karna
      3. Sirf data ki ghalti theek karna aur har tabdeeli ka record rakhna
    """
    master, hist, clean = d.master, d.hist, d.clean
    o = clean[["store_nbr", "family", "date", "sales", "onpromotion", "is_holiday", "is_holiday_season_day",
               "is_national_event", "is_earthquake_period"]].sort_values(["store_nbr", "family", "date"]).copy()
    o["y"] = np.log1p(o["sales"])
    o["dow"] = o["date"].dt.dayofweek
    base = (o.groupby(["store_nbr", "family", "dow"])["y"]
            .rolling(9, center=True, min_periods=5).median()
            .reset_index(level=[0, 1, 2], drop=True))
    o["baseline"] = base
    o["resid"] = o["y"] - o["baseline"]
    grp = o.groupby(["store_nbr", "family"])["resid"]
    med = grp.transform("median")
    mad = grp.transform(lambda x: (x - x.median()).abs().median())
    o["robust_z"] = (o["resid"] - med) / (1.4826 * mad.replace(0, np.nan))
    o["extreme"] = o["robust_z"].abs() > 5
    o["direction"] = np.where(o["robust_z"] > 5, "high", np.where(o["robust_z"] < -5, "low", ""))
    n_eval = int(o["robust_z"].notna().sum())
    res["o"], res["n_eval"] = o, n_eval

    logger.info("Rows checked: %s | extreme rows: %s (%.2f%%)", f"{n_eval:,}", f"{o['extreme'].sum():,}",
                o["extreme"].sum() / n_eval * 100)
    logger.info("Extreme direction counts:\n%s", o.loc[o["extreme"], "direction"].value_counts().to_string())
    logger.info("Rows jahan z-score nahi ban saka (series zyada tar zero, MAD = 0): %s", f"{o['robust_z'].isna().sum():,}")

    per_day = (o[o["extreme"]].groupby(["date", "direction"]).size().unstack(fill_value=0)
               .reindex(columns=["high", "low"], fill_value=0))
    per_day = per_day.reindex(pd.date_range(CLEAN_START, hist["date"].max()), fill_value=0)

    fig, ax = plt.subplots(figsize=(10.5, 4))
    ax.plot(per_day.index, per_day["high"], color=BLUE, linewidth=1.2, label="Normal se bohat zyada (high)")
    ax.plot(per_day.index, -per_day["low"], color=RED, linewidth=1.2, label="Normal se bohat kam (low)")
    ax.axhline(0, color=GRAY, linewidth=1)
    ax.set_title("Har din kitni series extreme thin (|robust z| > 5)")
    ax.set_ylabel("Series count (low neeche)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{abs(x):,.0f}"))
    ax.legend(loc="upper left")
    out.fig(fig, "20_outliers_per_day")

    names_by_date = master.drop_duplicates("date").set_index("date")["national_desc"].astype(str)
    top = per_day.assign(total=per_day.sum(axis=1)).sort_values("total", ascending=False).head(12)
    top["weekday"] = top.index.day_name()
    top["national holiday/event"] = names_by_date.reindex(top.index).values
    out.table(top, "20_top_extreme_days", "14. Sab se zyada extreme values wale 12 din")

    # Kaun si families mein extreme values zyada hain?
    fam_out = o.groupby("family").agg(rows=("robust_z", "count"), extreme=("extreme", "sum"))
    fam_out["extreme %"] = (fam_out["extreme"] / fam_out["rows"] * 100).round(2)
    high_promo = o[o["extreme"] & (o["direction"] == "high")]
    logger.info("High extremes jin din promotion bhi thi: %.1f%% (saare rows mein promotion: %.1f%%)",
                (high_promo["onpromotion"] > 0).mean() * 100, (o["onpromotion"] > 0).mean() * 100)
    out.table(fam_out.sort_values("extreme %", ascending=False).head(10), "20_extremes_by_family",
              "14. Families jin mein extreme values sab se zyada hain")

    # Har extreme value ki sab se pehli mumkin wajah (upar se neeche, pehli match wali)
    carnaval = master.loc[master["national_desc"].astype(str).str.contains("Carnaval"), "date"].drop_duplicates()
    carnaval_sunday = set(carnaval[carnaval.dt.dayofweek == 0] - pd.Timedelta(days=1))   # Carnaval Monday se pehle ka Sunday
    x = o[o["extreme"]].copy()
    mdx = x["date"].dt.strftime("%m-%d")
    x["reason"] = np.select(
        [x["is_earthquake_period"] == 1,
         (mdx >= "12-15") | (mdx <= "01-03"),
         x["date"].isin(carnaval_sunday),
         (x["is_holiday"] == 1) | (x["is_holiday_season_day"] == 1) | (x["is_national_event"] == 1),
         (x["direction"] == "high") & (x["onpromotion"] > 0)],
        ["Zalzala period", "Christmas / New Year season (15 Dec - 3 Jan)", "Carnaval se pehle ka Sunday",
         "Holiday / event ka din", "Promotion wala din (high)"],
        default="Abhi wajah nahi mili (Phase 03 mein check)")
    reasons = x["reason"].value_counts().to_frame("extreme rows")
    reasons["%"] = (reasons["extreme rows"] / len(x) * 100).round(1)
    logger.info("Carnaval se pehle ke Sundays: %s", sorted(str(c.date()) for c in carnaval_sunday if c >= CLEAN_START))
    out.table(reasons, "20_extreme_reasons", "14. Extreme values ki pehli mumkin wajah")

    # Carnaval se pehle wala Sunday: company ki sales, us se pichle aur agle Sunday ke muqable
    sun = d.daily_total
    rows = []
    for dt in sorted(carnaval_sunday):
        if dt - pd.Timedelta(days=7) in sun.index and dt + pd.Timedelta(days=7) in sun.index:
            ref = (sun[dt - pd.Timedelta(days=7)] + sun[dt + pd.Timedelta(days=7)]) / 2
            rows.append({"sunday": dt.date(), "sales": int(sun[dt]), "neighbour Sundays avg": int(ref),
                         "difference %": round((sun[dt] / ref - 1) * 100, 1),
                         "recording regime": "partial" if dt in d.partial_dates else "clean"})
    out.table(pd.DataFrame(rows).set_index("sunday") if rows else pd.DataFrame(), "20_carnaval_sunday_effect",
              "14. Carnaval se pehle Sunday vs aas paas ke Sundays")


# ======================================================================================
# 15. Forecast window vs history
# ======================================================================================
def section_15_forecast_window(d: EDAData, out: Outputs, res: dict[str, Any]) -> None:
    """
    Agar future ke inputs past se bohat alag hon (covariate shift) to accuracy gir sakti hai. Test ke 16 din
    ki sales hamare paas nahi, lekin inputs (promotion plan, calendar, holidays, stores) maloom hain, is liye
    unhein history se compare karna leakage nahi.

    Insight 15 (forecast window):
    - Test ke 16 dinon mein ~44% rows par promotion; pichle saal inhi dinon ~36%, 2015 mein us se kam. Forecast
      window history ke upar wale kinare par => halka covariate shift.
    - Test window mein koi national holiday/event nahi. Sirf Ambato (stores 23, 50) ki local chhutti 24 August.
    - Do payday din: 16 August (15 tareekh ki tankhwah ke agle din) aur 31 August (mahine ka aakhri din).
    - Store 52 test ke waqt sirf ~4 mahine purana (Section 17).
    - Asar (Phase 06): validation window (31 Jul-15 Aug) ka promotion level (~44%) bhi test ke qareeb => validation
      accuracy test ka acha andaza. Promotion features series ke apne normal level ke relative banayein.
    """
    master, hist = d.master, d.hist

    def promo_stats(df: pd.DataFrame) -> pd.Series:
        return pd.Series({"% rows with promotion": (df["onpromotion"] > 0).mean() * 100,
                          "avg items on promotion": df["onpromotion"].mean()})

    windows = {
        "16-31 Aug 2015": master[master["date"].between("2015-08-16", "2015-08-31")],
        "16-31 Aug 2016": master[master["date"].between("2016-08-16", "2016-08-31")],
        "Train ke aakhri 16 din": hist[hist["date"] > hist["date"].max() - pd.Timedelta(days=HORIZON_DAYS)],
        "Test (16-31 Aug 2017)": master[master["split"] == "test"],
    }
    window_cmp = pd.DataFrame({k: promo_stats(v) for k, v in windows.items()}).T.round(2)

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    bars = ax.bar(window_cmp.index, window_cmp["% rows with promotion"], color=[GRAY, GRAY, BLUE, ORANGE], width=0.55)
    for b, v in zip(bars, window_cmp["% rows with promotion"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontsize=9, color=INK2)
    ax.set_title("Promotion ka level: forecast window vs pehle ke isi tarah ke din")
    ax.set_ylabel("% rows with onpromotion > 0")
    ax.yaxis.set_major_formatter(pct)
    ax.set_ylim(0, window_cmp["% rows with promotion"].max() + 14)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.legend(handles=[Patch(facecolor=GRAY, label="Pichle saalon ke isi tarah ke din"),
                       Patch(facecolor=BLUE, label="Train ke aakhri 16 din"),
                       Patch(facecolor=ORANGE, label="Test window")], loc="upper left", ncol=3)
    out.fig(fig, "21_forecast_window_promotions")

    test_rows = master[master["split"] == "test"]
    store52_age = (test_rows["date"].min() - master.loc[master["store_nbr"] == 52, "store_open_date"].iloc[0]).days
    logger.info("Test window ke holidays: %s",
                test_rows.loc[test_rows["is_holiday"] == 1, ["date", "city"]].drop_duplicates().values.tolist())
    logger.info("Test window mein national holiday/event: %d / %d", int(test_rows["is_national_holiday"].max()),
                int(test_rows["is_national_event"].max()))
    logger.info("Sab se naya store: 52, test shuru hone par umar: %d din", store52_age)
    logger.info("Test window mein payday din: 31 August (mahine ka aakhri din); 16 August = 15 tareekh ki tankhwah ke agle din")
    out.table(window_cmp, "21_forecast_window_promotions", "15. Promotion level: forecast window vs history")


# ======================================================================================
# 16. Local aur regional holidays
# ======================================================================================
def section_16_local_holidays(d: EDAData, out: Outputs, res: dict[str, Any]) -> None:
    """
    Tareeqa: chhutti wale store ki sales usi weekday ke 2-3 hafte pehle/baad ke muqable kitni badli, MINUS usi
    din baqi stores kitne badle (company-wide asar, jaise mahine ki tareekh, hat jata hai).

    Insight 16 (local/regional holidays):
    - Local chhuttiyon ka median asar ~zero (-1% se -2%). Aadhe din -11% se +10% ke darmiyan, baqi bohat bikhre
      hue (-62% se +87%). Regional chhuttiyan sirf 15 store-din, asar pakka nahi.
    - Ambato (24 August, test window mein): pichle saalon mein stores 23 aur 50 ka asar -10% se +13% => test
      window mein bara farq aane ki umeed nahi.
    - Asar (Phase 05): `is_local_holiday`, `is_regional_holiday` kamzor features: rakhenge, lekin zyada asar
      ki umeed nahi; Phase 06 mein test karenge ke accuracy barhate hain ya sirf noise hain.
    """
    clean = d.clean
    sd = clean.groupby(["date", "store_nbr"]).agg(s=("sales", "sum"), loc=("is_local_holiday", "max"),
                                                   reg=("is_regional_holiday", "max")).reset_index()
    piv = sd.pivot_table(index="date", columns="store_nbr", values="s")
    store_city = d.master.drop_duplicates("store_nbr").set_index("store_nbr")["city"].astype(str)

    res_rows = []
    for (dt, st), g in sd[(sd["loc"] == 1) | (sd["reg"] == 1)].groupby(["date", "store_nbr"]):
        refs = [r for r in (dt + pd.Timedelta(weeks=k) for k in (-3, -2, 2, 3)) if r in piv.index]
        if len(refs) < 2 or pd.isna(piv.loc[dt, st]):
            continue
        own = piv.loc[dt, st] / piv.loc[refs, st].mean()
        others = (piv.loc[dt] / piv.loc[refs].mean()).drop(st).median()
        res_rows.append({"date": dt, "store": st, "city": store_city[st],
                         "kind": "local" if g["loc"].iloc[0] else "regional", "effect_%": (own / others - 1) * 100})
    loc_eff = pd.DataFrame(res_rows)
    median_eff = loc_eff["effect_%"].median()

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.hist(loc_eff["effect_%"].clip(-60, 60), bins=30, color=BLUE, edgecolor="white", linewidth=0.6,
            label="Store-days")
    ax.axvline(0, color=GRAY, linewidth=1, label="Koi farq nahi (0%)")
    ax.axvline(median_eff, color=ORANGE, linewidth=2, label=f"Median {median_eff:+.1f}%")
    ax.set_title("Local/regional chhutti wale store ki sales, baqi stores ke muqable (har store-din)")
    ax.set_xlabel("% difference (±60% par clip)")
    ax.set_ylabel("Store-days")
    ax.legend(loc="upper left")
    out.fig(fig, "22_local_holiday_effects")
    out.table(loc_eff.groupby("kind")["effect_%"].describe().round(1), "22_local_holiday_effect_summary",
              "16. Local/regional holiday effect (summary)")
    out.table(loc_eff[loc_eff["city"] == "Ambato"].assign(**{"effect_%": lambda x: x["effect_%"].round(1)})
              .set_index("date"), "22_ambato_holiday_history", "16. Ambato (test window wali chhutti) ki history")


# ======================================================================================
# 17. Naye stores (store 52)
# ======================================================================================
def section_17_new_stores(d: EDAData, out: Outputs, res: dict[str, Any]) -> None:
    """
    Store 52 April 2017 mein khula; test ke waqt sirf ~4 mahine purana hoga. 2015 ke baad khulne wale stores
    ki history dekhi gayi (opening ka josh, phir thehrao).

    Insight 17 (naye stores):
    - 6 mein se 5 naye stores pehle 4 hafton mein baad ke muqable zyada bechte hain (9% se 47% tak, "grand
      opening" asar), phir ek level par thehar jate hain. Har store ka pattern alag.
    - Store 52 test ke waqt opening ke josh se aage nikal chuka hoga, lekin history sirf ~3.5 mahine => "pichle
      saal ki sales" (lag 364) jaisa feature us ke liye maujood hi nahi.
    - Asar (Phase 05/06): "store ki umar" (days since opening) feature; store 52 ki accuracy alag se check.
    """
    open_dates = d.master.drop_duplicates("store_nbr").set_index("store_nbr")["store_open_date"]
    new_stores = open_dates[open_dates > "2015-01-01"].sort_values()
    sd_all = d.hist.groupby(["store_nbr", "date"])["sales"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10, 4))
    ramp = {}
    other_stores = [str(s) for s in new_stores.index if s != 52]
    for st, od in new_stores.items():
        x = sd_all[(sd_all["store_nbr"] == st) & (sd_all["date"] >= od)].copy()
        x["week"] = (x["date"] - od).dt.days // 7
        w = x.groupby("week")["sales"].mean().loc[:30]
        ref = w.loc[8:15].mean()                                    # opening ke 2-4 mahine baad ka level = 1.00
        ramp[st] = w / ref
        is52 = st == 52
        ax.plot(w.index, w / ref, color=BLUE if is52 else LIGHT_GRAY, linewidth=2.4 if is52 else 1.4,
                zorder=3 if is52 else 2, label="Store 52 (Apr 2017 mein khula)" if is52 else None)
        if is52:
            ax.text(w.index[-1] + 0.4, (w / ref).iloc[-1], "Store 52", fontsize=8.5, color=INK, va="center")
    ax.plot([], [], color=LIGHT_GRAY, linewidth=1.4, label="Baqi naye stores (" + ", ".join(other_stores) + ")")
    ax.axhline(1, color=GRAY, linewidth=1, label="Reference level (hafte 8-15 ka average = 1.00)")
    ax.set_title("Naye stores: khulne ke baad har hafte ki sales (hafte 8-15 ka average = 1.00)")
    ax.set_xlabel("Khulne ke baad hafte")
    ax.set_ylabel("Index")
    ax.set_xlim(-0.5, 33)
    ax.legend(loc="upper right")
    out.fig(fig, "23_new_store_ramp_up")
    first_weeks = pd.Series({st: r.loc[0:3].mean() for st, r in ramp.items()}, name="weeks 0-3 index").round(2)
    out.table(first_weeks, "23_new_store_first_weeks", "17. Pehle 4 hafton ka index (1.00 = hafte 8-15 ka level)")


# ======================================================================================
# 18. Co-movement
# ======================================================================================
def section_18_comovement(d: EDAData, out: Outputs, res: dict[str, Any]) -> None:
    """
    Agar saari series ek sath upar-neeche hoti hain (payday, mausam, ma'ashi halaat) to ek global model aur
    company-level features faida denge. Har series ke rozana utaar-charhao (trend hata kar) ka company ke
    average utaar-charhao se correlation nikala gaya.

    Insight 18 (co-movement):
    - Aam series ka company ke sath correlation ~0.4: rozana utaar-charhao ka ek hissa company-wide, ek apna.
    - Bari aur roz bikne wali families zyada ek sath chalti hain; chhoti aur kabhi kabhi bikne wali apni marzi se.
    - Asar (Phase 06): EK GLOBAL MODEL (saari 1,782 series ka) sahi chunao hai: company-wide patterns ek series
      se doosri mein seekhta hai. Company/family-level lag features bhi madad kar sakte hain.
    """
    c = d.clean[d.clean["date"] < d.hist["date"].max() - pd.Timedelta(days=14)].copy()
    c["y"] = np.log1p(c["sales"])
    c["y"] = c["y"] - c.groupby(["store_nbr", "family"])["y"].transform(
        lambda v: v.rolling(28, center=True, min_periods=14).mean())
    company_move = c.groupby("date")["y"].mean().rename("company")
    c = c.join(company_move, on="date")
    flat = c.groupby(["store_nbr", "family"])["y"].transform("std").fillna(0) == 0     # bilkul flat series
    co = c[~flat].groupby(["store_nbr", "family"]).apply(lambda g: g["y"].corr(g["company"])).dropna()

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.hist(co, bins=40, color=BLUE, edgecolor="white", linewidth=0.6, label="Series (store x family)")
    ax.axvline(co.median(), color=ORANGE, linewidth=2, label=f"Median {co.median():.2f}")
    ax.set_title("Har series ka company ke saath ek sath chalna (correlation)")
    ax.set_xlabel("Correlation with company-wide daily movement")
    ax.set_ylabel("Series")
    ax.legend(loc="upper left")
    out.fig(fig, "24_series_co_movement")

    by_fam = co.groupby(level="family").median().sort_values()
    logger.info("Series: %s | median correlation: %.2f | 75%% series ka correlation > %.2f",
                f"{len(co):,}", co.median(), co.quantile(0.25))
    extremes = pd.DataFrame({"sab se kam": by_fam.head(5).round(2).values, "family (kam)": by_fam.head(5).index,
                             "sab se zyada": by_fam.tail(5).round(2).values, "family (zyada)": by_fam.tail(5).index})
    out.table(extremes, "24_co_movement_by_family", "18. Co-movement: sab se kam aur sab se zyada families")


# ======================================================================================
# 19. Stock-out check
# ======================================================================================
def section_19_stockouts(d: EDAData, out: Outputs, res: dict[str, Any]) -> None:
    """
    Inventory data nahi. Lekin agar koi series roz bikti hai aur achanak kisi din zero ho jaye (store khula ho),
    to aksar yeh stock-out ki nishani hai. Aise din zyada hon to model ghalat seekhega ke demand zero thi.

    Insight 19 (stock-outs):
    - Roz bikne wali series mein zero sales takreeban hoti hi nahi (0.01% se kam) => POORA stock-out bara masla nahi.
    - Limitation (imaandari se): AADHA stock-out (maal kam pada, sales kam hui) is data se pakra nahi ja sakta
      kyunke inventory data nahi. Yeh dataset ki limitation hai jo README mein likhni hai.
    """
    clean = d.clean
    med_sales = clean.groupby(["store_nbr", "family"])["sales"].transform("median")
    busy = clean[med_sales >= 20]                                   # aam din mein 20+ units bechne wali series
    zeros = busy[busy["sales"] == 0]
    logger.info("Busy series (median >= 20 units): %s | rows: %s",
                f"{busy.groupby(['store_nbr', 'family']).ngroups:,}", f"{len(busy):,}")
    logger.info("Un mein zero sales wale din: %s (%.3f%%)", f"{len(zeros):,}", len(zeros) / max(len(busy), 1) * 100)
    logger.info("Kisi ek din 10 ya zyada busy series zero: %d din", int((zeros.groupby('date').size() >= 10).sum()))
    out.table(zeros.groupby("family").size().sort_values(ascending=False).head(5).to_frame("zero days"),
              "25_stockout_zero_days_by_family", "19. Busy series mein zero-sales din (top 5 families)")


# ======================================================================================
# 20. Khulasa: business insights aur aage ke faisle
# ======================================================================================
BUSINESS_INSIGHTS = [
    ("Weekend (Sat/Sun) sab se bare din, Thursday sab se kamzor", "Thursday/Friday ko shelves bharna, weekend staff barhana"),
    ("December saal ka peak, khas taur par 20-24 December", "Christmas se 2 hafte pehle inventory barhana"),
    ("Tankhwah ke foran baad (1-5 aur 16 tareekh, mahine ka aakhri din) sales zyada", "Payday ke aas paas stock aur promotions"),
    ("Volume chand families mein: GROCERY I, BEVERAGES, PRODUCE", "Accuracy ki nigrani sab se pehle inhi families par"),
    ("Stores ke size mein bohat bara farq; Quito aur Guayaquil haavi", "Har store ka forecast uske apne level par, ek size sab par nahi"),
    ("Promotion wale dinon par sales zyada (correlation, causation nahi)", "Promotion ka asli asar naapne ke liye controlled tests"),
    ("Holidays ke asar ulte bhi hain (+50% se -100%)", "Har holiday ki planning us ke pichle asar se"),
    ("2013-May 2015 ka kuch data aadha record hua (BEVERAGES waghaira)", "Purane reports par bharosa na karein; analysis aur training June 2015 se"),
    ("Oil price ka sales par koi asli asar nahi dikha", "Oil ko planning ka driver na samjha jaye"),
    ("Carnaval se pehle wale Sunday sales 25-34% kam", "Us Sunday ke liye kam stock aur staff"),
    ("Naye stores pehle hafton mein zyada bechte hain, phir thehar jate hain", "Opening ke waqt extra stock, phir normal"),
    ("Forecast window mein promotion history se zyada (~44%)", "Promotion ke asar ki nigrani"),
]

PHASE_05_DECISIONS = [
    ("Sales history: lag 16/21/28/35/42, lag 364, rolling averages (16 din pehle khatam)", "Section 12 (sirf lag >= 16 allowed)"),
    ("Same-weekday averages", "Section 3, 12 (weekly pattern)"),
    ("Calendar: day of week, day of month, payday distance, month, days to Christmas", "Section 3, 4, 5, 9"),
    ("Store/family identity aur attributes", "Section 6, 7"),
    ("Promotion features + `is_promo_tracked`", "Section 8"),
    ("Holiday-specific past effect, `is_earthquake_period`", "Section 9"),
    ("Oil (sirf lagged, test ke baad)", "Section 10 (kamzor umeed)"),
    ("Transactions: NAHI (same day), sirf 16+ din purani agar test mein faida de", "Section 11"),
    ("Carnaval se pehle ka Sunday (naya event)", "Section 14"),
    ("Store ki umar (days since opening)", "Section 17"),
    ("Promotion relative to series ka normal level", "Section 8, 15"),
    ("Local/regional holiday flags: kamzor, test karna", "Section 16"),
]

PHASE_06_DECISIONS = [
    "Target ko `log1p(sales)` par train karna (Section 1).",
    "Metrics: RMSLE aur WAPE, saath mein family, store aur demand-class ke hisaab se (Section 1, 6).",
    "Har model ko baseline (pichle hafton ki average) se behtar hona zaroori hai.",
    "Training data 1 June 2015 se (Section 2b); purana data sirf lag features ke liye.",
    "Ek global model saari series ke liye (Section 18). Store 52 ki accuracy alag se check (Section 17).",
]

MISSING_VALUE_DECISION = ("Sirf 3 columns mein missing values hain, aur teeno mein khaali hona khud information hai "
                          "(Section 0b). Koi imputation nahi.")

PHASE_03_INPUT = [
    "Extreme values ki list aur un ke din (Section 14), jinhein wajah ke saath classify karna hai.",
    "Yaad rahe: Christmas, 2 January, holidays, promotions aur zalzala asli demand hain, ghaltiyan nahi.",
]


def write_conclusions(path: Path) -> None:
    """Notebook ke aakhri markdown cell ko `reports/02_pre_eda_conclusions.md` mein likho."""
    lines = ["# Pre-EDA khulasa: business insights aur aage ke faisle", "",
             "## Business insights (planning team ke liye)", "",
             "| # | Insight | Business action |", "|---|---|---|"]
    lines += [f"| {i} | {a} | {b} |" for i, (a, b) in enumerate(BUSINESS_INSIGHTS, 1)]
    lines += ["", "## Phase 05 (feature engineering) ke faisle, sab Pre-EDA se nikle", "",
              "| Feature group | Kyun (kis section se) |", "|---|---|"]
    lines += [f"| {a} | {b} |" for a, b in PHASE_05_DECISIONS]
    lines += ["", "## Phase 06 (model aur evaluation) ke faisle", ""] + [f"- {x}" for x in PHASE_06_DECISIONS]
    lines += ["", "## Missing values ka faisla", "", f"- {MISSING_VALUE_DECISION}"]
    lines += ["", "## Phase 03 (outliers) ke liye input", ""] + [f"- {x}" for x in PHASE_03_INPUT]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def section_20_summary(d: EDAData, out: Outputs, res: dict[str, Any]) -> None:
    """Key numbers JSON mein, Phase 03 ke liye extreme rows CSV mein, aur conclusions markdown mein save karo."""
    s, o, acf = res["s"], res["o"], res["acf"]
    dow_index, month_index, cum = res["dow_index"], res["month_index"], res["cum"]
    summary_cls, lift, weekly, chg = res["summary_cls"], res["lift"], res["weekly"], res["chg"]

    summary = {
        "training_period": [str(d.hist["date"].min().date()), str(d.hist["date"].max().date())],
        "demand_rows": int(len(d.demand)),
        "target": {"median": float(s.median()), "zero_share": round(float((s == 0).mean()), 4),
                   "skew_raw": round(float(s.skew()), 2), "skew_log1p": round(float(np.log1p(s).skew()), 2)},
        "day_of_week_index": {DOW_NAMES[i]: round(float(v), 3) for i, v in dow_index.items()},
        "december_index": round(float(month_index[12]), 3),
        "clean_start": str(CLEAN_START.date()),
        "partial_recording_days_in_history": int(len(d.partial_dates)),
        "top3_family_share_pct": round(float(cum.iloc[2]), 1),
        "demand_classes_pct_series": summary_cls["% of series"].round(1).to_dict(),
        "demand_classes_pct_units": summary_cls["% of units"].round(1).to_dict(),
        "median_promo_lift_pct": round(float(lift.median()), 1),
        "oil_corr_levels": round(float(weekly["sales"].corr(weekly["oil"])), 2),
        "oil_corr_changes": round(float(chg["sales"].corr(chg["oil"])), 2),
        "acf": {f"lag_{k}": round(float(acf[k]), 3) for k in [1, 7, 16, 21, 28, 364]},
        "outliers_first_look": {"rows_checked": int(res["n_eval"]), "extreme_rows": int(o["extreme"].sum()),
                                "high": int((o["direction"] == "high").sum()),
                                "low": int((o["direction"] == "low").sum())},
    }
    summary_path = out.reports_dir / "02_pre_eda_summary.json"
    extreme_path = out.reports_dir / "02_pre_eda_extreme_values.csv"
    conclusions_path = out.reports_dir / "02_pre_eda_conclusions.md"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    # Phase 03 ke liye extreme rows ki list save
    (o.loc[o["extreme"], ["date", "store_nbr", "family", "sales", "baseline", "robust_z", "direction", "onpromotion"]]
     .to_csv(extreme_path, index=False))
    write_conclusions(conclusions_path)

    logger.info("Saved: %s, %s and %s", summary_path, extreme_path, conclusions_path)
    logger.info("Charts (%d): %s", len(list(out.fig_dir.glob('*.png'))), sorted(p.name for p in out.fig_dir.glob("*.png")))


# ======================================================================================
# Entry point
# ======================================================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 02: Pre-EDA for the Retail Demand Forecasting project.")
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"), help="Phase 01 ke processed files ka folder")
    p.add_argument("--raw-dir", type=Path, default=None, help="holidays_events.csv wala folder (default: auto-search)")
    p.add_argument("--reports-dir", type=Path, default=Path("reports"), help="Reports/figures ka output folder")
    p.add_argument("--show", action="store_true", help="Charts screen par bhi dikhao (default: sirf PNG save)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    if not args.show:
        plt.switch_backend("Agg")                       # server/terminal par bhi bina display ke chale
    apply_plot_style()

    out = Outputs(args.reports_dir, args.show)
    logger.info("Charts folder: %s", out.fig_dir)
    data = load_data(args.processed_dir)
    res: dict[str, Any] = {}

    # Sections ka tarteeb notebook jaisa hi hai; kuch sections pichle sections ke natije (`res`) use karte hain.
    pipeline = [
        ("0b/0c. Data overview", section_0_overview),
        ("1. Target variable", section_1_target),
        ("2. Company sales trend", section_2_trend),
        ("2b. Data recording regimes", section_2b_recording_regimes),
        ("3. Weekly seasonality", section_3_weekly),
        ("4. Monthly seasonality", section_4_monthly),
        ("5. Payday effect", section_5_payday),
        ("6/6b. Families aur demand classes", section_6_families),
        ("7. Stores", section_7_stores),
        ("8. Promotions", section_8_promotions),
        ("9. Holidays, Christmas, zalzala", lambda d, o, r: section_9_holidays(d, o, r, args.raw_dir)),
        ("10. Oil price", section_10_oil),
        ("11. Transactions aur basket", section_11_transactions),
        ("12. Autocorrelation", section_12_autocorrelation),
        ("13. Stationarity", section_13_stationarity),
        ("14. Outliers ki pehli nazar", section_14_outliers),
        ("15. Forecast window vs history", section_15_forecast_window),
        ("16. Local/regional holidays", section_16_local_holidays),
        ("17. Naye stores (store 52)", section_17_new_stores),
        ("18. Co-movement", section_18_comovement),
        ("19. Stock-out check", section_19_stockouts),
        ("20. Summary aur outputs", section_20_summary),
    ]
    for title, fn in pipeline:
        logger.info("=" * 20 + " Section %s " + "=" * 20, title)
        fn(data, out, res)

    logger.info("Phase 02 Pre-EDA complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
