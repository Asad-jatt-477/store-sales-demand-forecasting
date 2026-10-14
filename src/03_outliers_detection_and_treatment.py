#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Retail Demand Forecasting System
Phase 03: Outliers Detection and Treatment
===========================================

Project ka sequence
-------------------
01  01_data_preparation                          Master dataset aur flags
02  02_pre_eda                                   Data ki nature samajhna, outliers ki pehli nazar
03  03_outliers_detection_and_treatment (yeh)    Outliers dhoondna (2 visual + 2 statistical tareeqe), har ek ki
                                                 WAJAH saboot se pehchanna, aur sirf wahi theek karna jo asli demand nahi
04  04_post_eda                                  Treatment ke baad check, aur features ka faisla

Sab se aham usool
-----------------
Retail mein zyada tar "outliers" ASLI DEMAND hote hain (Christmas, holidays, promotions). Model ko inhi dinon ka
forecast karna seekhna hai. Is liye koi value sirf is wajah se nahi badli jayegi ke woh bari ya chhoti hai. Value tab
badlegi jab data khud saboot de ke yeh aam customers ki demand nahi thi.

Kahani ki misaal: ek dukaan par roz ~100 log aate hain. Ek din bikri 30 guna ho gayi.
  1. Kya us din customers bhi zyada aaye, ya doosri dukanon par bhi yahi hua, ya promotion thi? Agar haan => ASLI rush, rakho.
  2. Agar customers utne hi aaye, koi aur dukaan nahi uchhli, koi promotion nahi => kisi ek bare order (ya ghalat entry)
     ka nateeja. Yeh shak wala case hai.
Shak wale case ko bhi andaze se nahi chheda jata. FORECAST TEST (Section 11) batata hai ke isay badalne se forecast behtar
hota hai ya kharab. Is phase mein jo rules kaghaz par theek lagte thay, woh test mein noise se zyada faida nahi de sake,
is liye nahi lagaye gaye.

Sections
--------
0  Setup aur scope                          7  Wajah ki tafteesh (footprints)
1  Visual 1: Boxplot (har family)           8  Khaas cases ka saboot (store 2, store 1, LIQUOR, MEATS, L0)
2  Visual 2: Time-series + normal band      9  Treatment rules (data ki nature ke mutabiq)
3  Statistical 1: IQR (har series)         10  Candidate treatment aur audit log
4  Statistical 2: Contextual robust z      11  Test: har treatment alag, 4 backtest windows, noise ke muqable
5  Dono tareeqon ka muqabla                12  Final treatment aur pehle vs baad
6  Intermittent series                     13  Files save karna        14  Khulasa

Inputs  : <processed-dir>/retail_demand_master.parquet
          <processed-dir>/store_day_transactions.parquet
Outputs : <processed-dir>/retail_demand_master_treated.parquet   (5 naye columns, asli `sales` waisa hi)
          reports/figures/03_outliers/*.png                      (numbered charts)
          reports/tables/03_outliers/*.csv                       (har table ki CSV)
          reports/03_outlier_treatment_log.csv                   (har candidate + final faisla)
          reports/03_outliers_summary.json                       (key numbers)
          reports/03_outliers_conclusions.md                     (khulasa aur Phase 04 ke notes)

Usage
-----
    python 03_outliers_detection_and_treatment.py
    python 03_outliers_detection_and_treatment.py --processed-dir processed --reports-dir reports
    python 03_outliers_detection_and_treatment.py --show     # charts screen par bhi dikhao

Note: Section 11 mein 4 windows x 4 variants = 16 LightGBM models train hote hain, is liye yeh script lambi chalti hai.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*vert: bool was deprecated.*")
warnings.filterwarnings("ignore", message=".*obj.round has no effect.*")
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)

logger = logging.getLogger("outliers")

# --------------------------------------------------------------------------------------
# Project constants
# --------------------------------------------------------------------------------------
CLEAN_START = pd.Timestamp("2015-06-01")          # Phase 02, Section 2b ka faisla
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Detection / classification thresholds (Section 4, 7)
Z_THR = 5.0                                        # |robust z| > 5 => extreme
SHARE, DAY_SHARE, TX_HIGH, TX_LOW, PROMO_X = 0.20, 0.05, 1.3, 0.5, 2.0
# Treatment (Section 10)
POISSON_P = 0.001
TREAT_ACTIONS = ["cap", "replace_stockout", "replace_disruption"]

# Detection ke liye zaroori columns (Section 4)
COLS = ["store_nbr", "family", "date", "sales", "onpromotion", "is_holiday", "is_national_event",
        "is_holiday_season_day", "is_earthquake_period"]

# --------------------------------------------------------------------------------------
# Chart style: ek jaisa, saaf aur professional (grid lines bilkul nahi)
# --------------------------------------------------------------------------------------
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
GRAY, LIGHT_GRAY = "#9a9994", "#d9d8d4"
INK, INK2 = "#0b0b0b", "#52514e"

thousands = mtick.FuncFormatter(lambda x, _: f"{x:,.0f}")


def apply_plot_style() -> None:
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


class Outputs:
    """Charts, tables aur reports ko save karne wali ek jagah."""

    def __init__(self, reports_dir: Path, show: bool) -> None:
        self.reports_dir = reports_dir
        self.fig_dir = reports_dir / "figures" / "03_outliers"
        self.table_dir = reports_dir / "tables" / "03_outliers"
        self.show = show
        for p in (self.fig_dir, self.table_dir):
            p.mkdir(parents=True, exist_ok=True)

    def fig(self, fig: plt.Figure, name: str) -> None:
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
class Data:
    master: pd.DataFrame
    store_tx: pd.DataFrame
    train_end: pd.Timestamp
    carnaval_sundays: set
    scope: pd.Series              # boolean mask: treatment/detection ke scope ki rows


# ======================================================================================
# 0. Setup aur scope
# ======================================================================================
def load_data(processed_dir: Path) -> Data:
    """
    Scope: kaun si rows, aur kyun?
    - SIRF training ki clean period (June 2015 se 30 July 2017), sirf `is_trainable == 1` rows. Model inhi rows se
      seekhega. Band store, pre-opening aur partial-recording ki rows pehle hi training se bahar hain (Phase 01/02).
    - Validation (31 Jul - 15 Aug 2017) ko haath nahi lagaya jayega: woh hamara "mock exam" hai. Agar uske jawab bhi
      "saaf" kar diye to accuracy jhooti aayegi.
    - Test mein sales hain hi nahi, wahan kuch karne ko nahi.
    """
    master = pd.read_parquet(processed_dir / "retail_demand_master.parquet")
    master["family"] = master["family"].astype(str)
    store_tx = pd.read_parquet(processed_dir / "store_day_transactions.parquet")[["date", "store_nbr", "transactions"]]

    train_end = master.loc[master["split"] == "train", "date"].max()

    # Carnaval (Monday) se pehle wala Sunday: Phase 02 ki khoj
    carnaval = master.loc[master["national_desc"].astype(str).str.contains("Carnaval"), "date"].drop_duplicates()
    carnaval_sundays = set(carnaval[carnaval.dt.dayofweek == 0] - pd.Timedelta(days=1))

    scope = (master["split"] == "train") & (master["is_trainable"] == 1) & (master["date"] >= CLEAN_START)
    logger.info("Training ki aakhri tareekh : %s", train_end.date())
    logger.info("Scope rows (treatment yahin hoga): %s of %s", f"{scope.sum():,}", f"{len(master):,}")
    logger.info("Validation rows (bilkul nahi chhuye jayenge): %s", f"{(master['split'] == 'valid').sum():,}")
    return Data(master, store_tx, train_end, carnaval_sundays, scope)


# ======================================================================================
# 1. Visual tareeqa 1: Boxplot (har family)
# ======================================================================================
def section_1_boxplot(data: Data, out: Outputs, res: dict[str, Any]) -> None:
    """
    Boxplot ka dabba beech ki 50% values dikhata hai, dabbe se bahar ke nuqte "outliers" kehlate hain. `log1p(sales)`
    liya gaya hai, warna GROCERY I ke saamne chhoti families nazar hi na aatin.

    Boxplot kya batata hai, aur kya nahi:
    - Har family ki "normal range" ek nazar mein dikhti hai. GROCERY I, BEVERAGES, PRODUCE sab se upar; BOOKS aur
      BABY CARE sab se neeche.
    - Lekin yeh tareeqa akela kaafi nahi: ek dabbe mein store 44 (sab se bara) aur store 32 (sab se chhota) dono ki
      sales hain, is liye chhote store ka aam din bhi "outlier" lag sakta hai. Har Sunday aur har December bhi.
    - Boxplot ~25 hazar rows (~1.9%) ko "outlier" kehta hai, ~4/5 dabbe ke UPAR. Upar wale nuqte kahan se aate hain:
        * Sab se bari wajah do chhoti families: HOME APPLIANCES (~49%) aur BABY CARE (~24%): aksar zero bechti hain,
          is liye dabba zero ke paas hai aur koi bhi aam sale "outlier" ban jati hai.
        * 10 sab se bare stores ka hissa ~41% (saari rows mein ~19%): 54 stores ek dabbe mein milne ka asar.
        * Sunday (~21% vs 14%) aur December (~11% vs 8%) bhi zyada: peak din bhi "outlier" lagte hain.
    - Natija: boxplot sirf "kahan dekhna hai" ka andaza deta hai. Faisla har series ke apne normal se hoga (Section 4).
    """
    d = data.master.loc[data.scope, ["store_nbr", "family", "date", "sales"]].copy()
    d["y"] = np.log1p(d["sales"])
    res["d"] = d
    order = d.groupby("family")["y"].median().sort_values().index

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.boxplot([d.loc[d["family"] == f, "y"].values for f in order], vert=False, widths=0.6,
               flierprops=dict(marker="o", markersize=1.5, markerfacecolor=RED, markeredgecolor="none", alpha=0.25),
               medianprops=dict(color=BLUE, linewidth=1.6), boxprops=dict(color=INK2), whiskerprops=dict(color=INK2),
               capprops=dict(color=INK2))
    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("log1p(sales)   (0 = zero sale, 5 ~ 150 units, 10 ~ 22,000 units)")
    ax.set_title("Boxplot: har family ki sales (saare stores aur din ek sath)")
    ax.legend(handles=[Patch(facecolor="white", edgecolor=INK2, label="Dabba: beech ki 50% values (Q1 se Q3)"),
                       Line2D([0], [0], color=BLUE, linewidth=1.6, label="Median"),
                       Line2D([0], [0], color=INK2, linewidth=1, label="Whiskers (1.5 x IQR tak)"),
                       Line2D([0], [0], marker="o", linestyle="", markersize=4, markerfacecolor=RED,
                              markeredgecolor="none", label="Outlier nuqte")],
              loc="lower right")
    out.fig(fig, "01_boxplot_by_family")

    def iqr_bounds(s: pd.Series) -> tuple[float, float]:
        q1, q3 = s.quantile(.25), s.quantile(.75)
        return q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)

    fam_n = d.groupby("family").size()
    whisker_out = d.groupby("family")["y"].apply(lambda s: ((s < iqr_bounds(s)[0]) | (s > iqr_bounds(s)[1])).mean() * 100)
    logger.info("Boxplot ke hisaab se 'outlier' rows: %s (%.2f%%)", f"{whisker_out.mul(fam_n).sum() / 100:,.0f}",
                (whisker_out * fam_n).sum() / len(d))
    low_side = d.groupby("family")["y"].apply(lambda s: (s < iqr_bounds(s)[0]).sum())
    logger.info("In mein se dabbe ke NEECHE wale nuqte: %s", f"{low_side.sum():,}")

    # Upar wale nuqte kahan se aate hain? (daawa nahi, check)
    qq1 = d.groupby("family")["y"].transform(lambda s: s.quantile(.25))
    qq3 = d.groupby("family")["y"].transform(lambda s: s.quantile(.75))
    up = d[d["y"] > qq3 + 1.5 * (qq3 - qq1)]
    store_rank = d.groupby("store_nbr")["sales"].sum().rank(ascending=False)
    top10 = store_rank[store_rank <= 10].index
    check = pd.DataFrame({
        "upar wale outliers mein %": [up["family"].eq("HOME APPLIANCES").mean(), up["family"].eq("BABY CARE").mean(),
                                      up["store_nbr"].isin(top10).mean(), (up["date"].dt.dayofweek == 6).mean(),
                                      (up["date"].dt.month == 12).mean()],
        "saari rows mein %": [d["family"].eq("HOME APPLIANCES").mean(), d["family"].eq("BABY CARE").mean(),
                              d["store_nbr"].isin(top10).mean(), (d["date"].dt.dayofweek == 6).mean(),
                              (d["date"].dt.month == 12).mean()]},
        index=["HOME APPLIANCES family", "BABY CARE family", "10 sab se bare stores", "Sunday", "December"]) * 100
    out.table(check.round(1), "01_boxplot_outlier_sources", "1. Boxplot ke upar wale outliers kahan se aate hain")


# ======================================================================================
# 2. Visual tareeqa 2: Time-series + normal band
# ======================================================================================
def weekday_normal(s: pd.Series) -> pd.Series:
    """Usi weekday ki 9 hafton (4 peechhe + 4 aage) ki median."""
    return s.groupby(s.index.dayofweek).transform(lambda x: x.rolling(9, center=True, min_periods=5).median())


def section_2_timeseries_band(data: Data, out: Outputs, res: dict[str, Any]) -> None:
    """
    Time series mein outlier ko waqt ke context mein dekhna zaroori hai. Neeche chaar series hain jo Pre-EDA mein shak
    wali lagi thin. Har series ke sath us ka "normal" (usi weekday ki pichle/agle 4 hafton ki median) aur ek band
    (0.5x se 1.8x) hai. Band se bahar ke din aankh se dikh jate hain.

    Aankh se kya dikhta hai (abhi sirf shak, saboot Section 7-8 mein):
    1. Store 2, GROCERY I: do din bilkul alag (26 Apr aur 2 May 2016, ~63 hazar aur ~1.25 lakh units, aam din ~4 hazar).
       Is liye scale log rakha gaya.
    2. Store 1, GROCERY I: neeche bohat se lal nuqte, aur yeh holidays par hain. Data sirf yeh dikhata hai ke holidays
       par is ke customers bhi kam hote hain (Section 8b); wajah likhna andaza hoga.
    3. Store 44, LIQUOR: kuch din achanak neeche, December mein upar.
    4. Store 36, SCHOOL SUPPLIES: aam din ~zero, school season mein bara uchhaal: band ke bahar, lekin har saal ka
       season hai, ghalti nahi.
    Natija: har "bahar" wala nuqta ek jaisa nahi (season, holiday, aur kuch waqai ajeeb). Ab statistics se poore data par naapte hain.
    """
    d = res["d"]

    def series(store: int, fam: str) -> pd.Series:
        s = d[(d["store_nbr"] == store) & (d["family"] == fam)].set_index("date")["sales"]
        return s.reindex(pd.date_range(CLEAN_START, data.train_end))

    examples = [(2, "GROCERY I", "Store 2, GROCERY I"), (1, "GROCERY I", "Store 1, GROCERY I"),
                (44, "LIQUOR,WINE,BEER", "Store 44, LIQUOR,WINE,BEER"),
                (36, "SCHOOL AND OFFICE SUPPLIES", "Store 36, SCHOOL AND OFFICE SUPPLIES")]
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 11), sharex=True)
    for ax, (st, fam, title) in zip(axes, examples):
        s = series(st, fam)
        nrm = weekday_normal(s)
        ax.fill_between(s.index, nrm * 0.5, nrm * 1.8, color=BLUE, alpha=0.12, linewidth=0,
                        label="Normal band (0.5x to 1.8x)")
        ax.plot(s.index, s, color=INK2, linewidth=0.7, label="Asli sales")
        outside = s[(s > nrm * 1.8) | (s < nrm * 0.5)]
        ax.scatter(outside.index, outside, s=10, color=RED, zorder=3, label="Band se bahar")
        ax.set_title(title)
        ax.yaxis.set_major_formatter(thousands)
        ax.legend(loc="upper left", ncol=3)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("sales (log scale)")
    out.fig(fig, "02_timeseries_normal_band")


# ======================================================================================
# 3. Statistical tareeqa 1: IQR (har series alag)
# ======================================================================================
def section_3_iqr(data: Data, out: Outputs, res: dict[str, Any]) -> None:
    """
    Har series (store x family) ke liye `log1p(sales)` ka Q1 aur Q3; jo value `Q3 + 3xIQR` se upar ya `Q1 - 3xIQR` se
    neeche ho woh outlier. 1.5 ki jagah 3 is liye ke sirf BOHAT door wali values pakri jayen. Har series alag, is liye
    store ke size ka masla khatam.

    IQR ki kamzori:
    - IQR series ki saari history ek sath dekhta hai. Jo family saal ke kuch hafton mein hi bikti hai (SCHOOL AND OFFICE
      SUPPLIES, school season) us ke season ke din "outlier" lagte hain, jab ke yeh business ka sab se aham waqt hai.
    - HOME APPLIANCES aur BABY CARE jaisi chhoti families, jin ka level waqt ke sath badla, bohat zyada flag hoti hain.
    - IQR "kis din" ko nahi dekhta: store 1 ka holiday wala din saal ke kul range ke andar ho sakta hai, is liye IQR
      use nahi pakarta.
    - Natija: IQR ek seedha baseline tareeqa hai, lekin season ko outlier samajhta hai => context wala tareeqa chahiye.
    """
    d = res["d"].sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    grp_y = d.groupby(["store_nbr", "family"])["y"]
    q1, q3 = grp_y.transform(lambda x: x.quantile(.25)), grp_y.transform(lambda x: x.quantile(.75))
    d["iqr_flag"] = ((d["y"] > q3 + 3 * (q3 - q1)) | (d["y"] < q1 - 3 * (q3 - q1))).astype(int)
    res["d"] = d
    logger.info("IQR outliers: %s rows (%.2f%%)", f"{d['iqr_flag'].sum():,}", d["iqr_flag"].mean() * 100)
    fl = d[d["iqr_flag"] == 1]
    logger.info("Upar wale: %.0f%%", (fl["y"] > q3[fl.index]).mean() * 100)
    out.table(pd.DataFrame({"IQR outliers": fl["family"].value_counts().head(8),
                            "family ki rows mein %": (fl["family"].value_counts() / d["family"].value_counts() * 100).round(2)}
                           ).dropna(), "03_iqr_outliers_by_family", "3. IQR outliers (top families)")

    by_month = fl.groupby(fl["date"].dt.month).size().reindex(range(1, 13), fill_value=0)
    school = (fl[fl["family"] == "SCHOOL AND OFFICE SUPPLIES"].groupby(fl["date"].dt.month).size()
              .reindex(range(1, 13), fill_value=0))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.bar(by_month.index, by_month, color=LIGHT_GRAY, label="Saari IQR outliers")
    ax.bar(school.index, school, color=ORANGE, label="In mein se SCHOOL AND OFFICE SUPPLIES")
    ax.set_xticks(range(1, 13), MONTH_NAMES)
    ax.set_title("IQR outliers kis mahine mein: season bhi 'outlier' ban jata hai")
    ax.set_ylabel("rows")
    ax.legend(loc="upper left")
    out.fig(fig, "03_iqr_outliers_by_month")


# ======================================================================================
# 4. Statistical tareeqa 2: Contextual robust z-score
# ======================================================================================
def detect(df: pd.DataFrame, tx: pd.DataFrame, carnaval_sundays: set) -> pd.DataFrame:
    """
    df: scope rows (store_nbr, family, date, sales, onpromotion, calendar flags). Robust-z + footprints return karta hai.

    Har value ko usi series ke, usi weekday ke, aas paas ke hafton se compare kiya jata hai:
      1. y = log1p(sales)
      2. Normal (baseline) = usi weekday ki 9 hafton (4 peechhe, 4 aage) ki median (weekly pattern, trend, season hat jate hain)
      3. Farq (residual) = y - baseline
      4. Robust z = (residual - median) / (1.4826 x MAD). MAD mazboot spread hai (std ki tarah chand bari values se
         nahi bigarta); 1.4826 se MAD normal data ke std ke barabar ho jata hai.
      5. |z| > 5 = extreme (normal distribution mein ~35 lakh mein ek baar: bohat sakht hadd)

    Leakage ka sawal: baseline mein "agle 4 hafte" bhi shaamil hain. Yeh feature nahi, sirf purani training sales ki
    SAFAI hai, aur sirf training data (30 July tak) par chalti hai. Validation ki koi value shaamil nahi (Section 11 ke
    backtest mein har window ke liye alag se nibhai gayi hai).
    """
    o = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True).copy()
    o["y"] = np.log1p(o["sales"])
    o["dow"] = o["date"].dt.dayofweek
    o["baseline"] = (o.groupby(["store_nbr", "family", "dow"])["y"]
                     .rolling(9, center=True, min_periods=5).median()
                     .reset_index(level=[0, 1, 2], drop=True))
    o["resid"] = o["y"] - o["baseline"]
    g = o.groupby(["store_nbr", "family"])["resid"]
    o["res_med"] = g.transform("median")
    o["res_mad"] = g.transform(lambda x: (x - x.median()).abs().median())
    o["rz"] = (o["resid"] - o["res_med"]) / (1.4826 * o["res_mad"].replace(0, np.nan))

    # --- Footprints (Section 7 mein tafseel) ---
    o["hi3"], o["lo3"] = (o["rz"] > 3).astype(float), (o["rz"] < -3).astype(float)
    o.loc[o["rz"].isna(), ["hi3", "lo3"]] = np.nan
    o = o.join(o.groupby(["date", "family"])[["hi3", "lo3"]].mean().add_prefix("fam_"), on=["date", "family"])
    o = o.join(o.groupby(["date", "store_nbr"])[["hi3", "lo3"]].mean().add_prefix("store_"), on=["date", "store_nbr"])
    o = o.join(o.groupby("date")[["hi3", "lo3"]].mean().add_prefix("day_"), on="date")
    t = tx[tx["date"] <= df["date"].max()].sort_values(["store_nbr", "date"]).copy()
    t["dow"] = t["date"].dt.dayofweek
    t["tx_base"] = (t.groupby(["store_nbr", "dow"])["transactions"].rolling(9, center=True, min_periods=5).median()
                    .reset_index(level=[0, 1], drop=True))
    t["tx_ratio"] = t["transactions"] / t["tx_base"]
    o = o.merge(t[["date", "store_nbr", "tx_ratio"]], on=["date", "store_nbr"], how="left")
    o["promo_med"] = o.groupby(["store_nbr", "family"])["onpromotion"].transform("median")
    o["calendar"] = ((o["is_holiday"] == 1) | (o["is_national_event"] == 1) | (o["is_holiday_season_day"] == 1)
                     | o["date"].isin(carnaval_sundays))
    return o


def section_4_robust_z(data: Data, out: Outputs, res: dict[str, Any]) -> None:
    """
    Insight: 99% se zyada rows z = 0 ke aas paas hain. Extreme rows (lal lakeer ke bahar) bohat kam hain.
    """
    t0 = time.time()
    o = detect(data.master.loc[data.scope, COLS], data.store_tx, data.carnaval_sundays)
    d = res["d"]
    o = o.merge(d[["store_nbr", "family", "date", "iqr_flag"]], on=["store_nbr", "family", "date"])
    o["rz_flag"] = (o["rz"].abs() > Z_THR).astype(int)
    res["o"] = o
    logger.info("(%.0fs)  Rows: %s | robust z ban saka: %s | nahi ban saka (MAD = 0): %s", time.time() - t0,
                f"{len(o):,}", f"{o['rz'].notna().sum():,}", f"{o['rz'].isna().sum():,}")
    logger.info("Robust-z extremes: %s (%.2f%%) | high: %s | low: %s", f"{o['rz_flag'].sum():,}",
                o["rz_flag"].sum() / o["rz"].notna().sum() * 100, f"{(o['rz'] > Z_THR).sum():,}",
                f"{(o['rz'] < -Z_THR).sum():,}")

    z = o["rz"].dropna().clip(-15, 15)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.hist(z, bins=np.arange(-15, 15.25, 0.25), color=BLUE, alpha=0.85, label="Rows (robust z)")
    ax.set_yscale("log")
    ax.axvline(-Z_THR, color=RED, linestyle="--", linewidth=1.2, label="Extreme hadd (|z| = 5)")
    ax.axvline(Z_THR, color=RED, linestyle="--", linewidth=1.2)
    ax.set_xlabel("robust z (±15 par kaata gaya)")
    ax.set_ylabel("rows (log scale)")
    ax.set_title("Robust z ki distribution: takreeban sab rows normal ke qareeb")
    ax.legend(loc="upper right")
    out.fig(fig, "04_robust_z_distribution")


# ======================================================================================
# 5. Dono statistical tareeqon ka muqabla
# ======================================================================================
def section_5_compare(data: Data, out: Outputs, res: dict[str, Any]) -> None:
    """
    Faisla (outputs se):
    - IQR ke zyada tar outliers (~10 hazar) un chhoti series mein hain jo aksar zero bechti hain (school supplies,
      home appliances, baby care): wahan har season ya level badalna "outlier" ban jata hai.
    - Baqi series mein SIRF-IQR wali rows apne aas paas ke hafton ke hisaab se ~normal thin (median |z| ~1.6) => false alarm.
    - SIRF-robust-z wali rows zyada tar normal se KAM hain (jaise store 1 ke holidays): IQR inhein nahi dekh sakta.
    - Robust z hamara PRIMARY detector hoga. IQR sirf un series ke liye jahan robust z ban hi nahi sakta (Section 6).
    """
    o = res["o"]
    both = o[o["rz"].notna()]
    cmp_tab = pd.DataFrame({
        "rows": [((both.iqr_flag == 1) & (both.rz_flag == 1)).sum(), ((both.iqr_flag == 1) & (both.rz_flag == 0)).sum(),
                 ((both.iqr_flag == 0) & (both.rz_flag == 1)).sum()]},
        index=["Dono ne pakra", "Sirf IQR ne pakra", "Sirf robust z ne pakra"])
    out.table(cmp_tab, "05_iqr_vs_robust_z", "5. IQR vs robust z")

    only_iqr = both[(both.iqr_flag == 1) & (both.rz_flag == 0)]
    only_rz = both[(both.iqr_flag == 0) & (both.rz_flag == 1)]
    logger.info("IQR ke kul %s outliers mein se %s intermittent series (zyada tar zero, robust z nahi banta) mein hain",
                f"{o['iqr_flag'].sum():,}", f"{o.loc[o['rz'].isna(), 'iqr_flag'].sum():,}")
    logger.info("Sirf-IQR rows, top families: %s", only_iqr["family"].value_counts().head(4).to_dict())
    logger.info("Sirf-IQR rows ka median |robust z|: %.1f (yani apne hafton ke hisaab se takreeban normal thin)",
                only_iqr["rz"].abs().median())
    logger.info("Sirf-robust-z rows: %.0f%% normal se KAM wale (din ka asar, jo IQR ki saal bhar wali range mein chhup jata hai)",
                (only_rz["rz"] < 0).mean() * 100)

    s36 = o[(o["store_nbr"] == 36) & (o["family"] == "SCHOOL AND OFFICE SUPPLIES")].set_index("date")
    s1 = o[(o["store_nbr"] == 1) & (o["family"] == "GROCERY I")].set_index("date")
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.8))
    for ax, ser, title in [(axes[0], s1, "Store 1, GROCERY I: robust z holiday ki dips pakarta hai, IQR nahi"),
                           (axes[1], s36, "Store 36, SCHOOL SUPPLIES (2016 se): IQR season ko outlier kehta hai")]:
        ax.plot(ser.index, ser["sales"], color=LIGHT_GRAY, linewidth=0.8, label="Rozana sales")
        a, b = ser[ser["iqr_flag"] == 1], ser[ser["rz_flag"] == 1]
        ax.scatter(a.index, a["sales"], s=36, facecolor="none", edgecolor=ORANGE, linewidth=1.3,
                   label="IQR outlier", zorder=3)
        ax.scatter(b.index, b["sales"], s=12, color=RED, label="Robust-z extreme", zorder=4)
        ax.set_title(title)
        ax.yaxis.set_major_formatter(thousands)
        ax.legend(loc="upper left", ncol=3)
    axes[1].set_xlim(pd.Timestamp("2016-01-01"), data.train_end)
    out.fig(fig, "05_iqr_vs_robust_z")


# ======================================================================================
# 6. Intermittent series
# ======================================================================================
def section_6_intermittent(data: Data, out: Outputs, res: dict[str, Any]) -> None:
    """
    Kuch series zyada tar din zero ya 1-2 units bechti hain: MAD = 0, is liye robust z nahi banta. Un ke liye IQR
    sirf NON-ZERO values par lagaya, taa ke zeros ki bheed hadd na bigaare.

    Faisla: intermittent outliers ko waise hi rakho (koi treatment nahi). Wajahen:
    1. Chhote numbers hain (median kuch units); kul units ka hissa 0.01% se bhi kam => forecast par asar na hone ke barabar.
    2. Chhoti, kabhi kabhi bikne wali families (home appliances, school supplies, beauty, baby care), saal ke har
       mahine mein phaile hue. Outlier ki median sirf ~2 units: zyada tar zero wale din ke beech kabhi kabhi 2-3 piece
       bik gaye => chhoti families ke qudrati utaar-charhao ka nishaan, kisi ajeeb waqiye ka nahi.
    3. Model `log1p` par train hoga: log scale par 2 se 30 units ka farq bhi chhota, model nahi bigarta.
    """
    o = res["o"]
    inter = o[o["rz"].isna()].copy()
    n_series = inter[["store_nbr", "family"]].drop_duplicates().shape[0]
    nz = inter[inter["sales"] > 0]
    g = nz.groupby(["store_nbr", "family"])["y"]
    nq1, nq3 = g.transform(lambda x: x.quantile(.25)), g.transform(lambda x: x.quantile(.75))
    inter_flag = nz[nz["y"] > nq3 + 3 * (nq3 - nq1)]
    logger.info("Intermittent series: %d | rows: %s | zero din: %.0f%%", n_series, f"{len(inter):,}",
                (inter["sales"] == 0).mean() * 100)
    logger.info("Non-zero IQR outliers: %s", f"{len(inter_flag):,}")
    logger.info("In ki kul units: %s = poore scope ki units ka %.4f%%", f"{inter_flag['sales'].sum():,.0f}",
                inter_flag["sales"].sum() / o["sales"].sum() * 100)
    if len(inter_flag):
        logger.info("Median sale in outliers: %.0f units | max: %.0f", inter_flag["sales"].median(), inter_flag["sales"].max())
    logger.info("Top families: %s", inter_flag["family"].value_counts().head(5).to_dict())
    logger.info("Mahine: %s", inter_flag["date"].dt.month.value_counts().sort_index().to_dict())


# ======================================================================================
# 7. Wajah ki tafteesh (footprints)
# ======================================================================================
def classify(o: pd.DataFrame, share: float = SHARE, day_share: float = DAY_SHARE) -> pd.Series:
    """
    Har extreme value (|z| > 5) ke liye sawal: kya is ka asar data mein kahin aur bhi dikhta hai? Asli events ke nishaan:

    | Footprint                     | Kaise naapa                                              | Matlab                                |
    | Same family, bohat se stores  | Usi din usi family ke >= 20% stores mein z > 3 (< -3)    | Company-wide wajah (event, campaign)  |
    | Poori company ka din          | Usi din saari series mein >= 5% ka z > 3 (< -3)          | Har jagah alag din (Mother's Day...)  |
    | Same store, bohat si families | Usi store ki >= 20% families mein z > 3 (< -3)           | Store ka rush / holiday aadat         |
    | Customers (transactions)      | Store transactions normal ka > 1.3x ya < 0.5x            | Waqai log zyada ya kam aaye?          |
    | Promotion surge               | onpromotion series ki median ka 2x se zyada              | Discount ki wajah se bikri            |
    | Calendar                      | Holiday, national event, Christmas season, Carnaval Sun. | Pehle se maloom din                   |

    Thresholds (20%, 5%, 1.3x, 0.5x, 2x) "aam din" ke muqable bohat bari hadd hain (aam din saari series ka median sirf
    ~1% z > 3 hota hai, is liye 5% = 5 guna). Sensitivity test (Section 7) dikhata hai ke hadd badalne se faisla kitna badalta hai.

    Insight 7 (lal = sirf candidate, final faisla Section 11 ke test se):
    - Zyada tar HIGH extremes ke nishaan milte hain (same family bohat se stores, company ka khaas din, store rush,
      promotion): asli demand.
    - LOW extremes bhi zyada tar company ya store ki calendar aadat hain (Section 8).
    - Lal categories sirf woh hain jin ka koi nishaan nahi mila, ya jahan store takreeban band tha.
    """
    tx = o["tx_ratio"].fillna(1.0)          # transactions na ho to "normal" maano
    hi, lo = o["rz"] > Z_THR, o["rz"] < -Z_THR
    promo = (o["onpromotion"] > PROMO_X * o["promo_med"]) & (o["onpromotion"] > 0)
    # Store takreeban band (customers aadhe se kam), aur yeh calendar ya company-wide din nahi
    disrupted_day = (tx < TX_LOW) & ~o["calendar"] & (o["day_lo3"] < day_share)
    cause = np.select(
        [hi & (o["fam_hi3"] >= share), hi & (o["day_hi3"] >= day_share),
         hi & ((o["store_hi3"] >= share) | (tx > TX_HIGH)), hi & promo, hi,
         disrupted_day & (o["rz"] < -3),
         lo & (o["fam_lo3"] >= share), lo & (o["day_lo3"] >= day_share),
         lo & ((o["store_lo3"] >= share) | (tx < TX_LOW)), lo],
        ["H1 Same family, bohat se stores", "H2 Poori company ka khaas din", "H3 Store-wide rush",
         "H4 Promotion surge", "H5 Isolated spike (bulk order / ghalat entry)",
         "L0 Store takreeban band (disruption)",
         "L1 Same family, bohat se stores", "L2 Poori company ka khaas din", "L3 Store-wide kami (holiday aadat)",
         "L4 Isolated drop (stock-out / record nahi hua)"],
        default="")
    return pd.Series(cause, index=o.index)


def section_7_causes(data: Data, out: Outputs, res: dict[str, Any]) -> None:
    """
    Sensitivity ka matlab: hadd ko kaafi badal kar bhi (10% se 30%) "isolated" rows ki ginti usi range mein rehti hai.
    Yani faisla kisi ek jaadui number par nahi tika. 20% / 5% beech ka, mutawazin chunaav hai.
    """
    o = res["o"]
    o["cause"] = classify(o)
    causes = o.loc[o["cause"] != "", "cause"].value_counts().sort_index().to_frame("rows")
    causes["%"] = (causes["rows"] / causes["rows"].sum() * 100).round(1)
    res["causes"] = causes
    out.table(causes, "07_outlier_causes", "7. Extreme values ki wajah")

    sens = []
    for sh in (0.10, 0.20, 0.30):
        for ds in (0.03, 0.05, 0.10):
            cz = classify(o, sh, ds)
            sens.append({"family/store share": sh, "day share": ds,
                         "H5 isolated spikes": int(cz.str.startswith("H5").sum()),
                         "L0 disruption rows": int(cz.str.startswith("L0").sum()),
                         "L4 isolated drops": int(cz.str.startswith("L4").sum())})
    out.table(pd.DataFrame(sens).set_index(["family/store share", "day share"]), "07_threshold_sensitivity",
              "7. Threshold sensitivity")

    cc = causes["rows"].sort_index(ascending=False)
    colors = [RED if k.startswith(("H5", "L4", "L0")) else BLUE for k in cc.index]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.barh(cc.index, cc.values, color=colors)
    for i, v in enumerate(cc.values):
        ax.text(v + 30, i, f"{v:,}", va="center", fontsize=8, color=INK2)
    ax.set_title("Extreme values ki wajah: neela = asli event (rakho), lal = treatment ke candidate")
    ax.set_xlabel("rows")
    ax.legend(handles=[Patch(facecolor=BLUE, label="Asli event (rakho)"),
                       Patch(facecolor=RED, label="Treatment ke candidate (H5, L4, L0)")], loc="lower right")
    out.fig(fig, "06_outlier_causes")


# ======================================================================================
# 8. Khaas cases ka saboot
# ======================================================================================
def section_8_special_cases(data: Data, out: Outputs, res: dict[str, Any]) -> None:
    """
    8a. Store 2, GROCERY I: 1,24,717 units (2 May 2016). Saboot:
    - 26 April (~63 hazar) aur 2 May (~1.25 lakh) ko sales normal (~4 hazar) ka 15-30 guna thi.
    - Lekin customers (transactions) bilkul normal thay (ratio ~1.0): agar 30 guna log aate to transactions bhi barhti.
    - Doosre stores mein GROCERY I normal thi, aur isi store ki baqi families bhi normal. Promotion bhi aam din jitni.
    - Natija: customers ki ginti normal, phir bhi bikri 15-30 guna => kuch customers ne bohat zyada khareeda (bulk) ya entry
      ghalat. Receipt-level data nahi, is liye in dono mein farq nahi ho sakta.
    - Andaza (data se sabit nahi): yeh din zalzale (16 April) ke foran baad ke hain, shayad imdaadi saamaan. Faisla is
      andaze par NAHI kiya gaya. Yeh bhi andaza tha ke aisi value `lag_16` ko bigaarti hai; Section 11 ke test mein ghalat nikla.

    "Din khisakne" wali recording ki ghalti? Agar ho to ek din bohat kam aur agla bohat zyada hota:
    - Isolated drop ke baad agla din upar nahi uchhalta (balkay thora neeche): stock-out jaisa pattern.
    - Isolated spike ke baad agla din neeche nahi girta (z < -2 ~4%); ~31% baar agla din bhi upar: bulk khareedari aksar 2-3 din chalti hai.

    8b. Store 1: holidays par girawat. 80% girawat wale din holiday/event hain, store ke customers bhi kam (~70% kam,
        transactions ratio ~0.25-0.33). Kyun? Data nahi batata. Pattern har saal dohrata hai => ASLI pattern, rakho.
        Phase 04/05: holiday ka asar har store par alag (store x holiday interaction).

    8c. Company-wide low din. Sab se zyada girawat wale din har saal dohrane wale calendar din hain: Carnaval se pehle
        ke Sundays, Good Friday (MEATS 34-44% kam; bahar ki maloomat: Ecuador mein is din gosht kam khaya jata hai),
        Mother's Day, aur holiday se pehle/baad ke Sundays. LIQUOR 17 Feb aur 31 Mar 2017: ~80% stores mein bohat kam,
        agle do din sab stores mein zero (bahar ki maloomat: election 19 Feb aur runoff 2 Apr 2017 se pehle dry law).
        Sab ASLI, rakho. Phase 05 notes: Mother's Day file mein hai; election (dry law) aur "holiday se pehle ka
        Sunday" nahi. Forecast window (16-31 Aug 2017) mein inmein se koi din nahi.

    8d. Store takreeban band (L0): in 4 din store ki transactions normal ka sirf 0.3-15% thin, aur koi holiday ya
        company-wide din nahi. Store us din takreeban band tha (bijli, marammat...), Phase 01 ka closure rule nahi pakar
        saka kyun ke chand units bik gayi thin. Yeh demand nahi, store ki majboori hai.
    """
    o, master = res["o"], data.master

    # ---- 8a ----
    s2 = o[(o["store_nbr"] == 2) & (o["family"] == "GROCERY I") & o["date"].between("2016-04-24", "2016-05-08")]
    s2 = s2.assign(normal_units=np.expm1(s2["baseline"]).round(0),
                   other_stores_same_family_high=(s2["fam_hi3"] * 100).round(0).astype("Int64").astype(str) + "%",
                   same_store_other_families_high=(s2["store_hi3"] * 100).round(0).astype("Int64").astype(str) + "%")
    out.table(s2[["date", "sales", "normal_units", "rz", "tx_ratio", "onpromotion", "promo_med",
                  "other_stores_same_family_high", "same_store_other_families_high", "cause"]].round(2)
              .set_index("date"), "08a_store2_grocery_case", "8a. Store 2, GROCERY I (Apr-May 2016)")

    nxt = o.sort_values(["store_nbr", "family", "date"]).copy()
    gg = nxt.groupby(["store_nbr", "family"])["rz"]
    nxt["rz_next"], nxt["rz_prev"] = gg.shift(-1), gg.shift(1)
    iso_hi, iso_lo = nxt["cause"].str.startswith("H5"), nxt["cause"].str.startswith("L4")
    out.table(pd.DataFrame({
        "aam rows": [nxt["rz_next"].median(), (nxt["rz_next"] < -2).mean(), (nxt["rz_next"] > 2).mean()],
        "isolated spike ke baad": [nxt.loc[iso_hi, "rz_next"].median(), (nxt.loc[iso_hi, "rz_next"] < -2).mean(),
                                   (nxt.loc[iso_hi, "rz_next"] > 2).mean()],
        "isolated drop ke baad": [nxt.loc[iso_lo, "rz_next"].median(), (nxt.loc[iso_lo, "rz_next"] < -2).mean(),
                                  (nxt.loc[iso_lo, "rz_next"] > 2).mean()]},
        index=["agle din ka median z", "agle din z < -2 (share)", "agle din z > 2 (share)"]).round(3),
        "08a_next_day_behaviour", "8a. Isolated spike/drop ke agle din ka behaviour")

    # ---- 8b ----
    names = master.drop_duplicates("date").set_index("date")["national_desc"].astype(str)
    s1d = o[(o["store_nbr"] == 1) & (o["cause"].str.startswith("L3"))].drop_duplicates("date")
    s1d = s1d.assign(holiday=names.reindex(s1d["date"]).values, weekday=s1d["date"].dt.day_name())
    if len(s1d):
        logger.info("Store 1 ke 'store-wide kami' wale din: %d | in mein holiday/event: %.0f%%", len(s1d),
                    s1d["calendar"].mean() * 100)
    out.table(s1d[["date", "weekday", "holiday", "tx_ratio", "store_lo3"]].head(12).round(2).set_index("date"),
              "08b_store1_holiday_dips", "8b. Store 1: holidays par girawat")

    # ---- 8c ----
    day_tab = o.drop_duplicates("date").set_index("date")[["day_lo3"]].sort_values("day_lo3", ascending=False).head(12)
    day_tab["weekday"] = day_tab.index.day_name()
    day_tab["national_desc (file mein)"] = names.reindex(day_tab.index).values
    day_tab["carnaval sunday"] = day_tab.index.isin(list(data.carnaval_sundays))
    out.table(day_tab.round(3), "08c_company_wide_low_days", "8c. Company-wide sab se low din")

    liq = o[o["family"] == "LIQUOR,WINE,BEER"].groupby("date").agg(units=("sales", "sum"), stores_low=("lo3", "mean"))
    win = pd.concat([liq.loc["2017-02-14":"2017-02-22"], liq.loc["2017-03-28":"2017-04-05"]])
    all_dates = pd.date_range("2017-02-14", "2017-02-22").append(pd.date_range("2017-03-28", "2017-04-05"))
    win = win.reindex(all_dates)
    win["weekday"] = win.index.day_name()
    win["note"] = np.where(win["units"].isna(), "data mein nahi (Phase 01: family unavailable, sab stores zero)", "")
    out.table(win.round(2), "08c_liquor_election_windows", "8c. LIQUOR: election se pehle ke din")

    meats = o[o["family"] == "MEATS"].groupby("date")["sales"].sum()
    gf = []
    for dte in ["2016-03-25", "2017-04-14"]:
        dte = pd.Timestamp(dte)
        if dte in meats.index:
            ref = (meats[dte - pd.Timedelta(days=7)] + meats[dte + pd.Timedelta(days=7)]) / 2
            gf.append({"Good Friday": dte.date(), "MEATS units": round(meats[dte]),
                       "pichle/agle Friday ki average": round(ref), "farq %": round((meats[dte] / ref - 1) * 100, 1)})
    out.table(pd.DataFrame(gf), "08c_good_friday_meats", "8c. Good Friday par MEATS")

    # ---- 8d ----
    l0 = o[o["cause"].str.startswith("L0")]
    l0_days = l0.groupby(["date", "store_nbr"]).agg(rows=("sales", "size"), tx_ratio=("tx_ratio", "first"),
                                                    store_units=("sales", "sum")).reset_index()
    l0_days["weekday"] = l0_days["date"].dt.day_name()
    out.table(l0_days.round(3).set_index("date"), "08d_store_near_closed_days", "8d. Store takreeban band (L0) din")


# ======================================================================================
# 9 / 10. Treatment rules aur candidate treatment
# ======================================================================================
TREATMENT_RULES = [
    ("H1-H4: high, nishaan mila (family, company, store rush, promotion)", "RAKHO",
     "Asli demand. Model ko yahi seekhna hai (Christmas, events, promotions)"),
    ("L1-L3: low, nishaan mila (calendar, company din, store ki holiday aadat)", "RAKHO",
     "Asli aur dohrane wala pattern. Features se predict ho sakta hai"),
    ("Intermittent series ke spikes", "RAKHO", "Chhote numbers, seasonal, units ka < 0.01%"),
    ("H5: isolated spike", "CAP (z = 5 ki hadd tak)",
     "Customers normal, koi aur nishaan nahi => bara order ya ghalat entry. Din 'high' hi rehta hai, bekaboo nahi"),
    ("L4: isolated drop", "Normal (baseline) se badlo, agar Poisson test kahe ke qudrati utaar-charhao nahi",
     "Maal khatam (stock-out) ya record nahi hua. Bikri kam hui, DEMAND kam nahi hui"),
    ("L0: store takreeban band", "Normal se badlo (us store-din ki dabi hui rows)", "Store ki majboori, demand nahi. Closure jaisa"),
]


def treat(o: pd.DataFrame) -> pd.DataFrame:
    """
    Candidate treatment: `sales_clean` (saare candidate rules) aur `outlier_action`. Sirf scope rows par chalta hai.

    Section 9 (rules, data ki nature ke mutabiq; final faisla Section 11 ka test karega, TREATMENT_RULES dekhein):
    - High ko CAP, low ko REPLACE kyun? High (bulk order): customers ne waqai khareeda, kuch to asli tha => value
      "bohat zyada" se "zyada" (z = 5) tak laayi jati hai, mitai nahi. Low (stock-out): customers normal aaye, doosre
      stores mein family normal biki, phir bhi yahan bikri bohat kam => shelf khaali thi (wazahat, saboot nahi: inventory
      data nahi). Hum DEMAND forecast kar rahe hain, bikri nahi => us din ka normal behtareen andaza.
    - L4 ke liye Poisson hifazat: chhoti series (roz ~3 units) mein kabhi kabhi zero qudrati taur par aata hai. Is liye L4
      drop tabhi badla jayega jab Poisson test kahe ke itni kam value qudrati taur par 1,000 mein 1 baar se bhi kam aati hai.
    - Bilkul nahi hoga: koi row delete nahi, validation/test nahi chhuye jayenge, asli `sales` waisa hi rahega. Saaf value
      naye column `sales_clean` mein jayegi taa ke har faisla wapas dekha ja sake.
    """
    o = o.copy()
    o["sales_clean"] = o["sales"].astype(float)
    o["outlier_action"] = "none"
    base_units = np.expm1(o["baseline"] + o["res_med"])           # us din ka normal
    # H5: cap at z = 5
    cap = o["cause"].str.startswith("H5")
    cap_units = np.expm1(o["baseline"] + o["res_med"] + Z_THR * 1.4826 * o["res_mad"])
    o.loc[cap, "sales_clean"] = cap_units[cap]
    o.loc[cap, "outlier_action"] = "cap"
    # L4: Poisson test, phir normal se replace
    p_low = stats.poisson.cdf(np.floor(o["sales"]), base_units.clip(lower=1e-9))
    l4 = o["cause"].str.startswith("L4")
    o.loc[l4 & (p_low < POISSON_P), "sales_clean"] = base_units[l4 & (p_low < POISSON_P)]
    o.loc[l4 & (p_low < POISSON_P), "outlier_action"] = "replace_stockout"
    o.loc[l4 & (p_low >= POISSON_P), "outlier_action"] = "keep_natural_low"
    # L0: store takreeban band
    l0 = o["cause"].str.startswith("L0")
    o.loc[l0, "sales_clean"] = base_units[l0]
    o.loc[l0, "outlier_action"] = "replace_disruption"
    o.loc[(o["cause"] != "") & (o["outlier_action"] == "none"), "outlier_action"] = "keep_real_event"
    return o


def apply_actions(o: pd.DataFrame, actions: list[str]) -> np.ndarray:
    """Sirf chuni hui actions ki saaf value lo, baqi rows asli sales."""
    return np.where(o["outlier_action"].isin(actions), o["sales_clean"], o["sales"])


def section_10_candidates(data: Data, out: Outputs, res: dict[str, Any]) -> None:
    """Pehle saare rules candidate ke taur par lagte hain; final faisla Section 11 ke test ke baad hoga."""
    rules = pd.DataFrame(TREATMENT_RULES, columns=["Wajah", "Candidate rule", "Kyun (data ki nature)"]).set_index("Wajah")
    out.table(rules, "09_treatment_rules", "9. Candidate treatment rules")

    o = treat(res["o"])
    res["o"] = o
    act = o["outlier_action"].value_counts().to_frame("rows")
    act["units pehle"] = o.groupby("outlier_action")["sales"].sum().round(0)
    act["units candidate"] = o.groupby("outlier_action")["sales_clean"].sum().round(0)
    logger.info("L4 drops jo Poisson test ne 'qudrati' maan kar rakhe: %s", f"{(o['outlier_action'] == 'keep_natural_low').sum():,}")
    out.table(act, "10_candidate_actions", "10. Candidate actions")

    audit = o.loc[o["outlier_action"].isin(TREAT_ACTIONS),
                  ["date", "store_nbr", "family", "sales", "sales_clean", "rz", "tx_ratio", "onpromotion", "cause",
                   "outlier_action"]].rename(columns={"sales_clean": "candidate_value"})
    out.table(audit.sort_values("sales", ascending=False).head(10).round(2).set_index("date"),
              "10_audit_top10", "10. Audit log: sab se bari 10 candidate tabdeeliyan")


# ======================================================================================
# 11. Forecast test: har treatment alag, 4 windows, noise ke muqable
# ======================================================================================
def wape(y: np.ndarray, p: np.ndarray) -> float:
    return np.abs(y - p).sum() / y.sum() * 100


def rmsle(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.log1p(np.clip(p, 0, None)) - np.log1p(y)) ** 2)))


BASE_COLS = ["store_nbr", "family", "date", "split", "sales", "onpromotion", "store_type", "store_cluster", "city",
             "is_holiday", "is_national_holiday", "is_local_holiday", "is_regional_holiday", "is_holiday_season_day",
             "is_workday_override", "is_national_event", "is_earthquake_period", "is_trainable", "is_dead_series",
             "is_pre_opening", "is_store_closed", "is_closure_calendar_day"]
HIST = ["lag_16", "lag_21", "lag_28", "lag_35", "lag_42", "lag_364", "roll7_lag16", "roll28_lag16", "same_weekday_mean4"]
BASE_FEATURES = ["store_cat", "family", "store_type", "store_cluster", "city", "onpromotion", "dow", "dom", "month",
                 "is_holiday", "is_national_holiday", "is_local_holiday", "is_regional_holiday", "is_holiday_season_day",
                 "is_workday_override", "is_national_event", "is_earthquake_period"]
FEATURES = BASE_FEATURES + HIST

# (naam, kaun si actions, random seed)
VARIANTS = {"Raw": ([], 0), "Raw (seed 1, noise check)": ([], 1), "Cap (H5)": (["cap"], 0),
            "Replace (L4 + L0)": (["replace_stockout", "replace_disruption"], 0)}
WINDOWS = {"Validation (Aug 2017)": ("2017-07-30", "2017-07-31"),
           "Zalzale ke baad (May 2016)": ("2016-05-17", "2016-05-18"),
           "Christmas se pehle (Dec 2016)": ("2016-12-02", "2016-12-03"),
           "Aam window (Jun 2017)": ("2017-06-14", "2017-06-15")}


def build_model_frame(master: pd.DataFrame) -> pd.DataFrame:
    M = master.loc[master["date"] >= "2014-06-01", BASE_COLS].sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    for c in ["family", "store_type", "city"]:
        M[c] = M[c].astype("category")
    M["store_cat"] = M["store_nbr"].astype("category")
    M["dow"], M["dom"], M["month"] = M["date"].dt.dayofweek, M["date"].dt.day, M["date"].dt.month
    return M


def add_history(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby(["store_nbr", "family"], observed=True)[col]
    out = pd.DataFrame(index=df.index)
    for lag in (16, 21, 28, 35, 42, 364):
        out[f"lag_{lag}"] = g.shift(lag)
    s16 = g.shift(16)
    keys = [df["store_nbr"], df["family"]]
    out["roll7_lag16"] = s16.groupby(keys, observed=True).transform(lambda x: x.rolling(7, min_periods=1).mean())
    out["roll28_lag16"] = s16.groupby(keys, observed=True).transform(lambda x: x.rolling(28, min_periods=1).mean())
    out["same_weekday_mean4"] = out[["lag_21", "lag_28", "lag_35", "lag_42"]].mean(axis=1)
    return out


def run_window(data: Data, M: pd.DataFrame, train_end: str, eval_start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_end_ts, eval_start_ts = pd.Timestamp(train_end), pd.Timestamp(eval_start)
    eval_end = eval_start_ts + pd.Timedelta(days=15)
    master = data.master
    # 1) Detection + treatment SIRF train_end tak ke data par (leakage se hifazat)
    sc = (master["is_trainable"] == 1) & master["date"].between(CLEAN_START, train_end_ts)
    ow = detect(master.loc[sc, COLS], data.store_tx, data.carnaval_sundays)
    ow = treat(ow.assign(cause=classify(ow)))
    ow["family"] = pd.Categorical(ow["family"], categories=M["family"].cat.categories)
    df = M.merge(ow[["store_nbr", "family", "date", "sales_clean", "outlier_action"]],
                 on=["store_nbr", "family", "date"], how="left")
    known = (df["date"] <= train_end_ts).values
    ev_mask = df["date"].between(eval_start_ts, eval_end)
    tr_mask = (df["is_trainable"] == 1) & df["date"].between(CLEAN_START, train_end_ts)
    ev = df.loc[ev_mask, ["store_nbr", "family", "date", "sales", "is_dead_series", "is_pre_opening",
                          "is_store_closed", "is_closure_calendar_day"]].copy()
    zero_rule = ((ev["is_dead_series"] == 1) | (ev["is_pre_opening"] == 1) | (ev["is_closure_calendar_day"] == 1)).values
    for name, (actions, seed) in VARIANTS.items():
        col = np.where(df["outlier_action"].isin(actions), df["sales_clean"], df["sales"])
        df["_y"] = np.where(known, col, np.nan)                 # window ki sales kabhi features mein nahi
        X = pd.concat([df[BASE_FEATURES], add_history(df, "_y")], axis=1)
        mdl = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.07, num_leaves=127, min_child_samples=50,
                                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
        mdl.fit(X.loc[tr_mask, FEATURES], np.log1p(df.loc[tr_mask, "_y"]))
        p = np.expm1(mdl.predict(X.loc[ev_mask, FEATURES])).clip(0)
        ev["pred_" + name] = np.where(zero_rule, 0, p)
    open_rows = ((ev["is_pre_opening"] == 0) & (ev["is_store_closed"] == 0)).values
    rows = []
    for name in VARIANTS:                                       # imtihaan hamesha ASLI, bina chhuyi sales par
        y, p = ev["sales"].values[open_rows], ev["pred_" + name].values[open_rows]
        rows.append({"variant": name, "WAPE %": wape(y, p), "RMSLE": rmsle(y, p), "forecast / actual": p.sum() / y.sum()})
    n_t = ow["outlier_action"].isin(TREAT_ACTIONS).sum()
    logger.info("Train until %s | evaluate %s -> %s | treated rows in history: %s", train_end_ts.date(),
                eval_start_ts.date(), eval_end.date(), f"{n_t:,}")
    return pd.DataFrame(rows).set_index("variant"), ev


def section_11_forecast_test(data: Data, out: Outputs, res: dict[str, Any]) -> None:
    """
    Rules kitne bhi achay lagein, saboot test se aata hai. Wahi LightGBM model (Phase 01b jaisa, wahi features) har bar
    train hota hai; sirf target aur history features ka source badalta hai:

      Raw                        Asli `sales`
      Cap (H5)                   Sirf isolated spikes cap
      Replace (L4 + L0)          Sirf stock-out aur store-band din normal se badle
      Raw (seed 1)               Asli `sales`, sirf random seed alag (noise naapne ke liye)

    Imtihaan hamesha ASLI, bina chhuyi sales par (jo waqai bika).
    Leakage se hifazat: har window ke liye detection aur treatment SIRF us window se pehle ke data par dobara chalta hai.

    4 windows:
      1. Validation (31 Jul - 15 Aug 2017): asli mock exam.
      2. 18 May - 2 Jun 2016: zalzale ke bulk orders ke foran baad (store 2 ki 1.25 lakh wali value `lag_16` mein aati hai).
      3. 3 - 18 Dec 2016: Christmas se pehle ka mushkil waqt (Phase 01b ki backtest 3).
      4. 15 - 30 Jun 2017: ek aam window, jise rules banate waqt khaas taur par nahi dekha gaya.

    Imaandari wala note (design-level leakage): rules banate waqt poori training history ke extremes dekhe gaye thay (May
    aur Dec 2016 ke din bhi, misaal store 2). Rules generic hain (koi tareekh/store hard-code nahi) aur har window mein
    treatment sirf pichle data par chala, is liye asar chhota hai, phir bhi test 100% "andha" nahi. June 2017 isi liye shaamil.

    Noise ka andaza (seed test): LightGBM har baar rows/columns ka random hissa leta hai. Raw (seed 0) aur Raw (seed 1) ka
    farq NOISE hai, yani bina kuch badle bhi aane wala farq. Koi treatment tabhi "behtar" mana jayega jab uska faida
    is noise se bara ho.

    Test ka nateeja (original run ke numbers):
    1. Seed noise ~0.10 WAPE points (kuch windows mein 0.12-0.23).
    2. Replace (L4 + L0): average faida sirf -0.03 points, noise se chhota => FAIL.
    3. Cap (H5): chaaron windows mein forecast thora kharab (+0.04, +0.34, +0.10, +0.01) => FAIL. Wajah: "isolated" spikes
       ka ~54% un series mein hai jahan aise spikes 5 ya zyada baar aaye (kuch mein 20-38 baar): ek baar ke waqiat nahi,
       series ka baqaida rawaiya.
    4. `lag_16` mein 1.25 lakh ka dar sach nahi nikla: tree model formula se multiply nahi karta, bohat bari value sirf
       "bohat zyada" wali shaakh mein chali jati hai.
    Faisla: koi value nahi badli jayegi. Zyada tar outliers asli demand hain; shak wale badalne se forecast noise se
    zyada behtar nahi hua. Phase phir bhi zaroori tha: ab har extreme ki wajah (`outlier_cause`) hai.
    """
    M = build_model_frame(data.master)
    results: dict[str, pd.DataFrame] = {}
    evs: dict[str, pd.DataFrame] = {}
    for wname, (te, es) in WINDOWS.items():
        t0 = time.time()
        results[wname], evs[wname] = run_window(data, M, te, es)
        logger.info("   (%.0fs)", time.time() - t0)
    test_tab = pd.concat(results, axis=1)
    out.table(test_tab.round(4), "11_backtest_results", "11. Backtest results (4 windows x 4 variants)")
    res["test_tab"] = test_tab

    x = evs["Zalzale ke baad (May 2016)"]
    x = x[(x["store_nbr"] == 2) & (x["family"] == "GROCERY I")].set_index("date")
    out.table(x[["sales"] + ["pred_" + v for v in VARIANTS]].round(0).head(8), "11_store2_forecasts_may2016",
              "11. Store 2, GROCERY I: forecasts (zalzale ke baad window)")

    # Cap kyun fail hua? Kya "isolated" spikes waqai ek baar ke waqiat hain?
    h5 = res["o"][res["o"]["outlier_action"] == "cap"]
    per_series = h5.groupby(["store_nbr", "family"]).size().sort_values(ascending=False)
    logger.info("Isolated spikes: %s rows, %s series mein", f"{len(h5):,}", f"{len(per_series):,}")
    if len(h5):
        logger.info("Un series mein jahan 5 ya zyada spikes aaye: %s rows (%.0f%%)",
                    f"{per_series[per_series >= 5].sum():,}", per_series[per_series >= 5].sum() / len(h5) * 100)
    out.table(per_series.head(6).to_frame("spikes (clean period mein)"), "11_repeat_spike_series",
              "11. Baar baar spike wali series")

    # Faisle ka usool (review ke baad sakht kiya gaya, naye numbers dekhne se PEHLE): koi treatment tabhi rahega jab Raw
    # ke muqable (1) saari windows ki average WAPE ka faida, seed-noise ki average se BARA ho, aur
    # (2) kisi bhi window mein WAPE 0.10 point se zyada kharab na kare.
    wape_tab = test_tab.xs("WAPE %", axis=1, level=1)
    delta = wape_tab.sub(wape_tab.loc["Raw"], axis=1)
    noise = delta.loc["Raw (seed 1, noise check)", list(WINDOWS)].abs().mean()
    delta["average"] = delta[list(WINDOWS)].mean(axis=1)
    delta["worst window"] = delta[list(WINDOWS)].max(axis=1)
    delta["pass?"] = (delta["average"] < -noise) & (delta["worst window"] <= 0.10)
    delta.loc[["Raw", "Raw (seed 1, noise check)"], "pass?"] = False
    logger.info("Seed noise (Raw seed 0 vs seed 1, average |farq|): %.3f WAPE points", noise)
    out.table(delta.round(3), "11_wape_delta_vs_raw", "11. WAPE ka farq Raw ke muqable (manfi = behtar)")

    final_actions: list[str] = []
    for name, actions in [("Cap (H5)", ["cap"]), ("Replace (L4 + L0)", ["replace_stockout", "replace_disruption"])]:
        if delta.loc[name, "pass?"]:
            final_actions += actions
    logger.info("FINAL_ACTIONS (jo test pass kar gaye): %s", final_actions if final_actions else "koi nahi")
    res["wape_tab"], res["noise"], res["final_actions"] = wape_tab, float(noise), final_actions


# ======================================================================================
# 12. Final treatment aur pehle vs baad
# ======================================================================================
def section_12_final_treatment(data: Data, out: Outputs, res: dict[str, Any]) -> None:
    """
    Sirf woh treatment lagaya jata hai jo Section 11 ka test pass kare (`FINAL_ACTIONS`). Fail hone wale candidate rules
    ki values ASLI rehti hain, lekin un ki wajah (`outlier_cause`) aur faisla (`..._rejected_by_test`) audit ke liye
    mehfooz rehte hain.

    Insight 12 (original run): test ke baad koi treatment pass nahi hua, is liye ek bhi value nahi badli. `sales_clean`
    bilkul `sales` ke barabar hai. Asli events, bulk spikes aur stock-out jaise din sab waise hi hain, aur un ki wajah
    `outlier_cause` column mein darj hai.
    """
    o, final_actions = res["o"], res["final_actions"]
    o["sales_final"] = apply_actions(o, final_actions)
    o["final_action"] = np.where(o["outlier_action"].isin(final_actions), o["outlier_action"],
                                 np.where(o["outlier_action"].isin(TREAT_ACTIONS),
                                          o["outlier_action"].astype(str) + "_rejected_by_test", o["outlier_action"]))
    changed = o[o["outlier_action"].isin(final_actions)]
    res["changed"] = changed
    logger.info("Badli gayi rows: %s = scope ka %.3f%%", f"{len(changed):,}", len(changed) / len(o) * 100)
    logger.info("Kul units: pehle %s | baad %s | farq %+.3f%%", f"{o['sales'].sum():,.0f}", f"{o['sales_final'].sum():,.0f}",
                (o["sales_final"].sum() / o["sales"].sum() - 1) * 100)
    out.table(o["final_action"].value_counts().to_frame("rows"), "12_final_actions", "12. Final actions")

    def describe(s: pd.Series) -> pd.Series:
        return pd.Series({"mean": s.mean(), "median": s.median(), "p99": s.quantile(.99), "max": s.max(),
                          "zero share %": (s == 0).mean() * 100, "skew (log1p)": np.log1p(s).skew()})

    out.table(pd.DataFrame({"pehle (sales)": describe(o["sales"]), "baad (sales_clean)": describe(o["sales_final"])}).round(3),
              "12_before_after_summary", "12. Pehle vs baad")

    if len(changed):
        fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.4), sharex=True)
        s2 = o[(o["store_nbr"] == 2) & (o["family"] == "GROCERY I")].set_index("date").loc["2016-03-01":"2016-07-01"]
        axes[0].plot(s2.index, s2["sales"], color=LIGHT_GRAY, label="pehle (sales)")
        axes[0].plot(s2.index, s2["sales_final"], color=BLUE, linewidth=1.4, label="baad (sales_clean)")
        axes[0].set_title("Store 2, GROCERY I: final treatment ke baad")
        axes[0].yaxis.set_major_formatter(thousands)
        tot = o.groupby("date")[["sales", "sales_final"]].sum().loc["2016-03-01":"2016-07-01"]
        axes[1].plot(tot.index, tot["sales"], color=LIGHT_GRAY, label="pehle (sales)")
        axes[1].plot(tot.index, tot["sales_final"], color=BLUE, linewidth=1.2, label="baad (sales_clean)")
        axes[1].set_title("Company ki kul sales")
        axes[1].yaxis.set_major_formatter(thousands)
        for ax in axes:
            ax.legend(loc="upper left")
        out.fig(fig, "07_before_after")
    else:
        # Koi value nahi badli: before/after chart ka koi matlab nahi. Purane run ki chart files bhi hata do.
        for f in ("07_before_after.png", "08_change_by_family.png"):
            (out.fig_dir / f).unlink(missing_ok=True)
        logger.info("Koi value nahi badli, is liye before/after chart nahi banaya gaya.")


# ======================================================================================
# 13. Files save karna
# ======================================================================================
def section_13_save(data: Data, out: Outputs, res: dict[str, Any], processed_dir: Path) -> None:
    """
    Nayi file mein 5 naye columns (asli `sales` waisa hi):
      sales_clean         Phase 05 ka target aur history features isi se. Is run mein `sales` ke barabar (agar koi
                          treatment pass na hua). Column rakha gaya taa ke Phase 05 ka pipeline ek hi column use kare.
      outlier_robust_z    Section 4 ka z-score (sirf scope rows)
      outlier_cause       Section 7 ki wajah (H1-H5, L0-L4, ya `none`)
      outlier_action      Final faisla: keep_real_event, keep_natural_low, pass hui action, ya `..._rejected_by_test`
      is_outlier_treated  1 = value badli gayi. FEATURE NAHI (sales se bana hai), sirf audit ke liye
    """
    master, o = data.master, res["o"]
    final_actions, wape_tab, noise = res["final_actions"], res["wape_tab"], res["noise"]

    keep = o[["store_nbr", "family", "date", "sales_final", "rz", "cause", "final_action"]].rename(
        columns={"sales_final": "sales_clean", "rz": "outlier_robust_z", "cause": "outlier_cause",
                 "final_action": "outlier_action"})
    treated = master.merge(keep, on=["store_nbr", "family", "date"], how="left")
    treated["sales_clean"] = treated["sales_clean"].fillna(treated["sales"])       # scope ke bahar: asli sales (test mein NaN)
    treated["outlier_action"] = treated["outlier_action"].fillna("none")
    treated["outlier_cause"] = treated["outlier_cause"].fillna("").replace("", "none")
    treated["is_outlier_treated"] = treated["outlier_action"].isin(final_actions).astype("int8")
    for c in ["outlier_action", "outlier_cause"]:
        treated[c] = treated[c].astype("category")

    # --- Final checks ---
    assert len(treated) == len(master)
    v = treated["split"] == "valid"
    assert np.array_equal(treated.loc[v, "sales_clean"].values, treated.loc[v, "sales"].values), "validation badla!"
    assert treated.loc[treated["split"] == "test", "sales_clean"].isna().all()
    u = treated["is_outlier_treated"] == 0
    assert np.array_equal(treated.loc[u, "sales_clean"].fillna(-1).values, treated.loc[u, "sales"].fillna(-1).values)
    assert (treated["sales_clean"].dropna() >= 0).all()
    assert treated.loc[treated["is_outlier_treated"] == 1, "split"].eq("train").all()
    assert treated.loc[treated["is_outlier_treated"] == 1, "date"].ge(CLEAN_START).all()
    logger.info("Saare checks PASS")

    treated_path = processed_dir / "retail_demand_master_treated.parquet"
    treated.to_parquet(treated_path, index=False)
    log = o.loc[o["outlier_action"].isin(TREAT_ACTIONS),
                ["date", "store_nbr", "family", "sales", "sales_clean", "sales_final", "rz", "tx_ratio", "onpromotion",
                 "cause", "outlier_action", "final_action"]].rename(columns={"sales_clean": "candidate_value",
                                                                              "sales_final": "final_value"})
    log_path = out.reports_dir / "03_outlier_treatment_log.csv"
    log.to_csv(log_path, index=False)
    summary = {
        "scope_rows": int(len(o)), "robust_z_extremes": int(o["rz_flag"].sum()), "iqr_outliers": int(o["iqr_flag"].sum()),
        "causes": {k: int(x) for k, x in o.loc[o["cause"] != "", "cause"].value_counts().items()},
        "final_actions": final_actions,
        "actions": {k: int(x) for k, x in o["final_action"].value_counts().items()},
        "rows_changed": int(treated["is_outlier_treated"].sum()),
        "units_change_pct": float(o["sales_final"].sum() / o["sales"].sum() - 1) * 100,
        "test_wape": wape_tab.round(4).to_dict(),
        "seed_noise_wape_points": round(float(noise), 4),
        "thresholds": {"z": Z_THR, "share": SHARE, "day_share": DAY_SHARE, "tx_high": TX_HIGH, "tx_low": TX_LOW,
                       "promo_x": PROMO_X, "poisson_p": POISSON_P},
    }
    summary_path = out.reports_dir / "03_outliers_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    res["summary"] = summary
    logger.info("Saved: %s %s", treated_path, treated.shape)
    logger.info("Saved: %s (%d rows, har candidate final faisle ke sath)", log_path, len(log))
    logger.info("Saved: %s", summary_path)


# ======================================================================================
# 14. Khulasa aur Phase 04 ke liye notes
# ======================================================================================
CONCLUSIONS_MD = """\
## Detection (2 visual + 2 statistical)
1. **Boxplot:** har family ki range dikhata hai. Lekin upar wale "outliers" zyada tar do chhoti families (HOME APPLIANCES, BABY CARE), bare stores aur peak dinon ke hain, kyun ke 54 stores ek dabbe mein hain.
2. **Time-series + normal band:** dikhata hai ke har "bahar" wala din ek jaisa nahi (season, holiday, bulk order).
3. **IQR (har series):** zyada tar flags chhoti, aksar zero wali series ke season par hain, aur din ka context nahi dekhta.
4. **Contextual robust z:** har value ko usi series ke usi weekday ke normal se compare karta hai. **Primary detector.**

## Wajah ki tafteesh
Original run mein ~76% extremes (6,296 of 8,267) ke data mein nishaan mile: same family bohat se stores mein, company ka khaas din, store ka rush, promotion, aur calendar (Carnaval Sunday, Good Friday, Mother's Day, election ke din, store 1 ki holidays). **Yeh asli demand hain.**

## Treatment (data ki nature + forecast test, 4 windows, seed noise ke muqable)
| Wajah | Candidate rule | Test (original run) | Final |
|---|---|---|---|
| Nishaan wale extremes (H1-H4, L1-L3) | Rakho | - | **Rakho** |
| Intermittent spikes | Rakho | - | **Rakho** |
| Isolated spike (H5) | Cap | 4/4 windows mein thora kharab | **Rakho** |
| Isolated drop (L4) + store takreeban band (L0) | Normal se bharo | Faida noise se chhota | **Rakho** |

**Final nateeja (original run): data mein koi value nahi badli.** Yeh "kuch na karna" nahi, balkay test se sabit faisla hai ke is data aur is model setup ke liye outlier treatment ki zaroorat nahi. Validation/test nahi chhuye gaye, koi row delete nahi hui. Har candidate aur uska faisla `reports/03_outlier_treatment_log.csv` mein hai.

## Is phase ke sabaq
1. Jo rules kaghaz par theek lagte thay (bulk spikes kaatna, stock-out bharna), test mein fail hue. Treatment ka faisla **andaze se nahi, test se** hota hai.
2. Model comparison mein **noise naapna zaroori hai.** Seed test ke baghair 0.03-0.05 points ka farq "faida" lag raha tha, jab ke noise ~0.10 tha.

## Limitations (imaandari)
- Bulk order aur ghalat entry mein farq data se nahi ho sakta (receipt-level data nahi).
- Stock-out ka direct saboot (inventory data) nahi. L4 ko "stock-out" kehna sab se munasib wazahat hai, saboot nahi.
- Thresholds (20%, 5%, 1.3x, 0.5x, 2x, z = 5, Poisson 0.001) chune gaye hain. Sensitivity test sirf share thresholds ka hua.
- Rules poori training history dekh kar banaye gaye, is liye May/Dec 2016 ke test 100% andhe nahi (June 2017 window isi liye shaamil).
- Seed noise sirf 2 seeds se naapa gaya. Zyada seeds se andaza aur pakka hota.

## Phase 04 (Post-EDA) ke liye notes
- Patterns (weekday, payday, month) ko outlier labels ke sath dobara dekhna, misaal: "asli event" wale din hata kar ya alag kar ke.
- **Store x holiday** ka asar alag hai (store 1 mein holidays par customers ~70% kam).
- **Baar baar spike wali series** (H5, 5+ spikes) ek alag demand class ho sakti hain.
- Calendar din jo holidays file mein nahi: **election dry law**, **holiday se pehle ka Sunday**. Forecast window (16-31 Aug 2017) mein yeh nahi aate, lekin features ke taur par check karne layak hain.
"""


def section_14_conclusions(data: Data, out: Outputs, res: dict[str, Any]) -> None:
    """Notebook ke aakhri markdown cell ko `reports/03_outliers_conclusions.md` mein likho (is run ke numbers ke sath)."""
    s = res["summary"]
    header = [
        "# Phase 03 khulasa: Outliers Detection and Treatment", "",
        "## Is run ke numbers",
        f"- Scope rows: {s['scope_rows']:,} | robust-z extremes: {s['robust_z_extremes']:,} | IQR outliers: {s['iqr_outliers']:,}",
        f"- Final actions (test pass): {s['final_actions'] if s['final_actions'] else 'koi nahi'}",
        f"- Badli gayi rows: {s['rows_changed']:,} | units change: {s['units_change_pct']:+.3f}%",
        f"- Seed noise: {s['seed_noise_wape_points']} WAPE points", "",
        "*(Neeche ka narrative original notebook run se hai; naye data par numbers thore mukhtalif ho sakte hain.)*", ""]
    path = out.reports_dir / "03_outliers_conclusions.md"
    path.write_text("\n".join(header) + CONCLUSIONS_MD, encoding="utf-8")
    logger.info("Saved: %s", path)
    logger.info("Charts (%d): %s", len(list(out.fig_dir.glob("*.png"))), sorted(p.name for p in out.fig_dir.glob("*.png")))


# ======================================================================================
# Entry point
# ======================================================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 03: Outliers detection and treatment (Retail Demand Forecasting).")
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"), help="Phase 01 ke processed files ka folder")
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
        ("1. Boxplot (visual 1)", section_1_boxplot),
        ("2. Time-series + normal band (visual 2)", section_2_timeseries_band),
        ("3. IQR (statistical 1)", section_3_iqr),
        ("4. Contextual robust z (statistical 2)", section_4_robust_z),
        ("5. IQR vs robust z", section_5_compare),
        ("6. Intermittent series", section_6_intermittent),
        ("7. Wajah ki tafteesh", section_7_causes),
        ("8. Khaas cases ka saboot", section_8_special_cases),
        ("9/10. Treatment rules aur candidates", section_10_candidates),
        ("11. Forecast test (backtest)", section_11_forecast_test),
        ("12. Final treatment", section_12_final_treatment),
        ("13. Files save karna", lambda d, o, r: section_13_save(d, o, r, args.processed_dir)),
        ("14. Khulasa", section_14_conclusions),
    ]
    for title, fn in pipeline:
        logger.info("=" * 20 + " Section %s " + "=" * 20, title)
        fn(data, out, res)

    logger.info("Phase 03 Outliers Detection and Treatment complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
