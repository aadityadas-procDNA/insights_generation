"""
01_data_prep.py — Load gold layer, aggregate to market level, build feature set.

Reads:  GOLD_LABELLED  (UC table on Databricks; parquet file locally)
Writes: MARKET_SERIES  (~257 rows x ~30 cols; one row per week, market-level; UC table)

HYBRID EXECUTION
----------------
The expensive step is the HCP x week -> market aggregation over a potentially huge
gold table. That step runs in SPARK when a SparkSession is available: filtering and
the groupby happen in Spark, and ONLY the ~257-row aggregate is pulled into the driver
via .toPandas(). All downstream feature engineering (Fourier terms, splits, ~257 rows)
stays in pandas because it is trivially small and Spark would add overhead for no gain.

If Spark is NOT available, it falls back to a pure-pandas path. WARNING: the pandas
fallback must read the ENTIRE gold table into driver/local memory before filtering and
aggregating. On a large multi-million-row gold table this can exhaust memory and crash.
The pandas path is intended for local development on a small/sampled extract only.

Run locally:  uv run python models/01_data_prep.py
On Databricks: %run ./01_data_prep  (or as a notebook cell)
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from config import (
    ON_DATABRICKS, GOLD_LABELLED, MARKET_SERIES,
    read_parquet, write_parquet, PARAMS,
)


FOCUS = PARAMS["FOCUS_PRODUCT"]

# Channels that are market-level constants per week (same for all HCPs) — take MEAN
BROADCAST_COLS = [
    "tv_grps",
    "standard_display_impressions",
    "programmatic_display_impressions",
    "programmatic_video_impressions",
    "social_impressions",
    "audio_impressions",
    "competitor_spend",
]

# Field channels — SUM across HCPs (each HCP's activity is additive)
FIELD_COLS = [
    "f2f", "f2f_short_call", "f2f_accompanied", "phone_call", "total_calls",
    "samples", "speaker",
    "email_delivered", "email_opens", "email_clicked",
    "tp_email_delivered", "tp_email_opens", "tp_email_clicked",
    "doximity_opens", "epocrates_opens", "sermo_impressions",
    "ehr_impressions",
]

TARGET_COLS = ["sales", "trx", "nrx"]

# Known organic change-point dates (ground truth for validation — do NOT leak to model)
ORGANIC_CPS = {
    "otezla_maturity":  pd.Timestamp("2017-01-02"),
    "eucrisa_launch":   pd.Timestamp("2017-01-02"),
    "tremfya_launch":   pd.Timestamp("2017-07-03"),
}


# ── Spark availability guard ──────────────────────────────────────────────────
def get_spark():
    """Return an active SparkSession if one exists, else None.

    We do NOT create a new session here — on Databricks one already exists, and we
    don't want to spin one up locally just to aggregate a small extract.
    """
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession()
        return spark
    except Exception:
        return None


# ── Aggregation: Spark path (large table) ─────────────────────────────────────
def aggregate_to_market_spark(spark, source_ref: str) -> pd.DataFrame:
    """
    Filter + collapse HCP x product x week -> week x product entirely in Spark, then
    pull only the small market-level result into pandas.

    Field channels are summed; broadcast channels are averaged (summing would inflate
    them by ~n_HCPs and destroy the MMM coefficients); lifecycle_stage takes first().
    """
    from pyspark.sql import functions as F

    print("  [spark] reading + filtering + aggregating in Spark (driver stays light)")

    sdf = spark.table(source_ref) if ON_DATABRICKS else spark.read.parquet(source_ref)

    n_total = sdf.count()
    n_anom  = sdf.filter(F.col("is_anomaly") == 1).count()
    print(f"  {n_total:,} rows  |  {n_anom:,} labelled anomaly rows")

    # Keep clean rows only, focus product only — done in Spark before any collect
    sdf = sdf.filter((F.col("is_anomaly") == 0) & (F.col("product") == FOCUS))
    n_prod = sdf.count()
    n_hcp  = sdf.select("hcp_id").distinct().count()
    print(f"  Using clean {FOCUS} rows: {n_prod:,} rows across {n_hcp:,} HCPs")

    agg_exprs = (
        [F.sum(c).alias(c) for c in FIELD_COLS + TARGET_COLS]
        + [F.mean(c).alias(c) for c in BROADCAST_COLS]
        + [F.first("lifecycle_stage").alias("lifecycle_stage")]
    )
    mkt_sdf = sdf.groupBy("product", "week").agg(*agg_exprs)

    # Only the ~257-row aggregate crosses into driver memory
    mkt = mkt_sdf.toPandas()
    print(f"  [spark] collected {len(mkt):,} market-level rows to pandas")
    return mkt


# ── Aggregation: pandas fallback (small extract only) ─────────────────────────
def aggregate_to_market_pandas(source_ref: str) -> pd.DataFrame:
    """
    Pure-pandas aggregation. WARNING: reads the ENTIRE source into memory first.
    Safe only for a small/sampled local extract — NOT for a full gold table.
    """
    print("  [pandas] WARNING: loading the full table into memory before aggregating.")
    print("           If the gold table is large this may exhaust memory and crash.")
    print("           Use the Spark path (run on Databricks / an active SparkSession)")
    print("           or point GOLD_LABELLED at a sampled extract for local dev.")

    df = read_parquet(source_ref)
    print(f"  {len(df):,} rows  |  {df['is_anomaly'].sum():,} labelled anomaly rows")

    df = df[(df["is_anomaly"] == 0) & (df["product"] == FOCUS)].copy()
    print(f"  Using clean {FOCUS} rows: {len(df):,} rows across {df['hcp_id'].nunique():,} HCPs")

    agg_dict = {c: "sum" for c in FIELD_COLS + TARGET_COLS}
    agg_dict.update({c: "mean" for c in BROADCAST_COLS})
    agg_dict["lifecycle_stage"] = "first"

    mkt = df.groupby(["product", "week"]).agg(agg_dict).reset_index()
    return mkt


def aggregate_to_market(source_ref: str) -> pd.DataFrame:
    """Dispatch to Spark if available, else pandas with a memory warning."""
    spark = get_spark()
    if spark is not None:
        return aggregate_to_market_spark(spark, source_ref)
    return aggregate_to_market_pandas(source_ref)


# ── Feature engineering (pandas; operates on the ~257-row market series) ───────
def add_features(mkt: pd.DataFrame) -> pd.DataFrame:
    """Add log_sales, event flags, lifecycle numeric, and Fourier seasonality."""
    mkt = mkt.sort_values("week").reset_index(drop=True)

    mkt["log_sales"] = np.log(mkt["sales"].clip(lower=1e-6))

    stage_map = {"pre_launch": 0, "launch": 1, "growth": 2, "maturity": 3, "decline": 4}
    mkt["lc_num"] = mkt["lifecycle_stage"].map(stage_map).fillna(2).astype(int)

    mkt["week_idx"] = np.arange(len(mkt))

    for name, ts in ORGANIC_CPS.items():
        mkt[f"flag_{name}"] = (mkt["week"] >= ts).astype(int)

    K = PARAMS["FOURIER_K"]
    t = mkt["week_idx"].values
    for k in range(1, K + 1):
        mkt[f"sin_{k}"] = np.sin(2 * np.pi * k * t / 52)
        mkt[f"cos_{k}"] = np.cos(2 * np.pi * k * t / 52)

    for col in ["standard_display_impressions", "programmatic_display_impressions",
                "programmatic_video_impressions", "social_impressions",
                "audio_impressions", "ehr_impressions"]:
        if col in mkt.columns:
            mkt[f"{col}_log"] = np.log1p(mkt[col])

    return mkt


def split(mkt: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_train = PARAMS["TRAIN_WEEKS"]
    return mkt.iloc[:n_train].copy(), mkt.iloc[n_train:].copy()


def data_prep() -> None:
    print("=" * 60)
    print("  01  DATA PREP")
    print("=" * 60)

    print(f"\nLoading {GOLD_LABELLED} ...")

    # Market-level aggregation (Spark if available, else pandas + memory warning)
    mkt = aggregate_to_market(GOLD_LABELLED)
    print(f"  After aggregation: {len(mkt):,} market-level rows")

    # Ensure week is a proper datetime (Spark may return it as object/str)
    mkt["week"] = pd.to_datetime(mkt["week"])

    # Completeness check
    expected_weeks = 257
    actual_weeks   = mkt["week"].nunique()
    if actual_weeks < expected_weeks:
        print(f"  [warn] {actual_weeks} weeks found; expected {expected_weeks}. "
              f"Forward-filling gaps ...")
        full_spine = pd.date_range(mkt["week"].min(), mkt["week"].max(), freq="W-MON")
        mkt = (mkt.set_index("week")
                  .reindex(full_spine)
                  .ffill()
                  .reset_index()
                  .rename(columns={"index": "week"}))
    else:
        print(f"  Week spine complete: {actual_weeks} weeks ({mkt['week'].min().date()} -> "
              f"{mkt['week'].max().date()})")

    # Feature engineering (pandas — tiny)
    mkt = add_features(mkt)
    train, test = split(mkt)
    mkt["split"] = "train"
    mkt.loc[mkt.index >= PARAMS["TRAIN_WEEKS"], "split"] = "test"

    print(f"\n  Train: {len(train)} weeks | Test: {len(test)} weeks")
    print(f"  log_sales range: [{mkt['log_sales'].min():.2f}, {mkt['log_sales'].max():.2f}]")
    print(f"  Columns: {list(mkt.columns)}")

    # Write market series (UC table on Databricks via write_parquet -> saveAsTable)
    write_parquet(mkt, MARKET_SERIES)
    print(f"\n  Written -> {MARKET_SERIES}")
    print("=" * 60)


if __name__ == "__main__":
   data_prep()