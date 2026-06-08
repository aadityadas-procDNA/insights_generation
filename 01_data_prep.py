"""
01_data_prep.py — Load gold layer, aggregate to market level, build feature set.

Reads:  GOLD_LABELLED  (engagement_gold_layer_labelled.parquet)
Writes: MARKET_SERIES  (257 rows × ~30 cols; one row per week, market-level)

Run locally:  uv run python models/01_data_prep.py
On Databricks: %run ./01_data_prep  (or as a notebook cell)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    GOLD_LABELLED, MARKET_SERIES,
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


def aggregate_to_market(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse NPI × product × week to week × product (market level).

    Field channels are summed (each HCP contributes independently).
    Broadcast channels are averaged (they are the same value for all HCPs in a week;
    summing would inflate them ~10,000x and destroy the MMM coefficients).
    """
    agg_dict = {c: "sum" for c in FIELD_COLS + TARGET_COLS}
    agg_dict.update({c: "mean" for c in BROADCAST_COLS})
    agg_dict["lifecycle_stage"] = "first"   # identical per product×week by construction

    mkt = df.groupby(["product", "week"]).agg(agg_dict).reset_index()
    return mkt


def add_features(mkt: pd.DataFrame) -> pd.DataFrame:
    """Add log_sales, event flags, lifecycle numeric, and Fourier seasonality."""
    mkt = mkt.sort_values("week").reset_index(drop=True)

    # Log-transform target
    mkt["log_sales"] = np.log(mkt["sales"].clip(lower=1e-6))

    # Lifecycle numeric encoding
    stage_map = {"pre_launch": 0, "launch": 1, "growth": 2, "maturity": 3, "decline": 4}
    mkt["lc_num"] = mkt["lifecycle_stage"].map(stage_map).fillna(2).astype(int)

    # Linear week index (smooth trend proxy)
    mkt["week_idx"] = np.arange(len(mkt))

    # Event flags — organic structural breaks (for BOCPD ground-truth validation)
    for name, ts in ORGANIC_CPS.items():
        mkt[f"flag_{name}"] = (mkt["week"] >= ts).astype(int)

    # Fourier seasonality: K pairs at annual frequency
    K = PARAMS["FOURIER_K"]
    t = mkt["week_idx"].values
    for k in range(1, K + 1):
        mkt[f"sin_{k}"] = np.sin(2 * np.pi * k * t / 52)
        mkt[f"cos_{k}"] = np.cos(2 * np.pi * k * t / 52)

    # Log-transform large impression channels (log1p scale for MMM feature engineering)
    for col in ["standard_display_impressions", "programmatic_display_impressions",
                "programmatic_video_impressions", "social_impressions",
                "audio_impressions", "ehr_impressions"]:
        if col in mkt.columns:
            mkt[f"{col}_log"] = np.log1p(mkt[col])

    return mkt


def split(mkt: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_train = PARAMS["TRAIN_WEEKS"]
    return mkt.iloc[:n_train].copy(), mkt.iloc[n_train:].copy()


def main() -> None:
    print("=" * 60)
    print("  01  DATA PREP")
    print("=" * 60)

    print(f"\nLoading {GOLD_LABELLED} ...")
    df = read_parquet(GOLD_LABELLED)
    print(f"  {len(df):,} rows  |  {df['is_anomaly'].sum():,} labelled anomaly rows")

    # Keep only clean rows for model fitting; anomaly rows validated separately in 06
    df_clean = df[df["is_anomaly"] == 0].copy()
    print(f"  Using {len(df_clean):,} clean rows (is_anomaly==0) for model fitting")

    # Filter to focus product
    df_product = df_clean[df_clean["product"] == FOCUS].copy()
    print(f"  {FOCUS}: {len(df_product):,} rows across {df_product['hcp_id'].nunique():,} HCPs")

    # Market-level aggregation
    mkt = aggregate_to_market(df_product)
    print(f"  After aggregation: {len(mkt):,} market-level rows")

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

    # Feature engineering
    mkt = add_features(mkt)
    train, test = split(mkt)
    mkt["split"] = "train"
    mkt.loc[mkt.index >= PARAMS["TRAIN_WEEKS"], "split"] = "test"

    print(f"\n  Train: {len(train)} weeks | Test: {len(test)} weeks")
    print(f"  log_sales range: [{mkt['log_sales'].min():.2f}, {mkt['log_sales'].max():.2f}]")
    print(f"  Columns: {list(mkt.columns)}")

    write_parquet(mkt, MARKET_SERIES)
    print(f"\n  Written -> {MARKET_SERIES}")
    print("=" * 60)


if __name__ == "__main__":
    main()
