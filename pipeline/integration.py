"""
05_integration.py — Connect BOCPD change-points to MMM attribution shifts.

For each detected CP, compares channel contribution BEFORE vs AFTER the break
to produce a root-cause classification per change-point.

Root-cause classification logic:
  HIGH_RESIDUAL + no channel shift  → artifact_trx_spike candidate
  f2f/email contribution jumps       → new_channel_spike candidate
  all contributions rise uniformly   → legit_spike candidate
  lifecycle/competitor_spend shifts  → organic/competitive event

Reads:  CP_CANDIDATES  (from 02_bocpd.py)
        CONTRIBUTIONS  (from 04_mmm_fit.py)
Writes: outputs/model_outputs/integration_report.csv

Run locally:  uv run python models/05_integration.py
On Databricks: %run ./05_integration
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from .config import (
    CP_CANDIDATES, CONTRIBUTIONS, MODEL_OUT,
    read_parquet, read_csv, PARAMS,
)
try:
    from databricks.sdk.runtime import *  # noqa: F401, F403
except ImportError:
    pass

INTEG_OUT = Path(MODEL_OUT) / "integration_report.csv"

# Window (in weeks) to compute pre/post means around a CP
PRE_WINDOW  = 8   # weeks before the CP
POST_WINDOW = 8   # weeks after the CP

# Classification thresholds
RESIDUAL_ZSCORE_THRESH = 2.5   # residual |z| above this → artifact candidate
CONTRIB_SHIFT_THRESH   = 0.10  # relative shift in a channel's contribution (10%)


def pre_post_mean(series: pd.Series, weeks: pd.Series,
                  cp_week: pd.Timestamp,
                  pre: int, post: int) -> tuple[float, float]:
    """Compute mean of series in the [cp_week - pre, cp_week) and [cp_week, cp_week + post) windows."""
    pre_mask  = (weeks >= cp_week - pd.Timedelta(weeks=pre)) & (weeks < cp_week)
    post_mask = (weeks >= cp_week) & (weeks < cp_week + pd.Timedelta(weeks=post))
    pre_val   = float(series[pre_mask].mean()) if pre_mask.any()  else np.nan
    post_val  = float(series[post_mask].mean()) if post_mask.any() else np.nan
    return pre_val, post_val


def classify_cp(row: dict) -> str:
    """
    Rule-based root-cause classification for a single change-point.
    This is intentionally simple and interpretable — override with ML if needed.
    """
    if row["residual_z_post"] > RESIDUAL_ZSCORE_THRESH:
        return "artifact_trx_spike_candidate"

    field_shift = row.get("field_contrib_shift_rel", 0)
    if field_shift > CONTRIB_SHIFT_THRESH:
        total_shift = row.get("total_contrib_shift_rel", 0)
        if field_shift / (total_shift + 1e-9) > 0.7:
            return "new_channel_spike_candidate"
        return "legit_spike_candidate"

    if row.get("lifecycle_shift", 0) > 0.5:
        return "organic_lifecycle_event"

    if row.get("competitor_shift_rel", 0) > 0.3:
        return "competitive_spend_event"

    return "unclassified"


def integration() -> None:
    print("=" * 60)
    print("  05  INTEGRATION")
    print("=" * 60)

    # Load inputs
    cps    = spark.table(CP_CANDIDATES).toPandas()
    contribs = spark.table(CONTRIBUTIONS).toPandas()
    contribs["week"] = pd.to_datetime(contribs["week"])

    if len(cps) == 0:
        print("  No CP candidates found. Run 02_bocpd.py first.")
        return
    print(f"  {len(cps)} CP candidates from BOCPD")

    # Identify contribution columns
    ch_contrib_cols = [c for c in contribs.columns if c.startswith("contrib_") and "_sat" in c]
    ctrl_cols       = [c for c in contribs.columns if c.startswith("contrib_")
                       and "_sat" not in c and "competitor" not in c]
    comp_col        = [c for c in contribs.columns if "competitor" in c and c.startswith("contrib_")]

    # Residual z-score on the full series (for artifact detection)
    res_mean = contribs["residual"].mean()
    res_std  = contribs["residual"].std()
    contribs["residual_z"] = (contribs["residual"] - res_mean) / (res_std + 1e-9)

    records = []
    for _, cp in cps.iterrows():
        cp_week = pd.Timestamp(cp["week"])

        rec = {
            "cp_week":   cp_week.date(),
            "cp_prob":   round(float(cp["cp_prob"]), 4),
            "log_sales": round(float(cp["log_sales"]), 4),
        }

        # Pre/post for each channel contribution
        total_pre, total_post = 0.0, 0.0
        field_pre, field_post = 0.0, 0.0

        for col in ch_contrib_cols:
            pre, post = pre_post_mean(contribs[col], contribs["week"],
                                       cp_week, PRE_WINDOW, POST_WINDOW)
            rec[f"{col}_pre"]  = round(pre,  4) if not np.isnan(pre)  else None
            rec[f"{col}_post"] = round(post, 4) if not np.isnan(post) else None
            total_pre  += pre  if not np.isnan(pre)  else 0
            total_post += post if not np.isnan(post) else 0
            if any(x in col for x in ["f2f", "email", "phone", "samples"]):
                field_pre  += pre  if not np.isnan(pre)  else 0
                field_post += post if not np.isnan(post) else 0

        rec["total_contrib_shift_rel"] = round(
            (total_post - total_pre) / (abs(total_pre) + 1e-9), 4)
        rec["field_contrib_shift_rel"] = round(
            (field_post - field_pre) / (abs(field_pre) + 1e-9), 4)

        # Lifecycle shift
        if "contrib_lc_num" in contribs.columns:
            lc_pre, lc_post = pre_post_mean(contribs["contrib_lc_num"], contribs["week"],
                                             cp_week, PRE_WINDOW, POST_WINDOW)
            rec["lifecycle_shift"] = round((lc_post or 0) - (lc_pre or 0), 4)

        # Competitor spend shift
        if comp_col:
            cp_pre, cp_post = pre_post_mean(contribs[comp_col[0]], contribs["week"],
                                             cp_week, PRE_WINDOW, POST_WINDOW)
            rec["competitor_shift_rel"] = round(
                ((cp_post or 0) - (cp_pre or 0)) / (abs(cp_pre or 0) + 1e-9), 4)

        # Residual z-score post CP (artifact signature)
        post_mask = ((contribs["week"] >= cp_week) &
                     (contribs["week"] < cp_week + pd.Timedelta(weeks=POST_WINDOW)))
        rec["residual_z_post"] = round(
            float(contribs.loc[post_mask, "residual_z"].mean()) if post_mask.any() else 0, 4)

        # Root-cause classification
        rec["classification"] = classify_cp(rec)

        records.append(rec)

    report = pd.DataFrame(records)

    # Print summary
    print(f"\n  Classification summary:")
    print(report.groupby("classification")["cp_week"].count().to_string())
    print(f"\n  Full report preview:")
    print(report[["cp_week", "cp_prob", "classification",
                  "total_contrib_shift_rel", "field_contrib_shift_rel",
                  "residual_z_post"]].to_string(index=False))

    report.to_csv(INTEG_OUT, index=False)
    print(f"\n  Integration report written -> {INTEG_OUT}")
    print("=" * 60)


if __name__ == "__main__":
    integration()
