"""
03_mmm_data_prep.py — Adstock, saturation, scaling, final model matrix for MMM.

Reads:  MARKET_SERIES  (output of 01_data_prep.py)
Writes: MODEL_MATRIX   (X matrix + y vector, ready for PyMC in 04_mmm_fit.py)

Run locally:  uv run python models/03_mmm_data_prep.py
On Databricks: %run ./03_mmm_data_prep
"""

from __future__ import annotations

try:
    from databricks.sdk.runtime import *  # noqa: F401, F403
except ImportError:
    pass
import json
import numpy as np
import pandas as pd

from pipeline.config import (
    MARKET_SERIES, MODEL_MATRIX, MODEL_OUT,
    read_parquet, write_parquet, PARAMS,
)
from pathlib import Path

# ── Channel groups ────────────────────────────────────────────────────────────
# GROUP A: field channels — fast decay (0.15-0.40), mild concave saturation
GROUP_A = [
    "f2f", "phone_call", "f2f_short_call", "samples", "speaker",
    "email_opens", "tp_email_opens",
    "ehr_impressions", "doximity_opens", "epocrates_opens", "sermo_impressions",
]

# GROUP B: broadcast/digital — slow decay (0.50-0.80), S-curve saturation
GROUP_B = [
    "tv_grps",
    "standard_display_impressions",
    "programmatic_display_impressions",
    "programmatic_video_impressions",
    "social_impressions",
    "audio_impressions",
]

# GROUP C: exogenous controls — no adstock, enter model directly
GROUP_C = ["competitor_spend"]

# Lifecycle and seasonality covariates (no adstock/saturation)
COVARIATE_COLS = ["lc_num", "week_idx"]
FOURIER_COLS   = ["sin_1", "cos_1", "sin_2", "cos_2"]   # adjust if K != 2


# ── Transformations ───────────────────────────────────────────────────────────

def geometric_adstock(x: np.ndarray, decay: float) -> np.ndarray:
    """
    Geometric carry-over adstock.
    adstock[t] = x[t] + decay * adstock[t-1]
    Models the lagged effect of channel activity on prescribing behaviour.
    """
    out = np.zeros_like(x, dtype=float)
    for t in range(len(x)):
        out[t] = x[t] + decay * (out[t - 1] if t > 0 else 0.0)
    return out


def hill_saturation(x: np.ndarray, alpha: float, K: float) -> np.ndarray:
    """
    Hill (diminishing returns) saturation.
    sat(x) = x^alpha / (x^alpha + K^alpha)
    alpha < 1: fully concave (first unit dominates)
    alpha > 1: S-curve with a lagged inflection point
    K: half-saturation point (inflection) — set to median of adstocked series.
    """
    xa = np.power(np.maximum(x, 0), alpha)
    Ka = K ** alpha
    return xa / (xa + Ka + 1e-12)


def transform_channels(mkt: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Apply adstock + saturation to all channel groups.
    Returns the transformed DataFrame and a metadata dict (for inverse-transform).
    """
    decay  = PARAMS["DECAY"]
    alpha_a = PARAMS["HILL_ALPHA_FIELD"]
    alpha_b = PARAMS["HILL_ALPHA_BROADCAST"]
    meta   = {}

    for col in GROUP_A:
        if col not in mkt.columns:
            continue
        ads_col = f"{col}_ads"
        sat_col = f"{col}_sat"
        d = decay.get(col, 0.3)
        adstocked = geometric_adstock(mkt[col].values, d)
        K = float(np.median(adstocked[adstocked > 0])) if (adstocked > 0).any() else 1.0
        saturated = hill_saturation(adstocked, alpha_a, K)
        mkt[ads_col] = adstocked
        mkt[sat_col] = saturated
        meta[col] = {"type": "A", "decay": d, "hill_alpha": alpha_a, "hill_K": K}

    for col in GROUP_B:
        if col not in mkt.columns:
            continue
        ads_col = f"{col}_ads"
        sat_col = f"{col}_sat"
        d = decay.get(col, 0.6)
        adstocked = geometric_adstock(mkt[col].values, d)
        K = float(np.median(adstocked[adstocked > 0])) if (adstocked > 0).any() else 1.0
        saturated = hill_saturation(adstocked, alpha_b, K)
        mkt[ads_col] = adstocked
        mkt[sat_col] = saturated
        meta[col] = {"type": "B", "decay": d, "hill_alpha": alpha_b, "hill_K": K}

    return mkt, meta


def build_model_matrix(mkt: pd.DataFrame) -> tuple[np.ndarray, np.ndarray,
                                                    list[str], dict, dict]:
    """
    Build final X (feature matrix) and y (log_sales) for MMM.
    Returns: X_scaled, y, feature_names, scaler_params, channel_meta
    """
    # Final feature columns for MMM (saturated versions)
    feature_cols = (
        [f"{c}_sat" for c in GROUP_A if f"{c}_sat" in mkt.columns]
        + [f"{c}_sat" for c in GROUP_B if f"{c}_sat" in mkt.columns]
        + GROUP_C
        + COVARIATE_COLS
        + FOURIER_COLS
    )
    # Keep only columns that exist (some channels may be absent for certain products)
    feature_cols = [c for c in feature_cols if c in mkt.columns]

    X_raw = mkt[feature_cols].fillna(0).values.astype(float)
    y     = mkt["log_sales"].values.astype(float)

    # Standardise on TRAINING rows only; apply same scaler to full series
    train_mask = (mkt["split"] == "train").values
    X_mean = X_raw[train_mask].mean(axis=0)
    X_std  = X_raw[train_mask].std(axis=0)
    X_std  = np.where(X_std == 0, 1.0, X_std)   # avoid divide-by-zero for constant cols

    X_scaled = (X_raw - X_mean) / X_std

    scaler_params = {
        "feature_cols": feature_cols,
        "X_mean": X_mean.tolist(),
        "X_std":  X_std.tolist(),
        "y_mean": float(y[train_mask].mean()),
        "y_std":  float(y[train_mask].std()),
    }
    return X_scaled, y, feature_cols, scaler_params


def mmm_data_prep() -> None:
    print("=" * 60)
    print("  03  MMM DATA PREP")
    print("=" * 60)

    mkt = read_parquet(MARKET_SERIES)

    # Adstock + saturation
    mkt, channel_meta = transform_channels(mkt)
    print(f"  Transformed {len(channel_meta)} channels (adstock + Hill saturation)")

    # Model matrix
    X, y, feature_cols, scaler_params = build_model_matrix(mkt)

    print(f"  X shape: {X.shape}  (should be ~257 x 20-28)")
    print(f"  y range: [{y.min():.3f}, {y.max():.3f}]  (log_sales)")
    print(f"  Features: {feature_cols}")

    # Collinearity check
    corr = np.corrcoef(X.T)
    n    = len(feature_cols)
    hi_corr = [
        (feature_cols[i], feature_cols[j], float(corr[i, j]))
        for i in range(n) for j in range(i + 1, n)
        if abs(corr[i, j]) > 0.85
    ]
    if hi_corr:
        print(f"\n  [warn] High collinearity pairs (|r|>0.85) — consider grouping:")
        for a, b, r in hi_corr[:10]:
            print(f"    {a} <-> {b}  r={r:.3f}")
    else:
        print("  No high collinearity pairs (|r|>0.85).")

    # Persist transformed mkt + X/y as parquet
    # Store X as extra columns so the file is self-contained for 04_mmm_fit
    x_df = pd.DataFrame(X, columns=[f"X_{c}" for c in feature_cols])
    x_df["y"] = y
    x_df["week"] = mkt["week"].values
    x_df["split"] = mkt["split"].values
    write_parquet(x_df, MODEL_MATRIX)

    # Save scaler and channel metadata as JSON (needed in 05 and 06 for back-transform)
    meta_path = Path(MODEL_OUT) / "mmm_meta.json"
    with open(meta_path, "w") as f:
        json.dump({"scaler": scaler_params, "channel_meta": channel_meta}, f, indent=2)

    print(f"\n  Model matrix written -> {MODEL_MATRIX}")
    print(f"  Scaler + channel meta -> {meta_path}")
    print("=" * 60)


if __name__ == "__main__":
    mmm_data_prep()
