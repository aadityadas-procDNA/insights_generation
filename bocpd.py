"""
02_bocpd.py — Bayesian Online Change-Point Detection on the weekly sales series.

Algorithm: Adams & MacKay (2007), Normal-Gamma observation model.
Install:   pip install bayesian_changepoint_detection  (or uv add)

Reads:  MARKET_SERIES   (output of 01_data_prep.py)
Writes: CP_PROBS        (week, log_sales, cp_prob per row)
        CP_CANDIDATES   (flagged change-point dates with context)

Run locally:  uv run python models/02_bocpd.py
On Databricks: %run ./02_bocpd  (single-node; no Spark needed for BOCPD)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.special import gammaln

from databricks.sdk.runtime import *

from config import (
    MARKET_SERIES, CP_PROBS, CP_CANDIDATES,
    read_parquet, write_parquet, read_csv, PARAMS,
)


# ── Pure-numpy BOCPD (no external package required) ──────────────────────────
# Implements Adams & MacKay (2007) with Normal-Gamma (StudentT predictive).
# Equivalent to bayesian_changepoint_detection.bocd but self-contained for
# Databricks environments where pip installs may be restricted.

def _student_t_log_pred(x: float, mu: np.ndarray, kappa: np.ndarray,
                         alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Log predictive density of x under each run-length hypothesis."""
    nu      = 2 * alpha
    scale2  = beta * (kappa + 1) / (alpha * kappa)
    log_c   = (gammaln((nu + 1) / 2)
               - gammaln(nu / 2)
               - 0.5 * np.log(np.pi * nu * scale2))
    log_lik = log_c - ((nu + 1) / 2) * np.log(1 + (x - mu) ** 2 / (nu * scale2))
    return log_lik


def _ng_update(x: float, mu: np.ndarray, kappa: np.ndarray,
               alpha: np.ndarray, beta: np.ndarray):
    """Normal-Gamma conjugate update: return posterior hyperparameters."""
    kappa_new = kappa + 1
    mu_new    = (kappa * mu + x) / kappa_new
    alpha_new = alpha + 0.5
    beta_new  = beta + (kappa * (x - mu) ** 2) / (2 * kappa_new)
    return mu_new, kappa_new, alpha_new, beta_new


def run_bocpd(log_sales: np.ndarray,
              mu_0: float, kappa_0: float, alpha_0: float, beta_0: float,
              hazard_lam: int = 52) -> np.ndarray:
    """
    Run BOCPD over the full series. Returns cp_prob array (length T).
    cp_prob[t] = P(change-point occurred at step t).
    """
    T          = len(log_sales)
    hazard     = 1.0 / hazard_lam

    # Run-length posterior: R[r] = P(run length = r at current step)
    # Start with run length = 0 (certain CP at t=0)
    R      = np.array([1.0])
    mu     = np.array([mu_0])
    kappa  = np.array([kappa_0])
    alpha  = np.array([alpha_0])
    beta   = np.array([beta_0])

    cp_probs = np.zeros(T)

    for t, x in enumerate(log_sales):
        # Predictive probabilities under each run-length hypothesis
        log_pred = _student_t_log_pred(x, mu, kappa, alpha, beta)
        pred     = np.exp(log_pred - log_pred.max())   # stable softmax

        # Growth probability: existing run lengths survive (hazard not triggered)
        R_growth = R * pred * (1.0 - hazard)

        # CP probability: all existing run lengths collapse to 0
        R_cp = np.sum(R * pred * hazard)

        # New run-length distribution: prepend the CP term
        R_new   = np.append(R_cp, R_growth)
        R_new  /= R_new.sum()                          # normalise

        cp_probs[t] = R_new[0]                         # P(RL=0) = CP just happened

        # Update sufficient statistics for each hypothesis
        mu_new, kappa_new, alpha_new, beta_new = _ng_update(
            x, mu, kappa, alpha, beta)

        # Extend arrays: new run-length=0 gets prior hyperparams
        mu     = np.append(mu_0,    mu_new)
        kappa  = np.append(kappa_0, kappa_new)
        alpha  = np.append(alpha_0, alpha_new)
        beta   = np.append(beta_0,  beta_new)
        R      = R_new

    return cp_probs


def extract_candidates(weeks: pd.Series, log_sales: np.ndarray,
                        cp_probs: np.ndarray,
                        threshold: float, min_dist: int) -> pd.DataFrame:
    """Extract local-maximum CP candidates above threshold with minimum spacing."""
    peaks, props = find_peaks(cp_probs, height=threshold, distance=min_dist)
    if len(peaks) == 0:
        return pd.DataFrame(columns=["week", "cp_prob", "log_sales"])

    return pd.DataFrame({
        "week":      weeks.iloc[peaks].values,
        "cp_prob":   cp_probs[peaks],
        "log_sales": log_sales[peaks],
        "week_idx":  peaks,
    }).sort_values("cp_prob", ascending=False).reset_index(drop=True)


def bocpd() -> None:
    print("=" * 60)
    print("  02  BOCPD")
    print("=" * 60)

    mkt = read_parquet(MARKET_SERIES)
    train = mkt[mkt["split"] == "train"].copy()

    log_sales_all   = mkt["log_sales"].values
    log_sales_train = train["log_sales"].values

    # Prior from training data
    mu_0    = float(log_sales_train.mean())
    kappa_0 = 1.0
    alpha_0 = 1.0
    beta_0  = float(log_sales_train.var())

    print(f"\n  Prior: mu_0={mu_0:.3f}  beta_0={beta_0:.4f}")
    print(f"  Hazard lambda: {PARAMS['BOCPD_LAMBDA']} weeks  "
          f"(P(CP per week) = {1/PARAMS['BOCPD_LAMBDA']:.3f})")
    print(f"  Running BOCPD on {len(log_sales_all)} weeks ...")

    cp_probs = run_bocpd(
        log_sales_all,
        mu_0=mu_0, kappa_0=kappa_0, alpha_0=alpha_0, beta_0=beta_0,
        hazard_lam=PARAMS["BOCPD_LAMBDA"],
    )

    # ── Save full probability series ─────────────────────────────────────────
    prob_df = mkt[["week", "log_sales", "lifecycle_stage", "competitor_spend",
                   "split"]].copy()
    prob_df["cp_prob"] = cp_probs
    write_parquet(prob_df, CP_PROBS)
    print(f"\n  CP probabilities written -> {CP_PROBS}")

    # ── Extract candidates ────────────────────────────────────────────────────
    candidates = extract_candidates(
        mkt["week"], log_sales_all, cp_probs,
        threshold=PARAMS["BOCPD_THRESHOLD"],
        min_dist=PARAMS["BOCPD_MIN_DIST"],
    )
    candidates.to_csv(CP_CANDIDATES, index=False)
    print(f"  {len(candidates)} change-point candidates written -> {CP_CANDIDATES}")

    if len(candidates):
        print(f"\n  Top candidates:")
        print(candidates.head(8).to_string(index=False))

    # ── Quick validation against expected organic CPs ────────────────────────
    expected = {
        "OTEZLA maturity / EUCRISA launch": pd.Timestamp("2017-01-02"),
        "TREMFYA launch":                   pd.Timestamp("2017-07-03"),
    }
    LAG_TOLERANCE = pd.Timedelta("42 days")   # 6-week detection lag is acceptable

    print("\n  Organic CP validation (+-6 week window):")
    for name, ts in expected.items():
        window = candidates[
            (candidates["week"] >= ts - LAG_TOLERANCE) &
            (candidates["week"] <= ts + LAG_TOLERANCE)
        ]
        hit = "DETECTED" if len(window) else "MISSED"
        print(f"    {name} ({ts.date()}) -> {hit}")

    print("=" * 60)


if __name__ == "__main__":
    bocpd()
