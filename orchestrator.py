# Databricks notebook source
# MAGIC %pip install -r requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC # Enables autoreload; learn more at https://docs.databricks.com/en/files/workspace-modules.html#autoreload-for-python-modules
# MAGIC # To disable autoreload; run %autoreload 0

# COMMAND ----------

from config import *
from data_prep import *
from bocpd import *
from mmm_data_prep import *
from mmm_fit import *
from integration import *

# COMMAND ----------


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

# COMMAND ----------

if spark.catalog.tableExists(MARKET_SERIES):
    print(f"MARKET SERIES TABLE EXISTS as {MARKET_SERIES}")
else:
    data_prep()
    

# COMMAND ----------

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


# COMMAND ----------

if spark.catalog.tableExists(CP_PROBS) and spark.catalog.tableExists(CP_CANDIDATES):
    print(f"CP PROBS TABLE EXISTS as {CP_PROBS}")
else:
    bocpd()

# COMMAND ----------


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


# COMMAND ----------

mmm_data_prep()

# COMMAND ----------

def mmm_fit() -> None:
    print("=" * 60)
    print("  04  MMM FIT")
    print("=" * 60)

    # ── Load model matrix and metadata ────────────────────────────────────────
    mat = read_parquet(MODEL_MATRIX)
    meta_path = Path(MODEL_OUT) / "mmm_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    scaler       = meta["scaler"]
    feature_cols = scaler["feature_cols"]

    # Prefix used when storing X columns in parquet
    X_cols = [f"X_{c}" for c in feature_cols]
    X      = mat[X_cols].values
    y      = mat["y"].values

    train_mask = mat["split"] == "train"
    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[~train_mask], y[~train_mask]

    print(f"  X shape: {X.shape}  "
          f"(train={train_mask.sum()}, test={(~train_mask).sum()})")

    # ── Build and sample model ────────────────────────────────────────────────
    try:
        import pymc as pm
        import arviz as az
    except ImportError:
        print("  [error] PyMC/ArviZ not installed. Run: uv add pymc arviz")
        return

    mmm, ch_idx, comp_idx, ctrl_idx = build_pymc_model(
        X_train, y_train, feature_cols, scaler)

    print(f"\n  Sampling: {PARAMS['MCMC_CHAINS']} chains x "
          f"{PARAMS['MCMC_DRAWS']} draws (tune={PARAMS['MCMC_TUNE']}) ...")

    sampler_kwargs = dict(
        draws         = PARAMS["MCMC_DRAWS"],
        tune          = PARAMS["MCMC_TUNE"],
        chains        = PARAMS["MCMC_CHAINS"],
        target_accept = PARAMS["MCMC_TARGET_ACCEPT"],
        random_seed   = PARAMS["MCMC_SEED"],
        return_inferencedata = True,
    )
    # Use numpyro backend if available (10x faster via JAX)
    try:
        import numpyro  # noqa: F401
        sampler_kwargs["nuts_sampler"] = "numpyro"
        print("  numpyro backend detected — using JAX for faster sampling")
    except ImportError:
        pass

    with mmm:
        trace = pm.sample(**sampler_kwargs)

    # ── Convergence diagnostics ───────────────────────────────────────────────
    summary = az.summary(trace, var_names=["alpha", "beta_ch", "sigma"],
                         round_to=4)
    print(f"\n  Convergence summary (top rows):\n{summary.head(12).to_string()}")

    max_rhat = summary["r_hat"].max()
    min_ess  = summary["ess_bulk"].min()
    divs     = trace.sample_stats["diverging"].sum().item()

    print(f"\n  max R-hat:    {max_rhat:.4f}  (target < 1.01)")
    print(f"  min ESS_bulk: {min_ess:.0f}   (target > 400)")
    print(f"  Divergences:  {divs}           (target = 0)")

    if max_rhat > 1.05:
        warnings.warn("R-hat > 1.05: chains have NOT converged. "
                      "Increase draws or tighten priors.")
    if divs > 0:
        warnings.warn(f"{divs} divergences — raise target_accept to 0.95 "
                      "or add stronger priors for correlated channels.")

    # ── Posterior predictive check ─────────────────────────────────────────────
    with mmm:
        ppc = pm.sample_posterior_predictive(trace)

    y_hat_train = ppc.posterior_predictive["y_obs"].mean(dim=["chain", "draw"]).values
    mape_train  = float(np.mean(np.abs(np.exp(y_hat_train) - np.exp(y_train))
                                / np.exp(y_train)) * 100)
    print(f"\n  In-sample MAPE: {mape_train:.1f}%  (target < 10%)")

    # ── Save trace ─────────────────────────────────────────────────────────────
    try:
        trace.to_netcdf(MMM_TRACE)
        print(f"\n  Trace saved -> {MMM_TRACE}")
    except Exception as e:
        print(f"  [error] Failed to save trace: {e}")

    # ── Contribution decomposition ─────────────────────────────────────────────
    # posterior mean of each parameter
    beta_ch_mean   = trace.posterior["beta_ch"].mean(dim=["chain", "draw"]).values
    beta_ctrl_mean = trace.posterior["beta_ctrl"].mean(dim=["chain", "draw"]).values
    alpha_mean     = float(trace.posterior["alpha"].mean())

    contrib = pd.DataFrame({"week": mat["week"].values})
    contrib["baseline"] = alpha_mean

    for i, col_idx in enumerate(ch_idx):
        col_name = feature_cols[col_idx]
        contrib[f"contrib_{col_name}"] = beta_ch_mean[i] * X[:, col_idx]

    for i, col_idx in enumerate(ctrl_idx):
        col_name = feature_cols[col_idx]
        contrib[f"contrib_{col_name}"] = beta_ctrl_mean[i] * X[:, col_idx]

    if comp_idx:
        beta_comp_mean = trace.posterior["beta_comp"].mean(dim=["chain", "draw"]).values
        for i, col_idx in enumerate(comp_idx):
            col_name = feature_cols[col_idx]
            contrib[f"contrib_{col_name}"] = beta_comp_mean[i] * X[:, col_idx]

    contrib["y_actual"]    = y
    contrib["y_predicted"] = contrib[[c for c in contrib.columns
                                      if c.startswith("contrib_")]].sum(axis=1) + alpha_mean
    contrib["residual"]    = contrib["y_actual"] - contrib["y_predicted"]
    contrib["split"]       = mat["split"].values

    write_parquet(contrib, CONTRIBUTIONS)
    print(f"  Contributions written -> {CONTRIBUTIONS}")

    # ── Scaler for back-transform (saved to JSON in 03, used in 06) ───────────
    print("\n  Done. Run 05_integration.py to map CPs to attribution shifts.")
    print("=" * 60)


# COMMAND ----------

mmm_fit()

# COMMAND ----------


def integration() -> None:
    print("=" * 60)
    print("  05  INTEGRATION")
    print("=" * 60)

    # Load inputs
    cps    = pd.read_csv(CP_CANDIDATES, parse_dates=["week"])
    contribs = read_parquet(CONTRIBUTIONS)
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

# COMMAND ----------

integration()
