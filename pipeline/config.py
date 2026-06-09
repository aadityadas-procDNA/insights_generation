"""
config.py — Path and environment configuration for the BOCPD + MMM model layer.

Auto-detects Databricks runtime. On Databricks, data is referenced via Unity Catalog:
  - tabular data        -> catalog.schema.table   (read with spark.table / write saveAsTable)
  - file artifacts (.nc) -> /Volumes/catalog/schema/volume/...  (a UC Volume path)

Override by setting env vars:
  DATABRICKS_RUNTIME_VERSION   (set automatically in Databricks)
  UC_CATALOG, UC_SCHEMA, UC_VOLUME   (optional overrides for the UC location)
  GOLD_LABELLED_TABLE          (optional override for the input table name)
  MODEL_OUTPUT_SCHEMA          (optional override for where output tables are written)

Local usage:
  python 01_data_prep.py

Databricks usage:
  Ensure the input tables exist in UC (catalog.schema.table) and a Volume exists for
  file artifacts. Set UC_CATALOG / UC_SCHEMA / UC_VOLUME below or via env vars.
"""

import os
from pathlib import Path
try:
    from databricks.sdk.runtime import *  # noqa: F401, F403
except ImportError:
    pass

# ── Environment detection ─────────────────────────────────────────────────────
ON_DATABRICKS = bool(os.getenv("DATABRICKS_RUNTIME_VERSION", ""))

# ── Unity Catalog location — update to match your workspace ───────────────────
UC_CATALOG = os.getenv("UC_CATALOG", "nexora_poc_catalog")
UC_SCHEMA = os.getenv("UC_SCHEMA", "gold")  # schema holding input + output tables
UC_VOLUME = os.getenv(
    "UC_VOLUME", "model_artifacts"
)  # UC Volume for non-tabular files (.nc, etc.)

# Output tables can live in the same schema or a dedicated one
UC_OUT_SCHEMA = os.getenv("MODEL_OUTPUT_SCHEMA", UC_SCHEMA)

# Resolved UC prefixes
_UC_IN = f"{UC_CATALOG}.{UC_SCHEMA}"
_UC_OUT = f"{UC_CATALOG}.{UC_OUT_SCHEMA}"
_VOL_DIR = f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/{UC_VOLUME}"  # POSIX path on the cluster

# ── Local paths (relative to synth-pharma/outputs/) ──────────────────────────
_LOCAL_OUTPUTS = Path(__file__).parent.parent / "outputs"
_LOCAL_MODELS = Path(__file__).parent.parent / "outputs" / "model_outputs"

# ── Resolved references ───────────────────────────────────────────────────────
# NOTE: On Databricks these are TABLE NAMES (catalog.schema.table), except the
# trace which is a Volume FILE PATH. Locally they remain filesystem paths.
if ON_DATABRICKS:
    # Input data (UC tables)
    GOLD_LABELLED = os.getenv(
        "GOLD_LABELLED_TABLE", f"{_UC_IN}.engagement_mmm_labelled"
    )
    ANSWER_KEY = f"{_UC_IN}.anomaly_answer_key"

    # Output schema target
    MODEL_OUT = _VOL_DIR

    # Intermediate artefacts written between stages (UC tables)
    MARKET_SERIES = f"{_UC_OUT}.market_series"
    MODEL_MATRIX = f"{_UC_OUT}.model_matrix"
    CP_PROBS = f"{_UC_OUT}.bocpd_cp_probs"
    CP_CANDIDATES = f"{_UC_OUT}.bocpd_cp_candidates"
    CONTRIBUTIONS = f"{_UC_OUT}.mmm_contributions"
    VALIDATION_RPT = f"{_UC_OUT}.validation_report"

    # Non-tabular artefact -> UC Volume file path (NOT a table)
    MMM_TRACE = f"{_VOL_DIR}/mmm_trace.nc"  # ArviZ InferenceData (NetCDF)

else:
    # Input data (local files)
    GOLD_LABELLED = os.getenv(
        "GOLD_LABELLED_TABLE", str(_LOCAL_OUTPUTS / "engagement_mmm_labelled.parquet")
    )
    ANSWER_KEY = str(_LOCAL_OUTPUTS / "anomaly_answer_key.csv")

    _LOCAL_MODELS.mkdir(parents=True, exist_ok=True)
    MODEL_OUT = str(_LOCAL_MODELS)

    MARKET_SERIES = str(_LOCAL_MODELS / "market_series.parquet")
    MODEL_MATRIX = str(_LOCAL_MODELS / "model_matrix.parquet")
    CP_PROBS = str(_LOCAL_MODELS / "bocpd_cp_probs.parquet")
    CP_CANDIDATES = str(_LOCAL_MODELS / "bocpd_cp_candidates.csv")
    CONTRIBUTIONS = str(_LOCAL_MODELS / "mmm_contributions.parquet")
    VALIDATION_RPT = str(_LOCAL_MODELS / "validation_report.csv")
    MMM_TRACE = str(_LOCAL_MODELS / "mmm_trace.nc")


# ── UC-aware I/O helpers ──────────────────────────────────────────────────────
# On Databricks: read/write Unity Catalog TABLES (the `ref` is catalog.schema.table).
# Locally: read/write files (the `ref` is a filesystem path).
# `spark` is pre-defined in the Databricks notebook scope.


def read_parquet(ref: str, **kwargs):
    """Read a UC table (Databricks) or a parquet file (local) into pandas."""
    import pandas as pd

    if ON_DATABRICKS:
        return spark.table(ref).toPandas()  # noqa: F821
    return pd.read_parquet(ref, **kwargs)


def write_parquet(df, ref: str, **kwargs):
    """Write pandas DF as a UC table (Databricks) or parquet file (local)."""
    if ON_DATABRICKS:
        (
            spark.createDataFrame(df)  # noqa: F821
            .write.mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(ref)
        )
        return
    df.to_parquet(ref, index=False, compression="snappy", **kwargs)


def read_csv(ref: str, **kwargs):
    """Read a UC table (Databricks) or a CSV file (local) into pandas."""
    import pandas as pd

    if ON_DATABRICKS:
        return spark.table(ref).toPandas()  # noqa: F821
    return pd.read_csv(ref, **kwargs)


def write_csv(df, ref: str, **kwargs):
    """Write pandas DF as a UC table (Databricks) or a CSV file (local).

    NOTE: on Databricks a 'CSV output' becomes a UC table — that is the governed,
    queryable equivalent. If you genuinely need a .csv file on Databricks, write it
    to the UC Volume path instead (see _VOL_DIR).
    """
    if ON_DATABRICKS:
        (
            spark.createDataFrame(df)  # noqa: F821
            .write.mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(ref)
        )
        return
    df.to_csv(ref, index=False, **kwargs)


# ── Model hyper-parameters (centralised here so Databricks widgets can override)
PARAMS = {
    # Data prep
    "FOCUS_PRODUCT": "OTEZLA",
    "TRAIN_WEEKS": 180,  # weeks 1-180 for training; 181-257 for hold-out
    "FOURIER_K": 2,  # number of Fourier pairs for seasonality
    # BOCPD
    "BOCPD_LAMBDA": 52,  # expected run-length (weeks)
    "BOCPD_THRESHOLD": 0.30,  # CP probability threshold
    "BOCPD_MIN_DIST": 8,  # minimum weeks between CPs
    # MMM adstock decay starting values (learned in v2)
    "DECAY": {
        "f2f": 0.30,
        "phone_call": 0.25,
        "f2f_short_call": 0.25,
        "samples": 0.30,
        "speaker": 0.40,
        "email_opens": 0.20,
        "tp_email_opens": 0.20,
        "ehr_impressions": 0.15,
        "tv_grps": 0.70,
        "standard_display_impressions": 0.60,
        "programmatic_display_impressions": 0.55,
        "programmatic_video_impressions": 0.65,
        "social_impressions": 0.55,
        "audio_impressions": 0.45,
    },
    # MMM saturation (Hill) K = median of adstocked channel; alpha per group
    "HILL_ALPHA_FIELD": 0.8,
    "HILL_ALPHA_BROADCAST": 2.0,
    # MMM NUTS
    "MCMC_DRAWS": 2000,
    "MCMC_TUNE": 1000,
    "MCMC_CHAINS": 4,
    "MCMC_TARGET_ACCEPT": 0.90,
    "MCMC_SEED": 42,
}
