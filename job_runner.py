# Databricks notebook source
# Resolve the repo root from this notebook's workspace path and install the package.
# This works regardless of which user/Repo path the code is cloned to.

import subprocess, sys

_ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
_nb_path = _ctx.notebookPath().get()               # e.g. /Repos/user@co.com/insights_generation/job_runner
_repo_root = "/Workspace" + "/".join(_nb_path.split("/")[:-1])

print(f"Installing package from: {_repo_root}")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", _repo_root])

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from pipeline.config import *
from pipeline.data_prep import *
from pipeline.bocpd import *
from pipeline.mmm_data_prep import *
from pipeline.mmm_fit import *
from pipeline.integration import *
from pipeline.validation import *

def main():
    data_prep()
    bocpd()
    mmm_data_prep()
    mmm_fit()
    integration()

main()
