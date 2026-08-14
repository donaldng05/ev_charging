Notebooks are for exploration only. They are not the source of truth.

Pipelines, configs, models, and experiments live in `src/chargeopt` and `configs/`. If a notebook result matters, promote it into a tested module.

- `acn_data_tut1.ipynb` — ACN-Sim Lesson 1 onboarding. Not the ingest path.
- `02_acn_sessions_eda.ipynb` — ACN-Data `DataClient` EDA. Ingest lives in `chargeopt data pull`.

## Kernel

Use this project's `.venv`, not a global Python. After `uv sync`:

1. In Cursor: **Select Kernel** → **Python Environments** → `.venv`
2. Restart the notebook

Do not `pip install` or clone libraries inside notebook cells. Canonical session snapshots are CSV via `uv run chargeopt data pull`.
