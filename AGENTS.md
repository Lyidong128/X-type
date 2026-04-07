# AGENTS.md

## Cursor Cloud specific instructions

- **Tech stack**: Python 3.12, with numpy, matplotlib, and scipy as dependencies.
- **Install deps**: `pip install -r requirements.txt` (runs from repo root).
- **Run the app**: `python3 scripts/run_scan.py` — prints "Cloud agent running!" and writes `outputs/test.txt`.
- **No lint/test/build tooling** is configured in this repo. There are no linters, test frameworks, or build steps.
- **No services** are required — the project is a standalone Python script with no databases, servers, or external dependencies.
- The `outputs/` directory is created at runtime by `run_scan.py` and should not be committed.
