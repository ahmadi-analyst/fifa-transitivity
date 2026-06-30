# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Pipeline

Scripts must be run in order; each step depends on the previous:

```bash
pip install -r requirements.txt

python scripts/01_download_data.py    # ~3 MB download from GitHub; skips files already present
python scripts/02_process_data.py     # builds data/processed/ from raw CSVs
python scripts/03_analysis.py         # prints statistical results to stdout
python scripts/04_visualizations.py   # writes figures/ as PNG files
```

All scripts are idempotent. There are no tests; correctness is verified by inspecting printed output from `03_analysis.py` and the saved figures.

## Architecture

The project is a linear four-script data pipeline with no shared modules — each script is standalone and uses only stdlib + the packages in `requirements.txt`.

**Data flow:**

```
GitHub CSVs
  → data/raw/          (01_download_data.py)
  → data/processed/    (02_process_data.py)
  → stdout + figures/  (03_analysis.py, 04_visualizations.py)
```

**Key data structures built in `02_process_data.py`:**

- `group_stage_matches.csv` — 372 group-stage matches (1994–2022), one row per match, with `home_pts`/`away_pts` encoding wins as 1.0, draws as 0.5, losses as 0.0.
- `teams_with_rankings.csv` — one row per (tournament × team) with `pre_tournament_rank` pulled from the closest FIFA ranking snapshot before the tournament start date.
- `transitivity_triples.csv` — all ordered (A, B, C) permutations within each group. Rows where `is_transitivity_triple == 1` are the applicable cases (A strictly beat B, B strictly beat C). `transitive` is 1 if A also strictly beat C, else 0.

**Key design decisions to be aware of:**

- `rank_gap_ac = rank_A − rank_C`. Negative values mean A is ranked higher (better). The logistic regression shows OR ≈ 0.968 per unit — transitivity is more likely when A is strongly favored by ranking.
- FIFA ranking name → jfjelstul dataset name mismatches are resolved via `TEAM_NAME_MAP` in `02_process_data.py`. Adding tournaments beyond 2022 will require verifying this map.
- `rank_gap_ac` equals `rank_gap_ab + rank_gap_bc` exactly (perfect collinearity), so the multivariate logit in `03_analysis.py` uses `rank_gap_ab + rank_gap_bc` as components instead.
- Figure 4's network diagram (`04_visualizations.py`) hard-codes 2022 Group C as the illustrative group, with a fallback to 2018 Group F.
