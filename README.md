# ADS-B Future-Onset Interaction Forecasting

Code, frozen protocols, result artifacts, and manuscript sources for forecasting newly arising terminal-airspace proximity interactions from recent ADS-B observations.

The study uses the preceding two minutes of observed traffic state to forecast future-onset interaction counts and threshold-exceedance probabilities at 30, 120, and 300 seconds. It evaluates date-separated confirmation at KBTP and applies the frozen scheme to KAGC without airport-specific tuning.

## What this repository contains

- `src/airspace_complexity/` — target construction, feature extraction, models, and evaluation utilities.
- `scripts/` — the experiment, diagnostic, and figure-generation entry points used by the paper.
- `configs/` — frozen experiment protocols and source selections.
- `reports/references/` — machine-readable results supporting the paper's tables and figures.
- `tests/` — tests for the retained research pipeline.
- `paper/` — IEEE LaTeX source, bibliography, figures, and the current PDF.
- `docs/` — data access, schema, execution, and timezone documentation.

Internal planning notes, progress reports, reviewer prompts, translation intermediates, Word reading copies, raw ADS-B data, derived scenes, checkpoints, and temporary files are intentionally excluded.

## Scientific scope

The target is an ADS-B-derived geometric screening indicator. It is not a validated measure of controller workload, an official loss of separation, collision probability, accident risk, or universal airspace complexity.

The main methodological components are:

1. A future-onset target that excludes pairs already active at the forecast cutoff.
2. A representation ladder separating aggregate traffic state, observed pair geometry, and constant-velocity CPA features.
3. Source-day-blocked confirmation and zero-shot cross-airport evaluation.
4. A supplementary NB2 analysis for overdispersed interaction counts.

## Data

Raw TartanAviation ADS-B files are not redistributed. Obtain the dataset from its official source under the applicable terms, copy `configs/local_paths.example.yaml` to `configs/local_paths.yaml`, and configure local read-only data paths. Large raw and derived data remain outside Git.

## Environment

The latest experiment package was tested with Python 3.13:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-r03-repro-lock.txt
python -m pip install -e .
```

## Main entry points

```powershell
python scripts/run_r03_future_onset_confirmation.py --help
python scripts/run_r03_kagc_external_validation.py --help
python scripts/run_r03_nb2_review_supplement.py --help
python scripts/build_r03_paper_figures.py --help
```

Run the retained tests with:

```powershell
python -m pytest -q
```

Artifact-integrity tests that require non-redistributed derived ADS-B scenes are skipped when those files are absent.

## Paper

The revised IEEE Signal Processing Letters manuscript is available as [paper/paper.pdf](paper/paper.pdf) (4 pages). Its source is `paper/main.tex`, with bibliography in `paper/references.bib` and generated bibliography in `paper/main.bbl`.

The accompanying [paper/supplement.pdf](paper/supplement.pdf) contains one page of Supplementary Material and one page of Information for Reproducibility. Its source is `paper/supplement.tex`. The manuscript and supplement include clickable links to this repository. Author metadata is still awaiting final confirmation; these files are not a record of journal submission or acceptance.

From the `paper/` directory, compile the manuscript with `pdflatex main`, `bibtex main`, and two further runs of `pdflatex main`; copy the resulting `main.pdf` to `paper.pdf`. Compile the supplement with two runs of `pdflatex supplement`. The existing figure assets are retained for the research record.

## Author

Junwen Li, Sichuan University, Chengdu, China

## License

No open-source license has been assigned. Until a license file is added, reuse requires permission from the author. Third-party datasets and the IEEE template remain subject to their original terms.
