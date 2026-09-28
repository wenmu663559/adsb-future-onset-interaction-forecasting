# Paper reproduction dataset

`paper-data-v1.zip` is a real data archive committed to this repository, not a Git LFS pointer or a download stub. It is 9,719,465 bytes (9.27 MiB) and contains six JSONL files: the exact derived scenes and normalized observation reports for each dataset below. It does not mirror the complete upstream raw dataset, images or audio.

| Dataset | Airport | Scenes | Role |
|---|---|---:|---|
| Development | KBTP | 6,008 | 3,318 train / 1,247 validation / 1,443 retrospective test |
| Confirmation | KBTP | 2,605 | Eight source groups |
| External | KAGC | 1,268 | Ten source groups; no local fitting |

The archive preserves all file bytes, source identifiers, histories, labels and splits. `manifest.json` gives archive/member SHA-256 values, sizes, paths and split counts. Scene checksums match `configs/r03_nb2_review_protocol.json`. Each line of a scene file describes an airport, scene ID, cutoff time, observed history, split, targets and source references. Report files contain normalized observations needed for target reconstruction. See `docs/DATA_DICTIONARY.md` for the upstream field definitions and the source/configuration files for construction rules. These are observed proximity labels, not validated safety or workload outcomes.

## Attribution and terms

Derived from TartanAviation by Jay Patrikar, Joao P. A. Dantas, Brady Moon, Milad Hamidi, Sourish Ghosh, Nikhil Keetha, Ian Higgins, Atharva Chandak, Takashi Yoneyama and Sebastian Scherer (CMU AirLab and collaborators).

- Dataset site: https://theairlab.org/tartanaviation/
- Dataset/code archive: https://doi.org/10.5281/zenodo.14699102
- Data paper: https://doi.org/10.1038/s41597-025-04775-6
- The cited Zenodo record declares CC BY 4.0: https://creativecommons.org/licenses/by/4.0/
- Upstream collection/download software has its own BSD-3-Clause notice: https://github.com/castacks/TartanAviation/blob/main/LICENSE

Changes in this research subset: selection of airport observation windows, deterministic cleaning and unit conversion, construction of causal histories and cutoff-initialized interaction-onset labels, and fixed research splits. Preserve attribution and indicate further modifications when reusing the packaged data. No upstream-author endorsement is implied. This data attribution does not assign a new license to the project's code.

## Extract and verify

Run from the repository root, after installing `requirements-r03-repro-lock.txt` and the package (`python -m pip install -e .`).

```powershell
python -m pytest tests/test_packaged_data.py -q
python -m zipfile -e data/reproduction/paper-data-v1.zip .
```

Extraction creates only the three `outputs/experiments/...` directories listed in the manifest. Do not extract over files you have independently modified. The archive test checks hashes, membership, row counts, airports, unique scene IDs, disjoint scene IDs between datasets and the development split counts.

## Recompute the primary metrics

These commands write new results without overwriting historical references. The paths work on Windows, Linux and macOS; only the environment-activation syntax differs.

```text
python scripts/run_r03_future_onset_confirmation.py --development-root outputs/experiments/r03_future_onset_development/20260903T040754Z-online-complexity-pilot --confirmation-root outputs/experiments/r03_future_onset_confirmation/20260903T075352Z-online-complexity-pilot --confirmation-protocol configs/reproduction_kbtp_protocol.json --output outputs/reproduction/kbtp.json
python scripts/run_r03_kagc_external_validation.py --development-root outputs/experiments/r03_future_onset_development/20260903T040754Z-online-complexity-pilot --kagc-root outputs/experiments/r03_kagc_external_validation/20260903T080631Z-online-complexity-pilot --external-protocol configs/reproduction_kagc_protocol.json --output outputs/reproduction/kagc.json
python scripts/run_r03_nb2_review_supplement.py --output outputs/reproduction/nb2.json
```

Compare numerical `horizons` in the first two outputs with the corresponding historical confirmation/external JSON files under `reports/references/`. For NB2 compare `evaluations`; environment and run metadata can differ. Numerical tolerance used for this package audit is 1e-9.

The commands above were executed from the extracted archive in the recorded local environment. All 273 KBTP, 339 KAGC and 2,255 NB2 numerical values compared with the historical records matched exactly (maximum absolute difference 0). See `verification.json`. This checks recomputation from distributed derived inputs, not installation in a newly provisioned environment or a complete repeat of upstream raw ingestion.

The separate KBTP and KAGC reproduction protocols update dependency checksums for the distributed snapshot, including portable path replacements. `.gitattributes` keeps JSON line endings at LF across platforms to preserve these checksums. The KAGC configuration also records a checksum change for the currently distributed descriptive summary. The original protocol expects unavailable earlier summary bytes. This known provenance limitation is recorded inside `configs/reproduction_kagc_protocol.json`; the original protocol/results remain unchanged. No training inputs, model parameters, thresholds or source-group choices are changed by that separate reproduction configuration.

This archive supports recomputation from the paper's derived data, not a new independent experiment. It also contains normalized observations for target reconstruction. Repeating extraction and parsing from the full upstream raw corpus still requires downloading that corpus separately.

## Figures and validation

`python scripts/build_r03_paper_figures.py` recreates its figures from the retained result JSON files in ignored `outputs/figures/`. Pre-rendered figures unused by the current manuscript are not duplicated in Git. Run `python -m pytest -q` for the retained test suite. A legacy pre-onset dataset test may skip if its historical, non-packaged artifact is absent; the packaged-data test does not skip.
