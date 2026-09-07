# Data Access Policy

## Official Raw Data

Path: configured locally in `configs/local_paths.yaml`

Access mode: **READ ONLY**

## Prohibited Actions

- Modify, rename, move, delete, overwrite, or re-encode raw data.
- Generate outputs inside the raw directory.
- Commit raw data to Git or copy it into the project repository.

## Allowed Actions in R00

- Directory existence and read-permission checks.
- Top-level inventory, recursive file count, directory size estimate, and sample filenames.

## Derived Data

Derived data belongs under `data/interim` or `data/processed_research`, or in an explicitly configured external workspace.

## Traceability Requirement

Future derived records must retain source file identifiers and source row references.
