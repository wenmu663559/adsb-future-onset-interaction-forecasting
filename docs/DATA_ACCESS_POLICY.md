# Data access and redistribution

The exact derived scenes and normalized reports used in this paper are packaged in `data/reproduction/paper-data-v1.zip`. That directory contains attribution, upstream licensing sources, hashes and instructions. This research subset is not the full TartanAviation corpus.

For an audit beginning with the upstream raw files, obtain the complete corpus from https://theairlab.org/tartanaviation/ and configure the ignored `configs/local_paths.yaml` using the example file. Keep upstream originals read-only and generate outputs elsewhere.

Preserve the historical splits and results. Write reproduced evaluations under ignored `outputs/` paths and compare their metrics with `reports/references/`. The package preserves original scene bytes and source identifiers; publication does not create new independent evaluation data.
