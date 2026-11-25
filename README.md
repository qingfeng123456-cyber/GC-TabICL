# GC-TabICL

**Grouped conditional TabICLv2 adaptation for opening-responsibility forecasting in professional Counter-Strike 2.**

GC-TabICL is the study's task adaptation of the public TabICLv2 foundation model. A shared five-player choice interface predicts the first subsequent entrant into each Dust2 staging region from the friendly team's first two seconds of movement and completed-lineup history. A separate CatBoost occurrence head supplies the probability of regional entry. The implementation retains the method identifier `foundation_hurdle`.

The upstream TabICLv2 architecture, synthetic pretraining, and checkpoint are credited to their authors. This repository contains the downstream task interface, grouped adaptation, comparisons, evaluation protocol, and figure-reconstruction tools. It does not contain a newly pretrained foundation model.

## Contents

| Path | Purpose |
| --- | --- |
| `main.py` | Train, select on development data, lock artifacts, assess reserved matches, and package the returned results. |
| `foundation.py`, `features.py` | Grouped conditional adaptation and observed candidate features. |
| Other root Python modules | Rules, linear model, CatBoost, neural comparators, data and integrity checks. |
| `protocol.json` | Fixed observation interface, methods, seeds, training budgets, ablations and training-size subsets. |
| `data/` | Complete processed learning arrays required by the registered experiment. |
| `pretrained/` | Pinned public TabICLv2 checkpoint, upstream model card and source manifest. |
| `vendor/` | Model/dependency source and third-party license notices. |
| `provenance/` | Recording sources, SHA-256 inventories, cohort-selection record and data-quality audits. |
| `RELEASE_CONTENTS.json` | File hashes for this prepared release. |

## Data

The supplied experiment uses **1,584 development rounds** (59 training series / 1,244 rounds; 15 validation series / 340 rounds) and **413 reserved rounds from 20 additional series**. The files are physically separate:

- `data/development.npz` contains training and validation examples only.
- `data/external_motion.npz` contains the reserved assessment examples only.

These are the complete processed inputs and targets for this experiment. Raw DEM recordings are represented by canonical source URLs and fingerprints, rather than copied into this repository. See `data/DATA_CARD.md` and `DATA_INVENTORY.json` for array definitions and exact file hashes.

The original source ledger describes 94 earlier series, including an older contextual evaluation cohort. The runnable release uses its 74 development series plus 20 newly reserved series. Thus the 94 series in the released experiment are a different composition from the earlier 94-series source inventory. The combined acquisition inventory contains 114 distinct recordings and 55,697,979,820 raw bytes; raw acquisition volume is separate from the compressed learning-array size.

## Environment and execution

Use Python 3.11 or newer (development checks used Python 3.12). Install a CUDA-enabled PyTorch build that supports the target GPU, then install the requirements in that environment:

```bash
python -m pip install -r requirements.txt
python main.py
```

The formal run uses CUDA and checks actual device operations, read-only input integrity and repeated predictions before accepting results. It does not install or silently replace PyTorch. The bundled checkpoint and vendored model source allow training and inference without runtime model downloads.

For a development-only installation and flow check:

```bash
python main.py --smoke
```

The smoke run uses a smaller development subset and does not read reserved assessment arrays. It is a program check, not a performance result. On Windows, `Run-Windows.cmd` runs the same main entry in the active Python environment.

The checkpoint is prepared for Git LFS storage through `.gitattributes`. A Git clone should retrieve its LFS objects before execution:

```bash
git lfs install
git lfs pull
```

The original checkpoint can also be obtained from the pinned public `jingang/TabICL` source identified in `pretrained/MANIFEST.json`; verify its SHA-256 against that manifest.

For a ZIP download or an anonymous review copy that contains a Git LFS pointer instead of the binary checkpoint, retrieve the exact upstream file before running experiments:

```bash
python download_pretrained.py
python download_pretrained.py --verify-only
```

This setup command downloads only from the pinned upstream Hugging Face model source and verifies the full file size and SHA-256. It does not contact the project authors or change the model.

## Registered experiment

One complete run covers ten predictors: constant velocity, history prior, linear choice, CatBoost, flat prefix, goal-conditioned GRU, competitive graph, set Transformer, task-structured Transformer, and grouped foundation adaptation. Stochastic methods use seeds 2026, 2027 and 2028. The runner also includes the frozen foundation reference, three retrained input removals, and nested 15/30/59-training-series comparisons for the selected primary, GRU and task Transformer.

The primary score averages unique-outcome conditional identity log loss within each series, then gives series and fitted seeds equal weight. Model selection uses the development partition. The runner locks weights, preprocessing, source fingerprints and validation forecasts before opening the separate reserved data file. Shared occurrence heads keep conditional candidate-choice comparisons aligned within each data condition.

Upstream pretraining, model fitting, hyperparameter search and occurrence-head costs are recorded separately. Original model identifiers and the numerical protocol are preserved in the source.

## Results, continuation and figures

Run the same command in the same unmodified directory to continue an interrupted run. Completed forecasts are verified and reused. The formal result directory is `results/formal/`; the return archive is `results/CS2_Application_RESULTS_RETURN.zip`. Failures produce a separate return archive with diagnostics.

Results contain model selections, configuration and environment records, raw forecasts, compatible target masks, per-series scores, and figures with numerical evidence. To redraw returned figures on CPU:

```bash
python replot.py --logs results/formal --out results/redrawn
```

Replotting uses saved arrays and does not require training, original DEMs or a GPU.

## Provenance and licensing

The replay source is `blanchon/opencs2_dataset_demo`, pinned to revision `912a363098feec3f220e41d68634d2f71676c28b`. Original match and event pages and raw-recording hashes are preserved in the source ledgers. Data-quality records identify accepted rounds, compatible arrivals, observation deadlines and prior-only history reconstruction.

The original software and its documentation are released under the MIT license in `LICENSE`. Processed learning arrays are released under CC BY 4.0 with upstream attribution and transformation notes in `DATA_LICENSE.md`.

Bundled third-party code and pretrained weights retain their upstream terms. See `vendor/THIRD_PARTY_NOTICES.md`, `vendor/tabicl_raw/LICENSE`, `vendor/einops/LICENSE`, and the upstream model card.

## Release status

This release provides the code, processed data, protocol and reconstruction tools. The reserved-match assessment remains a separate execution stage; its model scores are produced by executing the registered experiment.
