# Processed data card

The release contains 1,997 attacking-round examples from Dust2. `development.npz` contains 1,244 training rounds from 59 series and 340 validation rounds from 15 series. `external_motion.npz` contains 413 reserved rounds from 20 disjoint series. Complete series, original round UIDs and source-row identifiers preserve the grouping used by evaluation.

| Array | Shape | Meaning |
| --- | --- | --- |
| `x` | N x 5 x 30 | Numerical state and completed-round history in stable friendly-player order. Column names are in `feature_names`. |
| `prefix` | N x 5 x 5 x 11 | Friendly observations at 0, 0.5, 1, 1.5 and 2 seconds after freeze end, using as-of native ticks. |
| `mask` | N x 2 | Compatible first-arriver player bits for regions A/B; zero represents no entry. |
| `risk` | N x 2 | Whether the regional event remains unobserved at the prediction cutoff. |
| `match`, `uid`, `round` | N | Source series and round correspondence. |
| `roster` | N x 5 | Stable friendly-player identities. These identifiers are not candidate features. |
| `source_rows` | N | Original row identifiers, disjoint between development and reserved data. |
| `prefix_tick`, `prefix_native`, `prefix_active` | N x 5 | Selected native ticks and elapsed/active clock values. |
| `centers` | 2 x 2 | Fixed regional horizontal reference centers. |
| `feature_names`, `prefix_names`, `prefix_grid` | metadata | Exact feature/channel definitions and sample grid. |

For `mask`, player j corresponds to bit `1 << j` for zero-based j. Multiple first arrivals in the same sampled frame retain all compatible bits. The target is the first sampled alive entrant within 25 active seconds or the earlier round end. The primary identity score uses unique at-risk regional outcomes; compatible ties and no-entry outcomes also contribute to the complete regional score.

The 11 prefix channels encode absolute position divided by 4,000, displacement from the first prefix sample divided by 1,000, velocity divided by 250, alive status, and sample age divided by 0.5 seconds. Velocities use the actual difference between selected native timestamps. Field names within each file are authoritative.

The physical development file contains no future-trajectory auxiliary tensors. Inference feature construction reads the observed state/prefix and fixed regional geometry. Historical channels use earlier completed accepted rounds of the same lineup within a series. Targets and identifying metadata are retained for grouping and scoring.

`DATA_INVENTORY.json` at the repository root supplies the exact dataset byte counts, SHA-256 values, split sizes and source-row inventory. The accompanying `provenance/` records link these processed examples to the pinned public replay source and data-quality checks. Raw DEMs and trajectory-extraction caches are not bundled.
