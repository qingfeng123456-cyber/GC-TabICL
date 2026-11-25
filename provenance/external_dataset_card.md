# Newly reserved professional-competition data

All 20 fixed sources have been downloaded, SHA-256 verified, parsed, joined and independently audited. This directory contains data preparation only; no model forecasts or performance comparisons have been run.

- Unique raw DEMs: **20**, **9,491,609,999 bytes** (9.491610 decimal GB; 8.839751 GiB).
- Match dates: April 13–20, 2026; development ended April 10.
- Map: Dust2. Patch counts: 14141: 20.
- Accepted attacking-side rounds: **413**; regional outcomes: **826**.
- Official-round accounting: 418 official rounds, 5 not retained: 4 without usable purchase-before-start state and 1 opening clock/motion conflict. Exact official timestamps and matching extraction/clock records are preserved in `external_dataset_audit.json`.
- Outcome inventory: 292 unique first arrivals, 5 sampled ties, 529 no-arrival outcomes.
- Snapshot-stage filtering: 3 nonofficial/restarted parsed cases and 1 opening clock/motion conflicts. Native parser-span and state-extraction rejection records are tabulated separately in the audit.
- The 20 match IDs, 20 raw SHA-256 values and accepted round IDs are disjoint from the original 94-series cohort.
- Combined with the original cohort: **114 unique recordings**, **55,697,979,820 raw bytes**. This is acquisition volume; the development, contextual and reserved evaluation roles remain distinct.
- Independent reconstruction of all prior-only history channels: maximum absolute error 8.34e-08; each current round is added only after its feature check.
- The acquisition uses the same pinned public dataset source and overlaps the historical-test calendar period. It supplies a new match-level replication cohort.

| Professional circuit | Series | Raw bytes | Accepted rounds |
| --- | ---: | ---: | ---: |
| CCT 2026 Challengers South America Series 1 | 1 | 434,073,545 | 18 |
| CCT Season 3 North America Series 4 | 2 | 812,454,925 | 43 |
| Dust2.us Eagle Masters Series 7 | 4 | 2,181,316,339 | 93 |
| IEM Rio 2026 | 3 | 1,700,991,840 | 68 |
| NODWIN Clutch Series 7 | 10 | 4,362,773,350 | 191 |

Source revision: `912a363098feec3f220e41d68634d2f71676c28b`.
Snapshot SHA-256: `75e0260e374796c903ed9ad1ff05341f250c1be100ad52f1dc7ca497c8e0d425`.

The 30-feature snapshot includes purchase state, operational regional first-entry labels and history from earlier accepted rounds with the same lineup. The native as-of two-second movement prefix is included in `../data/external_motion.npz`; its source-extraction and output fingerprints are recorded in `external_motion_card.json`.

The public source ledger contains canonical URLs and upstream event provenance. This project has not yet published a data/code repository or assigned a new redistribution license.
