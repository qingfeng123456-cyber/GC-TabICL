# Processed data attribution and license

The processed learning arrays in `data/` are adaptations of the OpenCS2 replay dataset published by **blanchon** at <https://huggingface.co/datasets/blanchon/opencs2_dataset_demo>, pinned to revision `912a363098feec3f220e41d68634d2f71676c28b`.

The upstream dataset is distributed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**. The processed arrays are distributed under the same license: <https://creativecommons.org/licenses/by/4.0/>. The legal code is available at <https://creativecommons.org/licenses/by/4.0/legalcode>.

Changes made for this release include selecting the documented series, reconciling attacking rounds, aligning players, computing causal completed-round histories, defining sampled regional first-arrival targets, and extracting the observed two-second motion prefix. Source URLs, per-recording fingerprints, data partitions and processing checks accompany the arrays in `provenance/` and the data cards.

Attribution should retain the upstream dataset creator and source, indicate these transformations, and acknowledge the GC-TabICL task preparation. The MIT software license does not replace the data license or the original terms of upstream software and pretrained weights. Original raw DEM files are not bundled here.
