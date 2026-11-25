# Bundled upstream components

`tabicl_raw` contains the model-computation subset of **TabICL 2.2.0**
(https://github.com/soda-inria/tabicl), copyright 2025 Soda team @ Inria,
under BSD-3-Clause. Its original `_model` and `_torch_devices.py` files are
preserved; package initializers are minimal so unused download/estimator
wrappers are not imported. Full terms are in `tabicl_raw/LICENSE`.

`einops` is **einops 0.8.1**, copyright 2018 Alex Rogozhnikov,
under the MIT license. Full terms are in `einops/LICENSE`.

The separate `pretrained/` directory contains the public TabICLv2 classifier
checkpoint, its source model card and revision/hash manifest. The checkpoint
is distributed under BSD-3-Clause according to that model card. The original
pretraining was performed by the upstream authors. This study adapts the
checkpoint to grouped, five-candidate CS2 responsibility prediction.

PyTorch, NumPy, CatBoost, Matplotlib, psutil and tqdm remain separately
installed environment dependencies; their binary distributions are not
vendored or modified by this package.
