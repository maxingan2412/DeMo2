# Repository Guidelines

This guide summarizes how to work inside the DeMo multi-modal ReID codebase and keep changes consistent.

## Project Structure & Module Organization
- Entry points: `train_net.py` (training), `test_net.py` (eval + checkpoint load), `test_sdtps.py` (SDTPS sanity check).
- Configs live in `configs/<dataset>/`; set `DATASETS.ROOT_DIR` and `OUTPUT_DIR` before runs.
- Core code: `modeling/`, `layers/`, `engine/processor.py`, `solver/`, `data/datasets/` for samplers/loaders.
- Utilities and assets: `utils/` (logging/meters), `visualize/` + `results/` (plots), `docs/` (SDTPS notes), `tools/` (helpers).

## Build, Test, and Development Commands
```bash
pip install -r requirements.txt                        # Python 3.8+, Torch 1.13 recommended
python train_net.py --config_file configs/RGBNT201/DeMo.yml OUTPUT_DIR=outputs/rgbnt201
python test_net.py --config_file configs/RGBNT201/DeMo.yml  # set checkpoint path; default is placeholder
python test_sdtps.py                                   # SDTPS integration/shape check
```
- For new datasets, copy a YAML and adjust `DATASETS`, `SOLVER`, `MODEL`.
- Use `MODEL.DEVICE_ID` or `CUDA_VISIBLE_DEVICES` for GPU selection; set `MODEL.DIST_TRAIN=True` for distributed runs.

## Coding Style & Naming Conventions
- PEP8 with 4-space indents; keep functions/variables in `snake_case` and classes in `CamelCase`.
- Keep config keys uppercase to match YAML/`cfg` access.
- Prefer small, pure functions; log shapes/metrics with `utils.logger.setup_logger`.
- Put dataset paths/toggles in configs, not hard-coded.

## Testing Guidelines
- Quick check: `python test_sdtps.py` when touching SDTPS or model heads.
- Full eval: `python test_net.py --config_file <cfg> --opts TEST.WEIGHT <path_to_pth>` after setting the checkpoint path; use a weight that matches the config’s dataset.
- Keep `OUTPUT_DIR` pointed to a writable experiment folder; archive `*.pth` and logs, not datasets, in PRs.

## Commit & Pull Request Guidelines
- Use short, imperative commit messages (e.g., `Update requirements.txt`, `Add SDTPS check`).
- PRs: brief summary, config used, dataset split, seed, and key metrics (mAP/Rank-1); add plots from `results/` or logs when behavior changes.
- Call out config changes that alter defaults and GPU memory/batch-size expectations.

## Security & Configuration Tips
- Do not commit datasets or large checkpoints; share download links as in `README.md`.
- Double-check YAML paths (`ROOT_DIR`, `OUTPUT_DIR`, pretrained weights) before running to avoid writing outside the repo.
