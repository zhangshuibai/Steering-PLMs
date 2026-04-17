# Repository Guidelines

## Project Structure & Module Organization
Core steering logic lives in `module/` (`steerable_esm2.py`, `steerable_esm3.py`, `steerable_prollama.py`) and shared helpers live in `utils/`. Use `scripts/extraction/` for steering-vector and activation extraction, `scripts/generation/` for sequence generation, `scripts/experiments/` for reproducibility studies, and `scripts/pipelines/` for longer end-to-end runs. Keep evaluation code in `evaluation/` (`oracle/`, `ppl/`, `esmfold/`). The GLP work is isolated in `generative_latent_prior/` with its own `pyproject.toml` and `environment.yaml`. Data, checkpoints, and outputs belong in `data/`, `saved_*`, `results/`, or `new-results/`, not alongside source files.

## Build, Test, and Development Commands
Set up the main environment from the root `README.md`, then run scripts directly from the repo root:

- `python scripts/extraction/extract_esm2_steering_vec.py --model 650M ...` builds steering vectors in `saved_steering_vectors/`.
- `python scripts/generation/steering_esm2_generation.py --model 650M --property sol --ref_data_path data/sol_easy.csv --output_file results/out.csv --steering --n 100` generates steered sequences.
- `python evaluation/oracle/evaluate_oracle.py --input_csv results/out.csv --property sol --ref_csv data/sol_easy.csv` scores generated sequences.
- `python evaluation/ppl/evaluate_ppl.py --input_csvs results/out.csv --model 3B --gpu_ids 0 1` computes pPPL.
- `bash run_all_experiments.sh` reruns the non-GLP experiment suite.
- `conda env create -f generative_latent_prior/environment.yaml` prepares the separate GLP environment.

## Coding Style & Naming Conventions
This is a Python-first repo; follow existing 4-space indentation, snake_case names, and small script-oriented modules. Put reusable model logic in `module/` or `utils/`, not inside one-off experiment drivers. Match the current CLI style: explicit flags, descriptive output paths, and minimal hidden defaults. There is no root formatter or linter config, so preserve surrounding import order and docstring/comment density.

## Testing Guidelines
There is no dedicated `tests/` package. Validate changes with targeted smoke runs on the script you touched, then run the downstream evaluator that proves behavior. For generation changes, prefer a small run such as `--n 2` before launching large GPU jobs. Record the exact output path you validated. Do not commit bulky generated CSVs, logs, or model weights unless they are intentional artifacts for the change.

## Commit & Pull Request Guidelines
Recent history uses short imperative subjects such as `Add ...`, `Update ...`, and `Reorganize ...`. Follow that pattern and keep each commit scoped to one change. In PRs, state the dataset/model path affected, list the commands you ran, note GPU or conda-env assumptions, and summarize any metric changes. If you touch `scripts/pipelines/*.sh`, call out any edited hard-coded values such as `PYTHON`, `WORK_DIR`, `DEVICE`, or `GPU_IDS`.
