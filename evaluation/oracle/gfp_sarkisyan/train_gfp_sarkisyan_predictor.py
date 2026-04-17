import os
import sys


PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
PROJECT_ROOT = os.path.abspath(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.oracle.fitness.train_fitness_predictor import main


DEFAULTS = {
    "dataset_name": "gfp_sarkisyan",
    "processed_csv": "data/benchmarks/processed/gfp_sarkisyan/gfp_sarkisyan_processed.csv",
    "save_path": "evaluation/oracle/gfp_sarkisyan/gfp_sarkisyan_predictor_final.pt",
    "features_dir": "saved_predictors/gfp_sarkisyan_features",
    "target_transform": "zscore",
    "top_k_fraction": 0.05,
}


if __name__ == "__main__":
    main(defaults=DEFAULTS)
