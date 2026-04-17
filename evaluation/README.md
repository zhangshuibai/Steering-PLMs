# Evaluation

Evaluation modules for generated protein sequences.

## Structure

```
evaluation/
├── common.py                          # Shared: PropertyPredictor, extract_features, load_esm2_model
├── oracle/                            # Property prediction (trained predictors)
│   ├── evaluate_oracle.py             # Main script for sol/therm evaluation
│   ├── fitness/                       # Shared trainer/evaluator for wet-lab fitness benchmarks
│   ├── solubility/
│   │   ├── sol_predictor_final.pt     # Trained on DeepSol, BCEWithLogitsLoss
│   │   └── sol_predictor_final_config.json
│   ├── trpb/
│   │   └── train_trpb_predictor.py
│   ├── creilov/
│   │   └── train_creilov_predictor.py
│   ├── gfp_sarkisyan/
│   │   └── train_gfp_sarkisyan_predictor.py
│   └── thermostability/
│       ├── therm_predictor_nocdhit.pt # Trained on Meltome, MSELoss
│       └── therm_predictor_nocdhit_config.json
├── ppl/                               # Pseudo-perplexity (naturalness)
│   └── evaluate_ppl.py                # ESM2-3B pPPL, supports multi-GPU
└── esmfold/                           # Structure quality (TODO)
    └── evaluate_esmfold.py            # Placeholder: pLDDT, pTM metrics
```

## Quick Start

```bash
# Oracle: solubility
python evaluation/oracle/evaluate_oracle.py \
    --input_csv results/ESM2_gen_steering_sol_easy.csv \
    --property sol --ref_csv data/sol_easy.csv

# Oracle: thermostability
python evaluation/oracle/evaluate_oracle.py \
    --input_csv results/ESM2_gen_steering_therm_easy.csv \
    --property therm --ref_csv data/therm_easy.csv

# Wet-lab fitness benchmark oracles
python evaluation/oracle/trpb/train_trpb_predictor.py
python evaluation/oracle/creilov/train_creilov_predictor.py
python evaluation/oracle/gfp_sarkisyan/train_gfp_sarkisyan_predictor.py

# pPPL (single GPU)
python evaluation/ppl/evaluate_ppl.py \
    --input_csvs results/ESM2_gen_steering_sol_easy.csv --model 3B

# pPPL (multi-GPU)
python evaluation/ppl/evaluate_ppl.py \
    --input_csvs results/ESM2_gen_steering_sol_easy.csv --model 3B --gpu_ids 0 1 2 3
```

## Programmatic Usage

```python
from evaluation.common import PropertyPredictor, load_predictor, extract_features, evaluate_sol, load_esm2_model
from evaluation.ppl.evaluate_ppl import compute_pseudo_perplexity, compute_pseudo_perplexity_multi_gpu
```
