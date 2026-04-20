# Paper Data: Prior-Aware Masked Diffusion for Protein Language Model Steering

## Folder structure

paper_data/
├── README.md                 <- this file
├── reference/                <- UniRef50 L17 statistics (for Mahal)
├── steering_vectors/         <- steering vectors used (sol/therm/trpb/gfp)
├── random_aa/                <- Random amino acid baseline (motivation)
├── goldilocks_sol_easy/      <- Cross-setting Mahal-pLDDT analysis
├── sol_easy/
│   ├── generated/            <- 500-seq generations per setting (.csv)
│   ├── oracle/               <- oracle-scored CSVs
│   ├── esmfold/              <- per-seq pLDDT (merged ESMFold output)
│   └── proxy/                <- per-seq Mahal, GLP resid, ppl 650M, ppl 3B
├── sol_hard/         (same structure)
├── therm_easy/       (same structure)
├── therm_hard/       (same structure)
├── trpb/             (fitness benchmark; same structure)
└── gfp/              (fitness benchmark; same structure)

## Settings

Each task × 5 settings:
- L17_a1: weak L17 single-layer steering (baseline near-natural)
- L17_a10: strong L17 single-layer (structure collapse)
- allL_a2: moderate all-layer steering (MAIN filter target)
- allL_a3: strong all-layer (oracle saturated)
- allL_a2_L17GLP_u0.5: modifier (GLP projection during generation)

## Proxies (per-seq)

- mahal: Mahalanobis² at L17 (using rep_statistics.pt)
- glp_resid: GLP denoising residual at u=0.15
- ppl_650m: pseudo-perplexity from ESM-2 650M (15 random positions)
- ppl_3b: pseudo-perplexity from ESM-2 3B (only main settings for speed)

## Metrics

- plddt: ESMFold predicted local distance difference test
- oracle: task-specific predictor output (sol=sol_prob, therm=pred_tm, trpb=fitness lookup, gfp=predictor)

