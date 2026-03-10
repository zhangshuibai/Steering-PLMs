#!/bin/bash
# ============================================================
# Solubility Steering 实验全流程 (ESM2-650M)
# Paper: "Steering Protein Language Models" (ICML'25)
#
# 流程:
#   Step 1: 提取 steering vectors (对比高/低溶解度序列的hidden representations)
#   Step 2: 使用 steering vectors 引导 ESM2 生成蛋白质序列 (easy + hard)
#   Step 3: 生成 baseline 序列 (无 steering 对照组)
#   Step 4: 用 oracle predictor 评估所有生成序列的溶解度
#
# 数据说明:
#   sol_filtered.csv: 720条序列, score 0~1, 用于提取 steering vectors
#     - pos: score >= 0.5 (高溶解度)
#     - neg: score <= 0.2 (低溶解度)
#   sol_easy.csv: 162条, score 0.25~0.30 (中低溶解度, 相对容易提升)
#   sol_hard.csv: 198条, score 0.001~0.10 (极低溶解度, 难以提升)
# ============================================================

set -e  # 遇错即停

PYTHON="/volume/demo/xlzhuang/MOE/miniconda3/envs/steering/bin/python"
DEVICE="cuda:7"
N_GEN=100  # 每组生成100条序列

mkdir -p results
mkdir -p saved_steering_vectors

echo "============================================================"
echo "Step 1: 提取 Solubility Steering Vectors"
echo "  输入: data/sol_filtered.csv (pos>=0.5, neg<=0.2)"
echo "  输出: saved_steering_vectors/650M_sol_steering_vectors.pt"
echo "============================================================"

$PYTHON extract_esm2_steering_vec.py \
    --model "650M" \
    --num_data 100 \
    --property "sol" \
    --data_path "data/sol_filtered.csv" \
    --theshold_pos 0.5 \
    --theshold_neg 0.2

echo ""
echo "============================================================"
echo "Step 2: Steering 引导生成 (Easy + Hard)"
echo "  Easy: 参考序列溶解度 0.25~0.30, 目标通过steering提升溶解度"
echo "  Hard: 参考序列溶解度 0.001~0.10, 难度更高"
echo "============================================================"

echo "--- Step 2a: Easy set with steering ---"
$PYTHON steering_esm2_generation.py \
    --model "650M" \
    --property "sol" \
    --device "$DEVICE" \
    --ref_data_path "data/sol_easy.csv" \
    --output_file "results/ESM2_gen_steering_sol_easy.csv" \
    --steering \
    --n $N_GEN

echo "--- Step 2b: Hard set with steering ---"
$PYTHON steering_esm2_generation.py \
    --model "650M" \
    --property "sol" \
    --device "$DEVICE" \
    --ref_data_path "data/sol_hard.csv" \
    --output_file "results/ESM2_gen_steering_sol_hard.csv" \
    --steering \
    --n $N_GEN

echo ""
echo "============================================================"
echo "Step 3: Baseline 生成 (无 Steering 对照组)"
echo "============================================================"

echo "--- Step 3a: Easy set without steering ---"
$PYTHON steering_esm2_generation.py \
    --model "650M" \
    --property "sol" \
    --device "$DEVICE" \
    --ref_data_path "data/sol_easy.csv" \
    --output_file "results/ESM2_gen_no_steering_sol_easy.csv" \
    --n $N_GEN

echo "--- Step 3b: Hard set without steering ---"
$PYTHON steering_esm2_generation.py \
    --model "650M" \
    --property "sol" \
    --device "$DEVICE" \
    --ref_data_path "data/sol_hard.csv" \
    --output_file "results/ESM2_gen_no_steering_sol_hard.csv" \
    --n $N_GEN

echo ""
echo "============================================================"
echo "Step 4: Oracle 评估 (用训好的 sol predictor 给生成序列打分)"
echo "  Predictor: saved_predictors/sol_predictor_final.pt"
echo "  对比: steering vs no-steering vs 原始参考序列"
echo "============================================================"

PREDICTOR="saved_predictors/sol_predictor_final.pt"

echo "--- Step 4a: 评估 Easy + Steering ---"
$PYTHON evaluate_generated_seqs.py \
    --input_csv "results/ESM2_gen_steering_sol_easy.csv" \
    --predictor_path "$PREDICTOR" \
    --property "sol" \
    --ref_csv "data/sol_easy.csv" \
    --device "$DEVICE"

echo ""
echo "--- Step 4b: 评估 Easy + No Steering ---"
$PYTHON evaluate_generated_seqs.py \
    --input_csv "results/ESM2_gen_no_steering_sol_easy.csv" \
    --predictor_path "$PREDICTOR" \
    --property "sol" \
    --ref_csv "data/sol_easy.csv" \
    --device "$DEVICE"

echo ""
echo "--- Step 4c: 评估 Hard + Steering ---"
$PYTHON evaluate_generated_seqs.py \
    --input_csv "results/ESM2_gen_steering_sol_hard.csv" \
    --predictor_path "$PREDICTOR" \
    --property "sol" \
    --ref_csv "data/sol_hard.csv" \
    --device "$DEVICE"

echo ""
echo "--- Step 4d: 评估 Hard + No Steering ---"
$PYTHON evaluate_generated_seqs.py \
    --input_csv "results/ESM2_gen_no_steering_sol_hard.csv" \
    --predictor_path "$PREDICTOR" \
    --property "sol" \
    --ref_csv "data/sol_hard.csv" \
    --device "$DEVICE"

echo ""
echo "============================================================"
echo "全流程完成！结果文件:"
echo "  results/ESM2_gen_steering_sol_easy_scored.csv"
echo "  results/ESM2_gen_steering_sol_hard_scored.csv"
echo "  results/ESM2_gen_no_steering_sol_easy_scored.csv"
echo "  results/ESM2_gen_no_steering_sol_hard_scored.csv"
echo "============================================================"
