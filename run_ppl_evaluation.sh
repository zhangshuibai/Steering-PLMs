#!/bin/bash
# =============================================================================
# Pseudo-Perplexity (pPPL) Evaluation Script
# =============================================================================
#
# 功能: 使用更大的 ESM2-3B 模型评估蛋白质序列的"自然度"(naturalness)
#
# 原理:
#   pPPL = exp(-1/L * Σ log P(x_i | x_\i))
#   对序列的每个位置 mask，用 ESM2-3B 预测该位置的真实氨基酸概率
#   pPPL 越低 = 序列越像天然蛋白质
#
# 为什么用 ESM2-3B 而不是 ESM2-650M?
#   生成用的是 650M，如果用同一模型评估存在"自评 bias"
#   用独立的、更强的 3B 模型做"第三方裁判"，更客观
#
# 评估三组序列:
#   1. Reference  — 天然蛋白质序列 (data/sol_easy.csv, 来自 UniRef50)
#   2. Steering   — 溶解度引导生成的序列
#   3. No-Steering — 无引导的 baseline 生成序列
#
# =============================================================================

PYTHON="/volume/demo/xlzhuang/MOE/miniconda3/envs/steering/bin/python"
WORK_DIR="/volume/demo/xlzhuang/MOE/Steering-PLMs"
cd $WORK_DIR

# ---------- 配置 ----------
MODEL="3B"                    # 评估用的 ESM2 模型: 150M / 650M / 3B
GPU_IDS="0 1 4 5"             # 多 GPU 并行 (3B 模型约 6GB/卡，每张卡独立处理一部分序列)
BATCH_MASKS=32                # 每次 forward pass 同时评估多少个 masked position
DEVICE_GEN="cuda:0"           # 生成序列用的单 GPU

# ---------- 输入文件 ----------
REF_CSV="data/sol_easy.csv"                              # 天然序列 (162条, 来自 UniRef50 lysozyme-like)
STEERING_CSV="results/ESM2_gen_steering_sol_easy.csv"    # steering 生成的序列 (100条)
NO_STEER_CSV="results/ESM2_gen_no_steering_sol_easy.csv" # no-steering 生成的序列 (100条)
OUTPUT_CSV="results/ppl_eval_3B_sol_easy.csv"            # 输出: 每条序列的 pPPL

# =============================================================================
# Step 1: 生成缺失的序列文件 (如果不存在)
# =============================================================================
# Steering 序列在之前的 pipeline 中已生成, 这里只补充 no-steering baseline

if [ ! -f "$NO_STEER_CSV" ]; then
    echo "=========================================="
    echo "Step 1: 生成 No-Steering baseline 序列"
    echo "=========================================="
    echo "  使用 ESM2-650M 做 mask-predict 迭代生成 (10轮, 每轮 mask 10%)"
    echo "  不加 steering vector, 作为对照组"
    echo ""
    $PYTHON steering_esm2_generation.py \
        --model "650M" \
        --property "sol" \
        --device "$DEVICE_GEN" \
        --ref_data_path "data/sol_easy.csv" \
        --output_file "$NO_STEER_CSV" \
        --n 100
    echo "  生成完毕: $NO_STEER_CSV"
else
    echo "Step 1: No-Steering 序列已存在, 跳过生成"
fi

# =============================================================================
# Step 2: 用 ESM2-3B 计算三组序列的 pPPL
# =============================================================================
echo ""
echo "=========================================="
echo "Step 2: ESM2-${MODEL} pPPL 评估"
echo "=========================================="
echo "  评估对象:"
echo "    - Reference:    $REF_CSV"
echo "    - Steering:     $STEERING_CSV"
echo "    - No-Steering:  $NO_STEER_CSV"
echo ""
echo "  使用 GPU: $GPU_IDS"
echo "  每条序列: mask 每个位置 → forward pass → 取 log P(真实 token)"
echo "  pPPL = exp(-mean(log_probs))"
echo ""

$PYTHON evaluate_ppl.py \
    --input_csvs "$REF_CSV" "$STEERING_CSV" "$NO_STEER_CSV" \
    --labels "Reference" "Steering" "No-Steering" \
    --model "$MODEL" \
    --gpu_ids $GPU_IDS \
    --batch_masks $BATCH_MASKS \
    --output_csv "$OUTPUT_CSV"

# =============================================================================
# 预期结果 (2025-03-11 实验):
# =============================================================================
# Group                     N    pPPL mean     pPPL med    log(pPPL)
# ------------------------------------------------------------------
# Reference               162       5.4659       4.9269       1.6229
# Steering                100      15.2278      15.3838       2.7176
# No-Steering             100       7.1862       6.2161       1.8702
#
# 结论:
#   - Reference (天然序列) pPPL 最低 → 最自然
#   - No-Steering 略高 → mask-predict 生成有轻微退化
#   - Steering pPPL 高 3 倍 → steering 把激活推到 off-manifold
#     → 验证了 GLP on-manifold projection 的必要性
# =============================================================================
