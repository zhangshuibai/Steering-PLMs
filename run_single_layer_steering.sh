#!/bin/bash
# =============================================================================
# Single-Layer Steering Experiment
# =============================================================================
#
# 目的: 找出 ESM2-650M (33层) 哪一层对溶解度 steering 最有效
#
# 实验设计:
#   - 对每一层 (0~32) 单独施加 steering vector (其他层不加)
#   - 每层生成 100 条序列
#   - 用 oracle predictor 评估溶解度 (sol prob, soluble ratio)
#   - 用 ESM2-3B 评估自然度 (pPPL)
#   - 与 all-layer steering 的结果对比
#
# Steering 原理:
#   原始: 所有层都加 steering vector (shape: [33, 1280])
#   本实验: 只在第 i 层加, 其余层置零
#     sv_single[i] = steering_vectors_all[i]
#     sv_single[j] = 0 (j ≠ i)
#
# Baseline 对比 (来自之前实验):
#   All-Layer Steering: sol_ratio ≈ 32%, pPPL ≈ 15.23
#   No Steering:        sol_ratio ≈ 11%, pPPL ≈ 7.19
#   Reference:          sol_ratio ≈ 5.6%, pPPL ≈ 5.47
#
# 预计耗时: ~50 分钟 (生成 ~15min + pPPL ~35min)
# =============================================================================

PYTHON="/volume/demo/xlzhuang/MOE/miniconda3/envs/steering/bin/python"
WORK_DIR="/volume/demo/xlzhuang/MOE/Steering-PLMs"
cd $WORK_DIR

# ---------- 配置 ----------
GPU_GEN="cuda:0"              # 生成 + sol eval 用的 GPU (ESM2-650M ~2.5GB)
GPU_PPL="0 1 4 5"             # pPPL eval 用的多 GPU (ESM2-3B ~6GB/卡)
N_GEN=100                     # 每层生成的序列数
PPL_MODEL="3B"                # pPPL 评估模型
OUTPUT_DIR="results/single_layer_steering"

# ---------- 输入文件 ----------
REF_DATA="data/sol_easy.csv"                                        # 参考序列 (UniRef50)
SV_PATH="saved_steering_vectors/650M_sol_steering_vectors.pt"       # steering vectors (33层)
PREDICTOR="saved_predictors/sol_predictor_final.pt"                 # 溶解度 oracle

# =============================================================================
# 运行实验
# =============================================================================
echo "=========================================="
echo "Single-Layer Steering Experiment"
echo "=========================================="
echo "  ESM2-650M: 33 layers (0~32)"
echo "  每层生成 ${N_GEN} 条序列"
echo "  评估: sol (oracle) + pPPL (ESM2-${PPL_MODEL})"
echo "  GPU: gen=${GPU_GEN}, pPPL=${GPU_PPL}"
echo ""

$PYTHON exp_single_layer_steering.py \
    --gpu_gen "$GPU_GEN" \
    --gpu_ppl $GPU_PPL \
    --n_gen $N_GEN \
    --ppl_model "$PPL_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --ref_data "$REF_DATA" \
    --sv_path "$SV_PATH" \
    --predictor_path "$PREDICTOR" \
    --batch_masks 32

echo ""
echo "=========================================="
echo "实验完成! 结果在: ${OUTPUT_DIR}/"
echo "  summary.csv  - 每层的 sol + pPPL 数据"
echo "  summary.json - 含最佳层的完整记录"
echo "  layer_*.csv  - 每层生成的序列"
echo "=========================================="

# =============================================================================
# All-Layer Steering Baseline (已有结果):
#   sol_ratio = 32.0%, pPPL = 15.23
#
# No-Steering Baseline (已有结果):
#   sol_ratio = 11.0%, pPPL = 7.19
#
# Reference (天然序列):
#   sol_ratio = 5.6%, pPPL = 5.47
#
# 实验结果 (2025-03-11):
#   Layer | Sol Ratio | pPPL mean
#   ------|-----------|----------
#      0  |    28.0%  |   7.23
#      1  |    31.0%  |   7.43
#      2  |    30.0%  |   6.99  ← best pPPL
#      3  |    29.0%  |   7.16
#      4  |    22.0%  |   7.07
#      5  |    26.0%  |   7.13
#      6  |    28.0%  |   7.20
#      7  |    25.0%  |   7.10
#      8  |    22.0%  |   7.25
#      9  |    25.0%  |   7.19
#     10  |    30.0%  |   7.08
#     11  |    26.0%  |   7.13
#     12  |    19.0%  |   7.09
#     13  |    21.0%  |   7.00
#     14  |    23.0%  |   7.24
#     15  |    24.0%  |   7.19
#     16  |    28.0%  |   7.38
#     17  |    32.0%  |   7.01  ← best sol & best trade-off
#     18  |    32.0%  |   7.33
#     19  |    29.0%  |   7.29
#     20  |    22.0%  |   7.20
#     21  |    24.0%  |   7.46
#     22  |    26.0%  |   7.25
#     23  |    24.0%  |   7.19
#     24  |    28.0%  |   7.45
#     25  |    25.0%  |   7.22
#     26  |    31.0%  |   7.32
#     27  |    30.0%  |   7.25
#     28  |    26.0%  |   7.28
#     29  |    27.0%  |   7.33
#     30  |    27.0%  |   7.07
#     31  |    22.0%  |   7.65
#     32  |    24.0%  |   7.53
#
# 最佳层:
#   Best sol layer:      17 (sol=32.0%, pPPL=7.01)
#   Best pPPL layer:      2 (sol=30.0%, pPPL=6.99)
#   Best trade-off:      17 (sol=32.0%, pPPL=7.01)
#
# 关键发现:
#   1. Layer 17 单独 steering 就能达到 all-layer 的 sol_ratio (32%)!
#   2. 但 pPPL 只有 7.01, 远低于 all-layer 的 15.23 → 自然度好很多
#   3. Layer 17-18 是 steering 效果最佳的"甜蜜点" (中间偏后层)
#   4. 所有单层 pPPL 都在 7.0-7.7 范围, 接近 no-steering 的 7.19
#      → 单层 steering 几乎不损害序列自然度
#   5. 相比之下, all-layer steering pPPL=15.23, 自然度严重退化
#      → 多层同时 steering 导致激活偏离蛋白质流形
# =============================================================================
