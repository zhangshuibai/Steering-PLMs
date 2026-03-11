#!/bin/bash
# =============================================================================
# GLP (Generative Latent Prior) On-Manifold Steering Pipeline
# =============================================================================
#
# 目的: 训练 GLP 学习 ESM2-650M Layer 17 的激活分布, 然后用 SDEdit
#       将 steered activations 投影回蛋白质流形, 提高生成序列的自然度
#
# 背景 (来自 single-layer steering 实验):
#   Layer 17 单独 steering: sol=32%, pPPL=7.01 (与 all-layer 效果相同)
#   All-layer steering:     sol=32%, pPPL=15.23 (自然度严重退化)
#   → 单层 steering 已经足够, GLP 的作用是进一步提高自然度
#
# GLP 原理:
#   1. 在大量天然蛋白质上提取 Layer 17 激活, 学习其分布 (flow matching)
#   2. Steering 后将激活加噪到 timestep u, 然后 denoise 回来
#   3. Denoise 过程将 off-manifold 的激活拉回到天然蛋白质的激活流形上
#   4. u 控制投影强度: u=0 不投影, u=1 完全重采样, u=0.5 推荐
#
# 三步流程:
#   Step 1: 从 UniRef50 提取 ESM2-650M Layer 17 token-level 激活
#   Step 2: 训练 GLP denoiser (flow matching)
#   Step 3: Layer 17 steering + GLP 投影, 评估 sol + pPPL
#
# 计算资源: 4× GPU (0,1,2,3), 预计总耗时 ~3-4 小时
# =============================================================================

PYTHON="/volume/demo/xlzhuang/MOE/miniconda3/envs/steering/bin/python"
WORK_DIR="/volume/demo/xlzhuang/MOE/Steering-PLMs"
GLP_DIR="$WORK_DIR/generative_latent_prior"
cd $WORK_DIR

# ---------- 配置 ----------
GPU_IDS="0 1 2 3"
LAYER=17
MAX_SEQS=4000000               # 4M 序列 → ~1B activations
BATCH_SIZE=16                  # 提取时每 GPU 的 batch size
GLP_N_LAYERS=6                 # Denoiser 深度
U=0.5                          # SDEdit 噪声水平

# ---------- 数据路径 ----------
FASTA="data/uniref50/uniref50.fasta.gz"
ACT_DIR="data/esm2_650m_layer17_uniref50"
GLP_RUN_DIR="$GLP_DIR/runs/glp-esm2-650m-layer17-d6"
GLP_CONFIG="$GLP_DIR/configs/train_esm2_650m_layer17.yaml"

# ---------- 评估相关 ----------
REF_DATA="data/sol_easy.csv"
SV_PATH="saved_steering_vectors/650M_sol_steering_vectors.pt"
PREDICTOR="saved_predictors/sol_predictor_final.pt"
OUTPUT_DIR="results/steering_with_glp"

# =============================================================================
# Step 1: 提取 ESM2-650M Layer 17 激活
# =============================================================================
# 输入: UniRef50 FASTA (4M 序列)
# 输出: memmap 格式激活文件 (data_XXXX.npy + rep_statistics.pt)
# 预计: ~1.5 小时 (4× GPU 并行)
# =============================================================================
echo "==========================================="
echo "Step 1: 提取 ESM2-650M Layer $LAYER 激活"
echo "==========================================="
echo "  数据: $FASTA"
echo "  输出: $ACT_DIR"
echo "  GPU: $GPU_IDS"
echo "  序列数: $MAX_SEQS"
echo ""

if [ ! -f "$ACT_DIR/rep_statistics.pt" ]; then
    $PYTHON extract_esm2_activations.py \
        --fasta "$FASTA" \
        --output_dir "$ACT_DIR" \
        --layer $LAYER \
        --max_seqs $MAX_SEQS \
        --batch_size $BATCH_SIZE \
        --gpu_ids $GPU_IDS
    echo "  提取完成!"
else
    echo "  激活文件已存在, 跳过提取"
fi

# =============================================================================
# Step 2: 训练 GLP Denoiser
# =============================================================================
# 输入: memmap 激活文件
# 输出: GLP checkpoint (denoiser weights + normalizer stats)
# 模型: TransformerMLP denoiser
#   d_input=1280 (ESM2-650M hidden dim)
#   d_model=2560, d_mlp=5120, n_layers=6
#   参数量 ~340M
# 训练: AdamW, lr=5e-5, batch=4096, cosine schedule
# 预计: ~1.5 小时 (单 GPU, denoiser 只有 340M 参数)
# =============================================================================
echo ""
echo "==========================================="
echo "Step 2: 训练 GLP Denoiser"
echo "==========================================="
echo "  Config: $GLP_CONFIG"
echo "  输出: $GLP_RUN_DIR"
echo "  架构: d_input=1280, d_model=2560, d_mlp=5120, n_layers=$GLP_N_LAYERS"
echo ""

if [ ! -f "$GLP_RUN_DIR/final.safetensors" ]; then
    cd $GLP_DIR
    CUDA_VISIBLE_DEVICES=0 $PYTHON glp_train.py config=configs/train_esm2_650m_layer17.yaml
    cd $WORK_DIR
    echo "  训练完成!"
else
    echo "  GLP 模型已存在, 跳过训练"
fi

# =============================================================================
# Step 3: Layer 17 Steering + GLP 投影评估
# =============================================================================
# 流程: 加载 ESM2-650M + GLP → Layer 17 steering + SDEdit → 生成 100 条序列
#       → oracle sol eval → ESM2-3B pPPL eval
# SDEdit 参数: u=0.5, 20 denoising steps
# 对比:
#   L17 单层 steering (无 GLP):  sol=32%, pPPL=7.01
#   All-layer steering:          sol=32%, pPPL=15.23
#   L17 + GLP:                   sol=??,  pPPL=?? (预期 pPPL 更低)
# =============================================================================
echo ""
echo "==========================================="
echo "Step 3: Steering + GLP 投影评估"
echo "==========================================="
echo "  GLP: $GLP_RUN_DIR"
echo "  SDEdit: u=$U, 20 steps"
echo "  生成: 100 条序列"
echo ""

$PYTHON steering_with_glp.py \
    --glp_path "$GLP_RUN_DIR" \
    --u $U \
    --num_timesteps 20 \
    --glp_layer $LAYER \
    --gpu_gen "cuda:0" \
    --gpu_ppl $GPU_IDS \
    --n_gen 100 \
    --output_dir "$OUTPUT_DIR" \
    --ref_data "$REF_DATA" \
    --sv_path "$SV_PATH" \
    --predictor_path "$PREDICTOR" \
    --ppl_model "3B" \
    --batch_masks 32

echo ""
echo "==========================================="
echo "Pipeline 完成!"
echo "  激活数据:  $ACT_DIR/"
echo "  GLP 模型:  $GLP_RUN_DIR/"
echo "  评估结果:  $OUTPUT_DIR/"
echo "==========================================="

# =============================================================================
# 预期对比:
#   Method                     | Sol Ratio | pPPL
#   ---------------------------|-----------|------
#   Reference (天然)           |     5.6%  |  5.47
#   No Steering                |    11.0%  |  7.19
#   L17 Single (no GLP)       |    32.0%  |  7.01
#   All-Layer Steering         |    32.0%  | 15.23
#   L17 + GLP (u=0.5)         |      ??%  |  ??   ← 本实验
#
# 预期: sol 保持在 ~30%, pPPL 接近或低于 7.01
# GLP 投影可能会略降低 sol (因为拉回流形), 但 pPPL 应该更低
#
# 实验结果 (待填):
#   TODO
# =============================================================================
