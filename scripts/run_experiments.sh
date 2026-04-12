#!/bin/bash
# ============================================================================
# LoLA + LoLCATs 完整实验脚本
# 模型: Llama-3.1-8B
# 包含: 9 个核心实验 + sparse_budget 消融 + budget 策略消融
# ============================================================================
#
# 使用方式:
#   bash scripts/run_experiments.sh <实验编号>
#   bash scripts/run_experiments.sh all        # 运行全部核心实验
#   bash scripts/run_experiments.sh ablation   # 运行全部消融实验
#
# 环境变量 (可在运行前设置):
#   HF_TOKEN          - HuggingFace token (访问 Llama-3.1-8B 需要)
#   CACHE_DIR         - 模型缓存目录 (默认: /scratch/)
#   CHECKPOINT_DIR    - 训练 checkpoint 保存目录 (默认: ./checkpoints)
#   LM_EVAL_PATH      - lm-evaluation-harness 路径
#   WANDB_ENTITY      - WandB 实体名
#   NO_WANDB          - 设为 1 则禁用 WandB
# ============================================================================

set -e

# ----- 配置 -----
HF_TOKEN="${HF_TOKEN:-}"
CACHE_DIR="${CACHE_DIR:-/scratch/}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"
LM_EVAL_PATH="${LM_EVAL_PATH:-}"
WANDB_ENTITY="${WANDB_ENTITY:-hazy-research}"
NO_WANDB="${NO_WANDB:-0}"

# 模型配置名 (对应 configs/model/ 下的 yaml 文件, 不含 .yaml 后缀)
MODEL_LOLCATS="distill_llama3_1_8b_lk_smd_wsw64_fd64_w01"
MODEL_LOLA="distill_llama3_1_8b_lk_smd_wsw64_fd64_w01_lola"
MODEL_LOLA_SINK="distill_llama3_1_8b_lk_smd_wsw64_fd64_w01_lola_sink"
MODEL_LOLA_BUDGET="distill_llama3_1_8b_lk_smd_wsw64_fd64_w01_lola_budget"
MODEL_LOLA_SINK_BUDGET="distill_llama3_1_8b_lk_smd_wsw64_fd64_w01_lola_sink_budget"
MODEL_BASELINE="base_llama3_1_8b"

# 实验配置名 (对应 configs/experiment/ 下的 yaml 文件)
DISTILL_CONFIG="distill_alpaca_clean_xent0_mse1000_lr1e-2"
FINETUNE_CONFIG="finetune_lora_qkvo_alpaca_clean"
EVAL_CONFIG="eval_alpaca_clean"

# WandB 参数
WANDB_ARGS=""
if [ "$NO_WANDB" = "1" ]; then
    WANDB_ARGS="--no_wandb"
else
    WANDB_ARGS="--wandb_entity ${WANDB_ENTITY}"
fi

HF_ARGS=""
if [ -n "$HF_TOKEN" ]; then
    HF_ARGS="--huggingface_token ${HF_TOKEN}"
fi

# ============================================================================
# 辅助函数
# ============================================================================

print_header() {
    echo ""
    echo "================================================================"
    echo "  $1"
    echo "================================================================"
    echo ""
}

# 运行 lm-evaluation-harness 评估 (6 个 benchmark)
run_lm_eval() {
    local model_type=$1     # lolcats_ckpt / model_config / huggingface
    local model_config=$2   # 模型配置名 (model_type=model_config 时使用)
    local attn_ckpt=$3      # attention MLP checkpoint 路径
    local ft_ckpt=$4        # finetune checkpoint 路径
    local desc=$5           # 实验描述

    print_header "评估: ${desc}"

    local base_args="--model_type ${model_type} ${WANDB_ARGS}"

    if [ "$model_type" = "lolcats_ckpt" ]; then
        base_args="${base_args} --attn_mlp_checkpoint_path ${attn_ckpt}"
        if [ -n "$ft_ckpt" ]; then
            base_args="${base_args} --finetune_checkpoint_path ${ft_ckpt}"
        fi
    elif [ "$model_type" = "model_config" ]; then
        base_args="${base_args} --model_config ${model_config}"
    fi

    # 评估任务列表
    declare -a tasks=("arc-challenge:25" "hellaswag:10" "winogrande:5" "truthfulqa-mc:0" "gsm8k:5" "hendrycksTest:5")

    for task_shot in "${tasks[@]}"; do
        IFS=':' read -r task shots <<< "$task_shot"
        echo "  -> 评估 ${task} (${shots}-shot)..."
        python lm_eval_harness/eval_lm_harness.py \
            ${base_args} \
            --task ${task} \
            --num_shots ${shots} \
            --batch_size 1
    done
}

# ============================================================================
# 实验 1: Baseline (原始 Llama-3.1-8B softmax)
# ============================================================================
exp1_baseline() {
    print_header "实验 1: Baseline - Llama-3.1-8B (softmax)"
    run_lm_eval "model_config" "${MODEL_BASELINE}" "" "" "Baseline softmax"
}

# ============================================================================
# 实验 2: LoLCATs (training-free)
# 直接加载 hazyresearch 预训练权重, 不额外训练
# 需要: hazyresearch 提供的 attn MLP checkpoint + LoRA checkpoint
# ============================================================================
exp2_lolcats_tf() {
    print_header "实验 2: LoLCATs (training-free)"

    # ---- 用户需要设置的路径 ----
    local LOLCATS_ATTN_CKPT="${LOLCATS_ATTN_CKPT:?请设置 LOLCATS_ATTN_CKPT 为 hazyresearch attn MLP checkpoint 路径}"
    local LOLCATS_FT_CKPT="${LOLCATS_FT_CKPT:?请设置 LOLCATS_FT_CKPT 为 hazyresearch LoRA checkpoint 路径}"

    run_lm_eval "lolcats_ckpt" "" "${LOLCATS_ATTN_CKPT}" "${LOLCATS_FT_CKPT}" \
        "LoLCATs (training-free, hazyresearch 权重)"
}

# ============================================================================
# 实验 3: LoLCATs (trainable, 两阶段训练)
# Stage 1: 注意力蒸馏 (feature map MLP)
# Stage 2: LoRA 微调 (q/k/v/o_proj)
# ============================================================================
exp3_lolcats_train() {
    print_header "实验 3: LoLCATs (trainable, 两阶段训练)"

    # Stage 1: 蒸馏
    echo "  [Stage 1] 注意力蒸馏..."
    python distill_llama.py \
        --model_config ${MODEL_LOLCATS} \
        --distill_config ${DISTILL_CONFIG} \
        --finetune_config ${FINETUNE_CONFIG} \
        --eval_config ${EVAL_CONFIG} \
        --checkpoint_dir ${CHECKPOINT_DIR} \
        ${WANDB_ARGS} ${HF_ARGS}

    # 找到最新的 checkpoint
    local LOLCATS_DISTILL_CKPT=$(ls -t ${CHECKPOINT_DIR}/${MODEL_LOLCATS}/*_distill.pt 2>/dev/null | head -1)
    local LOLCATS_FT_CKPT=$(ls -t ${CHECKPOINT_DIR}/${MODEL_LOLCATS}/*_ft.pt 2>/dev/null | head -1)

    echo "  蒸馏 checkpoint: ${LOLCATS_DISTILL_CKPT}"
    echo "  微调 checkpoint: ${LOLCATS_FT_CKPT}"

    # 评估
    run_lm_eval "lolcats_ckpt" "" "${LOLCATS_DISTILL_CKPT}" "${LOLCATS_FT_CKPT}" \
        "LoLCATs (trainable)"
}

# ============================================================================
# 实验 4: LoLA (training-free)
# 加载实验 3 的 LoLCATs 蒸馏+LoRA 权重, 用 LoLA 模型配置
# sparse_factor 使用默认初始值 (sigmoid(10.0)≈1.0)
# 注意: 训推不一致, 仅作为对比基线
# ============================================================================
exp4_lola_tf() {
    print_header "实验 4: LoLA (training-free, 加载 LoLCATs 权重)"

    # 需要实验 3 的 checkpoint
    local LOLCATS_DISTILL_CKPT="${LOLCATS_DISTILL_CKPT:?请设置 LOLCATS_DISTILL_CKPT 为实验3的蒸馏 checkpoint 路径}"
    local LOLCATS_FT_CKPT="${LOLCATS_FT_CKPT:?请设置 LOLCATS_FT_CKPT 为实验3的微调 checkpoint 路径}"

    # LoLA 模型加载 LoLCATs 权重 (strict=False, sparse_factors 用默认值)
    run_lm_eval "lolcats_ckpt" "" "${LOLCATS_DISTILL_CKPT}" "${LOLCATS_FT_CKPT}" \
        "LoLA (training-free, LoLCATs 权重)"
}

# ============================================================================
# 实验 5: LoLA (trainable, 用 LoLCATs 权重做 LoRA 微调)
# 加载实验 3 的蒸馏权重, 仅做 LoRA 微调 (不重新蒸馏)
# ============================================================================
exp5_lola_ft_lolcats() {
    print_header "实验 5: LoLA (trainable, LoLCATs 权重 + LoRA 微调)"

    local LOLCATS_DISTILL_CKPT="${LOLCATS_DISTILL_CKPT:?请设置 LOLCATS_DISTILL_CKPT 为实验3的蒸馏 checkpoint 路径}"

    # 跳过蒸馏, 直接 LoRA 微调
    python distill_llama.py \
        --model_config ${MODEL_LOLA} \
        --distill_config ${DISTILL_CONFIG} \
        --load_distill_checkpoint ${LOLCATS_DISTILL_CKPT} \
        --finetune_config ${FINETUNE_CONFIG} \
        --eval_config ${EVAL_CONFIG} \
        --checkpoint_dir ${CHECKPOINT_DIR} \
        ${WANDB_ARGS} ${HF_ARGS}

    local LOLA_FT_CKPT=$(ls -t ${CHECKPOINT_DIR}/${MODEL_LOLA}/*_ft.pt 2>/dev/null | head -1)
    echo "  微调 checkpoint: ${LOLA_FT_CKPT}"

    run_lm_eval "lolcats_ckpt" "" "${LOLCATS_DISTILL_CKPT}" "${LOLA_FT_CKPT}" \
        "LoLA (LoLCATs 蒸馏 + LoLA LoRA 微调)"
}

# ============================================================================
# 实验 6: LoLA (trainable, 完整两阶段自训练)
# Stage 1: LoLA 注意力蒸馏 (三层混合注意力, SRE 选择在 forward 中活跃)
# Stage 2: LoRA 微调
# 训推完全对齐
# ============================================================================
exp6_lola_train() {
    print_header "实验 6: LoLA (trainable, 完整两阶段自训练)"

    python distill_llama.py \
        --model_config ${MODEL_LOLA} \
        --distill_config ${DISTILL_CONFIG} \
        --finetune_config ${FINETUNE_CONFIG} \
        --eval_config ${EVAL_CONFIG} \
        --checkpoint_dir ${CHECKPOINT_DIR} \
        ${WANDB_ARGS} ${HF_ARGS}

    local LOLA_DISTILL_CKPT=$(ls -t ${CHECKPOINT_DIR}/${MODEL_LOLA}/*_distill.pt 2>/dev/null | head -1)
    local LOLA_FT_CKPT=$(ls -t ${CHECKPOINT_DIR}/${MODEL_LOLA}/*_ft.pt 2>/dev/null | head -1)

    echo "  蒸馏 checkpoint: ${LOLA_DISTILL_CKPT}"
    echo "  微调 checkpoint: ${LOLA_FT_CKPT}"

    run_lm_eval "lolcats_ckpt" "" "${LOLA_DISTILL_CKPT}" "${LOLA_FT_CKPT}" \
        "LoLA (完整两阶段自训练)"
}

# ============================================================================
# 实验 7: LoLA + Sink (trainable, 完整两阶段自训练)
# sink_size=4, sparse_budget=128
# ============================================================================
exp7_lola_sink() {
    print_header "实验 7: LoLA + Sink (trainable, sink_size=4)"

    python distill_llama.py \
        --model_config ${MODEL_LOLA_SINK} \
        --distill_config ${DISTILL_CONFIG} \
        --finetune_config ${FINETUNE_CONFIG} \
        --eval_config ${EVAL_CONFIG} \
        --checkpoint_dir ${CHECKPOINT_DIR} \
        ${WANDB_ARGS} ${HF_ARGS}

    local CKPT_DIR="${CHECKPOINT_DIR}/${MODEL_LOLA_SINK}"
    local DISTILL_CKPT=$(ls -t ${CKPT_DIR}/*_distill.pt 2>/dev/null | head -1)
    local FT_CKPT=$(ls -t ${CKPT_DIR}/*_ft.pt 2>/dev/null | head -1)

    run_lm_eval "lolcats_ckpt" "" "${DISTILL_CKPT}" "${FT_CKPT}" \
        "LoLA + Sink (sink_size=4)"
}

# ============================================================================
# 实验 8: LoLA + Budget (trainable, 跨层 uniform budget)
# budget_strategy=uniform, total_sparse_budget=4096
# ============================================================================
exp8_lola_budget() {
    print_header "实验 8: LoLA + Budget (trainable, uniform budget)"

    python distill_llama.py \
        --model_config ${MODEL_LOLA_BUDGET} \
        --distill_config ${DISTILL_CONFIG} \
        --finetune_config ${FINETUNE_CONFIG} \
        --eval_config ${EVAL_CONFIG} \
        --checkpoint_dir ${CHECKPOINT_DIR} \
        ${WANDB_ARGS} ${HF_ARGS}

    local CKPT_DIR="${CHECKPOINT_DIR}/${MODEL_LOLA_BUDGET}"
    local DISTILL_CKPT=$(ls -t ${CKPT_DIR}/*_distill.pt 2>/dev/null | head -1)
    local FT_CKPT=$(ls -t ${CKPT_DIR}/*_ft.pt 2>/dev/null | head -1)

    run_lm_eval "lolcats_ckpt" "" "${DISTILL_CKPT}" "${FT_CKPT}" \
        "LoLA + Budget (uniform)"
}

# ============================================================================
# 实验 9: LoLA + Sink + Budget (trainable)
# sink_size=4, budget_strategy=uniform, total_sparse_budget=4096
# ============================================================================
exp9_lola_sink_budget() {
    print_header "实验 9: LoLA + Sink + Budget (trainable)"

    python distill_llama.py \
        --model_config ${MODEL_LOLA_SINK_BUDGET} \
        --distill_config ${DISTILL_CONFIG} \
        --finetune_config ${FINETUNE_CONFIG} \
        --eval_config ${EVAL_CONFIG} \
        --checkpoint_dir ${CHECKPOINT_DIR} \
        ${WANDB_ARGS} ${HF_ARGS}

    local CKPT_DIR="${CHECKPOINT_DIR}/${MODEL_LOLA_SINK_BUDGET}"
    local DISTILL_CKPT=$(ls -t ${CKPT_DIR}/*_distill.pt 2>/dev/null | head -1)
    local FT_CKPT=$(ls -t ${CKPT_DIR}/*_ft.pt 2>/dev/null | head -1)

    run_lm_eval "lolcats_ckpt" "" "${DISTILL_CKPT}" "${FT_CKPT}" \
        "LoLA + Sink + Budget"
}

# ============================================================================
# 消融实验 A: sparse_budget 消融 (基于实验 6 的流程)
# budget = 64, 128, 256
# ============================================================================
ablation_sparse_budget() {
    print_header "消融实验 A: sparse_budget 消融"

    for BUDGET in 64 128 256; do
        echo ""
        echo "  ---- sparse_budget = ${BUDGET} ----"
        echo ""

        # 创建临时配置: 覆盖 sparse_budget
        # 注意: distill_llama.py 不支持直接覆盖 attention 子参数,
        # 因此需要为每个 budget 创建单独配置文件, 或使用符号链接方式
        # 这里使用 python 动态生成配置
        python -c "
import yaml
with open('configs/model/${MODEL_LOLA}.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['attention']['sparse_budget'] = ${BUDGET}
out_name = 'configs/model/${MODEL_LOLA}_budget${BUDGET}.yaml'
with open(out_name, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
print(f'  已生成配置: {out_name}')
"
        local MODEL_NAME="${MODEL_LOLA}_budget${BUDGET}"

        python distill_llama.py \
            --model_config ${MODEL_NAME} \
            --distill_config ${DISTILL_CONFIG} \
            --finetune_config ${FINETUNE_CONFIG} \
            --eval_config ${EVAL_CONFIG} \
            --checkpoint_dir ${CHECKPOINT_DIR} \
            ${WANDB_ARGS} ${HF_ARGS}

        local CKPT_DIR="${CHECKPOINT_DIR}/${MODEL_NAME}"
        local DISTILL_CKPT=$(ls -t ${CKPT_DIR}/*_distill.pt 2>/dev/null | head -1)
        local FT_CKPT=$(ls -t ${CKPT_DIR}/*_ft.pt 2>/dev/null | head -1)

        run_lm_eval "lolcats_ckpt" "" "${DISTILL_CKPT}" "${FT_CKPT}" \
            "LoLA sparse_budget=${BUDGET}"
    done
}

# ============================================================================
# 消融实验 B: budget 策略消融 (基于实验 8 的流程)
# strategy = uniform, pyramid
# ============================================================================
ablation_budget_strategy() {
    print_header "消融实验 B: budget 策略消融"

    for STRATEGY in uniform pyramid; do
        echo ""
        echo "  ---- budget_strategy = ${STRATEGY} ----"
        echo ""

        python -c "
import yaml
with open('configs/model/${MODEL_LOLA_BUDGET}.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['attention']['budget_strategy'] = '${STRATEGY}'
out_name = 'configs/model/${MODEL_LOLA_BUDGET}_${STRATEGY}.yaml'
with open(out_name, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
print(f'  已生成配置: {out_name}')
"
        local MODEL_NAME="${MODEL_LOLA_BUDGET}_${STRATEGY}"

        python distill_llama.py \
            --model_config ${MODEL_NAME} \
            --distill_config ${DISTILL_CONFIG} \
            --finetune_config ${FINETUNE_CONFIG} \
            --eval_config ${EVAL_CONFIG} \
            --checkpoint_dir ${CHECKPOINT_DIR} \
            ${WANDB_ARGS} ${HF_ARGS}

        local CKPT_DIR="${CHECKPOINT_DIR}/${MODEL_NAME}"
        local DISTILL_CKPT=$(ls -t ${CKPT_DIR}/*_distill.pt 2>/dev/null | head -1)
        local FT_CKPT=$(ls -t ${CKPT_DIR}/*_ft.pt 2>/dev/null | head -1)

        run_lm_eval "lolcats_ckpt" "" "${DISTILL_CKPT}" "${FT_CKPT}" \
            "LoLA budget_strategy=${STRATEGY}"
    done
}

# ============================================================================
# 单独评估 (可用已有 checkpoint 直接评估)
# 用法: bash scripts/run_experiments.sh eval <attn_ckpt> <ft_ckpt> <描述>
# ============================================================================
eval_checkpoint() {
    local attn_ckpt=$1
    local ft_ckpt=$2
    local desc=${3:-"自定义评估"}
    run_lm_eval "lolcats_ckpt" "" "${attn_ckpt}" "${ft_ckpt}" "${desc}"
}

# ============================================================================
# 单独训练 (可指定任意模型配置)
# 用法: bash scripts/run_experiments.sh train <model_config> [distill_ckpt]
# ============================================================================
train_model() {
    local model_config=$1
    local distill_ckpt=${2:-}

    print_header "训练: ${model_config}"

    local extra_args=""
    if [ -n "$distill_ckpt" ]; then
        extra_args="--load_distill_checkpoint ${distill_ckpt}"
        echo "  跳过蒸馏, 加载: ${distill_ckpt}"
    fi

    python distill_llama.py \
        --model_config ${model_config} \
        --distill_config ${DISTILL_CONFIG} \
        --finetune_config ${FINETUNE_CONFIG} \
        --eval_config ${EVAL_CONFIG} \
        --checkpoint_dir ${CHECKPOINT_DIR} \
        ${extra_args} \
        ${WANDB_ARGS} ${HF_ARGS}
}

# ============================================================================
# Demo 生成测试
# 用法: bash scripts/run_experiments.sh demo <attn_ckpt> <ft_ckpt>
# ============================================================================
demo_generation() {
    local attn_ckpt=$1
    local ft_ckpt=$2

    print_header "Demo 生成测试"

    python demo_lolcats_llm.py \
        --attn_mlp_checkpoint_path ${attn_ckpt} \
        --finetune_checkpoint_path ${ft_ckpt} \
        --max_new_tokens 256
}

# ============================================================================
# 主入口
# ============================================================================
case "${1:-help}" in
    1)  exp1_baseline ;;
    2)  exp2_lolcats_tf ;;
    3)  exp3_lolcats_train ;;
    4)  exp4_lola_tf ;;
    5)  exp5_lola_ft_lolcats ;;
    6)  exp6_lola_train ;;
    7)  exp7_lola_sink ;;
    8)  exp8_lola_budget ;;
    9)  exp9_lola_sink_budget ;;

    all)
        exp1_baseline
        exp3_lolcats_train
        exp6_lola_train
        exp7_lola_sink
        exp8_lola_budget
        exp9_lola_sink_budget
        ;;

    train_all)
        # 仅训练, 不评估 (适合先批量训练再统一评估)
        print_header "批量训练: LoLCATs + LoLA 全系列"
        train_model ${MODEL_LOLCATS}
        train_model ${MODEL_LOLA}
        train_model ${MODEL_LOLA_SINK}
        train_model ${MODEL_LOLA_BUDGET}
        train_model ${MODEL_LOLA_SINK_BUDGET}
        ;;

    ablation)
        ablation_sparse_budget
        ablation_budget_strategy
        ;;

    ablation_budget)   ablation_sparse_budget ;;
    ablation_strategy) ablation_budget_strategy ;;

    eval)
        shift
        eval_checkpoint "$@"
        ;;

    train)
        shift
        train_model "$@"
        ;;

    demo)
        shift
        demo_generation "$@"
        ;;

    help|*)
        echo "LoLA + LoLCATs 实验脚本"
        echo ""
        echo "用法: bash scripts/run_experiments.sh <命令>"
        echo ""
        echo "核心实验 (训练+评估):"
        echo "  1              实验1: Baseline (Llama-3.1-8B softmax)"
        echo "  2              实验2: LoLCATs (training-free, 需设置 LOLCATS_ATTN_CKPT, LOLCATS_FT_CKPT)"
        echo "  3              实验3: LoLCATs (两阶段训练)"
        echo "  4              实验4: LoLA (training-free, 需设置 LOLCATS_DISTILL_CKPT, LOLCATS_FT_CKPT)"
        echo "  5              实验5: LoLA (LoLCATs 权重 + LoRA 微调, 需设置 LOLCATS_DISTILL_CKPT)"
        echo "  6              实验6: LoLA (完整两阶段自训练)"
        echo "  7              实验7: LoLA + Sink (两阶段自训练, sink_size=4)"
        echo "  8              实验8: LoLA + Budget (两阶段自训练, uniform)"
        echo "  9              实验9: LoLA + Sink + Budget (两阶段自训练)"
        echo "  all            运行实验 1,3,6,7,8,9 (不含需要外部权重的实验)"
        echo "  train_all      仅训练全部模型 (不评估)"
        echo ""
        echo "消融实验:"
        echo "  ablation          运行全部消融实验"
        echo "  ablation_budget   sparse_budget 消融 (64/128/256)"
        echo "  ablation_strategy budget 策略消融 (uniform/pyramid)"
        echo ""
        echo "工具命令:"
        echo "  train <model_config> [distill_ckpt]   训练指定模型"
        echo "  eval <attn_ckpt> <ft_ckpt> [描述]     评估指定 checkpoint"
        echo "  demo <attn_ckpt> <ft_ckpt>            运行 demo 生成"
        echo ""
        echo "环境变量:"
        echo "  HF_TOKEN        HuggingFace token"
        echo "  CACHE_DIR       模型缓存目录 (默认: /scratch/)"
        echo "  CHECKPOINT_DIR  checkpoint 保存目录 (默认: ./checkpoints)"
        echo "  LM_EVAL_PATH    lm-evaluation-harness 路径"
        echo "  WANDB_ENTITY    WandB 实体名 (默认: hazy-research)"
        echo "  NO_WANDB=1      禁用 WandB"
        echo ""
        echo "示例:"
        echo "  # 运行 LoLA 完整训练+评估"
        echo "  bash scripts/run_experiments.sh 6"
        echo ""
        echo "  # 运行全部核心实验"
        echo "  NO_WANDB=1 bash scripts/run_experiments.sh all"
        echo ""
        echo "  # 评估已有 checkpoint"
        echo "  bash scripts/run_experiments.sh eval ./checkpoints/.../distill.pt ./checkpoints/.../ft.pt"
        ;;
esac
