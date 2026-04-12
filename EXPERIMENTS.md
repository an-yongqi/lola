# LoLA + LoLCATs 实验指南

## 概述

本项目在 LoLCATs 基础上实现了 **LoLA (Low-rank Linear Attention with Sparse Cache)** -- 三层混合注意力架构:

1. **滑动窗口 (softmax)**: 最近 `window_size` 个 token, 精确 softmax 注意力
2. **稀疏缓存 (softmax)**: 通过 Self-Recall Error (SRE) 选出的重要 token, 精确 softmax 注意力
3. **线性状态 (linear)**: 其余 token, 压缩到固定大小的 recurrent state

关键特性:
- **训推对齐**: 稀疏缓存在训练时通过 SRE 选择活跃, 与推理时一致
- **Sink Token**: 前 N 个 token 永久保留, 不被驱逐 (StreamingLLM 思想)
- **跨层 Budget**: 不同层使用不同的 sparse budget (uniform / pyramid / list 策略)
- **轻量训练**: 两阶段流程与 LoLCATs 完全一致 (蒸馏 + LoRA 微调)

## 环境准备

```bash
# 1. 安装基础依赖
pip install torch transformers omegaconf peft tqdm wandb

# 2. 访问 Llama-3.1-8B (需要 HuggingFace 授权)
export HF_TOKEN="your_hf_token"

# 3. (可选) 克隆 lm-evaluation-harness 用于评估
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
export LM_EVAL_PATH="/path/to/lm-evaluation-harness"

# 4. (可选) 禁用 WandB
export NO_WANDB=1
```

## 实验列表

### 核心实验 (9 个)

| # | 实验 | 配置文件 | 训练方式 | sparse_budget | sink_size | 训推对齐 |
|---|------|---------|---------|---------------|-----------|---------|
| 1 | Baseline (softmax) | `base_llama3_1_8b` | 无 | - | - | - |
| 2 | LoLCATs (training-free) | `distill_..._wsw64_fd64_w01` | 无 (加载权重) | - | - | - |
| 3 | LoLCATs (trainable) | `distill_..._wsw64_fd64_w01` | 蒸馏 + LoRA | - | - | Yes |
| 4 | LoLA (training-free) | `distill_..._lola` | 无 (加载 LoLCATs 权重) | 128 | 0 | **No** |
| 5 | LoLA (LoLCATs 权重 + LoRA) | `distill_..._lola` | 仅 LoRA | 128 | 0 | 部分 |
| 6 | LoLA (完整自训练) | `distill_..._lola` | 蒸馏 + LoRA | 128 | 0 | **Yes** |
| 7 | LoLA + Sink | `distill_..._lola_sink` | 蒸馏 + LoRA | 128 | 4 | **Yes** |
| 8 | LoLA + Budget | `distill_..._lola_budget` | 蒸馏 + LoRA | 按策略 | 0 | **Yes** |
| 9 | LoLA + Sink + Budget | `distill_..._lola_sink_budget` | 蒸馏 + LoRA | 按策略 | 4 | **Yes** |

> 配置文件前缀省略为 `distill_llama3_1_8b_lk_smd`

### 消融实验

**A. sparse_budget 消融** (基于实验 6):
- budget = 64, 128, 256
- 目标: 观察稀疏缓存大小对模型质量的影响

**B. budget 策略消融** (基于实验 8):
- uniform: 每层相同 budget
- pyramid: 首尾层多, 中间层少
- 目标: 观察异构 budget 分配是否优于均匀分配

## 运行指令

### 快速开始: 运行单个实验

```bash
# 实验 6: LoLA 完整自训练 (最推荐的核心实验)
bash scripts/run_experiments.sh 6

# 实验 1: Baseline 评估 (仅评估, 无需训练)
bash scripts/run_experiments.sh 1
```

### 运行全部可独立运行的实验

```bash
# 运行实验 1, 3, 6, 7, 8, 9 (不含需要外部权重的 2, 4, 5)
bash scripts/run_experiments.sh all
```

### 运行消融实验

```bash
# 全部消融
bash scripts/run_experiments.sh ablation

# 单独运行
bash scripts/run_experiments.sh ablation_budget    # sparse_budget 消融
bash scripts/run_experiments.sh ablation_strategy  # budget 策略消融
```

### 手动训练和评估

```bash
# 仅训练 (不评估)
bash scripts/run_experiments.sh train distill_llama3_1_8b_lk_smd_wsw64_fd64_w01_lola

# 跳过蒸馏, 仅做 LoRA 微调 (需要已有蒸馏 checkpoint)
bash scripts/run_experiments.sh train distill_llama3_1_8b_lk_smd_wsw64_fd64_w01_lola /path/to/distill.pt

# 评估已有 checkpoint
bash scripts/run_experiments.sh eval /path/to/distill.pt /path/to/ft.pt "自定义描述"

# Demo 生成测试
bash scripts/run_experiments.sh demo /path/to/distill.pt /path/to/ft.pt
```

### 直接使用 Python 命令

```bash
# Stage 1: 蒸馏
python distill_llama.py \
    --model_config distill_llama3_1_8b_lk_smd_wsw64_fd64_w01_lola \
    --distill_config distill_alpaca_clean_xent0_mse1000_lr1e-2 \
    --finetune_config finetune_lora_qkvo_alpaca_clean \
    --eval_config eval_alpaca_clean \
    --no_wandb

# 加载已有蒸馏权重, 仅做 LoRA 微调
python distill_llama.py \
    --model_config distill_llama3_1_8b_lk_smd_wsw64_fd64_w01_lola \
    --distill_config distill_alpaca_clean_xent0_mse1000_lr1e-2 \
    --load_distill_checkpoint ./checkpoints/.../xxx_distill.pt \
    --finetune_config finetune_lora_qkvo_alpaca_clean \
    --eval_config eval_alpaca_clean \
    --no_wandb

# lm-evaluation-harness 评估
python lm_eval_harness/eval_lm_harness.py \
    --model_type lolcats_ckpt \
    --attn_mlp_checkpoint_path ./checkpoints/.../xxx_distill.pt \
    --finetune_checkpoint_path ./checkpoints/.../xxx_ft.pt \
    --task arc-challenge --num_shots 25 \
    --no_wandb
```

## 训练流程说明

### 两阶段训练 (与 LoLCATs 完全一致)

**Stage 1 - 注意力蒸馏** (~2 epochs on alpaca-cleaned):
- 目标: 训练 feature map MLP, 使三层混合注意力输出匹配原始 softmax 输出
- Loss: MSE(y_pred, y_true)
- 可训参数: `feature_map_q.mlp`, `feature_map_k.mlp`, `window_factors`
- LoLA 特有: `sparse_factors` (稀疏缓存门控, 初始 sigmoid(10.0) ≈ 1.0)

**Stage 2 - LoRA 微调** (~2 epochs on alpaca-cleaned):
- 目标: 恢复因注意力替换造成的语言建模能力损失
- Loss: Cross-entropy (标准 LM loss)
- 可训参数: q/k/v/o_proj 的 LoRA (r=8, alpha=16)
- 三层混合注意力在 forward 中活跃, 保证训推对齐

### 训练速度

与 LoLCATs 相同的 O(n^2) quadratic attention (seq_len=1024 → 1024×1024 注意力矩阵)。SRE 计算的额外开销约 6% (O(n·f·d) vs 注意力的 O(n^2·d)), 可忽略。

## 配置文件说明

### 模型配置 (`configs/model/`)

| 配置 | attention_type | sparse_budget | sink_size | budget_strategy |
|------|---------------|---------------|-----------|-----------------|
| `base_llama3_1_8b` | softmax | - | - | - |
| `distill_..._wsw64_fd64_w01` | lolcats_llama_window_sw | - | - | - |
| `distill_..._lola` | lolcats_llama_window_lola | 128 | 0 | - |
| `distill_..._lola_sink` | lolcats_llama_window_lola | 128 | 4 | - |
| `distill_..._lola_budget` | lolcats_llama_window_lola | 128 | 0 | uniform |
| `distill_..._lola_sink_budget` | lolcats_llama_window_lola | 128 | 4 | uniform |

### LoLA 特有配置参数

```yaml
attention:
  attention_type: lolcats_llama_window_lola
  sparse_budget: 128          # 每层稀疏缓存大小 (per head)
  sink_size: 0                # sink token 数量 (0=禁用)
  init_sparse_factor: 10.0    # sparse_factor 初始值 (sigmoid 后 ≈ 1.0)
  train_sparse_factor: true   # 是否可训练
  # 跨层 budget (可选)
  budget_strategy: uniform    # uniform / pyramid / list
  total_sparse_budget: 4096   # 所有层的总 sparse budget
  # layer_budgets: [...]      # strategy=list 时手动指定每层
```

### 实验配置 (`configs/experiment/`)

| 配置 | 用途 |
|------|------|
| `distill_alpaca_clean_xent0_mse1000_lr1e-2` | Stage 1 蒸馏 |
| `finetune_lora_qkvo_alpaca_clean` | Stage 2 LoRA 微调 |
| `eval_alpaca_clean` | 评估 |

## 评估指标

使用 lm-evaluation-harness 评估以下 benchmark:

| 任务 | Shots | 指标 |
|------|-------|------|
| ARC-Challenge | 25 | acc_norm |
| HellaSwag | 10 | acc_norm |
| WinoGrande | 5 | acc |
| TruthfulQA-MC | 0 | mc2 |
| GSM8K | 5 | acc |
| MMLU (hendrycksTest) | 5 | acc |

## 预期结果格式

评估完成后, 结果保存在 `lm_eval_harness/results_lm_eval.csv`, 格式:

```
task,shots,acc,acc_norm,acc_stderr,acc_norm_stderr,attn_mlp_path,finetune_path
arc-challenge,25,0.xxx,0.xxx,...
hellaswag,10,0.xxx,0.xxx,...
```

建议整理为如下对比表:

| 模型 | ARC-C | HellaSwag | WinoGrande | TruthfulQA | GSM8K | MMLU | Avg |
|------|-------|-----------|------------|------------|-------|------|-----|
| Baseline (softmax) | | | | | | | |
| LoLCATs (training-free) | | | | | | | |
| LoLCATs (trainable) | | | | | | | |
| LoLA (training-free) | | | | | | | |
| LoLA (LoLCATs + LoRA) | | | | | | | |
| LoLA (自训练) | | | | | | | |
| LoLA + Sink | | | | | | | |
| LoLA + Budget | | | | | | | |
| LoLA + Sink + Budget | | | | | | | |

## 文件结构

```
src/model/linear_attention/
├── linear_window_attention_lola.py   # LoLA 核心实现
│   ├── compute_sre()                 # Self-Recall Error 计算
│   ├── select_sparse_tokens()        # 稀疏 token 选择 (per-head top-k)
│   ├── hybrid_attention_quadratic_lola()  # 三层混合注意力
│   ├── LolcatsLoLAAttention          # 注意力层 (继承 SlidingWindowAttention)
│   └── LinearAttentionLoLACache      # 缓存类 (窗口+稀疏+sink+线性状态)
├── linear_window_attention_sw.py     # LoLCATs 基础 (滑动窗口)
└── __init__.py                       # 导出 LoLA 类

src/model/convert_model.py
├── compute_layer_budgets()           # 跨层 budget 分配 (uniform/pyramid/list)
├── get_attention()                   # 注册 lolcats_llama_window_lola
└── get_attention_cache()             # 注册 LinearAttentionLoLACache

configs/model/
├── distill_..._lola.yaml             # LoLA 基础
├── distill_..._lola_sink.yaml        # LoLA + sink
├── distill_..._lola_budget.yaml      # LoLA + budget
└── distill_..._lola_sink_budget.yaml # LoLA + sink + budget

scripts/
└── run_experiments.sh                # 实验运行脚本
```

## 常见问题

### Q: LoLA 加载 LoLCATs 权重时报 missing keys 怎么办?
A: 正常现象。LoLA 新增的 `sparse_factors` 参数在 LoLCATs checkpoint 中不存在, `strict=False` 加载时会使用默认初始值 (sigmoid(10.0) ≈ 1.0)。但这是训推不一致的, 建议用实验 6 的完整两阶段训练。

### Q: sparse_budget 应该设多大?
A: 默认 128。消融实验覆盖 64/128/256。更大的 budget 保留更多精确注意力, 但增加内存和计算开销。建议从 128 开始。

### Q: 跨层 budget 分配的 "pyramid" 策略是什么意思?
A: 首尾层分配更多 sparse budget, 中间层分配更少。直觉是首层和尾层的注意力模式更复杂, 需要更多精确注意力补偿。

### Q: 训练需要多少 GPU 显存?
A: 与 LoLCATs 相同。Llama-3.1-8B + bf16 + LoRA (r=8) 大约需要 ~20GB 显存 (单 A100/H100)。蒸馏阶段因保留 teacher attention 会稍多。

### Q: 如何自定义每层的 budget?
A: 在模型配置中设置:
```yaml
attention:
  budget_strategy: list
  layer_budgets: [256, 256, 128, 128, 64, 64, ...]  # 32 层
```
