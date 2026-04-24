# LoLA 代码逐行解析

> 辅助人眼审查的详细文档。所有 `→` 链接均可跳转到源码对应行。
>
> 论文参考: [LoLA: Low-Rank Linear Attention With Sparse Caching (arxiv:2505.23666)](https://arxiv.org/abs/2505.23666)

---

## 目录

1. [整体架构](#1-整体架构)
2. [compute_sre — SRE 计算](#2-compute_sre)
3. [sparse token 选择 — serial 和 chunked 两版](#3-sparse-token-选择)
4. [hybrid_attention_quadratic_lola — 三层混合注意力 (训练)](#4-hybrid_attention_quadratic_lola)
5. [LolcatsLoLAAttention — 注意力层 (4 条 forward 路径)](#5-lolcatslolaattention)
6. [LinearAttentionLoLACache — 缓存类](#6-linearattentionlolacache)
   - [Cache.update — stateful training 路径](#61-cacheupdate)
   - [Cache.update_for_decoding — 解码路径](#62-cacheupdate_for_decoding)
7. [convert_model.py — 注册与 budget 分配](#7-convert_modelpy)
8. [论文公式 vs 实现对照表](#8-论文公式-vs-实现对照表)
9. [已知设计偏差](#9-已知设计偏差)

---

## 1. 整体架构

```
文件关系:
linear_attention.py                    ← 基类 LolcatsLinearAttention, LinearAttentionState
  └── linear_window_attention_sw.py    ← 父类 LolcatsSlidingWindowAttention, LinearAttentionSlidingWindowCache
        └── linear_window_attention_lola.py  ← LoLA 实现 (本文档主角)
              ├── compute_sre()
              ├── select_sparse_tokens_serial()    ← per-token 精确选择 (参考实现)
              ├── select_sparse_tokens_chunked()   ← per-chunk 高效选择 (默认)
              ├── hybrid_attention_quadratic_lola()
              ├── class LolcatsLoLAAttention
              └── class LinearAttentionLoLACache

convert_model.py
  ├── compute_layer_budgets()          ← 跨层 budget 分配
  ├── get_attention()                  ← 注册 lolcats_llama_window_lola
  └── get_attention_cache()            ← 注册 LinearAttentionLoLACache
```

**三层混合注意力** (论文 Eq.14):

```
y_t = [ φ(q)^T H_t  +  Σ_{sparse} exp(qk/√d)·v  +  Σ_{window} exp(qk/√d)·v ]
      / [ φ(q)^T s_t  +  Σ_{sparse} exp(qk/√d)    +  Σ_{window} exp(qk/√d)   ]
```

实现在此基础上增加了 **可学习 gating factors** (非论文内容):
- `window_factor`: sigmoid(learnable), 默认 ≈ 0.1
- `sparse_factor`: sigmoid(learnable), 默认 ≈ 1.0
- `linear_factor`: `1 - window_factor` 或 `1`

---

## 2. compute_sre

> 论文 Eq.10: `SRE(k, v | H, s) = ||v - φ(k)^T H / (φ(k)^T s) ||²`

→ [linear_window_attention_lola.py:35-68](../src/model/linear_attention/linear_window_attention_lola.py#L35-L68)

```python
def compute_sre(f_k, v, kv_state=None, k_state=None, eps=1e-12):
```

### 输入

| 参数 | Shape | 说明 |
|------|-------|------|
| `f_k` | `(b, h, l, f)` | feature-mapped keys φ(k) |
| `v` | `(b, h, l, d)` | values |
| `kv_state` | `(b, h, f, d)` or None | 线性状态 H。None → 从 f_k, v 计算 |
| `k_state` | `(b, h, 1, f)` or None | key 累积 s。None → 从 f_k 计算 |

### 逐行

→ [L53-56](../src/model/linear_attention/linear_window_attention_lola.py#L53-L56): **构建线性状态 H, s**
```python
if kv_state is None:
    H = einsum('bhlf,bhld->bhfd', f_k, v)   # Σ φ(k_i) v_i^T
    s = f_k.sum(dim=2, keepdim=True)         # Σ φ(k_i), shape (b,h,1,f)
```
- 仅在训练路径 kv_state=None 时触发
- **注意**: 调用方已通过 eligible_mask 过滤, 此处的 f_k, v 可能已被 mask

→ [L57-59](../src/model/linear_attention/linear_window_attention_lola.py#L57-L59): **使用外部传入的状态**
```python
else:
    H = kv_state.float()
    s = k_state.float()
```
- 解码路径: 传入实际累积的 decode_kv_states
- stateful training: 传入 chunk 间累积的 kv_states

→ [L62-64](../src/model/linear_attention/linear_window_attention_lola.py#L62-L64): **重建 value**
```python
numerator   = einsum('bhfd,bhlf->bhld', H, f_k)   # H @ φ(k_i), 对每个 i
denominator = einsum('bhnf,bhlf->bhl',  s, f_k)    # s^T @ φ(k_i), 对每个 i
v_hat = numerator / (denominator[..., None] + eps)
```
- `numerator` shape `(b,h,l,d)`: 每个 token 的重建 value
- `denominator` shape `(b,h,l)`: 每个 token 的归一化因子
- `eps` 防止除零

→ [L67](../src/model/linear_attention/linear_window_attention_lola.py#L67): **计算 SRE**
```python
sre = ((v.float() - v_hat) ** 2).sum(dim=-1)  # (b, h, l)
```
- L2 范数的平方, 在 head_dim 维度上求和
- 每个 (batch, head, token) 一个标量

### 审查要点
- [ ] `kv_state=None` 时走全局近似 — serial/chunked 选择函数内部调用时始终传入 kv_state (演化中的 H), 此分支仅在直接调用 compute_sre 时使用
- [ ] einsum 索引是否正确: `bhfd,bhlf->bhld` (H 的 f 维度与 f_k 的 f 维度缩约) ✓
- [ ] denominator shape `(b,h,l)` → `[..., None]` 扩展为 `(b,h,l,1)` 用于广播除法 ✓

---

## 3. sparse token 选择

> 论文 Eq.12-13, 15: per-query G_t 选择, 不可逆吸收 (irreversible absorption)
>
> 提供两个实现: **serial** (per-token 精确, 参考实现) 和 **chunked** (per-chunk 高效, 默认)

### 3.1 select_sparse_tokens_serial — 精确参考实现

→ [linear_window_attention_lola.py:74-174](../src/model/linear_attention/linear_window_attention_lola.py#L74-L174)

```python
def select_sparse_tokens_serial(f_k, v, sparse_budget, window_size,
                                 q_len, k_len, sink_size=0, eps=1e-12,
                                 kv_state=None, k_state=None):
```

**核心算法**: 逐 token 处理, 严格匹配论文 Eq.12-13

```
初始化: H_0 = 0, s_0 = 0, G_0 = ∅
For each query position t = 0, 1, ..., q_len-1:
    bw_boundary = t + offset - window_size   (below-window 边界)
    While 有新 key 被驱逐出窗口:
        p = next_evict
        If |G| >= budget:
            E = G ∪ {p}
            SRE(E) = compute_sre(f_k, v, H, s)
            absorb = argmin(SRE over E)
            H += φ(k_absorb) ⊗ v_absorb    (不可逆!)
            G = E \ {absorb}
        Else:
            G = G ∪ {p}
    sparse_mask[t] = G ∩ [0, bw_boundary]
```

**关键实现细节**:

→ [L112-117](../src/model/linear_attention/linear_window_attention_lola.py#L112-L117): **初始化线性状态**
```python
if kv_state is not None:  # stateful training: 继承先前 chunk 的 H
    H = kv_state.float().clone()
else:
    H = torch.zeros(b, h, f_dim, d, ...)
```

→ [L119-120](../src/model/linear_attention/linear_window_attention_lola.py#L119-L120): **per-(b,h) 追踪**
```python
is_in_G = torch.zeros(b, h, k_len, dtype=torch.bool)  # 哪些 key 在 sparse cache
```
- 使用 `(b, h, k_len)` boolean mask 追踪 G 的成员, 避免复杂的索引管理
- 不同 head 的 G 可以不同 (因 SRE 不同), 但驱逐时机相同

→ [L132-135](../src/model/linear_attention/linear_window_attention_lola.py#L132-L135): **驱逐时机**
```python
bw_boundary = t + offset - window_size
while next_evict <= bw_boundary and next_evict < k_len:
```
- `offset = k_len - q_len`: 支持 q_len ≠ k_len (stateful training)
- 每个 query 步最多驱逐一个 key (通常 `k_len == q_len` 时)
- 首个 query 可能有多个 below-window key (offset > window_size 时), while 循环处理

→ [L143-163](../src/model/linear_attention/linear_window_attention_lola.py#L143-L163): **超 budget 时吸收 min-SRE token**
```python
is_in_G[:, :, p] = True   # 临时加入
sre_all = compute_sre(f_k, v, kv_state=H, k_state=s)
sre_all[~is_in_G] = float('inf')  # 非 eligible → 不会被选为 min
_, absorb_pos = sre_all.min(dim=-1)  # (b, h) — 各 head 独立
```
- SRE 对全 k_len 计算, 非 eligible 位置设为 inf → argmin 不会选中
- `absorb_pos` per-head 不同: 各 head 基于自身 SRE 做独立决策

→ [L159-163](../src/model/linear_attention/linear_window_attention_lola.py#L159-L163): **不可逆吸收 (论文 Eq.13)**
```python
H = H + einsum('bhlf,bhld->bhfd', absorbed_fk, absorbed_v)
s = s + absorbed_fk
is_in_G.scatter_(-1, absorb_pos.unsqueeze(-1), False)
```
- 被吸收 token 的 φ(k)v^T 加入 H → **永远无法返回 sparse cache**
- scatter False: 从 G 中移除被吸收位置

→ [L170-172](../src/model/linear_attention/linear_window_attention_lola.py#L170-L172): **记录当前 G**
```python
if bw_boundary >= 0:
    end = min(bw_boundary + 1, k_len)
    sparse_mask[:, :, t, :end] = is_in_G[:, :, :end]
```
- 只写入 below-window 范围内的 G 位置 (causal constraint)

**返回值**: `(b, h, q_len, k_len)` — 每个 query 有独立的 sparse mask

### 审查要点 (serial)
- [ ] 驱逐同步: 所有 (b,h) 同一时刻驱逐同一 key → n_in_G 始终一致 ✓
- [ ] 不可逆: absorbed token 进入 H 后, is_in_G 设为 False, 永不再为 True ✓
- [ ] 自身可被吸收: 若刚加入的 p 有最小 SRE, 它会被立即吸收 (正确行为) ✓
- [ ] 复杂度: O(q_len × budget × k_len), 串行无法并行 — 仅作参考实现

---

### 3.2 select_sparse_tokens_chunked — 高效 chunk-wise 实现

→ [linear_window_attention_lola.py:180-293](../src/model/linear_attention/linear_window_attention_lola.py#L180-L293)

```python
def select_sparse_tokens_chunked(f_k, v, sparse_budget, window_size,
                                  q_len, k_len, chunk_size,
                                  sink_size=0, eps=1e-12,
                                  kv_state=None, k_state=None):
```

**核心算法**: 按 chunk 处理, chunk 内共享 G (论文 Section 3.3)

```
初始化: H_0 = 0, G_0 = ∅
For each chunk c = 0, 1, ...:
    1. 先处理驱逐 → 更新 G:
        evicted = 本 chunk 期间驱逐出窗口的 key 集合
        E = G_{prev} ∪ evicted
        If |E| > budget:
            SRE(E | H)
            G_c = top-budget(E by SRE)
            fallen = E \ G_c
            H += Σ_{fallen} φ(k)v^T   (不可逆!)
        Else:
            G_c = E

    2. 再写入 mask (用更新后的 G_c):
        queries [c*C, (c+1)*C):
            sparse_mask[t] = G_c (所有 query 共享, 受 causal 约束)
```

**关键实现细节** (注意: 先处理驱逐, 再写 mask):

→ [L249-280](../src/model/linear_attention/linear_window_attention_lola.py#L249-L280): **chunk 前处理 — 驱逐 + 选择 + 吸收**
```python
# 1. 构建 eligible set: G ∪ evicted
eligible = is_in_G.clone()
eligible[:, :, evict_start:evict_end + 1] = True

# 2. Top-k 选择 + 批量吸收
sre_all = compute_sre(f_k, v, kv_state=H, k_state=s_state)
sre_all[~eligible] = -float('inf')
_, topk_idx = sre_all.topk(sparse_budget, dim=-1)  # (b, h, budget)
new_G = zeros_like(is_in_G)
new_G.scatter_(-1, topk_idx, True)

fallen = eligible & ~new_G
H += einsum('bhlf,bhld->bhfd', f_k * fallen_mask, v * fallen_mask)
is_in_G = new_G
```
- topk: 保留 SRE 最高的 budget 个 (最重要, 线性状态表示最差)
- fallen: eligible 但未被选中 → **批量吸收**进 H
- 与 serial 的区别: chunk 内多个 fallen token 同时被吸收, 不逐个更新 H

→ [L284-292](../src/model/linear_attention/linear_window_attention_lola.py#L284-L292): **chunk 内 mask 写入 (向量化, 用更新后的 G)**
```python
query_indices = arange(chunk_start, chunk_end)
bw_boundaries = query_indices + offset - window_size
bw_mask = key_positions[None, :] <= bw_boundaries[:, None]  # (chunk_len, k_len)
sparse_mask[:, :, chunk_start:chunk_end, :] = (
    is_in_G[:, :, None, :] & bw_mask[None, None, :, :])
```
- 无内部 Python 循环, 完全向量化
- `bw_mask`: 每个 query 的 below-window 边界不同 → 同一 G 中不同 query 看到不同子集
- 广播: `(b,h,1,k) & (1,1,chunk,k)` → `(b,h,chunk,k)`

**返回值**: `(b, h, q_len, k_len)` — 同 serial, 但 chunk 内 query 共享同一 G

### 审查要点 (chunked)
- [ ] chunk 内共享 G: 所有 query 用同一 is_in_G, 但 bw_mask 保证 causal ✓
- [ ] eviction 范围计算: `evict_start = max(sink_size, prev + 1)`, `evict_end = last_query + offset - ws` ✓
- [ ] 批量吸收: fallen 的 f_k 和 v 用 mask 过滤, 一次 einsum 完成 ✓
- [ ] 不可逆: fallen tokens 进入 H 后, new_G 中不再包含 → 后续 chunk 不可恢复 ✓
- [ ] 复杂度: O(num_chunks × k_len × f × d), 比 serial 快约 chunk_size 倍

### 3.3 Serial vs Chunked 对比

| 特性 | serial | chunked |
|------|--------|---------|
| G 粒度 | per-token G_t | per-chunk G_c |
| H 更新 | 逐 token | chunk 边界批量 |
| 精度 | 精确 (论文 Eq.12) | 近似 (论文 Section 3.3) |
| 复杂度 | O(n × budget × (f+d)) | O(n/C × k × (f+d)) |
| 可并行 | 否 (Python loop) | 是 (chunk 内向量化) |
| 默认 | 否 | **是** |
| 配置 | `sparse_selection: serial` | `sparse_selection: chunked` |

---

## 4. hybrid_attention_quadratic_lola

> 论文 Eq.14 的训练时实现 (quadratic, O(n²))

→ [linear_window_attention_lola.py:299-409](../src/model/linear_attention/linear_window_attention_lola.py#L299-L409)

### 4.1 Mask 生成

→ [L324](../src/model/linear_attention/linear_window_attention_lola.py#L324): 复用父类的 `get_masks`
```python
mask_window, mask_linear = get_masks(window_size, q_len, k_len, q.device)
```
- `mask_window`: `(1,1,q,k)`, window 内为 1 (softmax 注意力区域)
- `mask_linear`: `(1,1,q,k)`, below-window 为 1 (线性注意力区域)
- 两者互斥, 并集 = causal mask

→ [L329](../src/model/linear_attention/linear_window_attention_lola.py#L329): **Sink 只在首 chunk 激活**
```python
effective_sink_size = sink_size if (sink_size > 0 and kv_state is None) else 0
```
- `kv_state is None` → 首次 chunk 或全序列 (包含 sink token)
- `kv_state is not None` → 后续 chunk (无 sink)

### 4.2 Per-query sparse 选择 (委托给 selection 函数)

→ [L332-344](../src/model/linear_attention/linear_window_attention_lola.py#L332-L344): **调用选择函数**
```python
with torch.no_grad():
    if sparse_selection == 'serial':
        sparse_mask = select_sparse_tokens_serial(
            f_k, v, sparse_budget, window_size, q_len, k_len,
            sink_size=effective_sink_size, kv_state=kv_state, k_state=k_state)
    else:  # 'chunked' (default)
        chunk_size = sparse_chunk_size or window_size
        sparse_mask = select_sparse_tokens_chunked(
            f_k, v, sparse_budget, window_size, q_len, k_len,
            chunk_size=chunk_size, sink_size=effective_sink_size,
            kv_state=kv_state, k_state=k_state)
```
- **重大变更**: SRE 计算和 H 构建/演化现在完全封装在选择函数内部
- 返回 `sparse_mask (b, h, q_len, k_len)` — **每个 query 有独立的 sparse token 集合**
- 选择函数内部维护 H 的不可逆演化 (token 被吸收后永不返回)
- `kv_state`/`k_state` 传入用于 stateful training (先前 chunk 的累积状态)

### 4.3 Mask 后处理

→ [L348](../src/model/linear_attention/linear_window_attention_lola.py#L348): **因果性保证**
```python
sparse_mask = sparse_mask & mask_linear.bool()
```
- `(b,h,q,k) & (1,1,q,k)` → 广播为 `(b,h,q,k)`
- 保证 query i 只能看到其 below-window 区域中被选中的 sparse token

→ [L351-358](../src/model/linear_attention/linear_window_attention_lola.py#L351-L358): **Sink mask**
```python
if effective_sink_size > 0:
    sink_mask = zeros(1, 1, q_len, k_len)
    sink_mask[:, :, :, :effective_sink_size] = True
    causal = ones(q_len, k_len).tril(k_len - q_len)
    sink_mask = sink_mask & causal[None, None, ...]
    sparse_mask = sparse_mask | sink_mask
```
- sink token 通过 sparse softmax 路径参与注意力
- causal mask 保证 sink 只对 pos >= sink_size 的 query 可见

→ [L361](../src/model/linear_attention/linear_window_attention_lola.py#L361): **更新 linear mask**
```python
mask_linear_remaining = mask_linear.bool() & ~sparse_mask
```
- 从 linear 区域中去掉 sparse + sink token
- 确保每个 token 恰好属于三路之一: window / sparse / linear

### 审查要点
- [ ] 三个 mask 互斥: `mask_window`, `sparse_mask`, `mask_linear_remaining` 两两无交集 ✓
- [ ] sink token 走 sparse softmax 路径, 不走 linear → 不会被压缩到线性状态 ✓

### 4.4 三路注意力计算

→ [L364](../src/model/linear_attention/linear_window_attention_lola.py#L364): **共享 QK 矩阵**
```python
qk = einsum('bhmd,bhnd->bhmn', q.float(), k.float()) * (d ** -0.5)
```
- 一次计算, window 和 sparse 共用 (避免重复 O(n²d) matmul)

→ [L367-370](../src/model/linear_attention/linear_window_attention_lola.py#L367-L370): **窗口 softmax**
```python
a_sm = qk.masked_fill(~mask_window.bool(), mask_value)  # mask_value = -1e8
a_sm_max = amax(a_sm, dim=-1, keepdim=True)
a_sm = window_factor * exp(a_sm - a_sm_max)
sum_sm = a_sm.sum(dim=-1, keepdim=True)
```
- 标准 softmax 的 log-sum-exp 技巧, 减去 max 防止溢出
- 乘 `window_factor` (gating, 非论文内容)
- `sum_sm` 是归一化分母的窗口部分

→ [L373-384](../src/model/linear_attention/linear_window_attention_lola.py#L373-L384): **稀疏 softmax + 数值稳定**
```python
if sparse_mask.any():
    a_sp = qk.masked_fill(~sparse_mask, mask_value)
    a_sp_max = amax(a_sp, dim=-1, keepdim=True)
    a_sp = sparse_factor * exp(a_sp - a_sp_max)
    sum_sp = a_sp.sum(dim=-1, keepdim=True)

    # 统一 max
    max_all = maximum(a_sm_max, a_sp_max)
    a_sm = a_sm * exp(a_sm_max - max_all)
    a_sp = a_sp * exp(a_sp_max - max_all)
    sum_sm = a_sm.sum(dim=-1, keepdim=True)  # 重算
    sum_sp = a_sp.sum(dim=-1, keepdim=True)  # 重算
```
- **关键**: window 和 sparse 使用同一个 softmax kernel `exp(qk/√d)`, 论文要求联合归一化
- 两个 max 可能不同 → 统一到 `max_all` 后 rescale, 等效于: `exp(qk - max_all)` 对全局
- rescale 后 **必须重算 sum** (L383-384)

### 审查要点
- [ ] max 统一逻辑: `a_sm` 原本是 `w * exp(qk - a_sm_max)`, rescale 为 `w * exp(qk - max_all)` ← 正确, 因为 `exp(a_sm_max - max_all)` 补偿了 max 差异 ✓
- [ ] `sparse_mask.any()` 为 False 时 (无 sparse token, 短序列): a_sp=0, sum_sp=0 → 退化为标准 window + linear ✓
- [ ] window 和 sparse 的 mask 互斥 → qk 的同一个位置不会同时在两路中被计入 ✓

→ [L390-392](../src/model/linear_attention/linear_window_attention_lola.py#L390-L392): **线性注意力**
```python
a_ln = einsum('bhmd,bhnd->bhmn', f_q.float(), f_k.float())
a_ln = linear_factor * a_ln.masked_fill(~mask_linear_remaining, 0)
sum_ln = a_ln.sum(dim=-1, keepdim=True)
```
- feature-mapped attention: `φ(q)^T φ(k)`, 非 softmax
- 仅保留 `mask_linear_remaining` 区域 (排除 window + sparse + sink)

→ [L395-404](../src/model/linear_attention/linear_window_attention_lola.py#L395-L404): **联合归一化**
```python
a_combined = a_sm + a_sp + a_ln
y = einsum('bhmn,bhnd->bhmd', a_combined, v.float())

if kv_state is not None:  # stateful training: 加上先前 chunk 的线性状态
    y += linear_factor * einsum('bhld,bhdf->bhlf', f_q, kv_state)
    sum_ln += linear_factor * einsum('bhld,bhnd->bhl', f_q, k_state)[..., None]

y = (y / (sum_sm + sum_sp + sum_ln + eps)).to(q.dtype)
```
- 三路的分子直接相加, 除以联合分母 (论文 Eq.14) ✓
- `kv_state` 贡献: `φ(q)^T H_prev` 加到分子, `φ(q)^T s_prev` 加到分母

### 审查要点
- [ ] 联合归一化分母: `sum_sm + sum_sp + sum_ln` — 三路各自的 unnormalized weight 之和 ✓
- [ ] `sum_sm`, `sum_sp` 已经过 max 统一 rescale, 与 `a_sm`, `a_sp` 一致 ✓
- [ ] 线性项 `a_ln` 没有 exp, 直接是 `φ(q)^T φ(k)` → 与 softmax 项量级可能不同, 靠 gating factor 调节 ✓
- [ ] `eps` 防止分母为零 ✓

---

## 5. LolcatsLoLAAttention

→ [linear_window_attention_lola.py:415-572](../src/model/linear_attention/linear_window_attention_lola.py#L415-L572)

### 5.1 __init__

→ [L419-442](../src/model/linear_attention/linear_window_attention_lola.py#L419-L442)

```python
def __init__(self, sparse_budget=128, sink_size=0,
             init_sparse_factor=10.0, train_sparse_factor=True,
             layer_sparse_budget=None,
             sparse_selection='chunked', sparse_chunk_size=None,
             **kwargs):
    super().__init__(**kwargs)
    self.sparse_budget = layer_sparse_budget or sparse_budget
    self.sink_size = sink_size
    self.sparse_selection = sparse_selection      # 'serial' or 'chunked'
    self.sparse_chunk_size = sparse_chunk_size    # None → use window_size
    self.sparse_factors = nn.Parameter(
        init_sparse_factor * ones(1, num_heads, 1, 1))
```

- `layer_sparse_budget`: 跨层 budget 分配时由 `convert_model.py` 注入
- `sparse_factors` shape `(1, num_heads, 1, 1)`: per-head 可学习
- `sparse_selection`: 选择算法, `'chunked'` (默认高效) 或 `'serial'` (精确参考)
- `sparse_chunk_size`: chunked 模式的 chunk 大小, None 时使用 window_size

### 审查要点
- [ ] 继承链: `LolcatsLoLAAttention → LolcatsSlidingWindowAttention → LolcatsLinearAttention → nn.Module` ✓
- [ ] `super().__init__(**kwargs)` 接收 `window_size`, `feature_map`, `learned_kernel`, `base_attn` 等 ✓
- [ ] `self.eps = 1e-12` 在 `LolcatsLinearAttention.__init__` 中设置, LoLA 继承 ✓

### 5.2 Forward 路径 1 — 蒸馏

→ [L470-486](../src/model/linear_attention/linear_window_attention_lola.py#L470-L486)

```python
if self.train_attention:
    with torch.no_grad():
        _y_true, a_true = softmax_attention(q, k, v)[:2]     # Ground truth
        y_true = _y_true.transpose(1,2).view(b, l, hidden_size)
        y_true = self.o_proj(y_true)

    y_pred, a_pred = hybrid_attention_quadratic_lola(...)     # Predicted
    attn_weights = ((a_pred, a_true), (y_pred, _y_true))
```

- Ground truth: 标准 softmax attention (causal)
- Predicted: 三层混合注意力
- 返回格式 `((a_pred, a_true), (y_pred, _y_true))` → trainer 用 `MSE(attns[1][0], attns[1][1])` 计算 loss

### 审查要点
- [ ] `_y_true` shape `(b,h,l,d)` → transpose + view → `(b,l,hidden_size)` ✓
- [ ] `y_pred` shape `(b,h,l,d)`, `_y_true` shape `(b,h,l,d)` — 二者在 attention head 空间对齐 ✓
- [ ] o_proj 仅应用于 y_true (ground truth 路径), y_pred 未经 o_proj (由 trainer 处理 MSE) ✓

### 5.3 Forward 路径 2 — 常规训练

→ [L489-500](../src/model/linear_attention/linear_window_attention_lola.py#L489-L500)

```python
if past_key_value is None:
    y_true, a_pred = hybrid_attention_quadratic_lola(
        q, k, f_q, f_k, v, window_factors, linear_factors, sparse_factors,
        window_size=self.window_size, sparse_budget=self.sparse_budget,
        sink_size=self.sink_size)
    attn_weights = a_pred
```

- 无 cache → 全序列注意力 (LoRA finetuning 阶段)
- 三层混合注意力在 forward 中活跃 → 训推对齐

### 5.4 Forward 路径 3 — 解码

→ [L506-544](../src/model/linear_attention/linear_window_attention_lola.py#L506-L544)

```python
if f_q.shape[2] == 1 and kv_seq_len > 1 and not self.training:
    _kv = past_key_value.update_for_decoding(k, v, self.layer_idx,
                                              self.feature_map_k, dtype=q.dtype)
    k_cache, v_cache, f_kv_state, f_k_state, sparse_k, sparse_v = _kv
```

解码时注意力计算:

→ [L515-518](../src/model/linear_attention/linear_window_attention_lola.py#L515-L518): **窗口 softmax** — `q @ k_cache`
→ [L521-531](../src/model/linear_attention/linear_window_attention_lola.py#L521-L531): **稀疏 softmax** — `q @ sparse_k` (含 sink)
→ [L335-339](../src/model/linear_attention/linear_window_attention_lola.py#L335-L339): **max 统一** — 与训练路径相同
→ [L345-352](../src/model/linear_attention/linear_window_attention_lola.py#L345-L352): **联合归一化**

```python
y = a_sm @ v_cache + a_sp @ sparse_v + linear_factors * f_q @ f_kv_state
sum_ln = linear_factors * f_q @ f_k_state
y = y / (sum_sm + sum_sp + sum_ln + self.eps)
```

### 审查要点
- [ ] 解码用 `self.eps` (= 1e-12), 与训练路径的 `eps=1e-12` 参数一致 ✓
- [ ] `sparse_k` 已含 sink token (Cache.update_for_decoding 负责拼接) ✓
- [ ] 线性状态 `f_kv_state`, `f_k_state` 来自 `decode_kv_states` (不含 window 内 token) ✓
- [ ] max 统一逻辑与训练路径 (L195-200) 完全一致 ✓

### 5.5 Forward 路径 4 — 状态训练 (分块)

→ [L354-372](../src/model/linear_attention/linear_window_attention_lola.py#L354-L372)

```python
else:  # stateful training
    try:
        kv_state = past_key_value.kv_states[self.layer_idx]
        k_state  = past_key_value.k_states[self.layer_idx]
    except IndexError:
        kv_state, k_state = None, None
    y_true, _ = hybrid_attention_quadratic_lola(
        ..., kv_state=kv_state, k_state=k_state)
    past_key_value.update(k, v, self.layer_idx, fmap_key_states=f_k, ...)
```

- 从 cache 取出先前 chunk 的累积状态
- `IndexError` → 首个 chunk, 无先前状态
- `hybrid_attention_quadratic_lola` 中 `kv_state is not None` → 加入先前线性状态
- 更新 cache (为下一 chunk 准备)

### 审查要点
- [ ] cache 参数同步: L310-313 在进入 decode/stateful 前设置 `past_key_value.sparse_budget = self.sparse_budget` ✓
- [ ] 输出投影 L375-376 在四条路径共用 (路径 1 例外, 在内部已处理) ✓

---

## 6. LinearAttentionLoLACache

→ [linear_window_attention_lola.py:384-617](../src/model/linear_attention/linear_window_attention_lola.py#L384-L617)

### 数据结构

```python
# 继承自 LinearAttentionSlidingWindowCache:
kv_states:        List[Tensor]  # (b,h,f,d) 全量线性状态 (含 window)
k_states:         List[Tensor]  # (b,h,1,f) 全量 key 状态
decode_kv_states: List[Tensor]  # (b,h,f,d) 解码用线性状态 (不含 window)
decode_k_states:  List[Tensor]  # (b,h,1,f) 解码用 key 状态
k_cache:          List[Tensor]  # (b,h,w,d) 滑动窗口 key
v_cache:          List[Tensor]  # (b,h,w,d) 滑动窗口 value

# LoLA 新增:
sparse_k_cache:   List[Tensor]  # (b,h,s,d) 稀疏缓存 key (s ≤ sparse_budget)
sparse_v_cache:   List[Tensor]  # (b,h,s,d) 稀疏缓存 value
sparse_sre:       List[Tensor]  # (b,h,s)   每个 sparse token 的 SRE 值
sink_k_cache:     List[Tensor]  # (b,h,sink_size,d) sink token key (永不驱逐)
sink_v_cache:     List[Tensor]  # (b,h,sink_size,d) sink token value
```

每个 List 按 layer_idx 索引, 每层一个 Tensor。

### 6.1 Cache.update

> 训练路径 (stateful training / chunked training)

→ [linear_window_attention_lola.py:402-485](../src/model/linear_attention/linear_window_attention_lola.py#L402-L485)

**流程**:

```
输入 chunk: [sink | pre-window | window]  (首 chunk)
         或 [pre-window | window]         (后续 chunk)

1. 首 chunk: 保存 sink token 到 sink_k/v_cache
2. 确定 start (首 chunk 排除 sink, 后续 chunk start=0)
3. pre-window → decode_kv_state (累积到线性状态)
4. window → k_cache/v_cache (覆盖旧 window)
5. 全量 kv_state = decode_kv_state + window 部分 (给下一 chunk 用)
```

→ [L424](../src/model/linear_attention/linear_window_attention_lola.py#L424): **首 chunk 判断**
```python
is_first_chunk = len(self.k_states) <= layer_idx
```

→ [L427-429](../src/model/linear_attention/linear_window_attention_lola.py#L427-L429): **Sink 初始化 (仅首次)**
```python
if self.sink_size > 0 and len(self.sink_k_cache) <= layer_idx:
    self.sink_k_cache.append(key_states[:, :, :self.sink_size, :])
    self.sink_v_cache.append(value_states[:, :, :self.sink_size, :].to(dtype))
```

→ [L432](../src/model/linear_attention/linear_window_attention_lola.py#L432): **排除 sink**
```python
start = self.sink_size if (self.sink_size > 0 and is_first_chunk) else 0
```
- 首 chunk: sink token 已单独保存, 从 linear 状态排除
- 后续 chunk: 无 sink, start=0

→ [L438-448](../src/model/linear_attention/linear_window_attention_lola.py#L438-L448): **分割 pre-window 和 window**
```python
if non_sink_len > self.window_size:
    fk_pre = fmap_key_states[:, :, start:seq_len - window_size]  # pre-window
    fk_win = fmap_key_states[:, :, seq_len - window_size:]       # window
else:
    fk_pre = fmap_key_states[:, :, :0]  # empty
    fk_win = fmap_key_states[:, :, start:]
```

→ [L450-453](../src/model/linear_attention/linear_window_attention_lola.py#L450-L453): **构建线性状态**
```python
decode_kv_state = einsum('bhlf,bhld->bhfd', fk_pre, v_pre)
kv_state = decode_kv_state + einsum('bhlf,bhld->bhfd', fk_win, v_win)
```
- `decode_kv_state`: 仅 pre-window (解码用, 不含 window)
- `kv_state`: 全量 (pre-window + window, 给下一 chunk 的 SRE 用)

### 审查要点
- [ ] `start` 仅在首 chunk 非零 → 后续 chunk 不会误跳 sink 位置 ✓
- [ ] 短序列 (`non_sink_len ≤ window_size`): `fk_pre` 为空, 所有 token 作为 window ✓
- [ ] `decode_kv_state` 不含 window 内 token → 解码时 window softmax + linear state 互斥 ✓
- [ ] 稀疏缓存在首 chunk 初始化为空 `(b,h,0,d)` → 解码开始前无 sparse token ✓

### 6.2 Cache.update_for_decoding

> 解码路径: 逐 token 生成, SRE 驱动的缓存管理

→ [linear_window_attention_lola.py:487-617](../src/model/linear_attention/linear_window_attention_lola.py#L487-L617)

**完整流程图**:

```
新 token (k, v) 进入
  │
  ├─ window 未满 (k_cache.shape < window_size)
  │    └─ 直接 append 到 window → 返回
  │
  └─ window 已满
       │
       ├─ 驱逐 k_cache[:,:,:1,:] (最旧 token)
       ├─ 计算 evicted_sre = SRE(evicted | H_t)
       │
       ├─ sparse cache 未满 (< budget)
       │    └─ 直接 append evicted 到 sparse cache → 移动窗口
       │
       └─ sparse cache 已满
            │
            ├─ 用当前 H_t 重新评分所有 sparse entries   ← 对齐论文 Eq.12
            ├─ min_sre = sparse_sre 中最小值
            │
            ├─ evicted_sre > min_sre (部分 head)
            │    ├─ displaced = sparse cache 中 min_sre 位置的 token
            │    ├─ displaced → 进入线性状态 (decode_kv_states += φ(k)v^T)
            │    ├─ evicted → 替换 displaced 在 sparse cache 中的位置
            │    └─ 未替换的 head: evicted → 进入线性状态
            │
            └─ evicted_sre ≤ min_sre (所有 head)
                 └─ evicted → 进入线性状态
```

#### 详细逐行

→ [L508-511](../src/model/linear_attention/linear_window_attention_lola.py#L508-L511): **窗口未满**
```python
if k_cache.shape[-2] < self.window_size:
    self.k_cache[layer_idx] = cat([k_cache, keys], dim=-2)
    self.v_cache[layer_idx] = cat([v_cache, values], dim=-2)
```
- prefill 后首几步, window 逐步填满

→ [L514-515](../src/model/linear_attention/linear_window_attention_lola.py#L514-L515): **驱逐最旧 token**
```python
evicted_k = k_cache[:, :, :1, :]  # (b, h, 1, d)
evicted_v = v_cache[:, :, :1, :]
```

→ [L518-526](../src/model/linear_attention/linear_window_attention_lola.py#L518-L526): **计算 evicted token 的 SRE**
```python
f_k_evicted = feature_map_k(evicted_k)          # φ(k_evicted)
kv_state = self.decode_kv_states[layer_idx]      # H_t
k_state_acc = self.decode_k_states[layer_idx]    # s_t
numerator   = einsum('bhfd,bhlf->bhld', kv_state, f_k_evicted)
denominator = einsum('bhnf,bhlf->bhl',  k_state_acc, f_k_evicted)
v_hat = numerator / (denominator[..., None] + 1e-12)
evicted_sre = ((evicted_v - v_hat) ** 2).sum(dim=-1)  # (b, h, 1)
```
- 用当前 H_t 计算: "如果 evicted token 被压缩到线性状态, 它会损失多少信息?"

→ [L532-536](../src/model/linear_attention/linear_window_attention_lola.py#L532-L536): **稀疏缓存未满 — 直接加入**
```python
if sparse_k.shape[-2] < self.sparse_budget:
    self.sparse_k_cache[layer_idx] = cat([sparse_k, evicted_k], dim=-2)
    self.sparse_v_cache[layer_idx] = cat([sparse_v, evicted_v.to(dtype)], dim=-2)
    self.sparse_sre[layer_idx] = cat([sparse_sre, evicted_sre], dim=-1)
```

→ [L538-544](../src/model/linear_attention/linear_window_attention_lola.py#L538-L544): **⭐ 重新评分 (对齐论文 Eq.12)**
```python
# 稀疏缓存已满 — 用当前 H_t 重新评分所有 sparse cache entries
f_k_sparse = feature_map_k(sparse_k)                                    # φ(k) for all cached
num_sp = einsum('bhfd,bhlf->bhld', kv_state, f_k_sparse)               # H @ φ(k_i)
den_sp = einsum('bhnf,bhlf->bhl',  k_state_acc, f_k_sparse)            # s^T @ φ(k_i)
v_hat_sp = num_sp / (den_sp[..., None] + 1e-12)
sparse_sre = ((sparse_v - v_hat_sp) ** 2).sum(dim=-1)                  # 重算 SRE
self.sparse_sre[layer_idx] = sparse_sre
```
- **关键修复**: 论文要求每步用当前 H_t 重新评分, 而非使用旧 SRE 值
- 代价: `feature_map_k(sparse_k)` = O(budget · f), einsum = O(budget · f · d)
- budget=128 时, 远小于 attention 计算量, 可接受

→ [L547-548](../src/model/linear_attention/linear_window_attention_lola.py#L547-L548): **比较决策**
```python
min_sre, min_idx = sparse_sre.min(dim=-1, keepdim=True)  # (b, h, 1)
should_replace = (evicted_sre > min_sre)                   # (b, h, 1)
```
- per-head 独立决策 ✓

→ [L550-591](../src/model/linear_attention/linear_window_attention_lola.py#L550-L591): **替换逻辑 (部分 head 替换)**
```python
if should_replace.any():
    # displaced = sparse cache 中 SRE 最小的 token
    displaced_k = gather(sparse_k, 2, min_idx_expanded_d)
    displaced_v = gather(sparse_v, 2, min_idx_expanded_d)

    # displaced → 进入线性状态
    f_displaced_k = feature_map_k(displaced_k)
    decode_kv_states += φ(displaced_k) @ displaced_v^T  * replace_mask

    # evicted → 替换 displaced 在 sparse cache 中的位置
    sparse_k_cache = scatter(sparse_k, ..., where(replace_mask, evicted_k, displaced_k))
    sparse_v_cache = scatter(sparse_v, ..., where(replace_mask, evicted_v, displaced_v))
    sparse_sre     = scatter(sparse_sre, ..., where(replace_mask, evicted_sre, min_sre))

    # 未替换的 head: evicted → 进入线性状态
    decode_kv_states += φ(evicted_k) @ evicted_v^T  * not_replace_mask
```

### 审查要点
- [ ] `replace_mask` shape `(b,h,1,1)` — 广播到 `(b,h,f,d)` 用于 kv_state 更新 ✓
- [ ] `scatter` 用 `torch.where` 确保只在 `should_replace` 的 head 上替换, 其余保持原值 ✓
- [ ] displaced token 进入线性状态前先 `feature_map_k()` 转换 → 正确, 线性状态存的是 φ(k)v^T ✓
- [ ] `not_replace_mask` 分支: evicted token 也用 `f_k_evicted` (已在 L518 计算), 无冗余调用 ✓
- [ ] `else` 分支 (L592-598): 全部 evicted 进入线性状态, 也用 `f_k_evicted` ✓

→ [L601-602](../src/model/linear_attention/linear_window_attention_lola.py#L601-L602): **移动窗口**
```python
self.k_cache[layer_idx] = cat([k_cache[:, :, 1:, :], keys], dim=-2)
self.v_cache[layer_idx] = cat([v_cache[:, :, 1:, :], values], dim=-2)
```
- 去掉最旧 (已驱逐), 加入新 token

→ [L608-617](../src/model/linear_attention/linear_window_attention_lola.py#L608-L617): **返回值 — 拼接 sink + sparse**
```python
sparse_k_out = self.sparse_k_cache[layer_idx]
sparse_v_out = self.sparse_v_cache[layer_idx]
if self.sink_size > 0 and len(self.sink_k_cache) > layer_idx:
    sparse_k_out = cat([self.sink_k_cache[layer_idx], sparse_k_out], dim=-2)
    sparse_v_out = cat([self.sink_v_cache[layer_idx], sparse_v_out], dim=-2)
return (k_cache, v_cache, decode_kv_states, decode_k_states, sparse_k_out, sparse_v_out)
```
- sink + sparse 拼接后统一作为 "sparse softmax" 路径的输入
- forward 路径 3 中 `a_sp = q @ sparse_k` 同时覆盖 sink 和 sparse token

---

## 7. convert_model.py

### 7.1 compute_layer_budgets

→ [convert_model.py:10-54](../src/model/convert_model.py#L10-L54)

| 策略 | 算法 | 示例 (32层, total=4096) |
|------|------|------------------------|
| `uniform` | `total // num_layers`, 余数分前几层 | 每层 128 |
| `pyramid` | `weight = 1 + |i - mid| / mid`, 按权重分配 | 首尾 ~170, 中间 ~90 |
| `list` | 直接指定 | 用户自定义 |

### 7.2 注册 LoLA

→ [convert_model.py:196-198](../src/model/convert_model.py#L196-L198): **attention class**
```python
elif attention_type == 'lolcats_llama_window_lola':
    from .linear_attention import LolcatsLoLAAttention
    return partial(LolcatsLoLAAttention, **kwargs)
```

→ [convert_model.py:221-223](../src/model/convert_model.py#L221-L223): **cache class**
```python
elif 'lola' in attention_type:
    from .linear_attention import LinearAttentionLoLACache
    return LinearAttentionLoLACache()
```
- 注意: cache 用默认参数创建, `sparse_budget` 和 `sink_size` 在 forward 中同步 (L311-313)

### 7.3 Per-layer budget 注入

→ [convert_model.py:137-146](../src/model/convert_model.py#L137-L146)
```python
if 'budget_strategy' in config:
    budgets = compute_layer_budgets(total, num_layers, strategy, ...)
    config['layer_sparse_budget'] = budgets[layer.self_attn.layer_idx]
```
- 在 `convert_llama_attention()` 中, 创建 attention 前注入 `layer_sparse_budget`
- `LolcatsLoLAAttention.__init__` 中: `self.sparse_budget = layer_sparse_budget or sparse_budget`

---

## 8. 论文公式 vs 实现对照表

| 论文公式 | 代码位置 | 实现方式 | 是否对齐 |
|---------|---------|---------|---------|
| Eq.10: SRE(k,v\|H,s) = \|\|v - H^T φ(k) / s^T φ(k)\|\|² | [compute_sre L31-64](../src/model/linear_attention/linear_window_attention_lola.py#L31-L64) | einsum 计算 v_hat, L2 距离平方 | ✅ |
| Eq.12: G_t = argmax top-λ by SRE, per-query | serial: [select_sparse_tokens_serial L74-174](../src/model/linear_attention/linear_window_attention_lola.py#L74-L174); chunked: [select_sparse_tokens_chunked L180-294](../src/model/linear_attention/linear_window_attention_lola.py#L180-L294); 解码: [update_for_decoding](../src/model/linear_attention/linear_window_attention_lola.py#L538-L544) | serial: per-token exact; chunked: per-chunk; 解码: 每步重评分 | ✅ |
| Eq.12: H_t 仅含已淘汰 token, 不可逆 | serial: [L151-155](../src/model/linear_attention/linear_window_attention_lola.py#L151-L155); chunked: [L237-243](../src/model/linear_attention/linear_window_attention_lola.py#L237-L243) | H 在选择函数内部逐步演化, absorbed token 永不返回 | ✅ |
| Eq.13: S_t = E_t \ G_t 进入线性状态 | [update_for_decoding L566-598](../src/model/linear_attention/linear_window_attention_lola.py#L566-L598) | displaced/evicted → decode_kv_states += φ(k)v^T | ✅ |
| Eq.14: 三路联合归一化 | [hybrid_attention L311-321](../src/model/linear_attention/linear_window_attention_lola.py#L311-L321) (训练) / [forward L408-416](../src/model/linear_attention/linear_window_attention_lola.py#L408-L416) (解码) | `y / (sum_sm + sum_sp + sum_ln + eps)` | ✅ |
| Eq.15: per-query G_t (serial) / per-chunk G_c (chunkwise) | serial: [L158-160](../src/model/linear_attention/linear_window_attention_lola.py#L158-L160); chunked: [L210-214](../src/model/linear_attention/linear_window_attention_lola.py#L210-L214) | serial: 每 query 独立 G; chunked: chunk 内共享 G | ✅ |
| Per-head 独立 sparse 选择 | serial: [L145 min per head](../src/model/linear_attention/linear_window_attention_lola.py#L145); chunked: [L233 topk per head](../src/model/linear_attention/linear_window_attention_lola.py#L233) | 各 head SRE 独立 → G 不同 | ✅ |
| Window + sparse 同一 softmax kernel | [hybrid_attention L301](../src/model/linear_attention/linear_window_attention_lola.py#L301) / [L307](../src/model/linear_attention/linear_window_attention_lola.py#L307) | 共享 `qk` 矩阵, 不同 mask | ✅ |
| 可学习 gating factors | [LolcatsLoLAAttention L349-351](../src/model/linear_attention/linear_window_attention_lola.py#L349-L351) | window_factor, sparse_factor, linear_factor | ⚠️ 非论文 |

---

## 9. 已知设计偏差

### 9.1 ~~训练时所有 query 共享同一组 sparse token~~ → **已修复**

- **论文**: 每个 query 位置 t 有自己的 G_t (随时间变化)
- **旧实现**: `select_sparse_tokens` 返回 `(b,h,1,k)`, 所有 query 共享
- **新实现**: 提供两版 per-query 选择:
  - `select_sparse_tokens_serial`: per-token exact G_t (论文 Eq.12)
  - `select_sparse_tokens_chunked`: per-chunk G_c (论文 Section 3.3), 默认
- 返回 `(b, h, q_len, k_len)`, 每个 query 有独立的 sparse mask
- H 不可逆演化: 被吸收 token 永不返回 sparse cache

### 9.2 Stateful training 的线性状态包含 sparse 应该缓存的 token

- **论文**: H_t 仅含被淘汰 token (S_t), 不含 G_t 中的 token
- **实现**: `Cache.update` 将所有 pre-window token 累积到 kv_state, 不区分 sparse vs linear
- **影响**: 中等。约 12% token (budget/seq_len) 被"多算"进线性状态
- **位置**: [L450](../src/model/linear_attention/linear_window_attention_lola.py#L450)
- **改进思路**: 在 Cache.update 中也做 SRE 选择, 将 sparse 部分单独存储

### 9.3 Gating factors (非论文)

- **论文**: 三路权重均为 1, 直接联合归一化
- **实现**: `window_factor` (sigmoid, ~0.1), `sparse_factor` (sigmoid, ~1.0), `linear_factor` (1 或 1-window_factor)
- **影响**: 无负面影响, 是合理的工程扩展。蒸馏可学习最优权重
- **位置**: [L278-280](../src/model/linear_attention/linear_window_attention_lola.py#L278-L280), [L185](../src/model/linear_attention/linear_window_attention_lola.py#L185), [L192](../src/model/linear_attention/linear_window_attention_lola.py#L192), [L207](../src/model/linear_attention/linear_window_attention_lola.py#L207)
