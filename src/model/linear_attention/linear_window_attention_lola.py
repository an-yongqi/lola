"""
LoLA: LoLCATs + Sparse Cache via Self-Recall Error (SRE)

三层混合注意力:
1. 滑动窗口 (softmax): 最近 window_size 个 token
2. 稀疏缓存 (softmax): SRE 选出的重要 token
3. 线性状态 (linear): 其余 token 压缩到 recurrent state

基于 linear_window_attention_sw.py 开发

Sparse selection modes:
- 'serial': per-token exact selection (reference, paper Eq.12-13)
- 'chunked': per-chunk selection (efficient, paper Section 3.3)
"""
from typing import List, Tuple, Optional, Callable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import Cache

from .linear_attention import (
    LolcatsLinearAttention, LinearAttentionState,
    softmax_attention
)
from .linear_window_attention_sw import (
    get_masks, LolcatsSlidingWindowAttention, LinearAttentionSlidingWindowCache
)


# ----------------------
# SRE computation
# ----------------------
def compute_sre(f_k: torch.Tensor, v: torch.Tensor,
                kv_state: torch.Tensor = None,
                k_state: torch.Tensor = None,
                eps: float = 1e-12) -> torch.Tensor:
    """
    计算 Self-Recall Error (SRE)
    SRE(i) = ||v_i - H @ φ(k_i) / (s^T @ φ(k_i))||²

    Args:
        f_k: feature-mapped keys, shape (b, h, l, f)
        v: values, shape (b, h, l, d)
        kv_state: 累积线性状态 H, shape (b, h, f, d). None 时从 f_k, v 计算全局状态
        k_state: 累积 key 状态 s, shape (b, h, 1, f). None 时从 f_k 计算
        eps: 数值稳定性

    Returns:
        sre: shape (b, h, l), 每个 token 的 SRE 值
    """
    if kv_state is None:
        # 全局近似: 用所有 token 构建线性状态
        H = torch.einsum('bhlf,bhld->bhfd', f_k.float(), v.float())
        s = f_k.float().sum(dim=2, keepdim=True)  # b, h, 1, f
    else:
        H = kv_state.float()
        s = k_state.float()

    # 重建 value: v_hat_i = H @ φ(k_i) / (s^T @ φ(k_i))
    numerator = torch.einsum('bhfd,bhlf->bhld', H, f_k.float())   # b, h, l, d
    denominator = torch.einsum('bhnf,bhlf->bhl', s, f_k.float())  # b, h, l
    v_hat = numerator / (denominator[..., None] + eps)

    # SRE = ||v - v_hat||^2
    sre = ((v.float() - v_hat) ** 2).sum(dim=-1)  # b, h, l
    return sre


# ----------------------
# Per-token serial sparse selection (exact reference)
# ----------------------
def select_sparse_tokens_serial(f_k: torch.Tensor, v: torch.Tensor,
                                sparse_budget: int, window_size: int,
                                q_len: int, k_len: int,
                                sink_size: int = 0, eps: float = 1e-12,
                                kv_state: torch.Tensor = None,
                                k_state: torch.Tensor = None) -> torch.Tensor:
    """
    Per-token serial sparse selection (exact reference implementation).
    Paper Eq.12-13, 15: sequential processing with irreversible absorption.

    For each query position t, as keys exit the sliding window:
    1. Evicted key joins eligible set: E_t = G_{t-1} ∪ {evicted}
    2. If |E_t| > sparse_budget: compute SRE with current H_t,
       absorb min-SRE token into H_t (irreversible)
    3. G_t = E_t \\ {absorbed}

    Key property: once a token is absorbed into H, it NEVER returns to sparse cache.

    Args:
        f_k: feature-mapped keys (b, h, k_len, f)
        v: values (b, h, k_len, d)
        sparse_budget: max sparse cache size per head
        window_size: sliding window size
        q_len, k_len: query and key sequence lengths
        sink_size: permanent sink tokens excluded from selection
        eps: numerical stability
        kv_state: pre-accumulated linear state H (b, h, f, d), for stateful training
        k_state: pre-accumulated key state s (b, h, 1, f), for stateful training

    Returns:
        sparse_mask: (b, h, q_len, k_len) per-query sparse token positions
    """
    b, h, _, f_dim = f_k.shape
    d = v.shape[-1]
    device = f_k.device
    offset = k_len - q_len

    # Initialize linear state
    if kv_state is not None:
        H = kv_state.float().clone()
        s = k_state.float().clone()
    else:
        H = torch.zeros(b, h, f_dim, d, device=device, dtype=torch.float32)
        s = torch.zeros(b, h, 1, f_dim, device=device, dtype=torch.float32)

    # Track which key positions are currently in sparse cache G, per (b, h)
    is_in_G = torch.zeros(b, h, k_len, device=device, dtype=torch.bool)

    # Output mask
    sparse_mask = torch.zeros(b, h, q_len, k_len, device=device, dtype=torch.bool)

    # Next key position to be evicted from window
    # Positions < sink_size are permanent sinks, never enter sparse/linear
    next_evict = sink_size

    for t in range(q_len):
        # Below-window boundary for query t: keys at positions [0, bw_boundary] are below window
        # From get_masks: linear_mask[t, j] = 1 iff j <= t + offset - window_size
        bw_boundary = t + offset - window_size

        # Process newly evicted keys (one per query step typically)
        while next_evict <= bw_boundary and next_evict < k_len:
            p = next_evict
            next_evict += 1

            # Check if adding p would exceed budget
            # Note: n_in_G is the same for all (b,h) due to synchronized eviction timing
            n_in_G = int(is_in_G[0, 0].sum().item())

            if n_in_G >= sparse_budget:
                # |E| = budget + 1 after adding p → must absorb one token
                is_in_G[:, :, p] = True  # temporarily add p to eligible set

                # Compute SRE for all eligible positions using current H
                sre_all = compute_sre(f_k, v, kv_state=H, k_state=s, eps=eps)  # (b, h, k_len)
                sre_all[~is_in_G] = float('inf')  # non-eligible → won't be selected as min

                # Absorb the token with minimum SRE (least important)
                _, absorb_pos = sre_all.min(dim=-1)  # (b, h) — position to absorb per head

                # Update linear state H with absorbed token
                pos_fk = absorb_pos[:, :, None, None].expand(-1, -1, 1, f_dim)
                pos_v = absorb_pos[:, :, None, None].expand(-1, -1, 1, d)
                absorbed_fk = torch.gather(f_k.float(), 2, pos_fk)  # (b, h, 1, f)
                absorbed_v = torch.gather(v.float(), 2, pos_v)      # (b, h, 1, d)
                H = H + torch.einsum('bhlf,bhld->bhfd', absorbed_fk, absorbed_v)
                s = s + absorbed_fk  # broadcast: (b, h, 1, f)

                # Remove absorbed token from G
                is_in_G.scatter_(-1, absorb_pos.unsqueeze(-1), False)
            else:
                # Room in cache, just add
                is_in_G[:, :, p] = True

        # Record current G for query t
        # Only include positions that are actually below window for this query
        if bw_boundary >= 0:
            end = min(bw_boundary + 1, k_len)
            sparse_mask[:, :, t, :end] = is_in_G[:, :, :end]

    return sparse_mask


# ----------------------
# Chunk-wise sparse selection (efficient)
# ----------------------
def select_sparse_tokens_chunked(f_k: torch.Tensor, v: torch.Tensor,
                                 sparse_budget: int, window_size: int,
                                 q_len: int, k_len: int,
                                 chunk_size: int,
                                 sink_size: int = 0, eps: float = 1e-12,
                                 kv_state: torch.Tensor = None,
                                 k_state: torch.Tensor = None) -> torch.Tensor:
    """
    Chunk-wise sparse selection (efficient for training and inference).
    Paper Section 3.3: all queries within a chunk share the same sparse cache G_c.

    At chunk boundaries:
    1. Newly evicted keys (exited window during chunk) join eligible set:
       E_c = G_{c-1} ∪ {evicted keys from chunk c}
    2. If |E_c| > sparse_budget: compute SRE, keep top-budget, absorb rest into H
    3. G_c used for all queries in next chunk

    Key property: once a token is absorbed into H, it NEVER returns to sparse cache.
    Chunk dependencies are sequential — G and H evolve at chunk boundaries.

    Args:
        f_k: feature-mapped keys (b, h, k_len, f)
        v: values (b, h, k_len, d)
        sparse_budget: max sparse cache size per head
        window_size: sliding window size
        q_len, k_len: query and key sequence lengths
        chunk_size: processing chunk size (default: window_size)
        sink_size: permanent sink tokens excluded from selection
        eps: numerical stability
        kv_state: pre-accumulated linear state H (b, h, f, d), for stateful training
        k_state: pre-accumulated key state s (b, h, 1, f), for stateful training

    Returns:
        sparse_mask: (b, h, q_len, k_len) per-query sparse token positions
    """
    b, h, _, f_dim = f_k.shape
    d = v.shape[-1]
    device = f_k.device
    offset = k_len - q_len

    # Initialize linear state
    if kv_state is not None:
        H = kv_state.float().clone()
        s_state = k_state.float().clone()
    else:
        H = torch.zeros(b, h, f_dim, d, device=device, dtype=torch.float32)
        s_state = torch.zeros(b, h, 1, f_dim, device=device, dtype=torch.float32)

    is_in_G = torch.zeros(b, h, k_len, device=device, dtype=torch.bool)
    sparse_mask = torch.zeros(b, h, q_len, k_len, device=device, dtype=torch.bool)

    # Last processed eviction boundary
    prev_evict_end = sink_size - 1

    num_chunks = (q_len + chunk_size - 1) // chunk_size
    key_positions = torch.arange(k_len, device=device)

    for c in range(num_chunks):
        chunk_start = c * chunk_size
        chunk_end = min((c + 1) * chunk_size, q_len)

        # --- Process evictions FIRST → update G before writing mask ---
        # This ensures queries in this chunk see the up-to-date sparse set.
        last_query = chunk_end - 1
        curr_evict_end = last_query + offset - window_size

        evict_start = max(sink_size, prev_evict_end + 1)
        evict_end = min(curr_evict_end, k_len - 1)

        if evict_end >= evict_start:
            # Build eligible set: current G + newly evicted tokens
            eligible = is_in_G.clone()
            eligible[:, :, evict_start:evict_end + 1] = True

            # n_eligible is the same for all (b,h) due to synchronized eviction
            n_eligible = int(eligible[0, 0].sum().item())

            if n_eligible > sparse_budget:
                # Compute SRE for eligible positions
                sre_all = compute_sre(f_k, v, kv_state=H, k_state=s_state, eps=eps)
                sre_all[~eligible] = -float('inf')  # non-eligible → won't be in top-k

                # Select top-budget (highest SRE → most important → keep)
                _, topk_idx = sre_all.topk(sparse_budget, dim=-1)  # (b, h, budget)

                # New G: only the selected positions
                new_G = torch.zeros_like(is_in_G)
                new_G.scatter_(-1, topk_idx, True)

                # Fallen tokens: eligible but not selected → absorb into H
                fallen = eligible & ~new_G
                fallen_mask = fallen.float().unsqueeze(-1)  # (b, h, k_len, 1)
                fallen_fk = f_k.float() * fallen_mask
                fallen_v = v.float() * fallen_mask
                H = H + torch.einsum('bhlf,bhld->bhfd', fallen_fk, fallen_v)
                s_state = s_state + fallen_fk.sum(dim=2, keepdim=True)

                is_in_G = new_G
            else:
                # All eligible fit within budget
                is_in_G = eligible

        prev_evict_end = curr_evict_end

        # --- THEN write updated G to sparse_mask for all queries in this chunk ---
        # For query t: sparse position j is valid if j is in G AND j <= t + offset - window_size
        query_indices = torch.arange(chunk_start, chunk_end, device=device)
        bw_boundaries = query_indices + offset - window_size  # (chunk_len,)
        # Vectorized: (chunk_len, k_len) mask of below-window positions per query
        bw_mask = key_positions[None, :] <= bw_boundaries[:, None]
        # Combine with current G: (b, h, chunk_len, k_len)
        sparse_mask[:, :, chunk_start:chunk_end, :] = (
            is_in_G[:, :, None, :] & bw_mask[None, None, :, :])

    return sparse_mask


# ----------------------
# 三层混合注意力
# ----------------------
def hybrid_attention_quadratic_lola(q: torch.Tensor, k: torch.Tensor,
                                    f_q: torch.Tensor, f_k: torch.Tensor,
                                    v: torch.Tensor,
                                    window_factor: torch.Tensor,
                                    linear_factor: torch.Tensor,
                                    sparse_factor: torch.Tensor,
                                    window_size: int,
                                    sparse_budget: int = 128,
                                    sink_size: int = 0,
                                    kv_state: torch.Tensor = None,
                                    k_state: torch.Tensor = None,
                                    sparse_selection: str = 'chunked',
                                    sparse_chunk_size: int = None,
                                    eps: float = 1e-12,
                                    mask_value: float = -1e8):
    """
    三层混合注意力: 窗口 softmax + 稀疏 softmax + 线性注意力

    Sparse selection modes:
    - 'serial': per-token exact selection (reference, O(n²) per head)
    - 'chunked': per-chunk selection (efficient, paper Section 3.3)
    """
    b, h, q_len, d = q.shape
    k_len = k.shape[2]

    mask_window, mask_linear = get_masks(window_size, q_len, k_len, q.device)

    # Sink token 仅在处理包含 sink 的 chunk 时激活:
    # - kv_state is None → 首次 chunk 或全序列 (包含 sink token)
    # - kv_state is not None → 后续 chunk (不包含 sink token)
    effective_sink_size = sink_size if (sink_size > 0 and kv_state is None) else 0

    # --- Per-query sparse token selection (handles SRE and H evolution internally) ---
    with torch.no_grad():
        if sparse_selection == 'serial':
            sparse_mask = select_sparse_tokens_serial(
                f_k, v, sparse_budget, window_size, q_len, k_len,
                sink_size=effective_sink_size, eps=eps,
                kv_state=kv_state, k_state=k_state)
        else:  # 'chunked' (default)
            chunk_size = sparse_chunk_size or window_size
            sparse_mask = select_sparse_tokens_chunked(
                f_k, v, sparse_budget, window_size, q_len, k_len,
                chunk_size=chunk_size,
                sink_size=effective_sink_size, eps=eps,
                kv_state=kv_state, k_state=k_state)

        # sparse_mask: (b, h, q_len, k_len) — per-query sparse positions
        # Intersect with linear mask for safety (ensures below-window + causal)
        sparse_mask = sparse_mask & mask_linear.bool()

        # Sink tokens: always attend via sparse softmax path
        if effective_sink_size > 0:
            sink_mask = torch.zeros(1, 1, q_len, k_len, device=q.device, dtype=torch.bool)
            sink_mask[:, :, :, :effective_sink_size] = True
            # 因果性: sink 只对 position >= sink_size 的 query 可见
            causal = torch.ones(q_len, k_len, device=q.device, dtype=torch.bool).tril(k_len - q_len)
            sink_mask = sink_mask & causal[None, None, ...]
            # sink 从 linear 区域移除, 并入 sparse 路径
            sparse_mask = sparse_mask | sink_mask

        # 更新 linear mask: 去掉 sparse/sink token
        mask_linear_remaining = mask_linear.bool() & ~sparse_mask

    # --- 共享 QK 计算 (避免重复 matmul) ---
    qk = torch.einsum('bhmd,bhnd->bhmn', q.float(), k.float()) * (d ** -0.5)

    # --- 1. 窗口 softmax ---
    a_sm = qk.masked_fill(~mask_window.bool(), mask_value)
    a_sm_max = torch.amax(a_sm, dim=-1, keepdim=True)
    a_sm = window_factor * torch.exp(a_sm - a_sm_max)
    sum_sm = a_sm.sum(dim=-1, keepdim=True)

    # --- 2. 稀疏 softmax ---
    if sparse_mask.any():
        a_sp = qk.masked_fill(~sparse_mask, mask_value)
        a_sp_max = torch.amax(a_sp, dim=-1, keepdim=True)
        a_sp = sparse_factor * torch.exp(a_sp - a_sp_max)
        sum_sp = a_sp.sum(dim=-1, keepdim=True)

        # 数值稳定: 统一 window 和 sparse 的 max 值
        max_all = torch.maximum(a_sm_max, a_sp_max)
        a_sm = a_sm * torch.exp(a_sm_max - max_all)
        a_sp = a_sp * torch.exp(a_sp_max - max_all)
        sum_sm = a_sm.sum(dim=-1, keepdim=True)
        sum_sp = a_sp.sum(dim=-1, keepdim=True)
    else:
        a_sp = torch.zeros_like(a_sm)
        sum_sp = torch.zeros_like(sum_sm)

    # --- 3. 线性注意力 (remaining below-window tokens) ---
    a_ln = torch.einsum('bhmd,bhnd->bhmn', f_q.float(), f_k.float())
    a_ln = linear_factor * a_ln.masked_fill(~mask_linear_remaining, 0)
    sum_ln = a_ln.sum(dim=-1, keepdim=True)

    # --- 4. 联合归一化 ---
    a_combined = a_sm + a_sp + a_ln
    y = torch.einsum('bhmn,bhnd->bhmd', a_combined, v.float())

    # 如有先前累积的线性状态 (stateful training)
    if kv_state is not None:
        y = y + linear_factor * torch.einsum('bhld,bhdf->bhlf', f_q.float(), kv_state.float())
        sum_ln = sum_ln + linear_factor * torch.einsum(
            'bhld,bhnd->bhl', f_q.float(), k_state.float())[..., None]

    y = (y / (sum_sm + sum_sp + sum_ln + eps)).to(q.dtype)

    # 注意力权重 (for distillation loss)
    a = (a_combined / (sum_sm + sum_sp + sum_ln + eps)).to(q.dtype)

    return y, a


# ----------------------
# LoLA 注意力层
# ----------------------
class LolcatsLoLAAttention(LolcatsSlidingWindowAttention):
    """
    LoLA: LoLCATs + Sparse Cache via Self-Recall Error
    """
    def __init__(self,
                 sparse_budget: int = 128,
                 sink_size: int = 0,
                 init_sparse_factor: float = 10.0,
                 train_sparse_factor: bool = True,
                 layer_sparse_budget: int = None,
                 sparse_selection: str = 'chunked',
                 sparse_chunk_size: int = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.sparse_budget = layer_sparse_budget if layer_sparse_budget is not None else sparse_budget
        self.sink_size = sink_size
        self.sparse_selection = sparse_selection
        self.sparse_chunk_size = sparse_chunk_size

        # 新增可学习参数: sparse_factor (sigmoid 后 ≈ 1.0)
        device, dtype = self.q_proj.weight.device, self.q_proj.weight.dtype
        if train_sparse_factor:
            self.sparse_factors = nn.Parameter(
                init_sparse_factor * torch.ones(1, self.num_heads, 1, 1, device=device, dtype=dtype))
        else:
            self.register_buffer(
                "sparse_factors",
                init_sparse_factor * torch.ones(1, self.num_heads, 1, 1, device=device, dtype=dtype))

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value: Optional[Cache] = None,
                output_attentions: bool = False,
                use_cache: bool = False,
                **kwargs,
               ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass with 4 paths:
        1. train_attention (蒸馏): ground truth + predicted 三层混合注意力
        2. regular training (无 cache): 全序列三层混合注意力
        3. decoding (生成): 窗口 + 稀疏 + 线性
        4. stateful training (分块): 三层混合注意力 + 累积状态
        """
        b, l, _ = hidden_states.size()
        q, k, v, kv_seq_len = self.process_qkv(hidden_states, attention_mask,
                                               position_ids, past_key_value)
        f_q, f_k = self.feature_map_q(q), self.feature_map_k(k)

        # 三个门控
        window_factors = F.sigmoid(self.window_factors)
        linear_factors = 1 - window_factors if self.affine_attention_factors else 1
        sparse_factors = F.sigmoid(self.sparse_factors)

        if self.train_attention:
            # --- 路径 1: 蒸馏 ---
            with torch.no_grad():
                _y_true, a_true = softmax_attention(q, k, v)[:2]
                y_true = _y_true.transpose(1, 2).contiguous().view(b, l, self.hidden_size)
                y_true = self.o_proj(y_true)

            y_pred, a_pred = hybrid_attention_quadratic_lola(
                q, k, f_q, f_k, v,
                window_factors, linear_factors, sparse_factors,
                window_size=self.window_size,
                sparse_budget=self.sparse_budget,
                sink_size=self.sink_size,
                sparse_selection=self.sparse_selection,
                sparse_chunk_size=self.sparse_chunk_size,
            )
            attn_weights = ((a_pred, a_true), (y_pred, _y_true))
        else:
            attn_weights = None
            if past_key_value is None:
                # --- 路径 2: 常规训练 ---
                y_true, a_pred = hybrid_attention_quadratic_lola(
                    q, k, f_q, f_k, v,
                    window_factors, linear_factors, sparse_factors,
                    window_size=self.window_size,
                    sparse_budget=self.sparse_budget,
                    sink_size=self.sink_size,
                    sparse_selection=self.sparse_selection,
                    sparse_chunk_size=self.sparse_chunk_size,
                )
                attn_weights = a_pred
            else:
                past_key_value.window_size = self.decode_window_size
                # 同步 LoLA 参数到缓存对象
                past_key_value.sparse_budget = self.sparse_budget
                past_key_value.sink_size = self.sink_size
                if f_q.shape[2] == 1 and kv_seq_len > 1 and not self.training:
                    # --- 路径 3: 解码 ---
                    assert use_cache is True
                    _kv = past_key_value.update_for_decoding(k, v, self.layer_idx,
                                                             self.feature_map_k,
                                                             dtype=q.dtype)
                    k_cache, v_cache, f_kv_state, f_k_state, sparse_k, sparse_v = _kv

                    # 窗口 softmax
                    a_sm = torch.einsum('bhmd,bhnd->bhmn', q.float(), k_cache.float()) * (k.shape[-1] ** -0.5)
                    a_sm_max = torch.amax(a_sm, dim=-1, keepdim=True)
                    a_sm = window_factors * torch.exp(a_sm - a_sm_max)
                    sum_sm = a_sm.sum(dim=-1, keepdim=True)

                    # 稀疏 softmax (sparse + sink)
                    if sparse_k is not None and sparse_k.shape[-2] > 0:
                        a_sp = torch.einsum('bhmd,bhnd->bhmn', q.float(), sparse_k.float()) * (k.shape[-1] ** -0.5)
                        a_sp_max = torch.amax(a_sp, dim=-1, keepdim=True)
                        a_sp = sparse_factors * torch.exp(a_sp - a_sp_max)
                        sum_sp = a_sp.sum(dim=-1, keepdim=True)
                        # 数值稳定: 统一 max
                        max_all = torch.maximum(a_sm_max, a_sp_max)
                        a_sm = a_sm * torch.exp(a_sm_max - max_all)
                        a_sp = a_sp * torch.exp(a_sp_max - max_all)
                        sum_sm = a_sm.sum(dim=-1, keepdim=True)
                        sum_sp = a_sp.sum(dim=-1, keepdim=True)
                    else:
                        a_sp = None
                        sum_sp = torch.zeros_like(sum_sm)

                    # 联合归一化
                    y_true = torch.einsum('bhmn,bhnd->bhmd', a_sm, v_cache.float())
                    if a_sp is not None:
                        y_true = y_true + torch.einsum('bhmn,bhnd->bhmd', a_sp, sparse_v.float())
                    y_true = y_true + linear_factors * torch.einsum(
                        'bhlf,bhfd->bhld', f_q.float(), f_kv_state.float())
                    sum_ln = linear_factors * torch.einsum(
                        'bhlf,bhnf->bhl', f_q.float(), f_k_state.float())[..., None]
                    y_true = (y_true / (sum_sm + sum_sp + sum_ln + self.eps)).to(q.dtype)

                else:
                    # --- 路径 4: 状态训练 (分块) ---
                    try:
                        kv_state = past_key_value.kv_states[self.layer_idx]
                        k_state = past_key_value.k_states[self.layer_idx]
                    except IndexError:
                        kv_state, k_state = None, None
                    y_true, _ = hybrid_attention_quadratic_lola(
                        q, k, f_q, f_k, v,
                        window_factors, linear_factors, sparse_factors,
                        window_size=self.window_size,
                        sparse_budget=self.sparse_budget,
                        sink_size=self.sink_size,
                        kv_state=kv_state,
                        k_state=k_state,
                        sparse_selection=self.sparse_selection,
                        sparse_chunk_size=self.sparse_chunk_size,
                    )
                    past_key_value.update(k, v, self.layer_idx,
                                          fmap_key_states=f_k,
                                          accumulate_in_fp32=True)

            # 输出投影
            y_true = y_true.transpose(1, 2).contiguous().view(b, l, self.hidden_size)
            y_true = self.o_proj(y_true)

        return y_true, attn_weights, past_key_value


# ----------------------
# LoLA 缓存类
# ----------------------
class LinearAttentionLoLACache(LinearAttentionSlidingWindowCache):
    """
    LoLA 缓存: 在 LinearAttentionSlidingWindowCache 基础上新增稀疏缓存
    """
    def __init__(self, window_size: int = 64,
                 sparse_budget: int = 128,
                 sink_size: int = 0) -> None:
        super().__init__(window_size=window_size)
        self.sparse_budget = sparse_budget
        self.sink_size = sink_size

        self.sparse_k_cache: List[torch.Tensor] = []
        self.sparse_v_cache: List[torch.Tensor] = []
        self.sparse_sre: List[torch.Tensor] = []

        self.sink_k_cache: List[torch.Tensor] = []
        self.sink_v_cache: List[torch.Tensor] = []

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
               layer_idx: Optional[int] = None, cache_kwargs: Optional[any] = None,
               accumulate_in_fp32: bool = False,
               fmap_key_states: torch.Tensor = None,
               grad_enabled: bool = False,
               **kwargs: any,
              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        训练路径 (stateful training): 更新 KV 状态和缓存
        与父类逻辑基本一致, 但需要处理 sink token 和稀疏缓存
        """
        with torch.set_grad_enabled(grad_enabled):
            if layer_idx == 0:
                self._seen_tokens += key_states.shape[-2]

            dtype = key_states.dtype
            if accumulate_in_fp32:
                fmap_key_states = fmap_key_states.float()
                value_states_fp = value_states.float()
            else:
                value_states_fp = value_states

            is_first_chunk = len(self.k_states) <= layer_idx

            # 初始化 sink 缓存 (仅首次)
            if self.sink_size > 0 and len(self.sink_k_cache) <= layer_idx:
                self.sink_k_cache.append(key_states[:, :, :self.sink_size, :])
                self.sink_v_cache.append(value_states[:, :, :self.sink_size, :].to(dtype))

            # 只有首次 chunk 需要排除 sink token; 后续 chunk 无 sink
            start = self.sink_size if (self.sink_size > 0 and is_first_chunk) else 0

            # 从 start 开始的 token 分为 pre-window 和 window 两部分
            # pre-window → decode_kv_state; window → k_cache/v_cache
            seq_len = key_states.shape[-2]
            non_sink_len = seq_len - start
            if non_sink_len > self.window_size:
                fk_pre = fmap_key_states[:, :, start:seq_len - self.window_size]
                v_pre = value_states_fp[:, :, start:seq_len - self.window_size]
                fk_win = fmap_key_states[:, :, seq_len - self.window_size:]
                v_win = value_states_fp[:, :, seq_len - self.window_size:]
            else:
                # 整个 non-sink 部分不足一个 window, 全部作为 window
                fk_pre = fmap_key_states[:, :, :0]  # empty
                v_pre = value_states_fp[:, :, :0]
                fk_win = fmap_key_states[:, :, start:]
                v_win = value_states_fp[:, :, start:]

            decode_kv_state = torch.einsum('bhlf,bhld->bhfd', fk_pre, v_pre)
            kv_state = decode_kv_state + torch.einsum('bhlf,bhld->bhfd', fk_win, v_win)
            decode_k_state = fk_pre.sum(dim=-2, keepdim=True)
            k_state = decode_k_state + fk_win.sum(dim=-2, keepdim=True)

            if is_first_chunk:
                self.kv_states.append(kv_state.to(dtype))
                self.k_states.append(k_state.to(dtype))
                self.decode_kv_states.append(decode_kv_state.to(dtype))
                self.decode_k_states.append(decode_k_state.to(dtype))
                self.k_cache.append(key_states[:, :, -self.window_size:, :])
                self.v_cache.append(value_states[:, :, -self.window_size:, :].to(dtype))
                # 初始化空稀疏缓存
                b, h = key_states.shape[:2]
                d = key_states.shape[-1]
                self.sparse_k_cache.append(torch.empty(b, h, 0, d, device=key_states.device, dtype=dtype))
                self.sparse_v_cache.append(torch.empty(b, h, 0, d, device=key_states.device, dtype=dtype))
                self.sparse_sre.append(torch.empty(b, h, 0, device=key_states.device, dtype=torch.float32))
            else:
                kv_state = (self.kv_states[layer_idx].to(kv_state.dtype) + kv_state).to(dtype)
                k_state = (self.k_states[layer_idx].to(kv_state.dtype) + k_state).to(dtype)
                self.kv_states[layer_idx] = kv_state
                self.k_states[layer_idx] = k_state

                decode_kv_state = (self.decode_kv_states[layer_idx].to(kv_state.dtype)
                                   + decode_kv_state).to(dtype)
                decode_k_state = (self.decode_k_states[layer_idx].to(kv_state.dtype)
                                  + decode_k_state).to(dtype)
                self.decode_kv_states[layer_idx] = decode_kv_state
                self.decode_k_states[layer_idx] = decode_k_state

                self.k_cache[layer_idx] = key_states[:, :, -self.window_size:, :]
                self.v_cache[layer_idx] = value_states[:, :, -self.window_size:, :]
            self._seen_tokens_by_layer[layer_idx] += key_states.shape[-2]

        return self.kv_states[layer_idx], self.k_states[layer_idx]

    def update_for_decoding(self, keys: torch.Tensor, values: torch.Tensor,
                            layer_idx: int, feature_map_k: Callable,
                            dtype: torch.dtype):
        """
        解码路径: SRE 驱动的缓存管理

        每步:
        1. 窗口未满 → 直接加入
        2. 窗口已满 → 驱逐最旧 token
           a. 计算被驱逐 token 的 SRE
           b. 稀疏缓存未满 → 加入稀疏缓存
           c. 稀疏缓存已满 → SRE 比较, 决定替换或压缩到线性状态

        Returns:
            (k_cache, v_cache, decode_kv_states, decode_k_states, sparse_k_concat, sparse_v_concat)
            sparse_k_concat/sparse_v_concat 包含 sink + sparse tokens
        """
        with torch.no_grad():
            k_cache = self.k_cache[layer_idx]
            v_cache = self.v_cache[layer_idx]

            if k_cache.shape[-2] < self.window_size:
                # 窗口未满, 直接加入
                self.k_cache[layer_idx] = torch.cat([k_cache, keys], dim=-2)
                self.v_cache[layer_idx] = torch.cat([v_cache, values], dim=-2)
            else:
                # 驱逐最旧 token
                evicted_k = k_cache[:, :, :1, :]  # b, h, 1, d
                evicted_v = v_cache[:, :, :1, :]

                # 计算被驱逐 token 的 SRE
                f_k_evicted = feature_map_k(evicted_k)
                kv_state = self.decode_kv_states[layer_idx]
                k_state_acc = self.decode_k_states[layer_idx]

                # v_hat = H @ φ(k) / (s^T @ φ(k))
                numerator = torch.einsum('bhfd,bhlf->bhld', kv_state.float(), f_k_evicted.float())
                denominator = torch.einsum('bhnf,bhlf->bhl', k_state_acc.float(), f_k_evicted.float())
                v_hat = numerator / (denominator[..., None] + 1e-12)
                evicted_sre = ((evicted_v.float() - v_hat) ** 2).sum(dim=-1)  # b, h, 1

                sparse_k = self.sparse_k_cache[layer_idx]
                sparse_v = self.sparse_v_cache[layer_idx]
                sparse_sre = self.sparse_sre[layer_idx]

                if sparse_k.shape[-2] < self.sparse_budget:
                    # 稀疏缓存未满, 直接加入
                    self.sparse_k_cache[layer_idx] = torch.cat([sparse_k, evicted_k], dim=-2)
                    self.sparse_v_cache[layer_idx] = torch.cat([sparse_v, evicted_v.to(dtype)], dim=-2)
                    self.sparse_sre[layer_idx] = torch.cat([sparse_sre, evicted_sre], dim=-1)
                else:
                    # 论文 Eq.12: 用当前 H_t 对所有 sparse cache entries 重新评分
                    f_k_sparse = feature_map_k(sparse_k)
                    num_sp = torch.einsum('bhfd,bhlf->bhld', kv_state.float(), f_k_sparse.float())
                    den_sp = torch.einsum('bhnf,bhlf->bhl', k_state_acc.float(), f_k_sparse.float())
                    v_hat_sp = num_sp / (den_sp[..., None] + 1e-12)
                    sparse_sre = ((sparse_v.float() - v_hat_sp) ** 2).sum(dim=-1)  # b, h, budget
                    self.sparse_sre[layer_idx] = sparse_sre

                    # 比较 evicted token SRE 与更新后的最小 SRE
                    min_sre, min_idx = sparse_sre.min(dim=-1, keepdim=True)  # b, h, 1
                    should_replace = (evicted_sre > min_sre)  # b, h, 1

                    if should_replace.any():
                        # 需要逐 batch/head 处理替换
                        min_idx_expanded_d = min_idx.unsqueeze(-1).expand_as(evicted_k)  # b, h, 1, d

                        # 取出被替换的 sparse token → 进入线性状态
                        displaced_k = torch.gather(sparse_k, 2, min_idx_expanded_d)
                        displaced_v = torch.gather(sparse_v, 2, min_idx_expanded_d)
                        f_displaced_k = feature_map_k(displaced_k)
                        displaced_kv = torch.einsum(
                            'bhlf,bhld->bhfd', f_displaced_k.float(), displaced_v.float()
                        ).to(dtype)

                        # 只对 should_replace 的位置更新
                        replace_mask = should_replace.unsqueeze(-1)  # b, h, 1, 1
                        replace_mask_sre = should_replace  # b, h, 1

                        # 更新线性状态 (被替换的 token 进入)
                        self.decode_kv_states[layer_idx] = self.decode_kv_states[layer_idx] + (
                            displaced_kv * replace_mask)
                        self.decode_k_states[layer_idx] = self.decode_k_states[layer_idx] + (
                            f_displaced_k.to(dtype) * replace_mask)

                        # 替换 sparse cache 中的对应位置
                        self.sparse_k_cache[layer_idx] = sparse_k.scatter(
                            2, min_idx_expanded_d,
                            torch.where(replace_mask, evicted_k, displaced_k))
                        self.sparse_v_cache[layer_idx] = sparse_v.scatter(
                            2, min_idx_expanded_d,
                            torch.where(replace_mask, evicted_v.to(dtype), displaced_v))
                        self.sparse_sre[layer_idx] = sparse_sre.scatter(
                            -1, min_idx,
                            torch.where(replace_mask_sre, evicted_sre, min_sre))

                        # 未被替换的 evicted token 进入线性状态
                        not_replace_mask = ~replace_mask
                        kv_evicted = torch.einsum(
                            'bhlf,bhld->bhfd', f_k_evicted.float(), evicted_v.float()
                        ).to(dtype)
                        self.decode_kv_states[layer_idx] = self.decode_kv_states[layer_idx] + (
                            kv_evicted * not_replace_mask)
                        self.decode_k_states[layer_idx] = self.decode_k_states[layer_idx] + (
                            f_k_evicted.to(dtype) * not_replace_mask)
                    else:
                        # 所有 evicted token 都进入线性状态
                        kv_state_new = torch.einsum(
                            'bhlf,bhld->bhfd', f_k_evicted.float(), evicted_v.float()
                        ).to(dtype)
                        self.decode_kv_states[layer_idx] = self.decode_kv_states[layer_idx] + kv_state_new
                        self.decode_k_states[layer_idx] = self.decode_k_states[layer_idx] + f_k_evicted.to(dtype)

                # 移动窗口
                self.k_cache[layer_idx] = torch.cat([k_cache[:, :, 1:, :], keys], dim=-2)
                self.v_cache[layer_idx] = torch.cat([v_cache[:, :, 1:, :], values], dim=-2)

            if layer_idx == 0:
                self._seen_tokens += keys.shape[-2]
            self._seen_tokens_by_layer[layer_idx] += keys.shape[-2]

            # 拼接 sink + sparse 作为 sparse 注意力的 KV
            sparse_k_out = self.sparse_k_cache[layer_idx]
            sparse_v_out = self.sparse_v_cache[layer_idx]
            if self.sink_size > 0 and len(self.sink_k_cache) > layer_idx:
                sparse_k_out = torch.cat([self.sink_k_cache[layer_idx], sparse_k_out], dim=-2)
                sparse_v_out = torch.cat([self.sink_v_cache[layer_idx], sparse_v_out], dim=-2)

            return (self.k_cache[layer_idx], self.v_cache[layer_idx],
                    self.decode_kv_states[layer_idx], self.decode_k_states[layer_idx],
                    sparse_k_out, sparse_v_out)
