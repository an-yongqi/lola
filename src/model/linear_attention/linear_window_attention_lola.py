"""
LoLA: LoLCATs + Sparse Cache via Self-Recall Error (SRE)

三层混合注意力:
1. 滑动窗口 (softmax): 最近 window_size 个 token
2. 稀疏缓存 (softmax): SRE 选出的重要 token
3. 线性状态 (linear): 其余 token 压缩到 recurrent state

基于 linear_window_attention_sw.py 开发
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
        sre: shape (b, h, l), 每个 token 的 SRE ��
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


def select_sparse_tokens(sre: torch.Tensor, sparse_budget: int,
                         mask_linear: torch.Tensor, sink_size: int = 0,
                         window_size: int = 64) -> torch.Tensor:
    """
    基于 SRE 选择 sparse cache tokens (per-head)

    Args:
        sre: shape (b, h, l)
        sparse_budget: 选择的 token 数量
        mask_linear: shape (1, 1, q_len, k_len), 1 表示 below-window 区域
        sink_size: sink token 数量 (从 SRE 选择中排除)
        window_size: 窗口大小

    Returns:
        sparse_mask: shape (b, h, 1, k_len), 被选中的 sparse token 位置为 True
    """
    b, h, l = sre.shape
    device = sre.device

    # 只对 below-window 区域做选择 (用最后一行 mask 作为全局可选范围)
    linear_eligible = mask_linear[0, 0, -1, :]  # (k_len,) - 最后一个 query 的 linear 区域

    sre_for_selection = sre.clone()
    # 排除 window 内的 token (linear_eligible 为 0 的位置)
    sre_for_selection = sre_for_selection * linear_eligible[None, None, :]
    # 排除 sink token
    if sink_size > 0:
        sre_for_selection[:, :, :sink_size] = 0

    # 实际可选 token 数量可能小于 sparse_budget
    n_eligible = linear_eligible.sum().item() - sink_size
    actual_budget = min(sparse_budget, max(0, int(n_eligible)))

    if actual_budget == 0:
        # 没有可选 token, 返回全 False mask
        return torch.zeros(b, h, 1, l, device=device, dtype=torch.bool)

    # Per-head top-k 选择
    _, sparse_idx = sre_for_selection.topk(actual_budget, dim=-1)  # b, h, actual_budget

    # 构建 sparse mask: (b, h, 1, k_len)
    sparse_mask = torch.zeros(b, h, 1, l, device=device, dtype=torch.bool)
    sparse_mask.scatter_(-1, sparse_idx.unsqueeze(2), True)

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
                                    eps: float = 1e-12,
                                    mask_value: float = -1e8):
    """
    三层混合注意力: 窗口 softmax + 稀疏 softmax + 线性注意力
    """
    b, h, q_len, d = q.shape
    k_len = k.shape[2]

    mask_window, mask_linear = get_masks(window_size, q_len, k_len, q.device)

    # --- SRE 计算和 sparse token 选择 ---
    with torch.no_grad():
        sre = compute_sre(f_k, v, kv_state=kv_state, k_state=k_state)
        sparse_mask = select_sparse_tokens(sre, sparse_budget, mask_linear,
                                           sink_size=sink_size, window_size=window_size)
        # sparse_mask: (b, h, 1, k_len), broadcast over queries
        # 与 mask_linear 交集: 确保因果性
        sparse_mask = sparse_mask.expand_as(mask_linear) & mask_linear.bool()
        # 如有 sink token, 为其创建 mask
        if sink_size > 0:
            sink_mask = torch.zeros(1, 1, q_len, k_len, device=q.device, dtype=torch.bool)
            sink_mask[:, :, :, :sink_size] = True
            # 因果性: sink 只对 position >= sink_size 的 query 可见 (对更早的 query 本身也可见)
            causal = torch.ones(q_len, k_len, device=q.device, dtype=torch.bool).tril(k_len - q_len)
            sink_mask = sink_mask & causal[None, None, ...]
            # sink 从 linear 区域移除, 并入 sparse 路径
            sparse_mask = sparse_mask | sink_mask
        # 更新 linear mask: 去掉 sparse/sink token
        mask_linear_remaining = mask_linear.bool() & ~sparse_mask

    # --- 1. 窗口 softmax ---
    a_sm = torch.einsum('bhmd,bhnd->bhmn', q.float(), k.float()) * (d ** -0.5)
    a_sm = a_sm.masked_fill(~mask_window.bool(), mask_value)
    a_sm_max = torch.amax(a_sm, dim=-1, keepdim=True)
    a_sm = window_factor * torch.exp(a_sm - a_sm_max)
    sum_sm = a_sm.sum(dim=-1, keepdim=True)

    # --- 2. 稀疏 softmax ---
    if sparse_mask.any():
        a_sp = torch.einsum('bhmd,bhnd->bhmn', q.float(), k.float()) * (d ** -0.5)
        a_sp = a_sp.masked_fill(~sparse_mask, mask_value)
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

    # 注意力权重 (for distillation loss, not strictly needed but matches interface)
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
                 **kwargs):
        super().__init__(**kwargs)
        self.sparse_budget = layer_sparse_budget if layer_sparse_budget is not None else sparse_budget
        self.sink_size = sink_size

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
                )
                attn_weights = a_pred
            else:
                past_key_value.window_size = self.decode_window_size
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
                    y_true = (y_true / (sum_sm + sum_sp + sum_ln)).to(q.dtype)

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

            # 初始化 sink 缓存 (仅首次)
            if self.sink_size > 0 and len(self.sink_k_cache) <= layer_idx:
                self.sink_k_cache.append(key_states[:, :, :self.sink_size, :])
                self.sink_v_cache.append(value_states[:, :, :self.sink_size, :].to(dtype))

            # 计算 decode 和 full KV state (与父类一致)
            # 注意: sink token 不进入线性状态
            start = self.sink_size if len(self.sink_k_cache) > layer_idx else 0
            if key_states.shape[-2] > start + self.window_size:
                fk_pre = fmap_key_states[:, :, start:-self.window_size]
                v_pre = value_states_fp[:, :, start:-self.window_size]
            else:
                fk_pre = fmap_key_states[:, :, :0]  # empty
                v_pre = value_states_fp[:, :, :0]

            decode_kv_state = torch.einsum('bhlf,bhld->bhfd', fk_pre, v_pre)
            kv_state = decode_kv_state + torch.einsum(
                'bhlf,bhld->bhfd',
                fmap_key_states[:, :, -self.window_size:],
                value_states_fp[:, :, -self.window_size:]
            )
            decode_k_state = fk_pre.sum(dim=-2, keepdim=True)
            k_state = decode_k_state + fmap_key_states[:, :, -self.window_size:].sum(dim=-2, keepdim=True)

            if len(self.k_states) <= layer_idx:
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
                    # 稀疏缓存已满, 比较 SRE
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
                        f_k_state = f_k_evicted.to(dtype)
                        kv_evicted = torch.einsum(
                            'bhlf,bhld->bhfd', f_k_evicted.float(), evicted_v.float()
                        ).to(dtype)
                        self.decode_kv_states[layer_idx] = self.decode_kv_states[layer_idx] + (
                            kv_evicted * not_replace_mask)
                        self.decode_k_states[layer_idx] = self.decode_k_states[layer_idx] + (
                            f_k_state * not_replace_mask)
                    else:
                        # 所有 evicted token 都进入线性状态
                        f_k_state = feature_map_k(evicted_k)
                        kv_state_new = torch.einsum(
                            'bhlf,bhld->bhfd', f_k_state.float(), evicted_v.float()
                        ).to(dtype)
                        self.decode_kv_states[layer_idx] = self.decode_kv_states[layer_idx] + kv_state_new
                        self.decode_k_states[layer_idx] = self.decode_k_states[layer_idx] + f_k_state.to(dtype)

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
