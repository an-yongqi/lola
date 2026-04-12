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
        kv_state: 累积线性状态 H, shape (b, h, f, d). None 时从 f_k, v 计算
        k_state: 累积 key 状态 s, shape (b, h, 1, f). None 时从 f_k 计算
        eps: 数值稳定性

    Returns:
        sre: shape (b, h, l), 每个 token 的 SRE 值
    """
    # TODO: Phase 2 实现
    raise NotImplementedError


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

    Args:
        q, k: 原始 query/key, shape (b, h, l, d)
        f_q, f_k: feature-mapped query/key, shape (b, h, l, f)
        v: values, shape (b, h, l, d)
        window_factor: 窗口注意力门控, shape (1, h, 1, 1)
        linear_factor: 线性注意力门控
        sparse_factor: 稀疏缓存注意力门控, shape (1, h, 1, 1)
        window_size: 滑动窗口大小
        sparse_budget: 稀疏缓存大小
        sink_size: sink token 数量
        kv_state: 先前累积的线性状态 (stateful training)
        k_state: 先前累积的 key 状态 (stateful training)

    Returns:
        y: 注意力输出, shape (b, h, l, d)
        a: 注意力权重 (用于蒸馏 loss)
    """
    # TODO: Phase 2 实现
    raise NotImplementedError


# ----------------------
# LoLA 注意力层
# ----------------------
class LolcatsLoLAAttention(LolcatsSlidingWindowAttention):
    """
    LoLA: LoLCATs + Sparse Cache via Self-Recall Error

    在 LolcatsSlidingWindowAttention 基础上新增:
    - sparse_budget: 稀疏缓存大小
    - sink_size: sink token 数量
    - sparse_factors: 稀疏缓存注意力门控 (可学习)
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
        # 覆盖 quadratic_attention 为 LoLA 版本
        self.quadratic_attention = hybrid_attention_quadratic_lola

        # 新增可学习参数: sparse_factor
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
        1. train_attention (蒸馏)
        2. regular training (无 cache)
        3. decoding (生成)
        4. stateful training (分块)
        """
        # TODO: Phase 2 实现
        raise NotImplementedError


# ----------------------
# LoLA 缓存类
# ----------------------
class LinearAttentionLoLACache(LinearAttentionSlidingWindowCache):
    """
    LoLA 缓存: 在 LinearAttentionSlidingWindowCache 基础上新增稀疏缓存

    新增状态:
    - sparse_k_cache, sparse_v_cache: 稀疏缓存的 KV
    - sparse_sre: 稀疏缓存中各 token 的 SRE 值 (用于 eviction)
    - sink_k_cache, sink_v_cache: sink token (永不驱逐)
    """
    def __init__(self, window_size: int = 64,
                 sparse_budget: int = 128,
                 sink_size: int = 0) -> None:
        super().__init__(window_size=window_size)
        self.sparse_budget = sparse_budget
        self.sink_size = sink_size

        # 稀疏缓存
        self.sparse_k_cache: List[torch.Tensor] = []
        self.sparse_v_cache: List[torch.Tensor] = []
        self.sparse_sre: List[torch.Tensor] = []

        # Sink 缓存
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
        训练路径: 更新 KV 状态和缓存
        """
        # TODO: Phase 2 实现
        raise NotImplementedError

    def update_for_decoding(self, keys: torch.Tensor, values: torch.Tensor,
                            layer_idx: int, feature_map_k: Callable,
                            dtype: torch.dtype):
        """
        解码路径: SRE 驱动的缓存管理
        """
        # TODO: Phase 2 实现
        raise NotImplementedError
