# Portions derived from Relational Transformer:
# https://github.com/stanford-star/relational-transformer
# Licensed under the Apache License, Version 2.0; see LICENSE.

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D205

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import cache
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ModuleDict
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from sdm.nn import LogScale, RMSNorm, SwiGLU

ATTENTION_TYPES = ("column", "feature", "neighbor")


class AttentionMask(NamedTuple):
    r"""The attention mask of one attention type.

    Args:
        mask: Boolean mask with shape ``[B, S, S]``; ``True`` lets a query
            cell attend to a key cell.
        key_len: The number of attended keys per query with shape ``[B, S]``.
        block_mask: The block-sparse form of ``mask`` for flex attention, or
            ``None`` to run dense scaled dot-product attention.
    """

    mask: Tensor
    key_len: Tensor
    block_mask: BlockMask | None


def attention_masks(
    node: Tensor,  # [B, S]
    parents: Tensor,  # [B, S, K]
    column: Tensor,  # [B, S]
    *,
    block_sparse: bool,
) -> dict[str, AttentionMask]:
    r"""Return the attention masks of a :class:`RelationalBlock`.

    Args:
        node: The row of each cell with shape ``[B, S]``; ``-1`` marks
            padding.
        parents: The parent rows of each cell's row with shape ``[B, S, K]``,
            padded with ``-1``.
        column: The column of each cell with shape ``[B, S]``.
        block_sparse: Whether to also build block-sparse masks for flex
            attention.

    Returns:
        Masks keyed by attention type. *Column* attention connects cells of
        the same column, *feature* attention connects cells of the same row
        and of its parent rows, and *neighbor* attention connects a cell to
        the cells of rows that reference its row.
    """
    valid = node >= 0
    pair = valid.unsqueeze(-1) & valid.unsqueeze(-2)  # [B, S, S]
    same_node = node.unsqueeze(-1) == node.unsqueeze(-2)
    key_in_parents = torch.zeros_like(same_node)
    for k in range(parents.size(-1)):  # Key row is a parent of query row.
        key_in_parents |= node.unsqueeze(-2) == parents[..., k].unsqueeze(-1)
    masks = {
        "column": (column.unsqueeze(-1) == column.unsqueeze(-2)) & pair,
        "feature": (same_node | key_in_parents) & pair,
        "neighbor": key_in_parents.transpose(-1, -2) & pair,
    }
    return {
        name: AttentionMask(
            mask=mask,
            key_len=mask.sum(dim=-1),
            block_mask=_block_mask(mask) if block_sparse else None,
        )
        for name, mask in masks.items()
    }


def _block_mask(mask: Tensor, block_size: int = 128) -> BlockMask:  # [B, S, S]
    # Read the block layout off the materialized mask, as a compiled
    # `create_block_mask` would, without a compilation per sequence length.
    B, S, _ = mask.size()
    num_blocks = -(-S // block_size)
    padding = num_blocks * block_size - S
    blocks = F.pad(mask, (0, padding, 0, padding))
    blocks = blocks.view(B, num_blocks, block_size, num_blocks, block_size)
    count = blocks.sum(dim=(2, 4))  # [B, Q_blocks, KV_blocks]
    full = count == block_size * block_size
    partial = (count > 0) & ~full

    def ordered(blocks: Tensor) -> tuple[Tensor, Tensor]:
        blocks = blocks.to(torch.int32).unsqueeze(
            1
        )  # [B, 1, Q_blocks, KV_blocks]
        indices = blocks.argsort(dim=-1, descending=True, stable=True)
        return blocks.sum(dim=-1), indices.to(torch.int32)

    def mask_mod(b: Tensor, h: Tensor, q: Tensor, kv: Tensor) -> Tensor:
        return mask[b, q, kv]

    return BlockMask.from_kv_blocks(
        *ordered(partial),
        *ordered(full),
        BLOCK_SIZE=block_size,
        mask_mod=mask_mod,
        seq_lengths=(S, S),
    )


@cache
def _compiled_flex_attention() -> Callable[..., Tensor]:
    # Static shapes compile once per context size; a second size switches to
    # dynamic shapes instead of recompiling for every new size.
    return torch.compile(  # ty: ignore[no-matching-overload]
        flex_attention, dynamic=None
    )


class RelationalAttention(torch.nn.Module):
    r"""Masked self-attention over cells for one attention type.

    Query and key heads are normalized with :class:`~sdm.nn.RMSNorm`, queries
    are scaled by :class:`~sdm.nn.LogScale` of the number of attended keys,
    logits use a ``1 / channels_per_head`` scale, and the attention output is
    gated by a sigmoid of the input. Block-sparse masks run through
    :func:`torch.nn.attention.flex_attention.flex_attention`; dense masks run
    through :func:`torch.nn.functional.scaled_dot_product_attention`. A cell
    without any key to attend to returns zero.

    Args:
        channels: The number of input and output channels.
        num_heads: The number of attention heads.
        device: The device.
        dtype: The dtype.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.num_heads = num_heads
        self.head_channels = channels // num_heads

        self.qkv_lin = Linear(
            channels, 3 * channels, bias=False, **factory_kwargs
        )
        self.query_norm = RMSNorm(
            self.head_channels, eps=1e-6, **factory_kwargs
        )
        self.key_norm = RMSNorm(self.head_channels, eps=1e-6, **factory_kwargs)
        self.query_scaling = LogScale(num_heads, **factory_kwargs)
        self.gate_lin = Linear(
            channels, channels, bias=False, **factory_kwargs
        )
        self.out_lin = Linear(channels, channels, bias=False, **factory_kwargs)

    def forward(
        self,
        x: Tensor,  # [B, S, C]
        attn_mask: AttentionMask,
    ) -> Tensor:  # [B, S, C]
        r"""The forward pass.

        Args:
            x: The cell embeddings with shape ``[B, S, C]``.
            attn_mask: The attention mask of this attention type.

        Returns:
            Tensor with shape ``[B, S, C]``.
        """
        query, key, value = (
            self.qkv_lin(x)
            .unflatten(-1, (3, self.num_heads, self.head_channels))
            .unbind(-3)
        )  # [B, S, H, C // H]
        query = self.query_scaling(
            self.query_norm(query),
            key_len=attn_mask.key_len,
        )
        key = self.key_norm(key)
        query, key, value = (
            tensor.transpose(1, 2) for tensor in (query, key, value)
        )  # [B, H, S, C // H]

        if attn_mask.block_mask is not None:
            out = _compiled_flex_attention()(
                query,
                key,
                value,
                block_mask=attn_mask.block_mask,
                scale=1.0 / self.head_channels,
            )
        else:
            # SDPA has no key to normalize over for an empty row: let it attend
            # to itself and drop the result afterwards.
            empty = attn_mask.key_len == 0  # [B, S]
            eye = torch.eye(x.size(1), dtype=torch.bool, device=x.device)
            out = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=(attn_mask.mask | (empty.unsqueeze(-1) & eye))[
                    :, None
                ],
                scale=1.0 / self.head_channels,
            )
            out = out.masked_fill(empty[:, None, :, None], 0.0)

        out = out.transpose(1, 2).flatten(-2)  # [B, S, C]
        return self.out_lin(out * torch.sigmoid(self.gate_lin(x)))


class RelationalBlock(torch.nn.Module):
    r"""Relational attention block from the `"Relational Transformer: Toward
    Zero-Shot Foundation Models for Relational Data"
    <https://arxiv.org/abs/2510.06377>`_ paper.

    Cells pass through three :class:`RelationalAttention` residuals in
    sequence, over the masks of :func:`attention_masks`: *column*
    attention among cells of the same column, *feature* attention among cells
    of the same row and of its foreign-key parent rows, and *neighbor*
    attention from a cell to the cells of rows that reference its row. A
    :class:`~sdm.nn.SwiGLU` feed-forward residual follows. Every residual is
    pre-normalized with :class:`~sdm.nn.RMSNorm`.

    Args:
        channels: The number of input and output channels.
        num_heads: The number of attention heads.
        hidden_channels: The hidden channels of the feed-forward network.
        device: The device.
        dtype: The dtype.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        hidden_channels: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.norms = ModuleDict(
            {
                name: RMSNorm(channels, eps=1e-6, **factory_kwargs)
                for name in (*ATTENTION_TYPES, "mlp")
            }
        )
        self.attns = ModuleDict(
            {
                name: RelationalAttention(
                    channels, num_heads, **factory_kwargs
                )
                for name in ATTENTION_TYPES
            }
        )
        self.mlp = SwiGLU(
            channels, hidden_channels, bias=False, **factory_kwargs
        )

    def forward(
        self,
        x: Tensor,  # [B, S, C]
        attn_masks: Mapping[str, AttentionMask],
    ) -> Tensor:  # [B, S, C]
        r"""The forward pass.

        Args:
            x: The cell embeddings with shape ``[B, S, C]``.
            attn_masks: The masks keyed by attention type, as returned by
                :func:`attention_masks`.

        Returns:
            Tensor with shape ``[B, S, C]``.
        """
        for name, attn in self.attns.items():
            x = x + attn(self.norms[name](x), attn_masks[name])
        return x + self.mlp(self.norms["mlp"](x))
