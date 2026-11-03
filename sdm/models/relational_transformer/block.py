# Portions derived from Relational Transformer:
# https://github.com/stanford-star/relational-transformer
# Licensed under the Apache License, Version 2.0; see LICENSE.

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D205

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor
from torch.nn import Linear, ModuleDict, Sequential, Sigmoid

from sdm.nn import Attention, LogScale, RMSNorm, SwiGLU

ATTENTION_TYPES = ("column", "feature", "neighbor")


def attention_masks(
    node: Tensor,  # [..., S]
    parents: Tensor,  # [..., S, K]
    column: Tensor,  # [..., S]
) -> dict[str, Tensor]:  # [..., 1, S, S]
    r"""Return the boolean attention masks of a :class:`RelationalBlock`.

    Args:
        node: The row of each cell with shape ``[..., S]``; ``-1`` marks
            padding.
        parents: The parent rows of each cell's row with shape
            ``[..., S, K]``, padded with ``-1``.
        column: The column of each cell with shape ``[..., S]``.

    Returns:
        Masks with shape ``[..., 1, S, S]`` keyed by attention type, where
        ``True`` lets a query cell attend to a key cell.
    """
    valid = node >= 0
    pair = valid.unsqueeze(-1) & valid.unsqueeze(-2)  # [..., S, S]
    same_node = node.unsqueeze(-1) == node.unsqueeze(-2)
    key_in_parents = torch.zeros_like(same_node)
    for k in range(
        parents.size(-1)
    ):  # The key's row is a parent of the query's row.
        key_in_parents |= node.unsqueeze(-2) == parents[..., k].unsqueeze(-1)
    return {
        "column": ((column.unsqueeze(-1) == column.unsqueeze(-2)) & pair)[
            ..., None, :, :
        ],
        "feature": ((same_node | key_in_parents) & pair)[..., None, :, :],
        "neighbor": (key_in_parents.transpose(-1, -2) & pair)[..., None, :, :],
    }


class RelationalBlock(torch.nn.Module):
    r"""Relational attention block from the `"Relational Transformer: Toward
    Zero-Shot Foundation Models for Relational Data"
    <https://arxiv.org/abs/2510.06377>`_ paper.

    Cells pass through three masked self-attention residuals in sequence:
    *column* attention among cells of the same column, *feature* attention
    among cells of the same row and of its foreign-key parent rows, and
    *neighbor* attention from a cell to the cells of rows that reference its
    row. A :class:`~sdm.nn.SwiGLU` feed-forward residual follows. Every
    residual is pre-normalized with :class:`~sdm.nn.RMSNorm`.

    Attention normalizes query and key heads, scales queries by
    :class:`~sdm.nn.LogScale` of the number of attended keys, uses a
    ``1 / channels_per_head`` logit scale, and gates its output with a sigmoid
    of the normalized input. A cell without any key to attend to leaves the
    residual unchanged.

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
        head_channels = channels // num_heads

        self.norms = ModuleDict(
            {
                name: RMSNorm(channels, eps=1e-6, **factory_kwargs)
                for name in (*ATTENTION_TYPES, "mlp")
            }
        )
        self.attns = ModuleDict(
            {
                name: Attention(
                    channels=channels,
                    num_query_heads=num_heads,
                    query_transform=RMSNorm(
                        head_channels, eps=1e-6, **factory_kwargs
                    ),
                    key_transform=RMSNorm(
                        head_channels, eps=1e-6, **factory_kwargs
                    ),
                    query_scaling=LogScale(num_heads, **factory_kwargs),
                    gate=Sequential(
                        Linear(
                            channels, channels, bias=False, **factory_kwargs
                        ),
                        Sigmoid(),
                    ),
                    scale=1.0 / head_channels,
                    bias=False,
                    **factory_kwargs,
                )
                for name in ATTENTION_TYPES
            }
        )
        self.mlp = SwiGLU(
            channels, hidden_channels, bias=False, **factory_kwargs
        )

    def forward(
        self,
        x: Tensor,  # [..., S, C]
        attn_masks: Mapping[str, Tensor],  # [..., 1, S, S]
    ) -> Tensor:  # [..., S, C]
        r"""The forward pass.

        Args:
            x: The cell embeddings with shape ``[..., S, C]``.
            attn_masks: Boolean masks with shape ``[..., 1, S, S]`` keyed by
                attention type, as returned by :func:`attention_masks`.

        Returns:
            Tensor with shape ``[..., S, C]``.
        """
        eye = torch.eye(x.size(-2), dtype=torch.bool, device=x.device)
        for name, attn in self.attns.items():
            attn_mask = attn_masks[name]
            # SDPA has no key to normalize over for an empty row: let it attend
            # to itself and drop the result afterwards.
            empty = ~attn_mask.any(dim=-1, keepdim=True)  # [..., 1, S, 1]
            out = attn(
                self.norms[name](x), attn_mask=attn_mask | (empty & eye)
            )
            x = x + out.masked_fill(empty.squeeze(-3), 0.0)
        return x + self.mlp(self.norms["mlp"](x))
