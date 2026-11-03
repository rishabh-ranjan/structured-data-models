# Portions derived from Relational Transformer:
# https://github.com/stanford-star/relational-transformer
# Licensed under the Apache License, Version 2.0; see LICENSE.

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

ENCODERS = (
    ("column", "col_name"),
    ("number", "number"),
    ("text", "text"),
    ("datetime", "datetime"),
)
ATTENTIONS = (("column", "col"), ("feature", "feat"), ("neighbor", "nbr"))


def remap_ckpt(ckpt: dict[str, Tensor]) -> dict[str, Tensor]:  # noqa: D103
    out: dict[str, Tensor] = {}

    for kind, upstream in ENCODERS:
        out[f"encoders.{kind}.0.weight"] = ckpt[f"enc_dict.{upstream}.weight"]
        out[f"encoders.{kind}.0.bias"] = ckpt[f"enc_dict.{upstream}.bias"]
        out[f"encoders.{kind}.1.weight"] = ckpt[f"norm_dict.{upstream}.scale"]
        if kind != "column":
            out[f"mask_embeddings.{kind}"] = ckpt[f"mask_embs.{upstream}"]

    num_layers = 1 + max(
        int(key.split(".")[1]) for key in ckpt if key.startswith("blocks.")
    )
    for layer in range(num_layers):
        for name, upstream in ATTENTIONS:
            src = f"blocks.{layer}.attns.{upstream}."
            dst = f"blocks.{layer}.attns.{name}."
            out[f"{dst}qkv_lin.weight"] = torch.cat(
                [ckpt[f"{src}{key}.weight"] for key in ("wq", "wk", "wv")]
            )
            # The upstream gate is `2 * sigmoid(.)`; fold the factor into the
            # output projection, which has no bias.
            out[f"{dst}out_lin.weight"] = 2 * ckpt[f"{src}wo.weight"]
            out[f"{dst}gate_lin.weight"] = ckpt[f"{src}wg.weight"]
            out[f"{dst}query_norm.weight"] = ckpt[f"{src}q_norm.scale"]
            out[f"{dst}key_norm.weight"] = ckpt[f"{src}k_norm.scale"]
            out[f"{dst}query_scaling.head_scale"] = ckpt[f"{src}scale"].view(
                -1
            )
            out[f"blocks.{layer}.norms.{name}.weight"] = ckpt[
                f"blocks.{layer}.norms.{upstream}.scale"
            ]
        out[f"blocks.{layer}.norms.mlp.weight"] = ckpt[
            f"blocks.{layer}.norms.ffn.scale"
        ]
        for name, upstream in (("gate", "w1"), ("up", "w3"), ("down", "w2")):
            out[f"blocks.{layer}.mlp.{name}_lin.weight"] = ckpt[
                f"blocks.{layer}.ffn.{upstream}.weight"
            ]

    out["norm.weight"] = ckpt["norm_out.scale"]
    out["head.weight"] = ckpt["dec_dict.number.weight"]
    out["head.bias"] = ckpt["dec_dict.number.bias"]

    return out
