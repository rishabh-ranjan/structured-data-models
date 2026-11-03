# Portions derived from Relational Transformer:
# https://github.com/stanford-star/relational-transformer
# Licensed under the Apache License, Version 2.0; see LICENSE.

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D205

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, ClassVar, Literal, cast

import torch
from safetensors.torch import load_file
from torch import Tensor
from torch.nn import (
    Linear,
    ModuleDict,
    ModuleList,
    Parameter,
    ParameterDict,
    Sequential,
)

from sdm import (
    NaT,
    Recipe,
    RelatedTables,
    StringTensor,
    Stype,
    TableTensor,
    TaskLike,
)
from sdm._inference import inference_mode
from sdm.cache import Cache
from sdm.models._huggingface import download_checkpoint
from sdm.models.base import ICLModel
from sdm.models.relational_transformer.block import (
    RelationalBlock,
    attention_masks,
)
from sdm.models.relational_transformer.cells import (
    KINDS,
    TEXT,
    Cells,
    build_cells,
)
from sdm.models.relational_transformer.ckpt import remap_ckpt
from sdm.models.relational_transformer.recipe import default_recipe
from sdm.models.relational_transformer.text import TextEmbedding
from sdm.nn import RMSNorm

CHECKPOINTS = {
    "rt-j": ("stanford-star/rt-j", "4ee4c583895304cb3529120bf86fc9e04711b4b6"),
    "rt-plurel": (
        "stanford-star/rt-plurel",
        "38bf7b54114057eac2c87ffa02c0b892341f70ec",
    ),
}
TEXT_MODEL = "sentence-transformers/all-MiniLM-L12-v2"


class RelationalTransformer(ICLModel):
    r"""The relational foundation model from the `"RT-J: Large-Scale
    Pretraining of Relational Transformers for Context-Efficient Predictions"
    <https://arxiv.org/abs/2610.17867>`_ paper.

    :class:`RelationalTransformer` tokenizes task rows and their related
    tables into cells: every non-missing, non-identifier value is one token,
    embedded from its value and from a text embedding of its
    ``"<column> of <table>"`` description. Numbers are standardized by the
    recipe, datetimes are standardized over the context, and text values and
    column descriptions are embedded with ``all-MiniLM-L12-v2``. The target
    cells of query rows are masked, while context rows keep theirs as
    in-context examples. Twelve relational attention blocks attend along
    columns, within rows and their foreign-key parent rows, and from rows to
    the rows that reference them, before a linear head reads the masked cells
    out.

    Related tables are read as foreign keys from the left to the right table
    of each :class:`~sdm.Relationship`, and task links attach task rows to
    their entity rows. Context and query related tables form separate row
    sets. On CUDA, attention runs block-sparse through
    :func:`torch.nn.attention.flex_attention.flex_attention` over cells
    ordered by column, as in the reference implementation; the released
    checkpoints were pretrained with contexts of up to 8,192 cells.

    The reference evaluation protocol builds one context per query row with
    the sampler of the ``relational-transformer`` package. Such contexts enter
    the model through :meth:`predict_cells` after conversion with
    :meth:`~sdm.models.relational_transformer.Cells.from_rt_batch`.

    Classification is restricted to two classes: the model predicts the
    standardized class code and returns the logits ``[0, z]`` over the two
    classes, so that the default recipe's softmax yields ``sigmoid(z)`` for
    the second class. Regression returns the prediction in a column named
    ``"pred"``.

    .. code-block:: python

        from sdm import RelatedTables, TableTensor
        from sdm.models import RelationalTransformer

        task_table = TableTensor.from_columns(
            {"user_id": [0, 1, 2, 3], "churn": [True, False, True, False]},
            stypes={"user_id": "id", "churn": "categorical"},
            device="cuda",
        )

        related_tables = RelatedTables(
            tables={
                "users": TableTensor.from_columns(
                    {"user_id": [0, 1, 2, 3], "age": [42, 23, 31, 26]},
                    stypes={"user_id": "id", "age": "numerical"},
                    device="cuda",
                ),
                "orders": TableTensor.from_columns(
                    {
                        "user_id": [0, 0, 1, 3, 3, 3],
                        "amount": [9.99, 4.99, 12.99, 7.99, 3.99, 5.99],
                    },
                    stypes={"user_id": "id", "amount": "numerical"},
                    device="cuda",
                ),
            },
            relationships=[{
                "left_table": "orders",
                "left_columns": "user_id",
                "right_table": "users",
                "right_columns": "user_id",
            }],
            task_links=[{
                "task_columns": "user_id",
                "table": "users",
                "table_columns": "user_id",
            }],
        )

        x_context = task_table[:2].drop_columns("churn")
        y_context = task_table[:2, "churn"]
        x_query = task_table[2:].drop_columns("churn")

        related_context_tables = related_tables.replace_tables({
            "users": related_tables.tables["users"][:2],
            "orders": related_tables.tables["orders"][:3],
        })
        related_query_tables = related_tables.replace_tables({
            "users": related_tables.tables["users"][2:],
            "orders": related_tables.tables["orders"][3:],
        })

        model = RelationalTransformer(device="cuda")

        # Default in-context learning forward pass:
        out = model(
            x_context=x_context,
            y_context=y_context,
            x_query=x_query,
            related_context_tables=related_context_tables,
            related_query_tables=related_query_tables,
        )

        # Fit+Predict forward pass:
        model.fit(x_context, y_context, related_context_tables)
        out = model.predict(x_query, related_query_tables)

    Args:
        task: The tasks to initialize. If ``None``, all tasks supported by this
            model are initialized.
        checkpoint: The pretrained checkpoint: ``"rt-j"`` is pretrained on
            PluRel and the Join, ``"rt-plurel"`` on PluRel only.
        pretrained: Whether to load the pretrained checkpoint.
        device: The device.

    Additional Forward Parameters:
        table_name (str): The name of the task table in column descriptions.
            (default: ``"task"``)
        block_sparse (bool | None): Whether to run block-sparse flex
            attention. If ``None``, flex attention is used on CUDA and dense
            attention otherwise. (default: ``None``)
    """

    supported_feature_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical, Stype.datetime, Stype.text}
    )
    supported_target_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical, Stype.categorical}
    )
    supports_multi_target: ClassVar[bool] = False
    supports_context_only_columns: ClassVar[bool] = False
    supports_related_tables: ClassVar[bool] = True

    def __init__(
        self,
        task: TaskLike | Iterable[TaskLike] | None = None,
        checkpoint: Literal["rt-j", "rt-plurel"] = "rt-j",
        pretrained: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(task=task)

        self.model = _RelationalTransformer(
            device="meta" if pretrained else device
        )
        self.text_embedding: Callable[[StringTensor], Tensor] = TextEmbedding(
            TEXT_MODEL
        )

        if pretrained:
            self._load_from_pretrained(checkpoint, device=device)

        self.eval()

    @classmethod
    def default_recipe(cls) -> Recipe:
        r""":meta private:"""  # noqa: D415
        return default_recipe()

    def predict_cells(
        self,
        cells: Cells,  # [B, S]
        *,
        block_sparse: bool | None = None,
    ) -> Tensor:  # [B, R]
        r"""Predict the masked target cells of tokenized contexts.

        This is the entry point for contexts built outside the
        :class:`~sdm.tensor.TableTensor` interface, such as the per-query
        contexts of the ``relational-transformer`` sampler converted with
        :meth:`~sdm.models.relational_transformer.Cells.from_rt_batch`.

        Args:
            cells: The cells with shape ``[B, S]`` and ``R`` target cells per
                sequence.
            block_sparse: Whether to run block-sparse flex attention. If
                ``None``, flex attention is used on CUDA and dense attention
                otherwise.

        Returns:
            The standardized target predictions with shape ``[B, R]``.
        """
        with inference_mode("none" if self.training else "inference"):
            return self.model(cells, block_sparse=block_sparse)

    def _load_from_pretrained(
        self,
        checkpoint: Literal["rt-j", "rt-plurel"],
        device: torch.device | str | None,
    ) -> RelationalTransformer:
        device = torch.get_default_device() if device is None else device

        repo_id, revision = CHECKPOINTS[checkpoint]
        path = download_checkpoint(
            repo_id=repo_id,
            filename="model.safetensors",
            revision=revision,
            config_filename="config.json",
        )
        # The checkpoints store bfloat16 weights.
        state = remap_ckpt(load_file(path, device=str(device)))
        state = {key: value.float() for key, value in state.items()}
        self.model.load_state_dict(state, assign=True)

        return self

    def _forward(
        self,
        x_context: TableTensor | None,  # [..., R_context, D]
        y_context: TableTensor | None,  # [..., R_context, 1]
        x_query: TableTensor | None,  # [..., R_query, D]
        x_context_only: TableTensor | None,  # [..., R_context, D_only]
        related_context_tables: RelatedTables[TableTensor] | None,
        related_query_tables: RelatedTables[TableTensor] | None,
        cache: Cache | None,
        generator: torch.Generator | None,
        **kwargs: Any,
    ) -> TableTensor:  # [..., R_query, 1 or 2]

        table_name = cast(str, kwargs.get("table_name", "task"))

        if cache is not None and cache.is_replaying:
            assert x_query is not None
            batch_shape = x_query.size()[:-2]
            classes = cast(Tensor | None, cache["classes"])
            target_column = cast(str, cache["target_column"])
            contexts = cast(list[Cells], cache["context"])
            vocabularies = cast(list[dict[str, int]], cache["vocabulary"])
            datetime_stats = cast(
                list[tuple[Tensor, Tensor]], cache["datetime"]
            )
        else:
            assert x_context is not None
            assert y_context is not None
            batch_shape = x_context.size()[:-2]
            classes = None
            if y_context.categorical.size(-1) > 0:
                classes = y_context.categorical.categories[0]
                if len(classes) != 2:
                    raise ValueError(
                        f"{self.__class__.__name__!r} supports binary "
                        f"classification only (got {len(classes)} classes)"
                    )
                target_column = y_context.categorical_columns[0]
                labels = y_context.categorical.code.squeeze(-1)
                labels = labels.to(x_context.numerical.dtype)
            else:
                target_column = y_context.numerical_columns[0]
                labels = y_context.numerical.squeeze(-1)

            contexts, vocabularies, datetime_stats = [], [], []
            for x, y in zip(
                _members(x_context),
                labels.reshape(-1, labels.size(-1)).unbind(0),
                strict=True,
            ):
                if classes is not None:  # Standardize the class codes.
                    y = (y - y.mean()) / _scale(y)
                related = (
                    []
                    if related_context_tables is None
                    else related_context_tables.tables.values()
                )
                stats = _datetime_stats([x, *related])
                cells, vocabulary = build_cells(
                    x=x,
                    y=y,
                    related_tables=related_context_tables,
                    table_name=table_name,
                    target_column=target_column,
                    datetime_mean=stats[0],
                    datetime_scale=stats[1],
                    embed=self.text_embedding,
                )
                contexts.append(cells)
                vocabularies.append(vocabulary)
                datetime_stats.append(stats)

            if cache is not None and cache.is_recording:
                cache["target_column"] = target_column
                cache["context"] = contexts
                cache["vocabulary"] = vocabularies
                cache["datetime"] = datetime_stats

        if x_query is None:
            assert x_context is not None
            out = x_context.numerical.new_empty((*batch_shape, 0))
        else:
            members = []
            for context, vocabulary, stats, x in zip(
                contexts,
                vocabularies,
                datetime_stats,
                _members(x_query),
                strict=True,
            ):
                query, _ = build_cells(
                    x=x,
                    y=None,
                    related_tables=related_query_tables,
                    table_name=table_name,
                    target_column=target_column,
                    datetime_mean=stats[0],
                    datetime_scale=stats[1],
                    embed=self.text_embedding,
                    vocabulary=vocabulary,
                )
                members.append(context.cat(query))
            out = self.model(
                Cells.stack(members),
                block_sparse=cast(bool | None, kwargs.get("block_sparse")),
            )  # [E, R_query]
            out = out.view(*batch_shape, out.size(-1))

        if classes is None:
            return TableTensor(
                columns={Stype.numerical: ["pred"]},
                numerical=out.unsqueeze(-1),
            )

        return TableTensor(
            columns={Stype.numerical: [str(c) for c in classes.tolist()]},
            numerical=torch.stack([torch.zeros_like(out), out], dim=-1),
        )


class _RelationalTransformer(torch.nn.Module):
    def __init__(
        self,
        num_layers: int = 12,
        channels: int = 512,
        text_channels: int = 384,
        num_heads: int = 8,
        hidden_channels: int = 2048,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.encoders = ModuleDict(
            {
                kind: Sequential(
                    Linear(in_channels, channels, **factory_kwargs),
                    RMSNorm(channels, eps=1e-6, **factory_kwargs),
                )
                for kind, in_channels in (
                    ("column", text_channels),
                    ("number", 1),
                    ("text", text_channels),
                    ("datetime", 1),
                )
            }
        )
        self.mask_embeddings = ParameterDict(
            {
                kind: Parameter(torch.randn(channels, **factory_kwargs))
                for kind in KINDS
            }
        )
        self.blocks = ModuleList(
            [
                RelationalBlock(
                    channels=channels,
                    num_heads=num_heads,
                    hidden_channels=hidden_channels,
                    **factory_kwargs,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(channels, eps=1e-6, **factory_kwargs)
        self.head = Linear(channels, 1, **factory_kwargs)

    def forward(
        self,
        cells: Cells,  # [B, S]
        *,
        block_sparse: bool | None = None,
    ) -> Tensor:  # [B, R]
        if block_sparse is None:
            block_sparse = cells.node.is_cuda

        # Order cells by column so that column attention is block-local.
        pad = cells.node < 0
        perm = cells.column.masked_fill(pad, torch.iinfo(torch.long).max)
        perm = perm.argsort(dim=-1, stable=True)  # [B, S]
        inverse = torch.empty_like(perm).scatter_(
            -1,
            perm,
            torch.arange(perm.size(-1), device=perm.device).expand_as(perm),
        )
        node = cells.node.gather(-1, perm)
        parents = cells.parents.gather(
            -2, perm.unsqueeze(-1).expand_as(cells.parents)
        )
        column = cells.column.gather(-1, perm)
        kind = cells.kind.gather(-1, perm)
        value = cells.value.gather(-1, perm)
        text = cells.text.gather(-1, perm)
        is_target = cells.is_target.gather(-1, perm).unsqueeze(-1)  # [B, S, 1]
        target = inverse.gather(-1, cells.target)  # [B, R]

        dtype = self.head.weight.dtype
        embedding = cells.embedding.to(dtype)
        x = self.encoders["column"](embedding[column])
        for index, name in enumerate(KINDS):
            if index == TEXT:
                inputs = embedding[text.clamp(min=0)]
            else:
                inputs = value.unsqueeze(-1).to(dtype)
            is_kind = (kind == index).unsqueeze(-1)
            x = x + self.encoders[name](inputs) * (is_kind & ~is_target)
            x = x + self.mask_embeddings[name] * (is_kind & is_target)
        x = x * ~pad.gather(-1, perm).unsqueeze(-1)  # [B, S, C]

        attn_masks = attention_masks(
            node, parents, column, block_sparse=block_sparse
        )
        for block in self.blocks:
            x = block(x, attn_masks)
        x = self.norm(x)

        out = self.head(x).squeeze(-1)  # [B, S]
        return out.gather(-1, target)  # [B, R]


def _members(x: TableTensor) -> list[TableTensor]:
    # Flatten the leading dimensions into separate two-dimensional tables.
    if x.dim() == 2:
        return [x]
    return [cast(TableTensor, member) for member in x.flatten(0, -3).unbind(0)]


def _scale(values: Tensor) -> Tensor:
    scale = values.std().nan_to_num(nan=1.0)
    return scale.masked_fill(scale == 0, 1.0)


def _datetime_stats(tables: Iterable[TableTensor]) -> tuple[Tensor, Tensor]:
    # Standardize all datetime values of a context jointly, in nanoseconds.
    values = torch.cat(
        [table.datetime[table.datetime != NaT] for table in tables]
    ).double()
    values *= 1_000
    if values.numel() == 0:
        return values.new_zeros(()), values.new_ones(())
    return values.mean(), _scale(values)
