# Portions derived from Relational Transformer:
# https://github.com/stanford-star/relational-transformer
# Licensed under the Apache License, Version 2.0; see LICENSE.

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from itertools import pairwise
from typing import NamedTuple, Self

import numpy as np
import pyarrow as pa
import torch
import torch.nn.functional as F
from torch import Tensor

from sdm import NaT, RelatedTables, StringTensor, TableTensor
from sdm.relational.join import join_index
from sdm.tensor.mixin import DeviceMixin

KINDS = ("number", "text", "datetime")
NUMBER, TEXT, DATETIME = range(3)
NANOSECONDS_PER_MICROSECOND = 1_000


class _Cells(NamedTuple):
    node: Tensor
    parents: Tensor
    column: Tensor
    kind: Tensor
    value: Tensor
    text: Tensor
    is_target: Tensor
    target: Tensor
    embedding: Tensor
    num_rows: int


_TENSOR_FIELDS = _Cells._fields[:-1]


class Cells(_Cells, DeviceMixin):
    r"""Cell tokens of task rows and their related tables.

    Every non-missing, non-identifier value of a row is one cell. A cell
    carries the row it belongs to, the foreign-key parent rows of that row,
    its column, its kind, and its value.

    Args:
        node: The row of each cell with shape ``[..., S]``; ``-1`` marks
            padding.
        parents: The parent rows of each cell's row with shape
            ``[..., S, K]``, padded with ``-1``.
        column: The index of each cell's column description in ``embedding``
            with shape ``[..., S]``.
        kind: The kind of each cell with shape ``[..., S]``: ``0`` for
            numbers, ``1`` for text, and ``2`` for datetimes.
        value: The number or standardized datetime value of each cell with
            shape ``[..., S]``.
        text: The index of each text value in ``embedding`` with shape
            ``[..., S]``; ``-1`` for other cells.
        is_target: Whether each cell is a masked prediction target with
            shape ``[..., S]``.
        target: The positions of the query target cells with shape
            ``[..., R]``.
        embedding: Text embeddings of column descriptions and text values
            with shape ``[T, D]``.
        num_rows: The number of rows across all tables.
    """

    def _tensors(self) -> Iterator[Tensor]:
        for name in _TENSOR_FIELDS:
            yield getattr(self, name)

    def _apply_tensor(self, fn: Callable[[Tensor], Tensor]) -> Self:
        return self.__class__(
            **{name: fn(getattr(self, name)) for name in _TENSOR_FIELDS},
            num_rows=self.num_rows,
        )

    def cat(self, other: Cells) -> Cells:
        r"""Append the cells of ``other``.

        Args:
            other: One-dimensional cells whose rows follow this object's rows
                and whose text indices continue beyond this object's
                ``embedding``.
        """
        num_parents = max(self.parents.size(-1), other.parents.size(-1))
        node = other.node.masked_fill(other.node < 0, -self.num_rows - 1)
        parents = other.parents.masked_fill(
            other.parents < 0, -self.num_rows - 1
        )
        return Cells(
            node=torch.cat([self.node, node + self.num_rows]),
            parents=torch.cat(
                [
                    _pad(self.parents, num_parents),
                    _pad(parents + self.num_rows, num_parents),
                ]
            ),
            column=torch.cat([self.column, other.column]),
            kind=torch.cat([self.kind, other.kind]),
            value=torch.cat([self.value, other.value]),
            text=torch.cat([self.text, other.text]),
            is_target=torch.cat([self.is_target, other.is_target]),
            target=torch.cat([self.target, other.target + self.node.numel()]),
            embedding=torch.cat([self.embedding, other.embedding]),
            num_rows=self.num_rows + other.num_rows,
        )

    @classmethod
    def stack(cls, members: Sequence[Cells]) -> Cells:
        r"""Stack one-dimensional cells into a padded batch.

        Args:
            members: The cells to stack along a new leading dimension.
        """
        num_cells = max(member.node.numel() for member in members)
        num_parents = max(member.parents.size(-1) for member in members)
        num_targets = max(member.target.numel() for member in members)
        offset = 0
        fields: dict[str, list[Tensor]] = {name: [] for name in _TENSOR_FIELDS}
        for member in members:
            fields["node"].append(_pad(member.node, num_cells))
            fields["parents"].append(
                _pad(_pad(member.parents, num_parents), num_cells, dim=-2)
            )
            fields["column"].append(
                _pad(member.column + offset, num_cells, fill_value=0)
            )
            fields["kind"].append(_pad(member.kind, num_cells, fill_value=0))
            fields["value"].append(
                _pad(member.value, num_cells, fill_value=0.0)
            )
            text = member.text.masked_fill(member.text < 0, -offset - 1)
            fields["text"].append(_pad(text + offset, num_cells))
            fields["is_target"].append(
                _pad(member.is_target, num_cells, fill_value=False)
            )
            fields["target"].append(
                _pad(member.target, num_targets, fill_value=0)
            )
            fields["embedding"].append(member.embedding)
            offset += member.embedding.size(0)
        return cls(
            **{
                name: torch.stack(tensors)
                for name, tensors in fields.items()
                if name != "embedding"
            },
            embedding=torch.cat(fields["embedding"]),
            num_rows=max(member.num_rows for member in members),
        )

    @classmethod
    def from_rt_batch(cls, batch: Mapping[str, Tensor]) -> Cells:
        r"""Create cells from a ``relational-transformer`` sampler batch.

        Args:
            batch: A batch as returned by ``rt.data.process_batch`` with
                ``node_idxs``, ``f2p_nbr_idxs``, ``col_name_idxs``,
                ``col_name_values``, ``sem_types``, ``number_values``,
                ``text_values``, ``datetime_values``, ``is_targets`` and
                ``is_padding`` entries of shape ``[B, S, ...]``, with the same
                number of target cells in every sequence.
        """
        is_padding = batch["is_padding"].bool()
        node = batch["node_idxs"].long().masked_fill(is_padding, -1)
        column = batch["col_name_idxs"].long()
        kind = batch["sem_types"].long()
        is_target = batch["is_targets"].bool()
        B, S = node.size()

        # One embedding row per column, then one per cell for text values.
        column_values = batch["col_name_values"]  # [B, S, D]
        num_columns = int(column.max()) + 1
        embedding = column_values.new_zeros(
            (num_columns + B * S, column_values.size(-1))
        )
        embedding[column.flatten()] = column_values.flatten(0, 1)
        embedding[num_columns:] = batch["text_values"].flatten(0, 1)
        text = torch.arange(B * S, device=node.device).view(B, S) + num_columns

        return cls(
            node=node,
            parents=batch["f2p_nbr_idxs"].long(),
            column=column,
            kind=kind,
            value=torch.where(
                kind == DATETIME,
                batch["datetime_values"].squeeze(-1),
                batch["number_values"].squeeze(-1),
            ),
            text=text.masked_fill(kind != TEXT, -1),
            is_target=is_target,
            target=is_target.nonzero()[:, 1].view(B, -1),
            embedding=embedding,
            num_rows=int(node.max()) + 1,
        )


def _pad(
    tensor: Tensor,
    size: int,
    dim: int = -1,
    fill_value: float = -1,
) -> Tensor:
    padding = [0, 0] * (-dim - 1) + [0, size - tensor.size(dim)]
    return F.pad(tensor, padding, value=fill_value)


def build_cells(
    x: TableTensor,  # [R, D]
    y: Tensor | None,  # [R]
    related_tables: RelatedTables[TableTensor] | None,
    *,
    table_name: str,
    target_column: str,
    datetime_mean: Tensor,
    datetime_scale: Tensor,
    embed: Callable[[StringTensor], Tensor],
    vocabulary: Mapping[str, int] | None = None,
) -> tuple[Cells, dict[str, int]]:
    r"""Tokenize task rows and their related tables into :class:`Cells`.

    Args:
        x: The task rows with shape ``[R, D]``.
        y: The target values of the task rows with shape ``[R]``, or
            ``None`` to mask the targets of query rows.
        related_tables: Related tables whose relationships are read as
            foreign keys from the left to the right table.
        table_name: The name of the task table in column descriptions.
        target_column: The name of the target column.
        datetime_mean: The mean of datetime values in nanoseconds.
        datetime_scale: The standard deviation of datetime values in
            nanoseconds.
        embed: Embeds a one-dimensional :class:`~sdm.StringTensor` into
            ``[T, D]``.
        vocabulary: Indices of already embedded strings. The returned
            ``embedding`` holds only the strings missing from it, in index
            order.

    Returns:
        The cells and the vocabulary of all strings.
    """
    device = x.device
    dtype = x.numerical.dtype
    num_rows = x.size(-2)
    tables: list[tuple[str, TableTensor, int]] = [(table_name, x, 0)]
    offsets: dict[str, int] = {}
    if related_tables is not None:
        for name, table in related_tables.tables.items():
            offsets[name] = num_rows
            tables.append((name, table, num_rows))
            num_rows += table.size(-2)

    descriptions = [f"{target_column} of {table_name}"]
    texts: list[pa.Array] = []
    for name, table, _ in tables:
        descriptions.extend(
            f"{column} of {name}"
            for column in (
                *table.numerical_columns,
                *table.text_columns,
                *table.datetime_columns,
            )
        )
        texts.append(table.text.to_arrow())
    vocabulary = dict(vocabulary or {})
    codes, new = _encode(descriptions, texts, vocabulary)
    description = torch.as_tensor(codes[0], device=device)

    nodes: list[Tensor] = []
    columns: list[Tensor] = []
    kinds: list[Tensor] = []
    values: list[Tensor] = []
    text_indices: list[Tensor] = []
    pointer = 1
    for (_, table, offset), text_codes in zip(tables, codes[1:]):
        size = (
            len(table.numerical_columns)
            + len(table.text_columns)
            + len(table.datetime_columns)
        )
        for block in _table_cells(
            table=table,
            offset=offset,
            description=description[pointer : pointer + size],
            text_codes=torch.as_tensor(text_codes, device=device).view(
                table.text.size()
            ),
            datetime_mean=datetime_mean,
            datetime_scale=datetime_scale,
        ):
            node, column, kind, value, text = block
            nodes.append(node)
            columns.append(column)
            kinds.append(kind)
            values.append(value)
            text_indices.append(text)
        pointer += size

    # Target cells of the task rows come last.
    arange = torch.arange(x.size(-2), device=device)
    nodes.append(arange)
    columns.append(description[0].expand(arange.numel()))
    kinds.append(torch.full_like(arange, NUMBER))
    values.append(
        y if y is not None else torch.zeros_like(arange, dtype=dtype)
    )
    text_indices.append(torch.full_like(arange, -1))
    node = torch.cat(nodes)
    is_target = torch.zeros_like(node, dtype=torch.bool)
    if y is None:
        is_target[-arange.numel() :] = True

    return Cells(
        node=node,
        parents=_parents(x, related_tables, offsets, num_rows)[node],
        column=torch.cat(columns),
        kind=torch.cat(kinds),
        value=torch.cat(values).to(dtype),
        text=torch.cat(text_indices),
        is_target=is_target,
        target=node.numel()
        - arange.numel()
        + arange[: 0 if y is not None else None],
        embedding=embed(StringTensor.from_arrow(new, device=device)),
        num_rows=num_rows,
    ), vocabulary


def _encode(
    descriptions: Sequence[str],
    texts: Sequence[pa.Array],
    vocabulary: dict[str, int],
) -> tuple[list[np.ndarray], pa.Array]:
    # Dictionary-encode all strings at once; new strings extend `vocabulary`.
    arrays = [pa.array(descriptions, type=pa.large_string())]
    arrays.extend(text.cast(pa.large_string()) for text in texts)
    encoded = pa.concat_arrays(arrays).dictionary_encode()
    new: list[str] = []
    index = np.empty(len(encoded.dictionary), dtype=np.int64)
    for i, string in enumerate(encoded.dictionary.to_pylist()):
        position = vocabulary.get(string)
        if position is None:
            position = len(vocabulary)
            vocabulary[string] = position
            new.append(string)
        index[i] = position
    codes = encoded.indices.fill_null(-1).to_numpy(zero_copy_only=False)
    codes = np.where(codes >= 0, index[np.maximum(codes, 0)], -1)
    sizes = np.cumsum([0, *(len(array) for array in arrays)])
    return [codes[start:end] for start, end in pairwise(sizes)], pa.array(
        new, type=pa.large_string()
    )


def _table_cells(
    table: TableTensor,  # [R, D]
    offset: int,
    description: Tensor,  # [D - D_id]
    text_codes: Tensor,  # [R, D_text]
    datetime_mean: Tensor,
    datetime_scale: Tensor,
) -> Iterator[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
    # Yield (node, column, kind, value, text) per kind in row-major order.
    num_numerical = table.numerical.size(-1)
    num_text = table.text.size(-1)
    blocks = (
        (NUMBER, table.numerical.isfinite(), description[:num_numerical]),
        (TEXT, text_codes >= 0, description[num_numerical:][:num_text]),
        (
            DATETIME,
            table.datetime != NaT,
            description[num_numerical + num_text :],
        ),
    )
    for kind, valid, column in blocks:
        row, col = valid.nonzero(as_tuple=True)
        if kind == NUMBER:
            value = table.numerical[row, col]
            text = torch.full_like(row, -1)
        elif kind == TEXT:
            value = torch.zeros_like(row, dtype=table.numerical.dtype)
            text = text_codes[row, col]
        else:
            value = table.datetime[row, col].double()
            value *= NANOSECONDS_PER_MICROSECOND
            value = (value - datetime_mean) / datetime_scale
            text = torch.full_like(row, -1)
        yield (
            row + offset,
            column[col],
            torch.full_like(row, kind),
            value,
            text,
        )


def _parents(
    x: TableTensor,
    related_tables: RelatedTables[TableTensor] | None,
    offsets: Mapping[str, int],
    num_rows: int,
) -> Tensor:  # [num_rows, K]
    # Foreign keys point from task rows to entity rows and from the left to
    # the right table of each relationship.
    children: list[Tensor] = []
    parents: list[Tensor] = []
    if related_tables is not None:
        for link in related_tables.task_links:
            child, parent = join_index(
                left_table=x,
                right_table=related_tables.tables[link.table],
                left_keys=link.task_columns,
                right_keys=link.table_columns,
            )
            children.append(child)
            parents.append(parent + offsets[link.table])
        for rel in related_tables.relationships:
            child, parent = join_index(
                left_table=related_tables.tables[rel.left_table],
                right_table=related_tables.tables[rel.right_table],
                left_keys=rel.left_columns,
                right_keys=rel.right_columns,
            )
            children.append(child + offsets[rel.left_table])
            parents.append(parent + offsets[rel.right_table])

    if len(children) == 0:
        return torch.empty((num_rows, 0), dtype=torch.long, device=x.device)

    child, perm = torch.cat(children).sort(stable=True)
    parent = torch.cat(parents)[perm]
    count = torch.bincount(child, minlength=num_rows)
    start = count.cumsum(0) - count
    position = torch.arange(child.numel(), device=child.device) - start[child]
    out = child.new_full((num_rows, int(count.max())), -1)
    out[child, position] = parent
    return out
