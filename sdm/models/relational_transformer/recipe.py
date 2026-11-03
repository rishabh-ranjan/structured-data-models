# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import torch

import sdm.processing as sp
from sdm import CategoricalTensor, StringTensor, Stype, TableTensor


def decode_categories(table: TableTensor) -> TableTensor:
    r"""Move string categories to text and other categories to numerical.

    Args:
        table: The table whose categorical columns are decoded. Missing values
            become null strings and ``NaN``, respectively.
    """
    numerical: list[torch.Tensor] = []
    numerical_columns: list[str] = []
    text: list[torch.Tensor] = []
    text_columns: list[str] = []
    for i, column in enumerate(table.categorical_columns):
        categories = table.categorical.categories[i]
        if isinstance(categories, StringTensor):
            column_tensor = table.categorical[..., i : i + 1]
            text.append(cast(CategoricalTensor, column_tensor).decode())
            text_columns.append(column)
            continue
        code = table.categorical.code[..., i]
        values = categories.to(table.numerical.dtype)[code.clamp(min=0)]
        numerical.append(values.masked_fill(code < 0, torch.nan))
        numerical_columns.append(column)

    out = table.drop_stypes(Stype.categorical)
    if numerical:
        numerical_table = TableTensor(
            columns={Stype.numerical: numerical_columns},
            numerical=torch.stack(numerical, dim=-1),
        )
        out = cast(TableTensor, torch.cat([out, numerical_table], dim=-1))
    if text:
        text_table = TableTensor(
            columns={Stype.text: text_columns},
            text=cast(StringTensor, torch.cat(text, dim=-1)),
        )
        out = cast(TableTensor, torch.cat([out, text_table], dim=-1))
    return out


def default_recipe() -> sp.Recipe:  # noqa: D103
    return sp.Recipe(
        features=[
            sp.StypeDispatch(categorical=decode_categories),
            sp.StypeDispatch(numerical=sp.Standardize()),
        ],
        target=sp.StypeDispatch(
            categorical=sp.AlignCategories(sort_by="value"),
            numerical=sp.Standardize(),
        ),
        output=[
            sp.AverageEstimators(),
            sp.TaskDispatch(classification=sp.Softmax()),
        ],
    )
