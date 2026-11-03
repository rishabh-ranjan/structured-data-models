# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import cast

from torch import Tensor

from sdm import StringTensor, TableTensor
from sdm.processing import SentenceTransformer


class TextEmbedding:
    r"""Embed strings with a sentence transformer loaded on first use.

    Args:
        model_name: Model name or local path passed to
            :class:`~sdm.processing.SentenceTransformer`.
        batch_size: The number of strings embedded per batch.
    """

    def __init__(self, model_name: str, *, batch_size: int = 256) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._processor: SentenceTransformer | None = None

    def __call__(self, text: StringTensor) -> Tensor:  # [T, D]
        r"""Embed a one-dimensional string tensor into ``[T, D]``."""
        if self._processor is None:
            self._processor = SentenceTransformer(
                self.model_name,
                batch_size=self.batch_size,
            )
        processor = self._processor.to(text.device)
        table = TableTensor(
            columns={"text": ("text",)},
            text=cast(StringTensor, text.view(-1, 1)),
        )
        return processor.transform(table).numerical
