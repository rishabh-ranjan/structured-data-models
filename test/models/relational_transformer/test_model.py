# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import hashlib

import pandas as pd
import pytest
import torch
from torch import Tensor

from sdm import (
    CategoricalTensor,
    RelatedTables,
    RelationalData,
    StringTensor,
    TableTensor,
)
from sdm.models import RelationalTransformer
from sdm.models.relational_transformer import model as rt_module
from sdm.models.relational_transformer.block import (
    AttentionMask,
    RelationalBlock,
    attention_masks,
)
from sdm.models.relational_transformer.cells import (
    DATETIME,
    NUMBER,
    TEXT,
    Cells,
    build_cells,
)
from sdm.models.relational_transformer.ckpt import remap_ckpt
from sdm.models.relational_transformer.recipe import decode_categories
from sdm.testing import onlyCUDA, withCUDA

CHANNELS = 64  # Flex attention requires at least 16 channels per head.
TEXT_CHANNELS = 16


class HashEmbedding:
    """Deterministic stand-in for the sentence transformer."""

    def __init__(self, channels: int) -> None:
        self.channels = channels
        self.strings: list[list[str]] = []

    def __call__(self, text: StringTensor) -> Tensor:
        strings = text.to_arrow().to_pylist()
        self.strings.append(strings)
        out = torch.empty(len(strings), self.channels)
        for i, string in enumerate(strings):
            seed = int(hashlib.sha256(string.encode()).hexdigest()[:8], 16)
            generator = torch.Generator().manual_seed(seed)
            out[i] = torch.randn(self.channels, generator=generator)
        return out.to(text.device)


@pytest.fixture
def model(
    monkeypatch: pytest.MonkeyPatch,
    device: torch.device,
) -> RelationalTransformer:
    monkeypatch.setattr(
        rt_module,
        "_RelationalTransformer",
        functools.partial(
            rt_module._RelationalTransformer,
            num_layers=2,
            channels=CHANNELS,
            text_channels=TEXT_CHANNELS,
            num_heads=4,
            hidden_channels=64,
        ),
    )
    model = RelationalTransformer(pretrained=False, device=device)
    # Make predictions sensitive to attention outputs.
    for parameter in model.parameters():
        if not parameter.any():
            torch.nn.init.normal_(parameter, std=0.02)
    model.text_embedding = HashEmbedding(TEXT_CHANNELS)
    return model


@pytest.fixture
def task_table(device: torch.device) -> TableTensor:
    return TableTensor.from_pandas(
        df=pd.DataFrame(
            {
                "user_id": [0, 1, 2, 3],
                "timestamp": pd.to_datetime(
                    ["2024-01-03", "2024-01-04", None, "2024-01-06"]
                ),
            }
        ),
        stypes={"user_id": "id", "timestamp": "datetime"},
        device=device,
    )


@pytest.fixture
def related_tables(
    relational_data: RelationalData,
) -> RelatedTables[TableTensor]:
    return RelatedTables(
        tables=relational_data.tables,
        relationships=relational_data.relationships,
        task_links=[
            {
                "task_column": "user_id",
                "table": "users",
                "table_column": "user_id",
            }
        ],
    )


def test_attention_masks() -> None:
    # Rows: 0 and 1 reference row 2; row 3 is a padding cell.
    node = torch.tensor([[0, 0, 1, 2, -1]])
    parents = torch.tensor([[[2], [2], [2], [-1], [-1]]])
    column = torch.tensor([[0, 1, 0, 1, 0]])

    masks = attention_masks(node, parents, column, block_sparse=False)

    expected = {
        "column": [
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        "feature": [
            [1, 1, 0, 1, 0],
            [1, 1, 0, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        "neighbor": [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ],
    }
    for name, value in expected.items():
        assert masks[name].mask.size() == (1, 5, 5)
        assert masks[name].mask[0].equal(torch.tensor(value, dtype=torch.bool))
        assert masks[name].key_len[0].tolist() == [sum(row) for row in value]
        assert masks[name].block_mask is None


@withCUDA
def test_block_skips_attention_without_keys(device: torch.device) -> None:
    block = RelationalBlock(
        channels=CHANNELS,
        num_heads=4,
        hidden_channels=64,
        device=device,
    )
    for parameter in block.parameters():
        if not parameter.any():
            torch.nn.init.normal_(parameter, std=0.02)

    S = 6
    x = torch.randn(1, S, CHANNELS, device=device)
    full = torch.ones(1, S, S, dtype=torch.bool, device=device)
    none = torch.zeros(1, S, S, dtype=torch.bool, device=device)
    masks = {
        name: AttentionMask(mask=mask, key_len=mask.sum(-1), block_mask=None)
        for name, mask in (
            ("column", full),
            ("feature", full),
            ("neighbor", none),
        )
    }
    out = block(x, masks)

    expected = x
    for name in ("column", "feature"):
        expected = expected + block.attns[name](
            block.norms[name](expected), masks[name]
        )
    expected = expected + block.mlp(block.norms["mlp"](expected))
    torch.testing.assert_close(out, expected)


def test_attention_matches_upstream_formulation() -> None:
    # The upstream attention of one type, written out as in the paper code.
    channels, num_heads = CHANNELS, 4
    head_channels = channels // num_heads
    weights = {
        key: torch.randn(channels, channels) * 0.1
        for key in ("wq", "wk", "wv", "wo", "wg")
    }
    q_norm, k_norm = torch.rand(head_channels), torch.rand(head_channels)
    scale = torch.rand(1, num_heads, 1, 1)

    S = 7
    x = torch.randn(1, S, channels)
    mask = torch.rand(1, S, S) < 0.5
    mask |= torch.eye(S, dtype=torch.bool)

    def rms_norm(value: Tensor, weight: Tensor) -> Tensor:
        return (
            value
            * (value.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
            * (weight)
        )

    q, k, v = (
        (x @ weights[key].T)
        .view(1, S, num_heads, head_channels)
        .transpose(1, 2)
        for key in ("wq", "wk", "wv")
    )
    q = (
        rms_norm(q, q_norm)
        * scale
        * mask.sum(-1).float().log().view(1, 1, S, 1)
    )
    k = rms_norm(k, k_norm)
    logits = (q @ k.transpose(-1, -2)) / head_channels
    probs = logits.masked_fill(~mask.unsqueeze(1), -torch.inf).softmax(-1)
    out = (probs @ v).transpose(1, 2).reshape(1, S, channels)
    expected = (2 * torch.sigmoid(x @ weights["wg"].T) * out) @ weights["wo"].T

    ckpt = {
        "blocks.0.attns.col.q_norm.scale": q_norm,
        "blocks.0.attns.col.k_norm.scale": k_norm,
        "blocks.0.attns.col.scale": scale,
        **{
            f"blocks.0.attns.col.{key}.weight": w for key, w in weights.items()
        },
    }
    block = RelationalBlock(channels, num_heads, hidden_channels=64)
    state = {
        key.removeprefix("blocks.0.attns.column."): value
        for key, value in remap_ckpt(_upstream_state(ckpt)).items()
        if key.startswith("blocks.0.attns.column.")
    }
    block.attns["column"].load_state_dict(state)

    attn_mask = AttentionMask(mask=mask, key_len=mask.sum(-1), block_mask=None)
    torch.testing.assert_close(block.attns["column"](x, attn_mask), expected)


def _upstream_state(overrides: dict[str, Tensor]) -> dict[str, Tensor]:
    # A complete single-block upstream state dict with random values.
    channels, text_channels, hidden = CHANNELS, TEXT_CHANNELS, 64
    state: dict[str, Tensor] = {}
    for kind, in_channels in (
        ("col_name", text_channels),
        ("number", 1),
        ("text", text_channels),
        ("datetime", 1),
    ):
        state[f"enc_dict.{kind}.weight"] = torch.randn(channels, in_channels)
        state[f"enc_dict.{kind}.bias"] = torch.randn(channels)
        state[f"norm_dict.{kind}.scale"] = torch.rand(channels)
    for kind, out_channels in (
        ("number", 1),
        ("text", text_channels),
        ("datetime", 1),
    ):
        state[f"dec_dict.{kind}.weight"] = torch.randn(out_channels, channels)
        state[f"dec_dict.{kind}.bias"] = torch.randn(out_channels)
        state[f"mask_embs.{kind}"] = torch.randn(channels)
    for attn in ("col", "feat", "nbr"):
        for key in ("wq", "wk", "wv", "wo", "wg"):
            state[f"blocks.0.attns.{attn}.{key}.weight"] = torch.randn(
                channels, channels
            )
        state[f"blocks.0.attns.{attn}.q_norm.scale"] = torch.rand(
            channels // 4
        )
        state[f"blocks.0.attns.{attn}.k_norm.scale"] = torch.rand(
            channels // 4
        )
        state[f"blocks.0.attns.{attn}.scale"] = torch.rand(1, 4, 1, 1)
        state[f"blocks.0.norms.{attn}.scale"] = torch.rand(channels)
    state["blocks.0.norms.ffn.scale"] = torch.rand(channels)
    state["blocks.0.ffn.w1.weight"] = torch.randn(hidden, channels)
    state["blocks.0.ffn.w2.weight"] = torch.randn(channels, hidden)
    state["blocks.0.ffn.w3.weight"] = torch.randn(hidden, channels)
    state["norm_out.scale"] = torch.rand(channels)
    state.update(overrides)
    return state


def test_remap_ckpt_loads_strictly() -> None:
    model = rt_module._RelationalTransformer(
        num_layers=1,
        channels=CHANNELS,
        text_channels=TEXT_CHANNELS,
        num_heads=4,
        hidden_channels=64,
    )
    model.load_state_dict(remap_ckpt(_upstream_state({})))


def test_decode_categories() -> None:
    table = TableTensor(
        columns={"categorical": ("flag", "level", "color")},
        categorical=CategoricalTensor(
            code=torch.tensor([[0, 2, 1], [1, -1, 0], [-1, 0, -1]]),
            categories=(
                torch.tensor([False, True]),
                torch.tensor([10, 20, 30]),
                StringTensor.from_list(["red", "blue"]),
            ),
        ),
    )
    out = decode_categories(table)
    assert out.numerical_columns == ("flag", "level")
    assert out.text_columns == ("color",)
    torch.testing.assert_close(
        out.numerical,
        torch.tensor([[0.0, 30.0], [1.0, torch.nan], [torch.nan, 10.0]]),
        equal_nan=True,
    )
    assert out.text.tolist() == [["blue"], ["red"], [None]]


@withCUDA
def test_build_cells(
    task_table: TableTensor,
    related_tables: RelatedTables[TableTensor],
    device: torch.device,
) -> None:
    embed = HashEmbedding(TEXT_CHANNELS)
    related_tables = related_tables.replace_tables(
        {
            name: decode_categories(table)
            for name, table in related_tables.tables.items()
        }
    )
    mean = torch.zeros((), dtype=torch.float64, device=device)
    scale = torch.ones((), dtype=torch.float64, device=device)
    y = torch.tensor([1.0, -1.0, 1.0, -1.0], device=device)

    cells, vocabulary = build_cells(
        x=task_table,
        y=y,
        related_tables=related_tables,
        table_name="churn",
        target_column="label",
        datetime_mean=mean,
        datetime_scale=scale,
        embed=embed,
    )

    # 3 task timestamps + 4 ages + 4 cities + 6 amounts + 5 timestamps
    # + 3 categories, plus 4 target cells.
    assert cells.node.size() == (29,)
    assert cells.num_rows == 4 + 4 + 6 + 3
    assert cells.target.numel() == 0
    assert not cells.is_target.any()
    assert cells.embedding.size() == (len(vocabulary), TEXT_CHANNELS)
    assert vocabulary["label of churn"] == 0
    assert "NYC" in vocabulary
    assert "timestamp of orders" in vocabulary
    assert (cells.column[-4:] == 0).all()
    torch.testing.assert_close(cells.value[-4:], y)

    # Task rows reference users, orders reference users and items.
    users, orders, items = 4, 8, 14
    assert cells.parents.size(-1) == 2
    assert cells.parents[cells.node == 0].unique().tolist() == [-1, users]
    assert sorted(cells.parents[cells.node == orders].unique().tolist()) == [
        users,
        items,
    ]
    assert (cells.parents[cells.node == users] == -1).all()

    kinds = {
        NUMBER: 4 + 6 + 4,
        TEXT: 4 + 3,
        DATETIME: 3 + 5,
    }
    for kind, count in kinds.items():
        assert int((cells.kind == kind).sum()) == count
    assert (cells.text[cells.kind == TEXT] >= 0).all()
    assert (cells.text[cells.kind != TEXT] == -1).all()
    assert cells.value.isfinite().all()

    query, _ = build_cells(
        x=task_table[:2],
        y=None,
        related_tables=related_tables,
        table_name="churn",
        target_column="label",
        datetime_mean=mean,
        datetime_scale=scale,
        embed=embed,
        vocabulary=vocabulary,
    )
    assert query.embedding.size(0) == 0
    assert query.is_target.tolist()[-2:] == [True, True]
    assert query.target.tolist() == [
        query.node.numel() - 2,
        query.node.numel() - 1,
    ]

    both = cells.cat(query)
    assert both.num_rows == cells.num_rows + query.num_rows
    assert both.node[-1] == cells.num_rows + 1
    assert both.target.tolist() == [
        both.node.numel() - 2,
        both.node.numel() - 1,
    ]
    assert both.embedding.size(0) == cells.embedding.size(0)

    batch = Cells.stack([both, cells])
    assert batch.node.size() == (2, both.node.numel())
    assert batch.embedding.size(0) == 2 * cells.embedding.size(0)
    assert (batch.node[1, cells.node.numel() :] == -1).all()
    assert batch.column[1, 0] == cells.column[0] + cells.embedding.size(0)
    assert batch.to(device).device == device


@withCUDA
@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_forward(
    model: RelationalTransformer,
    task_table: TableTensor,
    related_tables: RelatedTables[TableTensor],
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if device.type == "cpu":
        assert repr(model) == "RelationalTransformer()"
    else:
        assert repr(model) == "RelationalTransformer(device=cuda:0)"

    if dtype.is_floating_point:
        y = TableTensor(numerical=torch.randn(4, 1, device=device))
    else:
        y = TableTensor.from_columns(
            {"churn": [True, False, True, False]},
            stypes={"churn": "categorical"},
            device=device,
        )

    out = model(
        x_context=task_table,
        y_context=y,
        x_query=task_table[:3],
        related_context_tables=related_tables,
        related_query_tables=related_tables,
        table_name="churn",
    )
    assert out.dtype == task_table.dtype
    assert out.device == device
    assert torch.is_inference(out)
    if dtype.is_floating_point:
        assert out.numerical_columns == ("pred",)
        assert out.size() == (3, 1)
    else:
        assert out.numerical_columns == ("False", "True")
        assert out.size() == (3, 2)
        torch.testing.assert_close(
            out.numerical.sum(dim=-1),
            torch.ones(3, device=device),
        )

    model.fit(task_table, y, related_tables, table_name="churn")
    assert model._cache is not None
    assert model._cache.size() > 0
    pred = model.predict(task_table[:3], related_tables)
    assert pred.allclose(out, atol=1e-5, rtol=1e-4)
    model.clear()


def test_multiclass_target_raises(
    model: RelationalTransformer,
    task_table: TableTensor,
) -> None:
    y = TableTensor.from_columns(
        {"level": ["a", "b", "c", "a"]},
        stypes={"level": "categorical"},
    )
    with pytest.raises(ValueError, match="binary classification"):
        model(x_context=task_table, y_context=y, x_query=task_table)


@pytest.mark.parametrize("estimator_batch_size", [1, 2, None])
def test_num_estimators(
    model: RelationalTransformer,
    estimator_batch_size: int | None,
) -> None:
    R_context, R_query, C = 6, 3, 4
    x_context = torch.randn(3, R_context, C)
    y_context = torch.randn(3, R_context, 1)
    x_query = torch.randn(R_query, C).expand(3, R_query, C)

    out = model(
        x_context=x_context,
        y_context=y_context,
        x_query=x_query,
        num_estimators=None,
        estimator_batch_size=estimator_batch_size,
    )
    assert out.size() == (R_query, 1)

    expected = torch.stack(
        [
            model(x_context[i], y_context[i], x_query[i]).numerical
            for i in range(3)
        ]
    ).mean(0)
    torch.testing.assert_close(out.numerical, expected, atol=1e-5, rtol=1e-4)

    model.fit(
        x=x_context,
        y=y_context,
        num_estimators=None,
        estimator_batch_size=estimator_batch_size,
    )
    assert model.predict(x_query).allclose(out, atol=1e-5, rtol=1e-4)
    model.clear()


def test_missing_values_are_dropped(model: RelationalTransformer) -> None:
    x_context = torch.randn(5, 3)
    y_context = torch.randn(5, 1)
    x_query = torch.randn(2, 3)
    out = model(x_context, y_context, x_query)

    with_nan = x_context.clone()
    with_nan[1, 2] = torch.nan
    with_nan[4, 0] = torch.nan
    assert not model(with_nan, y_context, x_query).numerical.isnan().any()
    assert not model(with_nan, y_context, x_query).allclose(out)


def test_cell_order_does_not_change_predictions(
    model: RelationalTransformer,
) -> None:
    x_context = torch.randn(6, 4)
    y_context = torch.randn(6, 1)
    x_query = torch.randn(2, 4)
    out = model(x_context, y_context, x_query)
    # Reordering columns changes the cell order but not the prediction.
    perm = [2, 0, 3, 1]
    columns = [str(i) for i in perm]
    x_context_perm = TableTensor.from_tensor(x_context[:, perm], columns)
    x_query_perm = TableTensor.from_tensor(x_query[:, perm], columns)
    torch.testing.assert_close(
        model(x_context_perm, y_context, x_query_perm).numerical,
        out.numerical,
        atol=1e-5,
        rtol=1e-4,
    )


def _rt_batch(
    B: int, S: int, D: int, device: torch.device
) -> dict[str, Tensor]:
    generator = torch.Generator().manual_seed(0)
    node = torch.arange(S).div(3, rounding_mode="floor").expand(B, S).clone()
    parents = torch.full((B, S, 5), -1)
    parents[..., 0] = (node + 1) % (S // 3)
    kind = torch.randint(0, 3, (B, S), generator=generator)
    is_target = torch.zeros(B, S, dtype=torch.bool)
    is_target[:, 0] = True
    kind[:, 0] = NUMBER
    is_padding = torch.zeros(B, S, dtype=torch.bool)
    is_padding[:, -2:] = True
    column_values = torch.randn(3, D, generator=generator)
    batch = {
        "node_idxs": node.to(torch.int32),
        "f2p_nbr_idxs": parents.to(torch.int32),
        "col_name_idxs": (torch.arange(S) % 3).expand(B, S).to(torch.int32),
        "col_name_values": column_values[torch.arange(S) % 3].expand(B, S, D),
        "sem_types": kind.to(torch.int32),
        "number_values": torch.randn(B, S, 1, generator=generator),
        "text_values": torch.randn(B, S, D, generator=generator),
        "datetime_values": torch.randn(B, S, 1, generator=generator),
        "is_targets": is_target,
        "is_padding": is_padding,
    }
    return {key: value.to(device) for key, value in batch.items()}


@withCUDA
def test_from_rt_batch(device: torch.device) -> None:
    B, S, D = 2, 11, TEXT_CHANNELS
    batch = _rt_batch(B, S, D, device)
    cells = Cells.from_rt_batch(batch)

    assert cells.node.size() == (B, S)
    assert (cells.node[:, -2:] == -1).all()
    assert cells.parents.size() == (B, S, 5)
    assert cells.target.tolist() == [[0], [0]]
    assert cells.is_target.sum() == B
    text = cells.kind == TEXT
    assert (cells.text[text] >= 0).all()
    assert (cells.text[~text] == -1).all()
    torch.testing.assert_close(
        cells.embedding[cells.column],
        batch["col_name_values"],
    )
    torch.testing.assert_close(
        cells.embedding[cells.text[text]],
        batch["text_values"][text],
    )
    datetime = cells.kind == DATETIME
    torch.testing.assert_close(
        cells.value[datetime],
        batch["datetime_values"].squeeze(-1)[datetime],
    )
    torch.testing.assert_close(
        cells.value[~datetime],
        batch["number_values"].squeeze(-1)[~datetime],
    )


@withCUDA
def test_predict_cells(
    model: RelationalTransformer,
    device: torch.device,
) -> None:
    cells = Cells.from_rt_batch(_rt_batch(2, 11, TEXT_CHANNELS, device))
    out = model.predict_cells(cells, block_sparse=False)
    assert out.size() == (2, 1)
    assert out.isfinite().all()
    assert torch.is_inference(out)


@onlyCUDA
@pytest.mark.usefixtures("compilation")
def test_block_sparse_attention_matches_dense() -> None:
    device = torch.device("cuda")
    core = rt_module._RelationalTransformer(
        num_layers=2,
        channels=CHANNELS,
        text_channels=TEXT_CHANNELS,
        num_heads=4,
        hidden_channels=64,
        device=device,
    )
    for parameter in core.parameters():
        if not parameter.any():
            torch.nn.init.normal_(parameter, std=0.02)
    cells = Cells.from_rt_batch(_rt_batch(2, 300, TEXT_CHANNELS, device))

    with torch.inference_mode():
        dense = core(cells, block_sparse=False)
        sparse = core(cells, block_sparse=True)
    torch.testing.assert_close(sparse, dense, atol=1e-4, rtol=1e-3)
