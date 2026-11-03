# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RelationalTransformer on RelBench with the Relational Transformer sampler.

Reproduces the released RT-J evaluation protocol: one context of up to 8,192
cells per query row, sampled by the `relational-transformer` package from the
preprocessed RelBench databases, and scored with the RelBench metrics.
"""

import argparse
from collections.abc import Mapping, Sequence

import torch
from huggingface_hub import snapshot_download
from rt.data import get_tasks
from rt.eval import build_evaluator, run_and_report
from torch import Tensor

import sdm
from sdm.models.relational_transformer import Cells

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--task", type=str, required=True)
parser.add_argument(
    "--checkpoint", choices=("rt-j", "rt-plurel"), default="rt-j"
)
parser.add_argument("--ctx_size", type=int, default=8192)
parser.add_argument("--local_ctx_size", type=int, default=256)
parser.add_argument("--bfs_width", type=int, default=32)
parser.add_argument("--tokens_per_gpu", type=int, default=2**16)
parser.add_argument("--max_items", type=int, default=10_000_000)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--context_seed", type=int, default=0)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessed database and task #############################################
pre_dir = snapshot_download(
    "stanford-star/relbench-preprocessed",
    repo_type="dataset",
    allow_patterns=f"{args.dataset}/*",
    revision="1016626ddb30c027b92458bf866903850cc205e1",
)
tasks = get_tasks(pre_dir, [(args.dataset, args.task)], ("test",))
evaluator = build_evaluator(
    tasks,
    pre_dir,
    embedder="all-MiniLM-L12-v2",
    d_text=384,
    device=str(device),
    ctx_size_list=[args.ctx_size],
    local_ctx_size=args.local_ctx_size,
    bfs_width=args.bfs_width,
    prefer_latest=True,
    num_walks=10_000,
    walk_length=20,
    tokens_per_gpu=args.tokens_per_gpu,
    items_per_task=args.max_items,
    num_workers=args.num_workers,
    prefetch_factor=2,
    shuffle_seed=0,
    context_seed=args.context_seed,
    mmap_populate=True,
    vector_db_path=None,
    db_cutoff=None,
)


# Model #######################################################################
class Net(torch.nn.Module):
    """Adapt the model to the predictor interface of the RT evaluator."""

    def __init__(self, model: sdm.models.RelationalTransformer) -> None:
        super().__init__()
        self.model = model

    def predict(
        self,
        batch: Mapping[str, Tensor],
        ctx_size_list: Sequence[int],
        device: torch.device | str,
        task: object,
    ) -> dict[int, Tensor]:
        preds = {}
        for ctx_size in ctx_size_list:
            cells = Cells.from_rt_batch(
                {
                    key: value[:, :ctx_size].to(device, non_blocking=True)
                    for key, value in batch.items()
                }
            )
            preds[ctx_size] = self.model.predict_cells(cells)[:, 0].cpu()
        return preds


model = sdm.models.RelationalTransformer(
    checkpoint=args.checkpoint,
    device=device,
).to(torch.bfloat16)

run_and_report(
    Net(model),
    tasks,
    pre_dir,
    ctx_size=args.ctx_size,
    csv_out_dir=None,
    evaluator=evaluator,
    embedder="all-MiniLM-L12-v2",
)
