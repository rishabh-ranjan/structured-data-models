# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from typing import Any, cast

import pandas as pd
import relbench
import torch
import torchmetrics
from lag_target import add_lag_target
from tqdm import tqdm

import sdm

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--context_size", type=int, default=32)
parser.add_argument("--context_mode", choices=("last", "rnd"), default="last")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--max_test_steps", type=int, default=None)
parser.add_argument("--num_neighbors", type=int, nargs="*", default=[8, 8])
parser.add_argument("--num_lags", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Collect Relational Data #####################################################
dataset = relbench.load_dataset(args.dataset)
task = dataset.load_task(args.task)
db = task.get_db(upto_test_timestamp=False)
data = sdm.RelationalData(
    tables={
        name: sdm.TableTensor.from_pandas(
            df=table.df,
            stypes=sdm.infer_stypes(
                table.df.head(10_000),
                overrides={
                    cast(str, table.pkey_col): "id",
                    **dict.fromkeys(table.fkey_col_to_pkey_table, "id"),
                },
                text="infer",
                unsupported="drop",
            ),
        )
        for name, table in db.table_dict.items()
    },
    relationships=[
        {
            "left_table": left_table,
            "left_column": left_column,
            "right_table": right_table,
            "right_column": cast(str, db.table_dict[right_table].pkey_col),
        }
        for left_table, table in db.table_dict.items()
        for left_column, right_table in table.fkey_col_to_pkey_table.items()
    ],
)

# Collect Task Table ##########################################################
if task.task_type not in {
    relbench.base.TaskType.BINARY_CLASSIFICATION,
    relbench.base.TaskType.REGRESSION,
}:
    quit(f"Unsupported task {task.task_type.value!r}")

dfs = [
    task.get_table(split, mask_input_cols=False).df
    for split in ["train", "val", "test"]
]
df = pd.concat(dfs, ignore_index=True)
df = add_lag_target(df, task, args.num_lags)

if task.task_type == relbench.base.TaskType.REGRESSION:
    target_stype = "numerical"
else:
    target_stype = "categorical"

task_table = sdm.TableTensor.from_pandas(
    df=df,
    stypes={
        task.entity_col: "id",
        task.time_col: "datetime",
        task.target_col: target_stype,
        **{
            f"{task.target_col}_lag_{lag}": target_stype
            for lag in range(1, args.num_lags + 1)
        },
    },
)

context, query = task_table.split([len(dfs[0]) + len(dfs[1]), len(dfs[2])])
if len(context) > args.context_size:
    if args.context_mode == "last":
        order = context.datetime.reshape(-1).argsort(descending=True)
    else:
        order = torch.randperm(len(context))
    context = context[order[: args.context_size]]

# Create Subgraph Sampler #####################################################
sampler = data.sampler(
    time_columns={
        name: table.time_col
        for name, table in db.table_dict.items()
        if table.time_col is not None
    },
)
kwargs: dict[str, Any] = {
    "task_link": {
        "task_column": task.entity_col,
        "table": task.entity_table,
        "table_column": cast(str, db.table_dict[task.entity_table].pkey_col),
    },
    "num_neighbors": args.num_neighbors,
    "task_time_column": task.time_col,
}

# Execute Model ###############################################################
model = sdm.models.RelationalTransformer(device=device)

context, related_tables = sampler(context, **kwargs).to(device)
with torch.amp.autocast(device.type, torch.bfloat16, enabled=True):
    model.fit(
        x=context.drop_columns(task.target_col),
        y=context[task.target_col],
        related_tables=related_tables,
        table_name=args.task,
    )

if task.task_type == relbench.base.TaskType.REGRESSION:
    metric = torchmetrics.regression.MeanAbsoluteError()
else:
    metric = torchmetrics.classification.BinaryAUROC()
    positive_class = dfs[0][task.target_col].unique()[-1].item()
metric = metric.to(device)
for batch in tqdm(query.split(args.batch_size, -2)[: args.max_test_steps]):
    x_query = batch.drop_columns(task.target_col)
    y_query = batch[task.target_col].to(device)
    with torch.amp.autocast(device.type, torch.bfloat16, enabled=True):
        out = model.predict(*sampler(x_query, **kwargs).to(device))

    if task.task_type == relbench.base.TaskType.REGRESSION:
        pred = out["pred"].numerical.squeeze(-1)
        target = y_query.numerical.squeeze(-1)
    else:
        pred, target = sdm.evaluation.to_binary_class(
            out, y_query, positive_class
        )
    metric.update(pred, target)

if task.task_type == relbench.base.TaskType.REGRESSION:
    print(f"nMAE: {100 * metric.compute() / task.nmae_std:.2f}")
else:
    print(f"AUROC: {100 * metric.compute():.2f}")
