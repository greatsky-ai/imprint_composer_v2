"""
Demo script #4 from USAGE.md: micro-stepping encoder + next-step head.

Data handling is intentionally minimal – we call the helper from DATASETS.md
inspired code to fetch (or synthesize) a `[N, T, D]` sequence dataset and let
its metadata drive module dimensions.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import imprint
from imprint import SequenceDataset
from imprint.recipes import DemoConfig

Auto = imprint.Auto


MODEL = {
    "hidden_size": 128,
    "micro_steps": 1,
}

TRAINING = DemoConfig(
    seed=7,
    epochs=12,
    lr=1e-3,
    log_every=5,
    num_classes=None,
    grad_clip=1.0,
    val_every=1,
    train_split="train",
    val_split="val",
    data={
        "path": "solids_32x32.h5",  # Optional HDF5 dataset path
        "batch_size": 128,
        "synth_total": 320,
        "synth_seq_len": 160,
        "synth_feature_dim": 64,
        "synth_seed": 7,
    },
)


def build_graph(dataset: SequenceDataset) -> imprint.Graph:
    """
    Build the micro-stepping graph described in §4 of USAGE.md.
    """
    clock = imprint.Clock()
    graph = imprint.Graph(clock=clock)

    src = imprint.Source("x")
    hidden_size = MODEL["hidden_size"]
    output_dim = TRAINING.infer_num_classes(dataset)
    enc_slow = imprint.Module(
        name="enc_slow",
        proto=imprint.protos.GRUStack(hidden=hidden_size, layers=1, layernorm=True),
        ports=imprint.Ports(in_default=Auto, out_default=hidden_size),
        schedule=imprint.Rate.micro(MODEL["micro_steps"]),
    )
    head = imprint.Module(
        name="head",
        proto=imprint.protos.MLP(widths=[Auto]),
        ports=imprint.Ports(in_default=hidden_size, out_default=output_dim),
        schedule=imprint.Rate(inner_steps=1, emit_every=1),
    )

    graph.add(src, enc_slow, head)
    graph.connect(src["out"], enc_slow["in"])
    graph.connect(enc_slow["out"], head["in"])

    return graph


def run() -> None:
    dataset, val_dataset = TRAINING.load_datasets()

    # Build graph using CONFIG and dataset metadata (fixed output dim).
    graph = build_graph(dataset)

    # Prepare seq2static objective (CE on labels). Keep head emitting each tick.
    imprint.prepare_seq2static_classification(
        graph,
        dataset,
        head_name="head",
        label_key="y",
        emit_once=False,
    )

    TRAINING.train(
        graph,
        dataset,
        loss_fn=imprint.last_step_ce_loss(head_name="head", label_key="y"),
        metric_fn=imprint.last_step_accuracy(head_name="head", label_key="y"),
        val_dataset=val_dataset,
    )

    print("Done.")


if __name__ == "__main__":
    run()
