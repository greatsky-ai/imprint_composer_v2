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
from imprint import SequenceDataset, load_micro_step_demo_dataset

Auto = imprint.Auto


CONFIG = {
    "seed": 7,
    "data_path": "ball_drop.h5",  # Optional HDF5 dataset path
    "split": "train",
    "batch_size": 128,
    "hidden_size": 128,
    "epochs": 12,
    "lr": 0.5e-3,
    "log_every": 5,
    # Seq2static classification overrides; set to an int to force class count.
    "num_classes": None,
    "micro_steps": 8,
    "grad_clip": 1.0,
    "use_adamw": True,
    "weight_decay": 1e-2,
    "betas": (0.9, 0.95),
}


def build_graph(hidden_size: int, target_dim) -> imprint.Graph:
    """
    Build the micro-stepping graph described in §4 of USAGE.md.
    """
    clock = imprint.Clock()
    graph = imprint.Graph(clock=clock)

    src = imprint.Source("x")
    enc_slow = imprint.Module(
        name="enc_slow",
        proto=imprint.protos.GRUStack(hidden=hidden_size, layers=1, layernorm=True),
        ports=imprint.Ports(in_default=Auto, out_default=hidden_size),
        schedule=imprint.Rate(inner_steps=CONFIG["micro_steps"], emit_every=1),
    )
    head = imprint.Module(
        name="head",
        proto=imprint.protos.MLP(widths=[Auto]),
        ports=imprint.Ports(in_default=hidden_size, out_default=target_dim),
        schedule=imprint.Rate(inner_steps=1, emit_every=1),
    )

    graph.add(src, enc_slow, head)
    graph.connect(src["out"], enc_slow["in"])
    graph.connect(enc_slow["out"], head["in"])

    return graph


def run() -> None:
    cfg = CONFIG

    dataset = load_micro_step_demo_dataset(
        path=cfg["data_path"],
        split=cfg["split"],
        batch_size=cfg["batch_size"],
    )

    # Build with Auto head dim; dims will be inferred from the CE objective.
    graph = build_graph(hidden_size=cfg["hidden_size"], target_dim=Auto)

    # Prepare seq2static objective (emits once per sequence, CE on labels).
    imprint.prepare_seq2static_classification(
        graph,
        dataset,
        head_name="head",
        label_key="y",
        emit_once=True,
    )

    imprint.train_graph(
        graph,
        dataset,
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        log_every=cfg["log_every"],
        seed=cfg["seed"],
        grad_clip=cfg["grad_clip"],
        use_adamw=cfg["use_adamw"],
        weight_decay=cfg["weight_decay"],
        betas=cfg["betas"],
    )

    print("Done.")


if __name__ == "__main__":
    run()
