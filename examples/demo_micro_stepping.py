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
}


def build_graph(hidden_size: int, target_dim: int) -> imprint.Graph:
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
        schedule=imprint.Rate(inner_steps=1, emit_every=1),
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
    if cfg["seed"] is not None:
        torch.manual_seed(cfg["seed"])

    dataset = load_micro_step_demo_dataset(
        path=cfg["data_path"],
        split=cfg["split"],
        batch_size=cfg["batch_size"],
    )

    print("Loaded dataset:", dataset.summary())
    print(
        f"Metadata → feature_dim={dataset.feature_dim}, "
        f"target_dim={dataset.target_dim}, batches/epoch={dataset.batches_per_epoch}"
    )

    # Determine number of classes for seq2static classification.
    inferred_classes = dataset.num_classes
    if cfg["num_classes"] is not None:
        output_dim = int(cfg["num_classes"])
    elif inferred_classes is not None:
        output_dim = int(inferred_classes)
    elif getattr(dataset, "labels", None) is not None:
        # Derive class count from labels even if dtype is float in the HDF5.
        labels = dataset.labels  # type: ignore[attr-defined]
        max_label = int(labels.max().item())
        output_dim = max_label + 1
    else:
        output_dim = dataset.target_dim

    graph = build_graph(hidden_size=cfg["hidden_size"], target_dim=output_dim)

    # Emit once per sequence for seq2static classification.
    graph.modules["head"].schedule.emit_every = dataset.seq_len

    # Seq2static classification: compute CE on the last timestep only.
    def loss_fn(graph: imprint.Graph, batch: dict) -> torch.Tensor:
        head = graph.modules["head"]
        logits = head.state.output["out"]  # [B, T, C]
        last_logits = logits[:, -1, :]
        y = batch["y"]
        if y.dim() > 1:
            y = y.squeeze(-1)
        return torch.nn.functional.cross_entropy(last_logits, y.long())

    imprint.train_graph(
        graph,
        dataset,
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        log_every=cfg["log_every"],
        seed=None,  # already seeded
        loss_fn=loss_fn,
        grad_clip=1.0,
        use_adamw=True,
    )

    print("Done.")


if __name__ == "__main__":
    run()
