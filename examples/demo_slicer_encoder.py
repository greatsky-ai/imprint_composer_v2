"""
Demo script: slicer-style encoder followed by a context GRU head.

The graph mirrors the pattern in reference_material/slicer.py by:
  - Running a GRU slicer at the base tick rate but only emitting chunk summaries.
  - Feeding those chunk-level embeddings to a second GRU (context stage) that
    advances every time the slicer emits.
  - Applying an MLP head for sequence-to-static classification.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import imprint
from imprint import SequenceDataset, load_demo_dataset

Auto = imprint.Auto


CONFIG = {
    "seed": 111,
    "epochs": 10,
    "lr": 2e-3,
    "log_every": 5,
    "val_every": 1,
    "grad_clip": 1.0,
    "num_classes": None,
    "num_chunks": 32,
    "chunk_overlap": 0,
    "slicer_hidden": 128,
    "slicer_layers": 1,
    "slicer_inner_steps": 1,
    "context_hidden": 128,
    "context_layers": 1,
    "head_widths": [Auto],
    "train_split": "train",
    "val_split": "val",
    "data": {
        "path": "solids_32x32.h5",
        "batch_size": 128,
        "synth_total": 320,
        "synth_seq_len": 160,
        "synth_feature_dim": 64,
        "synth_seed": 11,
    },
}


def _chunk_hparams(dataset: SequenceDataset) -> tuple[int, int]:
    num_chunks = max(1, int(CONFIG["num_chunks"]))
    chunk_len = max(1, math.ceil(dataset.seq_len / num_chunks))
    overlap = float(CONFIG.get("chunk_overlap", 0.0))
    overlap = min(max(overlap, 0.0), 0.95)
    stride = max(1, int(round(chunk_len * (1.0 - overlap))))
    return chunk_len, stride


def build_graph(
    dataset: SequenceDataset,
    *,
    chunk_params: tuple[int, int] | None = None,
) -> imprint.Graph:
    clock = imprint.Clock()
    graph = imprint.Graph(clock=clock)

    chunk_len, stride = chunk_params or _chunk_hparams(dataset)

    src = imprint.Source("x")
    output_dim = imprint.infer_num_classes(dataset, override=CONFIG["num_classes"])

    reset_steps = max(1, chunk_len * CONFIG["slicer_inner_steps"])
    slicer = imprint.Module(
        name="slicer",
        proto=imprint.protos.GRUStack(
            hidden=CONFIG["slicer_hidden"],
            layers=CONFIG["slicer_layers"],
            layernorm=True,
            reset_every=reset_steps,
        ),
        ports=imprint.Ports(in_default=Auto, out_default=CONFIG["slicer_hidden"]),
        schedule=imprint.Rate(inner_steps=CONFIG["slicer_inner_steps"], emit_every=stride),
    )
    context = imprint.Module(
        name="context",
        proto=imprint.protos.GRUStack(
            hidden=CONFIG["context_hidden"],
            layers=CONFIG["context_layers"],
            layernorm=True,
        ),
        ports=imprint.Ports(
            in_default=CONFIG["slicer_hidden"],
            out_default=CONFIG["context_hidden"],
        ),
        schedule=imprint.Rate.follow(slicer, step_when_emits=True),
    )
    head = imprint.Module(
        name="head",
        proto=imprint.protos.MLP(widths=CONFIG["head_widths"]),
        ports=imprint.Ports(
            in_default=CONFIG["context_hidden"],
            out_default=output_dim,
        ),
        schedule=imprint.Rate.follow(context, step_when_emits=True),
    )

    graph.add(src, slicer, context, head)
    graph.connect(src["out"], slicer["in"])
    graph.connect(slicer["out"], context["in"])
    graph.connect(context["out"], head["in"])

    return graph


def run() -> None:
    cfg = CONFIG
    data_cfg = dict(cfg["data"])

    dataset = load_demo_dataset(split=cfg["train_split"], **data_cfg)
    val_dataset = load_demo_dataset(split=cfg["val_split"], **data_cfg)

    chunk_len, chunk_stride = _chunk_hparams(dataset)
    approx_chunks = math.ceil(dataset.seq_len / chunk_stride)
    graph = build_graph(dataset, chunk_params=(chunk_len, chunk_stride))

    print(
        f"Slicer emits every {chunk_stride} ticks "
        f"(chunk_len={chunk_len}, approx_chunks={approx_chunks})."
    )

    imprint.prepare_seq2static_classification(
        graph,
        dataset,
        head_name="head",
        label_key="y",
        emit_once=False,
    )

    imprint.train_graph(
        graph,
        dataset,
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        log_every=cfg["log_every"],
        seed=cfg["seed"],
        grad_clip=cfg["grad_clip"],
        loss_fn=imprint.last_step_ce_loss(head_name="head", label_key="y"),
        metric_fn=imprint.last_step_accuracy(head_name="head", label_key="y"),
        val_dataset=val_dataset,
        val_every=cfg["val_every"],
    )

    print("Done.")


if __name__ == "__main__":
    run()

