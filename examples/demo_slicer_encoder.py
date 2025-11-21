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
from imprint import SequenceDataset

Auto = imprint.Auto


CONFIG = {
    "seed": 151,
    "epochs": 10,
    "lr": 1e-3,
    "log_every": 5,
    "val_every": 1,
    "grad_clip": 1.0,
    "num_classes": None,
    "num_chunks": 16,
    "chunk_overlap": 0,
    "slicer_hidden": 256,
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
        ports=imprint.Ports(
            in_=imprint.InPort(size=Auto, combine="concat"),
            out_default=CONFIG["slicer_hidden"],
        ),
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
    graph.connect(src["out"], slicer["in"])      # input tokens to slicer               
    graph.connect(slicer["out"], context["in"])
    graph.connect(context["out"], head["in"])
    #graph.connect(context["out"], slicer["in"])      # feedback tokens to slicer

    return graph


def run() -> None:
    cfg = CONFIG
    data_cfg = dict(cfg["data"])

    dataset, val_dataset = imprint.load_train_val_splits(
        data_cfg,
        train_split=str(cfg["train_split"]),  # type: ignore[index]
        val_split=cfg.get("val_split", "val"),
    )

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
        register_objective=False,  # avoid double-counting CE across all timesteps
    )

    train_kwargs = imprint.trainer_kwargs_from_config(cfg, val_dataset=val_dataset)
    train_kwargs["loss_fn"] = imprint.combined_graph_and_ce_loss(
        head_name="head",
        label_key="y",
    )
    train_kwargs["metric_fn"] = imprint.last_step_accuracy(head_name="head", label_key="y")

    imprint.train_graph(
        graph,
        dataset,
        **train_kwargs,  # type: ignore[arg-type]
    )

    print("Done.")


if __name__ == "__main__":
    run()

