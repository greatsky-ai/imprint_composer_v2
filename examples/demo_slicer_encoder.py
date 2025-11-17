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
from imprint.recipes import DemoConfig

Auto = imprint.Auto


MODEL = {
    "num_chunks": 16,
    "chunk_overlap": 0,
    "slicer_hidden": 128,
    "slicer_layers": 1,
    "slicer_inner_steps": 1,
    "context_hidden": 128,
    "context_layers": 1,
    "head_widths": [Auto],
}

TRAINING = DemoConfig(
    seed=1561,
    epochs=10,
    lr=1e-3,
    log_every=5,
    val_every=1,
    grad_clip=1.0,
    num_classes=None,
    train_split="train",
    val_split="val",
    data={
        "path": "ball_drop.h5",
        "batch_size": 128,
        "synth_total": 320,
        "synth_seq_len": 160,
        "synth_feature_dim": 64,
        "synth_seed": 11,
    },
)


def _chunk_hparams(dataset: SequenceDataset) -> tuple[int, int]:
    num_chunks = max(1, int(MODEL["num_chunks"]))
    chunk_len = max(1, math.ceil(dataset.seq_len / num_chunks))
    overlap = float(MODEL.get("chunk_overlap", 0.0))
    overlap = min(max(overlap, 0.0), 0.95)
    stride = max(1, int(round(chunk_len * (1.0 - overlap))))
    return chunk_len, stride


def build_graph(
    dataset: SequenceDataset,
    *,
    chunk_params: tuple[int, int] | None = None,
    demo_cfg: DemoConfig = TRAINING,
) -> imprint.Graph:
    clock = imprint.Clock()
    graph = imprint.Graph(clock=clock)

    chunk_len, stride = chunk_params or _chunk_hparams(dataset)

    src = imprint.Source("x")
    output_dim = demo_cfg.infer_num_classes(dataset)

    reset_steps = max(1, chunk_len * MODEL["slicer_inner_steps"])
    slicer = imprint.Module(
        name="slicer",
        proto=imprint.protos.GRUStack(
            hidden=MODEL["slicer_hidden"],
            layers=MODEL["slicer_layers"],
            layernorm=True,
            reset_every=reset_steps,
        ),
        ports=imprint.Ports(out_default=MODEL["slicer_hidden"]),
        schedule=imprint.Rate.slicer(stride, inner_steps=MODEL["slicer_inner_steps"]),
    )
    slicer.enable_concat_input()
    context = imprint.Module(
        name="context",
        proto=imprint.protos.GRUStack(
            hidden=MODEL["context_hidden"],
            layers=MODEL["context_layers"],
            layernorm=True,
        ),
        ports=imprint.Ports(
            in_default=MODEL["slicer_hidden"],
            out_default=MODEL["context_hidden"],
        ),
    )
    head = imprint.Module(
        name="head",
        proto=imprint.protos.MLP(widths=MODEL["head_widths"]),
        ports=imprint.Ports(
            in_default=MODEL["context_hidden"],
            out_default=output_dim,
        ),
    )

    graph.add(src, slicer, context, head)
    graph.follow_when_emits(slicer, context)
    graph.follow_when_emits(context, head)
    graph.connect(src["out"], slicer["in"])      # input tokens to slicer               
    graph.connect(slicer["out"], context["in"])
    graph.connect(context["out"], head["in"])
    #graph.connect(context["out"], slicer["in"])      # feedback tokens to slicer

    return graph


def run() -> None:
    train_dataset, val_dataset = TRAINING.load_datasets()

    chunk_len, chunk_stride = _chunk_hparams(train_dataset)
    approx_chunks = math.ceil(train_dataset.seq_len / chunk_stride)
    graph = build_graph(train_dataset, chunk_params=(chunk_len, chunk_stride))

    print(
        f"Slicer emits every {chunk_stride} ticks "
        f"(chunk_len={chunk_len}, approx_chunks={approx_chunks})."
    )

    imprint.prepare_seq2static_classification(
        graph,
        train_dataset,
        head_name="head",
        label_key="y",
        emit_once=False,
    )

    TRAINING.train(
        graph,
        train_dataset,
        loss_fn=imprint.last_step_ce_loss(head_name="head", label_key="y"),
        metric_fn=imprint.last_step_accuracy(head_name="head", label_key="y"),
        val_dataset=val_dataset,
    )

    print("Done.")


if __name__ == "__main__":
    run()

