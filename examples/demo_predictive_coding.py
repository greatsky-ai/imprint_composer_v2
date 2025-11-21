"""
Demo script: two-layer predictive-coding style stack built from Imprint modules.

Each layer maintains its own GRU state, decodes a prediction of its input, and
pushes reconstruction errors upward while receiving context from the layer
above. Only lightweight glue is added here—the graph reuses core Imprint
modules (GRUs, MLPs, elementwise ops, objectives) to express the PC dynamics.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import imprint
from imprint import SequenceDataset, load_demo_dataset
from imprint.objectives import Targets

Auto = imprint.Auto


CONFIG: Dict[str, object] = {
    "seed": 5,
    "epochs": 1,
    "lr": 1e-3,
    "log_every": 5,
    "val_every": 1,
    "grad_clip": 1.0,
    "train_split": "train",
    "val_split": "val",
    "loss": {
        "rec": 1.0,
        "pred": 0.1,
        "sparse_err": 1e-3,
    },
    "layers": [
        {
            "name": "pc0",
            "hidden": 192,
            "layers": 1,
            "decoder_widths": [192, Auto],
        },
        {
            "name": "pc1",
            "hidden": 128,
            "layers": 1,
            "decoder_widths": [128, Auto],
        },
    ],
    "predictor_widths": [256, Auto],
    "data": {
        "path": "solids_32x32.h5",
        "batch_size": 96,
        "synth_total": 320,
        "synth_seq_len": 160,
        "synth_feature_dim": 64,
        "synth_seed": 13,
    },
}


def _add_pc_layer(
    graph: imprint.Graph,
    spec: Dict[str, object],
    *,
    signal_src: imprint.PortRef,
) -> Dict[str, imprint.Module]:
    """
    Assemble a single predictive-coding layer consisting of:
      - decoder: MLP that predicts the incoming signal
      - err: elementwise subtraction (signal - prediction)
      - gru: recurrent update driven by the error (and future top-down ctx)
    """
    base = spec["name"]
    hidden = int(spec["hidden"])
    layers = int(spec.get("layers", 1))
    decoder_widths = spec["decoder_widths"]  # type: ignore[index]

    decoder = imprint.Module(
        name=f"{base}_decoder",
        proto=imprint.protos.MLP(widths=decoder_widths),
        ports=imprint.Ports(
            in_=imprint.InPort(size=Auto, combine="concat"),
            out_default=Auto,
        ),
    )
    err = imprint.Module(
        name=f"{base}_err",
        proto=imprint.protos.Elementwise("sub"),
        ports=imprint.Ports(
            in_=imprint.InPortGroup(
                layout="disjoint",
                x=imprint.InPort(size=Auto),
                pred=imprint.InPort(size=Auto),
            ),
            out_default=Auto,
        ),
    )
    gru = imprint.Module(
        name=f"{base}_gru",
        proto=imprint.protos.GRUStack(hidden=hidden, layers=layers, layernorm=True),
        ports=imprint.Ports(
            in_=imprint.InPort(size=Auto, combine="concat"),
            out_default=hidden,
        ),
    )

    graph.add(decoder, err, gru)

    graph.connect(gru["out"], decoder["in"])
    graph.connect(signal_src, err["in.x"])
    graph.connect(decoder["out"], err["in.pred"])
    graph.connect(err["out"], gru["in"])

    return {"decoder": decoder, "err": err, "gru": gru}


def _connect_top_down(
    graph: imprint.Graph,
    higher: Dict[str, imprint.Module],
    lower: Dict[str, imprint.Module],
) -> None:
    ctx = higher["gru"]["out"]
    graph.connect(ctx, lower["gru"]["in"])
    graph.connect(ctx, lower["decoder"]["in"])


def _add_predictor_head(
    graph: imprint.Graph,
    base_gru: imprint.Module,
) -> imprint.Module:
    predictor = imprint.Module(
        name="pc_pred_head",
        proto=imprint.protos.MLP(widths=CONFIG["predictor_widths"]),  # type: ignore[index]
        ports=imprint.Ports(
            in_default=Auto,
            out_default=Auto,
        ),
    )
    graph.add(predictor)
    graph.connect(base_gru["out"], predictor["in"])
    return predictor


def _register_objectives(
    layers: List[Dict[str, imprint.Module]],
    *,
    predictor: imprint.Module | None,
    source: imprint.Module,
) -> None:
    loss_cfg = CONFIG["loss"]  # type: ignore[index]
    rec_w = float(loss_cfg.get("rec", 1.0))
    sparse_w = float(loss_cfg.get("sparse_err", 0.0))
    pred_w = float(loss_cfg.get("pred", 0.0))

    for layer in layers:
        decoder = layer["decoder"]
        err = layer["err"]
        target = Targets.port_drive(err, "in.x")
        decoder.objectives.mse("out", target, weight=rec_w)
        if sparse_w > 0:
            err.objectives.activity_l1("out", weight=sparse_w)

    if predictor is not None and pred_w > 0:
        predictor.objectives.mse(
            "out",
            Targets.shifted_input(source, +1),
            weight=pred_w,
        )


def build_graph(dataset: SequenceDataset) -> imprint.Graph:
    clock = imprint.Clock()
    graph = imprint.Graph(clock=clock)

    src = imprint.Source("x")
    graph.add(src)

    signal = src["out"]
    built_layers: List[Dict[str, imprint.Module]] = []

    for idx, layer_spec in enumerate(CONFIG["layers"]):  # type: ignore[index]
        layer = _add_pc_layer(graph, layer_spec, signal_src=signal)  # type: ignore[arg-type]
        built_layers.append(layer)
        signal = layer["err"]["out"]
        if idx > 0:
            _connect_top_down(graph, built_layers[idx], built_layers[idx - 1])

    predictor = None
    if float(CONFIG["loss"]["pred"]) > 0:  # type: ignore[index]
        predictor = _add_predictor_head(graph, built_layers[0]["gru"])

    _register_objectives(built_layers, predictor=predictor, source=src)
    return graph


def run() -> None:
    cfg = CONFIG
    data_cfg = dict(cfg["data"])  # type: ignore[index]

    dataset = load_demo_dataset(split=cfg["train_split"], **data_cfg)  # type: ignore[index]
    val_dataset = load_demo_dataset(split=cfg["val_split"], **data_cfg)  # type: ignore[index]

    graph = build_graph(dataset)

    imprint.train_graph(
        graph,
        dataset,
        epochs=cfg["epochs"],  # type: ignore[index]
        lr=cfg["lr"],  # type: ignore[index]
        log_every=cfg["log_every"],  # type: ignore[index]
        seed=cfg["seed"],  # type: ignore[index]
        grad_clip=cfg["grad_clip"],  # type: ignore[index]
        val_dataset=val_dataset,
        val_every=cfg["val_every"],  # type: ignore[index]
    )

    print("Done.")


if __name__ == "__main__":
    run()
