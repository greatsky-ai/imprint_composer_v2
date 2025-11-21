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
from imprint import SequenceDataset
from imprint.objectives import Targets

Auto = imprint.Auto


CONFIG: Dict[str, object] = {
    "seed": 5,
    "epochs": 5,
    "lr": 0.5e-3,
    "log_every": 5,
    "val_every": 1,
    "grad_clip": 1.0,
    "use_adamw": True,
    "weight_decay": 5e-2,
    "train_split": "train",
    "val_split": "val",
    "two_layers": True,
    "use_feedback": False,
    "loss": {
        "rec": 0,
        "pred": 1,
        "sparse_err": 0.0,
    },
    "layers": [
        {
            "name": "pc0",
            "hidden": 128,
            "layers": 1,
            "decoder_widths": [128, Auto],
            "out_dim": 32,

        },
        {
            "name": "pc1",
            "hidden": 128,
            "layers": 1,
            "decoder_widths": [128, Auto],
            "out_dim": 32,
        },
    ],
    "predictor_widths": [256, Auto],
    # Task head configuration (aux GRU consumes all PC GRU latents)
    "task": {
        "enabled": True,
        "aux_hidden": 64,
        "aux_layers": 1,
        "head_widths": [Auto],
        # If labels are present, we use classification; otherwise we fall back to next-step MSE.
        "num_classes": None,  # override; None -> infer from dataset if available
        "emit_once": False,   # keep head emitting each tick; CE computed on final step
        "label_key": "y",
        "head_name": "task_head",
        "weight": 1.0,        # relative weight for classification loss when combined with graph objectives
        "stop_grad": True,    # detach gradients from PC GRU outputs into aux (prevents upstream updates)
    },
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
    gru_out_dim = spec.get("out_dim", None)
    gru = imprint.Module(
        name=f"{base}_gru",
        proto=imprint.protos.GRUStack(
            hidden=hidden,
            layers=layers,
            layernorm=True,
            out_dim=(None if gru_out_dim is None else int(gru_out_dim)),  # type: ignore[arg-type]
        ),
        ports=imprint.Ports(
            in_=imprint.InPort(size=Auto, combine="concat"),
            out_default=Auto,  # let graph infer from proto.out_dim or hidden
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
    task_head: imprint.Module | None,
    dataset: SequenceDataset,
) -> None:
    loss_cfg = CONFIG["loss"]  # type: ignore[index]
    rec_w = float(loss_cfg.get("rec", 1.0))
    sparse_w = float(loss_cfg.get("sparse_err", 0.0))
    pred_w = float(loss_cfg.get("pred", 0.0))

    for layer in layers:
        decoder = layer["decoder"]
        err = layer["err"]
        target = Targets.port_drive(err, "in.x")
        decoder.objectives.mse("out", target, weight=rec_w, name="rec")
        if sparse_w > 0:
            err.objectives.activity_l1("out", weight=sparse_w, name="sparse_err")

    if predictor is not None and pred_w > 0:
        predictor.objectives.mse(
            "out",
            Targets.shifted_input(source, +1),
            weight=pred_w,
            name="pred",
        )

    # Optional fallback task objective: if no labels/classes, train the task head for next-step prediction.
    if task_head is not None:
        try:
            out_dim = imprint.infer_num_classes(dataset, override=CONFIG.get("task", {}).get("num_classes"))  # type: ignore[index]
            has_labels = out_dim is not None and int(out_dim) > 0
        except Exception:
            has_labels = False
        if not has_labels:
            task_head.objectives.mse(
                "out",
                Targets.shifted_input(source, +1),
                weight=1.0,
                name="task_pred",
            )


def build_graph(dataset: SequenceDataset) -> imprint.Graph:
    clock = imprint.Clock()
    graph = imprint.Graph(clock=clock)

    src = imprint.Source("x")
    graph.add(src)

    signal = src["out"]
    built_layers: List[Dict[str, imprint.Module]] = []

    layer_specs = CONFIG["layers"]  # type: ignore[index]
    if not bool(CONFIG.get("two_layers", True)):
        layer_specs = layer_specs[:1]

    for idx, layer_spec in enumerate(layer_specs):
        layer = _add_pc_layer(graph, layer_spec, signal_src=signal)  # type: ignore[arg-type]
        built_layers.append(layer)
        signal = layer["err"]["out"]
        if idx > 0 and bool(CONFIG.get("use_feedback", True)):
            _connect_top_down(graph, built_layers[idx], built_layers[idx - 1])

    predictor = None
    if float(CONFIG["loss"]["pred"]) > 0:  # type: ignore[index]
        predictor = _add_predictor_head(graph, built_layers[0]["gru"])

    task_cfg = CONFIG.get("task", {})  # type: ignore[assignment]
    task_head = None
    if isinstance(task_cfg, dict):
        feature_refs = [layer["gru"]["out"] for layer in built_layers]
        extra_refs: List[tuple[imprint.PortRef, bool]] = []
        task_head = imprint.attach_task_head(
            graph,
            dataset=dataset,
            config=task_cfg,
            feature_refs=feature_refs,
            extra_refs=extra_refs,
        )
        is_classification, _ = imprint.detect_task_mode(
            dataset, num_classes_override=task_cfg.get("num_classes", None)
        )
        if is_classification and task_head is not None:
            imprint.prepare_seq2static_classification(
                graph,
                dataset,
                head_name=str(task_cfg.get("head_name", "task_head")),
                label_key=str(task_cfg.get("label_key", "y")),
                emit_once=bool(task_cfg.get("emit_once", False)),
                register_objective=False,
            )

    _register_objectives(built_layers, predictor=predictor, source=src, task_head=task_head, dataset=dataset)
    return graph


def run() -> None:
    cfg = CONFIG
    data_cfg = dict(cfg["data"])  # type: ignore[index]

    dataset, val_dataset = imprint.load_train_val_splits(
        data_cfg,
        train_split=str(cfg["train_split"]),  # type: ignore[index]
        val_split=cfg.get("val_split", "val"),  # type: ignore[arg-type]
    )

    graph = build_graph(dataset)

    train_kwargs = imprint.trainer_kwargs_from_config(cfg, val_dataset=val_dataset)
    task_cfg = cfg.get("task", {})  # type: ignore[assignment]
    if isinstance(task_cfg, dict) and task_cfg.get("enabled", False):
        is_classification, _ = imprint.detect_task_mode(
            dataset, num_classes_override=task_cfg.get("num_classes", None)
        )
        if is_classification:
            head_name = str(task_cfg.get("head_name", "task_head"))
            label_key = str(task_cfg.get("label_key", "y"))
            task_w = float(task_cfg.get("weight", 1.0))
            train_kwargs["loss_fn"] = imprint.combined_graph_and_ce_loss(
                head_name=head_name,
                label_key=label_key,
                weight=task_w,
            )
            train_kwargs["metric_fn"] = imprint.last_step_accuracy(
                head_name=head_name,
                label_key=label_key,
            )

    imprint.train_graph(
        graph,
        dataset,
        **train_kwargs,  # type: ignore[arg-type]
    )

    print("Done.")


if __name__ == "__main__":
    run()
