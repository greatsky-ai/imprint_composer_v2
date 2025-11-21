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
    "epochs": 5,
    "lr": 0.5e-3,
    "log_every": 5,
    "val_every": 1,
    "grad_clip": 1.0,
    "train_split": "train",
    "val_split": "val",
    "loss": {
        "rec": 0.3,
        "pred": 1.0,
        "sparse_err": 0,
    },
    "layers": [
        {
            "name": "pc0",
            "hidden": 128,
            "layers": 1,
            "decoder_widths": [128, Auto],
            "out_dim": 16,

        },
        {
            "name": "pc1",
            "hidden": 128,
            "layers": 1,
            "decoder_widths": [128, Auto],
            "out_dim": 16,
        },
    ],
    "predictor_widths": [256, Auto],
    # Task head configuration (aux GRU consumes all PC GRU latents)
    "task": {
        "enabled": True,
        "aux_hidden": 128,
        "aux_layers": 1,
        "head_widths": [Auto],
        # If labels are present, we use classification; otherwise we fall back to next-step MSE.
        "num_classes": None,  # override; None -> infer from dataset if available
        "emit_once": False,   # keep head emitting each tick; CE computed on final step
        "label_key": "y",
        "head_name": "task_head",
        "weight": 1.0,        # relative weight for classification loss when combined with graph objectives
        "stop_grad": True,    # detach gradients from PC GRU outputs into aux (prevents upstream updates)
        "use_top_only": False,# if True, feed only the highest PC GRU into aux (instead of all)
        "include_raw_x": False,  # optionally also feed raw input x into aux for debugging
        "include_errors": False, # optionally feed per-layer reconstruction errors into aux
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


def _add_aux_and_task_head(
    graph: imprint.Graph,
    *,
    gru_latents: List[imprint.Module],
    dataset: SequenceDataset,
    source_module: imprint.Module | None = None,
    pc_layers: List[Dict[str, imprint.Module]] | None = None,
) -> imprint.Module | None:
    """
    Add an auxiliary GRU that consumes all PC GRU latents (concatenated) and feed a task head.
    Returns the task head module or None if disabled.
    """
    task_cfg = CONFIG.get("task", {})  # type: ignore[assignment]
    if not isinstance(task_cfg, dict) or not task_cfg.get("enabled", False):
        return None

    aux_hidden = int(task_cfg.get("aux_hidden", 160))
    aux_layers = int(task_cfg.get("aux_layers", 1))
    head_widths = task_cfg.get("head_widths", [Auto])  # type: ignore[assignment]
    head_name = str(task_cfg.get("head_name", "task_head"))

    # Determine whether dataset has labels (classification). If not, we'll fall back to next-step MSE.
    try:
        out_dim = imprint.infer_num_classes(dataset, override=task_cfg.get("num_classes", None))
        classification_mode = out_dim is not None and int(out_dim) > 0
    except Exception:
        out_dim = None
        classification_mode = False

    aux_out_dim = task_cfg.get("aux_out_dim", None)
    aux = imprint.Module(
        name="pc_aux",
        proto=imprint.protos.GRUStack(
            hidden=aux_hidden,
            layers=aux_layers,
            layernorm=True,
            out_dim=(None if aux_out_dim is None else int(aux_out_dim)),  # type: ignore[arg-type]
        ),
        ports=imprint.Ports(
            in_=imprint.InPort(size=Auto, combine="concat"),
            out_default=Auto,  # infer from proto.out_dim or hidden
        ),
    )
    head = imprint.Module(
        name=head_name,
        proto=imprint.protos.MLP(widths=head_widths),  # type: ignore[arg-type]
        ports=imprint.Ports(
            in_default=Auto,  # infer from aux out
            out_default=(out_dim if classification_mode else Auto),  # Auto when using MSE fallback
        ),
    )
    graph.add(aux, head)
    # Optionally restrict to the highest PC GRU only
    sources = [gru_latents[-1]] if bool(task_cfg.get("use_top_only", False)) and gru_latents else list(gru_latents)
    for gru in sources:
        e = graph.connect(gru["out"], aux["in"])
        # Control whether task loss updates upstream PC GRUs.
        e.set_stop_grad(bool(task_cfg.get("stop_grad", True)))
    # Optional debug fan-in: include raw x and/or per-layer error signals
    if bool(task_cfg.get("include_raw_x", False)) and source_module is not None:
        e = graph.connect(source_module["out"], aux["in"])
        e.set_stop_grad(True)  # ensure task loss cannot update the source data path
    if bool(task_cfg.get("include_errors", False)) and pc_layers is not None:
        for layer in pc_layers:
            e = graph.connect(layer["err"]["out"], aux["in"])
            e.set_stop_grad(True)
    graph.connect(aux["out"], head["in"])

    return head


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

    for idx, layer_spec in enumerate(CONFIG["layers"]):  # type: ignore[index]
        layer = _add_pc_layer(graph, layer_spec, signal_src=signal)  # type: ignore[arg-type]
        built_layers.append(layer)
        signal = layer["err"]["out"]
        if idx > 0:
            _connect_top_down(graph, built_layers[idx], built_layers[idx - 1])

    predictor = None
    if float(CONFIG["loss"]["pred"]) > 0:  # type: ignore[index]
        predictor = _add_predictor_head(graph, built_layers[0]["gru"])

    # Add auxiliary GRU that consumes all GRU latents and a task head that trains via task loss
    task_head = _add_aux_and_task_head(
        graph,
        gru_latents=[layer["gru"] for layer in built_layers],
        dataset=dataset,
        source_module=src,
        pc_layers=built_layers,
    )

    # If labels exist, prepare seq2static classification on the specified head.
    task_cfg = CONFIG.get("task", {})  # type: ignore[assignment]
    if isinstance(task_cfg, dict) and task_cfg.get("enabled", False):
        try:
            out_dim = imprint.infer_num_classes(dataset, override=task_cfg.get("num_classes", None))
            classification_mode = out_dim is not None and int(out_dim) > 0
        except Exception:
            classification_mode = False
        if classification_mode and task_head is not None:
            imprint.prepare_seq2static_classification(
                graph,
                dataset,
                head_name=str(task_cfg.get("head_name", "task_head")),
                label_key=str(task_cfg.get("label_key", "y")),
                emit_once=bool(task_cfg.get("emit_once", False)),
                register_objective=False,  # we'll add CE via combined training loss (final step)
            )

    _register_objectives(built_layers, predictor=predictor, source=src, task_head=task_head, dataset=dataset)
    return graph


def run() -> None:
    cfg = CONFIG
    data_cfg = dict(cfg["data"])  # type: ignore[index]

    dataset = load_demo_dataset(split=cfg["train_split"], **data_cfg)  # type: ignore[index]
    val_dataset = load_demo_dataset(split=cfg["val_split"], **data_cfg)  # type: ignore[index]

    graph = build_graph(dataset)

    # Build training kwargs and conditionally include task loss/metric if classification labels are present.
    train_kwargs = dict(
        epochs=cfg["epochs"],  # type: ignore[index]
        lr=cfg["lr"],  # type: ignore[index]
        log_every=cfg["log_every"],  # type: ignore[index]
        seed=cfg["seed"],  # type: ignore[index]
        grad_clip=cfg["grad_clip"],  # type: ignore[index]
        val_dataset=val_dataset,
        val_every=cfg["val_every"],  # type: ignore[index]
    )
    task_cfg = CONFIG.get("task", {})  # type: ignore[assignment]
    include_task_loss = False
    if isinstance(task_cfg, dict) and task_cfg.get("enabled", False):
        try:
            out_dim = imprint.infer_num_classes(dataset, override=task_cfg.get("num_classes", None))
            include_task_loss = out_dim is not None and int(out_dim) > 0
        except Exception:
            include_task_loss = False
    if include_task_loss:
        head_name = str(task_cfg.get("head_name", "task_head"))
        label_key = str(task_cfg.get("label_key", "y"))
        task_w = float(task_cfg.get("weight", 1.0))
        ce_loss = imprint.last_step_ce_loss(head_name=head_name, label_key=label_key)
        def _combined_loss(graph: imprint.Graph, batch: Dict[str, "imprint.torch.Tensor"]) -> "imprint.torch.Tensor":  # type: ignore[name-defined]
            # Combine graph-local objectives (rec, pred, etc.) with final-step CE for the task head.
            return graph.loss() + task_w * ce_loss(graph, batch)
        # Expose a logging component so the trainer can report CE alongside graph objective breakdowns.
        try:
            def _aux_in_norm(g: imprint.Graph, b: Dict[str, "imprint.torch.Tensor"]):  # type: ignore[name-defined]
                m = g.modules.get("pc_aux")
                if m is None or "in" not in m.state.input_drive:
                    return imprint.torch.tensor(0.0)  # type: ignore[attr-defined]
                x = m.state.input_drive["in"]
                if x.dim() == 3:
                    x = x[:, -1, :]
                return x.abs().mean()
            def _aux_out_std(g: imprint.Graph, b: Dict[str, "imprint.torch.Tensor"]):  # type: ignore[name-defined]
                m = g.modules.get("pc_aux")
                if m is None or "out" not in m.state.output:
                    return imprint.torch.tensor(0.0)  # type: ignore[attr-defined]
                y = m.state.output["out"]
                if y.dim() == 3:
                    y = y[:, -1, :]
                return y.float().std()
            def _head_logit_std(g: imprint.Graph, b: Dict[str, "imprint.torch.Tensor"]):  # type: ignore[name-defined]
                h = g.modules.get(head_name)
                if h is None or "out" not in h.state.output:
                    return imprint.torch.tensor(0.0)  # type: ignore[attr-defined]
                z = h.state.output["out"]
                if z.dim() == 3:
                    z = z[:, -1, :]
                return z.float().std()
            _combined_loss._components = {  # type: ignore[attr-defined]
                f"{head_name}:ce": lambda g, b: task_w * ce_loss(g, b),
                "aux_in_norm": _aux_in_norm,
                "aux_out_std": _aux_out_std,
                "head_logit_std": _head_logit_std,
            }
        except Exception:
            pass
        train_kwargs.update(
            loss_fn=_combined_loss,  # type: ignore[assignment]
            metric_fn=imprint.last_step_accuracy(head_name=head_name, label_key=label_key),
        )

    imprint.train_graph(
        graph,
        dataset,
        **train_kwargs,  # type: ignore[arg-type]
    )

    print("Done.")


if __name__ == "__main__":
    run()
