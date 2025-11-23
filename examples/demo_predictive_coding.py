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
from typing import Dict, List, Sequence

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import imprint
from imprint import SequenceDataset
from imprint.objectives import Targets

Auto = imprint.Auto


CONFIG: Dict[str, object] = {
    "seed": 15,
    "epochs": 11,
    "lr": 1e-3,
    "log_every": 1,
    "val_every": 1,
    #"grad_clip": 1.0,
    "use_adamw": True,
    "weight_decay": 4e-2,
    "train_split": "train",
    "val_split": "val",
    "two_layers": True,
    "use_feedback": True,
    "confine_pc_gradients": True,
    "log_gradients": False,
    "visualize_val_sample": True,
    "input_scale": 11.0,
    "loss": {
        "rec": 1,
        "pred": 0,
        "sparse_err": 0,
    },
    "layers": [
        {
            "name": "pc0",
            "hidden": 64,
            "layers": 1,
            "decoder_widths": [32, Auto],
            "out_dim": 16,

        },
        {
            "name": "pc1",
            "hidden": 32,
            "layers": 1,
            "decoder_widths": [32, Auto],
            "out_dim": 16,
        },
    ],
    "predictor_widths": [128, Auto],
    "film_widths": [Auto],
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
        "path": "solids_16x16.h5",
        "batch_size": 192,
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
    apply_film: bool,
    film_widths: Sequence[int] | None,
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
    edge = graph.connect(signal_src, err["in.x"])
    if bool(CONFIG.get("confine_pc_gradients", False)):
        edge.set_stop_grad(True)
    # Freeze both error-path projections to identity so reconstruction is measured in the err space.
    edge.set_identity(True)
    pred_edge = graph.connect(decoder["out"], err["in.pred"])
    pred_edge.set_identity(True)

    film = None
    drive_src = err["out"]
    if apply_film:
        resolved_widths = spec.get("film_widths", film_widths)
        if resolved_widths is None:
            raise ValueError("FiLM widths must be provided when apply_film is enabled.")
        film = imprint.Module(
            name=f"{base}_film",
            proto=imprint.protos.FiLMConditioner(widths=list(resolved_widths)),
            ports=imprint.Ports(
                in_=imprint.InPortGroup(
                    layout="disjoint",
                    signal=imprint.InPort(size=Auto),
                    cond=imprint.InPort(size=Auto),
                ),
                out_default=Auto,
            ),
        )
        graph.add(film)
        graph.connect(err["out"], film["in.signal"])
        drive_src = film["out"]

    graph.connect(drive_src, gru["in"])

    modules: Dict[str, imprint.Module] = {"decoder": decoder, "err": err, "gru": gru}
    if film is not None:
        modules["film"] = film
    return modules


def _connect_top_down(
    graph: imprint.Graph,
    higher: Dict[str, imprint.Module],
    lower: Dict[str, imprint.Module],
) -> None:
    ctx = higher["gru"]["out"]
    edge_dec = graph.connect(ctx, lower["decoder"]["in"])
    film = lower.get("film")
    edge_target = None
    if film is not None:
        edge_target = graph.connect(ctx, film["in.cond"])
    else:
        edge_target = graph.connect(ctx, lower["gru"]["in"])
    if bool(CONFIG.get("confine_pc_gradients", False)):
        edge_dec.set_stop_grad(True)
        edge_target.set_stop_grad(True)


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
        # Keep targets in the err space, but edges into err are identity/frozen (see _add_pc_layer).
        target = Targets.port_drive(err, "in.x")
        decoder.objectives.mse("out", target, weight=rec_w, name="rec")
        if sparse_w > 0:
            err.objectives.activity_l1("out", weight=sparse_w, name="sparse_err")

    if predictor is not None and pred_w > 0:
        predictor.objectives.mse(
            "out",
            Targets.shifted_input(source, +8),
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

    default_film_widths = CONFIG.get("film_widths", None)
    use_feedback = bool(CONFIG.get("use_feedback", True))

    for idx, layer_spec in enumerate(layer_specs):
        apply_film = use_feedback and idx < len(layer_specs) - 1
        film_spec = layer_spec.get("film_widths")
        film_widths: Sequence[int] | None = None
        if isinstance(film_spec, Sequence) and not isinstance(film_spec, (str, bytes)):
            film_widths = film_spec  # type: ignore[assignment]
        elif isinstance(default_film_widths, Sequence) and not isinstance(
            default_film_widths, (str, bytes)
        ):
            film_widths = default_film_widths  # type: ignore[assignment]
        layer = _add_pc_layer(
            graph,
            layer_spec,  # type: ignore[arg-type]
            signal_src=signal,
            apply_film=apply_film,
            film_widths=film_widths,  # type: ignore[arg-type]
        )
        built_layers.append(layer)
        signal = layer["err"]["out"]
        if idx > 0 and use_feedback:
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
    output_dir = os.path.dirname(os.path.abspath(__file__))

    dataset, val_dataset = imprint.load_train_val_splits(
        data_cfg,
        train_split=str(cfg["train_split"]),  # type: ignore[index]
        val_split=cfg.get("val_split", "val"),  # type: ignore[arg-type]
    )

    scale = float(cfg.get("input_scale", 1.0))
    if scale != 1.0:
        imprint.scale_sequence_dataset(dataset, scale)
        if val_dataset is not None:
            imprint.scale_sequence_dataset(val_dataset, scale)

    graph = build_graph(dataset)
    grad_monitor = None
    if bool(cfg.get("log_gradients", False)):
        grad_monitor = imprint.GradientWatcher(
            graph,
            track_ports=True,
            track_edges=True,
        )

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

    viz_entries: List[Dict[str, object]] = []
    if bool(cfg.get("visualize_val_sample", False)):
        viz_entries.append(
            {
                "enabled": True,
                "module": cfg.get("visualize_val_module", "pc0_decoder"),
                "port": cfg.get("visualize_val_port", "out"),
                "mode": cfg.get("visualize_val_mode", "frame"),
                "dir": output_dir,
                "filename": cfg.get("visualize_val_path", "val_viz.png"),
            }
        )
        if bool(cfg.get("visualize_val_compare_input", True)):
            viz_entries.append(
                {
                    "enabled": True,
                    "module": cfg.get("visualize_val_input_module", "src_x"),
                    "port": cfg.get("visualize_val_input_port", "out"),
                    "mode": cfg.get("visualize_val_mode", "frame"),
                    "dir": output_dir,
                    "filename": cfg.get("visualize_val_input_path", "val_input.png"),
                }
            )
        viz_entries.append(
            {
                "enabled": float(cfg.get("loss", {}).get("pred", 0.0)) > 0.0,
                "module": cfg.get("visualize_val_pred_module", "pc_pred_head"),
                "port": cfg.get("visualize_val_pred_port", "out"),
                "mode": cfg.get(
                    "visualize_val_pred_mode", cfg.get("visualize_val_mode", "frame")
                ),
                "dir": output_dir,
                "filename": cfg.get("visualize_val_pred_path", "val_pred.png"),
            }
        )

    viz_cfg = {"train": [], "val": viz_entries}

    imprint.train_graph(
        graph,
        dataset,
        grad_monitor=grad_monitor,
        viz_config=viz_cfg,
        **train_kwargs,  # type: ignore[arg-type]
    )

    # Simple post-train monitoring: decoder variability across batch/time on a val batch.
    try:
        import torch  # type: ignore
        if val_dataset is not None:
            first_batch = next(val_dataset.iter_batches(shuffle=False))
        else:
            first_batch = next(dataset.iter_batches(shuffle=False))
        graph.rollout(first_batch)
        mod_name = str(cfg.get("visualize_val_module", "pc0_decoder"))
        if mod_name in graph.modules and "out" in graph.modules[mod_name].state.output:
            tensor = graph.modules[mod_name].state.output["out"]  # [B, T, D]
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(1)
            B, T, D = tensor.shape
            overall_std = float(tensor.std().item())
            std_over_time = float(tensor.std(dim=1).mean().item())     # variability across ticks
            std_over_feat = float(tensor.std(dim=2).mean().item())     # variability across features
            min_val = float(tensor.min().item())
            max_val = float(tensor.max().item())
            print(
                f"[monitor] {mod_name}.out stats on {'val' if val_dataset is not None else 'train'} batch: "
                f"shape=[{B},{T},{D}] std_all={overall_std:.4e} "
                f"mean(std_over_time)={std_over_time:.4e} mean(std_over_feat)={std_over_feat:.4e} "
                f"min={min_val:.4e} max={max_val:.4e}"
            )
    except Exception as _exc:
        # Best-effort logging only; keep demo robust.
        pass

    print("Done.")
    if grad_monitor is not None:
        grad_monitor.close()


if __name__ == "__main__":
    run()
