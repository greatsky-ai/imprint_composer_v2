# imprint/__init__.py

from .core import (
    Graph,
    Clock,
    Rate,
    Auto,
    Module,
    Source,
    Ports,
    InPort,
    InPortGroup,
    Nodes,
    PortRef,
    Edge,
)

from .objectives import Targets
from .record import record, Trace, Plan
from .data_helper import SequenceDataset, load_demo_dataset, scale_sequence_dataset
from .training import train_graph
from .recipes import (
    prepare_seq2static_classification,
    last_step_ce_loss,
    last_step_accuracy,
    infer_num_classes,
    detect_task_mode,
    load_train_val_splits,
    trainer_kwargs_from_config,
    combined_graph_and_ce_loss,
    attach_task_head,
)
from .diagnostics import (
    GradientWatcher,
    GradientSummary,
    plot_gradient_heatmap,
    visualize_module_output,
)
from . import masks
from . import protos

__all__ = [
    "Graph",
    "Clock",
    "Rate",
    "Auto",
    "Module",
    "Source",
    "Ports",
    "InPort",
    "InPortGroup",
    "Nodes",
    "PortRef",
    "Edge",
    "Targets",
    "record",
    "Trace",
    "Plan",
    "SequenceDataset",
    "load_demo_dataset",
    "scale_sequence_dataset",
    "train_graph",
    "prepare_seq2static_classification",
    "last_step_ce_loss",
    "last_step_accuracy",
    "infer_num_classes",
    "detect_task_mode",
    "load_train_val_splits",
    "trainer_kwargs_from_config",
    "combined_graph_and_ce_loss",
    "attach_task_head",
    "GradientWatcher",
    "GradientSummary",
    "plot_gradient_heatmap",
    "visualize_module_output",
    "masks",
    "protos",
]
