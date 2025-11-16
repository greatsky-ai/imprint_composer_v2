from __future__ import annotations

from typing import Optional

from .core import Graph
from .objectives import Targets
from .data_helper import SequenceDataset


def prepare_seq2static_classification(
    graph: Graph,
    dataset: SequenceDataset,
    *,
    head_name: str = "head",
    label_key: str = "y",
    emit_once: bool = True,
) -> None:
    """
    Configure a graph for sequence-to-static classification:
      - Optionally make the head emit once per sequence.
      - Add a CE objective on the head's 'out' port against batch[label_key].
    """
    head = graph.modules[head_name]
    if emit_once:
        head.schedule.emit_every = dataset.seq_len
    head.objectives.ce(on="out", target=Targets.batch_key(label_key))


