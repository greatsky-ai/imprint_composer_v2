# imprint/record.py

from __future__ import annotations
from typing import Any, Dict, Iterator, Optional
from contextlib import contextmanager
import torch
from .core import Graph


class Trace:
    """
    Recording of a Graph rollout.

    Responsibilities:
      - Capture the sequence of module activations and port operations.
      - Provide an API to compile this schedule into a faster executable Plan.
    """

    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        # Internal structures to store schedule, op graphs, etc.
        self._events: Any = None

    def compile(
        self,
        backend: str = "cuda_graph",
        **options: Any,
    ) -> "Plan":
        """
        Compile the recorded execution into a reusable Plan.

        Args:
          backend: Compilation backend ('cuda_graph', 'torch_compile', 'xla', ...).
          options: Backend-specific options such as fusion flags.

        Returns:
          Plan object with rollout() and loss() methods mirroring Graph.
        """
        return Plan(self.graph, self, backend=backend, options=options)


class Plan:
    """
    Executable compiled from a Trace.

    Responsibilities:
      - Replay the recorded schedule efficiently for new batches.
      - Expose rollout() and loss() similar to Graph.
    """

    def __init__(
        self,
        graph: Graph,
        trace: Trace,
        backend: str,
        options: Dict[str, Any],
    ) -> None:
        self.graph = graph
        self.trace = trace
        self.backend = backend
        self.options = options

    def rollout(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Execute the compiled schedule on a new batch.
        """
        ...

    def loss(self) -> torch.Tensor:
        """
        Compute loss from module-local objectives for the last rollout.
        """
        ...


@contextmanager
def record(graph: Graph) -> Iterator[Trace]:
    """
    Context manager to record a single Graph rollout.

    Usage:
      with imprint.record(g) as trace:
          g.rollout(batch)
      plan = trace.compile(...)
    """
    trace = Trace(graph)
    # Implementation would enable recording mode on the graph/scheduler
    try:
        yield trace
    finally:
        # Disable recording mode
        ...
