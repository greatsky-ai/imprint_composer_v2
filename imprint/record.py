# imprint/record.py

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

from .core import Graph


@dataclass(frozen=True)
class RolloutEvent:
    """
    Lightweight description of a single module emission captured during rollout.
    """

    tick: int
    module: str
    port: str
    shape: Tuple[int, ...]
    dtype: Optional[str]
    device: Optional[str]


class Trace:
    """
    Recording of a Graph rollout.

    Responsibilities:
      - Capture the sequence of module emissions and timing metadata.
      - Provide a compile() API that materialises a callable Plan.
    """

    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self._events: List[RolloutEvent] = []
        self._tick_count: Optional[int] = None
        self._module_emits: Dict[str, int] = {}
        self._metadata: Dict[str, Any] = {}
        self._active = False

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        if self._active:
            return
        self._events.clear()
        self._module_emits.clear()
        self._metadata.clear()
        self._tick_count = None
        self.graph.register_event_listener(self._handle_event)
        self._active = True

    def stop(self) -> None:
        if not self._active:
            return
        self.graph.unregister_event_listener(self._handle_event)
        self._active = False

    # ---------------------------------------------------------------- listeners
    def _handle_event(self, payload: Dict[str, Any]) -> None:
        kind = payload.get("event")
        if kind == "module_emit":
            event = RolloutEvent(
                tick=int(payload["tick"]),
                module=str(payload["module"]),
                port=str(payload["port"]),
                shape=tuple(int(dim) for dim in payload.get("shape", ())),
                dtype=payload.get("dtype"),
                device=payload.get("device"),
            )
            self._events.append(event)
            self._module_emits[event.module] = self._module_emits.get(event.module, 0) + 1
        elif kind == "tick_end":
            tick = int(payload["tick"])
            self._tick_count = max(self._tick_count or 0, tick + 1)
        elif kind == "rollout_start":
            self._metadata["requested_ticks"] = payload.get("ticks")
        elif kind == "rollout_end":
            self._metadata["completed_ticks"] = payload.get("ticks")

    # ----------------------------------------------------------------- metadata
    @property
    def events(self) -> Tuple[RolloutEvent, ...]:
        return tuple(self._events)

    @property
    def tick_count(self) -> Optional[int]:
        return self._tick_count

    def summary(self) -> Dict[str, Any]:
        return {
            "ticks": self._metadata.get("completed_ticks", self._tick_count),
            "modules": dict(self._module_emits),
            "events": len(self._events),
        }

    # ------------------------------------------------------------------- output
    def compile(
        self,
        backend: str = "python",
        **options: Any,
    ) -> "Plan":
        """
        Compile the recorded execution into a reusable Plan facade.
        """
        return Plan(
            graph=self.graph,
            tick_count=self._metadata.get("completed_ticks", self._tick_count),
            module_schedule=dict(self._module_emits),
            backend=backend,
            options=options,
        )


class Plan:
    """
    Executable compiled from a Trace.

    This reference implementation simply replays the original Graph using the
    recorded tick count. Future backends could swap in accelerators.
    """

    def __init__(
        self,
        graph: Graph,
        tick_count: Optional[int],
        module_schedule: Dict[str, int],
        backend: str,
        options: Dict[str, Any],
    ) -> None:
        self.graph = graph
        self.tick_count = tick_count
        self.module_schedule = module_schedule
        self.backend = backend
        self.options = dict(options)

    def rollout(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Execute the recorded schedule on a new batch.
        """
        return self.graph.rollout(batch, max_ticks=self.tick_count)

    def loss(self) -> torch.Tensor:
        """
        Compute loss from module-local objectives for the last rollout.
        """
        return self.graph.loss()

    def describe(self) -> Dict[str, Any]:
        """
        Provide lightweight metadata about the compiled plan.
        """
        return {
            "backend": self.backend,
            "tick_count": self.tick_count,
            "module_schedule": self.module_schedule,
            "options": self.options,
        }


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
    trace.start()
    try:
        yield trace
    finally:
        trace.stop()
