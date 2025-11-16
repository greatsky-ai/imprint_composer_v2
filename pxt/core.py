# imprint/core.py

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Iterable, Callable, Tuple
import torch
import torch.nn as nn

# Sentinel for auto-inferred dimensions
class _AutoDim:
    """Sentinel used in configuration to mean 'infer this dimension at bind() time'."""
    def __repr__(self) -> str:
        return "Auto"

Auto = _AutoDim()


class Clock:
    """
    Global clock for a Graph.

    Responsibilities:
      - Maintain a monotonically increasing 'tick' counter for external events.
      - Provide APIs to step the clock and query current time.
      - Used by the scheduler to coordinate multi-rate modules.
    """

    def __init__(self) -> None:
        self._tick: int = 0

    @property
    def tick(self) -> int:
        """Current external tick (integer)."""
        return self._tick

    def step(self, n: int = 1) -> int:
        """Advance clock by n ticks (default 1) and return new tick."""
        self._tick += n
        return self._tick


class Rate:
    """
    Scheduling metadata for a module.

    Semantics:
      - inner_steps: number of internal compute steps per external clock tick.
      - emit_every: emit outputs every Nth external tick.
      - follow: optionally tie this module's emission schedule to another module.

    Notes:
      - When 'follow' is set, inner_steps/emit_every may be ignored or interpreted
        relative to the followed module's emissions.
    """

    def __init__(
        self,
        inner_steps: int = 1,
        emit_every: int = 1,
        follow: Optional[Union[str, "Module"]] = None,
    ) -> None:
        if inner_steps < 1:
            raise ValueError("inner_steps must be >= 1")
        if emit_every < 1:
            raise ValueError("emit_every must be >= 1")
        self.inner_steps = inner_steps
        self.emit_every = emit_every
        self.follow = follow

    @classmethod
    def follow(cls, module_or_name: Union[str, "Module"]) -> "Rate":
        """
        Convenience constructor: emit in lockstep with another module.
        """
        return cls(inner_steps=1, emit_every=1, follow=module_or_name)


class InPort:
    """
    Definition of a single input port.

    Responsibilities:
      - Declare the port's target dimension (may be Auto).
      - Hold metadata about how incoming edges are combined (sum, concat, etc.).
      - Own a mapping from incoming Edge -> projection tensor (handled by Graph/Module at bind()).
    """

    def __init__(
        self,
        size: Union[int, _AutoDim] = Auto,
        combine: str = "sum",
        name: Optional[str] = None,
    ) -> None:
        """
        Args:
          size: Target feature dimension for this port (Auto to infer at bind()).
          combine: How to combine contributions from multiple edges ('sum', 'concat').
          name: Optional logical name; set automatically when used within InPortGroup.
        """
        self.size = size
        self.combine = combine
        self.name = name  # Assigned by InPortGroup if not set


class InPortGroup:
    """
    A named group of input subports, with a layout invariant.

    Responsibilities:
      - Expose subports as attributes: group.deep, group.wide, etc.
      - Enforce layout constraints like 'disjoint' or 'shared'.
      - Provide a unified representation for the module's internal input state.
    """

    def __init__(
        self,
        layout: str = "disjoint",
        **subports: InPort,
    ) -> None:
        """
        Args:
          layout:
            - 'disjoint': each subport occupies a disjoint slice of a shared internal
              representation.
            - 'shared': subports may overlap; layout managed by edges/masks.
          subports: Mapping from subport name -> InPort(size=..., ...).
        """
        if layout not in ("disjoint", "shared"):
            raise ValueError("layout must be 'disjoint' or 'shared'")
        self.layout = layout
        self.subports: Dict[str, InPort] = {}

        for name, port in subports.items():
            port.name = name
            self.subports[name] = port

    def __getitem__(self, name: str) -> InPort:
        return self.subports[name]

    def __iter__(self):
        return iter(self.subports.items())


class Ports:
    """
    Container for module port declarations.

    Responsibilities:
      - Describe default input and output ports.
      - Optionally hold a grouped input port (InPortGroup).
      - Provide a registry for additional named output ports defined later.
    """

    def __init__(
        self,
        in_default: Union[int, _AutoDim, None] = None,
        out_default: Union[int, _AutoDim, None] = None,
        in_: Optional[Union[InPort, InPortGroup]] = None,
        out: Optional[Any] = None,
    ) -> None:
        """
        Args:
          in_default: Default input port size for 'in'; ignored if in_ is provided.
          out_default: Default output port size for 'out'; may be Auto.
          in_: Optional explicit InPort or InPortGroup for input.
          out: Reserved for future structured output port groups; typically omitted.
        """
        self.in_default = in_default
        self.out_default = out_default
        self.in_ = in_
        self.out = out

        # Populated during Module construction/bind
        self.output_ports: Dict[str, "Nodes"] = {}  # name -> Nodes selection


class Nodes:
    """
    Selection of internal nodes (neurons) within a module.

    Responsibilities:
      - Encode layer index and index/slice information.
      - Used to define output ports that view subsets of internal activations.
    """

    def __init__(
        self,
        layer: Optional[int] = None,
        indices: Optional[Iterable[int]] = None,
        slice_: Optional[slice] = None,
    ) -> None:
        """
        Exactly one of indices or slice_ should be provided (or both None for 'all').
        """
        self.layer = layer
        self.indices = list(indices) if indices is not None else None
        self.slice_ = slice_

    @classmethod
    def of(
        cls,
        layer: Optional[int] = None,
        indices: Optional[Iterable[int]] = None,
        slice: Optional[slice] = None,
    ) -> "Nodes":
        """
        Convenience constructor.
        """
        return cls(layer=layer, indices=indices, slice_=slice)


class PortRef:
    """
    A reference to a specific port on a module.

    Responsibilities:
      - Identify (module, port_name).
      - Provide access to metadata like port size.
      - Allows Graph.connect() to be expressed as enc["out"] → agg["in.deep"].
    """

    def __init__(self, module: "Module", name: str) -> None:
        self.module = module
        self.name = name

    def __repr__(self) -> str:
        return f"PortRef({self.module.name!r}, {self.name!r})"


class Edge:
    """
    Directed connection between two ports.

    Responsibilities:
      - Track source/target PortRef.
      - Own projection parameters from source dimension → target port dimension.
      - Handle persistent masks and magnitude constraints on its weights.
    """

    def __init__(
        self,
        name: Optional[str],
        src: PortRef,
        dst: PortRef,
    ) -> None:
        """
        Args:
          name: Optional human-readable name for the edge.
          src: Source port.
          dst: Destination port.
        """
        self.name = name or f"{src.module.name}.{src.name}→{dst.module.name}.{dst.name}"
        self.src = src
        self.dst = dst

        # Populated at bind() time:
        self.proj: Optional[nn.Linear] = None
        self._mask: Optional[torch.Tensor] = None
        self._max_abs: Optional[float] = None

    def constrain(
        self,
        mask: Optional[torch.Tensor] = None,
        max_abs: Optional[float] = None,
        persist: bool = True,
    ) -> None:
        """
        Set structured constraints on this edge's projection weights.

        Args:
          mask: Optional binary or float mask broadcastable to proj.weight, applied
                multiplicatively.
          max_abs: Optional maximum absolute value for proj weights.
          persist: If True, mask is reapplied after each optimizer step.
        """
        self._mask = mask
        self._max_abs = max_abs
        # Registration of hooks / buffers is left to implementation.


class ModuleState:
    """
    Per-module state tensors.

    Responsibilities:
      - Store input-drive per input port (after projections).
      - Store output activations per output port.
      - Store recurrent hidden state if applicable.
    """

    def __init__(self) -> None:
        # port_name -> tensor [B, T, D_port]
        self.input_drive: Dict[str, torch.Tensor] = {}
        # port_name -> tensor [B, T, D_port]
        self.output: Dict[str, torch.Tensor] = {}
        # Implementation-defined recurrent state (e.g., GRU hidden)
        self.recurrent: Any = None


class Objectives:
    """
    Container for per-module training objectives.

    Responsibilities:
      - Store a list of local loss terms and their weights.
      - Provide convenience constructors for common objectives.

    The Graph aggregates these into a global loss.
    """

    def __init__(self, module: "Module") -> None:
        self.module = module
        self._terms: List[Tuple[Callable[[], torch.Tensor], float]] = []

    def add(self, fn: Callable[[], torch.Tensor], weight: float = 1.0) -> None:
        """Register a custom objective as a zero-arg callable returning a scalar loss."""
        self._terms.append((fn, weight))

    # Convenience APIs; implementations will capture the right tensors/targets

    def mse(
        self,
        on: str,
        target: "TargetsSpec",
        weight: float = 1.0,
    ) -> None:
        """Mean squared error between module.state.output[on] and target."""
        ...

    def ce(
        self,
        on: str,
        target: "TargetsSpec",
        weight: float = 1.0,
    ) -> None:
        """Cross-entropy loss between logits at port 'on' and target labels."""
        ...

    def activity_l1(self, on: str, weight: float = 1.0) -> None:
        """L1 penalty on module.state.output[on]."""
        ...

    def activity_l2(self, on: str, weight: float = 1.0) -> None:
        """L2 penalty on module.state.output[on]."""
        ...

    def compute(self) -> torch.Tensor:
        """Sum all local loss terms (each multiplied by its weight)."""
        ...


class Module(nn.Module):
    """
    Core building block: a module with ports, internal state, and a schedule.

    Responsibilities:
      - Wrap a 'proto' (the internal computation, e.g., GRUStack, MLP).
      - Expose named input/output ports.
      - Maintain input-drive and output states per port.
      - Provide an interface to a scheduler via Rate.
      - Own local Objectives.
    """

    def __init__(
        self,
        name: str,
        proto: nn.Module,
        ports: Optional[Ports] = None,
        schedule: Optional[Rate] = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.proto = proto
        self.ports = ports or Ports()
        self.schedule = schedule or Rate()
        self.state = ModuleState()
        self.objectives = Objectives(self)

        # Port metadata populated at bind() time
        self._input_ports: Dict[str, Union[InPort, InPortGroup]] = {}
        self._output_ports: Dict[str, Nodes] = {}  # 'out', plus any custom ports

        # Mask constraints for module-level weights (e.g. recurrent)
        self._input_masks: Dict[str, torch.Tensor] = {}
        self._recurrent_mask: Optional[torch.Tensor] = None
        self._recurrent_max_abs: Optional[float] = None

    def __getitem__(self, port_name: str) -> PortRef:
        """
        Shortcut to construct a PortRef: enc["out"].
        """
        return PortRef(self, port_name)

    # --- Port definition APIs ---

    def define_port(self, name: str, nodes: Nodes) -> None:
        """
        Define a named output port as a view of internal nodes.

        Args:
          name: Port name (e.g., 'out_deep').
          nodes: Selection of internal nodes this port exposes.
        """
        self._output_ports[name] = nodes

    def port_ref(self, name: str) -> PortRef:
        """Explicitly get a PortRef to a named port."""
        return PortRef(self, name)

    # --- Weight constraint APIs ---

    def set_input_mask(
        self,
        port: str,
        mask: torch.Tensor,
        persist: bool = True,
    ) -> None:
        """
        Apply a persistent mask to the input projection weights for a given port.
        """
        self._input_masks[port] = mask
        ...

    def set_recurrent_mask(
        self,
        mask: torch.Tensor,
        persist: bool = True,
    ) -> None:
        """
        Apply a persistent mask to the recurrent weights of the proto (if any).
        """
        self._recurrent_mask = mask
        ...

    def limit_weights(self, port: str, max_abs: float) -> None:
        """
        Enforce a max absolute value on the input projection weights for a port.
        """
        ...

    def limit_recurrent(self, max_abs: float) -> None:
        """
        Enforce a max absolute value on the recurrent weights of the proto.
        """
        ...

    # --- Hooks used by the scheduler / Graph ---

    def step_once(
        self,
        drive: Dict[str, torch.Tensor],
    ) -> None:
        """
        Perform a single internal compute step given current input-drive.

        This is the 'single-step' API the scheduler calls inner_steps times per external tick.
        Implementation calls into self.proto and updates self.state.recurrent and possibly
        self.state.output[...] for ports that emit at this step.
        """
        raise NotImplementedError

    def should_emit_at(self, tick: int, ref_tick: Optional[int] = None) -> bool:
        """
        Return True if this module should emit outputs at the given external tick.

        ref_tick is used when Rate.follow(...) is in effect.
        """
        ...


class Source(Module):
    """
    Special module that injects external sequences into the graph.

    Responsibilities:
      - Read tensors from the input batch (e.g., batch['x']).
      - Emit events on its 'out' port at each clock tick.
    """

    def __init__(
        self,
        batch_key: str,
        name: Optional[str] = None,
    ) -> None:
        """
        Args:
          batch_key: Key into the batch dict whose value is a tensor [B, T, D].
          name: Optional name; default 'src_{batch_key}'.
        """
        super().__init__(
            name=name or f"src_{batch_key}",
            proto=nn.Identity(),  # No learnable params by default
            ports=Ports(in_default=None, out_default=Auto),
            schedule=Rate(inner_steps=1, emit_every=1),
        )
        self.batch_key = batch_key


class Graph:
    """
    Container for modules and their connectivity.

    Responsibilities:
      - Own a Clock and a set of Modules/Edges.
      - Provide methods to add modules and connect ports.
      - Bind shapes and construct projections/masks at bind().
      - Run multi-rate rollouts given a batch.
      - Aggregate module-local objectives into a global loss.
    """

    def __init__(self, clock: Optional[Clock] = None) -> None:
        self.clock = clock or Clock()
        self.modules: Dict[str, Module] = {}
        self.edges: List[Edge] = []

    # --- Construction APIs ---

    def add(self, *modules: Module) -> None:
        """
        Register one or more modules (or Sources) with the graph.
        """
        for m in modules:
            if m.name in self.modules:
                raise ValueError(f"Duplicate module name {m.name!r}")
            self.modules[m.name] = m

    def connect(
        self,
        src: PortRef,
        dst: PortRef,
        name: Optional[str] = None,
    ) -> Edge:
        """
        Connect src → dst with an Edge. Returns the created Edge.

        Args:
          src: Source port reference (e.g., enc["out"]).
          dst: Destination port reference (e.g., agg["in.deep"]).
          name: Optional edge name.
        """
        edge = Edge(name=name, src=src, dst=dst)
        self.edges.append(edge)
        return edge

    def edge(
        self,
        name: Optional[str] = None,
        src: Optional[PortRef] = None,
        dst: Optional[PortRef] = None,
    ) -> Edge:
        """
        Retrieve an edge by name or (src, dst) pair.
        """
        ...

    # --- Binding & rollout ---

    def bind(self, batch: Dict[str, torch.Tensor]) -> None:
        """
        Infer Auto dimensions, build projections, and initialize module states.

        Responsibilities:
          - Inspect Source modules to determine input shapes.
          - Infer all Auto dimensions for ports/protos.
          - Allocate nn.Linear projections for edges with appropriate shapes.
          - Set up port groups layouts (e.g., disjoint slices).
        """
        ...

    def rollout(
        self,
        batch: Optional[Dict[str, torch.Tensor]] = None,
        max_ticks: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Execute the graph over a sequence of external ticks.

        Args:
          batch: Optional input batch dict. If None, reuse last bound batch.
          max_ticks: Optional limit on number of external ticks.

        Returns:
          A mapping from 'module.port' to collected output tensors.
        """
        ...

    def loss(self) -> torch.Tensor:
        """
        Compute total loss as the sum of all module-local objectives.

        Returns:
          Scalar tensor suitable for backprop.
        """
        ...


# Type alias used by Objectives/Targets
TargetsSpec = Any
