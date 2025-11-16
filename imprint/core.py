# imprint/core.py

from __future__ import annotations

import math
from collections import defaultdict, OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .objectives import Objectives, TargetsSpec

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
    Reference to a module port. Carries basic direction metadata.
    """

    def __init__(self, module: "Module", name: str, kind: str = "output") -> None:
        self.module = module
        self.name = name
        self.kind = kind  # 'output', 'input', or 'drive'

    def __repr__(self) -> str:
        return f"PortRef({self.module.name!r}, {self.name!r}, kind={self.kind!r})"

    @property
    def drive(self) -> "PortRef":
        if self.kind != "input":
            raise ValueError("Only input ports expose an input-drive reference.")
        return PortRef(self.module, self.name, kind="drive")


class MaskedLinear(nn.Linear):
    """
    nn.Linear that supports persistent masks and max-abs constraints.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.register_buffer("_mask", None)
        self._max_abs: Optional[float] = None

    def set_mask(self, mask: Optional[torch.Tensor]) -> None:
        if mask is None:
            self._mask = None
            return
        mask = mask.to(dtype=self.weight.dtype, device=self.weight.device)
        if mask.shape == self.weight.shape:
            expanded = mask
        elif mask.shape == (self.weight.shape[1],):
            expanded = mask.unsqueeze(0).expand_as(self.weight)
        elif mask.shape == (self.weight.shape[0],):
            expanded = mask.unsqueeze(1).expand_as(self.weight)
        else:
            raise ValueError(
                f"Mask shape {mask.shape} must match or broadcast to weight shape {self.weight.shape}."
            )
        self._mask = expanded

    def set_max_abs(self, max_abs: Optional[float]) -> None:
        self._max_abs = None if max_abs is None else float(max_abs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self._max_abs is not None:
            with torch.no_grad():
                self.weight.clamp_(-self._max_abs, self._max_abs)
        weight = self.weight
        if self._mask is not None:
            weight = weight * self._mask
        return F.linear(input, weight, self.bias)


class Edge:
    """
    Directed connection between two ports backed by a projection.
    """

    def __init__(
        self,
        name: Optional[str],
        src: PortRef,
        dst: PortRef,
    ) -> None:
        self.name = name or f"{src.module.name}.{src.name}→{dst.module.name}.{dst.name}"
        self.src = src
        self.dst = dst
        self.proj: Optional[MaskedLinear] = None
        self._mask: Optional[torch.Tensor] = None
        self._max_abs: Optional[float] = None

    def constrain(
        self,
        mask: Optional[torch.Tensor] = None,
        max_abs: Optional[float] = None,
        persist: bool = True,
    ) -> None:
        del persist  # MaskedLinear inherently persists constraints.
        self._mask = mask
        self._max_abs = max_abs
        if self.proj is not None:
            if mask is not None:
                self.proj.set_mask(mask)
            if max_abs is not None:
                self.proj.set_max_abs(max_abs)


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
        self.recurrent: Any = None


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

        self._input_specs: Dict[str, Union[InPort, InPortGroup]] = {}
        self._input_groups: Dict[str, InPortGroup] = {}
        self._register_input_ports()

        self._input_masks: Dict[str, torch.Tensor] = {}
        self._input_max_abs: Dict[str, float] = {}
        self._recurrent_mask: Optional[torch.Tensor] = None
        self._recurrent_max_abs: Optional[float] = None

        self._output_dims: Dict[str, Optional[Union[int, _AutoDim]]] = {
            "out": self.ports.out_default
        }
        self._custom_output_nodes: Dict[str, Nodes] = dict(self.ports.output_ports)

        self._pending_outputs: Dict[str, torch.Tensor] = {}
        self._last_layer_outputs: Optional[List[torch.Tensor]] = None
        self._input_history: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self._output_history: Dict[str, List[torch.Tensor]] = defaultdict(list)

    def _register_input_ports(self) -> None:
        self._input_specs.clear()
        self._input_groups.clear()
        if isinstance(self.ports.in_, InPortGroup):
            self._input_groups["in"] = self.ports.in_
            self._input_specs["in"] = self.ports.in_
            for name, sub in self.ports.in_.subports.items():
                self._input_specs[f"in.{name}"] = sub
        elif isinstance(self.ports.in_, InPort):
            port = self.ports.in_
            port.name = port.name or "in"
            self._input_specs[port.name] = port
        elif self.ports.in_default is not None or self.ports.in_default is Auto:
            self._input_specs["in"] = InPort(size=self.ports.in_default, name="in")

    def _is_input_port(self, name: str) -> bool:
        return name in self._input_specs

    def __getitem__(self, port_name: str) -> PortRef:
        kind = "input" if self._is_input_port(port_name) else "output"
        return PortRef(self, port_name, kind=kind)

    # --- Port definition APIs ---

    def define_port(self, name: str, nodes: Nodes) -> None:
        """
        Define a named output port as a view of internal nodes.

        Args:
          name: Port name (e.g., 'out_deep').
          nodes: Selection of internal nodes this port exposes.
        """
        self._custom_output_nodes[name] = nodes
        self._output_dims.setdefault(name, None)

    def port_ref(self, name: str) -> PortRef:
        """Explicitly get a PortRef to a named port."""
        kind = "input" if self._is_input_port(name) else "output"
        return PortRef(self, name, kind=kind)

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

    def set_recurrent_mask(
        self,
        mask: torch.Tensor,
        persist: bool = True,
    ) -> None:
        """
        Apply a persistent mask to the recurrent weights of the proto (if any).
        """
        self._recurrent_mask = mask

    def limit_weights(self, port: str, max_abs: float) -> None:
        """
        Enforce a max absolute value on the input projection weights for a port.
        """
        self._input_max_abs[port] = max_abs

    def limit_recurrent(self, max_abs: float) -> None:
        """
        Enforce a max absolute value on the recurrent weights of the proto.
        """
        self._recurrent_max_abs = max_abs

    # --- Hooks used by the scheduler / Graph ---

    def should_emit_at(self, tick: int, follow_emitted: Optional[bool]) -> bool:
        if self.schedule.follow is not None:
            return bool(follow_emitted)
        return tick % self.schedule.emit_every == 0

    # --- Rollout bookkeeping -------------------------------------------------

    def _device(self) -> torch.device:
        param = next(self.parameters(), None)
        if param is not None:
            return param.device
        return torch.device("cpu")

    def _prepare_for_rollout(self, batch_size: int, ticks: int, device: torch.device) -> None:
        self._input_history = defaultdict(list)
        self._output_history = defaultdict(list)
        self._pending_outputs = {}
        self._last_layer_outputs = None
        self.state.input_drive = {}
        self.state.output = {}
        init_state = getattr(self.proto, "init_state", None)
        if callable(init_state):
            self.state.recurrent = init_state(batch_size, device=device)
        else:
            self.state.recurrent = None

    def _record_input_drive(self, port: str, tensor: torch.Tensor) -> None:
        self._input_history[port].append(tensor)

    def _record_output(self, port: str, tensor: torch.Tensor) -> None:
        self._output_history[port].append(tensor)

    def _finalize_state(
        self,
        batch_size: int,
        ticks: int,
        device: torch.device,
        port_dims: Dict[Tuple[str, str], int],
    ) -> None:
        input_tensors: Dict[str, torch.Tensor] = {}
        for port, history in self._input_history.items():
            if history:
                input_tensors[port] = torch.stack(history, dim=1)
            else:
                dim = port_dims.get((self.name, port))
                if dim is None:
                    continue
                input_tensors[port] = torch.zeros(batch_size, ticks, dim, device=device)
        self.state.input_drive = input_tensors

        output_tensors: Dict[str, torch.Tensor] = {}
        for port, history in self._output_history.items():
            if history:
                output_tensors[port] = torch.stack(history, dim=1)
            else:
                dim = port_dims.get((self.name, port))
                if dim is None:
                    continue
                output_tensors[port] = torch.zeros(batch_size, 0, dim, device=device)
        self.state.output = output_tensors

    # --- Stepping ------------------------------------------------------------

    def _prepare_proto_input(self, drive: Dict[str, Any]) -> Any:
        if "in" not in self._input_specs:
            return None
        spec = self._input_specs["in"]
        if isinstance(spec, InPortGroup):
            group_inputs: Dict[str, torch.Tensor] = {}
            for name in spec.subports:
                key = f"in.{name}"
                if key not in drive:
                    raise KeyError(f"Missing drive for subport {key!r}")
                group_inputs[name] = drive[key]
            return group_inputs
        return drive.get("in")

    def _render_custom_port(self, nodes: Nodes) -> torch.Tensor:
        if nodes.layer is None:
            base = self._pending_outputs["out"]
        else:
            if self._last_layer_outputs is None:
                raise ValueError(
                    f"Module {self.name} defines layered nodes but proto does not expose layers."
                )
            layer_idx = nodes.layer
            if layer_idx < 0 or layer_idx >= len(self._last_layer_outputs):
                raise IndexError(f"Invalid layer index {layer_idx} for module {self.name}.")
            base = self._last_layer_outputs[layer_idx]
        if nodes.indices is not None:
            index = torch.as_tensor(nodes.indices, device=base.device, dtype=torch.long)
            return base.index_select(dim=-1, index=index)
        if nodes.slice_ is not None:
            return base[..., nodes.slice_]
        return base

    def _run_step(self, drive: Dict[str, Any], tick: int) -> None:
        del tick  # unused by default
        primary = self._prepare_proto_input(drive)
        if hasattr(self.proto, "step") and hasattr(self.proto, "init_state"):
            if primary is None:
                raise ValueError(f"Module {self.name} requires an input drive but none provided.")
            if self.state.recurrent is None:
                raise RuntimeError("Recurrent module missing state; bind() must be called first.")
            new_state, layer_outputs = self.proto.step(primary, self.state.recurrent)  # type: ignore[attr-defined]
            self.state.recurrent = new_state
            self._last_layer_outputs = layer_outputs
            self._pending_outputs = {"out": layer_outputs[-1]}
        else:
            payload = primary if primary is not None else drive.get("in")
            if payload is None:
                raise ValueError(f"Module {self.name} requires an input drive but none provided.")
            if isinstance(payload, dict):
                if hasattr(self.proto, "mode"):
                    output = self.proto(payload)
                else:
                    ordered_args = [payload[name] for name in payload]
                    output = self.proto(*ordered_args)
            else:
                output = self.proto(payload)
            if isinstance(output, dict):
                raise TypeError("Module proto returned dict; expected tensor.")
            self._last_layer_outputs = None
            self._pending_outputs = {"out": output}

        for name, nodes in self._custom_output_nodes.items():
            self._pending_outputs[name] = self._render_custom_port(nodes)


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
            proto=nn.Identity(),
            ports=Ports(in_default=None, out_default=Auto),
            schedule=Rate(inner_steps=1, emit_every=1),
        )
        self.batch_key = batch_key
        self._data: Optional[torch.Tensor] = None

    def bind_data(self, tensor: torch.Tensor) -> None:
        if tensor.dim() != 3:
            raise ValueError(f"Source {self.name} expects [B, T, D] tensor.")
        self._data = tensor
        self._output_dims["out"] = tensor.shape[-1]

    def _run_step(self, drive: Dict[str, Any], tick: int) -> None:  # type: ignore[override]
        del drive
        if self._data is None:
            raise RuntimeError(f"Source {self.name} has not been bound to data.")
        B, T, D = self._data.shape
        if tick < T:
            value = self._data[:, tick, :]
        else:
            value = torch.zeros(B, D, device=self._data.device, dtype=self._data.dtype)
        self._pending_outputs = {"out": value}
        self._last_layer_outputs = None


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
        self.modules: "OrderedDict[str, Module]" = OrderedDict()
        self.edges: List[Edge] = []

        self._incoming: Dict[Tuple[str, str], List[Edge]] = defaultdict(list)
        self._outgoing: Dict[Tuple[str, str], List[Edge]] = defaultdict(list)
        self._drive_outgoing: Dict[Tuple[str, str], List[Edge]] = defaultdict(list)
        self._structure_dirty = True

        self._current_batch: Optional[Dict[str, torch.Tensor]] = None
        self._batch_size: Optional[int] = None
        self._max_ticks: int = 0
        self._port_dims: Dict[Tuple[str, str], int] = {}
        self._module_devices: Dict[str, torch.device] = {}
        self._follow_targets: Dict[str, Optional[str]] = {}

    # --- Construction APIs ---

    def add(self, *modules: Module) -> None:
        """
        Register one or more modules (or Sources) with the graph.
        """
        for m in modules:
            if m.name in self.modules:
                raise ValueError(f"Duplicate module name {m.name!r}")
            self.modules[m.name] = m
        self._structure_dirty = True

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
        if src.module.name not in self.modules or dst.module.name not in self.modules:
            raise ValueError("Both modules must be added to the graph before connecting.")
        if dst.kind != "input":
            raise ValueError("Destination of an edge must be an input port.")
        if src.kind not in ("output", "drive"):
            raise ValueError("Source must be an output port or an input-drive reference.")
        edge = Edge(name=name, src=src, dst=dst)
        self.edges.append(edge)
        self._structure_dirty = True
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
        if name is not None:
            for edge in self.edges:
                if edge.name == name:
                    return edge
            raise KeyError(f"No edge named {name!r}")
        if src is not None and dst is not None:
            for edge in self.edges:
                if edge.src is src and edge.dst is dst:
                    return edge
            raise KeyError("Edge with given src/dst not found.")
        raise ValueError("Provide either name or both src and dst to locate an edge.")

    # --- Binding & rollout ---

    def bind(self, batch: Dict[str, torch.Tensor]) -> None:
        """
        Infer Auto dimensions, build projections, and initialize module states.
        """
        if not self.modules:
            raise ValueError("Graph has no modules to bind.")
        self._current_batch = batch
        self._ensure_structure()
        self._bind_sources(batch)
        self._infer_port_dimensions(batch)
        self._materialize_edges()
        self._prepare_modules()

    def rollout(
        self,
        batch: Optional[Dict[str, torch.Tensor]] = None,
        max_ticks: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Execute the graph over a sequence of external ticks.
        """
        if batch is not None:
            self.bind(batch)
        if self._current_batch is None:
            raise RuntimeError("Graph has not been bound to a batch.")
        ticks = max_ticks or self._max_ticks
        edge_buffers: Dict[Edge, Optional[torch.Tensor]] = {}
        results: Dict[str, torch.Tensor] = {}

        for module in self.modules.values():
            device = self._module_devices[module.name]
            module._prepare_for_rollout(self._batch_size or 0, ticks, device)

        self.clock._tick = 0
        for tick in range(ticks):
            emitted_flags: Dict[str, bool] = {}
            for module in self.modules.values():
                device = self._module_devices[module.name]
                drive = self._collect_drive(module, edge_buffers, device, tick)
                for port_name, tensor in drive.items():
                    if isinstance(tensor, dict):
                        continue
                    module._record_input_drive(port_name, tensor)
                inner_steps = module.schedule.inner_steps
                for _ in range(inner_steps):
                    module._run_step(drive, tick)
                follow_name = self._follow_targets.get(module.name)
                follow_emitted = emitted_flags.get(follow_name) if follow_name else None
                if module.should_emit_at(tick, follow_emitted):
                    for port, tensor in module._pending_outputs.items():
                        module._record_output(port, tensor)
                    self._propagate_outputs(module, edge_buffers)
                    emitted_flags[module.name] = True
                else:
                    emitted_flags[module.name] = False
            self.clock.step()

        for module in self.modules.values():
            module._finalize_state(
                self._batch_size or 0,
                ticks,
                self._module_devices[module.name],
                self._port_dims,
            )
            for port, tensor in module.state.output.items():
                results[f"{module.name}.{port}"] = tensor
        return results

    def loss(self) -> torch.Tensor:
        """
        Compute total loss as the sum of all module-local objectives.
        """
        if self._current_batch is None:
            raise RuntimeError("Graph.loss() called before bind().")
        total: Optional[torch.Tensor] = None
        for module in self.modules.values():
            loss = module.objectives.compute(self._current_batch, self.modules)
            total = loss if total is None else total + loss
        if total is None:
            raise RuntimeError("No objectives registered across modules.")
        return total

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        for module in self.modules.values():
            yield from module.parameters()
        for edge in self.edges:
            if edge.proj is not None:
                yield from edge.proj.parameters()

    def named_parameters(self) -> Iterable[Tuple[str, torch.nn.Parameter]]:
        for name, module in self.modules.items():
            for param_name, param in module.named_parameters():
                yield f"{name}.{param_name}", param
        for edge in self.edges:
            if edge.proj is None:
                continue
            prefix = f"edge[{edge.name}]"
            for param_name, param in edge.proj.named_parameters():
                yield f"{prefix}.{param_name}", param

    # --- Internal helpers ----------------------------------------------------

    def _ensure_structure(self) -> None:
        if not self._structure_dirty:
            return
        self._incoming = defaultdict(list)
        self._outgoing = defaultdict(list)
        self._drive_outgoing = defaultdict(list)
        for edge in self.edges:
            dst_key = (edge.dst.module.name, edge.dst.name)
            self._incoming[dst_key].append(edge)
            if edge.src.kind == "drive":
                src_key = (edge.src.module.name, edge.src.name)
                self._drive_outgoing[src_key].append(edge)
            else:
                src_key = (edge.src.module.name, edge.src.name)
                self._outgoing[src_key].append(edge)
        self._structure_dirty = False

    def _bind_sources(self, batch: Dict[str, torch.Tensor]) -> None:
        self._batch_size = None
        self._max_ticks = 0
        for module in self.modules.values():
            if isinstance(module, Source):
                if module.batch_key not in batch:
                    raise KeyError(f"Batch missing key {module.batch_key!r} required by {module.name}.")
                tensor = batch[module.batch_key]
                module.bind_data(tensor)
                B, T, _ = tensor.shape
                if self._batch_size is None:
                    self._batch_size = B
                elif self._batch_size != B:
                    raise ValueError("All sources must share the same batch size.")
                self._max_ticks = max(self._max_ticks, T)
        if self._batch_size is None:
            raise ValueError("Graph requires at least one Source to determine batch size.")

    def _infer_port_dimensions(self, batch: Dict[str, torch.Tensor]) -> None:
        output_dims: Dict[Tuple[str, str], Optional[int]] = {}
        input_dims: Dict[Tuple[str, str], Optional[int]] = {}

        for module in self.modules.values():
            device = module._device()
            self._module_devices[module.name] = device
            out = module._output_dims.get("out")
            if isinstance(module, Source) and module._data is not None:
                out = module._data.shape[-1]
            elif hasattr(module.proto, "hidden"):
                out = getattr(module.proto, "hidden")
            elif hasattr(module.proto, "out_dim"):
                out = getattr(module.proto, "out_dim")
            output_dims[(module.name, "out")] = None if out is Auto else out
            for custom_port in module._custom_output_nodes:
                output_dims[(module.name, custom_port)] = None
            for port_name, spec in module._input_specs.items():
                if isinstance(spec, InPortGroup):
                    input_dims[(module.name, port_name)] = None
                    for sub_name, sub in spec.subports.items():
                        value = sub.size
                        input_dims[(module.name, f"{port_name}.{sub_name}")] = (
                            None if value is Auto else int(value)
                        )
                else:
                    value = spec.size
                    input_dims[(module.name, port_name)] = None if value is Auto else int(value)

        changed = True
        while changed:
            changed = False
            for (module_name, port_name), edges in self._incoming.items():
                port_spec = self.modules[module_name]._input_specs.get(port_name)
                if port_spec is None or isinstance(port_spec, InPortGroup):
                    continue
                key = (module_name, port_name)
                if input_dims[key] is not None:
                    continue
                src_dims = []
                unresolved = False
                for edge in edges:
                    dim = output_dims.get((edge.src.module.name, edge.src.name))
                    if dim is None and edge.src.kind == "drive":
                        dim = input_dims.get((edge.src.module.name, edge.src.name))
                    if dim is None:
                        unresolved = True
                        break
                    src_dims.append(dim)
                if unresolved or not src_dims:
                    continue
                if port_spec.combine == "sum":
                    if len(set(src_dims)) != 1:
                        raise ValueError(
                            f"Port {module_name}.{port_name} receives mismatched dims {src_dims}"
                        )
                    resolved = src_dims[0]
                elif port_spec.combine == "concat":
                    resolved = sum(src_dims)
                else:
                    raise ValueError(f"Unsupported combine mode {port_spec.combine}")
                input_dims[key] = resolved
                changed = True

            for edge in self.edges:
                if edge.src.kind == "drive":
                    continue
                src_key = (edge.src.module.name, edge.src.name)
                if output_dims.get(src_key) is not None:
                    continue
                dst_key = (edge.dst.module.name, edge.dst.name)
                dst_dim = input_dims.get(dst_key)
                if dst_dim is not None:
                    output_dims[src_key] = dst_dim
                    changed = True

        for module in self.modules.values():
            key = (module.name, "out")
            if output_dims.get(key) is None:
                inferred = self._infer_output_dim_from_targets(module, batch, input_dims, output_dims)
                if inferred is None:
                    raise ValueError(f"Unable to infer output dimension for module {module.name!r}.")
                output_dims[key] = inferred

        # Assign group port sizes based on subports.
        for (module_name, port_name), value in list(input_dims.items()):
            if value is not None:
                continue
            module = self.modules[module_name]
            spec = module._input_specs.get(port_name)
            if isinstance(spec, InPortGroup):
                sub_dims = [
                    input_dims[(module_name, f"{port_name}.{sub_name}")]
                    for sub_name in spec.subports
                ]
                if any(dim is None for dim in sub_dims):
                    raise ValueError(f"Unable to resolve grouped port size for {module_name}.{port_name}")
                total = 0
                for dim in sub_dims:
                    assert dim is not None
                    total += int(dim)
                input_dims[(module_name, port_name)] = total

        self._port_dims = {
            key: int(value)
            for key, value in {**input_dims, **output_dims}.items()
            if value is not None
        }

        for module in self.modules.values():
            for name, nodes in module._custom_output_nodes.items():
                dim = self._compute_nodes_width(module, nodes)
                self._port_dims[(module.name, name)] = dim

    def _infer_output_dim_from_targets(
        self,
        module: Module,
        batch: Dict[str, torch.Tensor],
        input_dims: Dict[Tuple[str, str], Optional[int]],
        output_dims: Dict[Tuple[str, str], Optional[int]],
    ) -> Optional[int]:
        dims: List[int] = []
        terms = getattr(module.objectives, "_terms", [])
        for term in terms:
            if term.get("kind") not in {"mse", "ce"}:
                continue
            if term.get("on") != "out":
                continue
            spec = term.get("target")
            if not spec:
                continue
            kind = spec[0]
            if kind == "batch_key":
                tensor = batch[spec[1]]
                if tensor.dim() >= 2:
                    dims.append(tensor.shape[-1])
                elif term["kind"] == "ce":
                    dims.append(int(tensor.max().item()) + 1)
            elif kind == "port_drive":
                target_module = self.modules[spec[1]]
                dim = input_dims.get((target_module.name, spec[2]))
                if dim is not None:
                    dims.append(dim)
            elif kind == "shifted_input":
                src_module = self.modules[spec[1]]
                dim = output_dims.get((src_module.name, "out"))
                if dim is not None:
                    dims.append(dim)
        if not dims:
            return None
        if len(set(dims)) != 1:
            raise ValueError(f"Conflicting target dims for module {module.name}: {dims}")
        return dims[0]

    def _compute_nodes_width(self, module: Module, nodes: Nodes) -> int:
        if nodes.indices is not None:
            return len(nodes.indices)
        if nodes.slice_ is not None:
            base_dim = self._base_dimension_for_nodes(module, nodes)
            start, stop, step = nodes.slice_.indices(base_dim)
            if step == 0:
                raise ValueError("Slice step cannot be zero.")
            length = max(0, math.ceil((stop - start) / step))
            return length
        return self._base_dimension_for_nodes(module, nodes)

    def _base_dimension_for_nodes(self, module: Module, nodes: Nodes) -> int:
        if nodes.layer is None:
            dim = self._port_dims.get((module.name, "out"))
            if dim is None:
                raise ValueError(f"Unable to determine base dimension for module {module.name} output.")
            return dim
        hidden = getattr(module.proto, "hidden", None)
        if hidden is None:
            raise ValueError(
                f"Module {module.name} defines layered nodes but proto does not expose 'hidden' attribute."
            )
        return int(hidden)

    def _materialize_edges(self) -> None:
        for edge in self.edges:
            src_dim = self._port_dims.get((edge.src.module.name, edge.src.name))
            dst_dim = self._port_dims.get((edge.dst.module.name, edge.dst.name))
            if src_dim is None or dst_dim is None:
                raise ValueError(f"Edge {edge.name} has unresolved dimensions.")
            device = self._module_devices[edge.dst.module.name]
            if edge.proj is None:
                edge.proj = MaskedLinear(src_dim, dst_dim).to(device)
            else:
                if edge.proj.in_features != src_dim or edge.proj.out_features != dst_dim:
                    raise ValueError(f"Edge {edge.name} projection shape mismatch.")
            if edge._mask is not None:
                edge.proj.set_mask(edge._mask.to(device))
            if edge._max_abs is not None:
                edge.proj.set_max_abs(edge._max_abs)
            dst_module = edge.dst.module
            mask = dst_module._input_masks.get(edge.dst.name)
            if mask is not None:
                edge.proj.set_mask(mask.to(device))
            max_abs = dst_module._input_max_abs.get(edge.dst.name)
            if max_abs is not None:
                edge.proj.set_max_abs(max_abs)

    def _prepare_modules(self) -> None:
        from . import protos  # Local import to avoid circular dependency

        for module in self.modules.values():
            in_dim = self._port_dims.get((module.name, "in"))
            out_dim = self._port_dims.get((module.name, "out"))
            bind_fn = getattr(module.proto, "bind", None)
            if callable(bind_fn):
                if isinstance(module.proto, protos.Aggregator):
                    group = module._input_groups.get("in")
                    if group is None:
                        raise ValueError(f"Aggregator module {module.name} requires an InPortGroup named 'in'.")
                    dims = {
                        sub_name: self._port_dims[(module.name, f"in.{sub_name}")]
                        for sub_name in group.subports
                    }
                    module.proto.bind(dims)
                elif isinstance(module.proto, protos.MLP):
                    if in_dim is None or out_dim is None:
                        raise ValueError(f"Module {module.name} requires resolved in/out dims.")
                    module.proto.bind(in_dim, out_dim)
                elif isinstance(module.proto, protos.GRUStack):
                    if in_dim is None:
                        raise ValueError(f"Module {module.name} requires resolved input dim.")
                    module.proto.bind(in_dim)
                else:
                    try:
                        if in_dim is not None and out_dim is not None:
                            module.proto.bind(in_dim, out_dim)
                        elif in_dim is not None:
                            module.proto.bind(in_dim)
                    except TypeError:
                        pass

            if module._recurrent_mask is not None and hasattr(module.proto, "parameters"):
                mask = module._recurrent_mask
                for name, param in module.proto.named_parameters():
                    if "recurrent" in name or "weight_hh" in name:
                        param.data.mul_(mask.to(param.device))
            if module._recurrent_max_abs is not None:
                limit = module._recurrent_max_abs
                for name, param in module.proto.named_parameters():
                    if "recurrent" in name or "weight_hh" in name:
                        param.data.clamp_(-limit, limit)

            follow = module.schedule.follow
            if isinstance(follow, Module):
                self._follow_targets[module.name] = follow.name
            elif isinstance(follow, str):
                self._follow_targets[module.name] = follow
            else:
                self._follow_targets[module.name] = None

    def _collect_drive(
        self,
        module: Module,
        edge_buffers: Dict[Edge, Optional[torch.Tensor]],
        device: torch.device,
        tick: int,
    ) -> Dict[str, Any]:
        drive: Dict[str, Any] = {}
        for port_name, spec in module._input_specs.items():
            if isinstance(spec, InPortGroup):
                continue
            edges = self._incoming.get((module.name, port_name), [])
            dim = self._port_dims.get((module.name, port_name))
            tensors: List[torch.Tensor] = []
            for edge in edges:
                value = edge_buffers.get(edge)
                if value is None:
                    value = torch.zeros(self._batch_size or 0, dim or 0, device=device)
                tensors.append(value)
            if not tensors and dim is not None:
                tensors.append(torch.zeros(self._batch_size or 0, dim, device=device))
            if isinstance(spec, InPort):
                if spec.combine == "sum":
                    combined = torch.stack(tensors, dim=0).sum(dim=0) if tensors else torch.zeros(self._batch_size or 0, dim or 0, device=device)
                elif spec.combine == "concat":
                    combined = torch.cat(tensors, dim=-1) if tensors else torch.zeros(self._batch_size or 0, dim or 0, device=device)
                else:
                    raise ValueError(f"Unsupported combine mode {spec.combine}")
                drive[port_name] = combined
            else:
                drive[port_name] = tensors[0] if tensors else torch.zeros(0, device=device)

        for group_name, group in module._input_groups.items():
            group_values = {}
            for sub_name in group.subports:
                key = f"{group_name}.{sub_name}"
                group_values[sub_name] = drive.get(
                    key,
                    torch.zeros(self._batch_size or 0, self._port_dims.get((module.name, key), 0), device=device),
                )
            drive[group_name] = group_values

        for port_name, tensor in drive.items():
            if isinstance(tensor, torch.Tensor):
                self._propagate_drive_edges(module, port_name, tensor, edge_buffers)

        return drive

    def _propagate_outputs(self, module: Module, edge_buffers: Dict[Edge, Optional[torch.Tensor]]) -> None:
        for port, tensor in module._pending_outputs.items():
            for edge in self._outgoing.get((module.name, port), []):
                projected = edge.proj(tensor) if edge.proj is not None else tensor
                edge_buffers[edge] = projected

    def _propagate_drive_edges(
        self,
        module: Module,
        port_name: str,
        tensor: torch.Tensor,
        edge_buffers: Dict[Edge, Optional[torch.Tensor]],
    ) -> None:
        for edge in self._drive_outgoing.get((module.name, port_name), []):
            projected = edge.proj(tensor) if edge.proj is not None else tensor
            edge_buffers[edge] = projected
