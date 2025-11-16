# `imprint` – Port-Graph RNN Composition (Usage Guide)

> **Status:** design skeleton, not a working library yet.  
> This document describes the intended *surface API* and usage patterns.

`imprint` is a small layer on top of PyTorch for building **composable, multi-rate, port-based RNN models**. It focuses on:

- **Modules**: self-contained computational units with internal state.
- **Ports**: named input/output interfaces backed by internal node groups.
- **Graphs**: collections of modules and edges with learnable projections.
- **Multi-rate scheduling**: modules can take many internal steps per external tick and emit at different rates.
- **Local objectives**: per-module losses on their own states or port drives.
- **Tracing & compilation**: record a rollout and compile it into a fast executable (e.g., CUDA graph).

This guide walks through the main concepts and shows how to build interesting RNN architectures using the current skeleton API.

---

## 1. Core Concepts

### 1.1 Graph & Clock

A `Graph` is the container for modules and their connectivity. A `Clock` provides a monotonic external “event” time.

```python
import torch
import imprint

clock = imprint.Clock()
g = imprint.Graph(clock=clock)
```

- `clock.tick` is an integer external “time”.
- `Graph.rollout` steps the clock, schedules modules, and runs them according to their `Rate`.

---

### 1.2 Modules, Protos, and Ports

A `Module` wraps an internal PyTorch `nn.Module` (“proto”) and exposes **ports**:

- **Input ports**: where incoming signals land (after projections).
- **Output ports**: views of internal neurons that send signals outward.
- **Port groups**: structured sets of subports (e.g., `deep`, `wide`), with layout constraints.

Simple example:

```python
Auto = imprint.Auto

enc = imprint.Module(
    name="enc",
    proto=imprint.protos.GRUStack(hidden=256, layers=2, layernorm=True),
    ports=imprint.Ports(
        in_default=Auto,   # infer based on connections / batch at bind()
        out_default=256,   # encoder exposes a 256-dim "out" port
    ),
)
```

- `ports.in_default` / `ports.out_default` define default `in` / `out` port sizes.
- `Auto` means “infer later at `g.bind(batch)` time”.

Modules expose ports via indexing:

```python
enc_in  = enc["in"]   # PortRef(enc, "in")
enc_out = enc["out"]  # PortRef(enc, "out")
```

You can also define **custom output ports** that pick specific internal nodes:

```python
enc.define_port(
    "out_deep",
    nodes=imprint.Nodes.of(layer=2, slice=slice(0, 192)),  # 192 units from deepest layer
)

enc.define_port(
    "out_wide",
    nodes=imprint.Nodes.of(layer=1, indices=range(0, 384, 2)),  # even indices in layer 1
)
```

These named ports can be wired independently.

---

### 1.3 Sources

A `Source` is a special `Module` that reads from the batch and emits events.

```python
src = imprint.Source(batch_key="x")  # expects batch["x"] as [B, T, D]
```

- It has a default `out` port.
- At rollout, it emits `batch["x"][:, t, :]` at each external tick `t`.

---

### 1.4 Rate (Multi-Rate Scheduling)

Each module has a `Rate` that controls how often it computes and emits:

```python
rate = imprint.Rate(inner_steps=4, emit_every=2)
```

- `inner_steps`: number of **internal proto steps** per external tick.
- `emit_every`: emit outputs every N external ticks.
- `Rate.follow("other_module_name")`: emit when another module emits.

Example:

```python
enc_hi = imprint.Module(
    name="enc_hi",
    proto=imprint.protos.GRUStack(hidden=192, layers=1),
    ports=imprint.Ports(in_default=Auto, out_default=192),
    schedule=imprint.Rate(inner_steps=1, emit_every=1),  # runs once per tick, emits every tick
)

enc_lo = imprint.Module(
    name="enc_lo",
    proto=imprint.protos.GRUStack(hidden=192, layers=2),
    ports=imprint.Ports(in_default=Auto, out_default=192),
    schedule=imprint.Rate(inner_steps=4, emit_every=2),  # 4 micro-steps per tick, emit every 2 ticks
)
```

Internally, the scheduler calls the module’s `step_once` method `inner_steps` times per external tick and only saves outputs when `should_emit_at` is true.

---

### 1.5 Edges & Projections

Connections between modules are **edges**, each with its own learnable projection:

```python
g.add(src, enc_hi)
g.connect(src["out"], enc_hi["in"])  # creates an Edge with its own nn.Linear
```

- The edge owns a projection `proj: src_dim → port_dim`.
- Multiple edges into the same port are combined (sum or concat) as defined by the port/group.

You can access and constrain an edge later:

```python
edge = g.connect(enc_hi["out"], enc_lo["in"], name="hi→lo")
edge.constrain(mask=imprint.masks.band(bandwidth=32), max_abs=0.3)
```

---

### 1.6 Per-Module Objectives & Targets

Each module has an `objectives` container where you can register local losses:

```python
enc.objectives.activity_l2(on="out", weight=1e-4)
```

Common patterns:

- `m.objectives.mse(on="out", target=Targets.batch_key("y"))`
- `m.objectives.ce(on="out", target=Targets.batch_key("labels"))`
- `m.objectives.mse(on="out", target=Targets.port_drive(enc, port="in"))`
- `m.objectives.activity_l1/on("out")`

`Targets` provide *declarative* target specifiers:

```python
from imprint import Targets

# direct from batch
Targets.batch_key("y")

# predict next input from a Source
Targets.shifted_input(src, shift=+1)

# reconstruct encoder input-drive
Targets.port_drive(enc, port="in")
```

The graph aggregates them:

```python
loss = g.loss()  # sum of all module-local objectives
```

---

### 1.7 Internal States: Input-Drive and Output

Each module tracks:

```python
m.state.input_drive[port_name]  # [B, T, D_port]
m.state.output[port_name]       # [B, T, D_port]
m.state.recurrent               # e.g., [layers, B, hidden]
```

These are convenient for:

- Analysis / logging.
- Local objectives.
- Wiring predictive-coding-like circuits (error units, etc.).

---

### 1.8 Tracing and Compilation

You can record **what actually happens** in a rollout (modules firing, port ops) and compile into a `Plan`:

```python
from imprint import record

with record(g) as trace:
    out = g.rollout(batch)

plan = trace.compile(
    backend="cuda_graph",
    fuse_projections=True,
    fuse_port_sums=True,
)

# Later:
out = plan.rollout(batch2)
loss = plan.loss()
loss.backward()
```

`Plan` mirrors `Graph`’s `rollout` and `loss`, but with a fixed schedule and more efficient execution.

---

## 2. A Simple End-to-End Example

Start with a straightforward chain: `Source → Encoder → MLP → Head`.

```python
import torch
import imprint

Auto = imprint.Auto

clock = imprint.Clock()
g = imprint.Graph(clock=clock)

# Source: reads batch["x"]
src = imprint.Source("x")

enc = imprint.Module(
    name="enc",
    proto=imprint.protos.GRUStack(hidden=256, layers=2, layernorm=True),
    ports=imprint.Ports(in_default=Auto, out_default=256),
)

mlp = imprint.Module(
    name="mlp",
    proto=imprint.protos.MLP(widths=[256, 512], act="gelu"),
    ports=imprint.Ports(in_default=256, out_default=512),
)

head = imprint.Module(
    name="head",
    proto=imprint.protos.MLP(widths=[512, Auto]),  # Auto out-dim (e.g. labels)
    ports=imprint.Ports(in_default=512, out_default=Auto),
)

g.add(src, enc, mlp, head)

g.connect(src["out"], enc["in"])
g.connect(enc["out"], mlp["in"])
g.connect(mlp["out"], head["in"])

# Local regularization and supervised objective
enc.objectives.activity_l2(on="out", weight=1e-4)
head.objectives.ce(on="out", target=imprint.Targets.batch_key("y"))

# Optional: constrain input weights (mask + magnitude)
enc.set_input_mask(port="in", mask=imprint.masks.keep_every_kth(k=2), persist=True)
enc.limit_weights(port="in", max_abs=0.25)

batch = {
    "x": torch.randn(32, 200, 128),    # [B, T, input_dim]
    "y": torch.randint(0, 1000, (32,)) # labels
}

# Infer Auto dims, allocate projections, etc.
g.bind(batch)

# Roll out and train
out = g.rollout(batch)
loss = g.loss()
loss.backward()
```

Key takeaways:

- You never manually create projections; `Graph.bind` handles that.
- Ports & dims with `Auto` get inferred from connections and batch shapes.
- All objectives are module-local and aggregated automatically.

---

## 3. Multi-Rate Encoders → Aggregator → Readout

Now build something more interesting:

- High-rate encoder (`enc_hi`) that emits every tick.
- Low-rate encoder (`enc_lo`) that runs multiple internal steps and emits less frequently.
- Aggregator with **disjoint input subports** (`in.deep`, `in.wide`).
- Readout that emits at the aggregator’s cadence.

```python
clock = imprint.Clock()
g = imprint.Graph(clock=clock)

src = imprint.Source("x")

enc_hi = imprint.Module(
    name="enc_hi",
    proto=imprint.protos.GRUStack(hidden=192, layers=1),
    ports=imprint.Ports(in_default=Auto, out_default=192),
    schedule=imprint.Rate(inner_steps=1, emit_every=1),  # basic RNN
)

enc_lo = imprint.Module(
    name="enc_lo",
    proto=imprint.protos.GRUStack(hidden=192, layers=2),
    ports=imprint.Ports(in_default=Auto, out_default=192),
    schedule=imprint.Rate(inner_steps=4, emit_every=2),  # 4 internal steps, emit every 2 ticks
)

agg = imprint.Module(
    name="agg",
    proto=imprint.protos.Aggregator(mode="concat→linear", out_dim=256),
    ports=imprint.Ports(
        in_=imprint.InPortGroup(
            deep=imprint.InPort(size=192),  # high-rate encoder
            wide=imprint.InPort(size=192),  # low-rate encoder
            layout="disjoint",          # subports occupy disjoint slices internally
        ),
        out_default=256,
    ),
    schedule=imprint.Rate(inner_steps=1, emit_every=4),  # emit every 4 external ticks
)

readout = imprint.Module(
    name="readout",
    proto=imprint.protos.MLP(widths=[256, Auto]),
    ports=imprint.Ports(in_default=256, out_default=Auto),
    schedule=imprint.Rate.follow("agg"),  # emit whenever agg emits
)

g.add(src, enc_hi, enc_lo, agg, readout)

g.connect(src["out"], enc_hi["in"])
g.connect(src["out"], enc_lo["in"])

# Fan-in via structured group ports; disjointness handled by InPortGroup
g.connect(enc_hi["out"], agg["in.deep"])
g.connect(enc_lo["out"], agg["in.wide"])

g.connect(agg["out"], readout["in"])

# Local objectives
agg.objectives.activity_l1(on="out", weight=2e-4)
readout.objectives.ce(on="out", target=imprint.Targets.batch_key("y"))

batch = {
    "x": torch.randn(32, 256, 80),          # fine-grained sequence
    "y": torch.randint(0, 500, (32, 64)),   # labels at aggregator cadence (64 emits over 256 ticks)
}

g.bind(batch)

# Optionally record for later compilation
from imprint import record

with record(g) as trace:
    out = g.rollout(batch)

loss = g.loss()
loss.backward()

plan = trace.compile(
    backend="cuda_graph",
    fuse_projections=True,
    fuse_port_sums=True,
)

# Training loop using the compiled plan
for batch in train_loader:
    out = plan.rollout(batch)
    loss = plan.loss()
    loss.backward()
    optimizer.step()
```

Important points:

- `InPortGroup(layout="disjoint")` handles “disjoint channels” semantics.
- Multi-rate behavior is driven purely by `Rate`s.
- Recording/compilation see the exact multi-rate schedule, so nothing special needs to be done for multi-rate graphs.

---

## 4. Micro-Stepping: Many Internal Steps per External Event

You don’t need to write custom wrappers for “micro-stepping”; the `Rate` handles it.

Example: encoder takes 8 internal steps per external tick and predicts the next input.

```python
clock = imprint.Clock()
g = imprint.Graph(clock=clock)

src = imprint.Source("x")

enc_slow = imprint.Module(
    name="enc_slow",
    proto=imprint.protos.GRUStack(hidden=320, layers=2, layernorm=True),
    ports=imprint.Ports(in_default=Auto, out_default=320),
    schedule=imprint.Rate(inner_steps=8, emit_every=1),  # 8 internal steps per event
)

head = imprint.Module(
    name="head",
    proto=imprint.protos.MLP(widths=[320, Auto]),
    ports=imprint.Ports(in_default=320, out_default=Auto),
    schedule=imprint.Rate(inner_steps=1, emit_every=1),
)

g.add(src, enc_slow, head)
g.connect(src["out"], enc_slow["in"])
g.connect(enc_slow["out"], head["in"])

# Self-supervised next-step prediction
head.objectives.mse(
    on="out",
    target=imprint.Targets.shifted_input(src, shift=+1),
)

batch = {"x": torch.randn(16, 128, 64)}
g.bind(batch)
g.rollout(batch)
loss = g.loss()
loss.backward()

# Inspect internal state:
drive_t = enc_slow.state.input_drive["in"]  # [B, T, in_dim]
feat_t  = enc_slow.state.output["out"]      # [B, T, 320] (after 8 micro-steps per tick)
```

All the “loop K times” logic is hidden; you just specify `inner_steps=8`.

Want a runnable version? `python examples/demo_micro_stepping.py` wires this graph to
`imprint.data_helper.load_micro_step_demo_dataset`, so batch shapes, sequence length,
and head dimensions are inferred directly from a DATASETS.md-style source.

---

## 5. Predictive-Coding-Like Micro-Circuit

Now build a “predictive coding” motif:

- Encoder `enc` processes inputs.
- Predictor `pred` tries to reconstruct `enc`’s **input-drive**.
- Error unit `err` computes absolute difference and optionally drives a controller `ctrl`.

```python
clock = imprint.Clock()
g = imprint.Graph(clock=clock)

src = imprint.Source("x")

enc = imprint.Module(
    name="enc",
    proto=imprint.protos.GRUStack(hidden=192, layers=1),
    ports=imprint.Ports(in_default=Auto, out_default=192),
    schedule=imprint.Rate(inner_steps=1, emit_every=1),
)

pred = imprint.Module(
    name="pred",
    proto=imprint.protos.MLP(widths=[192, Auto]),  # Auto → enc.in dim
    ports=imprint.Ports(in_default=192, out_default=Auto),
)

err = imprint.Module(
    name="err",
    proto=imprint.protos.Elementwise("sub→abs"),   # |a - b|
    ports=imprint.Ports(
        in_=imprint.InPortGroup(
            actual=imprint.InPort(size=Auto),
            predicted=imprint.InPort(size=Auto),
            layout="disjoint",  # conceptually separate streams
        ),
        out_default=Auto,
    ),
)

g.add(src, enc, pred, err)

g.connect(src["out"], enc["in"])
g.connect(enc["out"], pred["in"])

# Wire actual input-drive vs. predicted input-drive into error unit
g.connect(enc.port_ref("in").drive, err["in.actual"])  # the encoder's input-drive
g.connect(pred["out"],              err["in.predicted"])

# Local objective: predictor reconstructs enc's input-drive
pred.objectives.mse(
    on="out",
    target=imprint.Targets.port_drive(enc, port="in"),
    weight=1.0,
)

# Optional: use error as input to a controller
ctrl = imprint.Module(
    name="ctrl",
    proto=imprint.protos.GRUStack(hidden=128, layers=1),
    ports=imprint.Ports(in_default=Auto, out_default=128),
)
g.add(ctrl)
g.connect(err["out"], ctrl["in"])

batch = {"x": torch.randn(8, 300, 48)}
g.bind(batch)
g.rollout(batch)
loss = g.loss()
loss.backward()
```

Key features:

- `enc.port_ref("in").drive` gives a handle to the **port’s input-drive** tensor.
- `Targets.port_drive(enc, "in")` can be used as a declarative target.
- All objectives are local to their modules; you can build rich error-driven patterns without a single monolithic loss.

---

## 6. Multiple Output Ports + Structured Aggregator

Combine:

- A module that exposes several **named output ports** bound to different node groups.
- An aggregator with corresponding **disjoint input subports**.

```python
clock = imprint.Clock()
g = imprint.Graph(clock=clock)

src = imprint.Source("x")

enc = imprint.Module(
    name="enc",
    proto=imprint.protos.GRUStack(hidden=384, layers=3, layernorm=True),
    ports=imprint.Ports(in_default=Auto, out_default=256),  # default "out"
)

# Define specialized output ports
enc.define_port(
    "out_deep",
    nodes=imprint.Nodes.of(layer=2, slice=slice(0, 192)),   # deep layer
)
enc.define_port(
    "out_wide",
    nodes=imprint.Nodes.of(layer=1, indices=range(0, 384, 2)),  # mid layer, even indices
)

agg = imprint.Module(
    name="agg",
    proto=imprint.protos.Aggregator(mode="sum→linear", out_dim=256),
    ports=imprint.Ports(
        in_=imprint.InPortGroup(
            deep=imprint.InPort(size=192),
            wide=imprint.InPort(size=192),
            layout="disjoint",
        ),
        out_default=256,
    ),
)

head = imprint.Module(
    name="head",
    proto=imprint.protos.MLP(widths=[256, Auto]),
    ports=imprint.Ports(in_default=256, out_default=Auto),
)

g.add(src, enc, agg, head)

g.connect(src["out"], enc["in"])
g.connect(enc["out_deep"], agg["in.deep"])
g.connect(enc["out_wide"], agg["in.wide"])
g.connect(agg["out"], head["in"])

batch = {
    "x": torch.randn(12, 100, 96),
    "y": torch.randint(0, 50, (12,)),
}
g.bind(batch)
out = g.rollout(batch)
loss = g.loss()
loss.backward()
```

If you want special sparsity patterns on a particular mapping, you can still use masks at the *edge* level:

```python
edge_deep = g.connect(enc["out_deep"], agg["in.deep"])
edge_deep.constrain(
    mask=imprint.masks.band(bandwidth=32),
    max_abs=0.4,
)
```

But for standard “disjoint contributions per source,” the port group layout suffices.

---

## 7. Putting It All Together

The skeleton API supports:

- **Composable graphs of modules** with explicit connectivity.
- **Port-level semantics** for node selection and structured fan-in (disjoint vs shared layouts).
- **Multi-rate behavior** via `Rate(inner_steps, emit_every, follow)`.
- **Internal states**: module `state.input_drive` and `state.output` per port.
- **Local objectives** and declarative `Targets` for self-supervision, predictive coding, etc.
- **Masking & magnitude limits** on edge projections and module recurrent weights (for structured sparsity / constraints).
- **Trace → compile** to amortize Python overhead after discovering an actual multi-rate schedule.