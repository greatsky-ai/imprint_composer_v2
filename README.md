# Imprint (Composer v2)

Imprint is a minimal, shapes‑driven toolkit for building and training modular sequence models via a declarative graph. It aims to keep demos and experiments focused on the unique parts (graph topology, objectives, and a handful of hyperparameters) while pushing boilerplate into small helpers.


## Philosophy

- Graph‑first: declare modules and edges; let the system infer shapes and build projections.
- Lazy binding: parameters and projections are materialized only once the first batch is seen.
- Scheduling: micro‑stepping, emission cadence, and follow relationships are first‑class.
- Objectives as data: per‑module objectives reference batch keys, port drives, or shifted inputs.
- Minimal boilerplate: helpers for training loops, metrics, validation cadence, and common recipes.
- Extensible prototypes: simple `bind()` and optional `step()` APIs to plug in new compute blocks.


## Features

- Graph, Clock, Module, Ports, Edges
- `Rate(inner_steps, emit_every, follow, step_when_emits)` enables micro‑stepping encoders, controlled
  emissions, and lets downstream modules step only when their source emits (slicer/context patterns).
  - `Source("x")` injects external sequences (e.g., `batch["x"]`) as a module in the graph.
  - Auto dimension inference for ports/edges; named `Nodes` views for custom output ports.
- Projections with constraints
  - Edges are backed by `MaskedLinear` and support persistent masks and per‑edge max‑abs clamps.
  - Modules can also apply input masks/limits per port.
- Gradient flow control
  - Edge‑level stop‑gradient lets you block gradients from flowing back into a source module while still training the edge projection and downstream modules (ideal for auxiliary heads that should not steer backbone encoders).
- Prototypes (protos)
  - `GRUStack` with `bind(input_dim)` and single‑step `step(drive, state)`; optional layernorm.
  - `MLP` with Auto final width (derived from objectives or outputs).
  - `Aggregator` for grouped inputs (`concat` or `sum` then `Linear`).
  - `Elementwise` for simple tensor ops (e.g., `sub→abs`).
- Objectives and targets
  - Losses: `mse`, `ce`, `activity_l1`, `activity_l2`, plus user‑supplied callables.
  - Targets: `Targets.batch_key("y")`, `Targets.shifted_input(src, shift)`, `Targets.port_drive(module, "in")`.
  - Parameter regularization (tag‑based): `params_l1(tag)`, `params_l2(tag)` with tags:
    - `proto_params`: all proto parameters
    - `recurrent_params`: common recurrent matrices (e.g., GRU/LSTM `weight_hh`)
    - `all_params`: same as `proto_params` (module‑level)
- Data helpers
  - `SequenceDataset` (`[N, T, D]`) with metadata and batch iteration.
  - `load_demo_dataset(path=None, split="train", ...)` for synthetic or HDF5 data.
- Training helpers
  - `train_graph(graph, dataset, ...)` with:
    - `epochs`, `lr`, `log_every`, `seed`
    - `loss_fn`, `metric_fn`
    - `grad_clip`, `use_adamw`, `weight_decay`, `betas`
    - `val_dataset`, `val_every` and optional `val_loss_fn`, `val_metric_fn` (default to training fns)
- Recipes
  - `prepare_seq2static_classification(graph, dataset, head_name="head", label_key="y", emit_once=False)`
  - `last_step_ce_loss(...)`, `last_step_accuracy(...)`
  - `infer_num_classes(dataset, override=None)`


## Install / Run

This repository is intended for local development and experimentation. A typical workflow:

```bash
git clone https://github.com/your-org/imprint_composer_v2.git
cd imprint_composer_v2
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # if you maintain one; otherwise ensure torch is available
python examples/demo_micro_stepping.py
```


## Core API (bird’s‑eye)

```python
import imprint

# Graph and scheduling
clock = imprint.Clock()
graph = imprint.Graph(clock=clock)
src = imprint.Source("x")
enc = imprint.Module(
    name="enc",
    proto=imprint.protos.GRUStack(hidden=128, layers=2, layernorm=True),
    ports=imprint.Ports(in_default=imprint.Auto, out_default=128),
    schedule=imprint.Rate(inner_steps=8, emit_every=1),
)
head = imprint.Module(
    name="head",
    proto=imprint.protos.MLP(widths=[128, imprint.Auto]),
    ports=imprint.Ports(in_default=128, out_default=imprint.Auto),
)

graph.add(src, enc, head)
graph.connect(src["out"], enc["in"])
graph.connect(enc["out"], head["in"])

# Objective: seq2static classification on the final timestep
imprint.prepare_seq2static_classification(graph, dataset, head_name="head", label_key="y")

# Optional param regularization
enc.objectives.params_l2("recurrent_params", weight=5e-5)

# Train (with classification loss/accuracy computed on final timestep)
imprint.train_graph(
    graph,
    dataset,
    epochs=5,
    lr=3e-3,
    log_every=10,
    seed=7,
    loss_fn=imprint.last_step_ce_loss(head_name="head", label_key="y"),
    metric_fn=imprint.last_step_accuracy(head_name="head", label_key="y"),
    val_dataset=val_dataset,
    val_every=1,
)
```


## Usage patterns

- Seq2Seq next‑step prediction
  - Use `Targets.shifted_input(src, +1)` with an MSE head that emits every tick.
- Seq2Static classification
  - Keep the head emitting each tick but compute CE on the final timestep only (`last_step_ce_loss`).
  - Alternatively, set the head’s `emit_every = dataset.seq_len` to emit once per sequence (be mindful of how early ticks might leak into an “easy” first‑frame prediction for your dataset).
- Micro‑stepping encoders
  - Increase `Rate.inner_steps` on an encoder to perform multiple internal updates per external tick.
- Grouped inputs and aggregation
  - Use `InPortGroup` and `protos.Aggregator` to combine multiple streams via `concat` or `sum`.

## Gradient flow control (stop‑grad on edges)

Imprint allows isolating gradient flow at the edge level. This is critical when attaching auxiliary/task heads to a backbone: you often want the aux head and its projections to learn from the task loss without steering the upstream feature extractors that are trained under different objectives.

- What it does: Setting stop‑grad on an edge detaches the source activation before projection. Gradients still flow
  - into the edge’s projection weights, and
  - into all downstream modules (e.g., aux GRUs, MLP heads),
  but do not flow back into the source module that produced the activation.
- Why it matters: Prevents task objectives from inadvertently updating backbone modules (e.g., predictive‑coding layers trained via reconstruction/prediction) and reduces gradient interference in multi‑objective setups.

Usage:

```python
# Prevent task loss from updating the upstream encoder while still training the
# edge projection and downstream auxiliary module / head.
edge = graph.connect(encoder["out"], aux["in"])
edge.set_stop_grad(True)
```

Notes:
- Works for both standard output edges and input‑drive edges; the detach is applied before any projection.
- Source modules remain trainable by their own objectives; only the task loss routed through the stop‑grad edge is blocked from flowing upstream.
- See `examples/demo_predictive_coding.py`: the aux GRU consumes PC GRU latents via stop‑grad edges so the task head trains the aux path while PC layers train on their reconstruction/predictor losses.

## Gradient diagnostics

Understanding whether gradients are flowing (or being blocked) is often more informative than raw losses. The `imprint.diagnostics` helpers provide a lightweight way to inspect per-module, per-edge, and per-port gradients every few logging steps.

- `imprint.GradientWatcher(graph, track_ports=True, track_edges=True)` registers parameter/activation hooks and accumulates L2 norms, max magnitudes, and zero fractions.
- Pass the watcher to `imprint.train_graph(..., grad_monitor=watcher)` to print formatted gradient summaries alongside the existing loss breakdown. The predictive-coding demo exposes a `CONFIG["log_gradients"]` knob that enables this automatically.
- Call `summary = watcher.pop_summary(top_k=5)` manually to inspect the latest stats, or `watcher.close()` to remove hooks.
- For richer visualization, feed a `GradientSummary` into `imprint.plot_gradient_heatmap(summary, section="modules")` (matplotlib optional) to view hot/cold modules at a glance.

Example:

```python
graph = build_graph(dataset)
grad_monitor = imprint.GradientWatcher(graph, track_ports=True, track_edges=True)
imprint.train_graph(
    graph,
    dataset,
    epochs=cfg["epochs"],
    lr=cfg["lr"],
    grad_monitor=grad_monitor,
)
grad_monitor.close()
```

Use the summaries to quickly confirm that:
- stop-grad edges are doing their job (port gradients drop to zero),
- predictive layers confine their updates when `confine_pc_gradients=True`,
- downstream heads are not starving (non-zero gradients on their parameters).


## Objectives: targets and regularization

- Targets
  - `Targets.batch_key("y")`: supervised labels or continuous targets from the batch.
  - `Targets.shifted_input(src, +1)`: self‑supervised next‑step targets from a `Source`'s output.
  - `Targets.port_drive(module, "in")`: reference a module’s input drive tensor.
- Activity regularizers
  - `activity_l1("out", w)` and `activity_l2("out", w)` operate on module outputs.
- Parameter regularizers (tag‑based; no fragile name matching)
  - `params_l1("proto_params", w)`, `params_l2("recurrent_params", w)`
  - Module exposes `iter_parameters_by_tag(tag)` with tags:
    - `proto_params`, `recurrent_params`, `all_params`


## Prototypes (bind/step model)

- `GRUStack(hidden, layers=1, layernorm=False, reset_every=None)`
  - `bind(input_dim)` allocates cells; `init_state(batch_size, device)` allocates `[layers, B, hidden]`.
  - Optional `reset_every` zeros the hidden state on a fixed cadence (e.g., per chunk for slicer encoders).
  - `step(drive, state)` returns `(new_state, [layer_outputs...])` with final output used as module `out`.
- `MLP(widths=[..., Auto], act="relu")`
  - `bind(input_dim, output_dim)` constructs linear stack; final width can be `Auto` and inferred.
- `Aggregator(mode="concat→linear", out_dim)`
  - `bind(input_dims: Dict[str, int])` sizes internal linear layer post‑concat/sum.
- `Elementwise(op)` supports basic ops (e.g., `sub→abs`, `add`, `mul`).


## Training helper

`imprint.train_graph` provides a minimal, pragmatic loop with optional validation:

- Construct parameters on first rollout (lazy bind).
- Create optimizer lazily (`Adam` or `AdamW`).
- Options: `grad_clip`, `weight_decay`, `betas`, `seed`.
- Metrics: pass `metric_fn` and (optionally) `val_metric_fn`; validation defaults to training fns.


## Data conventions

- `SequenceDataset` wraps `torch.Tensor` data `[N, T, D]` and optional labels, and exposes:
  - `feature_dim`, `target_dim`, `num_classes`, `batches_per_epoch`, `summary()`.
  - `iter_batches(shuffle=True, device=None)` → yields `{"x": ..., "y": ...}` when labels present.
- HDF5 datasets
  - See `DATASETS.md` for layout. If an HDF5 path isn’t provided, the demo synthesizes a tiny set.
- `load_demo_dataset` accepts a config/kwargs bundle (path, batch size, synthetic
  sequence length, feature dimension, seed, etc.) so every example can share the same helper
  without bespoke data-loading code.


## Extending

- Implement a prototype with `bind()` and optionally `step()` + `init_state()` to participate in the scheduler.
- Add custom objectives via `Objectives.add(fn, weight)` or extend `Objectives` if widely useful.
- For richer param tagging, your proto may expose a tiny registry (optional):
  - e.g., keep references to recurrent matrices to avoid name heuristics; Imprint will use them when present.


## Examples

- `examples/demo_micro_stepping.py`
  - Micro‑stepping encoder with a next-step head wired up to the generalized data helper.
- `examples/demo_slicer_encoder.py`
  - Two-stage slicer/context GRU graph that chunks sequences via emission cadence and feeds a seq2static head.


## Tests

Run the included tests to verify the graph, schedules and projections:

```bash
pytest -q
```


## Roadmap

- Additional recipes (contrastive, masked modeling)
- Richer state‑level regularization (tagged states with cheap accumulators)
- More protos (CNNs, Transformers with step APIs)


