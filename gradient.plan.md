# Gradient Debug Visualization Plan

1. **Define Requirements & Hooks**  
   - Enumerated gradient metrics (per-module L2/max/mean, zero %, stop-grad edges) and hook points in `imprint/core.py` (module parameters, edge projections, proto outputs).

2. **Implement Gradient Capture Utility**  
   - Added `imprint/diagnostics.py` with `GradientWatcher`, `GradientSummary`, and tensor stat helpers.

3. **Visualization / Reporting Layer**  
   - Textual summaries via `GradientSummary.to_text` + optional matplotlib heatmap helper.

4. **Demo Integration & Config Knob**  
   - `examples/demo_predictive_coding.py` gains `CONFIG["log_gradients"]` and wires the watcher into `train_graph`.

5. **Documentation**  
   - README section explains how to enable gradient logging and plotting.

### To-dos

- [x] List gradient metrics and hook points in core graph
- [x] Implement capture/reset helpers in new diagnostics module
- [x] Add textual/plot summaries + trainer logging hooks
- [x] Expose config flag and wire diagnostics in predictive coding demo
- [x] Document gradient debugging workflow

