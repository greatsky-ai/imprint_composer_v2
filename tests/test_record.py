import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import imprint  # noqa: E402


def _build_simple_graph(output_dim: int = 4) -> imprint.Graph:
    g = imprint.Graph()
    src = imprint.Source("x")
    enc = imprint.Module(
        name="enc",
        proto=imprint.protos.GRUStack(hidden=16, layers=1),
        ports=imprint.Ports(in_default=imprint.Auto, out_default=16),
    )
    head = imprint.Module(
        name="head",
        proto=imprint.protos.MLP(widths=[16, output_dim]),
        ports=imprint.Ports(in_default=16, out_default=output_dim),
    )
    g.add(src, enc, head)
    g.connect(src["out"], enc["in"])
    g.connect(enc["out"], head["in"])
    head.objectives.mse(on="out", target=imprint.Targets.batch_key("y"))
    return g


def test_record_trace_creates_plan_and_summary():
    torch.manual_seed(5)
    graph = _build_simple_graph()
    batch = {
        "x": torch.randn(3, 6, 8),
        "y": torch.randn(3, 6, 4),
    }

    with imprint.record(graph) as trace:
        outputs = graph.rollout(batch)

    summary = trace.summary()
    assert summary["events"] > 0
    assert summary["ticks"] == 6
    assert "head" in summary["modules"]
    assert trace.tick_count == 6
    assert len(trace.events) == summary["events"]
    assert "head.out" in outputs

    plan = trace.compile()
    plan_desc = plan.describe()
    assert plan_desc["tick_count"] == 6
    plan_out = plan.rollout(batch)

    torch.testing.assert_close(plan_out["head.out"], graph.rollout(batch)["head.out"])
    plan_loss = plan.loss()
    assert torch.is_tensor(plan_loss)
