import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import imprint as imprint


Auto = imprint.Auto


def _has_grads(graph: imprint.Graph) -> bool:
    return any(param.grad is not None for param in graph.parameters())


def test_simple_chain_rollout_backward():
    torch.manual_seed(0)
    g = imprint.Graph()
    src = imprint.Source("x")
    enc = imprint.Module(
        name="enc",
        proto=imprint.protos.GRUStack(hidden=32, layers=2, layernorm=True),
        ports=imprint.Ports(in_default=Auto, out_default=32),
    )
    mlp = imprint.Module(
        name="mlp",
        proto=imprint.protos.MLP(widths=[32, 64], act="gelu"),
        ports=imprint.Ports(in_default=32, out_default=64),
    )
    head = imprint.Module(
        name="head",
        proto=imprint.protos.MLP(widths=[64, Auto]),
        ports=imprint.Ports(in_default=64, out_default=Auto),
    )

    g.add(src, enc, mlp, head)
    g.connect(src["out"], enc["in"])
    g.connect(enc["out"], mlp["in"])
    g.connect(mlp["out"], head["in"])

    enc.objectives.activity_l2(on="out", weight=1e-3)
    head.objectives.ce(on="out", target=imprint.Targets.batch_key("y"))

    batch = {
        "x": torch.randn(8, 20, 16),
        "y": torch.randint(0, 10, (8,)),
    }

    outputs = g.rollout(batch)
    assert outputs["head.out"].shape == (8, 20, 10)
    loss = g.loss()
    loss.backward()
    assert _has_grads(g)


def test_multi_rate_follow_schedule():
    torch.manual_seed(1)
    g = imprint.Graph()
    src = imprint.Source("x")
    enc_hi = imprint.Module(
        name="enc_hi",
        proto=imprint.protos.GRUStack(hidden=24, layers=1),
        ports=imprint.Ports(in_default=Auto, out_default=24),
        schedule=imprint.Rate(inner_steps=1, emit_every=1),
    )
    enc_lo = imprint.Module(
        name="enc_lo",
        proto=imprint.protos.GRUStack(hidden=24, layers=2),
        ports=imprint.Ports(in_default=Auto, out_default=24),
        schedule=imprint.Rate(inner_steps=2, emit_every=2),
    )
    agg = imprint.Module(
        name="agg",
        proto=imprint.protos.Aggregator(mode="concat→linear", out_dim=48),
        ports=imprint.Ports(
            in_=imprint.InPortGroup(
                deep=imprint.InPort(size=24),
                wide=imprint.InPort(size=24),
            ),
            out_default=48,
        ),
        schedule=imprint.Rate(inner_steps=1, emit_every=4),
    )
    readout = imprint.Module(
        name="readout",
        proto=imprint.protos.MLP(widths=[48, Auto]),
        ports=imprint.Ports(in_default=48, out_default=Auto),
        schedule=imprint.Rate.follow("agg"),
    )

    g.add(src, enc_hi, enc_lo, agg, readout)
    g.connect(src["out"], enc_hi["in"])
    g.connect(src["out"], enc_lo["in"])
    g.connect(enc_hi["out"], agg["in.deep"])
    g.connect(enc_lo["out"], agg["in.wide"])
    g.connect(agg["out"], readout["in"])

    readout.objectives.mse(
        on="out",
        target=imprint.Targets.batch_key("y"),
        weight=0.5,
    )

    batch = {
        "x": torch.randn(4, 40, 12),
        "y": torch.randn(4, 10, 16),  # 40 ticks with emit_every=4 -> 10 outputs
    }

    outputs = g.rollout(batch)
    assert outputs["agg.out"].shape[1] == 10
    assert outputs["readout.out"].shape == (4, 10, 16)
    loss = g.loss()
    loss.backward()
    assert _has_grads(g)


def test_port_drive_connections_and_targets():
    torch.manual_seed(2)
    g = imprint.Graph()
    src = imprint.Source("x")
    enc = imprint.Module(
        name="enc",
        proto=imprint.protos.GRUStack(hidden=16, layers=1),
        ports=imprint.Ports(in_default=Auto, out_default=16),
    )
    pred = imprint.Module(
        name="pred",
        proto=imprint.protos.MLP(widths=[16, 16, 10]),
        ports=imprint.Ports(in_default=16, out_default=10),
    )
    err = imprint.Module(
        name="err",
        proto=imprint.protos.Elementwise("sub→abs"),
        ports=imprint.Ports(
            in_=imprint.InPortGroup(
                actual=imprint.InPort(size=10),
                predicted=imprint.InPort(size=10),
            ),
            out_default=16,
        ),
    )

    g.add(src, enc, pred, err)
    g.connect(src["out"], enc["in"])
    g.connect(enc["out"], pred["in"])
    g.connect(enc["out"], err["in.predicted"])
    g.connect(enc.port_ref("in").drive, err["in.actual"])

    pred.objectives.mse(
        on="out",
        target=imprint.Targets.port_drive(enc, "in"),
        weight=1.0,
    )

    batch = {"x": torch.randn(3, 15, 10)}

    outputs = g.rollout(batch)
    assert "err.out" in outputs
    assert outputs["err.out"].shape[1] == 15
    loss = g.loss()
    loss.backward()
    assert _has_grads(g)


def test_edge_and_module_masks_apply_constraints():
    torch.manual_seed(3)
    g = imprint.Graph()
    src = imprint.Source("x")
    linear = imprint.Module(
        name="linear",
        proto=imprint.protos.MLP(widths=[8, 12]),
        ports=imprint.Ports(in_default=Auto, out_default=12),
    )
    g.add(src, linear)
    edge = g.connect(src["out"], linear["in"])

    keep_mask = imprint.masks.keep_every_kth(2, size=8)
    linear.set_input_mask("in", keep_mask)
    edge_mask = imprint.masks.eye(8, 8)
    edge.constrain(mask=edge_mask)

    batch = {"x": torch.randn(2, 5, 8)}
    g.rollout(batch)

    assert edge.proj is not None
    expanded_keep = keep_mask.unsqueeze(0).expand_as(edge.proj.weight)
    sample = torch.randn(3, 8)
    with torch.no_grad():
        masked_weight = edge.proj.weight * expanded_keep
        expected = F.linear(sample, masked_weight, edge.proj.bias)
        actual = edge.proj(sample)
    assert torch.allclose(actual, expected)


def _run_all():
    tests = [
        test_simple_chain_rollout_backward,
        test_multi_rate_follow_schedule,
        test_port_drive_connections_and_targets,
        test_edge_and_module_masks_apply_constraints,
    ]
    for test in tests:
        name = test.__name__
        print(f"Running {name}...")
        test()
        print(f"✓ {name}")
    print("All tests passed ✔")


if __name__ == "__main__":
    _run_all()
