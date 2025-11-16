"""
Demo script #4 from USAGE.md: micro-stepping encoder + next-step head.

Data handling is intentionally minimal – we call the helper from DATASETS.md
inspired code to fetch (or synthesize) a `[N, T, D]` sequence dataset and let
its metadata drive module dimensions.
"""

from __future__ import annotations

import argparse
from typing import Iterable

import torch

import imprint
from imprint import SequenceDataset, load_micro_step_demo_dataset

Auto = imprint.Auto


def build_graph(hidden_size: int, target_dim: int) -> imprint.Graph:
    """
    Build the micro-stepping graph described in §4 of USAGE.md.
    """
    clock = imprint.Clock()
    graph = imprint.Graph(clock=clock)

    src = imprint.Source("x")
    enc_slow = imprint.Module(
        name="enc_slow",
        proto=imprint.protos.GRUStack(hidden=hidden_size, layers=2, layernorm=True),
        ports=imprint.Ports(in_default=Auto, out_default=hidden_size),
        schedule=imprint.Rate(inner_steps=8, emit_every=1),
    )
    head = imprint.Module(
        name="head",
        proto=imprint.protos.MLP(widths=[hidden_size, Auto]),
        ports=imprint.Ports(in_default=hidden_size, out_default=target_dim),
        schedule=imprint.Rate(inner_steps=1, emit_every=1),
    )

    graph.add(src, enc_slow, head)
    graph.connect(src["out"], enc_slow["in"])
    graph.connect(enc_slow["out"], head["in"])

    head.objectives.mse(
        on="out",
        target=imprint.Targets.shifted_input(src, shift=+1),
        weight=1.0,
    )
    return graph


def graph_parameters(graph: imprint.Graph) -> Iterable[torch.nn.Parameter]:
    for module in graph.modules.values():
        for param in module.parameters():
            yield param


def run_epoch(
    graph: imprint.Graph,
    dataset: SequenceDataset,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    log_every: int,
) -> float:
    total_loss = 0.0
    for step, batch in enumerate(dataset.iter_batches(shuffle=True), start=1):
        optimizer.zero_grad()
        graph.rollout(batch)
        loss = graph.loss()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if step % log_every == 0 or step == dataset.batches_per_epoch:
            print(
                f"[epoch {epoch}] step {step}/{dataset.batches_per_epoch} "
                f"loss={loss.item():.4f}"
            )

    return total_loss / dataset.batches_per_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Micro-stepping demo (USAGE.md §4).",
    )
    parser.add_argument("--data-path", type=str, default=None, help="Optional HDF5 dataset path.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    dataset = load_micro_step_demo_dataset(
        path=args.data_path,
        split=args.split,
        batch_size=args.batch_size,
    )

    print("Loaded dataset:", dataset.summary())
    print(
        f"Metadata → feature_dim={dataset.feature_dim}, "
        f"target_dim={dataset.target_dim}, batches/epoch={dataset.batches_per_epoch}"
    )

    graph = build_graph(hidden_size=args.hidden_size, target_dim=dataset.target_dim)
    params = list(graph_parameters(graph))
    if not params:
        raise RuntimeError("Graph has no trainable parameters.")
    optimizer = torch.optim.Adam(params, lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        avg_loss = run_epoch(
            graph,
            dataset,
            optimizer,
            epoch=epoch,
            log_every=args.log_every,
        )
        print(f"Epoch {epoch} average loss: {avg_loss:.4f}")

    print("Done.")


if __name__ == "__main__":
    main()
