# imprint/objectives.py

from __future__ import annotations
from typing import Any, Dict
import torch
from .core import Module, TargetsSpec


class Targets:
    """
    Helper constructors for target specifications used by Objectives.

    Responsibilities:
      - Represent targets in a way that can be resolved at rollout time.
      - Provide common target types like batch_key, shifted_input, port_drive, etc.
    """

    @staticmethod
    def batch_key(key: str) -> TargetsSpec:
        """
        Target is batch[key], aligned with the module's output time dimension.
        """
        return ("batch_key", key)

    @staticmethod
    def shifted_input(source: "Module", shift: int) -> TargetsSpec:
        """
        Target is the source module's output (typically a Source), shifted in time.

        Example:
          shifted_input(src, +1) means predict x[t+1] at time t.
        """
        return ("shifted_input", source.name, shift)

    @staticmethod
    def port_drive(module: Module, port: str) -> TargetsSpec:
        """
        Target is module.state.input_drive[port].
        """
        return ("port_drive", module.name, port)


def resolve_target(
    spec: TargetsSpec,
    batch: Dict[str, torch.Tensor],
    modules: Dict[str, Module],
) -> torch.Tensor:
    """
    Internal: turn a TargetsSpec into a concrete tensor given
    the current batch and module states.
    """
    ...
