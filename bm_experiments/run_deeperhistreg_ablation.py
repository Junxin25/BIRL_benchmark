"""
Run DeeperHistReg with benchmark-only ablation patches.

This wrapper keeps the upstream DeeperHistReg files untouched. It is intended
for BIRL ablation runs where selected initial-alignment functions need a small
runtime compatibility shim.
"""

import importlib.util
import os
import sys

import torch as tc


DHR_ROOT = "/home/junxinfu/registration/Model/DeeperHistReg/deeperhistreg"
DHR_REGISTRATION = os.path.join(DHR_ROOT, "dhr_registration")
for path in (DHR_REGISTRATION, DHR_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from dhr_initial_alignment import exhaustive_rigid_search as ers
from dhr_registration import initial_alignment_methods as ia
from dhr_utils import warping as w


def exhaustive_rigid_search(source: tc.Tensor, target: tc.Tensor, params: dict) -> tc.Tensor:
    """Normalize exhaustive search fallback output to the theta format."""
    transform = ers.exhaustive_rigid_search(source, target, params)
    if isinstance(transform, tc.Tensor) and transform.ndim == 2:
        if tuple(transform.shape) == (3, 3):
            transform = transform[:2, :]
        if tuple(transform.shape) == (2, 3):
            transform = w.affine2theta(
                transform.detach().cpu().numpy().astype("float64"),
                (source.size(2), source.size(3)),
            ).type_as(source).unsqueeze(0).to(source.device)
    return transform


ia.exhaustive_rigid_search = exhaustive_rigid_search


def _load_original_run_module():
    path_run = os.path.join(DHR_ROOT, "run.py")
    spec = importlib.util.spec_from_file_location("deeperhistreg_original_run", path_run)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    dhr_run = _load_original_run_module()
    config = dhr_run.parse_args(sys.argv[1:])
    dhr_run.run_registration(**config)
