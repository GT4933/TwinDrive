"""
Size and speed of TwinDrive v1, TwinDrive v2 and the base TwinLiteNet, measured the same way:
parameters, conv/transposed-conv multiply-adds for one 360x640 frame, and model-only GPU latency at
batch 1 (warm-up, then CUDA-synchronised timing over several repeats).

Run from the repo root:  python twin/src/benchmark.py
"""
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(ROOT, "twin"))
sys.path.append(os.path.join(ROOT, "base_paper"))

import torch

from src.models.twinlitenet import TwinLiteNet
from src.models.twindrive_v2 import TwinDriveV2, TwinDriveV2LaneFix
from model.TwinLite import TwinLiteNet as BaseTwinLiteNet


def conv_macs(model, x):
    total = 0
    def hook(mod, inputs, output):
        nonlocal total
        if isinstance(mod, torch.nn.Conv2d):
            total += output.numel() * (mod.in_channels // mod.groups) * mod.kernel_size[0] * mod.kernel_size[1]
        elif isinstance(mod, torch.nn.ConvTranspose2d):
            total += inputs[0].numel() * mod.out_channels * mod.kernel_size[0] * mod.kernel_size[1] // mod.groups
    handles = [m.register_forward_hook(hook) for m in model.modules()]
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    return total


def latency_ms(model, x, warmup=20, iters=100, repeats=3):
    """Returns per-repeat mean latency in ms (model only, batch 1)."""
    results = []
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(iters):
                model(x)
            torch.cuda.synchronize()
            results.append((time.perf_counter() - start) / iters * 1000)
    return results


if __name__ == "__main__":
    models = {"TwinDrive v1": TwinLiteNet, "TwinDrive v2": TwinDriveV2, "TwinDrive v2lf": TwinDriveV2LaneFix,
              "TwinLiteNet (base)": BaseTwinLiteNet}
    print(f"{'model':20s} {'params':>10s} {'conv GMACs':>11s} {'latency ms (3 repeats)':>26s}")
    for name, cls in models.items():
        model = cls().eval()
        params = sum(p.numel() for p in model.parameters())
        macs = conv_macs(model, torch.zeros(1, 3, 360, 640))
        line = f"{name:20s} {params:>10,d} {macs / 1e9:>11.2f}"
        if torch.cuda.is_available():
            model = model.cuda()
            runs = latency_ms(model, torch.randn(1, 3, 360, 640, device="cuda"))
            line += "   " + ", ".join(f"{r:.2f}" for r in runs)
        print(line)
    if torch.cuda.is_available():
        print(f"(fp32, batch 1, {torch.cuda.get_device_name(0)}, model only; MACs exclude attention matmuls)")
