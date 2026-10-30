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
for p in [ROOT, os.path.join(ROOT, "twin"), os.path.join(ROOT, "base_paper")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from torch.utils.flop_counter import FlopCounterMode

from src.models.twindrive_v1 import TwinDriveV1
from src.models.twindrive_v2 import TwinDriveV2, TwinDriveV2LaneFix
try:
    from base_paper.model.TwinLite import TwinLiteNet as BaseTwinLiteNet
except ImportError:
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

def total_flops(model, x):
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops()

if __name__ == "__main__":
    models = {"TwinDrive v1": TwinDriveV1, "TwinDrive v2": TwinDriveV2, "TwinDrive v2lf": TwinDriveV2LaneFix,
              "TwinLiteNet (base)": BaseTwinLiteNet}
    print(f"{'model':20s} {'params':>10s} {'conv GMACs':>11s} {'conv GFLOPs':>12s} {'all GFLOPs':>11s} "
          f"{'latency ms (3 repeats)':>26s} {'TFLOPS':>7s}")
    for name, cls in models.items():
        model = cls().eval()
        x_cpu = torch.zeros(1, 3, 360, 640)
        params = sum(p.numel() for p in model.parameters())
        macs = conv_macs(model, x_cpu)
        flops_all = total_flops(model, x_cpu)
        line = f"{name:20s} {params:>10,d} {macs / 1e9:>11.2f} {2 * macs / 1e9:>12.2f} {flops_all / 1e9:>11.2f}"
        if torch.cuda.is_available():
            model = model.cuda()
            runs = latency_ms(model, torch.randn(1, 3, 360, 640, device="cuda"))
            tflops = flops_all / (sum(runs) / len(runs) / 1e3) / 1e12
            line += "   " + ", ".join(f"{r:.2f}" for r in runs) + f"   {tflops:>6.2f}"
        print(line)
    if torch.cuda.is_available():
        print(f"(fp32, batch 1, {torch.cuda.get_device_name(0)}, model only)")