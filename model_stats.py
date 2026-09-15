import sys
import argparse

import torch
import torch.nn as nn

import model


def count_parameters(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def fmt_int(x):
    return f"{x:,}"


def print_top_level_parameters(net):
    print("\n" + "=" * 88)
    print("Per-module parameter count")
    print("=" * 88)
    print(f"{'Module':<24} {'Type':<30} {'Params':>16} {'Trainable':>16}")
    print("-" * 88)

    total = 0
    trainable = 0

    for name, child in net.named_children():
        p, t = count_parameters(child)
        total += p
        trainable += t
        print(f"{name:<24} {child.__class__.__name__:<30} {fmt_int(p):>16} {fmt_int(t):>16}")

    print("-" * 88)
    print(f"{'SUM':<24} {'':<30} {fmt_int(total):>16} {fmt_int(trainable):>16}")


class ModelWrapper(nn.Module):
    """
    Matches the current total.py style:
        {"face": x, "pairedface": x}
    """
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net(
            {"face": x, "pairedface": x},
            require_img=False
        )
        return out[0]


def profile_complexity(net, image_size, device):
    try:
        from thop import profile
    except ImportError:
        print("\n[ERROR] thop is not installed.")
        print(f"Install with:\n{sys.executable} -m pip install thop")
        return

    wrapper = ModelWrapper(net).to(device)
    wrapper.eval()

    x = torch.randn(1, 3, image_size, image_size, device=device)

    print("\n" + "=" * 88)
    print("MACs / FLOPs")
    print("=" * 88)
    print(f"Input: face={tuple(x.shape)}, pairedface={tuple(x.shape)}")

    try:
        macs, thop_params = profile(
            wrapper,
            inputs=(x,),
            verbose=False
        )
    except Exception as e:
        print("\n[ERROR] THOP profiling failed:")
        print(repr(e))
        print("\nParameter counts above are still valid.")
        print("If this happens, send me the error and I can add custom THOP rules.")
        return

    flops = 2 * macs

    print(f"MACs      : {macs / 1e9:.6f} G")
    print(f"FLOPs     : {flops / 1e9:.6f} G  (2 x MACs)")
    print(f"THOP Params: {thop_params / 1e6:.6f} M")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size",
        type=int,
        default=160,
        help="Input image size. Default: 160"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Profiling device. Default: cpu"
    )
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA is unavailable; using CPU instead.")
        device = "cpu"

    net = model.Model()
    net.eval()

    total_params, trainable_params = count_parameters(net)
    frozen_params = total_params - trainable_params

    print("=" * 88)
    print("Current Model Statistics")
    print("=" * 88)
    print(f"Model       : {net.__class__.__name__}")
    print(f"Input size  : 3 x {args.size} x {args.size}")
    print(f"Device      : {device}")

    print("\n" + "=" * 88)
    print("Overall parameter count")
    print("=" * 88)
    print(f"Total parameters     : {fmt_int(total_params)} ({total_params / 1e6:.6f} M)")
    print(f"Trainable parameters : {fmt_int(trainable_params)} ({trainable_params / 1e6:.6f} M)")
    print(f"Frozen parameters    : {fmt_int(frozen_params)} ({frozen_params / 1e6:.6f} M)")

    print_top_level_parameters(net)
    profile_complexity(net, args.size, device)


if __name__ == "__main__":
    main()
