"""
Repeated measurement of the Gram-vs-standard Newton-Schulz cost across the two
dimensions that matter: the absolute size k = min(m, n) and the aspect ratio m / k.

Everything is written to CSV in long format (one row per repeat) so no raw datum is
lost, under tb/reports/gram_ns/raw/. Repeats alternate measurement order so that the
first-measured block is not systematically favoured (the SM clock ramps out of idle,
which penalises whoever goes first).

    python tb/sweep_gram_ns.py --out tb/reports/gram_ns
"""

import argparse
import csv
import os
import statistics
import subprocess
import time

import torch

from zipmap.layers.ttt import (
    FastWeightGluMLPMultihead,
    TTTOperator,
    zeropower_via_gram_newtonschulz5,
    zeropower_via_newtonschulz5,
)

STEPS = 5


def burn_in(seconds, size, device):
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        torch.mm(a, b)
    torch.cuda.synchronize()


def sm_clock():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.split()
        return max(int(v) for v in out if v.isdigit())
    except Exception:
        return -1


def bench(fn, G, inner):
    for _ in range(3):
        fn(G, STEPS)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(inner):
        fn(G, STEPS)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / inner


def operator_sweep(rows, ks, ratios, batches, repeats, inner, device):
    for batch in batches:
        for k in ks:
            for ratio in ratios:
                for orientation in ("tall", "wide"):
                    m, n = (k * ratio, k) if orientation == "tall" else (k, k * ratio)
                    G = torch.randn(batch, m, n, device=device)
                    std_ms, gram_ms = [], []
                    for r in range(repeats):
                        # Alternate order so neither operator always goes first.
                        if r % 2 == 0:
                            std_ms.append(bench(zeropower_via_newtonschulz5, G, inner))
                            gram_ms.append(bench(zeropower_via_gram_newtonschulz5, G, inner))
                            order = "std_first"
                        else:
                            gram_ms.append(bench(zeropower_via_gram_newtonschulz5, G, inner))
                            std_ms.append(bench(zeropower_via_newtonschulz5, G, inner))
                            order = "gram_first"
                        rows.append(dict(
                            batch=batch, k=k, ratio=ratio, orientation=orientation,
                            m=m, n=n, repeat=r, order=order,
                            standard_ms=std_ms[-1], gram_ms=gram_ms[-1],
                        ))
                    print(f"  batch={batch} k={k:5d} {ratio}:1 {orientation:4s} "
                          f"[{m:6d},{n:6d}]  std={statistics.median(std_ms):8.4f}  "
                          f"gram={statistics.median(gram_ms):8.4f}  "
                          f"ratio={statistics.median(std_ms) / statistics.median(gram_ms):6.3f}")


def block_sweep(rows, shapes, repeats, device):
    info = {"ttt_op_order": [[0, -1, True, True]]}
    for (label, dim, head_dim, inter_multi, seq_len) in shapes:
        num_heads = dim // head_dim
        x = torch.randn(1, seq_len, dim, device=device)

        def build(flag):
            torch.manual_seed(0)
            m = FastWeightGluMLPMultihead(
                dim=dim, head_dim=head_dim, inter_multi=inter_multi,
                base_lr=0.01, muon_update_steps=STEPS,
                use_gate_fn=(num_heads == 1), use_fused_kernels=False,
                use_gram_newton_schulz=flag,
            ).to(device)
            return m

        std_block, gram_block = build(False), build(True)
        for _ in range(3):
            for blk in (std_block, gram_block):
                o, _ = blk(x, info=info)
                o.sum().backward()
                x.grad = None

        std_ms, gram_ms = [], []
        for r in range(repeats):
            def one(blk):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                o, _ = blk(x, info=info)
                o.sum().backward()
                e.record()
                torch.cuda.synchronize()
                x.grad = None
                return s.elapsed_time(e)

            if r % 2 == 0:
                std_ms.append(one(std_block)); gram_ms.append(one(gram_block)); order = "std_first"
            else:
                gram_ms.append(one(gram_block)); std_ms.append(one(std_block)); order = "gram_first"

            rows.append(dict(
                label=label, dim=dim, head_dim=head_dim, inter_multi=inter_multi,
                num_heads=num_heads, seq_len=seq_len, repeat=r, order=order,
                standard_ms=std_ms[-1], gram_ms=gram_ms[-1],
            ))

        flag_note = "SQUARE -> gram path not taken" if head_dim == head_dim * inter_multi else "2:1 rectangular"
        print(f"  {label:22s} [{num_heads}, {head_dim}, {head_dim * inter_multi}]  "
              f"{flag_note}  std={statistics.median(std_ms):7.3f}  "
              f"gram={statistics.median(gram_ms):7.3f}  "
              f"ratio={statistics.median(std_ms) / statistics.median(gram_ms):6.3f}")


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tb/reports/gram_ns")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--inner", type=int, default=10)
    ap.add_argument("--burn-in", type=float, default=45.0)
    ap.add_argument("--ks", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    ap.add_argument("--ratios", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 12])
    args = ap.parse_args()

    raw = os.path.join(args.out, "raw")
    print(f"device={args.device} ({torch.cuda.get_device_name(args.device)})")
    print(f"repeats={args.repeats} inner={args.inner} burn_in={args.burn_in}s "
          f"clock_after_burn={sm_clock()} MHz")

    print("\nBurning in...")
    burn_in(args.burn_in, 4096, args.device)
    print(f"  clock now {sm_clock()} MHz")

    print("\n[01] operator sweep, isolated orthogonalizers")
    rows = []
    operator_sweep(rows, args.ks, args.ratios, args.batches, args.repeats, args.inner, args.device)
    write_csv(os.path.join(raw, "01_operator_sweep.csv"), rows)

    print("\n[02] end-to-end TTT block A/B")
    shapes = [
        ("profile_ttt square", 768, 64, 1, 2048),
        ("profile_ttt +mult2", 768, 64, 2, 2048),
        ("real config", 1024, 1024, 2, 2048),
        ("real config mult4", 1024, 1024, 4, 2048),
    ]
    rows2 = []
    block_sweep(rows2, shapes, args.repeats, args.device)
    write_csv(os.path.join(raw, "02_block_ab.csv"), rows2)

    print(f"\nclock at end: {sm_clock()} MHz")
    print("Done.")


if __name__ == "__main__":
    main()
