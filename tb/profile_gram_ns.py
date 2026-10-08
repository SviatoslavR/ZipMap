"""
A/B profiling of the `use_gram_newton_schulz` option in the TTT block.

Times forward + backward for both settings at the shapes the shipped configs use
and reports peak memory, so the cost of the Gram iteration can be compared against
the standard one under real Inductor codegen. Needs CUDA.

    python tb/profile_gram_ns.py
    python tb/profile_gram_ns.py --seq-len 4096 --iters 30

Defaults mirror `training/config/default_debug.yaml`: head_dim 1024, inter_multi 2,
muon_update_steps 5, at the ViT-L embedding width.
"""

import argparse
import time

import torch

from zipmap.layers.ttt import (
    FastWeightGluMLPMultihead,
    TTTOperator,
    zeropower_via_gram_newtonschulz5,
    zeropower_via_newtonschulz5,
)


def real_op_order(seq_len):
    """The bidirectional offline order the aggregator builds."""
    return [
        TTTOperator(start=0, end=-1, update=True, apply=False),
        TTTOperator(start=0, end=-1, update=False, apply=True),
    ]


def build_block(args, flag):
    torch.manual_seed(0)
    # use_gate_fn only works when num_heads == 1: the gate multiplies o_norm(output),
    # shaped [(b*h), l, head_dim], by gate_fn(x), shaped [b, l, dim].
    num_heads = args.dim // args.head_dim
    return FastWeightGluMLPMultihead(
        dim=args.dim,
        head_dim=args.head_dim,
        inter_multi=args.inter_multi,
        base_lr=0.01,
        muon_update_steps=args.steps,
        use_gate_fn=(num_heads == 1),
        use_fused_kernels=False,
        use_gram_newton_schulz=flag,
    ).to(args.device)


def bench(block, x, info, iters, warmup):
    for _ in range(warmup):
        out, _ = block(x, info=info)
        out.float().sum().backward()
        x.grad = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(iters):
        out, _ = block(x, info=info)
        out.float().sum().backward()
        x.grad = None
    torch.cuda.synchronize()

    ms = (time.perf_counter() - start) / iters * 1000
    peak_mib = torch.cuda.max_memory_allocated() / 2**20
    return ms, peak_mib, out


def bench_operator(fn, G, steps, iters, warmup):
    for _ in range(warmup):
        fn(G, steps)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn(G, steps)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def operator_sweep(args):
    """
    Time the two orthogonalizers on their own across aspect ratios.

    This is where the Gram variant's cost actually lives. Note the standard
    implementation already transposes a tall G so that its A = X X^T is the small
    [k, k] Gram matrix -- so it is *not* paying O(m^2 n). What Gram buys is not
    having to multiply the polynomial into X every iteration; it pays for that with
    extra [k, k]^3 products used to carry R and Q along. The trade is therefore
    favourable only when m is large relative to k = min(m, n), and the crossover is
    what this sweep measures.
    """
    print("\nOperator-level sweep (steps=%d, batch=%d, min dim k=%d):"
          % (args.steps, args.batch, args.sweep_k))
    print(f"  {'shape':>16}  {'ratio':>5}  {'standard':>10}  {'gram':>10}  {'speedup':>8}")

    for ratio in args.ratios:
        for orientation in ("tall", "wide"):
            if orientation == "tall":
                m, n = args.sweep_k * ratio, args.sweep_k
            else:
                m, n = args.sweep_k, args.sweep_k * ratio
            G = torch.randn(args.batch, m, n, device=args.device)
            std_ms = bench_operator(zeropower_via_newtonschulz5, G, args.steps,
                                    args.iters, args.warmup)
            gram_ms = bench_operator(zeropower_via_gram_newtonschulz5, G, args.steps,
                                     args.iters, args.warmup)
            print(f"  {f'[{m}, {n}]':>16}  {f'{ratio}:1':>5}  "
                  f"{std_ms:9.3f}ms  {gram_ms:9.3f}ms  {std_ms / gram_ms:7.3f}x")


def burn_in(seconds, size, device, dtype):
    """
    Hold the GPU under real load before measuring.

    This machine idles its SM clock far below boost (observed 300-1350 MHz vs ~1900
    under load). A measurement started from idle is inflated, and because the ramp
    takes time it penalises whichever block is measured first. Measured directly: the
    same block ran 10.24 ms at 1350 MHz and 8.32 ms at 1868 MHz.
    """
    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        torch.mm(a, b)
    torch.cuda.synchronize()

    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.split()
        if out:
            print(f"  burn-in done, SM clocks (MHz): {', '.join(out)}")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--head-dim", type=int, default=1024)
    ap.add_argument("--inter-multi", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-sweep", action="store_true",
                    help="skip the isolated orthogonalizer sweep")
    ap.add_argument("--burn-in", type=float, default=0.0,
                    help="seconds of GPU load before measuring, to leave the idle SM clock")
    ap.add_argument("--burn-in-size", type=int, default=4096,
                    help="square GEMM size used for the burn-in")
    ap.add_argument("--sweep-k", type=int, default=1024,
                    help="min matrix dimension held fixed across the sweep")
    ap.add_argument("--ratios", type=int, nargs="+", default=[1, 2, 3, 4, 8, 16])
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this profiling script.")

    dtype = getattr(torch, args.dtype)
    d_h = args.head_dim * args.inter_multi
    print(f"device={args.device} ({torch.cuda.get_device_name(args.device)})  dtype={args.dtype}")
    print(f"dim={args.dim} head_dim={args.head_dim} inter_multi={args.inter_multi} "
          f"-> fast-weight grads [{args.batch}, {args.dim}, {d_h}] / [{args.batch}, {d_h}, {args.dim}]")
    print(f"seq_len={args.seq_len} batch={args.batch} steps={args.steps} "
          f"iters={args.iters} warmup={args.warmup}")

    if args.burn_in > 0:
        print(f"Burning in for {args.burn_in:.0f}s to leave the idle SM clock...")
        burn_in(args.burn_in, args.burn_in_size, args.device, dtype)

    x = torch.randn(args.batch, args.seq_len, args.dim, device=args.device,
                    dtype=dtype, requires_grad=True)
    info = {"ttt_op_order": real_op_order(args.seq_len)}

    results = {}
    outputs = {}
    for flag in (False, True):
        torch.cuda.empty_cache()
        block = build_block(args, flag)
        ms, peak, out = bench(block, x, info, args.iters, args.warmup)
        results[flag] = (ms, peak)
        outputs[flag] = out.detach().float()
        print(f"  use_gram_newton_schulz={flag!s:<5}  {ms:8.2f} ms/iter   peak {peak:7.1f} MiB")

    baseline_ms, baseline_peak = results[False]
    gram_ms, gram_peak = results[True]
    print()
    print(f"  speedup (gram vs standard): {baseline_ms / gram_ms:.3f}x")
    print(f"  peak memory delta:          {gram_peak - baseline_peak:+.1f} MiB")

    # Both settings must produce a usable output; they are expected to differ, since
    # the flag changes the coefficient schedule as well as the iteration.
    delta = (outputs[False] - outputs[True]).abs().max().item()
    print(f"  max|output(F) - output(T)|: {delta:.4e}  "
          f"({'differs, as expected' if delta > 0 else 'IDENTICAL -- flag had no effect!'})")
    print(f"  both finite: {torch.isfinite(outputs[False]).all().item() and torch.isfinite(outputs[True]).all().item()}")

    if not args.skip_sweep:
        operator_sweep(args)


if __name__ == "__main__":
    main()
