"""
nsys profiling bench for the Gram Newton-Schulz option in the TTT block.

Same shape as `tb/profile_ttt.py`, but the two nvtx ranges are the TTT block
*without* the option (baseline) and *with* it, instead of TTT vs a transformer block.

    nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true \
        -o tb/reports/ttt_gram_ns --force-overwrite=true \
        python tb/profile_ttt_gram_ns.py

IMPORTANT -- `inter_multi` must be >= 2 for this comparison to mean anything.
The fast-weight gradients are [b*h, head_dim, head_dim*inter_multi] (and the transpose
for w1). With the default inter_multi=1 they are *square*, `zeropower_orthogonalize`
falls back to the standard iteration, and the two blocks compute bit-identical results
-- you would be profiling the same thing twice. The two shapes below are the shipped
config's (head_dim == embed_dim, inter_multi 2), i.e. 2:1.

The script checks for that and says so before profiling.
"""

import argparse
import torch

from zipmap.layers.ttt import FastWeightGluMLPMultihead


def build(args, use_gram_newton_schulz):
    # Same seed for both blocks so they start from identical parameters and the
    # output comparison below is a clean A/B rather than an init difference.
    torch.manual_seed(0)
    # use_gate_fn requires num_heads == 1 (the gate multiplies [(b*h), l, head_dim]
    # by [b, l, dim]). The shipped configs have head_dim == embed_dim, so the gate is
    # on there; this keeps the profiled module matching whatever shape is requested.
    num_heads = args.dim // args.head_dim
    return FastWeightGluMLPMultihead(
        dim=args.dim,
        head_dim=args.head_dim,
        inter_multi=args.inter_multi,
        base_lr=0.01,
        muon_update_steps=args.muon_update_steps,
        use_gate_fn=(num_heads == 1),
        use_fused_kernels=False,
        use_gram_newton_schulz=use_gram_newton_schulz,
    ).to(args.device).to(args.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--inter-multi", type=int, default=2)
    ap.add_argument("--muon-update-steps", type=int, default=5)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    args.dtype = getattr(torch, args.dtype)
    d_h = args.head_dim * args.inter_multi
    num_heads = args.dim // args.head_dim

    print(f"Profiling with Tensor Shape: [Batch={args.batch_size}, Seq={args.seq_len}, "
          f"Dim={args.dim}]")
    print(f"  head_dim={args.head_dim} inter_multi={args.inter_multi} -> "
          f"num_heads={num_heads}, hidden={d_h}")
    print(f"  fast-weight grad shapes: [{args.batch_size * num_heads}, {args.head_dim}, {d_h}] "
          f"and [{args.batch_size * num_heads}, {d_h}, {args.head_dim}]")

    if args.head_dim == d_h:
        print("\n  !! inter_multi == 1: the gradients are SQUARE, so the Gram path is")
        print("  !! not taken and both blocks will compute identical results. Use")
        print("  !! --inter-multi 2 or more, or this profile compares a thing to itself.\n")

    print("Initializing modules...")

    baseline_block = build(args, use_gram_newton_schulz=False)
    gram_block = build(args, use_gram_newton_schulz=True)

    x = torch.randn(args.batch_size, args.seq_len, args.dim, device=args.device,
                    dtype=args.dtype, requires_grad=True)

    info = {
        "ttt_op_order": [[0, -1, True, True]]
    }

    print("Warming up GPU pipelines...")
    for _ in range(args.warmup):
        out_b, _ = baseline_block(x, info=info)
        out_b.sum().backward()
        x.grad = None

        out_g, _ = gram_block(x, info=info)
        out_g.sum().backward()
        x.grad = None

    torch.cuda.synchronize()

    # Confirm the flag is actually reaching the kernels before spending a profile on it.
    delta = (out_b.float() - out_g.float()).abs().max().item()
    print(f"Baseline vs Gram output max|diff| = {delta:.4e} "
          f"({'differs -- flag is live' if delta > 0 else 'IDENTICAL -- flag had no effect, see inter_multi note above'})")

    print("Running Profiling...")

    torch.cuda.nvtx.range_push("TTT_NS_Standard")
    out_b, _ = baseline_block(x, info=info)
    out_b.sum().backward()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    x.grad = None

    torch.cuda.nvtx.range_push("TTT_NS_Gram")
    out_g, _ = gram_block(x, info=info)
    out_g.sum().backward()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    x.grad = None

    print("Profiling Script Executed Successfully.")


if __name__ == "__main__":
    main()
