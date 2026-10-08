"""
Correctness checks for the `use_gram_newton_schulz` option on the TTT module
(`zipmap/layers/ttt.py`).

Run from the repo root:
    python tb/test_gram_newton_schulz.py

Device-agnostic: uses CUDA when available, CPU otherwise, so the math can be
checked without a GPU. Set TORCHDYNAMO_DISABLE=1 to check the eager math with
torch.compile out of the picture.
"""

import os

import torch

from zipmap.layers.ttt import (
    GRAM_NEWTON_SCHULZ_COEFFICIENTS,
    GRAM_NEWTON_SCHULZ_RESET_ITERATIONS,
    FastWeightGluMLPMultihead,
    fast_weight_swish_glu_weight_norm_mini_batch_apply,
    zeropower_orthogonalize,
    zeropower_via_gram_newtonschulz5,
    zeropower_via_newtonschulz5,
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
STEPS = 5

_failures = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        _failures.append(name)


def reference_gram_newton_schulz(X, steps, reset_iterations=GRAM_NEWTON_SCHULZ_RESET_ITERATIONS,
                                 epsilon=1e-7):
    """
    Independent transcription of the reference implementation
    (`gram_newton_schulz/gram_newton_schulz.py`, `_make_compiled_gram` with the
    torch backend).

    Written with torch.baddbmm to mirror the reference's sym_baddbmm / mm_add
    helpers, so it does not share code shape with the implementation under test.
    The reference runs the iterations in fp16; this runs them in fp32.
    """
    tall_skinny = X.size(-2) > X.size(-1)
    X = X.to(torch.float32)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + epsilon)

    if tall_skinny:
        R = X.mT @ X
    else:
        R = X @ X.mT

    I = torch.eye(R.size(-1), device=X.device, dtype=X.dtype).expand(R.size(0), -1, -1)
    Q = None
    coeffs = list(GRAM_NEWTON_SCHULZ_COEFFICIENTS)[:steps]

    for i, (a, b, c) in enumerate(coeffs):
        if i in reset_iterations and i != 0:
            X = X @ Q if tall_skinny else Q @ X
            R = X.mT @ X if tall_skinny else X @ X.mT
            Q = None

        Z = torch.baddbmm(R, R, R, alpha=c, beta=b)

        if i == 0 or i in reset_iterations:
            Q = Z + a * I
        else:
            Q = torch.baddbmm(Q, Q, Z, beta=a)

        if i < len(coeffs) - 1 and (i + 1) not in reset_iterations:
            RZ = torch.baddbmm(R, R, Z, beta=a)
            R = torch.baddbmm(RZ, Z, RZ, beta=a)

    return X @ Q if tall_skinny else Q @ X


def standard_ns_with_coefficients(X, coefficients):
    """
    The standard iteration, but able to take a per-iteration coefficient schedule.

    This is the reference implementation's `_make_compiled_standard`. It exists so
    that the Gram-vs-standard comparison can be run with *identical* coefficients,
    which isolates the iteration from the coefficient schedule.
    """
    tall = X.size(-2) > X.size(-1)
    X = X.to(torch.float32)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    Y = X.transpose(-1, -2) if tall else X
    for (a, b, c) in coefficients:
        A = Y @ Y.transpose(-1, -2)
        B = b * A + c * (A @ A)
        Y = a * Y + B @ Y
    return Y.transpose(-1, -2) if tall else Y


def polar_factor(X):
    """The exact orthogonal polar factor U V^T, via SVD."""
    U, _, Vh = torch.linalg.svd(X.float(), full_matrices=False)
    return U @ Vh


def orthogonality_residual(Y):
    """max |singular_value - 1| of the (batched) output matrix."""
    sv = torch.linalg.svdvals(Y.float())
    return (sv - 1.0).abs().max().item()


def test_matches_reference():
    print("\n[1] compiled implementation vs. independent transcription of the reference")
    torch.manual_seed(0)
    for m, n in [(128, 32), (32, 128), (64, 64), (256, 64)]:
        G = torch.randn(2, m, n, device=DEVICE)
        got = zeropower_via_gram_newtonschulz5(G, STEPS)
        want = reference_gram_newton_schulz(G, STEPS)
        # Not bitwise: the transcription applies the polynomial as mul+add where the
        # implementation under test folds it into @. fp32 rounding then propagates
        # through 5 chained iterations whose coefficients reach ~20, so agreement is
        # ~1e-5 relative. A real defect (wrong coefficient / missed reset / transposed
        # matmul) moves the output by O(1), which this threshold still catches.
        err = (got - want).abs().max().item() / max(want.abs().max().item(), 1e-6)
        check(f"shape [{m}, {n}] matches reference", err < 1e-3, f"rel|diff|={err:.2e}")


def test_orthogonalizes():
    print("\n[2] orthogonalization quality vs. the standard Newton-Schulz iteration")
    torch.manual_seed(1)
    for m, n in [(128, 32), (32, 128), (256, 64)]:
        G = torch.randn(2, m, n, device=DEVICE)
        gram = zeropower_via_gram_newtonschulz5(G, STEPS)
        std = zeropower_via_newtonschulz5(G, STEPS)

        sv_gram = torch.linalg.svdvals(gram.float())[0]
        sv_std = torch.linalg.svdvals(std.float())[0]
        r_gram = (sv_gram - 1.0).abs().max().item()
        r_std = (sv_std - 1.0).abs().max().item()

        # Both are approximate orthogonalizers: Muon-style NS deliberately lands on
        # US'V^T with S' spread around 1 rather than on exactly UV^T. What has to hold
        # is that the Gram variant is not materially worse than the standard one.
        check(f"shape [{m}, {n}] gram quality within 1.15x of standard",
              r_gram < r_std * 1.15 + 1e-3,
              f"gram |sv-1|max={r_gram:.4f} (sv {sv_gram.min():.3f}..{sv_gram.max():.3f}), "
              f"std |sv-1|max={r_std:.4f} (sv {sv_std.min():.3f}..{sv_std.max():.3f})")


def test_same_operator_as_standard():
    """
    The Gram iteration must compute the *same operator* as the standard iteration.

    It is the same polynomial evaluated in a different association order: every
    intermediate multiplier is a polynomial in X X^T, so they all commute with each
    other (and with X), which makes the accumulated product order-independent. Any
    deviation beyond fp32 rounding would mean the port is wrong -- and a far more
    dangerous failure than a graph break, because the model would still train.

    This is also the check that the earlier "independent transcription" comparison
    cannot make: that one was written from the same source as the implementation, so
    a shared misreading would pass both.
    """
    print("\n[7] same operator as the standard iteration (identical coefficients)")
    torch.manual_seed(6)
    worst = 0.0
    for steps in (1, 2, 3, 4, 5):
        coeffs = list(GRAM_NEWTON_SCHULZ_COEFFICIENTS)[:steps]
        for m, n in [(1024, 512), (512, 1024), (256, 512), (512, 512), (64, 64)]:
            G = torch.randn(2, m, n, device=DEVICE)
            gram = zeropower_via_gram_newtonschulz5(G, steps)
            std = standard_ns_with_coefficients(G, coeffs)
            rel = ((gram - std).norm() / std.norm()).item()
            worst = max(worst, rel)
    # Measured worst case is 9.6e-05 at 5 steps; a real algorithmic difference is
    # O(0.1-1) relative, so this threshold separates the two by orders of magnitude.
    check("gram == standard iteration up to fp32 rounding",
          worst < 1e-3, f"worst rel|diff|={worst:.2e} over steps 1-5, square and non-square")

    # And both must sit at the same distance from the exact polar factor.
    G = torch.randn(2, 512, 256, device=DEVICE)
    P = polar_factor(G)
    e_gram = ((zeropower_via_gram_newtonschulz5(G, STEPS) - P).norm() / P.norm()).item()
    e_std = ((standard_ns_with_coefficients(G, list(GRAM_NEWTON_SCHULZ_COEFFICIENTS)[:STEPS]) - P).norm()
             / P.norm()).item()
    check("gram and standard match their target equally well",
          abs(e_gram - e_std) < 1e-4,
          f"dist to true polar factor: gram={e_gram:.5f} std={e_std:.5f}")


def test_steps_edge_cases():
    print("\n[3] step-count edge cases")
    torch.manual_seed(2)
    G = torch.randn(2, 64, 32, device=DEVICE)
    want = G.float() / (G.float().norm(dim=(1, 2), keepdim=True) + 1e-7)

    for steps in (0, -1, -5):
        got = zeropower_via_gram_newtonschulz5(G, steps)
        err = (got - want).abs().max().item()
        check(f"steps={steps} == normalized G", err < 1e-6, f"max|diff|={err:.2e}")

    # More steps than coefficient triples: capped at the schedule length rather than
    # cycling or indexing out of range.
    over = zeropower_via_gram_newtonschulz5(G, 99)
    full = zeropower_via_gram_newtonschulz5(G, len(GRAM_NEWTON_SCHULZ_COEFFICIENTS))
    check("steps > len(coefficients) is capped", torch.equal(over, full))


def test_dispatch():
    print("\n[4] dispatcher picks the right branch")
    torch.manual_seed(3)
    square = torch.randn(2, 64, 64, device=DEVICE)
    tall = torch.randn(2, 128, 32, device=DEVICE)
    wide = torch.randn(2, 32, 128, device=DEVICE)

    base = zeropower_via_newtonschulz5(square, STEPS)
    check("square + flag on  -> standard (bitwise)",
          torch.equal(zeropower_orthogonalize(square, STEPS, True), base))
    check("square + flag off -> standard (bitwise)",
          torch.equal(zeropower_orthogonalize(square, STEPS, False), base))
    check("tall + flag off   -> standard (bitwise)",
          torch.equal(zeropower_orthogonalize(tall, STEPS, False),
                      zeropower_via_newtonschulz5(tall, STEPS)))
    check("tall + flag on    -> gram (bitwise)",
          torch.equal(zeropower_orthogonalize(tall, STEPS, True),
                      zeropower_via_gram_newtonschulz5(tall, STEPS)))
    check("wide + flag on    -> gram (bitwise)",
          torch.equal(zeropower_orthogonalize(wide, STEPS, True),
                      zeropower_via_gram_newtonschulz5(wide, STEPS)))


def test_module_forward_backward():
    print("\n[5] FastWeightGluMLPMultihead forward/backward")

    for inter_multi, label in [(2, "non-square grads (gram active)"),
                               (1, "square grads (gram falls back)")]:
        for flag in [False, True]:
            torch.manual_seed(4)
            block = FastWeightGluMLPMultihead(
                dim=128,
                head_dim=32,
                inter_multi=inter_multi,
                muon_update_steps=STEPS,
                use_gram_newton_schulz=flag,
            ).to(DEVICE)
            x = torch.randn(2, 16, 128, device=DEVICE, requires_grad=True)
            # Two segments, both raising output, so the fast weights carry over from
            # the first update into the second (the multi-segment TTT path).
            info = {"ttt_op_order": [[0, 8, True, True], [8, -1, True, True]]}
            out, state = block(x, info=info)
            loss = out.float().sum()
            loss.backward()

            grads_ok = all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in block.parameters()
            )
            # The returned fast weights are the per-batch expansion of the parameters.
            shapes_ok = out.shape == (2, 16, 128) and all(
                state[k].shape == getattr(block, k).repeat(2, 1, 1).shape
                for k in ("w0", "w1", "w2")
            )
            check(f"inter_multi={inter_multi} flag={flag} {label}: fwd/bwd finite",
                  torch.isfinite(out).all().item() and grads_ok,
                  f"out.shape={tuple(out.shape)}")
            check(f"inter_multi={inter_multi} flag={flag} {label}: shapes",
                  shapes_ok)

    # The deprecated fused path hardcodes the standard iteration; combining it with
    # the flag must error instead of silently ignoring it.
    torch.manual_seed(4)
    fused = FastWeightGluMLPMultihead(
        dim=128, head_dim=32, muon_update_steps=STEPS,
        use_fused_kernels=True, use_gram_newton_schulz=True,
    ).to(DEVICE)
    x = torch.randn(2, 16, 128, device=DEVICE)
    try:
        fused(x, info={"ttt_op_order": [[0, -1, True, True]]})
        check("fused kernels + gram flag raises", False, "no error raised")
    except ValueError as e:
        check("fused kernels + gram flag raises", "not implemented" in str(e))
    except Exception as e:  # noqa: BLE001
        check("fused kernels + gram flag raises", False, f"{type(e).__name__}: {e}")


def test_block_output_delta():
    """
    How far does the block's OUTPUT move when the flag is flipped?

    Same module object, same input, same initial fast weights -- only the flag differs,
    so this is the isolated effect of the switch rather than run-to-run noise.

    The flag is expected to *not* be numerically interchangeable (it changes the
    coefficient schedule), so this does not assert equality. What it asserts is that
    the change stays in the same functional regime: same output scale, high
    correlation, no outlier tokens, and a perturbation smaller than a genuinely
    different update rule. That last comparison is the point -- `muon_update_steps=0`
    (no orthogonalization at all) is measured on the same scale to calibrate what
    "big" means here.
    """
    print("\n[8] block output with the flag on vs off (isolated, same weights)")
    torch.manual_seed(7)

    # num_heads == 1 is required by the gate path, so head_dim == dim (matches the
    # shipped configs, where head_dim == embed_dim == 1024).
    dim, head_dim, inter_multi, seq_len, batch = 128, 128, 2, 64, 2
    x = torch.randn(batch, seq_len, dim, device=DEVICE)
    order = {"ttt_op_order": [[0, -1, True, False], [0, -1, False, True]]}

    def make(steps):
        torch.manual_seed(11)
        m = FastWeightGluMLPMultihead(
            dim=dim, head_dim=head_dim, inter_multi=inter_multi,
            base_lr=0.01, muon_update_steps=steps, use_gate_fn=True,
            use_gram_newton_schulz=False,
        ).to(DEVICE)
        m.eval()
        return m

    with torch.no_grad():
        block = make(STEPS)
        out_std, _ = block(x, info=order)
        block.use_gram_newton_schulz = True
        out_gram, state_gram = block(x, info=order)
        block.use_gram_newton_schulz = False
        _, state_std = block(x, info=order)

        out_cal, _ = make(0)(x, info=order)  # calibration: no orthogonalization

    out_std, out_gram, out_cal = (o.float() for o in (out_std, out_gram, out_cal))

    def relf(a, b):
        return ((a - b).norm() / b.norm()).item()

    def cos(a, b):
        return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()

    rel, c = relf(out_gram, out_std), cos(out_gram, out_std)
    rel_cal, c_cal = relf(out_cal, out_std), cos(out_cal, out_std)
    scale_ratio = out_gram.abs().mean().item() / out_std.abs().mean().item()
    per_tok = ((out_gram - out_std).norm(dim=-1) / out_std.norm(dim=-1)).max().item()

    print(f"    output magnitude: standard={out_std.abs().mean():.4f} "
          f"gram={out_gram.abs().mean():.4f} calibration={out_cal.abs().mean():.4f}")
    print(f"    gram vs standard:   relF={rel:.4f}  cos={c:.5f}")
    print(f"    calibration (no-orthogonalization vs standard): relF={rel_cal:.4f} "
          f"cos={c_cal:.5f}")

    check("output scale preserved", 0.8 < scale_ratio < 1.25, f"gram/standard = {scale_ratio:.4f}")
    check("output stays highly correlated", c > 0.95, f"cos = {c:.5f}")
    check("no outlier tokens", per_tok < 3 * rel, f"worst per-token relF = {per_tok:.4f}")
    check("milder than dropping orthogonalization entirely", rel < rel_cal,
          f"relF {rel:.4f} vs calibration {rel_cal:.4f} (ratio {rel / rel_cal:.2f})")
    check("fast-weight states move less than the output",
          max(relf(state_gram[k].float(), state_std[k].float()) for k in ("w0", "w1", "w2")) < rel,
          f"worst state relF = {max(relf(state_gram[k].float(), state_std[k].float()) for k in ('w0', 'w1', 'w2')):.4f}")


def test_fullgraph_tracing():
    """
    Verify dynamo can trace the new code with no graph breaks.

    Uses the eager backend because Inductor on this machine has no C++ compiler;
    fullgraph=True is what matters here -- it raises on any graph break, which is
    the failure mode `@torch.compile` would hit in production.
    """
    print("\n[6] dynamo traces without graph breaks (backend=eager, fullgraph=True)")
    torch._dynamo.config.suppress_errors = False
    torch.manual_seed(5)

    for name, fn in [("zeropower_via_newtonschulz5", zeropower_via_newtonschulz5),
                     ("zeropower_via_gram_newtonschulz5", zeropower_via_gram_newtonschulz5)]:
        inner = getattr(fn, "_torchdynamo_orig_callable", fn)
        for m, n in [(64, 32), (64, 64)]:
            G = torch.randn(2, m, n, device=DEVICE)
            try:
                out = torch.compile(inner, backend="eager", fullgraph=True)(G, STEPS)
                ok, detail = out.shape == G.shape, ""
            except Exception as e:  # noqa: BLE001 - report any tracing failure
                ok, detail = False, f"{type(e).__name__}: {str(e)[:120]}"
            check(f"{name} [{m}, {n}] traces", ok, detail)

    # The mini-batch apply is the real integration point: it has dynamic=True in
    # production and calls the orthogonalizers inside its own graph.
    inner = getattr(fast_weight_swish_glu_weight_norm_mini_batch_apply,
                    "_torchdynamo_orig_callable",
                    fast_weight_swish_glu_weight_norm_mini_batch_apply)
    b, l, d, dh = 2, 16, 32, 64
    for flag in [False, True]:
        args = (
            torch.randn(b, d, dh, device=DEVICE) * 0.1,
            torch.randn(b, dh, d, device=DEVICE) * 0.1,
            torch.randn(b, d, dh, device=DEVICE) * 0.1,
            torch.randn(b, l, d, device=DEVICE),
            torch.randn(b, l, d, device=DEVICE),
            torch.randn(b, l, d, device=DEVICE),
            torch.rand(b, l, 1, device=DEVICE) * 0.1,
            torch.rand(b, l, 1, device=DEVICE) * 0.1,
            torch.rand(b, l, 1, device=DEVICE) * 0.1,
            [[0, 8, True, True], [8, -1, True, True]],
            STEPS,
            flag,
        )
        try:
            out = torch.compile(inner, backend="eager", fullgraph=True, dynamic=True)(*args)
            ok, detail = out[0].shape == (b, l, d), ""
        except Exception as e:  # noqa: BLE001 - report any tracing failure
            ok, detail = False, f"{type(e).__name__}: {str(e)[:120]}"
        check(f"mini_batch_apply flag={flag} traces (dynamic=True)", ok, detail)


def require_working_compile_backend():
    """
    Fail early with a readable message if torch.compile cannot run here.

    The math checks go through the @torch.compile-decorated functions, so they need a
    working backend. Inductor needs a C++ compiler, which a CPU-only Windows box
    usually lacks -- but that is an environment problem, not a code problem. Section
    [6] checks tracing separately with the eager backend, which needs no compiler.
    """
    if os.environ.get("TORCHDYNAMO_DISABLE"):
        return
    try:
        torch.compile(lambda t: t * 2, backend="inductor", fullgraph=True)(torch.zeros(8))
    except Exception:  # noqa: BLE001 - any backend failure means the same thing here
        print(
            "\n" + "=" * 60 + "\n"
            "torch.compile is unavailable with the default (inductor) backend on this\n"
            "machine -- typically a missing C++ compiler. This is environmental; the\n"
            "code under test is fine.\n\n"
            "Re-run with compilation disabled to check the math:\n"
            "    TORCHDYNAMO_DISABLE=1 python tb/test_gram_newton_schulz.py\n"
        )
        raise SystemExit(2)


def main():
    print(f"device={DEVICE}  steps={STEPS}")
    print(f"gram coefficients: {len(GRAM_NEWTON_SCHULZ_COEFFICIENTS)} entries, "
          f"reset at {GRAM_NEWTON_SCHULZ_RESET_ITERATIONS}")

    # The checks below deliberately compile the same functions for several input
    # shapes, which is precisely what dynamo's per-code-object recompile budget
    # limits (default 8). Without this, hitting that budget surfaces as
    # RecompileLimitExceeded and masquerades as a tracing failure. Production is
    # unaffected: the dynamic=True caller inlines these operators into one graph.
    torch._dynamo.config.cache_size_limit = 64

    require_working_compile_backend()
    test_matches_reference()
    test_same_operator_as_standard()
    test_orthogonalizes()
    test_steps_edge_cases()
    test_dispatch()
    test_module_forward_backward()
    test_block_output_delta()
    test_fullgraph_tracing()

    print("\n" + "=" * 60)
    if _failures:
        print(f"FAILED: {len(_failures)} check(s)")
        for name in _failures:
            print(f"  - {name}")
        raise SystemExit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
