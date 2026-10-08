# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

ZipMap (CVPR 2026) — linear-time stateful 3D reconstruction via test-time training.
This is a **reimplementation** of the paper's code, verified to match the original's
performance. Upstream/author: Haian Jin.

## Environment

```bash
conda activate zipmap
pip install -e .        # from the repo root; torch is pinned to 2.6.0
python -c "import torch; print(torch.__version__)"
```

Working-copy- and machine-specific notes live in `CLAUDE.local.md`, which is not
tracked.

## Layout

```
zipmap/layers/ttt.py        TTT layer
zipmap/layers/block_ttt.py  Block wrapper; `ttt_mode` swaps attention for TTT
zipmap/models/aggregator_ttt.py  Builds blocks, reads ttt_config -> ttt_params
zipmap/models/ZipMap.py     Top-level model
training/config/*.yaml      Hydra configs (this is where TTT knobs are set)
tb/profile_ttt.py           Profiling testbench for the TTT block
tb/profile_ttt_gram_ns.py   Same, but nvtx ranges are Gram NS vs standard NS
tb/profile_gram_ns.py       A/B and operator sweep for the Gram NS option
tb/sweep_gram_ns.py         Repeated-measurement sweep, writes CSV
tb/test_gram_newton_schulz.py  Correctness tests for the Gram NS option
demo_gradio_zipmap*.py      Interactive demos
```

## The TTT module (`zipmap/layers/ttt.py`)

`FastWeightGluMLPMultihead` replaces attention with a SwiGLU MLP whose weights
(`w0`, `w1`, `w2`) are *fast weights*: they are updated at test time by gradient
descent on a self-supervised objective, then applied to the query.

`inter_multi` is the hidden-width multiplier of that MLP (the same role as `mlp_ratio`
in an ordinary transformer): `d_h = head_dim * inter_multi`, and the fast-weight
parameter count scales linearly with it. It also sets the shape of the fast-weight
gradients — see the orthogonalization section below.

Things that are easy to get wrong:

- **Fast weights are per-batch.** Parameters are `[num_heads, d_in, d_h]`; `forward`
  expands them with `.repeat(b, 1, 1)` and returns `[b * num_heads, ...]`. A returned
  `state["w0"]` never has the same shape as `module.w0`.
- **Weight norm is re-applied after every update.** `w0_norm`/`w1_norm`/`w2_norm` are
  captured *before* the loop and renormalized onto the updated weights each step.
- **`ttt_op_order`** is a list of `[start, end, update, apply]` segments. `end == -1`
  means "to the end of the sequence". A segment with `apply=False` produces **no
  output tokens**, so the concatenated output is shorter than the input — the
  segments that raise output must cover the sequence between them.
- **`muon_update_steps=0` is not "skip the update"** — the Newton-Schulz routine then
  returns the gradient divided by its Frobenius norm, i.e. plain normalized SGD.
- **`use_gate_fn=True` only works when `num_heads == 1`.** The gate multiplies
  `o_norm(output)`, shaped `[(b·h), l, head_dim]`, by `gate_fn(x)`, shaped `[b, l, dim]`;
  those broadcast only when `h == 1`. Every shipped config has
  `head_dim == embed_dim == 1024`, so the constraint is invisible there — but it fails
  immediately at e.g. `dim=768, head_dim=64` with a size-mismatch error.
- Two forward paths exist. `use_fused_kernels=False` (the default, and what every
  shipped config uses) runs `fast_weight_swish_glu_weight_norm_mini_batch_apply`.
  `use_fused_kernels=True` runs `bidirectional_lact_swiglu_fused_ckpt`, which is
  marked deprecated in the source and hardcodes 5 NS steps.

### Fast-weight gradient orthogonalization

The fast-weight gradients are orthogonalized Muon-style before being added to the
weights. Two implementations exist, selected by `use_gram_newton_schulz`:

| | function | coefficients |
|---|---|---|
| standard (default) | `zeropower_via_newtonschulz5` | Moonlight, `(3.4445, -4.7750, 2.0315)` repeated |
| Gram | `zeropower_via_gram_newtonschulz5` | Polar Express, one triple per iteration |

`zeropower_orthogonalize` dispatches between them.

The Gram variant iterates on the *Gram matrix* `R = XᵀX` and accumulates the polynomial
multiplier `Q`, applying `Q` to `X` only at the end and at every restart iteration,
rather than re-applying the polynomial to `X` every step. Its
`GRAM_NEWTON_SCHULZ_COEFFICIENTS` / `GRAM_NEWTON_SCHULZ_RESET_ITERATIONS` correspond to
the reference implementation's `ns_coefficients` / `gram_newton_schulz_reset_iterations`.

Two properties that are easy to get wrong when reasoning about it:

- **It is the same operator as the standard iteration given the same coefficients** —
  every intermediate multiplier is a polynomial in `X Xᵀ`, so they all commute and the
  accumulated product is order-independent. The two shipped paths differ only in their
  coefficient schedule (Polar Express vs Moonlight), which is a real behavioural change.
  So flipping the flag changes the *schedule*, not the iteration; to isolate the
  iteration, point `GRAM_NEWTON_SCHULZ_COEFFICIENTS` at a schedule of your choosing, or
  compare against `standard_ns_with_coefficients` in the test file.
- **The cost model is not the obvious one.** The standard implementation already
  transposes a tall `G` so that its `A = X Xᵀ` is the small `[k, k]` Gram matrix
  (`k = min(m, n)`); it is not paying `O(m²n)`. The Gram loop instead spends extra `k³`
  matmuls carrying `R` and `Q`, saving the two `k²m` matmuls per iteration that
  re-apply the polynomial to `X`. Whether that wins depends on both the aspect ratio
  and the absolute size `k` — full measurements are in `CLAUDE.local.md` and
  `tb/reports/gram_ns/SUMMARY.md`.
- **Only non-square gradients take the Gram path.** Square gradients fall back to the
  standard iteration, matching the reference implementation's own dispatch. Since
  `d_h = head_dim * inter_multi`, the gradients are square exactly when
  `inter_multi == 1`, which makes the flag a no-op at that setting.
- `use_fused_kernels=True` **raises** if the flag is also set, rather than silently
  training without it (that deprecated path hardcodes the standard iteration, and a
  silent A/B run would come back identical and read as "no effect").

Other deliberate deviations from the reference: iterations run in **fp32** (the reference
casts to fp16 for its CuTeDSL kernels; the existing standard routine is fp32 to stay
2080ti-safe), and `steps` means *number of coefficient triples used*, capped at
`len(GRAM_NEWTON_SCHULZ_COEFFICIENTS)` (5), with `steps <= 0` degrading to plain
Frobenius normalization.

## Setting TTT knobs

TTT options live in the Hydra configs and reach the module as `**kwargs`:

```yaml
# training/config/*.yaml
model:
  ttt_config:
    params:
      head_dim: 1024
      inter_multi: 2
      base_lr: 0.01
      muon_update_steps: 5
      use_gate_fn: True
      use_gram_newton_schulz: False
```

`aggregator_ttt.py` passes `ttt_config["params"]` through to the block, and
`block_ttt.py` splats it into `FastWeightGluMLPMultihead(dim, **ttt_params)`, so a new
constructor kwarg needs no wiring — only the constructor signature. Unknown keys raise
`TypeError`, so keep the configs and the signature in sync.

## Testing

```bash
python tb/test_gram_newton_schulz.py            # math + tracing + fwd/bwd
python tb/profile_ttt.py                        # profiling testbench (needs CUDA)
python tb/profile_gram_ns.py                    # A/B the flag, end-to-end and per-operator
python tb/sweep_gram_ns.py                      # repeated sweep, writes CSV under tb/reports/
```

`tb/test_gram_newton_schulz.py` checks the compiled implementation against a
transcription of the reference, that the Gram iteration is the same operator as the
standard one under identical coefficients, orthogonalization quality, the dispatcher,
module forward/backward with the flag both ways, the fused-path guard, how far the block
output moves when the flag is flipped, and dynamo tracing with
`backend="eager", fullgraph=True` (which catches graph breaks without needing a working
Inductor backend).

For nsys, `tb/profile_ttt_gram_ns.py` mirrors `tb/profile_ttt.py`'s nvtx pattern with the
two TTT variants in place of TTT-vs-transformer:

```bash
nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true --force-overwrite=true \
    -o tb/reports/ttt_gram_ns python tb/profile_ttt_gram_ns.py --inter-multi 2
```

**`--inter-multi` must be >= 2** or the gradients are square, the Gram path is skipped,
and the two ranges profile identical work. The script warns if that happens, and the
`max|diff|` it prints tells you the flag actually reached the kernels.

## Conventions

- Keep the `@torch.compile` decoration on the orthogonalization routines — they are
  called from inside an already-compiled function, and this is the established pattern
  in `ttt.py`.
- Configs are the source of truth for what a run does. Don't silently flip a config
  value to enable an experiment; add the key and say so.
- Personal working notes belong in `CLAUDE.local.md` (ignored), not here.
