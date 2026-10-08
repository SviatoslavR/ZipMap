import collections
import math

import torch
from torch import nn

import torch.nn.functional as F
from einops import rearrange

try:
    # from l2norm_triton_kernels import l2_norm_add_fused
    from lact_with_act_ckpt_plain import (
        lact_swiglu_ffn_fast_weight_grads_with_ckpt,
        fused_swiglu_ffn_fwd_with_ckpt,
    )
except ImportError:
    # from .l2norm_triton_kernels import l2_norm_add_fused
    from .lact_with_act_ckpt_plain import (
        lact_swiglu_ffn_fast_weight_grads_with_ckpt,
        fused_swiglu_ffn_fwd_with_ckpt,
    )


TTTOperator = collections.namedtuple("TTTOperator", ["start", "end", "update", "apply"])


@torch.compile
def inv_softplus(x):
    y = x + math.log(-math.expm1(-x))
    return y

@torch.compile
def silu_backprop(dy: torch.Tensor, x: torch.Tensor):
    """
    Args:
        dy: [b, d, l], gradient of the outer loss wrt the y
        x: [b, d, l], input of the silu activation
    outs:
        dx: [b, d, l], gradient of the outer loss wrt the x
        dx = dy * sigma * (1 + x * (1 - sigma))
    """
    sigma = torch.sigmoid(x)
    dx = dy * sigma * (1 + x * (1 - sigma))
    return dx

@torch.compile()
def zeropower_via_newtonschulz5(G, steps):
    """
    modified from https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py#L49
    Major change: G is [b, d, d] rather than [d, d]
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    Args:
        G: [b, d, d]
        steps: int
    Returns:
        X: [b, d, d]
    """
    assert len(G.shape) == 3
    a, b, c = (3.4445, -4.7750, 2.0315)
    # Previous version: X = G.bfloat16()
    # Use float32 to be compatible with 2080ti
    X = G.to(dtype=torch.float32, device=G.device).contiguous()
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.transpose(1, 2)
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    return X


# Coefficients of the Gram Newton-Schulz iteration, ported from the reference
# implementation (gram_newton_schulz/coefficients.py in the gram-newton-schulz repo).
# Source: https://arxiv.org/pdf/2505.16932 (Polar Express).
#
# Dividing a, b, c by sf, sf^3, sf^5 is exactly equivalent to evaluating the
# polynomial a*x + b*x^3 + c*x^5 at x / sf, i.e. it widens the region of convergence
# by 5%. That absorbs the gap between the Frobenius normalization used below and the
# spectral normalization the schedule was fitted against.
_NS_SAFETY_FACTOR = 1.05
_NS_POLAR_EXPRESS_COEFFICIENTS = (
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
)
GRAM_NEWTON_SCHULZ_COEFFICIENTS = tuple(
    (a / _NS_SAFETY_FACTOR, b / _NS_SAFETY_FACTOR**3, c / _NS_SAFETY_FACTOR**5)
    for (a, b, c) in _NS_POLAR_EXPRESS_COEFFICIENTS
)

# Iterations at which the Gram Newton-Schulz iteration restarts: the accumulated
# multiplier is folded back into X and the Gram matrix is recomputed from scratch.
# Between restarts R is only carried along by the polynomial, so it drifts away from
# the true Gram matrix of X; the restart is what keeps that drift from accumulating.
GRAM_NEWTON_SCHULZ_RESET_ITERATIONS = (2,)


@torch.compile()
def zeropower_via_gram_newtonschulz5(G, steps):
    """
    Gram Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

    This is the "Gram" variant of `zeropower_via_newtonschulz5`. Rather than folding
    the polynomial into X at every iteration, it iterates on the Gram matrix R and
    accumulates the multiplier Q, touching X only at the start, at every restart, and
    at the end.

    The standard implementation is *not* naive by comparison: it already transposes a
    tall G so that its A = X X^T is the small [k, k] Gram matrix (k = min(m, n)), so
    it is not paying O(m^2 n). Both variants do the same k x k work. The difference is
    that the standard one spends two k^2 * m matmuls per iteration re-applying the
    polynomial to X, while this one spends extra k^3 matmuls carrying R and Q. The
    trade only pays off when m is large relative to k -- measured (RTX 2080 Ti, fp32,
    k=1024, 5 steps, batch 1), with tall and wide agreeing:

        m / k :    1      2      3      4      8     16
        gain  :  0.83x  1.15x  1.35x  1.52x  1.90x  2.19x

    That table holds k at 1024. Absolute size matters as much as shape: this loop
    issues more kernels per iteration (four k^3 products plus extra elementwise ops,
    against the standard one's two k^2 * m products), which only pays off once those
    kernels are too big to be launch-bound. At the same 2:1 aspect ratio, the whole
    block runs at 0.93x with k=64 (head_dim 64, 12 heads) and 1.06x with k=1024
    (head_dim 1024, 1 head). So it loses on square matrices, is a modest win at the
    shipped config's k=1024 / 2:1, and wins clearly from 3:1 up.

    `zeropower_orthogonalize` therefore keeps the standard iteration for square G,
    matching the reference implementation; the measured crossover is nearer 1.5:1,
    so a caller that cares can tighten it.

    Args:
        G: [b, m, n]
        steps: int, number of iterations. At most len(GRAM_NEWTON_SCHULZ_COEFFICIENTS);
            extra steps are ignored. steps <= 0 degrades to plain normalization,
            matching `zeropower_via_newtonschulz5(G, 0)`.
    Returns:
        X: [b, m, n]
    """
    assert len(G.shape) == 3
    tall_skinny = G.size(1) > G.size(2)
    # Keep float32 for consistency with zeropower_via_newtonschulz5 (2080ti-safe).
    X = G.to(dtype=torch.float32, device=G.device).contiguous()
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)

    # Not a plain slice: a negative slice bound would silently drop trailing steps.
    coefficients = GRAM_NEWTON_SCHULZ_COEFFICIENTS[: max(steps, 0)]
    if len(coefficients) == 0:
        return X

    if tall_skinny:
        R = X.transpose(1, 2) @ X  # [b, n, n]
    else:
        R = X @ X.transpose(1, 2)  # [b, m, m]
    I = (
        torch.eye(R.size(-1), device=X.device, dtype=X.dtype)
        .unsqueeze(0)
        .expand(R.size(0), -1, -1)
    )
    Q = None

    for i, (a, b, c) in enumerate(coefficients):
        if i in GRAM_NEWTON_SCHULZ_RESET_ITERATIONS and i != 0:
            X = X @ Q if tall_skinny else Q @ X
            R = X.transpose(1, 2) @ X if tall_skinny else X @ X.transpose(1, 2)
            Q = None

        Z = b * R + c * (R @ R)
        if i == 0 or i in GRAM_NEWTON_SCHULZ_RESET_ITERATIONS:
            Q = Z + a * I
        else:
            Q = a * Q + Q @ Z

        if i < len(coefficients) - 1 and (i + 1) not in GRAM_NEWTON_SCHULZ_RESET_ITERATIONS:
            RZ = a * R + R @ Z
            R = a * RZ + Z @ RZ

    X = X @ Q if tall_skinny else Q @ X
    return X


def zeropower_orthogonalize(G, steps, use_gram_newton_schulz=False):
    """
    Orthogonalize the fast-weight gradients with Newton-Schulz, optionally via the
    Gram variant. Both branches are separately torch.compiled, so this dispatcher
    is traced through and the branch is resolved at trace time.

    The Gram variant is only used for non-square G -- it is slower than the standard
    iteration on square matrices, and the cost model behind that is documented on
    `zeropower_via_gram_newtonschulz5`. Square G therefore falls back to the standard
    iteration, exactly as the reference implementation does.
    """
    if use_gram_newton_schulz and G.size(-2) != G.size(-1):
        return zeropower_via_gram_newtonschulz5(G, steps)
    return zeropower_via_newtonschulz5(G, steps)


@torch.compile(dynamic=True)
def fast_weight_swish_glu_weight_norm_mini_batch_apply(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    ttt_ua_order: list,
    muon_update_steps: int = 0,
    use_gram_newton_schulz: bool = False,
):
    """
    Note:
    Forward:
    (silu(x @ w0) * (x @ w2)) @ w1

    w0, w2: [b, d, dh]
    w1:     [b, dh, d]
    q: [b, l, d]
    k: [b, l, d]
    v: [b, l, d]
    lr0, lr1, lr2: [b, l, 1]
    """
    w0_norm = w0.detach().norm(dim=1, keepdim=True)
    w1_norm = w1.detach().norm(dim=1, keepdim=True)
    w2_norm = w2.detach().norm(dim=1, keepdim=True)

    output = []
    for start, end, update, apply in ttt_ua_order:
        w0_now, w1_now, w2_now = w0, w1, w2
        # all tokens
        if end == -1:
            end = q.shape[1]


        if update:
            ki, vi = k[:, start:end, :], v[:, start:end, :]  # bf16
            lr0i = lr0[:, start:end, :]  # [b, l, d/1] fp32
            lr1i = lr1[:, start:end, :]  # [b, l, d/1] fp32
            lr2i = lr2[:, start:end, :]  # [b, l, d/1] fp32

            gate_before_act = ki @ w0_now       # b[b, l, dh] = [b, l, d] @ [b, d, dh]
            hidden_before_mul = ki @ w2_now     # b[b, l, dh] = [b, l, d] @ [b, d, dh]
            hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

            dhidden = vi @ w1_now.transpose(-1, -2)  # [b, l, dh] = [b, l, d] @ [b, d, dh]
            dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
            dgate = dhidden * hidden_before_mul
            dgate_before_act = silu_backprop(dgate, gate_before_act)

            # [b, dh, l] @ [b, l, d] -> [b, dh, d]
            w1_grad = zeropower_orthogonalize(
                (hidden * lr1i).transpose(-1, -2) @ vi,
                muon_update_steps,
                use_gram_newton_schulz,
            )
            w0_grad = zeropower_orthogonalize(
                (ki * lr0i).transpose(-1, -2) @ dgate_before_act,
                muon_update_steps,
                use_gram_newton_schulz,
            )
            w2_grad = zeropower_orthogonalize(
                (ki * lr2i).transpose(-1, -2) @ dhidden_before_mul,
                muon_update_steps,
                use_gram_newton_schulz,
            )


            w1_now = w1_now + w1_grad
            w0_now = w0_now + w0_grad
            w2_now = w2_now + w2_grad


            # do weight norm here
            w0_now = w0_now / (w0_now.norm(dim=1, keepdim=True) + 1e-5) * w0_norm
            w1_now = w1_now / (w1_now.norm(dim=1, keepdim=True) + 1e-5) * w1_norm
            w2_now = w2_now / (w2_now.norm(dim=1, keepdim=True) + 1e-5) * w2_norm


            w0, w1, w2 = w0_now, w1_now, w2_now

        if apply:
            # Only calculate the output in the last repeat.
            qi = q[:, start:end, :]
            oi = (F.silu(qi @ w0_now, inplace=True) * (qi @ w2_now)) @ w1_now
            output.append(oi)

    output = torch.cat(output, dim=1)

    return output, w0, w1, w2



@torch.compile(dynamic=True)
def bidirectional_lact_swiglu_fused_ckpt(
    w0: torch.Tensor,  # [b, dh, dk]
    w1: torch.Tensor,  # [b, dv, dh]
    w2: torch.Tensor,  # [b, dh, dk]
    q: torch.Tensor,  # [b, l, dk]
    k: torch.Tensor,  # [b, l, dk]
    v: torch.Tensor,  # [b, l, dv]
    lr0: torch.Tensor,  # [b, l, 1]
    lr1: torch.Tensor,  # [b, l, 1]
    lr2: torch.Tensor,  # [b, l, 1]
    ttt_ua_order: list
) -> torch.Tensor:
    """
    Note this function takes flattened k, v and lr.
      by flattent, the batch dimension B is merged into the sequence dimension L.

    The query Q is not flattend.
    """

    BatchSize = q.size(0)
    # adding detach here sometimes improves stability.
    w0_norm = w0.detach().norm(dim=2, keepdim=True)
    w1_norm = w1.detach().norm(dim=2, keepdim=True)
    w2_norm = w2.detach().norm(dim=2, keepdim=True)

    output = []
    for start, end, update, apply in ttt_ua_order:
        # all tokens
        if end == -1:
            end = q.shape[1]

        ######### update the fast weight w0, w1, w2 with test-time training #########
        if update:
            ki, vi = k[:, start:end, :], v[:, start:end, :]  
            lr0i = lr0[:, start:end, :]  
            lr1i = lr1[:, start:end, :]  
            lr2i = lr2[:, start:end, :]
            # make the shape to be (b, 1, l)
            lr0i, lr1i, lr2i = lr0i.reshape(BatchSize, 1, -1), lr1i.reshape(BatchSize, 1, -1), lr2i.reshape(BatchSize, 1, -1)
            # [BatchSize, Hidden, D] for dw0, dw2, [BatchSize, D, Hidden] for dw1
            dw0, dw1, dw2 = lact_swiglu_ffn_fast_weight_grads_with_ckpt(
                w0,
                w1,
                w2,
                ki,
                vi,
                lr0i,
                lr1i,
                lr2i,
            )


            dw0 = zeropower_via_newtonschulz5(dw0, 5)
            dw1 = zeropower_via_newtonschulz5(dw1, 5)
            dw2 = zeropower_via_newtonschulz5(dw2, 5)

            w1 = w1 + dw1
            w0 = w0 + dw0
            w2 = w2 + dw2

            w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
            w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
            w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

        ######### apply the updated fast weights to the query #########
        if apply:
            qi = q[:, start:end, :]
            oi = fused_swiglu_ffn_fwd_with_ckpt(w0, w1, w2, qi)
            output.append(oi)
    output = torch.cat(output, dim=1)

    return output, w0, w1, w2



class FastWeightGluMLPMultihead(nn.Module):
    """
    On init of fast_weight:

    Let's start with the magnitude of the value.
    value_proj is initialized with uniform distribution with range [-1.0/sqrt(d), 1.0/sqrt(d)]
        x is layernormed. So during init, value is unit norm total (not per head, per head is 1.0/sqrt(num_head))
        After silu, value is around norm of 2.7 per head.  (why? seems wired)

    Then for the fast weight, assume initial lr = 0.
    Then with l2_norm of q,k, input is unit normed.
    if w0 is initialized with kaiming, relu(w0 @ q) is unit normed.
    Then w1 is initialized with kaiming, so w1 @ relu(w0 @ q) is of norm sqrt(2) per head
    Since I compute total norm, it is sqrt(2) * sqrt(num_head), which is around 2.7 for dim=512, num_head=4.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int,
        inter_multi: int = 1,
        bias: bool = False,
        base_lr=0.01,
        muon_update_steps=0,
        use_gate_fn = False,
        use_fused_kernels: bool = False,
        use_gram_newton_schulz: bool = False,
    ):
        """
        Args:
            dim: input dimension, which should be the same as the local window attention dim and output dimension
            head_dim: dimension of each head
            inter_multi: the hidden dimension is head_dim * inter_multi
            bias: whether to use bias in linear layers
            base_lr: the base learning rate for the fast weight update
            muon_update_steps: number of steps for muon update
            use_gate_fn: whether to use gate function after the output
            use_gram_newton_schulz: whether to orthogonalize the fast-weight gradients with the
                Gram Newton-Schulz iteration instead of the standard one. Only affects non-square
                gradients; square ones keep using the standard iteration. Ignored when
                use_fused_kernels is True (deprecated path).
        """
        super().__init__()
        self.dim = dim
        assert dim % head_dim == 0
        self.num_heads = dim // head_dim
        self.muon_update_steps = muon_update_steps
        self.use_gram_newton_schulz = use_gram_newton_schulz

        d_in = d_out = head_dim
        d_h = int(head_dim * inter_multi)

        gain = math.sqrt(2)  # for relu activations
        self.w0 = nn.Parameter(
            torch.randn(self.num_heads, d_in, d_h) * gain / math.sqrt(d_in)
        )  # [d_h * num_heads,  d_in]
        self.w1 = nn.Parameter(
            torch.randn(self.num_heads, d_h, d_out) * gain / math.sqrt(d_h)
        )  # [d_in * num_heads,  d_h]
        self.w2 = nn.Parameter(
            torch.randn(self.num_heads, d_in, d_h) * gain / math.sqrt(d_in)
        )  # [d_h * num_heads,  d_in]

        self.to_qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.c_proj = nn.Linear(dim, dim, bias=bias)

        self.lr_dim = self.num_heads
        self.lr_fc = nn.Linear(dim, self.lr_dim * 3)
        self.base_lr_inv = inv_softplus(base_lr)

        self.use_gate_fn = use_gate_fn
        if self.use_gate_fn:
            self.gate_fn = nn.Sequential(
                nn.Linear(dim, dim, bias=bias),
                nn.SiLU()
            )
        self.use_fused_kernels = use_fused_kernels
        self.o_norm = torch.nn.RMSNorm(head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, x: torch.Tensor, info={}, *args):
        """
        x: (b, l, d)
        """
        qkv = F.silu(self.to_qkv(x), inplace=True)  # Silu - Linear
        q, k, v = rearrange(
            qkv, "b l (qkv h d) -> qkv (b h) l d",
            qkv=3, h=self.num_heads
        )
        q = q / (q.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)
        k = k / (k.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)

        lr = self.lr_fc(x)  # [b, l, lr_dim]
        lr = torch.nn.functional.softplus(lr.float() + self.base_lr_inv)

        
        lr0, lr1, lr2 = rearrange(
            lr, "b l (lrs h d) -> lrs (b h) l d",
            lrs=3, h=self.num_heads
        )

        # deprecated
        if self.use_fused_kernels:
            if self.use_gram_newton_schulz:
                # Fail loudly rather than silently training without the option: the
                # fused path hardcodes the standard Newton-Schulz iteration, so an
                # A/B run would come back identical and read as "no effect".
                raise ValueError(
                    "use_gram_newton_schulz is not implemented for the fused-kernel path "
                    "(use_fused_kernels=True), which hardcodes the standard Newton-Schulz "
                    "iteration. Set use_fused_kernels=False to use it."
                )

            if "w0" in info:
                assert "w1" in info and "w2" in info
                w0 = info["w0"]
                w1 = info["w1"]
                w2 = info["w2"]
            else:
                w0 = self.w0.transpose(-1, -2).repeat(x.shape[0], 1, 1)
                w1 = self.w1.transpose(-1, -2).repeat(x.shape[0], 1, 1)
                w2 = self.w2.transpose(-1, -2).repeat(x.shape[0], 1, 1)

            output, w0, w1, w2 = bidirectional_lact_swiglu_fused_ckpt(
                w0, w1, w2, q, k, v, lr0, lr1, lr2, info["ttt_op_order"],
            )      
        else:
            if "w0" in info:
                assert "w1" in info and "w2" in info
                w0 = info["w0"]
                w1 = info["w1"]
                w2 = info["w2"]
            else:
                w0 = self.w0.repeat(x.shape[0], 1, 1)
                w1 = self.w1.repeat(x.shape[0], 1, 1)
                w2 = self.w2.repeat(x.shape[0], 1, 1)
            output, w0, w1, w2 = fast_weight_swish_glu_weight_norm_mini_batch_apply(
                w0, w1, w2, q, k, v, lr0, lr1, lr2, info["ttt_op_order"],
                muon_update_steps=self.muon_update_steps,
                use_gram_newton_schulz=self.use_gram_newton_schulz,
            )


        if self.use_gate_fn:
            output = self.o_norm(output) * self.gate_fn(x)
        else:
            output = self.o_norm(output) 

        output = rearrange(
            output, "(b h) l d -> b l (h d)", h=self.num_heads, b=x.shape[0]
        )

        output = self.c_proj(output)
        return output, {"w0": w0, "w1": w1, "w2": w2}

    def extra_repr(self) -> str:
        return (f"w0 shape: {self.w0.shape}, w1 shape: {self.w1.shape}, w2 shape: {self.w2.shape}, "
                f"Muon update steps: {self.muon_update_steps}, "
                f"Gram Newton-Schulz: {self.use_gram_newton_schulz}, "
                f"Base lr: {math.log(1 + math.exp(self.base_lr_inv))}, ")


