"""CUDA DDTree best-first builder.

Rule: liranringel/ddtree c96427a185677bf4133ed865dd1626a5041aef9b,
DDTree (MIT), build_ddtree_tree. Adapted from our validated 2026-09-07 GPU port.

Use full-vocabulary normalized draft log probabilities, temperature 1, and the
official lexicographic rank-path tie rule. Budget excludes the root.
"""
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass
class Tree:
    draft_tokens: torch.Tensor
    parent_indices: torch.Tensor
    depths: torch.Tensor
    node_mask: torch.Tensor
    draft_logprobs: torch.Tensor


@triton.jit
def _build(Root, Ids, Logp, Tok, Par, Dep, Mask, D: tl.constexpr,
           K: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    batch = tl.program_id(0)
    x = tl.arange(0, BLOCK)
    # Each pop replaces its sibling and publishes at most one child. At most
    # N frontier slots are needed. FP64 matches the Python heap score arithmetic.
    first = tl.load(Logp + batch * D * K).to(tl.float64)
    score = tl.where(x == 0, first, -float('inf'))
    parent = tl.full((BLOCK,), 0, tl.int32)
    depth = tl.full((BLOCK,), 1, tl.int32)
    rank = tl.full((BLOCK,), 0, tl.int32)
    # Preserve the original base-128 path through budget 64. Budget 128 needs
    # base-256 digits and unsigned keys (the first digit can set bit 63).
    BITS: tl.constexpr = 8 if K > 64 else 7
    KEY: tl.constexpr = tl.uint64 if K > 64 else tl.int64
    KEY_MAX: tl.constexpr = 18446744073709551615 if K > 64 else 9223372036854775807
    hi = tl.where(x == 0, 1 << (BITS * 7), 0).to(KEY)
    lo = tl.full((BLOCK,), 0, KEY)
    root = tl.load(Root + batch)
    tl.store(Tok + batch * N, root)
    tl.store(Par + batch * N, -1)
    tl.store(Dep + batch * N, 0)
    tl.store(Mask + batch * N, 1)
    for node in range(1, N):
        best = tl.max(score, 0)
        best_hi = tl.min(tl.where(score == best, hi, tl.full((), KEY_MAX, KEY)), 0)
        best_lo = tl.min(tl.where((score == best) & (hi == best_hi), lo, tl.full((), KEY_MAX, KEY)), 0)
        winner = tl.min(tl.where((score == best) & (hi == best_hi) & (lo == best_lo), x, BLOCK), 0)
        selected = x == winner
        p = tl.sum(tl.where(selected, parent, 0), 0)
        d = tl.sum(tl.where(selected, depth, 0), 0)
        r = tl.sum(tl.where(selected, rank, 0), 0)
        valid = best != -float('inf')
        token = tl.load(Ids + (batch * D + d - 1) * K + r, mask=valid, other=0)
        tl.store(Tok + batch * N + node, token)
        tl.store(Par + batch * N + node, tl.where(valid, p, -1))
        tl.store(Dep + batch * N + node, tl.where(valid, d, 0))
        tl.store(Mask + batch * N + node, valid)
        old = tl.load(Logp + (batch * D + d - 1) * K + r, mask=valid, other=0).to(tl.float64)
        sibling = tl.load(Logp + (batch * D + d - 1) * K + r + 1,
                          mask=valid & (r + 1 < K), other=-float('inf')).to(tl.float64)
        sibling_score = (best - old) + sibling
        score = tl.where(selected, sibling_score, score)
        rank = tl.where(selected, r + 1, rank)
        hi = tl.where(selected & (d <= 8), best_hi + (tl.full((), 1, KEY) << (BITS * (8 - d))), hi)
        lo = tl.where(selected & (d > 8), best_lo + (tl.full((), 1, KEY) << (BITS * (15 - d))), lo)
        child_logp = tl.load(Logp + (batch * D + d) * K,
                            mask=valid & (d < D), other=-float('inf')).to(tl.float64)
        child = x == node
        score = tl.where(child, best + child_logp, score)
        parent = tl.where(child, node, parent)
        depth = tl.where(child, d + 1, depth)
        rank = tl.where(child, 0, rank)
        child_hi = best_hi + tl.where(d + 1 <= 8, (tl.full((), 1, KEY) << (BITS * (7 - d))), 0)
        child_lo = best_lo + tl.where(d + 1 > 8, (tl.full((), 1, KEY) << (BITS * (14 - d))), 0)
        hi = tl.where(child, child_hi, hi)
        lo = tl.where(child, child_lo, lo)


def build_ddtree_gpu(*, root_ids, top_token_ids, top_log_probs, budget):
    b, d, k = top_token_ids.shape
    assert 0 < d <= 15 and 0 < k <= 128 and 0 < budget <= 128
    n = budget + 1
    tok = torch.empty((b, n), dtype=torch.int64, device=root_ids.device)
    par = torch.empty_like(tok)
    dep = torch.empty_like(tok)
    mask = torch.empty((b, n), dtype=torch.bool, device=root_ids.device)
    logp = torch.zeros((b, n), dtype=torch.float32, device=root_ids.device)
    _build[(b,)](root_ids, top_token_ids.contiguous(), top_log_probs.contiguous(),
                 tok, par, dep, mask, d, k, n, triton.next_power_of_2(n),
                 num_warps=4, enable_fp_fusion=False)
    return Tree(tok, par, dep, mask, logp)
