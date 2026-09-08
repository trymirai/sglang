from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TreeSample:
    accepted_nodes: tuple[int, ...]
    bonus_token: int


def _normalize(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.to(dtype=torch.float64)
    total = probs.sum()
    assert bool(torch.isfinite(total)) and float(total) > 0.0
    return probs / total


def _sample_categorical(probs: torch.Tensor, uniform: float) -> int:
    probs = _normalize(probs)
    assert 0.0 <= uniform < 1.0
    threshold = uniform * float(probs.sum())
    cumulative = torch.cumsum(probs, dim=0)
    index = int(torch.searchsorted(cumulative, threshold, right=True).item())
    return min(index, probs.numel() - 1)


def filter_target_probs(
    probs: torch.Tensor,
    *,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    sequential: bool,
) -> torch.Tensor:
    """Apply the target sampler's top-k, top-p, and min-p filters."""
    assert probs.ndim == 2
    rows = probs.shape[0]
    assert top_ks.shape == top_ps.shape == min_ps.shape == (rows,)

    if sequential:
        ranks = torch.arange(probs.shape[1], device=probs.device)[None, :]
        sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
        sorted_probs = torch.where(ranks < top_ks[:, None], sorted_probs, 0.0)
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
        cumulative_before = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
        sorted_probs = torch.where(
            cumulative_before <= top_ps[:, None], sorted_probs, 0.0
        )
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    else:
        sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
        ranks = torch.arange(probs.shape[1], device=probs.device)[None, :]
        cumulative_before = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
        keep = (ranks < top_ks[:, None]) & (cumulative_before <= top_ps[:, None])
        sorted_probs = torch.where(keep, sorted_probs, 0.0)

    min_threshold = sorted_probs[:, :1] * min_ps[:, None]
    sorted_probs = torch.where(sorted_probs >= min_threshold, sorted_probs, 0.0)
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    return torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)


def plackett_luce_order(probs: torch.Tensor, uniforms: torch.Tensor) -> torch.Tensor:
    """Sample a complete weighted-without-replacement order."""
    probs = _normalize(probs)
    assert probs.ndim == 1
    assert uniforms.shape == probs.shape
    assert bool(torch.all((uniforms > 0.0) & (uniforms < 1.0)))
    keys = torch.log(probs) - torch.log(-torch.log(uniforms.to(torch.float64)))
    return torch.argsort(keys, descending=True)


def target_only_tree_verify(
    *,
    tokens: torch.Tensor,
    parents: torch.Tensor,
    target_probs: torch.Tensor,
    uniforms: torch.Tensor,
) -> TreeSample:
    """Verify a deterministic candidate tree by sampling from the target first."""
    num_nodes = int(tokens.shape[0])
    assert tokens.shape == parents.shape == (num_nodes,)
    assert target_probs.shape[0] == num_nodes
    assert uniforms.shape == (num_nodes,)
    assert int(parents[0]) == -1

    accepted = [0]
    current = 0
    while True:
        target_token = _sample_categorical(
            target_probs[current], float(uniforms[current])
        )
        children = torch.nonzero(parents == current, as_tuple=False).flatten()
        matching = children[tokens[children] == target_token]
        if matching.numel() == 0:
            return TreeSample(tuple(accepted), target_token)
        assert matching.numel() == 1
        current = int(matching[0])
        accepted.append(current)


def traversal_tree_verify(
    *,
    tokens: torch.Tensor,
    parents: torch.Tensor,
    target_probs: torch.Tensor,
    draft_logprobs: torch.Tensor,
    sibling_keys: torch.Tensor,
    accept_uniforms: torch.Tensor,
    bonus_uniform: float,
) -> TreeSample:
    """Literal CPU implementation of Traversal Verification Algorithm 3.

    Each parent's children must be the complete support of its proposal
    distribution. ``sibling_keys`` fixes the sampled without-replacement order.
    """
    num_nodes = int(tokens.shape[0])
    vocab_size = int(target_probs.shape[1])
    assert tokens.shape == parents.shape == draft_logprobs.shape == sibling_keys.shape
    assert tokens.shape == (num_nodes,)
    assert target_probs.shape == (num_nodes, vocab_size)
    assert accept_uniforms.shape == (num_nodes,)
    assert int(parents[0]) == -1

    active = torch.ones(num_nodes, dtype=torch.bool)
    target = [_normalize(row) for row in target_probs]
    draft = [torch.zeros(vocab_size, dtype=torch.float64) for _ in range(num_nodes)]

    for parent in range(num_nodes):
        children = torch.nonzero(parents == parent, as_tuple=False).flatten()
        if children.numel() == 0:
            continue
        probs = torch.softmax(draft_logprobs[children].to(torch.float64), dim=0)
        child_tokens = tokens[children].to(torch.long)
        assert child_tokens.unique().numel() == child_tokens.numel()
        draft[parent][child_tokens] = probs

    acceptance = torch.zeros(num_nodes, dtype=torch.float64)
    acceptance[0] = 1.0

    def update_descendants(parent: int) -> None:
        children = torch.nonzero(active & (parents == parent), as_tuple=False).flatten()
        for child_tensor in children:
            child = int(child_tensor)
            token = int(tokens[child])
            q = float(draft[parent][token])
            assert q > 0.0
            acceptance[child] = min(
                float(acceptance[parent]) * float(target[parent][token]) / q,
                1.0,
            )
            update_descendants(child)

    update_descendants(0)

    for verify_step in range(num_nodes):
        current = 0
        while True:
            children = torch.nonzero(
                active & (parents == current), as_tuple=False
            ).flatten()
            if children.numel() == 0:
                leaf = current
                break
            current = int(children[torch.argmax(sibling_keys[children])])

        if leaf == 0 or float(accept_uniforms[verify_step]) < float(acceptance[leaf]):
            path = []
            current = leaf
            while current >= 0:
                path.append(current)
                current = int(parents[current])
            path.reverse()
            bonus = _sample_categorical(target[leaf], bonus_uniform)
            return TreeSample(tuple(path), bonus)

        parent = int(parents[leaf])
        parent_acceptance = float(acceptance[parent])
        residual = torch.clamp(
            parent_acceptance * target[parent] - draft[parent], min=0.0
        )
        residual_mass = float(residual.sum())
        target[parent] = (
            residual / residual_mass
            if residual_mass > 0.0
            else torch.zeros_like(residual)
        )

        rejected_token = int(tokens[leaf])
        draft[parent][rejected_token] = 0.0
        remaining_mass = float(draft[parent].sum())
        if remaining_mass > 0.0:
            draft[parent] /= remaining_mass

        denominator = residual_mass + 1.0 - parent_acceptance
        acceptance[parent] = residual_mass / denominator if denominator > 0.0 else 0.0
        active[leaf] = False
        update_descendants(parent)

    raise AssertionError("Traversal verification exhausted a non-empty tree")
