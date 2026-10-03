"""Per-step dump of the served weaver_uzu tree before its prune, plus extra diagnostic trees.

Enabled by --speculative-dflash-tfm-uzu-tree-dump-dir. On every decode step the served tree is
grown a second time eagerly, with the same inputs, to recover what the CUDA graph discards: the
construction slots and the still-active frontier before the final prune. Each shape given in
--speculative-dflash-tfm-uzu-tree-dump-shapes is grown on the same step too, on the same draft,
pool and shared Gumbel field. None of these trees is verified: the verifier keeps reading the
graph's own output, so the generated text is the one the server produces without the dump.

The served tree cannot be dumped by replaying its graph a second time: the replay overwrites the
tree the verifier reads, and the graph key does not include the tree shape. Everything here is
therefore an eager call, like the warm-up the graph capture itself runs.

After verification the step's accept tokens (accepted drafts plus the bonus token) are attached
and the step is written as one .npz file per step under one directory per request, in request
order. The realised continuation of a request is the concatenation of its accept tokens, which is
what an offline analysis covers each dumped tree with.

With --speculative-dflash-tfm-uzu-tree-dump-internals the dump is the training record of a model of
the tree's acceptance, and the served tree is built once: eagerly, with the Weaver internals, and
that very tree goes to the verifier (build_served). There is no second build to compare against,
so the dumped tree is the verified tree by construction. Each step is written in its final form
(see _tree_record and _target_record): the pre-prune tree S with its builder scores and the kept
nodes, the candidate pool of every depth with the drafter's log-normaliser over the vocabulary
and the shared Gumbel field, the Weaver residual logits and read-out state of every expanded
node, the drafter's and the target's hidden states, and at every verified node the target's top
logits, its log-normaliser, its filtered sampling distribution on those top tokens and the token
the shared field makes it choose. bf16 tensors are kept as their raw 16-bit patterns (uint16),
since numpy has no bfloat16.

Diagnostic only: one request at a time (--max-running-requests 1), every step pays several extra
tree builds and a device-to-host copy.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import msgspec
import numpy as np
import torch
import triton
import triton.language as tl

from sglang.srt.speculative.dflash_tfm import WeaverTree
from sglang.srt.speculative.shared_gumbel import filtered_probs, gumbel_at, sample_probs
from sglang.srt.speculative.uzu_tree import (
    UzuConstructedTree,
    UzuTreeInternals,
    UzuTreeShape,
    check_uzu_tree_shape,
    construct_uzu_tree,
)

# Target logits kept per verified node. The request filters of these runs (top-k 20 inside a
# top-p nucleus) never reach past this, so the filtered target distribution is recoverable.
TARGET_TOP_K = 64

# Narrowest dtype that holds each field exactly; tokens and slot indices fit in int32.
_FIELD_DTYPES = {
    "tokens": np.int32,
    "parents": np.int32,
    "depths": np.uint8,
    "mask": np.bool_,
    "active": np.bool_,
    "logprobs": np.float32,
    "scores": np.float32,
    "prune_scores": np.float32,
}


class _PendingStep(msgspec.Struct):
    rid: str
    arrays: dict[str, np.ndarray]


def _to_numpy(tensor: torch.Tensor, field: str) -> np.ndarray:
    # Longest suffix first, so "prune_scores" is not taken for "scores".
    suffix = next(
        key for key in sorted(_FIELD_DTYPES, key=len, reverse=True) if field.endswith(key)
    )
    return tensor.detach().to("cpu").numpy().astype(_FIELD_DTYPES[suffix], copy=False)


def _constructed_arrays(prefix: str, tree: UzuConstructedTree) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_{field}": _to_numpy(getattr(tree, field)[0], field)
        for field in UzuConstructedTree.__struct_fields__
    }


def _pruned_arrays(prefix: str, tree: WeaverTree) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_tokens": _to_numpy(tree.draft_tokens[0], "tokens"),
        f"{prefix}_parents": _to_numpy(tree.parent_indices[0], "parents"),
        f"{prefix}_depths": _to_numpy(tree.depths[0], "depths"),
        f"{prefix}_mask": _to_numpy(tree.node_mask[0], "mask"),
    }


def _same_tree(left: WeaverTree, right: WeaverTree) -> bool:
    return bool(
        torch.equal(left.draft_tokens, right.draft_tokens)
        and torch.equal(left.parent_indices, right.parent_indices)
        and torch.equal(left.node_mask, right.node_mask)
    )


def _finite_or_none(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(value) else value


def _raw(tensor: torch.Tensor) -> np.ndarray:
    """A float tensor on the host, unrounded: bf16 as its bit pattern, anything else as fp32."""
    tensor = tensor.detach()
    if tensor.dtype == torch.bfloat16:
        return tensor.contiguous().view(torch.int16).to("cpu").numpy().view(np.uint16)
    return tensor.to("cpu", torch.float32).numpy()


@triton.jit
def _pool_gumbel_kernel(
    noise_ptr, tokens_ptr, seeds_ptr, positions_ptr, POOL: tl.constexpr, BLOCK: tl.constexpr
):
    # One program per pool row d of request 0. The children of a node at depth d come from pool
    # row d and draw the field at the node's position, the root position plus d, exactly as the
    # publish kernel does; slots without a token read token -1 there too.
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < POOL
    tokens = tl.load(tokens_ptr + row * POOL + offsets, mask=mask, other=-1)
    noise = gumbel_at(tl.load(seeds_ptr), tl.load(positions_ptr) + row, tokens)
    tl.store(noise_ptr + row * POOL + offsets, noise, mask=mask)


def _pool_gumbel(worker, candidate_ids: torch.Tensor) -> torch.Tensor:
    """The shared Gumbel field on the candidate pool of every depth, as the tree kernel saw it."""
    seeds, positions = worker._gumbel_tree_inputs[1][:2]
    rows = candidate_ids[0].contiguous()
    noise = torch.empty(rows.shape, dtype=torch.float32, device=rows.device)
    _pool_gumbel_kernel[(rows.shape[0],)](
        noise, rows, seeds, positions, POOL=int(rows.shape[1]),
        BLOCK=int(triton.next_power_of_2(rows.shape[1])),
    )
    return noise


def _draft_vocab_logits(lm_head, hidden_states: torch.Tensor) -> torch.Tensor:
    """The drafter's logits over the whole vocabulary: the products _topk_from_lm_head takes."""
    shard = lm_head.shard_indices
    weight = lm_head.weight
    num_org = int(shard.num_org_elements)
    num_org_padded = int(shard.num_org_elements_padded)
    num_added = int(shard.num_added_elements)
    hs = hidden_states.to(dtype=weight.dtype)
    parts = []
    if num_org > 0:
        parts.append(torch.matmul(hs, weight[:num_org].T).float())
    if num_added > 0:
        added = weight[num_org_padded : num_org_padded + num_added]
        parts.append(torch.matmul(hs, added.T).float())
    return torch.cat(parts, dim=-1)


NOT_EXPANDED = 255


def _bf16_words(tensor: torch.Tensor, field: str) -> np.ndarray:
    """An fp32 tensor that holds bf16 values (a bf16 matmul upcast), as raw bf16 words; exact or an error."""
    narrow = tensor.to(torch.bfloat16)
    if not torch.equal(narrow.float(), tensor.float()):
        raise RuntimeError(f"The uzu tree dump expected {field} to hold bf16 values exactly.")
    return _raw(narrow)


def _tree_record(
    worker,
    inputs: dict[str, torch.Tensor],
    tree: UzuConstructedTree,
    sources: torch.Tensor,
    internals: UzuTreeInternals,
    shape: UzuTreeShape,
    max_depth: int,
) -> dict[str, np.ndarray]:
    """The step's tree and pools in training form, for request 0.

    S is the construction slots that hold a node followed by the active frontier, in that source
    order (topological). Per node of S: its parent in S (-1 at the root), its depth, the index of
    its token in the pool row of its parent's depth (row = depth - 1), its expansion round (slot 0
    is the root, then each round fills parents_per_round slots; frontier nodes are not expanded)
    and the builder's scores. ``verified`` lists the kept nodes in verification order (-1 for an
    empty verifier slot). Expanded nodes, which are the slots above the depth cap, carry the
    Weaver residual logits and read-out state, in S order.
    """
    candidate_ids = inputs["candidate_ids"][0]
    present = torch.cat((tree.node_mask[0], tree.frontier_active[0]))
    num_slots = int(tree.node_mask.shape[1])
    s_sources = torch.nonzero(present).squeeze(1)
    s_of_source = torch.full((present.numel(),), -1, dtype=torch.long, device=present.device)
    s_of_source[s_sources] = torch.arange(s_sources.numel(), device=present.device)
    take = lambda node, frontier: torch.cat((node[0], frontier[0]))[s_sources]
    parent_sources = take(tree.node_parents, tree.frontier_parents)
    parents = torch.where(parent_sources >= 0, s_of_source[parent_sources.clamp_min(0)], -1)
    depths = take(tree.node_depths, tree.frontier_depths).long()
    tokens = take(tree.node_tokens, tree.frontier_tokens).long()
    hits = candidate_ids[depths[1:] - 1] == tokens[1:, None]
    if not torch.all(hits.sum(dim=1) == 1):
        raise RuntimeError("The uzu tree dump found a node token not exactly once in its pool row.")
    pool_index = torch.cat((torch.zeros(1, dtype=torch.long, device=hits.device), hits.int().argmax(dim=1)))
    width = int(shape.parents_per_round)
    rounds = torch.where(
        s_sources < num_slots,
        torch.where(s_sources == 0, 0, 1 + (s_sources - 1).clamp_min(0) // width),
        NOT_EXPANDED,
    )
    kept = sources[0]
    verified = torch.where(present[kept], s_of_source[kept], -1)
    expanded_slots = s_sources[(s_sources < num_slots) & (depths < max_depth)]
    _, lm_head = worker._target_embedding_and_lm_head()
    proposal_features = inputs["proposal_features"]
    depth = int(candidate_ids.shape[0])
    vocab_logits = _draft_vocab_logits(
        lm_head, proposal_features[0].reshape(depth, proposal_features.shape[-1])
    )
    return {
        "root_token": tokens[0].to("cpu", torch.int32).numpy(),
        "parent": parents.to("cpu", torch.int16).numpy(),
        "depth": depths.to("cpu", torch.uint8).numpy(),
        "pool_index": pool_index.to("cpu", torch.int16).numpy(),
        "round": rounds.to("cpu", torch.uint8).numpy(),
        "edge_logprob": _raw(take(tree.node_logprobs, tree.frontier_logprobs).float()),
        "priority": _raw(take(tree.node_scores, tree.frontier_scores).float()),
        "prune_score": _raw(take(tree.node_prune_scores, tree.frontier_prune_scores).float()),
        "verified": verified.to("cpu", torch.int16).numpy(),
        "pool_tokens": candidate_ids.to("cpu", torch.int32).numpy(),
        "pool_logits": _bf16_words(inputs["candidate_scores"][0], "the drafter's pool logits"),
        "pool_log_norms": _raw(torch.logsumexp(vocab_logits, dim=-1)),
        "pool_gumbels": _raw(_pool_gumbel(worker, inputs["candidate_ids"])),
        "rho": _bf16_words(internals.slot_residuals[0][expanded_slots], "the Weaver residual logits"),
        "state": _raw(internals.slot_states[0][expanded_slots]),
        "draft_h": _raw(proposal_features[0]),
        "target_h": _raw(inputs["output_norm"][0]),
    }


def _target_record(
    logits: torch.Tensor, sampling_info, positions: torch.Tensor, num_nodes: int
) -> dict[str, np.ndarray]:
    """The target at every verified node: its top logits and log-normaliser, its filtered sampling
    distribution on those top tokens and its shared-Gumbel choice.

    The filtered distribution is the very filtered_probs target sampling runs, read off on the top
    tokens; the request filters of these runs (top-k inside a top-p nucleus) keep it inside them,
    and a step where they do not is an error rather than a silently truncated record.
    """
    logits = logits.view(-1, logits.shape[-1])[:num_nodes]
    top = torch.topk(logits.float(), TARGET_TOP_K, dim=-1)
    probs = filtered_probs(logits, sampling_info, repeats=num_nodes)
    seeds = sampling_info.sampling_seed.repeat_interleave(num_nodes)
    choices = sample_probs(probs, seeds, positions.reshape(-1)[:num_nodes])
    filtered = probs.gather(1, top.indices)
    if float((1.0 - filtered.sum(dim=-1)).abs().max()) > 1e-5:
        raise RuntimeError(
            f"The target's filtered distribution leaves its top {TARGET_TOP_K} tokens; raise TARGET_TOP_K."
        )
    return {
        "target_top_tokens": top.indices.to("cpu", torch.int32).numpy(),
        "target_top_logits": _raw(top.values),
        "target_log_norm": _raw(torch.logsumexp(logits.float(), dim=-1)),
        "target_filtered": _raw(filtered.float()),
        "target_choice": choices.to("cpu", torch.int32).numpy(),
    }


class UzuTreeDumper:
    def __init__(
        self, worker, dump_dir: str, shapes_spec: str | None, internals: bool = False
    ) -> None:
        self.internals = internals
        self.dump_dir = Path(dump_dir)
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        self.served_shape = UzuTreeShape.from_worker(worker)
        self.extra_shapes = [
            UzuTreeShape.parse(spec) for spec in (shapes_spec or "").split(",") if spec.strip()
        ]
        for shape in self.extra_shapes:
            check_uzu_tree_shape(shape, int(worker.candidate_pool_size), int(worker.tree_budget))
        self.final_num_nodes = int(worker.tree_budget) + 1
        self.meta = {
            "served_shape": str(self.served_shape),
            "extra_shapes": [str(shape) for shape in self.extra_shapes],
            "tree_budget": int(worker.tree_budget),
            "tree_max_depth": int(worker.tree_max_depth),
            "candidate_pool_size": int(worker.candidate_pool_size),
            "weight_sigma_inv": float(worker.gumbel_weight_sigma_inv),
            "prune_sigma_inv": worker.gumbel_prune_sigma_inv,
            "draft_temperature": _finite_or_none(worker.draft_temperature),
            "weight_temperature": _finite_or_none(worker.weight_temperature),
            "draft_top_k": int(worker.draft_top_k),
            "draft_top_p": float(worker.draft_top_p),
            "draft_min_p": float(worker.draft_min_p),
            "internals": internals,
        }
        if internals:
            self.meta["target_top_k"] = TARGET_TOP_K
        self._request_dirs: dict[str, Path] = {}
        self._step_counts: dict[str, int] = {}
        self._pending: _PendingStep | None = None

    def record_build(
        self,
        worker,
        *,
        rid: str,
        root_position: int,
        served_tree: WeaverTree,
        inputs: dict[str, torch.Tensor],
    ) -> None:
        """Without internals: grow the served shape and every extra shape on this step's inputs
        next to the graph-built served tree, and hold the step until verify."""
        if self.internals:
            raise RuntimeError("With internals the served tree comes from build_served.")
        if int(served_tree.draft_tokens.shape[0]) != 1:
            raise RuntimeError("The uzu tree dump supports one request per step only.")
        if self._pending is not None:
            raise RuntimeError(
                "The uzu tree dump saw two tree builds without a verify between them."
            )
        with torch.inference_mode():
            served = construct_uzu_tree(worker, **inputs, shape=self.served_shape)
            arrays = _constructed_arrays("served", served)
            arrays |= _pruned_arrays("served_pruned", served_tree)
            # The eager rebuild must prune to exactly the tree the graph handed to the verifier,
            # otherwise the dumped pre-prune tree is not the one this step was served from.
            arrays["served_match"] = np.array(
                _same_tree(served.prune(self.final_num_nodes), served_tree)
            )
            for index, shape in enumerate(self.extra_shapes):
                extra = construct_uzu_tree(worker, **inputs, shape=shape)
                arrays |= _constructed_arrays(f"extra{index}", extra)
        arrays["root_position"] = np.array(root_position, dtype=np.int64)
        self._pending = _PendingStep(rid=rid, arrays=arrays)

    def build_served(
        self, worker, *, rid: str, root_position: int, inputs: dict[str, torch.Tensor]
    ) -> WeaverTree:
        """With internals: build this step's served tree once, eagerly with the Weaver internals,
        record it in training form and return it for verification."""
        if not self.internals:
            raise RuntimeError("build_served is the internals dump; use record_build without it.")
        if int(inputs["candidate_ids"].shape[0]) != 1:
            raise RuntimeError("The uzu tree dump supports one request per step only.")
        if self._pending is not None:
            raise RuntimeError(
                "The uzu tree dump saw two tree builds without a verify between them."
            )
        with torch.inference_mode():
            candidate_ids = inputs["candidate_ids"]
            internals = UzuTreeInternals.empty(
                1,
                self.served_shape.num_slots,
                int(worker.weaver.d_rank),
                int(candidate_ids.shape[2]),
                device=candidate_ids.device,
                dtype=inputs["candidate_weights"].dtype,
            )
            constructed = construct_uzu_tree(
                worker, **inputs, shape=self.served_shape, internals=internals
            )
            tree, sources = constructed.prune_with_sources(self.final_num_nodes)
            arrays = _tree_record(
                worker,
                inputs,
                constructed,
                sources,
                internals,
                self.served_shape,
                int(worker.tree_max_depth),
            )
        arrays["root_position"] = np.array(root_position, dtype=np.int64)
        self._pending = _PendingStep(rid=rid, arrays=arrays)
        return tree

    def record_accept(
        self,
        accept_tokens: torch.Tensor,
        *,
        logits: torch.Tensor | None = None,
        sampling_info=None,
        positions: torch.Tensor | None = None,
    ) -> None:
        """Attach this step's accept tokens (drafts plus bonus) and write the step.

        With internals the verifier's logits, sampling info and node positions are needed too.
        """
        pending = self._pending
        if pending is None:
            raise RuntimeError("The uzu tree dump saw a verify without a recorded tree build.")
        self._pending = None
        pending.arrays["accept_tokens"] = _to_numpy(accept_tokens, "tokens")
        if self.internals:
            if logits is None or sampling_info is None or positions is None:
                raise RuntimeError(
                    "The uzu tree dump with internals needs the verifier's logits, "
                    "sampling info and node positions."
                )
            with torch.inference_mode():
                pending.arrays |= _target_record(
                    logits, sampling_info, positions, self.final_num_nodes
                )
        request_dir = self._request_dir(pending.rid)
        step = self._step_counts[pending.rid]
        self._step_counts[pending.rid] = step + 1
        np.savez(request_dir / f"{step:06d}.npz", **pending.arrays)

    def _request_dir(self, rid: str) -> Path:
        if rid not in self._request_dirs:
            request_dir = self.dump_dir / f"{len(self._request_dirs):03d}_{rid}"
            request_dir.mkdir(parents=True, exist_ok=False)
            (request_dir / "meta.json").write_text(
                json.dumps({"rid": rid, "ordinal": len(self._request_dirs), **self.meta}, indent=2)
                + "\n"
            )
            self._request_dirs[rid] = request_dir
            self._step_counts[rid] = 0
        return self._request_dirs[rid]
