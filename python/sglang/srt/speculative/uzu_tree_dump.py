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

With --speculative-dflash-tfm-uzu-tree-dump-graph the internals dump is fast enough to collect
with: the served tree with its internals, the pool Gumbel field and the drafter's log-normaliser
are one CUDA graph, its outputs are copied without blocking into pinned host buffers, and a
background thread turns them into the very same step record (see _host_tree_record) and writes
it. The decode loop never waits for the host. A step file appears under its final name only once
it is complete, possibly a little after the request returns; the reader waits for the count of
steps the server reports. --speculative-dflash-tfm-uzu-tree-dump-graph-check also builds the eager
record on every step and fails on any difference.
"""

from __future__ import annotations

import json
import math
import os
import queue
import threading
from pathlib import Path

import msgspec
import numpy as np
import torch

from sglang.srt.speculative.dflash_tfm import WeaverTree
from sglang.srt.speculative.shared_gumbel import filtered_probs, sample_probs
from sglang.srt.speculative.uzu_tree import (
    UzuConstructedTree,
    UzuTreeInternals,
    UzuTreeShape,
    check_uzu_tree_shape,
    construct_uzu_tree,
    pool_gumbels,
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
        "prune_score": _raw(
            take(tree.node_prune_scores, tree.frontier_prune_scores).float()
        ),
        "verified": verified.to("cpu", torch.int16).numpy(),
        "pool_tokens": candidate_ids.to("cpu", torch.int32).numpy(),
        "pool_logits": _bf16_words(
            inputs["candidate_scores"][0], "the drafter's pool logits"
        ),
        "pool_log_norms": _raw(torch.logsumexp(vocab_logits, dim=-1)),
        "pool_gumbels": _raw(pool_gumbels(worker, inputs["candidate_ids"])[0]),
        "rho": _bf16_words(
            internals.slot_residuals[0][expanded_slots], "the Weaver residual logits"
        ),
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


# Steps recorded but not yet written: the decode loop waits for the writer only this far behind.
GRAPH_SLOTS = 4
DEVIATION_LIMIT = 1e-5


class _GraphTree(msgspec.Struct):
    """The served tree with its internals as one CUDA graph: static inputs and outputs."""

    graph: torch.cuda.CUDAGraph
    inputs: dict[str, torch.Tensor]
    constructed: UzuConstructedTree
    tree: WeaverTree
    sources: torch.Tensor
    internals: UzuTreeInternals
    gumbels: torch.Tensor
    log_norms: torch.Tensor


class _GraphItem(msgspec.Struct):
    """One step on its way to the writer: its pinned slot and the event its copies end at."""

    slot: int
    event: torch.cuda.Event
    rid: str
    root_position: int
    num_accept: int


def _tree_device_fields(state: _GraphTree) -> dict[str, torch.Tensor]:
    """The device tensors a step record is built from, request 0."""
    fields = {
        f"node_{name}": getattr(state.constructed, f"node_{name}")[0]
        for name in (
            "tokens",
            "parents",
            "depths",
            "mask",
            "logprobs",
            "scores",
            "prune_scores",
        )
    }
    fields |= {
        f"frontier_{name}": getattr(state.constructed, f"frontier_{name}")[0]
        for name in (
            "tokens",
            "parents",
            "depths",
            "active",
            "logprobs",
            "scores",
            "prune_scores",
        )
    }
    return fields | {
        "sources": state.sources[0],
        "candidate_ids": state.inputs["candidate_ids"][0],
        "candidate_scores": state.inputs["candidate_scores"][0],
        "proposal_features": state.inputs["proposal_features"][0],
        "output_norm": state.inputs["output_norm"][0],
        "slot_residuals": state.internals.slot_residuals[0],
        "slot_states": state.internals.slot_states[0],
        "gumbels": state.gumbels[0],
        "log_norms": state.log_norms,
    }


def _target_device_fields(
    logits: torch.Tensor, sampling_info, positions: torch.Tensor, num_nodes: int
) -> dict[str, torch.Tensor]:
    """The target's record at every verified node, on the device (see _target_record)."""
    logits = logits.view(-1, logits.shape[-1])[:num_nodes]
    top = torch.topk(logits.float(), TARGET_TOP_K, dim=-1)
    probs = filtered_probs(logits, sampling_info, repeats=num_nodes)
    seeds = sampling_info.sampling_seed.repeat_interleave(num_nodes)
    choices = sample_probs(probs, seeds, positions.reshape(-1)[:num_nodes])
    filtered = probs.gather(1, top.indices)
    return {
        "top_indices": top.indices,
        "top_values": top.values,
        "log_norm": torch.logsumexp(logits.float(), dim=-1),
        "filtered": filtered.float(),
        "choices": choices,
        "deviation": (1.0 - filtered.sum(dim=-1)).abs().max(),
    }


def _host(tensor: torch.Tensor) -> np.ndarray:
    """A pinned host tensor as an owned numpy array; bf16 as its raw 16-bit patterns."""
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.int16).numpy().view(np.uint16).copy()
    return tensor.numpy().copy()


def _host_bf16_words(values: np.ndarray, field: str) -> np.ndarray:
    """fp32 values that must be bf16 exactly, as raw bf16 words (as _bf16_words on the device)."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    if np.isnan(values).any() or np.any(bits & np.uint32(0xFFFF)):
        raise RuntimeError(
            f"The uzu tree dump expected {field} to hold bf16 values exactly."
        )
    return (bits >> np.uint32(16)).astype(np.uint16)


def _host_tree_record(
    host: dict[str, np.ndarray], shape: UzuTreeShape, max_depth: int
) -> dict[str, np.ndarray]:
    """_tree_record on host copies of the graph's outputs: the same arrays, bit for bit."""
    num_slots = int(host["node_mask"].shape[0])
    present = np.concatenate((host["node_mask"], host["frontier_active"]))
    s_sources = np.flatnonzero(present)
    s_of_source = np.full(present.size, -1, dtype=np.int64)
    s_of_source[s_sources] = np.arange(s_sources.size)
    take = lambda name: np.concatenate(
        (host[f"node_{name}"], host[f"frontier_{name}"])
    )[s_sources]
    parent_sources = take("parents")
    parents = np.where(
        parent_sources >= 0, s_of_source[np.maximum(parent_sources, 0)], -1
    )
    depths = take("depths").astype(np.int64)
    tokens = take("tokens").astype(np.int64)
    candidate_ids = host["candidate_ids"]
    hits = candidate_ids[depths[1:] - 1] == tokens[1:, None]
    if not np.all(hits.sum(axis=1) == 1):
        raise RuntimeError(
            "The uzu tree dump found a node token not exactly once in its pool row."
        )
    pool_index = np.concatenate(([0], hits.argmax(axis=1)))
    width = int(shape.parents_per_round)
    rounds = np.where(
        s_sources < num_slots,
        np.where(s_sources == 0, 0, 1 + np.maximum(s_sources - 1, 0) // width),
        NOT_EXPANDED,
    )
    kept = host["sources"]
    verified = np.where(present[kept], s_of_source[kept], -1)
    expanded_slots = s_sources[(s_sources < num_slots) & (depths < max_depth)]
    return {
        "root_token": np.asarray(tokens[0], dtype=np.int32),
        "parent": parents.astype(np.int16),
        "depth": depths.astype(np.uint8),
        "pool_index": pool_index.astype(np.int16),
        "round": rounds.astype(np.uint8),
        "edge_logprob": take("logprobs").astype(np.float32),
        "priority": take("scores").astype(np.float32),
        "prune_score": take("prune_scores").astype(np.float32),
        "verified": verified.astype(np.int16),
        "pool_tokens": candidate_ids.astype(np.int32),
        "pool_logits": _host_bf16_words(
            host["candidate_scores"], "the drafter's pool logits"
        ),
        "pool_log_norms": host["log_norms"].astype(np.float32),
        "pool_gumbels": host["gumbels"].astype(np.float32),
        "rho": _host_bf16_words(
            host["slot_residuals"][expanded_slots], "the Weaver residual logits"
        ),
        "state": host["slot_states"][expanded_slots],
        "draft_h": host["proposal_features"],
        "target_h": host["output_norm"],
    }


def _host_target_record(host: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """_target_record on host copies of _target_device_fields."""
    if float(host["deviation"]) > DEVIATION_LIMIT:
        raise RuntimeError(
            f"The target's filtered distribution leaves its top {TARGET_TOP_K} tokens; raise TARGET_TOP_K."
        )
    return {
        "target_top_tokens": host["top_indices"].astype(np.int32),
        "target_top_logits": host["top_values"].astype(np.float32),
        "target_log_norm": host["log_norm"].astype(np.float32),
        "target_filtered": host["filtered"].astype(np.float32),
        "target_choice": host["choices"].astype(np.int32),
    }


# The target filters (top-p, top-k renormalisation kernels) are not bitwise deterministic: on the
# same probabilities, 18 of 20 calls differed from the first by up to 3e-8 with the same support.
# The graph check compares the filtered distribution to this bound and its support exactly.
FILTERED_TOLERANCE = 1e-6


def _same_target(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray]
) -> str | None:
    """_same_arrays for the target record, target_filtered compared up to FILTERED_TOLERANCE."""
    rest = lambda record: {k: v for k, v in record.items() if k != "target_filtered"}
    differs = _same_arrays(rest(left), rest(right))
    if differs is not None:
        return differs
    a, b = left["target_filtered"], right["target_filtered"]
    if a.shape != b.shape or not np.array_equal(a > 0, b > 0):
        return "target_filtered support"
    if np.abs(a.astype(np.float64) - b).max(initial=0.0) > FILTERED_TOLERANCE:
        return "target_filtered"
    return None


def _same_arrays(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray]
) -> str | None:
    """The first field where two records differ (in keys, dtype, shape or bits), None if none."""
    if left.keys() != right.keys():
        return f"fields {sorted(left.keys() ^ right.keys())}"
    for name in sorted(left):
        a, b = np.asarray(left[name]), np.asarray(right[name])
        if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
            return name
    return None


class UzuTreeDumper:

    def __init__(
        self,
        worker,
        dump_dir: str,
        shapes_spec: str | None,
        internals: bool = False,
        plain_check: bool = False,
        graph: bool = False,
        graph_check: bool = False,
    ) -> None:
        if graph and not internals:
            raise RuntimeError(
                "The graph dump is the internals dump built under a CUDA graph."
            )
        if graph and plain_check:
            raise RuntimeError(
                "The plain check runs with the eager internals dump only."
            )
        self.internals = internals
        self.plain_check = plain_check
        self.graph = graph
        self.graph_check = graph_check
        self._graph_tree: _GraphTree | None = None
        self._slots: list[dict[str, torch.Tensor]] = []
        self._free_slots: queue.Queue[int] = queue.Queue()
        self._writes: queue.Queue[_GraphItem | None] = queue.Queue()
        self._writer: threading.Thread | None = None
        self._writer_error: BaseException | None = None
        self._pending_graph: (
            tuple[int, str, int, dict[str, np.ndarray] | None] | None
        ) = None
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
            "plain_check": plain_check,
            # The confidence head, or None: it ranks the prune unless prune_sigma_inv is set, and
            # with confidence_head_expand it also sets the expansion priority (log r, or with
            # confidence_head_expand_coverage log r + log kappa).
            "confidence_head": (
                None
                if worker.uzu_confidence_head is None
                else worker.uzu_confidence_head.source
            ),
            "confidence_head_expand": bool(worker.uzu_confidence_head_expand),
            "confidence_head_expand_coverage": bool(
                worker.uzu_confidence_head_expand_coverage
            ),
            # The tree grown on the drafter alone (children and priority without the Weaver
            # residual); the dumped rho is still the Weaver residual of every expanded slot.
            "expand_without_weaver": bool(worker.uzu_expand_without_weaver),
            # A separate head that sets the expansion priority (its local log r), or None.
            "expansion_head": (
                None
                if worker.uzu_expansion_head is None
                else worker.uzu_expansion_head.source
            ),
            # The expanding head's local part runs incrementally (the fused round kernel, the prune
            # on the kept parts) rather than over the whole tree after every round.
            "incremental_head": worker.uzu_round_kernel is not None,
        }
        if internals:
            self.meta["target_top_k"] = TARGET_TOP_K
            self.meta["graph"] = graph
        self.max_depth = int(worker.tree_max_depth)
        self._request_dirs: dict[str, Path] = {}
        self._step_counts: dict[str, int] = {}
        self._pending: _PendingStep | None = None
        if graph:
            self._slots = [{} for _ in range(GRAPH_SLOTS)]
            for slot in range(GRAPH_SLOTS):
                self._free_slots.put(slot)

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
            served = construct_uzu_tree(
                worker,
                **inputs,
                shape=self.served_shape,
                confidence_head=worker.uzu_confidence_head,
                expansion_head=worker.uzu_expansion_head,
            )
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
        if self._pending is not None or self._pending_graph is not None:
            raise RuntimeError(
                "The uzu tree dump saw two tree builds without a verify between them."
            )
        if self.graph:
            return self._build_served_graph(
                worker, rid=rid, root_position=root_position, inputs=inputs
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
                worker,
                **inputs,
                shape=self.served_shape,
                internals=internals,
                confidence_head=worker.uzu_confidence_head,
                expansion_head=worker.uzu_expansion_head,
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
            if self.plain_check:
                arrays |= self._plain_check(worker, inputs, constructed, tree)
        arrays["root_position"] = np.array(root_position, dtype=np.int64)
        self._pending = _PendingStep(rid=rid, arrays=arrays)
        return tree

    def _plain_check(
        self,
        worker,
        inputs: dict[str, torch.Tensor],
        constructed: UzuConstructedTree,
        tree: WeaverTree,
    ) -> dict[str, np.ndarray]:
        """The same step grown with the plain Weaver step the CUDA graph runs: whether its
        pruned tree is the verified one, and the largest prune-score difference over the
        sources present in both (inf if the construction differs)."""
        plain = construct_uzu_tree(
            worker,
            **inputs,
            shape=self.served_shape,
            confidence_head=worker.uzu_confidence_head,
            expansion_head=worker.uzu_expansion_head,
        )
        present = torch.cat((constructed.node_mask[0], constructed.frontier_active[0]))
        same_nodes = bool(
            torch.equal(
                present, torch.cat((plain.node_mask[0], plain.frontier_active[0]))
            )
            and torch.equal(constructed.node_tokens, plain.node_tokens)
            and torch.equal(constructed.frontier_tokens, plain.frontier_tokens)
        )
        error = float("inf")
        if same_nodes:
            served = torch.cat(
                (constructed.node_prune_scores[0], constructed.frontier_prune_scores[0])
            )
            other = torch.cat(
                (plain.node_prune_scores[0], plain.frontier_prune_scores[0])
            )
            error = float((served - other)[present].abs().max())
        return {
            "plain_tree_match": np.array(
                _same_tree(plain.prune(self.final_num_nodes), tree)
            ),
            "plain_same_nodes": np.array(same_nodes),
            "plain_prune_score_err": np.array(error, dtype=np.float32),
        }

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
        if self.graph:
            if logits is None or sampling_info is None or positions is None:
                raise RuntimeError(
                    "The uzu tree dump with internals needs the verifier's logits, "
                    "sampling info and node positions."
                )
            self._record_accept_graph(accept_tokens, logits, sampling_info, positions)
            return
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

    def _eager_served(
        self, worker, inputs: dict[str, torch.Tensor]
    ) -> tuple[WeaverTree, dict[str, np.ndarray]]:
        """The eager internals build of build_served: its pruned tree and its step record."""
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
            worker,
            **inputs,
            shape=self.served_shape,
            internals=internals,
            confidence_head=worker.uzu_confidence_head,
            expansion_head=worker.uzu_expansion_head,
        )
        tree, sources = constructed.prune_with_sources(self.final_num_nodes)
        record = _tree_record(
            worker,
            inputs,
            constructed,
            sources,
            internals,
            self.served_shape,
            self.max_depth,
        )
        return tree, record

    def _capture(self, worker, inputs: dict[str, torch.Tensor]) -> _GraphTree:
        """Capture the served tree with its internals, the pool Gumbel field and the drafter's
        log-normaliser as one graph, after an eager warm-up run on the same static buffers.
        """
        static = {
            name: torch.empty_like(value)
            for name, value in inputs.items()
            if name != "token_embed"
        }
        for name, buffer in static.items():
            buffer.copy_(inputs[name])
        token_embed = inputs["token_embed"]
        candidate_ids = static["candidate_ids"]
        depth = int(candidate_ids.shape[1])
        internals = UzuTreeInternals.empty(
            1,
            self.served_shape.num_slots,
            int(worker.weaver.d_rank),
            int(candidate_ids.shape[2]),
            device=candidate_ids.device,
            dtype=static["candidate_weights"].dtype,
        )
        _, lm_head = worker._target_embedding_and_lm_head()

        def run():
            constructed = construct_uzu_tree(
                worker,
                **static,
                token_embed=token_embed,
                shape=self.served_shape,
                internals=internals,
                confidence_head=worker.uzu_confidence_head,
                expansion_head=worker.uzu_expansion_head,
            )
            tree, sources = constructed.prune_with_sources(self.final_num_nodes)
            features = static["proposal_features"]
            log_norms = torch.logsumexp(
                _draft_vocab_logits(
                    lm_head, features[0].reshape(depth, features.shape[-1])
                ),
                dim=-1,
            )
            return (
                constructed,
                tree,
                sources,
                pool_gumbels(worker, candidate_ids),
                log_norms,
            )

        run()
        torch.cuda.synchronize(candidate_ids.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            constructed, tree, sources, gumbels, log_norms = run()
        torch.cuda.synchronize(candidate_ids.device)
        return _GraphTree(
            graph=graph,
            inputs=static | {"token_embed": token_embed},
            constructed=constructed,
            tree=tree,
            sources=sources,
            internals=internals,
            gumbels=gumbels,
            log_norms=log_norms,
        )

    def _graph_for(self, worker, inputs: dict[str, torch.Tensor]) -> _GraphTree:
        state = self._graph_tree
        if state is None:
            state = self._graph_tree = self._capture(worker, inputs)
            return state
        if inputs["token_embed"].data_ptr() != state.inputs[
            "token_embed"
        ].data_ptr() or any(
            state.inputs[name].shape != value.shape
            or state.inputs[name].dtype != value.dtype
            for name, value in inputs.items()
        ):
            raise RuntimeError(
                "The uzu tree dump graph saw inputs of another shape or embedding."
            )
        return state

    def _raise_writer_error(self) -> None:
        if self._writer_error is not None:
            raise RuntimeError(
                "The uzu tree dump writer failed."
            ) from self._writer_error

    def _pinned(self, slot: int, name: str, tensor: torch.Tensor) -> torch.Tensor:
        buffers = self._slots[slot]
        if name not in buffers:
            buffers[name] = torch.empty(
                tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=True
            )
        return buffers[name]

    def _acquire_slot(self) -> int:
        while True:
            self._raise_writer_error()
            try:
                return self._free_slots.get(timeout=1.0)
            except queue.Empty:
                continue

    def _build_served_graph(
        self, worker, *, rid: str, root_position: int, inputs: dict[str, torch.Tensor]
    ) -> WeaverTree:
        """The served tree from the graph; its record is copied to a pinned slot without blocking."""
        self._raise_writer_error()
        with torch.inference_mode():
            state = self._graph_for(worker, inputs)
            for name, buffer in state.inputs.items():
                if name != "token_embed":
                    buffer.copy_(inputs[name])
            state.graph.replay()
            slot = self._acquire_slot()
            for name, tensor in _tree_device_fields(state).items():
                self._pinned(slot, name, tensor).copy_(tensor, non_blocking=True)
            check = None
            if self.graph_check:
                torch.cuda.synchronize(state.tree.draft_tokens.device)
                host = {
                    name: _host(self._slots[slot][name])
                    for name in _tree_device_fields(state)
                }
                graph_record = _host_tree_record(
                    host, self.served_shape, self.max_depth
                )
                eager_tree, eager_record = self._eager_served(worker, inputs)
                differs = _same_arrays(graph_record, eager_record)
                if differs is not None or not _same_tree(eager_tree, state.tree):
                    raise RuntimeError(
                        f"The graph dump differs from the eager one in {differs or 'the tree'}."
                    )
                check = eager_record
        self._pending_graph = (slot, rid, root_position, check)
        return state.tree

    def _record_accept_graph(
        self,
        accept_tokens: torch.Tensor,
        logits: torch.Tensor,
        sampling_info,
        positions: torch.Tensor,
    ) -> None:
        self._raise_writer_error()
        if self._pending_graph is None:
            raise RuntimeError(
                "The uzu tree dump saw a verify without a recorded tree build."
            )
        slot, rid, root_position, check = self._pending_graph
        self._pending_graph = None
        with torch.inference_mode():
            target = _target_device_fields(
                logits, sampling_info, positions, self.final_num_nodes
            )
            for name, tensor in target.items():
                self._pinned(slot, f"target_{name}", tensor).copy_(
                    tensor, non_blocking=True
                )
            count = int(accept_tokens.numel())
            buffer = self._pinned(
                slot,
                "accept_tokens",
                torch.empty(self.final_num_nodes + 1, dtype=accept_tokens.dtype),
            )
            buffer[:count].copy_(accept_tokens, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
            if check is not None:
                event.synchronize()
                host = {
                    name: _host(self._slots[slot][f"target_{name}"]) for name in target
                }
                eager = _target_record(
                    logits, sampling_info, positions, self.final_num_nodes
                )
                differs = _same_target(_host_target_record(host), eager)
                if differs is not None:
                    raise RuntimeError(
                        f"The graph dump differs from the eager one in {differs}."
                    )
        if self._writer is None:
            self._writer = threading.Thread(
                target=self._write_loop, name="uzu-tree-dump-writer", daemon=True
            )
            self._writer.start()
        self._writes.put(_GraphItem(slot, event, rid, root_position, count))

    def _write_loop(self) -> None:
        """Wait for each step's copies, build its record on the host and write it."""
        try:
            while (item := self._writes.get()) is not None:
                item.event.synchronize()
                buffers = self._slots[item.slot]
                tree = {
                    name: _host(tensor)
                    for name, tensor in buffers.items()
                    if not name.startswith("target_") and name != "accept_tokens"
                }
                target = {
                    name.removeprefix("target_"): _host(tensor)
                    for name, tensor in buffers.items()
                    if name.startswith("target_")
                }
                accept = _host(buffers["accept_tokens"])[: item.num_accept].astype(
                    np.int32
                )
                self._free_slots.put(item.slot)
                arrays = _host_tree_record(tree, self.served_shape, self.max_depth)
                arrays["root_position"] = np.array(item.root_position, dtype=np.int64)
                arrays["accept_tokens"] = accept
                arrays |= _host_target_record(target)
                self._write_step(item.rid, arrays)
        except BaseException as error:  # noqa: BLE001 - surfaced on the next dump call
            self._writer_error = error

    def _write_step(self, rid: str, arrays: dict[str, np.ndarray]) -> None:
        """One step file, visible under its final name only once complete."""
        request_dir = self._request_dir(rid)
        step = self._step_counts[rid]
        self._step_counts[rid] = step + 1
        path = request_dir / f"{step:06d}.npz"
        partial = path.with_name(path.name + ".part")
        with partial.open("wb") as handle:
            np.savez(handle, **arrays)
        os.replace(partial, path)

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
