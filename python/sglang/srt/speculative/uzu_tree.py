"""Uzu-style grow-and-prune tree with deterministic or shared-Gumbel Top-C.

Child selection is always driven by the shared Gumbel field when it is enabled.
The edge weight, and therefore frontier priority and final pruning, is noise-free
only at --speculative-dflash-tfm-uzu-weight-sigma inf (the default); a finite sigma
mixes the same field into the weight as softmax(logits / T + G / sigma).

The drafter is also calibrated independently of the verifier: --...-uzu-draft-temperature
replaces the request temperature in the score above, --...-uzu-weight-temperature gives the edge
weight a temperature of its own while children are still selected at the draft temperature, and
--...-uzu-draft-top-k/-top-p/-min-p (their nucleus taken at the verifier's temperature)
truncate the drafter's pool distribution before the tree is scored. Truncated candidates are
masked to -inf, so they leave Top-C selection and the edge-weight normaliser together and the
edge stays a log-softmax of one consistent vector.

--...-uzu-prune-sigma gives the final prune a node score of its own: the same edge weight
with the field at 1 / sigma_prune, accumulated along the path. Frontier priority keeps the score
built with --...-uzu-weight-sigma. Without it there is one score for both, as before.

A finite sigma, a draft temperature and draft filtering all change node scores and draft log
probabilities, which is safe only because shared_gumbel requires target_only verification, and
target_only never reads draft_logprobs (only the traversal path does). They therefore alter tree
topology, never the distribution of accepted tokens.

Weaver inference and verification remain in dflash_tfm.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import msgspec
import torch
import triton
import triton.language as tl

# Loaded lazily by DFlashTfmWorker after dflash_tfm finishes importing.
from sglang.srt.speculative.dflash_tfm import WeaverTree
from sglang.srt.speculative.shared_gumbel import gumbel_at
from sglang.srt.speculative.uzu_confidence_head import (
    CONFIG,
    WEIGHTS,
    ConfidenceHead,
    ExpandedTree,
    TreeInputs,
    VectorConfidenceHead,
)
from sglang.srt.speculative.uzu_confidence_kernel import (
    RoundState,
    VectorKernelWeights,
    child_positions,
    expand_round,
    supports,
)

logger = logging.getLogger(__name__)


class UzuRoundKernel(msgspec.Struct, frozen=True):
    """The expanding head whose local part the rounds compute incrementally, and its packed weights."""

    head: VectorConfidenceHead
    weights: VectorKernelWeights


class UzuTreeShape(msgspec.Struct, frozen=True):
    """How a tree grows: R expansion rounds, W parents per round, C children per parent."""

    rounds: int
    parents_per_round: int
    children_per_parent: int

    @classmethod
    def from_worker(cls, worker) -> UzuTreeShape:
        return cls(
            worker.tree_rounds,
            worker.tree_parents_per_round,
            worker.tree_children_per_parent,
        )

    @classmethod
    def parse(cls, spec: str) -> UzuTreeShape:
        """Parse "RxWxC", e.g. "16x32x16"."""
        parts = spec.strip().lower().split("x")
        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            raise ValueError(f"A tree shape must be RxWxC in positive integers, got {spec!r}.")
        return cls(*(int(part) for part in parts))

    @property
    def num_slots(self) -> int:
        """Construction slots: the root plus W parents in each of the R - 1 later rounds."""
        return 1 + (self.rounds - 1) * self.parents_per_round

    @property
    def max_candidates(self) -> int:
        """Nodes before the prune: the root plus C children of every construction slot."""
        return 1 + self.children_per_parent * self.num_slots

    def __str__(self) -> str:
        return f"{self.rounds}x{self.parents_per_round}x{self.children_per_parent}"


def check_uzu_tree_shape(shape: UzuTreeShape, candidate_pool_size: int, tree_budget: int) -> None:
    """The limits configure_uzu_tree enforces on the served shape, for an extra (dump) shape."""
    if shape.rounds < 1:
        raise ValueError(f"Tree shape {shape}: R must be >= 1.")
    if not 1 <= shape.parents_per_round <= 32:
        raise ValueError(
            f"Tree shape {shape}: W must be in 1..32 (the fused frontier kernel supports at "
            "most 32 parents)."
        )
    if not 1 <= shape.children_per_parent <= candidate_pool_size:
        raise ValueError(
            f"Tree shape {shape}: C must be in 1..{candidate_pool_size} (the effective candidate "
            "pool)."
        )
    if not 1 <= tree_budget < shape.max_candidates:
        raise ValueError(
            f"Tree shape {shape}: the tree budget {tree_budget} must be below its "
            f"{shape.max_candidates} candidates, or pruning it would keep everything."
        )


class UzuConstructedTree(msgspec.Struct):
    """A grown tree before the final prune: construction slots plus the still-active frontier.

    Slots are in topological order and ``*_parents`` index slots. ``*_scores`` accumulate the
    frontier-priority score, ``*_prune_scores`` the score the prune reads (the very same tensors
    without --...-uzu-prune-sigma), and ``*_logprobs`` hold per-edge log weights.
    """

    node_tokens: torch.Tensor
    node_parents: torch.Tensor
    node_depths: torch.Tensor
    node_mask: torch.Tensor
    node_logprobs: torch.Tensor
    node_scores: torch.Tensor
    node_prune_scores: torch.Tensor
    frontier_tokens: torch.Tensor
    frontier_parents: torch.Tensor
    frontier_depths: torch.Tensor
    frontier_active: torch.Tensor
    frontier_logprobs: torch.Tensor
    frontier_scores: torch.Tensor
    frontier_prune_scores: torch.Tensor

    def prune(self, final_num_nodes: int) -> WeaverTree:
        """Keep the top final_num_nodes nodes by prune score, as the served tree does."""
        return self.prune_with_sources(final_num_nodes)[0]

    def prune_with_sources(self, final_num_nodes: int) -> tuple[WeaverTree, torch.Tensor]:
        """The pruned tree and, per kept node, its source: the index into slots followed by frontier."""
        return _weaver_prune_constructed_tree(
            node_tokens=self.node_tokens,
            node_parents=self.node_parents,
            node_depths=self.node_depths,
            node_mask=self.node_mask,
            node_logprobs=self.node_logprobs,
            node_scores=self.node_prune_scores,
            frontier_tokens=self.frontier_tokens,
            frontier_parents=self.frontier_parents,
            frontier_depths=self.frontier_depths,
            frontier_active=self.frontier_active,
            frontier_logprobs=self.frontier_logprobs,
            frontier_scores=self.frontier_prune_scores,
            final_num_nodes=final_num_nodes,
        )


class UzuTreeInternals(msgspec.Struct):
    """What each construction slot's Weaver step saw, filled by construct_uzu_tree on request.

    Row ``s`` of every tensor belongs to construction slot ``s`` (the same index as
    ``UzuConstructedTree.node_*``): ``slot_states`` is the read-out query the pool was scored
    against, ``slot_residuals`` the residual logits before the drafter's candidate scores are
    added, ``slot_logits`` the pool logits the slot's children were taken from. Rows of slots
    that were never filled, or were filled but not expanded, hold whatever the step computed and
    must be read through the tree's own masks.
    """

    slot_states: torch.Tensor
    slot_residuals: torch.Tensor
    slot_logits: torch.Tensor

    @classmethod
    def empty(
        cls, bs: int, num_slots: int, d_rank: int, pool_size: int, *, device, dtype
    ) -> UzuTreeInternals:
        return cls(
            slot_states=torch.zeros((bs, num_slots, d_rank), dtype=dtype, device=device),
            slot_residuals=torch.zeros(
                (bs, num_slots, pool_size), dtype=torch.float32, device=device
            ),
            slot_logits=torch.zeros(
                (bs, num_slots, pool_size), dtype=torch.float32, device=device
            ),
        )


def configure_uzu_tree(worker) -> None:
    args = worker.server_args
    worker.tree_rounds = args.speculative_dflash_tfm_uzu_rounds
    worker.tree_parents_per_round = args.speculative_dflash_tfm_uzu_width
    worker.tree_children_per_parent = args.speculative_dflash_tfm_uzu_children
    available_depth = min(worker.block_size - 1, worker.weaver.K)
    requested_depth = args.speculative_dflash_tfm_uzu_max_depth
    worker.tree_max_depth = (
        available_depth if requested_depth is None else requested_depth
    )
    if worker.tree_rounds < 1:
        raise ValueError("--speculative-dflash-tfm-uzu-rounds must be >= 1.")
    if not 1 <= worker.tree_parents_per_round <= 32:
        raise ValueError(
            "--speculative-dflash-tfm-uzu-width must be in 1..32 "
            "(the fused frontier kernel supports at most 32 parents)."
        )
    if not 1 <= worker.tree_children_per_parent <= worker.candidate_pool_size:
        raise ValueError(
            "--speculative-dflash-tfm-uzu-children must be in "
            f"1..{worker.candidate_pool_size} (the effective candidate pool)."
        )
    if not 1 <= worker.tree_max_depth <= available_depth:
        raise ValueError(
            "--speculative-dflash-tfm-uzu-max-depth must be in "
            f"1..{available_depth}, limited by the backbone and Weaver checkpoint."
        )
    max_candidates = 1 + worker.tree_children_per_parent * (
        1 + (worker.tree_rounds - 1) * worker.tree_parents_per_round
    )
    if not 1 <= worker.tree_budget < max_candidates:
        raise ValueError(
            "--speculative-dflash-tfm-tree-budget must be positive and below "
            f"{max_candidates} for this R/W/C configuration (budget excludes root)."
        )
    if worker.tree_sampling_mode != "target_only":
        raise ValueError(
            "weaver_uzu requires target_only verification; "
            "its pruned proposals do not implement traversal rejection."
        )
    worker.uzu_confidence_head = None
    worker.uzu_confidence_head_expand = bool(
        args.speculative_dflash_tfm_uzu_confidence_head_expand
    )
    worker.uzu_confidence_head_expand_coverage = bool(
        args.speculative_dflash_tfm_uzu_confidence_head_expand_coverage
    )
    worker.uzu_expansion_head = None
    head_dir = args.speculative_dflash_tfm_uzu_confidence_head
    if head_dir is not None:
        head = _load_confidence_head(worker, head_dir)
        worker.uzu_confidence_head = head
        if worker.gumbel_prune_sigma_inv is None:
            logger.info("weaver_uzu prunes with the confidence head %s", head.source)
        if worker.uzu_confidence_head_expand_coverage and head.expansion is None:
            raise ValueError(
                "--speculative-dflash-tfm-uzu-confidence-head-expand-coverage needs a confidence "
                f"head with a coverage branch (format 3); {head.source} has none."
            )
        if worker.uzu_confidence_head_expand:
            logger.info(
                "weaver_uzu expands by the %s confidence head %s%s",
                (
                    "local part (no tree attention) of the"
                    if head.tree is not None
                    else ""
                ),
                head.source,
                (
                    " with its coverage branch (log r + log kappa)"
                    if worker.uzu_confidence_head_expand_coverage
                    else ""
                ),
            )
    expansion_dir = args.speculative_dflash_tfm_uzu_expansion_head
    if expansion_dir is not None:
        prune_head = worker.uzu_confidence_head
        if prune_head is not None and _same_export(expansion_dir, prune_head):
            # One head both expands (its local part) and prunes: one instance, one token table.
            expansion_head = prune_head
        else:
            expansion_head = _load_confidence_head(worker, expansion_dir)
        worker.uzu_expansion_head = expansion_head
        logger.info(
            "weaver_uzu expands by the %sconfidence head %s; the prune keeps its own score",
            (
                "local part (no tree attention) of the "
                if expansion_head.tree is not None
                else ""
            ),
            expansion_head.source,
        )
    worker.uzu_round_kernel = None
    expanding = (
        worker.uzu_expansion_head
        if worker.uzu_expansion_head is not None
        else worker.uzu_confidence_head if worker.uzu_confidence_head_expand else None
    )
    if expanding is not None:
        if (
            isinstance(expanding, VectorConfidenceHead)
            and supports(expanding.config)
            and not worker.uzu_confidence_head_expand_coverage
            and not args.speculative_dflash_tfm_uzu_confidence_head_recompute
        ):
            worker.uzu_round_kernel = UzuRoundKernel(
                head=expanding, weights=VectorKernelWeights.pack(expanding)
            )
            logger.info(
                "weaver_uzu computes the expanding head's local part only for the slots each "
                "round expands (fused round kernel)%s",
                (
                    "; the prune reuses it"
                    if expanding is worker.uzu_confidence_head
                    else ""
                ),
            )
        else:
            logger.info(
                "weaver_uzu re-scores the whole tree with the expanding head after every round"
            )


def _load_confidence_head(worker, directory: str) -> ConfidenceHead:
    # The head projects the target's raw input embedding, not the Weaver token embedding the
    # tree is grown with.
    embed_module, _ = worker._target_embedding_and_lm_head()
    embedding = embed_module.weight
    head = ConfidenceHead.load(directory, embedding, embedding.device, compile=True)
    head.check_tree(
        worker.tree_children_per_parent,
        worker.tree_max_depth,
        worker.candidate_pool_size,
        int(embedding.shape[1]),
    )
    return head


def _same_export(directory: str, head: ConfidenceHead) -> bool:
    """Whether the export in directory is the head already loaded: the same config and weights."""
    path = Path(directory)
    config = json.loads((path / CONFIG).read_text())
    digest = hashlib.sha256((path / WEIGHTS).read_bytes()).hexdigest()
    return config == head.config and digest == head.config["weights_sha256"]


# The frontier limit changes every round (slot_start * C). As a constexpr it compiled one variant
# per round, which a 64-round diagnostic tree pays for in tens of minutes; as a runtime value, left
# unspecialised, there is one variant per block size. It only bounds the frontier mask, an exact
# integer compare, so the selection is the same either way.
@triton.jit(do_not_specialize=["frontier_limit"])
def _uzu_materialize_frontier_kernel(
    frontier_tokens_ptr,
    frontier_parents_ptr,
    frontier_depths_ptr,
    frontier_scores_ptr,
    frontier_logprobs_ptr,
    frontier_active_ptr,
    slot_ancestors_ptr,
    tokens_ptr,
    parents_ptr,
    depths_ptr,
    node_mask_ptr,
    draft_logprobs_ptr,
    node_scores_ptr,
    selected_tokens_ptr,
    selected_depths_ptr,
    selected_position_ids_ptr,
    selected_candidate_rows_ptr,
    selected_batch_indices_ptr,
    selected_scores_ptr,
    selected_active_ptr,
    selected_parent_ancestors_ptr,
    frontier_prune_scores_ptr,
    node_prune_scores_ptr,
    selected_prune_scores_ptr,
    slot_start,
    frontier_limit,
    NUM_NODES: tl.constexpr,
    DEPTH: tl.constexpr,
    MAX_TREE_DEPTH: tl.constexpr,
    FRONTIER_SLOTS: tl.constexpr,
    SELECT_WIDTH: tl.constexpr,
    SCRATCH_WIDTH: tl.constexpr,
    BLOCK_DEPTH: tl.constexpr,
    BLOCK_FRONTIER: tl.constexpr,
    WRITE_ANCESTORS: tl.constexpr,
    SEPARATE_PRUNE: tl.constexpr,
):
    batch = tl.program_id(0)
    selected_offset = tl.arange(0, 32)
    selected_mask = selected_offset < SELECT_WIDTH
    frontier_offset = tl.arange(0, BLOCK_FRONTIER)
    frontier_mask = frontier_offset < frontier_limit
    frontier_scores = tl.load(
        frontier_scores_ptr + batch * FRONTIER_SLOTS + frontier_offset,
        mask=frontier_mask,
        other=-float("inf"),
    ).to(tl.float32)
    frontier_active = (
        tl.load(
            frontier_active_ptr + batch * FRONTIER_SLOTS + frontier_offset,
            mask=frontier_mask,
            other=0,
        )
        != 0
    )
    frontier_depth = tl.load(
        frontier_depths_ptr + batch * FRONTIER_SLOTS + frontier_offset,
        mask=frontier_mask,
        other=DEPTH,
    )
    frontier_valid = (
        frontier_mask & frontier_active & (frontier_scores != -float("inf"))
    )
    # Match Uzu's integer priority key. A float32 +1e20 offset erases
    # all ordinary log-probability differences among expandable parents.
    bits = frontier_scores.to(tl.uint32, bitcast=True)
    score_key = tl.where((bits & 0x80000000) == 0, bits ^ 0x80000000, bits ^ 0xFFFFFFFF)
    effective_max_depth: tl.constexpr = min(MAX_TREE_DEPTH, DEPTH)
    priority_key = ((frontier_depth < effective_max_depth).to(tl.uint32) << 31) | (
        score_key >> 1
    )
    priority_key = tl.where(frontier_valid, priority_key, 0).to(tl.uint32)
    frontier_parent = tl.load(
        frontier_parents_ptr + batch * FRONTIER_SLOTS + frontier_offset,
        mask=frontier_mask,
        other=2147483647,
    )
    frontier_token = tl.load(
        frontier_tokens_ptr + batch * FRONTIER_SLOTS + frontier_offset,
        mask=frontier_mask,
        other=2147483647,
    )
    frontier_index = tl.full((32,), 0, tl.int32)
    selected_valid = tl.full((32,), False, tl.int1)
    for pick in tl.static_range(0, SELECT_WIDTH):
        top_key = tl.max(priority_key, axis=0)
        top_parent = tl.min(
            tl.where(priority_key == top_key, frontier_parent, 2147483647), axis=0
        )
        top_token = tl.min(
            tl.where(
                (priority_key == top_key) & (frontier_parent == top_parent),
                frontier_token,
                2147483647,
            ),
            axis=0,
        )
        matches = (
            (priority_key == top_key)
            & (frontier_parent == top_parent)
            & (frontier_token == top_token)
        )
        top_index = tl.min(tl.where(matches, frontier_offset, 2147483647), axis=0)
        top_index = tl.where(top_key != 0, top_index, 0)
        frontier_index = tl.where(selected_offset == pick, top_index, frontier_index)
        selected_valid = tl.where(selected_offset == pick, top_key != 0, selected_valid)
        priority_key = tl.where(frontier_offset == top_index, 0, priority_key).to(
            tl.uint32
        )
    selected_index = batch * FRONTIER_SLOTS + frontier_index
    score = tl.load(
        frontier_scores_ptr + selected_index,
        mask=selected_mask,
        other=0.0,
    ).to(tl.float32)
    valid = selected_valid & (
        tl.load(frontier_active_ptr + selected_index, mask=selected_mask, other=0) != 0
    )
    token = tl.load(frontier_tokens_ptr + selected_index, mask=selected_mask, other=0)
    parent = tl.load(frontier_parents_ptr + selected_index, mask=selected_mask, other=0)
    depth = tl.load(frontier_depths_ptr + selected_index, mask=selected_mask, other=0)
    logprob = tl.load(
        frontier_logprobs_ptr + selected_index,
        mask=selected_mask,
        other=-float("inf"),
    )

    output_index = batch * NUM_NODES + slot_start + selected_offset
    tl.store(
        tokens_ptr + output_index,
        tl.where(valid, token, 0),
        mask=selected_mask,
    )
    tl.store(
        parents_ptr + output_index,
        tl.where(valid, parent, -1),
        mask=selected_mask,
    )
    tl.store(
        depths_ptr + output_index,
        tl.where(valid, depth, 0),
        mask=selected_mask,
    )
    tl.store(node_mask_ptr + output_index, valid, mask=selected_mask)
    tl.store(
        draft_logprobs_ptr + output_index,
        tl.where(valid, logprob, -float("inf")),
        mask=selected_mask,
    )
    tl.store(
        node_scores_ptr + output_index,
        tl.where(valid, score, -float("inf")),
        mask=selected_mask,
    )

    scratch_index = batch * SCRATCH_WIDTH + selected_offset
    tl.store(
        selected_tokens_ptr + scratch_index,
        tl.where(valid, token, 0),
        mask=selected_mask,
    )
    tl.store(
        selected_depths_ptr + scratch_index,
        tl.where(valid, depth, 0),
        mask=selected_mask,
    )
    position = tl.minimum(depth, DEPTH - 1)
    tl.store(
        selected_position_ids_ptr + scratch_index,
        tl.where(valid, position, 0),
        mask=selected_mask,
    )
    tl.store(
        selected_candidate_rows_ptr + scratch_index,
        tl.where(valid, batch * DEPTH + position, 0),
        mask=selected_mask,
    )
    tl.store(
        selected_batch_indices_ptr + scratch_index,
        batch,
        mask=selected_mask,
    )
    tl.store(
        selected_scores_ptr + scratch_index,
        tl.where(valid, score, -float("inf")),
        mask=selected_mask,
    )
    if SEPARATE_PRUNE:
        # The prune score travels with the node: into its tree slot for the final prune, and
        # into the scratch row as the parent prune score of its own children.
        prune_score = tl.load(
            frontier_prune_scores_ptr + selected_index, mask=selected_mask, other=0.0
        ).to(tl.float32)
        tl.store(
            node_prune_scores_ptr + output_index,
            tl.where(valid, prune_score, -float("inf")),
            mask=selected_mask,
        )
        tl.store(
            selected_prune_scores_ptr + scratch_index,
            tl.where(valid, prune_score, -float("inf")),
            mask=selected_mask,
        )
    tl.store(
        selected_active_ptr + scratch_index,
        valid & (depth < effective_max_depth),
        mask=selected_mask,
    )
    tl.store(
        frontier_active_ptr + selected_index,
        False,
        mask=selected_mask & selected_valid,
    )

    if WRITE_ANCESTORS:
        ancestor_offsets = tl.arange(0, BLOCK_DEPTH)[None, :]
        ancestor_mask = (ancestor_offsets < DEPTH) & selected_mask[:, None]
        parent_safe = tl.minimum(tl.maximum(parent, 0), NUM_NODES - 1)[:, None]
        ancestors = tl.load(
            slot_ancestors_ptr
            + (batch * NUM_NODES + parent_safe) * DEPTH
            + ancestor_offsets,
            mask=ancestor_mask & valid[:, None],
            other=-1,
        )
        tl.store(
            selected_parent_ancestors_ptr
            + scratch_index[:, None] * DEPTH
            + ancestor_offsets,
            ancestors,
            mask=ancestor_mask,
        )


@triton.jit
def _uzu_publish_frontier_kernel(
    current_keys_ptr,
    current_values_ptr,
    node_keys_ptr,
    node_values_ptr,
    parent_ancestors_ptr,
    slot_ancestors_ptr,
    logits_ptr,
    candidate_ids_ptr,
    prefix_score_ptr,
    prefix_prune_score_ptr,
    valid_ptr,
    node_depth_ptr,
    frontier_tokens_ptr,
    frontier_parents_ptr,
    frontier_depths_ptr,
    frontier_scores_ptr,
    frontier_prune_scores_ptr,
    frontier_logprobs_ptr,
    frontier_active_ptr,
    slot_start,
    gumbel_seeds,
    gumbel_positions,
    gumbel_temperatures,
    gumbel_scale,
    gumbel_weight_scale,
    gumbel_weight_temperatures,
    gumbel_prune_scale,
    SHARED_GUMBEL: tl.constexpr,
    SEPARATE_PRUNE: tl.constexpr,
    BS: tl.constexpr,
    WIDTH: tl.constexpr,
    DEPTH: tl.constexpr,
    NUM_NODES: tl.constexpr,
    NUM_LAYERS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TOTAL_KV: tl.constexpr,
    TOTAL_ANCESTORS: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    EXPAND_WIDTH: tl.constexpr,
    FRONTIER_SLOTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_POOL: tl.constexpr,
):
    program = tl.program_id(0)
    offsets = program * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    kv_mask = offsets < TOTAL_KV
    hd = offsets % HEAD_DIM
    head = (offsets // HEAD_DIM) % NUM_HEADS
    layer = (offsets // (HEAD_DIM * NUM_HEADS)) % NUM_LAYERS
    row_in_width = (offsets // (HEAD_DIM * NUM_HEADS * NUM_LAYERS)) % WIDTH
    batch = (offsets // (HEAD_DIM * NUM_HEADS * NUM_LAYERS * WIDTH)) % BS
    row = batch * WIDTH + row_in_width
    valid = tl.load(valid_ptr + row, mask=kv_mask, other=0) != 0
    current_index = ((layer * BS * WIDTH + row) * NUM_HEADS + head) * HEAD_DIM + hd
    slot = slot_start + row_in_width
    node_index = (
        ((batch * NUM_NODES + slot) * NUM_LAYERS + layer) * NUM_HEADS + head
    ) * HEAD_DIM + hd
    key_value = tl.load(
        current_keys_ptr + current_index, mask=kv_mask & valid, other=0.0
    )
    value_value = tl.load(
        current_values_ptr + current_index, mask=kv_mask & valid, other=0.0
    )
    tl.store(node_keys_ptr + node_index, key_value, mask=kv_mask)
    tl.store(node_values_ptr + node_index, value_value, mask=kv_mask)

    ancestor_mask = offsets < TOTAL_ANCESTORS
    ancestor_depth = offsets % DEPTH
    ancestor_row = (offsets // DEPTH) % WIDTH
    ancestor_batch = (offsets // (DEPTH * WIDTH)) % BS
    ancestor_flat_row = ancestor_batch * WIDTH + ancestor_row
    ancestor_valid = (
        tl.load(valid_ptr + ancestor_flat_row, mask=ancestor_mask, other=0) != 0
    )
    current_pos = tl.load(
        node_depth_ptr + ancestor_flat_row,
        mask=ancestor_mask,
        other=0,
    )
    current_pos = tl.minimum(current_pos, DEPTH - 1)
    parent_value = tl.load(parent_ancestors_ptr + offsets, mask=ancestor_mask, other=-1)
    ancestor_slot = slot_start + ancestor_row
    ancestor_value = tl.where(
        ancestor_depth == current_pos, ancestor_slot, parent_value
    )
    ancestor_value = tl.where(ancestor_valid, ancestor_value, -1)
    out_index = (ancestor_batch * NUM_NODES + ancestor_slot) * DEPTH + ancestor_depth
    tl.store(slot_ancestors_ptr + out_index, ancestor_value, mask=ancestor_mask)

    if program < BS * WIDTH:
        candidate_offsets = tl.arange(0, BLOCK_POOL)
        pool_mask = candidate_offsets < POOL_SIZE
        candidate_base = program * POOL_SIZE + candidate_offsets
        token_ids = tl.load(
            candidate_ids_ptr + candidate_base, mask=pool_mask, other=-1
        )
        raw_scores = tl.load(
            logits_ptr + candidate_base,
            mask=pool_mask,
            other=-float("inf"),
        ).to(tl.float32)
        raw_scores = tl.where((token_ids >= 0) & pool_mask, raw_scores, -float("inf"))
        parent_score = tl.load(prefix_score_ptr + program)
        parent_depth = tl.load(node_depth_ptr + program)
        if SHARED_GUMBEL:
            row_batch = program // WIDTH
            valid_slot = (token_ids >= 0) & pool_mask
            # Selection and weight divide by temperatures of their own. The weight temperature
            # defaults to a copy of the selection one, and then the two quotients are bitwise
            # equal, so the tree is exactly what a single shared temperature gave.
            # Each quotient is re-masked after the division: an infinite temperature turns the
            # -inf of a padded or token-less slot into NaN, which would then win a max and
            # corrupt the tree. For any finite temperature this is an exact no-op.
            selection_logits = tl.where(
                valid_slot, raw_scores / tl.load(gumbel_temperatures + row_batch),
                -float("inf"),
            )
            weight_logits = tl.where(
                valid_slot, raw_scores / tl.load(gumbel_weight_temperatures + row_batch),
                -float("inf"),
            )
            noise = gumbel_at(
                tl.load(gumbel_seeds + row_batch),
                tl.load(gumbel_positions + row_batch) + parent_depth, token_ids)
            selection_scores = selection_logits + tl.load(gumbel_scale) * noise
            # The weight sees the same field scaled by 1 / sigma. At sigma = inf the scale is
            # 0.0 and this is an exact no-op, so the edge weight stays what it is today.
            weight_scores = weight_logits + tl.load(gumbel_weight_scale) * noise
            if SEPARATE_PRUNE:
                # The prune score sees the same field scaled by 1 / sigma_prune. Only the final
                # prune reads it; frontier priority keeps the sigma score above.
                prune_scores = weight_logits + tl.load(gumbel_prune_scale) * noise
        else:
            selection_scores = raw_scores
            weight_scores = raw_scores
        parent_active = (tl.load(valid_ptr + program) != 0) & (parent_depth < DEPTH)
        frontier_batch = program // WIDTH
        frontier_row = program - frontier_batch * WIDTH
        max_score = tl.max(weight_scores, axis=0)
        exp_scores = tl.where(
            weight_scores == -float("inf"), 0.0, tl.exp(weight_scores - max_score)
        )
        log_denom = tl.log(tl.sum(exp_scores, axis=0)) + max_score
        if SEPARATE_PRUNE:
            prune_max = tl.max(prune_scores, axis=0)
            prune_exp = tl.where(
                prune_scores == -float("inf"), 0.0, tl.exp(prune_scores - prune_max)
            )
            prune_log_denom = tl.log(tl.sum(prune_exp, axis=0)) + prune_max
            parent_prune_score = tl.load(prefix_prune_score_ptr + program)
        child_base = (
            frontier_batch * FRONTIER_SLOTS + (slot_start + frontier_row) * EXPAND_WIDTH
        )
        child_depth = parent_depth + 1
        for child in tl.static_range(0, EXPAND_WIDTH):
            selected_value, top_index = tl.max(
                selection_scores,
                axis=0,
                return_indices=True,
                return_indices_tie_break_left=True,
            )
            # Gather from the same vector the normalizer came from, so the edge stays a
            # log-softmax of one consistent score vector (temperature already applied there).
            top_value = tl.sum(
                tl.where(candidate_offsets == top_index, weight_scores, 0.0), axis=0
            )
            child_token = tl.load(candidate_ids_ptr + program * POOL_SIZE + top_index)
            child_valid = (
                parent_active
                & (child_token >= 0)
                & (top_value != -float("inf"))
                & (selected_value != -float("inf"))
            )
            # Compute the edge first, as native Uzu does. Cancelling logits
            # after adding the parent can round a child above its parent,
            # invalidating the monotonicity required by final stable pruning.
            edge_logprob = tl.minimum(top_value - log_denom, 0.0)
            child_score = tl.minimum(parent_score, parent_score + edge_logprob)
            child_index = child_base + child
            tl.store(
                frontier_tokens_ptr + child_index,
                tl.where(child_valid, child_token, 0),
            )
            tl.store(
                frontier_parents_ptr + child_index,
                tl.where(child_valid, slot_start + frontier_row, 0),
            )
            tl.store(
                frontier_depths_ptr + child_index,
                tl.where(child_valid, child_depth, 0),
            )
            tl.store(
                frontier_scores_ptr + child_index,
                tl.where(
                    child_valid,
                    child_score,
                    -float("inf"),
                ),
            )
            tl.store(
                frontier_logprobs_ptr + child_index,
                tl.where(child_valid, edge_logprob, -float("inf")),
            )
            if SEPARATE_PRUNE:
                # Same child, same monotone accumulation, the other weight vector.
                prune_value = tl.sum(
                    tl.where(candidate_offsets == top_index, prune_scores, 0.0), axis=0
                )
                prune_edge = tl.minimum(prune_value - prune_log_denom, 0.0)
                prune_child = tl.minimum(
                    parent_prune_score, parent_prune_score + prune_edge
                )
                tl.store(
                    frontier_prune_scores_ptr + child_index,
                    tl.where(child_valid, prune_child, -float("inf")),
                )
            tl.store(frontier_active_ptr + child_index, child_valid)
            selection_scores = tl.where(
                candidate_offsets == top_index,
                -float("inf"),
                selection_scores,
            )


@triton.jit
def _pool_gumbel_kernel(
    noise_ptr,
    tokens_ptr,
    seeds_ptr,
    positions_ptr,
    DEPTH: tl.constexpr,
    POOL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program per (request, pool row d). The children of a node at depth d come from
    # pool row d and draw the field at the request's position plus d, exactly as the
    # publish kernel does; slots without a token read token -1 there too.
    program = tl.program_id(0)
    batch = program // DEPTH
    row = program - batch * DEPTH
    offsets = tl.arange(0, BLOCK)
    mask = offsets < POOL
    tokens = tl.load(tokens_ptr + program * POOL + offsets, mask=mask, other=-1)
    noise = gumbel_at(
        tl.load(seeds_ptr + batch), tl.load(positions_ptr + batch) + row, tokens
    )
    tl.store(noise_ptr + program * POOL + offsets, noise, mask=mask)


def pool_gumbels(worker, candidate_ids: torch.Tensor) -> torch.Tensor:
    """The shared Gumbel field on the candidate pool of every depth, [bs, depth, pool], as
    the tree kernel draws it."""
    bs, depth, pool = candidate_ids.shape
    seeds, positions = worker._gumbel_tree_inputs[bs][:2]
    rows = candidate_ids.contiguous()
    noise = torch.empty(rows.shape, dtype=torch.float32, device=rows.device)
    _pool_gumbel_kernel[(bs * depth,)](
        noise,
        rows,
        seeds,
        positions,
        DEPTH=int(depth),
        POOL=int(pool),
        BLOCK=int(triton.next_power_of_2(pool)),
    )
    return noise


def _frontier_select_warps(block_frontier: int) -> int:
    """Warps for the frontier select: one up to 1024 slots, then one per 1024 up to 16.

    The select is a single program per request that unrolls SELECT_WIDTH exact integer
    max/min reductions over the whole frontier block, so its result does not depend on the
    warp count. One warp keeps the small served frontiers exactly as they were; a large
    diagnostic frontier on one warp would hold a thousand elements per thread and take
    minutes to compile and to run.
    """
    return 1 if block_frontier <= 1024 else min(16, block_frontier // 1024)


def _tree_frontier_expandable(
    valid: torch.Tensor,
    node_depth: torch.Tensor,
    max_depth: int,
) -> torch.Tensor:
    """Return parents whose children would not exceed ``max_depth``."""
    return valid & (node_depth < max_depth)


def _weaver_prune_constructed_tree(
    *,
    node_tokens: torch.Tensor,
    node_parents: torch.Tensor,
    node_depths: torch.Tensor,
    node_mask: torch.Tensor,
    node_logprobs: torch.Tensor,
    node_scores: torch.Tensor,
    frontier_tokens: torch.Tensor,
    frontier_parents: torch.Tensor,
    frontier_depths: torch.Tensor,
    frontier_active: torch.Tensor,
    frontier_logprobs: torch.Tensor,
    frontier_scores: torch.Tensor,
    final_num_nodes: int,
) -> tuple[WeaverTree, torch.Tensor]:
    """Apply Uzu's final cumulative-path-score prune on the GPU; also return the kept sources.

    Construction slots are already topological. Active frontier nodes are
    appended after them. Stable score sorting therefore resolves exact ties in
    parent-before-child order, matching Uzu's trie pruning contract. The
    retained sources are then restored to topological order for TreeAttention.
    """

    all_tokens = torch.cat((node_tokens, frontier_tokens), dim=1)
    all_parents = torch.cat((node_parents, frontier_parents), dim=1)
    all_depths = torch.cat((node_depths, frontier_depths), dim=1)
    all_mask = torch.cat((node_mask, frontier_active), dim=1)
    all_logprobs = torch.cat((node_logprobs, frontier_logprobs), dim=1)
    all_scores = torch.cat((node_scores, frontier_scores), dim=1)
    all_scores = all_scores.masked_fill(~all_mask, -torch.inf)
    # A shallow or narrow configuration may exhaust the frontier before B.
    # Keep unused verifier slots masked, with safe token/parent/depth values.
    all_tokens = all_tokens.masked_fill(~all_mask, 0)
    all_parents = all_parents.masked_fill(~all_mask, -1)
    all_depths = all_depths.masked_fill(~all_mask, 0)

    # Stable descending order gives score first and topological source index
    # second. Sorting the retained source IDs restores the compact tree's
    # parent-before-child order without changing the retained set.
    selected_sources = torch.argsort(all_scores, dim=1, descending=True, stable=True)[
        :, :final_num_nodes
    ]
    selected_sources = torch.sort(selected_sources, dim=1).values
    compact_indices = torch.arange(
        final_num_nodes, dtype=torch.long, device=node_tokens.device
    )[None, :].expand_as(selected_sources)
    inverse = torch.full_like(all_parents, -1)
    inverse.scatter_(1, selected_sources, compact_indices)

    selected_parents = torch.gather(all_parents, 1, selected_sources)
    compact_parents = torch.gather(
        inverse, 1, selected_parents.clamp_min(0)
    ).masked_fill(selected_parents < 0, -1)
    tree = WeaverTree(
        torch.gather(all_tokens, 1, selected_sources),
        compact_parents,
        torch.gather(all_depths, 1, selected_sources),
        torch.gather(all_mask, 1, selected_sources),
        torch.gather(all_logprobs, 1, selected_sources),
    )
    return tree, selected_sources


def _filter_draft_logits(
    logits: torch.Tensor,
    row_candidate_ids: torch.Tensor,
    temperatures: torch.Tensor | None,
    width: int,
    top_k: int,
    top_p: float,
    min_p: float,
) -> torch.Tensor:
    """Truncate the drafter's pool distribution, independently of the verifier's.

    The probabilities are taken at the VERIFIER's temperature over the candidate pool the
    drafter actually scores, so a nucleus here is a share of the pool mass and not of the
    vocabulary. The verifier temperature, not the selection one, because the nucleus stands in
    for the verifier's own, which no drafter knob moves. Nothing is lost by that: children are
    a Top-C, which sees only the order of f(l)/T_sel + G, and that order equals the order of
    f(l) + T_sel * G, so the selection temperature acts as a scale on the Gumbel noise and never
    needs tempered logits. Filtering happens after the temperature, matching the order the
    verifier's own path uses.

    Dropped candidates are masked to -inf, which removes them from Top-C selection and from the
    edge-weight normaliser at once; the kernel already treats -inf as "not a child". One
    descending sort serves all three filters.

    The filters chain rather than intersect: top-k first, then top-p over the probabilities
    renormalised on the top-k survivors, then min-p over what is left. This is what
    apply_dflash_draft_truncation does on the DFLASH chain path, and what the verifier's own
    top_k_renorm_prob / top_p_renorm_prob do. Intersecting three thresholds taken on one
    full-pool softmax would keep strictly more candidates and give the same flags a second
    meaning.
    """
    if top_k == 0 and top_p >= 1.0 and min_p <= 0.0:
        return logits
    if temperatures is None:
        raise RuntimeError(
            "Draft-side filtering needs the shared Gumbel context for the draft temperature."
        )
    # Pool slots without a token carry meaningless logits and must not occupy a top-k slot or
    # eat nucleus mass; the kernel masks them the same way.
    logits = logits.masked_fill(row_candidate_ids < 0, -float("inf"))
    sorted_logits, order = logits.sort(dim=-1, descending=True)
    # Rows are batch-major, width rows per request, matching the kernel's program // WIDTH.
    row_temperatures = temperatures.repeat_interleave(width).unsqueeze(-1)
    tempered = sorted_logits / row_temperatures
    positions = torch.arange(logits.shape[-1], device=logits.device)
    keep = torch.ones_like(tempered, dtype=torch.bool)

    def survivor_probs() -> torch.Tensor:
        """Distribution over the candidates still alive, so each stage sees the previous one."""
        return torch.softmax(
            torch.where(keep, tempered, torch.full_like(tempered, -float("inf"))), dim=-1
        )

    if top_k > 0:
        keep &= positions < top_k
    if top_p < 1.0:
        probs = survivor_probs()
        # Keep every candidate whose predecessors have not yet covered top_p, so the candidate
        # that crosses the threshold survives and the kept mass is at least top_p.
        keep &= (probs.cumsum(-1) - probs) < top_p
    if min_p > 0.0:
        probs = survivor_probs()
        # Position 0 always survives the stages above, so its probability is the surviving max.
        keep &= probs >= min_p * probs[:, :1]
    # The most likely candidate always survives, whatever the thresholds say.
    keep |= positions == 0
    mask = torch.zeros_like(keep).scatter_(-1, order, keep)
    return logits.masked_fill(~mask, -float("inf"))


def build_uzu_tree(
    self,
    *,
    root_ids: torch.Tensor,
    output_norm: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_weights: torch.Tensor,
    candidate_scores: torch.Tensor,
    proposal_features: torch.Tensor,
    token_embed: torch.Tensor,
) -> WeaverTree:
    shape = UzuTreeShape.from_worker(self)
    node_budget = int(self.tree_budget)
    if node_budget <= 0 or candidate_ids.shape[1] <= 0:
        return _root_only_tree(root_ids, shape.num_slots)
    return construct_uzu_tree(
        self,
        root_ids=root_ids,
        output_norm=output_norm,
        candidate_ids=candidate_ids,
        candidate_weights=candidate_weights,
        candidate_scores=candidate_scores,
        proposal_features=proposal_features,
        token_embed=token_embed,
        shape=shape,
        confidence_head=self.uzu_confidence_head,
        expansion_head=self.uzu_expansion_head,
    ).prune(node_budget + 1)


def _root_only_tree(root_ids: torch.Tensor, num_nodes: int) -> WeaverTree:
    """The tree when there is nothing to grow (no budget or no drafted depth): the root alone."""
    bs = root_ids.shape[0]
    device = root_ids.device
    tokens = torch.zeros((bs, num_nodes), dtype=torch.long, device=device)
    parents = torch.full((bs, num_nodes), -1, dtype=torch.long, device=device)
    depths = torch.zeros((bs, num_nodes), dtype=torch.long, device=device)
    node_mask = torch.zeros((bs, num_nodes), dtype=torch.bool, device=device)
    draft_logprobs = torch.full(
        (bs, num_nodes), -torch.inf, dtype=torch.float32, device=device
    )
    tokens[:, 0] = root_ids
    node_mask[:, 0] = True
    draft_logprobs[:, 0] = 0.0
    return WeaverTree(tokens, parents, depths, node_mask, draft_logprobs)


def construct_uzu_tree(
    self,
    *,
    root_ids: torch.Tensor,
    output_norm: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_weights: torch.Tensor,
    candidate_scores: torch.Tensor,
    proposal_features: torch.Tensor,
    token_embed: torch.Tensor,
    shape: UzuTreeShape,
    internals: UzuTreeInternals | None = None,
    confidence_head: ConfidenceHead | None = None,
    expansion_head: ConfidenceHead | None = None,
) -> UzuConstructedTree:
    """Grow a tree of the given shape on this step's inputs, without the final prune.

    Everything but the shape comes from the worker: coupling, filters, depth cap. The served tree
    is this with the worker's own shape, pruned; any other shape is a diagnostic tree grown on
    the same draft, the same pool and the same Gumbel field.

    With ``internals`` every Weaver step runs the compiled variant that also returns its read-out
    state and residual logits, and writes them into the slots it expands. That variant may differ
    from the plain one in the last bits, so the internals dump serves this very tree rather than
    comparing it with a separately built one.

    With ``confidence_head`` the prune scores are the head's log r instead of the cumulative
    path score; the pool logits of every slot are kept for it on the way. With the worker's
    ``uzu_confidence_head_expand`` the head's local part (a head with tree attention without its
    attention corrections) is run after every round as well and its log r becomes the frontier
    priority, so the tree grows best-first by the head's score; a sigma_prune then keeps its own
    prune score and the head only expands. With ``uzu_confidence_head_expand_coverage`` the
    priority is log r + log kappa of the head's coverage branch instead; the prune still reads
    log r. With ``expansion_head`` that head's local part sets the frontier priority instead of
    the prune head, and the prune keeps its own score (the prune head's log r or sigma_prune).
    """
    bs, depth, pool_size = candidate_ids.shape
    if depth <= 0:
        raise ValueError("A Weaver tree needs at least one drafted depth to grow.")
    coupling = self._gumbel_tree_inputs[bs] if self.use_gumbel_sampling else (None,) * 7
    separate_prune = self.use_gumbel_sampling and self.gumbel_prune_sigma_inv is not None
    expand_head = (
        expansion_head
        if expansion_head is not None
        else confidence_head if self.uzu_confidence_head_expand else None
    )
    expand_with_head = expand_head is not None
    prune_with_head = confidence_head is not None and not separate_prune
    filter_temperatures = (
        self._draft_filter_temperatures[bs] if self.use_gumbel_sampling else None
    )
    batch_expand_width = int(shape.parents_per_round)
    expand_width = int(shape.children_per_parent)
    num_nodes = shape.num_slots
    device = root_ids.device
    tokens = torch.zeros((bs, num_nodes), dtype=torch.long, device=device)
    parents = torch.full((bs, num_nodes), -1, dtype=torch.long, device=device)
    depths = torch.zeros((bs, num_nodes), dtype=torch.long, device=device)
    node_mask = torch.zeros((bs, num_nodes), dtype=torch.bool, device=device)
    draft_logprobs = torch.full(
        (bs, num_nodes), -torch.inf, dtype=torch.float32, device=device
    )
    node_scores = torch.full_like(draft_logprobs, -torch.inf)
    tokens[:, 0] = root_ids
    node_mask[:, 0] = True
    draft_logprobs[:, 0] = 0.0
    node_scores[:, 0] = 0.0
    # Pool logits l = a + rho of every slot, for the confidence head; the publish kernel
    # consumes them and keeps only the children.
    slot_logits = (
        None
        if confidence_head is None and expansion_head is None
        else torch.zeros((bs, num_nodes, pool_size), dtype=torch.float32, device=device)
    )

    frontier_slots = num_nodes * expand_width
    batch_indices = torch.arange(bs, dtype=torch.long, device=device)
    num_layers = self.weaver.num_layers
    num_heads = self.weaver.num_heads
    head_dim = self.weaver.d_rank // self.weaver.num_heads
    if batch_expand_width > 32:
        raise RuntimeError(
            "Fused Weaver frontier materialization supports at most "
            f"32 selected nodes per expansion, got {batch_expand_width}."
        )
    external_keys, external_values, external_mask = self.weaver.prompt_external_kv(
        output_norm[:, None], proposal_features
    )
    candidate_ids_rows = candidate_ids.reshape(bs * depth, pool_size)
    candidate_weights_rows = candidate_weights.reshape(
        bs * depth, pool_size, candidate_weights.shape[-1]
    )
    candidate_scores_rows = candidate_scores.reshape(bs * depth, pool_size)
    node_keys = torch.empty(
        (bs, num_nodes, num_layers, num_heads, head_dim),
        dtype=proposal_features.dtype,
        device=device,
    )
    node_values = torch.empty_like(node_keys)
    slot_ancestors = torch.empty(
        (bs, num_nodes, depth), dtype=torch.long, device=device
    )

    frontier_tokens = torch.empty((bs, frontier_slots), dtype=torch.long, device=device)
    frontier_parents = torch.empty_like(frontier_tokens)
    frontier_depths = torch.empty_like(frontier_tokens)
    frontier_scores = torch.empty(
        (bs, frontier_slots), dtype=torch.float32, device=device
    )
    frontier_logprobs = torch.empty_like(frontier_scores)
    frontier_active = torch.empty((bs, frontier_slots), dtype=torch.bool, device=device)
    frontier_scores.fill_(-torch.inf)
    frontier_logprobs.fill_(-torch.inf)
    frontier_active.zero_()
    if expand_with_head:
        # The head reads every source each round: entries no publish has written yet must be
        # well-formed absent sources, not uninitialised memory.
        frontier_tokens.zero_()
        frontier_parents.fill_(-1)
        frontier_depths.zero_()
    selected_tokens = torch.empty(
        (bs, batch_expand_width), dtype=torch.long, device=device
    )
    selected_depths = torch.empty_like(selected_tokens)
    selected_position_ids = torch.empty_like(selected_tokens)
    selected_candidate_rows = torch.empty_like(selected_tokens)
    selected_batch_indices = torch.empty_like(selected_tokens)
    selected_scores = torch.empty(
        (bs, batch_expand_width), dtype=torch.float32, device=device
    )
    selected_active = torch.empty(
        (bs, batch_expand_width), dtype=torch.bool, device=device
    )
    selected_parent_ancestors = torch.empty(
        (bs, batch_expand_width, depth),
        dtype=torch.long,
        device=device,
    )
    # Prune scores: a second cumulative score with the field at 1 / sigma_prune, read only by the
    # final prune. Without sigma_prune these are the very score tensors, the kernels never touch
    # them, and the prune is exactly the one before this knob existed.
    if separate_prune:
        node_prune_scores = torch.full_like(node_scores, -torch.inf)
        node_prune_scores[:, 0] = 0.0
        frontier_prune_scores = torch.full_like(frontier_scores, -torch.inf)
        selected_prune_scores = torch.empty_like(selected_scores)
    else:
        node_prune_scores = node_scores
        frontier_prune_scores = frontier_scores
        selected_prune_scores = selected_scores
    if device.type != "cuda":
        raise RuntimeError("Weaver tree construction requires Triton on CUDA.")

    def publish_frontier(
        logits: torch.Tensor,
        row_candidate_ids: torch.Tensor,
        current_keys: torch.Tensor,
        current_values: torch.Tensor,
        parent_ancestors: torch.Tensor,
        prefix_score: torch.Tensor,
        prefix_prune_score: torch.Tensor,
        valid: torch.Tensor,
        node_depth: torch.Tensor,
        slot_start: int,
        width: int,
    ) -> None:
        logits = _filter_draft_logits(
            logits,
            row_candidate_ids,
            filter_temperatures,
            width,
            self.draft_top_k,
            self.draft_top_p,
            self.draft_min_p,
        )
        # Keep children at tree_max_depth, but never expand them further.
        # Applying the cap before publishing leaves the fixed node budget
        # available to other, shallower frontier branches.
        valid = _tree_frontier_expandable(
            valid,
            node_depth,
            self.tree_max_depth,
        )
        total_kv = bs * width * num_layers * num_heads * head_dim
        total_ancestors = bs * width * depth
        block_size = 256
        grid = (
            max(
                triton.cdiv(total_kv, block_size),
                triton.cdiv(total_ancestors, block_size),
                bs * width,
            ),
        )
        _uzu_publish_frontier_kernel[grid](
            current_keys,
            current_values,
            node_keys,
            node_values,
            parent_ancestors,
            slot_ancestors,
            logits,
            row_candidate_ids,
            prefix_score,
            prefix_prune_score,
            valid,
            node_depth,
            frontier_tokens,
            frontier_parents,
            frontier_depths,
            frontier_scores,
            frontier_prune_scores,
            frontier_logprobs,
            frontier_active,
            int(slot_start),
            *coupling,
            SHARED_GUMBEL=self.use_gumbel_sampling,
            SEPARATE_PRUNE=separate_prune,
            BS=int(bs),
            WIDTH=int(width),
            DEPTH=int(depth),
            NUM_NODES=int(num_nodes),
            NUM_LAYERS=int(num_layers),
            NUM_HEADS=int(num_heads),
            HEAD_DIM=int(head_dim),
            TOTAL_KV=int(total_kv),
            TOTAL_ANCESTORS=int(total_ancestors),
            POOL_SIZE=int(pool_size),
            EXPAND_WIDTH=int(expand_width),
            FRONTIER_SLOTS=int(frontier_slots),
            BLOCK_SIZE=int(block_size),
            BLOCK_POOL=int(triton.next_power_of_2(pool_size)),
        )

    def materialize_frontier(slot_start: int, width: int) -> None:
        frontier_limit = max(width, slot_start * expand_width)
        block_frontier = int(triton.next_power_of_2(frontier_limit))
        _uzu_materialize_frontier_kernel[(bs,)](
            frontier_tokens,
            frontier_parents,
            frontier_depths,
            frontier_scores,
            frontier_logprobs,
            frontier_active,
            slot_ancestors,
            tokens,
            parents,
            depths,
            node_mask,
            draft_logprobs,
            node_scores,
            selected_tokens,
            selected_depths,
            selected_position_ids,
            selected_candidate_rows,
            selected_batch_indices,
            selected_scores,
            selected_active,
            selected_parent_ancestors,
            frontier_prune_scores,
            node_prune_scores,
            selected_prune_scores,
            int(slot_start),
            int(frontier_limit),
            NUM_NODES=int(num_nodes),
            DEPTH=int(depth),
            MAX_TREE_DEPTH=int(self.tree_max_depth),
            FRONTIER_SLOTS=int(frontier_slots),
            SELECT_WIDTH=int(width),
            SCRATCH_WIDTH=int(batch_expand_width),
            BLOCK_DEPTH=int(triton.next_power_of_2(depth)),
            BLOCK_FRONTIER=block_frontier,
            # Construction may intentionally materialize more nodes than
            # the final target-verification budget before the stable T-node
            # prune.  Every construction slot still needs its ancestor
            # chain for subsequent Weaver rounds.
            WRITE_ANCESTORS=slot_start + width <= num_nodes,
            SEPARATE_PRUNE=separate_prune,
            num_warps=_frontier_select_warps(block_frontier),
        )

    def expand_node_indexed(
        token_ids: torch.Tensor,
        position_ids: torch.Tensor,
        candidate_row_index: torch.Tensor,
        parent_ancestors: torch.Tensor,
        row_batch_indices: torch.Tensor,
        slot_start: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        step_kwargs = dict(
            token_ids=token_ids,
            candidate_ids=candidate_ids_rows,
            candidate_weights=candidate_weights_rows,
            candidate_scores=candidate_scores_rows,
            candidate_row_index=candidate_row_index,
            external_keys=external_keys,
            external_values=external_values,
            external_mask=external_mask,
            position_ids=position_ids,
            node_keys=node_keys,
            node_values=node_values,
            parent_ancestors=parent_ancestors.reshape(
                bs * parent_ancestors.shape[1], depth
            ).contiguous(),
            row_batch_indices=row_batch_indices,
            token_embed=token_embed,
        )
        if internals is None:
            (
                logits,
                current_keys,
                current_values,
                row_candidate_ids,
            ) = self._weaver_indexed_step_compiled(
                **step_kwargs,
            )
        else:
            (
                logits,
                current_keys,
                current_values,
                row_candidate_ids,
                states,
                residuals,
            ) = self._weaver_indexed_step_compiled(**step_kwargs, with_internals=True)
            # Rows are batch-major, width rows per request, as in the publish kernel.
            slots = slice(slot_start, slot_start + width)
            internals.slot_states[:, slots] = states.view(bs, width, -1)
            internals.slot_residuals[:, slots] = residuals.view(bs, width, -1)
            internals.slot_logits[:, slots] = logits.float().view(bs, width, -1)
        if slot_logits is not None:
            slot_logits[:, slot_start : slot_start + width] = logits.float().view(
                bs, width, -1
            )
        return (
            logits.float(),
            row_candidate_ids,
            current_keys,
            current_values,
        )

    gumbels = (
        None
        if confidence_head is None and expansion_head is None
        else pool_gumbels(self, candidate_ids)
    )

    def head_inputs() -> TreeInputs:
        """The tree grown so far as the head reads it."""
        assert slot_logits is not None and gumbels is not None
        return TreeInputs(
            slot_logits=slot_logits,
            pool_logits=candidate_scores,
            pool_gumbels=gumbels,
            pool_tokens=candidate_ids,
            draft_hidden=proposal_features.reshape(bs, depth, -1),
            tokens=torch.cat((tokens, frontier_tokens), dim=1),
            parents=torch.cat((parents, frontier_parents), dim=1),
            depths=torch.cat((depths, frontier_depths), dim=1),
            present=torch.cat((node_mask, frontier_active), dim=1),
        )

    # The head's local part scores a child from its parent and the path alone, so a node keeps its
    # log r as the tree grows and the last round's scores are those of the finished tree; for a head
    # without tree attention they are its prune scores too.
    head_log_r: torch.Tensor | None = None

    def rescore_frontier() -> None:
        nonlocal head_log_r
        assert expand_head is not None
        if self.uzu_confidence_head_expand_coverage and expansion_head is None:
            head_log_r, priority = expand_head.expansion_scores(head_inputs())
        else:
            head_log_r = priority = expand_head.local_scores(head_inputs())
        frontier_scores.copy_(priority[:, num_nodes:])

    # The incremental path: the expanding head's local part only for the slots a round has just
    # expanded, kept per slot for the prune (same scores as re-scoring the whole tree).
    round_kernel = (
        self.uzu_round_kernel
        if self.uzu_round_kernel is not None
        and expand_head is self.uzu_round_kernel.head
        else None
    )
    if round_kernel is not None:
        assert slot_logits is not None and gumbels is not None
        # The kernel reads every pool tensor as dense [bs, depth, pool].
        constants = round_kernel.head.step_constants(
            candidate_scores.contiguous(),
            gumbels,
            candidate_ids.contiguous(),
            proposal_features.reshape(bs, depth, -1),
        )
        state = RoundState.zeros(
            bs, num_nodes, expand_width, round_kernel.weights.hidden, device
        )

    def rescore(slot_start: int, width: int) -> None:
        """The frontier priority of the children a round has just published."""
        if round_kernel is not None:
            expand_round(
                round_kernel.weights,
                slot_start=slot_start,
                width=width,
                slot_logits=slot_logits,
                pool_logits=constants.pool_logits,
                pool_gumbels=constants.pool_gumbels,
                pool_tokens=constants.pool_tokens,
                draft_rows=constants.draft_rows,
                depths=depths,
                node_scores=node_scores,
                frontier_tokens=frontier_tokens,
                frontier_active=frontier_active,
                frontier_scores=frontier_scores,
                state=state,
            )
        elif expand_with_head:
            rescore_frontier()

    root_parent_ancestors = torch.full(
        (bs, 1, depth), -1, dtype=torch.long, device=device
    )
    root_prefix_score = torch.zeros((bs,), dtype=torch.float32, device=device)
    root_depth = torch.zeros((bs,), dtype=torch.long, device=device)
    root_active = torch.ones((bs,), dtype=torch.bool, device=device)
    (
        root_logits,
        root_candidate_ids,
        root_keys,
        root_values,
    ) = expand_node_indexed(
        root_ids,
        root_depth,
        batch_indices * depth,
        root_parent_ancestors,
        batch_indices,
        0,
        1,
    )
    publish_frontier(
        root_logits,
        root_candidate_ids,
        root_keys,
        root_values,
        root_parent_ancestors,
        root_prefix_score,
        root_prefix_score,
        root_active[:, None],
        root_depth[:, None],
        0,
        1,
    )
    rescore(0, 1)

    for round_index in range(1, shape.rounds):
        slot_start = 1 + (round_index - 1) * batch_expand_width
        width = batch_expand_width
        materialize_frontier(slot_start, width)
        valid = selected_active[:, :width]
        token = selected_tokens[:, :width]
        node_depth = selected_depths[:, :width]
        node_score = selected_scores[:, :width]
        parent_ancestors = selected_parent_ancestors[:, :width]
        position_ids = selected_position_ids[:, :width]
        candidate_row_index = selected_candidate_rows[:, :width]
        row_batch_indices = selected_batch_indices[:, :width]

        token_flat = token.reshape(bs * width)
        node_score_flat = node_score.reshape(bs * width)
        node_prune_score_flat = selected_prune_scores[:, :width].reshape(bs * width)
        node_depth_flat = node_depth.reshape(bs * width)
        valid_flat = valid.reshape(bs * width)
        position_ids_flat = position_ids.reshape(bs * width)
        candidate_row_index_flat = candidate_row_index.reshape(bs * width)
        row_batch_indices_flat = row_batch_indices.reshape(bs * width)

        (
            logits,
            row_candidate_ids,
            current_keys,
            current_values,
        ) = expand_node_indexed(
            token_flat,
            position_ids_flat,
            candidate_row_index_flat,
            parent_ancestors,
            row_batch_indices_flat,
            slot_start,
            width,
        )
        publish_frontier(
            logits,
            row_candidate_ids,
            current_keys,
            current_values,
            parent_ancestors,
            node_score_flat,
            node_prune_score_flat,
            valid_flat,
            node_depth_flat,
            slot_start,
            width,
        )
        rescore(slot_start, width)
    constructed = UzuConstructedTree(
        node_tokens=tokens,
        node_parents=parents,
        node_depths=depths,
        node_mask=node_mask,
        node_logprobs=draft_logprobs,
        node_scores=node_scores,
        node_prune_scores=node_prune_scores,
        frontier_tokens=frontier_tokens,
        frontier_parents=frontier_parents,
        frontier_depths=frontier_depths,
        frontier_active=frontier_active,
        frontier_logprobs=frontier_logprobs,
        frontier_scores=frontier_scores,
        frontier_prune_scores=frontier_prune_scores,
    )
    if not prune_with_head:
        return constructed
    assert confidence_head is not None
    if round_kernel is not None and confidence_head is round_kernel.head:
        # The full head on what the rounds kept: no slot's pool or drafter row is read again.
        scores = round_kernel.head.prune_expanded(
            ExpandedTree(
                slot_logits=slot_logits,
                nodes=state.nodes,
                children=state.children,
                slots=state.slots,
                child_indices=state.child_indices,
                child_real=state.child_real,
                child_positions=child_positions(
                    tokens, parents, node_mask, frontier_tokens, expand_width
                ),
                tokens=torch.cat((tokens, frontier_tokens), dim=1),
                parents=torch.cat((parents, frontier_parents), dim=1),
                depths=torch.cat((depths, frontier_depths), dim=1),
                present=torch.cat((node_mask, frontier_active), dim=1),
            ),
            constants,
        )
    elif (
        head_log_r is not None
        and expand_head is confidence_head
        and confidence_head.tree is None
    ):
        scores = head_log_r
    else:
        scores = confidence_head.prune_scores(head_inputs())
    return msgspec.structs.replace(
        constructed,
        node_prune_scores=scores[:, :num_nodes],
        frontier_prune_scores=scores[:, num_nodes:],
    )
