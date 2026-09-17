"""Uzu-style grow-and-prune tree with deterministic or shared-Gumbel Top-C.

Gumbel noise affects child selection only. Frontier priority and final pruning
use unperturbed cumulative log probabilities. Weaver inference and verification
remain in dflash_tfm.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Loaded lazily by DFlashTfmWorker after dflash_tfm finishes importing.
from sglang.srt.speculative.dflash_tfm import WeaverTree
from sglang.srt.speculative.shared_gumbel import gumbel_at

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


@triton.jit
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
    slot_start,
    NUM_NODES: tl.constexpr,
    DEPTH: tl.constexpr,
    MAX_TREE_DEPTH: tl.constexpr,
    FRONTIER_SLOTS: tl.constexpr,
    SELECT_WIDTH: tl.constexpr,
    SCRATCH_WIDTH: tl.constexpr,
    BLOCK_DEPTH: tl.constexpr,
    BLOCK_FRONTIER: tl.constexpr,
    FRONTIER_LIMIT: tl.constexpr,
    WRITE_ANCESTORS: tl.constexpr,
):
    batch = tl.program_id(0)
    selected_offset = tl.arange(0, 32)
    selected_mask = selected_offset < SELECT_WIDTH
    frontier_offset = tl.arange(0, BLOCK_FRONTIER)
    frontier_mask = frontier_offset < FRONTIER_LIMIT
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
    valid_ptr,
    node_depth_ptr,
    frontier_tokens_ptr,
    frontier_parents_ptr,
    frontier_depths_ptr,
    frontier_scores_ptr,
    frontier_logprobs_ptr,
    frontier_active_ptr,
    slot_start,
    gumbel_seeds,
    gumbel_positions,
    gumbel_temperatures,
    gumbel_scale,
    SHARED_GUMBEL: tl.constexpr,
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
            raw_scores = raw_scores / tl.load(gumbel_temperatures + row_batch)
            selection_scores = raw_scores + tl.load(gumbel_scale) * gumbel_at(
                tl.load(gumbel_seeds + row_batch),
                tl.load(gumbel_positions + row_batch) + parent_depth, token_ids)
        else:
            selection_scores = raw_scores
        parent_active = (tl.load(valid_ptr + program) != 0) & (parent_depth < DEPTH)
        frontier_batch = program // WIDTH
        frontier_row = program - frontier_batch * WIDTH
        max_score = tl.max(raw_scores, axis=0)
        exp_scores = tl.where(
            raw_scores == -float("inf"), 0.0, tl.exp(raw_scores - max_score)
        )
        log_denom = tl.log(tl.sum(exp_scores, axis=0)) + max_score
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
            top_value = tl.load(logits_ptr + program * POOL_SIZE + top_index).to(
                tl.float32
            )
            if SHARED_GUMBEL:
                top_value = top_value / tl.load(gumbel_temperatures + program // WIDTH)
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
            tl.store(frontier_active_ptr + child_index, child_valid)
            selection_scores = tl.where(
                candidate_offsets == top_index,
                -float("inf"),
                selection_scores,
            )


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
) -> WeaverTree:
    """Apply Uzu's final cumulative-path-score prune on the GPU.

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
    return WeaverTree(
        torch.gather(all_tokens, 1, selected_sources),
        compact_parents,
        torch.gather(all_depths, 1, selected_sources),
        torch.gather(all_mask, 1, selected_sources),
        torch.gather(all_logprobs, 1, selected_sources),
    )


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
    bs, depth, pool_size = candidate_ids.shape
    coupling = self._gumbel_tree_inputs[bs] if self.use_gumbel_sampling else (None,) * 4
    node_budget = int(self.tree_budget)
    final_num_nodes = node_budget + 1
    batch_expand_width = int(self.tree_parents_per_round)
    expand_width = int(self.tree_children_per_parent)
    num_nodes = 1 + (self.tree_rounds - 1) * batch_expand_width
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
    if node_budget <= 0 or depth <= 0:
        return WeaverTree(tokens, parents, depths, node_mask, draft_logprobs)

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
    if device.type != "cuda":
        raise RuntimeError("Weaver tree construction requires Triton on CUDA.")

    def publish_frontier(
        logits: torch.Tensor,
        row_candidate_ids: torch.Tensor,
        current_keys: torch.Tensor,
        current_values: torch.Tensor,
        parent_ancestors: torch.Tensor,
        prefix_score: torch.Tensor,
        valid: torch.Tensor,
        node_depth: torch.Tensor,
        slot_start: int,
        width: int,
    ) -> None:
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
            valid,
            node_depth,
            frontier_tokens,
            frontier_parents,
            frontier_depths,
            frontier_scores,
            frontier_logprobs,
            frontier_active,
            int(slot_start),
            *coupling,
            SHARED_GUMBEL=self.use_gumbel_sampling,
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
            int(slot_start),
            NUM_NODES=int(num_nodes),
            DEPTH=int(depth),
            MAX_TREE_DEPTH=int(self.tree_max_depth),
            FRONTIER_SLOTS=int(frontier_slots),
            SELECT_WIDTH=int(width),
            SCRATCH_WIDTH=int(batch_expand_width),
            BLOCK_DEPTH=int(triton.next_power_of_2(depth)),
            BLOCK_FRONTIER=int(triton.next_power_of_2(frontier_limit)),
            FRONTIER_LIMIT=int(frontier_limit),
            # Construction may intentionally materialize more nodes than
            # the final target-verification budget before the stable T-node
            # prune.  Every construction slot still needs its ancestor
            # chain for subsequent Weaver rounds.
            WRITE_ANCESTORS=slot_start + width <= num_nodes,
            num_warps=1,
        )

    def expand_node_indexed(
        token_ids: torch.Tensor,
        position_ids: torch.Tensor,
        candidate_row_index: torch.Tensor,
        parent_ancestors: torch.Tensor,
        row_batch_indices: torch.Tensor,
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
        (
            logits,
            current_keys,
            current_values,
            row_candidate_ids,
        ) = self._weaver_indexed_step_compiled(
            **step_kwargs,
        )
        return (
            logits.float(),
            row_candidate_ids,
            current_keys,
            current_values,
        )

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
    )
    publish_frontier(
        root_logits,
        root_candidate_ids,
        root_keys,
        root_values,
        root_parent_ancestors,
        root_prefix_score,
        root_active[:, None],
        root_depth[:, None],
        0,
        1,
    )

    for round_index in range(1, self.tree_rounds):
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
        )
        publish_frontier(
            logits,
            row_candidate_ids,
            current_keys,
            current_values,
            parent_ancestors,
            node_score_flat,
            valid_flat,
            node_depth_flat,
            slot_start,
            width,
        )
    return _weaver_prune_constructed_tree(
        node_tokens=tokens,
        node_parents=parents,
        node_depths=depths,
        node_mask=node_mask,
        node_logprobs=draft_logprobs,
        node_scores=node_scores,
        frontier_tokens=frontier_tokens,
        frontier_parents=frontier_parents,
        frontier_depths=frontier_depths,
        frontier_active=frontier_active,
        frontier_logprobs=frontier_logprobs,
        frontier_scores=frontier_scores,
        final_num_nodes=final_num_nodes,
    )
