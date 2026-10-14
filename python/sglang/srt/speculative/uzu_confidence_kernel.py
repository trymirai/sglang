"""Fused per-round kernel of the vector confidence head's local part (format 5).

A round of the weaver_uzu tree expands a few slots and publishes their children. The head's local
part of a slot reads only that slot's own drafter row, its own pool and its own children, so after
the publish one program per new slot computes it and writes what the expansion and the final prune
need:

  - the frontier priority of every real child: log r = the slot's log r + min(log q_u(c), 0);
  - z_u, the children's token vectors, their slot numbers, pool indices and which are real.

It is the function of VectorConfidenceHead.expand_slots (uzu_confidence_head.py) for the heads that
supports() accepts: the drafter row and the pool set encoder summed into z_u with no node layer
(the export folds a dense one), the child token and the four slot numbers, the residual edge MLP of
depth one over [z_u; e_c; z_u * e_c; slot], alpha and beta straight from z_u, no input SiLU. A
head outside that set keeps the whole-tree path. Everything is fp32 (dots in IEEE precision, as the
head's _full_fp32), with no host synchronisation, so the kernel runs inside the tree's CUDA graph.

The module imports only torch, triton and msgspec, so offline tests can load it by file path.
"""

from __future__ import annotations

import msgspec
import torch
import triton
import triton.language as tl

POOL_INPUTS = ("pool_draft", "pool_weaver", "pool_noise")
INPUTS = frozenset({"draft", "child_token", *POOL_INPUTS})
SLOT_NUMBERS = 4
# Rows of the edge MLP's dots: the children of one slot, padded to the smallest dot height.
EDGE_ROWS = 16
LOG_PROB_FLOOR = -30.0


def supports(config: dict) -> bool:
    """Whether the fused kernel computes this exported head's local part."""
    vector = config.get("vector")
    if config.get("format") != 5 or vector is None:
        return False
    hidden = int(config["hidden"])
    return (
        set(vector["inputs"]) == INPUTS
        and tuple(vector["pool_inputs"]) == POOL_INPUTS
        and bool(vector["slot"])
        and bool(vector["node_sum"])
        and vector["node_depth"] is None
        and vector["child_depth"] is None
        and vector["node_out_depth"] is None
        and int(vector["edge_depth"]) == 1
        and vector["mlp"] == "residual"
        and not vector["input_act"]
        and hidden >= 16
        and hidden & (hidden - 1) == 0
        and int(config["children"]) <= EDGE_ROWS
    )


class VectorKernelWeights(msgspec.Struct, frozen=True):
    """The head's local-part weights as contiguous f32 tensors on the head's device."""

    entry_weight: torch.Tensor  # [pool hidden, 3]
    entry_bias: torch.Tensor
    hidden_weight: torch.Tensor  # [pool hidden, pool hidden]
    hidden_bias: torch.Tensor
    out_weight: torch.Tensor  # [hidden, 2 * pool hidden]
    out_bias: torch.Tensor
    slot_mean: torch.Tensor  # [4]
    slot_std: torch.Tensor
    slot_keep: torch.Tensor
    edge_scale: torch.Tensor  # [3 * hidden + 4]
    edge_shift: torch.Tensor
    edge_weight: torch.Tensor  # [hidden, 3 * hidden + 4]
    edge_bias: torch.Tensor
    block_scale: torch.Tensor  # [hidden]
    block_shift: torch.Tensor
    up_weight: torch.Tensor  # [up, hidden]
    up_bias: torch.Tensor
    down_weight: torch.Tensor  # [hidden, up]
    down_bias: torch.Tensor
    final_scale: torch.Tensor  # [hidden]
    final_shift: torch.Tensor
    delta_weight: torch.Tensor  # [hidden]
    delta_bias: torch.Tensor  # [1]
    output_weight: torch.Tensor  # [3, hidden]: rows alpha, beta (the third is unused)
    output_bias: torch.Tensor
    token_table: torch.Tensor  # [vocab, hidden]
    hidden: int
    pool_hidden: int
    up: int
    cap: float
    eps: float

    @classmethod
    def pack(cls, head) -> VectorKernelWeights:
        """From a loaded VectorConfidenceHead (its config, weights, slot_keep and token table)."""
        if not supports(head.config):
            raise ValueError("The fused kernel does not compute this confidence head.")
        weights = head.weights

        def take(name: str) -> torch.Tensor:
            return weights[name].float().contiguous()

        hidden = int(head.config["hidden"])
        pool_hidden = int(weights["pool_set.entry.weight"].shape[0])
        up = int(weights["edge_mlp.blocks.0.up.weight"].shape[0])
        if weights["pool_set.entry.weight"].shape[1] != len(POOL_INPUTS):
            raise ValueError("The fused kernel reads three numbers per pool candidate.")
        if weights["edge_mlp.entry.weight"].shape != (
            hidden,
            3 * hidden + SLOT_NUMBERS,
        ):
            raise ValueError(
                "The fused kernel needs the edge input [z_u; e_c; z_u * e_c; slot]."
            )
        if head.token_table.shape[1] != hidden:
            raise ValueError("The fused kernel needs token vectors as wide as z_u.")
        if pool_hidden & (pool_hidden - 1) or up < 16 or up & (up - 1):
            raise ValueError(
                "The fused kernel needs a power-of-two pool width and an edge block of at "
                "least 16."
            )
        return cls(
            entry_weight=take("pool_set.entry.weight"),
            entry_bias=take("pool_set.entry.bias"),
            hidden_weight=take("pool_set.hidden.weight"),
            hidden_bias=take("pool_set.hidden.bias"),
            out_weight=take("pool_set.out.weight"),
            out_bias=take("pool_set.out.bias"),
            slot_mean=take("slot_mean"),
            slot_std=take("slot_std"),
            slot_keep=head.slot_keep.float().contiguous(),
            edge_scale=take("edge_mlp.input_norm.scale"),
            edge_shift=take("edge_mlp.input_norm.shift"),
            edge_weight=take("edge_mlp.entry.weight"),
            edge_bias=take("edge_mlp.entry.bias"),
            block_scale=take("edge_mlp.blocks.0.norm.scale"),
            block_shift=take("edge_mlp.blocks.0.norm.shift"),
            up_weight=take("edge_mlp.blocks.0.up.weight"),
            up_bias=take("edge_mlp.blocks.0.up.bias"),
            down_weight=take("edge_mlp.blocks.0.down.weight"),
            down_bias=take("edge_mlp.blocks.0.down.bias"),
            final_scale=take("edge_mlp.final_norm.scale"),
            final_shift=take("edge_mlp.final_norm.shift"),
            delta_weight=take("edge_output.weight").reshape(-1),
            delta_bias=take("edge_output.bias").reshape(-1),
            output_weight=take("output.weight"),
            output_bias=take("output.bias"),
            token_table=head.token_table.float().contiguous(),
            hidden=hidden,
            pool_hidden=pool_hidden,
            up=up,
            cap=float(head.cap),
            eps=float(head.eps),
        )


class RoundState(msgspec.Struct, frozen=True):
    """What the rounds keep for the final prune, one row per construction slot."""

    # [bs, slots, hidden] f32: z_u.
    nodes: torch.Tensor
    # [bs, slots, C, hidden] f32: the children's token vectors.
    children: torch.Tensor
    # [bs, slots, C, 4] f32: their standardised slot numbers.
    slots: torch.Tensor
    # [bs, slots, C] long: their indices in the slot's pool row.
    child_indices: torch.Tensor
    # [bs, slots, C] bool: which are real.
    child_real: torch.Tensor

    @classmethod
    def zeros(
        cls, bs: int, slots: int, children: int, hidden: int, device: torch.device
    ) -> RoundState:
        return cls(
            nodes=torch.zeros((bs, slots, hidden), dtype=torch.float32, device=device),
            children=torch.zeros(
                (bs, slots, children, hidden), dtype=torch.float32, device=device
            ),
            slots=torch.zeros(
                (bs, slots, children, SLOT_NUMBERS), dtype=torch.float32, device=device
            ),
            child_indices=torch.zeros(
                (bs, slots, children), dtype=torch.long, device=device
            ),
            child_real=torch.zeros(
                (bs, slots, children), dtype=torch.bool, device=device
            ),
        )


def child_positions(
    tokens: torch.Tensor,
    parents: torch.Tensor,
    node_mask: torch.Tensor,
    frontier_tokens: torch.Tensor,
    children: int,
) -> torch.Tensor:
    """The source holding every child of every construction slot, [bs, slots, C].

    Sources are the slots, then the frontier, where entry slot * C + rank is that slot's child of
    that rank. A child still in the frontier sits at its entry; once expanded it sits at the slot it
    went to. Its rank is found by its token among its parent's C entries, which the frontier keeps
    after the selection (pool tokens are distinct). tokens, parents and node_mask are the slots'
    [bs, slots], frontier_tokens the frontier's [bs, slots * C]."""
    bs, slots = tokens.shape
    device = tokens.device
    ranks = torch.arange(children, device=device)
    entries = slots + torch.arange(slots, device=device)[:, None] * children + ranks
    entries = entries.reshape(1, slots * children).expand(bs, -1)
    safe = parents.clamp_min(0)
    siblings = frontier_tokens.view(bs, slots, children).gather(
        1, safe[..., None].expand(-1, -1, children)
    )
    rank = (siblings == tokens[..., None]).int().argmax(dim=-1)
    moved = node_mask & (parents >= 0)
    # Every expanded slot goes to its entry under its parent; the rest to one spare column.
    targets = torch.where(
        moved, safe * children + rank, torch.full_like(safe, slots * children)
    )
    table = torch.cat((entries, entries.new_zeros((bs, 1))), dim=1).scatter(
        1, targets, torch.arange(slots, device=device).expand(bs, -1)
    )
    return table[:, : slots * children].view(bs, slots, children)


@triton.jit
def _layer_norm_rows(values, scale, shift, width: tl.constexpr, eps):
    mean = tl.sum(values, axis=1) / width
    centred = values - mean[:, None]
    variance = tl.sum(centred * centred, axis=1) / width
    return (
        centred * (1.0 / tl.sqrt(variance + eps))[:, None] * scale[None, :]
        + shift[None, :]
    )


@triton.jit
def _softplus(values):
    return tl.maximum(values, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(values)))


# slot_start changes every round: unspecialised, one compiled variant serves them all.
@triton.jit(do_not_specialize=["slot_start"])
def _vector_expand_kernel(
    slot_logits_ptr,
    pool_logits_ptr,
    pool_gumbels_ptr,
    pool_tokens_ptr,
    draft_rows_ptr,
    depths_ptr,
    node_scores_ptr,
    frontier_tokens_ptr,
    frontier_active_ptr,
    token_table_ptr,
    entry_weight_ptr,
    entry_bias_ptr,
    hidden_weight_ptr,
    hidden_bias_ptr,
    out_weight_ptr,
    out_bias_ptr,
    slot_mean_ptr,
    slot_std_ptr,
    slot_keep_ptr,
    edge_scale_ptr,
    edge_shift_ptr,
    edge_weight_ptr,
    edge_bias_ptr,
    block_scale_ptr,
    block_shift_ptr,
    up_weight_ptr,
    up_bias_ptr,
    down_weight_ptr,
    down_bias_ptr,
    final_scale_ptr,
    final_shift_ptr,
    delta_weight_ptr,
    delta_bias_ptr,
    output_weight_ptr,
    output_bias_ptr,
    frontier_scores_ptr,
    nodes_ptr,
    children_ptr,
    numbers_ptr,
    child_indices_ptr,
    child_real_ptr,
    slot_start,
    cap,
    eps,
    NUM_NODES: tl.constexpr,
    DEPTH: tl.constexpr,
    POOL: tl.constexpr,
    BLOCK_POOL: tl.constexpr,
    WIDTH: tl.constexpr,
    CHILDREN: tl.constexpr,
    HIDDEN: tl.constexpr,
    POOL_HIDDEN: tl.constexpr,
    UP: tl.constexpr,
    ROWS: tl.constexpr,
    SLOT: tl.constexpr,
    FLOOR: tl.constexpr,
):
    program = tl.program_id(0)
    batch = program // WIDTH
    slot = slot_start + program - batch * WIDTH
    node = batch * NUM_NODES + slot
    depth = tl.load(depths_ptr + node)
    row = tl.minimum(tl.maximum(depth, 0), DEPTH - 1)
    pool = tl.arange(0, BLOCK_POOL)
    in_pool = pool < POOL
    row_base = (batch * DEPTH + row) * POOL
    logits = tl.load(
        slot_logits_ptr + node * POOL + pool, mask=in_pool, other=-float("inf")
    )
    draft = tl.load(
        pool_logits_ptr + row_base + pool, mask=in_pool, other=-float("inf")
    )
    gumbels = tl.load(pool_gumbels_ptr + row_base + pool, mask=in_pool, other=0.0)
    tokens = tl.load(pool_tokens_ptr + row_base + pool, mask=in_pool, other=-1)

    # Log-softmax over the pool as torch.log_softmax forms it, clamped as the head's _log_probs.
    draft_max = tl.max(draft, axis=0)
    draft_shifted = draft - draft_max
    draft_sum = tl.sum(tl.where(in_pool, tl.exp(draft_shifted), 0.0), axis=0)
    log_a = tl.maximum(draft_shifted - tl.log(draft_sum), FLOOR)
    logit_max = tl.max(logits, axis=0)
    logit_shifted = logits - logit_max
    logit_exp = tl.where(in_pool, tl.exp(logit_shifted), 0.0)
    logit_sum = tl.sum(logit_exp, axis=0)
    log_l = tl.maximum(logit_shifted - tl.log(logit_sum), FLOOR)
    pi_l = logit_exp / logit_sum

    # The pool set encoder: three numbers per candidate, dense + SiLU, dense; pi_l-weighted mean and max.
    unit = tl.arange(0, POOL_HIDDEN)
    pool_weaver = log_l - log_a
    # The second layer as POOL_HIDDEN rank-one updates, one unit of the first at a time: any width works.
    per_candidate = (
        tl.zeros((BLOCK_POOL, POOL_HIDDEN), dtype=tl.float32)
        + tl.load(hidden_bias_ptr + unit)[None, :]
    )
    for inner in tl.static_range(POOL_HIDDEN):
        first = (
            log_a * tl.load(entry_weight_ptr + inner * 3)
            + pool_weaver * tl.load(entry_weight_ptr + inner * 3 + 1)
            + gumbels * tl.load(entry_weight_ptr + inner * 3 + 2)
            + tl.load(entry_bias_ptr + inner)
        )
        first = first * tl.sigmoid(first)
        per_candidate += (
            first[:, None]
            * tl.load(hidden_weight_ptr + unit * POOL_HIDDEN + inner)[None, :]
        )
    mean = tl.sum(pi_l[:, None] * per_candidate, axis=0)
    peak = tl.max(tl.where(in_pool[:, None], per_candidate, -float("inf")), axis=0)

    width = tl.arange(0, HIDDEN)
    out_rows = out_weight_ptr + width[:, None] * (2 * POOL_HIDDEN)
    summary = (
        tl.sum(tl.load(out_rows + unit[None, :]) * mean[None, :], axis=1)
        + tl.sum(
            tl.load(out_rows + POOL_HIDDEN + unit[None, :]) * peak[None, :], axis=1
        )
        + tl.load(out_bias_ptr + width)
    )
    z = tl.load(draft_rows_ptr + (batch * DEPTH + row) * HIDDEN + width) + summary
    tl.store(nodes_ptr + node * HIDDEN + width, z)
    alpha = tl.minimum(
        _softplus(
            tl.sum(tl.load(output_weight_ptr + width) * z, axis=0)
            + tl.load(output_bias_ptr)
        ),
        cap,
    )
    beta = tl.minimum(
        _softplus(
            tl.sum(tl.load(output_weight_ptr + HIDDEN + width) * z, axis=0)
            + tl.load(output_bias_ptr + 1)
        ),
        cap,
    )

    # The slot's children: the frontier entries the publish just wrote, slot * C + rank.
    child = tl.arange(0, ROWS)
    is_child = child < CHILDREN
    entry = batch * (NUM_NODES * CHILDREN) + slot * CHILDREN + child
    child_tokens = tl.load(frontier_tokens_ptr + entry, mask=is_child, other=-1)
    real = is_child & (
        tl.load(frontier_active_ptr + entry, mask=is_child, other=0) != 0
    )
    match = (
        (tokens[None, :] == child_tokens[:, None]) & in_pool[None, :] & real[:, None]
    )
    # Pool tokens are distinct, so a real child matches once; a child that is not real reads index 0.
    index = tl.sum(tl.where(match, pool[None, :], 0), axis=1)
    at_child = pool[None, :] == index[:, None]
    child_draft = tl.sum(tl.where(at_child, draft[None, :], 0.0), axis=1)
    rank = tl.sum(
        tl.where((draft[None, :] > child_draft[:, None]) & in_pool[None, :], 1.0, 0.0),
        axis=1,
    )
    raw = tl.where(
        width[None, :] == 0,
        tl.sum(tl.where(at_child, log_a[None, :], 0.0), axis=1)[:, None],
        tl.where(
            width[None, :] == 1,
            tl.sum(tl.where(at_child, log_l[None, :], 0.0), axis=1)[:, None],
            tl.where(
                width[None, :] == 2,
                tl.sum(tl.where(at_child, gumbels[None, :], 0.0), axis=1)[:, None],
                tl.where(width[None, :] == 3, tl.log(1.0 + rank)[:, None], 0.0),
            ),
        ),
    )
    is_number = width < SLOT
    numbers = (
        (raw - tl.load(slot_mean_ptr + width, mask=is_number, other=0.0)[None, :])
        / tl.load(slot_std_ptr + width, mask=is_number, other=1.0)[None, :]
        * tl.load(slot_keep_ptr + width, mask=is_number, other=0.0)[None, :]
    )
    vector_tokens = tl.sum(tl.where(at_child, tokens[None, :], 0), axis=1)
    vectors = tl.load(
        token_table_ptr + vector_tokens[:, None] * HIDDEN + width[None, :],
        mask=is_child[:, None],
        other=0.0,
    )

    # The edge MLP over [z_u; e_c; z_u * e_c; slot] (layer norm over all 3H + 4 columns), one block, layer norm.
    nodes_rows = tl.zeros((ROWS, HIDDEN), dtype=tl.float32) + z[None, :]
    product = nodes_rows * vectors
    columns = 3 * HIDDEN + SLOT
    total = (
        tl.sum(nodes_rows, axis=1)
        + tl.sum(vectors, axis=1)
        + tl.sum(product, axis=1)
        + tl.sum(numbers, axis=1)
    )
    centre = total / columns
    spread = (
        tl.sum((nodes_rows - centre[:, None]) * (nodes_rows - centre[:, None]), axis=1)
        + tl.sum((vectors - centre[:, None]) * (vectors - centre[:, None]), axis=1)
        + tl.sum((product - centre[:, None]) * (product - centre[:, None]), axis=1)
        + tl.sum(
            tl.where(
                is_number[None, :],
                (numbers - centre[:, None]) * (numbers - centre[:, None]),
                0.0,
            ),
            axis=1,
        )
    )
    inverse = (1.0 / tl.sqrt(spread / columns + eps))[:, None]
    weight_rows = edge_weight_ptr + width[None, :] * columns + width[:, None]
    edge = tl.load(edge_bias_ptr + width)[None, :] + tl.dot(
        (nodes_rows - centre[:, None])
        * inverse
        * tl.load(edge_scale_ptr + width)[None, :]
        + tl.load(edge_shift_ptr + width)[None, :],
        tl.load(weight_rows),
        input_precision="ieee",
    )
    edge += tl.dot(
        (vectors - centre[:, None])
        * inverse
        * tl.load(edge_scale_ptr + HIDDEN + width)[None, :]
        + tl.load(edge_shift_ptr + HIDDEN + width)[None, :],
        tl.load(weight_rows + HIDDEN),
        input_precision="ieee",
    )
    edge += tl.dot(
        (product - centre[:, None])
        * inverse
        * tl.load(edge_scale_ptr + 2 * HIDDEN + width)[None, :]
        + tl.load(edge_shift_ptr + 2 * HIDDEN + width)[None, :],
        tl.load(weight_rows + 2 * HIDDEN),
        input_precision="ieee",
    )
    edge += tl.dot(
        (numbers - centre[:, None])
        * inverse
        * tl.load(edge_scale_ptr + 3 * HIDDEN + width, mask=is_number, other=0.0)[
            None, :
        ]
        + tl.load(edge_shift_ptr + 3 * HIDDEN + width, mask=is_number, other=0.0)[
            None, :
        ],
        tl.load(weight_rows + 3 * HIDDEN, mask=is_number[:, None], other=0.0),
        input_precision="ieee",
    )
    normed = _layer_norm_rows(
        edge,
        tl.load(block_scale_ptr + width),
        tl.load(block_shift_ptr + width),
        HIDDEN,
        eps,
    )
    up = tl.arange(0, UP)
    raised = (
        tl.dot(
            normed,
            tl.load(up_weight_ptr + up[None, :] * HIDDEN + width[:, None]),
            input_precision="ieee",
        )
        + tl.load(up_bias_ptr + up)[None, :]
    )
    raised = raised * tl.sigmoid(raised)
    edge += (
        tl.dot(
            raised,
            tl.load(down_weight_ptr + width[None, :] * UP + up[:, None]),
            input_precision="ieee",
        )
        + tl.load(down_bias_ptr + width)[None, :]
    )
    edge = _layer_norm_rows(
        edge,
        tl.load(final_scale_ptr + width),
        tl.load(final_shift_ptr + width),
        HIDDEN,
        eps,
    )
    delta = tl.sum(edge * tl.load(delta_weight_ptr + width)[None, :], axis=1) + tl.load(
        delta_bias_ptr
    )
    delta = tl.where(real, delta, 0.0)

    # log q over the whole pool, delta on the real children only; their log r into the frontier.
    scores = alpha * logits + beta * gumbels
    for rank_index in tl.static_range(CHILDREN):
        own = child == rank_index
        own_index = tl.sum(tl.where(own, index, 0), axis=0)
        own_delta = tl.sum(tl.where(own, delta, 0.0), axis=0)
        scores = tl.where(pool == own_index, scores + own_delta, scores)
    top = tl.max(tl.where(in_pool, scores, -float("inf")), axis=0)
    normaliser = top + tl.log(
        tl.sum(tl.where(in_pool, tl.exp(scores - top), 0.0), axis=0)
    )
    log_q = tl.sum(tl.where(at_child, scores[None, :], 0.0), axis=1) - normaliser
    log_r = tl.load(node_scores_ptr + node) + tl.minimum(log_q, 0.0)
    tl.store(frontier_scores_ptr + entry, log_r, mask=real)

    kept = node * CHILDREN + child
    tl.store(child_indices_ptr + kept, index.to(tl.int64), mask=is_child)
    tl.store(child_real_ptr + kept, real, mask=is_child)
    tl.store(
        children_ptr + kept[:, None] * HIDDEN + width[None, :],
        vectors,
        mask=is_child[:, None],
    )
    tl.store(
        numbers_ptr + kept[:, None] * SLOT + width[None, :],
        numbers,
        mask=is_child[:, None] & is_number[None, :],
    )


def expand_round(
    weights: VectorKernelWeights,
    *,
    slot_start: int,
    width: int,
    slot_logits: torch.Tensor,
    pool_logits: torch.Tensor,
    pool_gumbels: torch.Tensor,
    pool_tokens: torch.Tensor,
    draft_rows: torch.Tensor,
    depths: torch.Tensor,
    node_scores: torch.Tensor,
    frontier_tokens: torch.Tensor,
    frontier_active: torch.Tensor,
    frontier_scores: torch.Tensor,
    state: RoundState,
) -> None:
    """The local part of slots [slot_start, slot_start + width) after their children are published:
    frontier_scores of their real children and their rows of state, in place.

    slot_logits [bs, slots, pool] f32, pool_* [bs, depth, pool] (logits and Gumbel field f32),
    draft_rows [bs, depth, hidden] f32 (step_constants), depths and node_scores [bs, slots] (a slot's
    log r, 0 at the root), the frontier [bs, slots * C]."""
    bs, num_nodes, pool = slot_logits.shape
    children = state.child_indices.shape[2]
    _vector_expand_kernel[(bs * width,)](
        slot_logits,
        pool_logits,
        pool_gumbels,
        pool_tokens,
        draft_rows,
        depths,
        node_scores,
        frontier_tokens,
        frontier_active,
        weights.token_table,
        weights.entry_weight,
        weights.entry_bias,
        weights.hidden_weight,
        weights.hidden_bias,
        weights.out_weight,
        weights.out_bias,
        weights.slot_mean,
        weights.slot_std,
        weights.slot_keep,
        weights.edge_scale,
        weights.edge_shift,
        weights.edge_weight,
        weights.edge_bias,
        weights.block_scale,
        weights.block_shift,
        weights.up_weight,
        weights.up_bias,
        weights.down_weight,
        weights.down_bias,
        weights.final_scale,
        weights.final_shift,
        weights.delta_weight,
        weights.delta_bias,
        weights.output_weight,
        weights.output_bias,
        frontier_scores,
        state.nodes,
        state.children,
        state.slots,
        state.child_indices,
        state.child_real,
        int(slot_start),
        weights.cap,
        weights.eps,
        NUM_NODES=int(num_nodes),
        DEPTH=int(pool_logits.shape[1]),
        POOL=int(pool),
        BLOCK_POOL=int(triton.next_power_of_2(pool)),
        WIDTH=int(width),
        CHILDREN=int(children),
        HIDDEN=weights.hidden,
        POOL_HIDDEN=weights.pool_hidden,
        UP=weights.up,
        ROWS=EDGE_ROWS,
        SLOT=SLOT_NUMBERS,
        FLOOR=LOG_PROB_FLOOR,
    )
