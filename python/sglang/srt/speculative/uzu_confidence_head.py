"""Confidence head for the weaver_uzu final prune.

The prune keeps the B nodes with the highest score. Without the head that score is the
cumulative path log-weight (at --...-uzu-prune-sigma, with the shared Gumbel field mixed
in). With the head it is log r(n), the head's estimate of the log-probability that node n
lies on the target's path. For every expanded node u the head predicts the target's choice
over u's candidate pool,

    log q_u(t) = log_softmax_t(alpha_u * l_ut + beta_u * G_t + delta_ut),

and log r(n) is the sum of min(log q, 0) along the path to n (0 at the root). Here
l = a + rho is the pool logit the tree was grown with (drafter logit plus Weaver residual)
and G the shared Gumbel field at the children's position, the one the target samples with.
alpha_u and beta_u are positive outputs of a small network over u's own features and the
drafter hidden state of u's depth; delta_ut is a per-child correction, non-zero only on the
C children of u. Scores never grow along a path, so the kept set stays closed under
ancestors.

A head of format 2 first passes the tree through attention: every present source is a token
(its child vector, plus the node part for an expanded slot, plus a depth embedding), pre-LN
layers attend over all tokens of the step with a learned bias per pair relation (self,
parent, child, ancestor, descendant, sibling and optionally the same vocabulary token), and
the result corrects the node part of every slot and the child vectors before alpha, beta and
delta are computed (confidence_head/model/tree.py). Its local part, the same head without the
attention corrections, scores a child from its parent and path alone; when the server expands by
the head, that part ranks the frontier after every round (local_scores).

A head of format 3 (no tree attention) also has a coverage branch for the expansion: kappa(n), the
probability that the target's choice after n is among n's C children, given that n is reached. It
reads only what is known before n is expanded: n's node part computed as a child (its parent's pool,
its path and the drafter hidden state of its own depth), n's child vector and three scalars of the
pool its children will come from (drafter logits and the shared Gumbel field of that depth). The
expansion priority is then e(n) = log r(n) + log kappa(n), the log-probability that expanding n adds
a node of the target's path (expansion_scores).

A head of format 4 (no tree attention, no coverage branch) reads two more inputs in the node part
of a slot u: the embedding of u's own token (the step's root token at the root), through the
children's token projection and a dense layer of its own, and the drafter hidden state of the depth
after u's (row d(u) + 1, the position of u's children's children; zero for a slot on the last row),
with its own layer norm and projection. Neither depends on the rest of the tree, so a node keeps its
score as the tree grows, and the expansion can rank by it.

A head of format 5 is the vector head (confidence_head/model/vector.py): the node part reads the
drafter hidden state of the slot's depth (layer norm, projection) and a summary of the slot's pool
(a small network over every candidate's log pi_a, log pi_l - log pi_a and G, pooled by the
pi_l-weighted mean and the maximum), joined by concatenation or a sum and passed through an MLP
(residual blocks with layer norms, or plain dense + SiLU layers) to z_u, or summed into z_u with no
node layer at all (node_depth null: the export folds a single dense node layer into the input
projections, which is the same function); alpha and beta come from z_u
(or from an output MLP of it). A child's delta reads its token's projected embedding e_c and four
standardised slot numbers (log pi_a(c), log pi_l(c), G_c, log(1 + drafter rank)): either an edge MLP
over [z_u; e_c; z_u * e_c; slot], or a child MLP over [e_c; slot] merged with z_u (sum, concatenation,
product or both). Optionally a SiLU follows every input projection. With a tree section, attention
over the step's sources (tokens: projections of z_u and of the per-child inputs) corrects z_u and e_c
before the outputs; its local part (no corrections) ranks the expansion.

A vector head also has an incremental path for the server. The local part of a slot reads only its
own drafter row, its own pool and its own children, so a round needs it only for the slots it has
just expanded: step_constants once per step, expand_slots per round (or the fused kernel of
uzu_confidence_kernel), and prune_expanded once at the end, which runs the attention and the
outputs on the kept z_u, child vectors and slot numbers. It computes the same function as
local_scores and prune_scores on the whole tree.

The head is trained offline on rollouts dumped by uzu_tree_dump (research repository:
Projects/Tree Construction, confidence_head/; exported by confidence_head.tools.export_head).
This module mirrors its definitions in confidence_head/model/features.py, model/head.py and
train/loss.py (log_q, log_reach).

Everything is a fixed-shape tensor computation without host synchronisation, so it runs
inside the tree's CUDA graph. The module imports only torch, msgspec and safetensors, so
offline tests can load it by file path.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import math
import operator
from collections.abc import Iterator
from pathlib import Path

import msgspec
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

FORMAT = 1
TREE_FORMAT = 2
EXPANSION_FORMAT = 3
NODE_INPUTS_FORMAT = 4
VECTOR_FORMAT = 5
FORMATS = (FORMAT, TREE_FORMAT, EXPANSION_FORMAT, NODE_INPUTS_FORMAT)
LOG_PROB_FLOOR = -30.0
EXPANSION_FEATURES = 3
WEIGHTS = "head.safetensors"
CONFIG = "head.json"
SUPPORTED_BLOCKS = frozenset({"scalars", "draft", "draft_next"})
# Node scalars the head can compute at prune time: the "noise" and "path" groups. The
# "pool" group needs the drafter's log-normaliser over the vocabulary and the "query" group
# the request's acceptance history, so a head that keeps any of them is refused at load.
COMPUTED_NODE_FEATURES = frozenset(
    {
        "leader_pi",
        "leader_gap",
        "leader_is_top",
        "leader_logit_gap",
        "depth",
        "path_log_weight",
        "is_leader",
        "rank_z",
        "rank_l",
    }
)
TOKEN_TABLE_CHUNK = 16384
TREE_RELATIONS = 6


class TreeInputs(msgspec.Struct, frozen=True):
    """One step's grown tree, before the final prune, as the head reads it.

    Sources are the construction slots followed by the frontier, in the order the prune
    sorts them; parents index the slots. Rows of slots that were not expanded (absent, or
    at the depth cap) are only ever read through masks.
    """

    # [bs, slots, pool] f32: l = a + rho of every construction slot.
    slot_logits: torch.Tensor
    # [bs, depth, pool]: drafter logits a of the candidate pool of each depth.
    pool_logits: torch.Tensor
    # [bs, depth, pool] f32: shared Gumbel field on the pool of each depth.
    pool_gumbels: torch.Tensor
    # [bs, depth, pool] long: candidate tokens of the pool of each depth.
    pool_tokens: torch.Tensor
    # [bs, depth, width]: drafter hidden state behind the pool of each depth.
    draft_hidden: torch.Tensor
    # [bs, sources] long: source tokens.
    tokens: torch.Tensor
    # [bs, sources] long: parent slot, -1 at the root and for absent sources.
    parents: torch.Tensor
    # [bs, sources] long: source depths.
    depths: torch.Tensor
    # [bs, sources] bool: sources holding a node.
    present: torch.Tensor


class StepConstants(msgspec.Struct, frozen=True):
    """What every slot of a step reads from the drafter, computed once after it
    (VectorConfidenceHead.step_constants)."""

    # [bs, depth, hidden] f32: the drafter-row term of z_u for a slot of each depth; None for a
    # head without the drafter row.
    draft_rows: torch.Tensor | None
    # [bs, depth, pool] f32: drafter logits a of the candidate pool of each depth.
    pool_logits: torch.Tensor
    # [bs, depth, pool] f32: shared Gumbel field on the pool of each depth.
    pool_gumbels: torch.Tensor
    # [bs, depth, pool] long: candidate tokens of the pool of each depth.
    pool_tokens: torch.Tensor


class SlotExpansion(msgspec.Struct, frozen=True):
    """The head's local part on the slots one round expands (VectorConfidenceHead.expand_slots)."""

    # [bs, width, hidden] f32: z_u of every slot.
    nodes: torch.Tensor
    # [bs, width, C, token] f32: the token vectors of its children.
    children: torch.Tensor
    # [bs, width, C, 4] f32: their standardised slot numbers; None for a head without them.
    slots: torch.Tensor | None
    # [bs, width, C] f32: log q_u of every child under the local part.
    log_q: torch.Tensor
    # [bs, width, C] f32: log r of every child, the slot's log r plus min(log q, 0); -inf for a
    # child that is not real.
    log_r: torch.Tensor


class ExpandedTree(msgspec.Struct, frozen=True):
    """A grown tree whose slots keep the local part of the round that expanded them, as the final
    prune reads it (VectorConfidenceHead.prune_expanded). Sources are those of TreeInputs: the
    construction slots, then the frontier."""

    # [bs, slots, pool] f32: l = a + rho of every construction slot.
    slot_logits: torch.Tensor
    # [bs, slots, hidden] f32: z_u of every slot (any finite value for a slot without children).
    nodes: torch.Tensor
    # [bs, slots, C, token] f32: the token vectors of the C children of every slot.
    children: torch.Tensor
    # [bs, slots, C, 4] f32 or None: their standardised slot numbers.
    slots: torch.Tensor | None
    # [bs, slots, C] long: each child's index in its parent's pool row.
    child_indices: torch.Tensor
    # [bs, slots, C] bool: which children are real.
    child_real: torch.Tensor
    # [bs, slots, C] long: the source holding each real child (its slot once expanded, else its
    # frontier entry); read only where real.
    child_positions: torch.Tensor
    # [bs, sources] long / long / long / bool: as in TreeInputs.
    tokens: torch.Tensor
    parents: torch.Tensor
    depths: torch.Tensor
    present: torch.Tensor


@contextlib.contextmanager
def _full_fp32() -> Iterator[None]:
    """fp32 matmuls without TF32, so the head matches the exported model's arithmetic."""
    precision = torch.get_float32_matmul_precision()
    cudnn = torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cudnn.allow_tf32 = cudnn


def _layer_norm(
    values: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float
) -> torch.Tensor:
    mean = values.mean(dim=-1, keepdim=True)
    variance = (values - mean).square().mean(dim=-1, keepdim=True)
    return (values - mean) * torch.rsqrt(variance + eps) * scale + shift


def _softplus(values: torch.Tensor) -> torch.Tensor:
    return torch.logaddexp(values, torch.zeros_like(values))


def _among_siblings(
    values: torch.Tensor, gap_cap: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rank (siblings strictly above), gap to the best sibling, gap to the nearest below."""
    own = values[..., :, None]
    other = values[..., None, :]
    rank = (other > own).sum(dim=-1).float()
    lead = values.amax(dim=-1, keepdim=True) - values
    below = torch.where(other < own, other, torch.full_like(other, -torch.inf))
    return rank, lead, torch.clamp(values - below.amax(dim=-1), max=gap_cap)


def _rows(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """values[b, index[b, ...]] for a per-batch index of any trailing shape."""
    batch = torch.arange(values.shape[0], device=values.device)
    return values[batch.view(-1, *([1] * (index.dim() - 1))), index]


def pool_indices(
    pool_tokens: torch.Tensor, tokens: torch.Tensor, depths: torch.Tensor
) -> torch.Tensor:
    """Each source's index in the pool row it came from (its parent's depth); 0 at the root.

    Tokens of a pool row are distinct (the drafter's top-k over the vocabulary), so the
    match is unique.
    """
    rows = pool_tokens.shape[1]
    row_tokens = _rows(pool_tokens, (depths - 1).clamp(0, rows - 1))
    return (row_tokens == tokens[..., None]).int().argmax(dim=-1)


def child_sources(
    parents: torch.Tensor, present: torch.Tensor, slots: int, children: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Source positions of the children of every slot, [bs, slots, C], and which are real.

    The children are the tree's own: the present sources whose parent is the slot, the
    first C in source order. An expanded slot has exactly C of them as long as every
    candidate of its pool is a valid token (no drafter filters, which the server enforces
    with a head); the rest hold the number of sources and are masked.
    """
    bs, sources = parents.shape
    positions = torch.arange(sources, device=parents.device)
    earlier = positions[None, :] < positions[:, None]
    siblings = (
        (parents[:, :, None] == parents[:, None, :]) & present[:, None, :] & earlier
    )
    ranks = siblings.sum(dim=-1)
    real = present & (parents >= 0) & (ranks < children)
    # Each real child goes to bucket parent * C + rank; everything else to one spare bucket.
    buckets = torch.where(real, parents * children + ranks, slots * children)
    table = torch.full(
        (bs, slots * children + 1), sources, dtype=torch.long, device=parents.device
    )
    table = table.scatter(1, buckets, positions.expand(bs, -1))
    first = table[:, : slots * children].view(bs, slots, children)
    return first, first < sources


def child_indices(
    parents: torch.Tensor,
    present: torch.Tensor,
    indices: torch.Tensor,
    slots: int,
    children: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool indices of the children of every slot, [bs, slots, C], and which are real."""
    first, real = child_sources(parents, present, slots, children)
    return _rows(indices, first.clamp_max(parents.shape[1] - 1)), real


def tree_relations(
    parents: torch.Tensor,
    present: torch.Tensor,
    tokens: torch.Tensor,
    max_depth: int,
    same_token: bool,
) -> torch.Tensor:
    """Pair relations between sources, [bs, R, S, S] f32: self, parent, child, ancestor two
    or more edges up, descendant likewise, sibling and, with same_token, the same vocabulary
    token (the root excluded). Only between present sources; no source is deeper than
    max_depth, so max_depth steps up reach the root."""
    sources = parents.shape[1]
    index = torch.arange(sources, device=parents.device)
    is_parent = parents[:, :, None] == index
    ancestor = torch.zeros_like(is_parent)
    up = parents
    for _ in range(max_depth):
        ancestor = ancestor | (up[:, :, None] == index)
        up = torch.where(up >= 0, _rows(parents, up.clamp_min(0)), up)
    far = ancestor & ~is_parent
    same = torch.eye(sources, dtype=torch.bool, device=parents.device).expand_as(
        is_parent
    )
    rooted = parents >= 0
    sibling = (parents[:, :, None] == parents[:, None, :]) & rooted[:, :, None] & ~same
    pairs = [
        same,
        is_parent,
        is_parent.transpose(1, 2),
        far,
        far.transpose(1, 2),
        sibling,
    ]
    if same_token:
        pairs.append(
            (tokens[:, :, None] == tokens[:, None, :])
            & rooted[:, :, None]
            & rooted[:, None, :]
            & ~same
        )
    both = present[:, :, None] & present[:, None, :]
    return (torch.stack(pairs, dim=1) & both[:, None]).float()


def _jumps(parents: torch.Tensor, steps: int) -> list[torch.Tensor]:
    """Ancestors 1, 2, 4, ... levels up; the root (parent -1) is its own ancestor."""
    own = torch.arange(parents.shape[1], device=parents.device).expand_as(parents)
    jump = torch.where(parents < 0, own, parents)
    jumps = [jump]
    for _ in range(steps - 1):
        jump = _rows(jump, jump)
        jumps.append(jump)
    return jumps


def _path_sums(edges: torch.Tensor, jumps: list[torch.Tensor]) -> torch.Tensor:
    """Sum of the edges on the path from the root to every node (the root's edge is 0)."""
    sums = edges
    for jump in jumps:
        sums = sums + _rows(sums, jump)
    return sums


def _path_minima(values: torch.Tensor, jumps: list[torch.Tensor]) -> torch.Tensor:
    """Minimum of the values on the path from the root to every node. Exact: it makes the
    score of a node never exceed its parent's whatever the summation order was."""
    minima = values
    for jump in jumps:
        minima = torch.minimum(minima, _rows(minima, jump))
    return minima


def project_token_embedding(
    embedding: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """token_projection(token_norm(E)) for the whole vocabulary in f32, once at load."""
    with _full_fp32():
        parts = [
            _layer_norm(
                embedding[start : start + TOKEN_TABLE_CHUNK].float(), scale, shift, eps
            )
            @ weight.T
            + bias
            for start in range(0, embedding.shape[0], TOKEN_TABLE_CHUNK)
        ]
    return torch.cat(parts)


class ConfidenceHead:
    """An exported head on one device: weights, frozen feature statistics and, for a head
    with a token embedding, the projected target input embedding of every token."""

    def __init__(
        self,
        config: dict,
        weights: dict[str, torch.Tensor],
        token_table: torch.Tensor | None,
        source: str = "",
        compile: bool = False,
    ) -> None:
        if config.get("format") not in FORMATS:
            raise ValueError(
                f"Confidence head format {config.get('format')}, expected one of "
                f"{FORMATS}."
            )
        node_inputs = bool(config.get("node_token")) or "draft_next" in config["blocks"]
        if (config["format"] == NODE_INPUTS_FORMAT) != node_inputs:
            raise ValueError(
                f"A confidence head of format {NODE_INPUTS_FORMAT}, and only it, reads the "
                "node's token or the next drafter row."
            )
        if config.get("node_token") and config["child"] != "embedding":
            raise ValueError(
                "The node-token input shares the children's token projection."
            )
        if (config["format"] == TREE_FORMAT) != ("tree" in config):
            raise ValueError(
                f"A confidence head of format {TREE_FORMAT}, and only it, has a tree section."
            )
        if (config["format"] == EXPANSION_FORMAT) != ("expansion" in config):
            raise ValueError(
                f"A confidence head of format {EXPANSION_FORMAT}, and only it, has an "
                "expansion section."
            )
        if "tree" in config and config["child"] == "none":
            raise ValueError("Tree attention needs the per-child term.")
        if "expansion" in config and config["child"] == "none":
            raise ValueError("The coverage branch reads the per-child vectors.")
        if config["form"] != "b":
            raise ValueError("The confidence head supports form B only.")
        extra = set(config["blocks"]) - SUPPORTED_BLOCKS
        if extra:
            raise ValueError(
                f"The confidence head cannot compute blocks {sorted(extra)} at prune time."
            )
        kept = {
            name
            for name, keep in zip(
                config["node_features"], config["node_keep"], strict=True
            )
            if keep
        }
        if kept - COMPUTED_NODE_FEATURES:
            raise ValueError(
                "The confidence head keeps node features "
                f"{sorted(kept - COMPUTED_NODE_FEATURES)}, unavailable at prune time."
            )
        if config["child"] == "embedding" and token_table is None:
            raise ValueError(
                "A confidence head with a token embedding needs the target's input "
                "embedding."
            )
        self.config = config
        self.weights = weights
        self.token_table = token_table
        self.source = source
        self.linear = bool(config["linear"])
        self.cap = float(config["cap"])
        self.eps = float(config["layer_norm_eps"])
        self.gap_cap = float(config["gap_cap"])
        self.children = int(config["children"])
        self.max_depth = int(config["max_depth"])
        self.child = config["child"]
        self.node_token = bool(config.get("node_token", False))
        self.blocks = frozenset(config["blocks"])
        self.tree = config.get("tree")
        self.expansion = config.get("expansion")
        device = weights["node_mean"].device
        self.node_keep = torch.tensor(
            config["node_keep"], dtype=torch.float32, device=device
        )
        self.child_keep = torch.tensor(
            config["child_keep"], dtype=torch.float32, device=device
        )
        # Pointer jumping sums 2 ** steps edges along a path: a slot is at most max_depth
        # edges below the root, and the root's own edge is zero.
        self.jump_steps = max(1, math.ceil(math.log2(self.max_depth)))
        # Compiled once per server: fuses the head's many small kernels for the CUDA graph. The
        # local scores (no tree attention) rank the expansion; a head without attention has only those.
        self._scores_fn = (
            torch.compile(self._full_scores, dynamic=False, fullgraph=True)
            if compile
            else self._full_scores
        )
        if self.tree is None:
            self._local_scores_fn = self._scores_fn
        else:
            self._local_scores_fn = (
                torch.compile(self._local_scores, dynamic=False, fullgraph=True)
                if compile
                else self._local_scores
            )
        self._expansion_scores_fn = (
            torch.compile(self._expansion_scores, dynamic=False, fullgraph=True)
            if compile and self.expansion is not None
            else self._expansion_scores
        )

    @classmethod
    def load(
        cls,
        directory: str | Path,
        token_embedding: torch.Tensor | None,
        device: torch.device | str,
        compile: bool = False,
    ) -> ConfidenceHead:
        """Read head.json and head.safetensors (sha256-checked).

        token_embedding is the target's raw input embedding table [vocab, width], needed
        only by a head with a token embedding.
        """
        directory = Path(directory)
        config = json.loads((directory / CONFIG).read_text())
        digest = hashlib.sha256((directory / WEIGHTS).read_bytes()).hexdigest()
        if digest != config["weights_sha256"]:
            raise ValueError(
                f"{directory / WEIGHTS}: sha256 {digest} differs from the one in {CONFIG}."
            )
        weights = load_file(str(directory / WEIGHTS), device=str(device))
        if config.get("format") == VECTOR_FORMAT:
            return VectorConfidenceHead.load_vector(
                directory, config, weights, digest, token_embedding, device, compile
            )
        table = None
        if config["child"] == "embedding" and token_embedding is not None:
            table = project_token_embedding(
                token_embedding.to(device),
                weights["token_norm.scale"],
                weights["token_norm.shift"],
                weights["token_projection.weight"],
                weights["token_projection.bias"],
                float(config["layer_norm_eps"]),
            )
        source = f"{directory} sha256={digest}"
        return cls(config, weights, table, source=source, compile=compile)

    def check_tree(
        self, children_per_parent: int, max_depth: int, pool: int, width: int
    ) -> None:
        """The head is defined on the tree it was trained on; refuse any other at startup."""
        expected = (
            self.children,
            self.max_depth,
            int(self.config["pool"]),
            int(self.config["model_width"]),
        )
        actual = (children_per_parent, max_depth, pool, width)
        if actual != expected:
            raise ValueError(
                "The confidence head was trained with children per parent, max depth, "
                f"pool, drafter width = {expected}; the server runs {actual}."
            )

    def _dense(self, name: str, values: torch.Tensor) -> torch.Tensor:
        return values @ self.weights[f"{name}.weight"].T + self.weights[f"{name}.bias"]

    def _node_scalars(
        self,
        logits: torch.Tensor,
        gumbels: torch.Tensor,
        log_pi: torch.Tensor,
        parents: torch.Tensor,
        own_indices: torch.Tensor,
        depths: torch.Tensor,
        jumps: list[torch.Tensor],
    ) -> torch.Tensor:
        """Standardised node scalars of every slot, dropped columns zero."""
        scores = logits + gumbels
        is_root = parents < 0
        safe_parents = parents.clamp_min(0)
        zeros = torch.zeros_like(depths, dtype=torch.float32)
        # The "noise" group reads the slot's own pool.
        leaders = scores.argmax(dim=-1, keepdim=True)
        leader_logits = logits.gather(-1, leaders)[..., 0]
        top_logits = logits.amax(dim=-1)
        top2 = scores.topk(2, dim=-1).values
        computed = {
            "leader_pi": log_pi.gather(-1, leaders)[..., 0].exp(),
            "leader_gap": torch.clamp(top2[..., 0] - top2[..., 1], max=self.gap_cap),
            "leader_is_top": (leader_logits >= top_logits).float(),
            "leader_logit_gap": top_logits - leader_logits,
        }
        # The "path" group places the slot in its parent's pool and accumulates log pi.
        parent_scores = _rows(scores, safe_parents)
        parent_logits = _rows(logits, safe_parents)
        indices = own_indices[..., None]
        own_scores = parent_scores.gather(-1, indices)
        own_logits = parent_logits.gather(-1, indices)
        edges = _rows(log_pi, safe_parents).gather(-1, indices)[..., 0]
        edges = torch.where(is_root, zeros, edges)
        leader_of_parent = (parent_scores.argmax(dim=-1) == own_indices).float()
        computed |= {
            "depth": depths.float(),
            "path_log_weight": _path_sums(edges, jumps),
            "is_leader": torch.where(is_root, torch.ones_like(zeros), leader_of_parent),
            "rank_z": torch.where(
                is_root, zeros, (parent_scores > own_scores).sum(dim=-1).float()
            ),
            "rank_l": torch.where(
                is_root, zeros, (parent_logits > own_logits).sum(dim=-1).float()
            ),
        }
        raw = torch.stack(
            [computed.get(name, zeros) for name in self.config["node_features"]], dim=-1
        )
        standard = (raw - self.weights["node_mean"]) / self.weights["node_std"]
        return standard * self.node_keep

    def _child_vectors(
        self,
        children: torch.Tensor,
        draft: torch.Tensor,
        logits: torch.Tensor,
        gumbels: torch.Tensor,
        log_pi: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Vector of the C children of every slot: child features through child_input."""

        def take(values: torch.Tensor) -> torch.Tensor:
            return values.gather(-1, children)

        child_draft = take(draft)
        child_logits = take(logits)
        child_gumbels = take(gumbels)
        child_scores = child_logits + child_gumbels
        rank_l, lead_l, next_l = _among_siblings(child_logits, self.gap_cap)
        rank_z, lead_z, next_z = _among_siblings(child_scores, self.gap_cap)
        pool_ranks = (draft[..., None, :] > child_draft[..., None]).sum(dim=-1).float()
        features = torch.stack(
            [
                child_draft,
                child_logits - child_draft,
                child_logits,
                take(log_pi),
                child_gumbels,
                child_scores,
                rank_l,
                rank_z,
                lead_l,
                lead_z,
                next_l,
                next_z,
                torch.log1p(pool_ranks),
                (pool_ranks == 0).float(),
            ],
            dim=-1,
        )
        standard = (features - self.weights["child_mean"]) / self.weights["child_std"]
        parts = [standard * self.child_keep]
        if self.child == "embedding":
            assert self.token_table is not None
            parts.append(self.token_table[take(tokens)])
        return self._dense("child_input", torch.cat(parts, dim=-1))

    def _child_delta(self, hidden: torch.Tensor, child: torch.Tensor) -> torch.Tensor:
        """delta of the C children of every slot from the slot's node part and their vectors."""
        node = hidden[..., None, :].expand_as(child)
        inner = F.silu(
            self._dense("child_hidden", torch.cat([node, child, node * child], dim=-1))
        )
        return self._dense("child_output", inner)[..., 0]

    def _tree_layer(
        self, index: int, values: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        """One pre-LN layer: attention over all sources of the step, then the MLP."""
        prefix = f"tree.layers.{index}"
        bs, sources, width = values.shape
        heads = bias.shape[1]
        size = width // heads
        normed = _layer_norm(
            values,
            self.weights[f"{prefix}.attention_norm.scale"],
            self.weights[f"{prefix}.attention_norm.shift"],
            self.eps,
        )
        query, key, value = (
            part.reshape(bs, sources, heads, size).transpose(1, 2)
            for part in self._dense(f"{prefix}.qkv", normed).chunk(3, dim=-1)
        )
        logits = query @ key.transpose(-1, -2) / math.sqrt(size) + bias
        mixed = torch.softmax(logits, dim=-1) @ value
        values = values + self._dense(
            f"{prefix}.out", mixed.transpose(1, 2).reshape(bs, sources, width)
        )
        normed = _layer_norm(
            values,
            self.weights[f"{prefix}.mlp_norm.scale"],
            self.weights[f"{prefix}.mlp_norm.shift"],
            self.eps,
        )
        return values + self._dense(
            f"{prefix}.down", F.silu(self._dense(f"{prefix}.up", normed))
        )

    def _tree_shifts(
        self,
        hidden: torch.Tensor,
        child: torch.Tensor,
        first: torch.Tensor,
        real: torch.Tensor,
        parents: torch.Tensor,
        depths: torch.Tensor,
        present: torch.Tensor,
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Corrections of the node part of every slot and of the child vectors from attention
        over the tree: a source's token is its child vector, plus the node part if it is a
        slot with children (an expanded node of the training tree), plus a depth embedding.
        """
        assert self.tree is not None
        bs, slots, children, width = child.shape
        sources = parents.shape[1]
        # Every real child adds its vector at its source; the rest go to a spare row.
        rows = torch.where(real, first, torch.full_like(first, sources))
        values = child.new_zeros((bs, sources + 1, width)).scatter_add(
            1,
            rows.view(bs, slots * children, 1).expand(-1, -1, width),
            child.reshape(bs, slots * children, width),
        )
        node_part = torch.where(
            real.any(dim=-1, keepdim=True), hidden, torch.zeros_like(hidden)
        )
        values = torch.cat(
            (values[:, :slots] + node_part, values[:, slots:sources]), dim=1
        )
        values = (
            values
            + self.weights["tree.depth_embedding"][depths.clamp(0, self.max_depth)]
        )
        pairs = tree_relations(
            parents, present, tokens, self.max_depth, bool(self.tree["same_token"])
        )
        bias = torch.einsum(
            "ak,bknm->banm", self.weights["tree.relation_weight"], pairs
        )
        bias = bias.masked_fill(~present[:, None, None, :], -torch.inf)
        for index in range(int(self.tree["layers"])):
            values = self._tree_layer(index, values, bias)
        values = _layer_norm(
            values,
            self.weights["tree.final_norm.scale"],
            self.weights["tree.final_norm.shift"],
            self.eps,
        )
        node_shift = self._dense("tree.couple_node", values[:, :slots])
        child_shift = self._dense(
            "tree.couple_child", _rows(values, first.clamp_max(sources - 1))
        )
        return node_shift, child_shift

    def _log_q(
        self,
        slot_logits: torch.Tensor,
        pool_logits: torch.Tensor,
        pool_gumbels: torch.Tensor,
        pool_tokens: torch.Tensor,
        draft_hidden: torch.Tensor,
        tokens: torch.Tensor,
        parents: torch.Tensor,
        depths: torch.Tensor,
        present: torch.Tensor,
        indices: torch.Tensor,
        jumps: list[torch.Tensor],
        attention: bool = True,
    ) -> torch.Tensor:
        return self._outputs(
            slot_logits,
            pool_logits,
            pool_gumbels,
            pool_tokens,
            draft_hidden,
            tokens,
            parents,
            depths,
            present,
            indices,
            jumps,
            attention,
        )[0]

    def _outputs(
        self,
        slot_logits: torch.Tensor,
        pool_logits: torch.Tensor,
        pool_gumbels: torch.Tensor,
        pool_tokens: torch.Tensor,
        draft_hidden: torch.Tensor,
        tokens: torch.Tensor,
        parents: torch.Tensor,
        depths: torch.Tensor,
        present: torch.Tensor,
        indices: torch.Tensor,
        jumps: list[torch.Tensor],
        attention: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """log q over the pool of every slot, and for a head with the per-child term the vectors
        of every slot's C children and their pool indices."""
        slots = slot_logits.shape[1]
        rows = pool_logits.shape[1]
        logits = slot_logits.float()
        slot_depths = depths[:, :slots]
        slot_rows = slot_depths.clamp(0, rows - 1)
        draft = _rows(pool_logits.float(), slot_rows)
        gumbels = _rows(pool_gumbels.float(), slot_rows)
        log_pi = torch.log_softmax(logits, dim=-1)
        hidden_width = int(self.config["hidden"])
        pre = logits.new_zeros((*slot_depths.shape, hidden_width))
        if "scalars" in self.blocks:
            scalars = self._node_scalars(
                logits,
                gumbels,
                log_pi,
                parents[:, :slots],
                indices[:, :slots],
                slot_depths,
                jumps,
            )
            pre = pre + self._dense("scalars", scalars)
        if "draft" in self.blocks:
            normed = _layer_norm(
                draft_hidden.float(),
                self.weights["draft.norm.scale"],
                self.weights["draft.norm.shift"],
                self.eps,
            )
            pre = pre + _rows(self._dense("draft", normed), slot_rows)
        if "draft_next" in self.blocks:
            normed = _layer_norm(
                draft_hidden.float(),
                self.weights["draft_next.norm.scale"],
                self.weights["draft_next.norm.shift"],
                self.eps,
            )
            following = slot_depths + 1
            last = draft_hidden.shape[1] - 1
            taken = _rows(self._dense("draft_next", normed), following.clamp(max=last))
            pre = pre + torch.where(
                (following <= last)[..., None], taken, torch.zeros_like(taken)
            )
        if self.node_token:
            assert self.token_table is not None
            pre = pre + self._dense("node_token", self.token_table[tokens[:, :slots]])
        hidden = pre if self.linear else F.silu(pre + self.weights["hidden_bias"])
        deltas = None
        child = children = None
        if self.child != "none":
            children, valid = child_indices(
                parents, present, indices, slots, self.children
            )
            child = self._child_vectors(
                children,
                draft,
                logits,
                gumbels,
                log_pi,
                _rows(pool_tokens, slot_rows),
            )
            if self.tree is not None and attention:
                first, _ = child_sources(parents, present, slots, self.children)
                node_shift, child_shift = self._tree_shifts(
                    hidden, child, first, valid, parents, depths, present, tokens
                )
                pre = pre + node_shift
                hidden = (
                    pre if self.linear else F.silu(pre + self.weights["hidden_bias"])
                )
                child = child + child_shift
            deltas = self._child_delta(hidden, child)
            deltas = torch.where(valid, deltas, torch.zeros_like(deltas))
        out = self._dense("output", hidden)
        alpha = torch.clamp(_softplus(out[..., 0]), max=self.cap)
        beta = torch.clamp(_softplus(out[..., 1]), max=self.cap)
        scores = alpha[..., None] * logits + beta[..., None] * gumbels
        if deltas is not None:
            scores = scores.scatter_add(-1, children, deltas)
        return scores - torch.logsumexp(scores, dim=-1, keepdim=True), child, children

    def _reach(
        self,
        lq: torch.Tensor,
        parents: torch.Tensor,
        present: torch.Tensor,
        indices: torch.Tensor,
        jumps: list[torch.Tensor],
    ) -> torch.Tensor:
        """log r of every source from log q of the slots; -inf for absent sources."""
        slots = lq.shape[1]
        safe_parents = parents.clamp_min(0)
        edges = _rows(lq, safe_parents).gather(-1, indices[..., None])[..., 0]
        edges = torch.where(
            parents < 0, torch.zeros_like(edges), torch.clamp(edges, max=0.0)
        )
        # log r of the slots: path sums, then path minima so that a slot never scores
        # above its parent. A frontier node adds its own edge (<= 0) to its parent's.
        reach = _path_minima(_path_sums(edges[:, :slots], jumps), jumps)
        frontier = _rows(reach, safe_parents[:, slots:]) + edges[:, slots:]
        scores = torch.cat((reach, frontier), dim=1)
        return torch.where(present, scores, torch.full_like(scores, -torch.inf))

    def _coverage_logits(
        self,
        slot_logits: torch.Tensor,
        pool_logits: torch.Tensor,
        pool_gumbels: torch.Tensor,
        draft_hidden: torch.Tensor,
        parents: torch.Tensor,
        depths: torch.Tensor,
        indices: torch.Tensor,
        jumps: list[torch.Tensor],
        child: torch.Tensor,
        children: torch.Tensor,
    ) -> torch.Tensor:
        """logit kappa of the C children of every slot, [bs, slots, C].

        A child's node part is the one it gets once expanded, from what is known before: the
        "path" group placed in its parent's pool, the drafter hidden state of its own depth, the
        other groups zero (the export refuses a head that keeps them). The three scalars read
        the pool row of the child's own depth, the one its children are drawn from.
        """
        slots = slot_logits.shape[1]
        rows = pool_logits.shape[1]
        logits = slot_logits.float()
        slot_depths = depths[:, :slots]
        gumbels = _rows(pool_gumbels.float(), slot_depths.clamp(0, rows - 1))
        scores = logits + gumbels
        log_pi = torch.log_softmax(logits, dim=-1)
        parent_rows = parents[:, :slots]
        edges = _rows(log_pi, parent_rows.clamp_min(0)).gather(
            -1, indices[:, :slots, None]
        )[..., 0]
        edges = torch.where(parent_rows < 0, torch.zeros_like(edges), edges)
        own_scores = scores.gather(-1, children)
        own_logits = logits.gather(-1, children)
        weight = _path_sums(edges, jumps)[..., None] + log_pi.gather(-1, children)
        computed = {
            "depth": (slot_depths + 1).float()[..., None].expand_as(weight),
            "path_log_weight": weight,
            "is_leader": (scores.argmax(dim=-1, keepdim=True) == children).float(),
            "rank_z": (scores[..., None, :] > own_scores[..., None])
            .sum(dim=-1)
            .float(),
            "rank_l": (logits[..., None, :] > own_logits[..., None])
            .sum(dim=-1)
            .float(),
        }
        zeros = torch.zeros_like(weight)
        raw = torch.stack(
            [computed.get(name, zeros) for name in self.config["node_features"]], dim=-1
        )
        standard = (raw - self.weights["node_mean"]) / self.weights["node_std"]
        hidden_width = int(self.config["hidden"])
        pre = weight.new_zeros((*weight.shape, hidden_width))
        if "scalars" in self.blocks:
            pre = pre + self._dense("scalars", standard * self.node_keep)
        next_rows = (slot_depths + 1).clamp(max=rows - 1)
        if "draft" in self.blocks:
            normed = _layer_norm(
                draft_hidden.float(),
                self.weights["draft.norm.scale"],
                self.weights["draft.norm.shift"],
                self.eps,
            )
            pre = pre + _rows(self._dense("draft", normed), next_rows)[..., None, :]
        hidden = pre if self.linear else F.silu(pre + self.weights["hidden_bias"])
        draft = _rows(pool_logits.float(), next_rows)
        next_scores = draft + _rows(pool_gumbels.float(), next_rows)
        top2 = next_scores.topk(2, dim=-1).values
        top_mass = (
            torch.log_softmax(draft, dim=-1).topk(self.children, dim=-1).values.exp()
        )
        pool_scalars = torch.stack(
            [
                torch.clamp(top2[..., 0] - top2[..., 1], max=self.gap_cap),
                (next_scores.argmax(dim=-1) == draft.argmax(dim=-1)).float(),
                top_mass.sum(dim=-1),
            ],
            dim=-1,
        )
        pool_scalars = (pool_scalars - self.weights["expand_mean"]) / self.weights[
            "expand_std"
        ]
        inputs = torch.cat(
            (hidden, child, pool_scalars[..., None, :].expand(*child.shape[:-1], -1)),
            dim=-1,
        )
        inner = F.silu(self._dense("kappa_hidden", inputs))
        return self._dense("kappa_output", inner)[..., 0]

    def _expansion_scores(
        self,
        slot_logits: torch.Tensor,
        pool_logits: torch.Tensor,
        pool_gumbels: torch.Tensor,
        pool_tokens: torch.Tensor,
        draft_hidden: torch.Tensor,
        tokens: torch.Tensor,
        parents: torch.Tensor,
        depths: torch.Tensor,
        present: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        slots = slot_logits.shape[1]
        sources = parents.shape[1]
        rows = pool_logits.shape[1]
        indices = pool_indices(pool_tokens, tokens, depths)
        jumps = _jumps(parents[:, :slots], self.jump_steps)
        lq, child, children = self._outputs(
            slot_logits,
            pool_logits,
            pool_gumbels,
            pool_tokens,
            draft_hidden,
            tokens,
            parents,
            depths,
            present,
            indices,
            jumps,
            False,
        )
        assert child is not None and children is not None
        log_r = self._reach(lq, parents, present, indices, jumps)
        logit = self._coverage_logits(
            slot_logits,
            pool_logits,
            pool_gumbels,
            draft_hidden,
            parents,
            depths,
            indices,
            jumps,
            child,
            children,
        )
        # kappa is defined for a real child whose own depth has a pool row to draw children
        # from; a child at the depth cap is never expanded.
        first, real = child_sources(parents, present, slots, self.children)
        defined = real & (depths[:, :slots, None] + 1 < rows)
        log_kappa = torch.where(
            defined, F.logsigmoid(logit), torch.full_like(logit, -torch.inf)
        )
        target = torch.where(defined, first, torch.full_like(first, sources))
        per_source = log_kappa.new_full((log_kappa.shape[0], sources + 1), -torch.inf)
        per_source = per_source.scatter(1, target.flatten(1), log_kappa.flatten(1))[
            :, :sources
        ]
        return log_r, log_r + per_source

    def _scores(
        self,
        slot_logits: torch.Tensor,
        pool_logits: torch.Tensor,
        pool_gumbels: torch.Tensor,
        pool_tokens: torch.Tensor,
        draft_hidden: torch.Tensor,
        tokens: torch.Tensor,
        parents: torch.Tensor,
        depths: torch.Tensor,
        present: torch.Tensor,
        attention: bool,
    ) -> torch.Tensor:
        slots = slot_logits.shape[1]
        indices = pool_indices(pool_tokens, tokens, depths)
        jumps = _jumps(parents[:, :slots], self.jump_steps)
        lq = self._log_q(
            slot_logits,
            pool_logits,
            pool_gumbels,
            pool_tokens,
            draft_hidden,
            tokens,
            parents,
            depths,
            present,
            indices,
            jumps,
            attention,
        )
        return self._reach(lq, parents, present, indices, jumps)

    def _full_scores(self, *inputs: torch.Tensor) -> torch.Tensor:
        return self._scores(*inputs, attention=True)

    def _local_scores(self, *inputs: torch.Tensor) -> torch.Tensor:
        return self._scores(*inputs, attention=False)

    def log_q(self, tree: TreeInputs) -> torch.Tensor:
        """log q over the pool of every slot, [bs, slots, pool] (unexpanded rows unused)."""
        slots = tree.slot_logits.shape[1]
        with _full_fp32():
            return self._log_q(
                tree.slot_logits,
                tree.pool_logits,
                tree.pool_gumbels,
                tree.pool_tokens,
                tree.draft_hidden,
                tree.tokens,
                tree.parents,
                tree.depths,
                tree.present,
                pool_indices(tree.pool_tokens, tree.tokens, tree.depths),
                _jumps(tree.parents[:, :slots], self.jump_steps),
            )

    def prune_scores(self, tree: TreeInputs) -> torch.Tensor:
        """log r of every source, [bs, sources]; -inf for absent sources."""
        with _full_fp32():
            return self._scores_fn(
                tree.slot_logits,
                tree.pool_logits,
                tree.pool_gumbels,
                tree.pool_tokens,
                tree.draft_hidden,
                tree.tokens,
                tree.parents,
                tree.depths,
                tree.present,
            )

    def local_scores(self, tree: TreeInputs) -> torch.Tensor:
        """log r of every source by the head's local part: without tree attention, a child scored from
        its parent and path alone, so a node keeps its score as the tree grows. This is what the
        expansion ranks by; for a head without attention it is prune_scores."""
        with _full_fp32():
            return self._local_scores_fn(
                tree.slot_logits,
                tree.pool_logits,
                tree.pool_gumbels,
                tree.pool_tokens,
                tree.draft_hidden,
                tree.tokens,
                tree.parents,
                tree.depths,
                tree.present,
            )

    def expansion_scores(self, tree: TreeInputs) -> tuple[torch.Tensor, torch.Tensor]:
        """log r and the expansion priority e = log r + log kappa of every source, [bs, sources]
        each. e is -inf where kappa is undefined: absent sources, the root and nodes at the
        depth cap. Only a head of format 3 has kappa; like local_scores, a node keeps both as
        the tree grows."""
        if self.expansion is None:
            raise ValueError("This confidence head has no coverage branch (format 3).")
        with _full_fp32():
            return self._expansion_scores_fn(
                tree.slot_logits,
                tree.pool_logits,
                tree.pool_gumbels,
                tree.pool_tokens,
                tree.draft_hidden,
                tree.tokens,
                tree.parents,
                tree.depths,
                tree.present,
            )


def _log_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.log_softmax(logits, dim=-1), min=LOG_PROB_FLOOR)


class VectorConfidenceHead(ConfidenceHead):
    """A vector head (format 5): the same scoring entry points as ConfidenceHead, its own outputs."""

    def __init__(
        self,
        config: dict,
        weights: dict[str, torch.Tensor],
        token_table: torch.Tensor,
        source: str = "",
        compile: bool = False,
    ) -> None:
        if config.get("format") != VECTOR_FORMAT:
            raise ValueError(f"A vector head has format {VECTOR_FORMAT}.")
        if config["form"] != "b":
            raise ValueError("The confidence head supports form B only.")
        vector = config["vector"]
        unknown = set(vector["inputs"]) - {
            "draft",
            "child_token",
            "pool_draft",
            "pool_weaver",
            "pool_noise",
        }
        if unknown or "child_token" not in vector["inputs"]:
            raise ValueError(
                f"The vector head reads inputs {sorted(vector['inputs'])}; the fork computes the drafter "
                "row, the child token and the pool."
            )
        self.config = config
        self.weights = weights
        self.token_table = token_table
        self.source = source
        self.cap = float(config["cap"])
        self.eps = float(config["layer_norm_eps"])
        self.children = int(config["children"])
        self.max_depth = int(config["max_depth"])
        self.child = "embedding"
        self.vector = vector
        device = weights["output.weight"].device
        self.slot_keep = torch.ones(4, dtype=torch.float32, device=device)
        self.slot_keep[list(vector["slot_drop"])] = 0.0
        self.tree = config.get("tree")
        self.expansion = None
        self.jump_steps = max(1, math.ceil(math.log2(self.max_depth)))
        self._scores_fn = (
            torch.compile(self._full_scores, dynamic=False, fullgraph=True)
            if compile
            else self._full_scores
        )
        if self.tree is None:
            self._local_scores_fn = self._scores_fn
        else:
            self._local_scores_fn = (
                torch.compile(self._local_scores, dynamic=False, fullgraph=True)
                if compile
                else self._local_scores
            )
        self._expansion_scores_fn = self._expansion_scores
        # The incremental path: the drafter rows once per step, the prune once per step on
        # the kept local parts; the rounds run a kernel of their own (uzu_confidence_kernel).
        self._draft_rows_fn = (
            torch.compile(self._draft_rows, dynamic=False, fullgraph=True)
            if compile
            else self._draft_rows
        )
        self._prune_expanded_fn = (
            torch.compile(self._prune_expanded, dynamic=False, fullgraph=True)
            if compile
            else self._prune_expanded
        )

    @classmethod
    def load_vector(
        cls,
        directory: Path,
        config: dict,
        weights: dict[str, torch.Tensor],
        digest: str,
        token_embedding: torch.Tensor | None,
        device: torch.device | str,
        compile: bool,
    ) -> VectorConfidenceHead:
        if token_embedding is None:
            raise ValueError("A vector head needs the target's input embedding.")
        table = project_token_embedding(
            token_embedding.to(device),
            weights["token_norm.scale"],
            weights["token_norm.shift"],
            weights["token_projection.weight"],
            weights["token_projection.bias"],
            float(config["layer_norm_eps"]),
        )
        source = f"{directory} sha256={digest}"
        return cls(config, weights, table, source=source, compile=compile)

    def _act(self, values: torch.Tensor) -> torch.Tensor:
        return F.silu(values) if self.vector["input_act"] else values

    def _mlp(self, prefix: str, depth: int, values: torch.Tensor) -> torch.Tensor:
        """The head's MLP of that depth: residual (layer norm, dense, residual SiLU blocks, layer
        norm) or plain (depth layers of dense + SiLU); depth 0 is one dense layer in both."""
        if self.vector["mlp"] == "plain":
            for index in range(max(depth, 1)):
                values = self._dense(f"{prefix}.layers.{index}", values)
                if depth > 0:
                    values = F.silu(values)
            return values
        if depth > 0:
            values = _layer_norm(
                values,
                self.weights[f"{prefix}.input_norm.scale"],
                self.weights[f"{prefix}.input_norm.shift"],
                self.eps,
            )
        hidden = self._dense(f"{prefix}.entry", values)
        for index in range(depth):
            block = f"{prefix}.blocks.{index}"
            normed = _layer_norm(
                hidden,
                self.weights[f"{block}.norm.scale"],
                self.weights[f"{block}.norm.shift"],
                self.eps,
            )
            hidden = hidden + self._dense(
                f"{block}.down", F.silu(self._dense(f"{block}.up", normed))
            )
        if depth > 0:
            hidden = _layer_norm(
                hidden,
                self.weights[f"{prefix}.final_norm.scale"],
                self.weights[f"{prefix}.final_norm.shift"],
                self.eps,
            )
        return hidden

    def _pool_summary(
        self, logits: torch.Tensor, draft: torch.Tensor, gumbels: torch.Tensor
    ) -> torch.Tensor:
        """The set encoder over every slot's pool: per candidate the enabled numbers, a shared
        two-layer network, the pi_l-weighted mean and the maximum, a dense layer."""
        log_a = _log_probs(draft)
        columns = {
            "pool_draft": log_a,
            "pool_weaver": _log_probs(logits) - log_a,
            "pool_noise": gumbels,
        }
        features = torch.stack(
            [columns[name] for name in self.vector["pool_inputs"]], dim=-1
        )
        per_slot = self._dense(
            "pool_set.hidden", F.silu(self._dense("pool_set.entry", features))
        )
        weights = torch.softmax(logits, dim=-1)
        mean = (weights[..., None] * per_slot).sum(dim=-2)
        return self._dense(
            "pool_set.out", torch.cat([mean, per_slot.amax(dim=-2)], dim=-1)
        )

    def _slot_numbers(
        self,
        logits: torch.Tensor,
        draft: torch.Tensor,
        gumbels: torch.Tensor,
        children: torch.Tensor,
    ) -> torch.Tensor:
        """Standardised slot numbers of every child: log pi_a(c), log pi_l(c), G_c and
        log(1 + the number of candidates with a strictly larger drafter logit)."""
        child_draft = draft.gather(-1, children)
        rank = (draft[..., None, :] > child_draft[..., None]).sum(dim=-1).float()
        numbers = torch.stack(
            [
                _log_probs(draft).gather(-1, children),
                _log_probs(logits).gather(-1, children),
                gumbels.gather(-1, children),
                torch.log1p(rank),
            ],
            dim=-1,
        )
        numbers = (numbers - self.weights["slot_mean"]) / self.weights["slot_std"]
        return numbers * self.slot_keep

    def _edge(
        self, node: torch.Tensor, child: torch.Tensor, slot: torch.Tensor | None
    ) -> torch.Tensor:
        """delta of every child from z_u, the child token's vector and the slot numbers."""
        vector = self.vector
        expanded = node[..., None, :].expand_as(child)
        if vector["child_depth"] is None:
            parts = [expanded, child, expanded * child]
            if slot is not None:
                parts.append(slot)
            edge = self._mlp(
                "edge_mlp", int(vector["edge_depth"]), torch.cat(parts, dim=-1)
            )
            return self._dense("edge_output", edge)[..., 0]
        own = [child]
        if slot is not None and not vector["slot_to_edge"]:
            own.append(slot)
        branch = self._mlp(
            "child_mlp", int(vector["child_depth"]), torch.cat(own, dim=-1)
        )
        merge = vector["edge_merge"]
        pieces = {
            "sum": [expanded + branch],
            "concat": [expanded, branch],
            "product": [expanded * branch],
            "sum_product": [expanded + branch, expanded * branch],
            "concat_product": [expanded, branch, expanded * branch],
        }[merge]
        if vector["slot_to_edge"]:
            assert slot is not None
            pieces.append(slot)
        edge = self._mlp(
            "edge_mlp", int(vector["edge_depth"]), torch.cat(pieces, dim=-1)
        )
        return self._dense("edge_output", edge)[..., 0]

    def _outputs(
        self,
        slot_logits: torch.Tensor,
        pool_logits: torch.Tensor,
        pool_gumbels: torch.Tensor,
        pool_tokens: torch.Tensor,
        draft_hidden: torch.Tensor,
        tokens: torch.Tensor,
        parents: torch.Tensor,
        depths: torch.Tensor,
        present: torch.Tensor,
        indices: torch.Tensor,
        jumps: list[torch.Tensor],
        attention: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        vector = self.vector
        slots = slot_logits.shape[1]
        rows = pool_logits.shape[1]
        logits = slot_logits.float()
        slot_depths = depths[:, :slots]
        slot_rows = slot_depths.clamp(0, rows - 1)
        draft = _rows(pool_logits.float(), slot_rows)
        gumbels = _rows(pool_gumbels.float(), slot_rows)
        parts = []
        if "draft" in vector["inputs"]:
            normed = _layer_norm(
                draft_hidden.float(),
                self.weights["draft.norm.scale"],
                self.weights["draft.norm.shift"],
                self.eps,
            )
            parts.append(_rows(self._act(self._dense("draft", normed)), slot_rows))
        if vector["pool_inputs"]:
            parts.append(self._act(self._pool_summary(logits, draft, gumbels)))
        if vector["node_sum"]:
            joined = parts[0]
            for part in parts[1:]:
                joined = joined + part
        else:
            joined = torch.cat(parts, dim=-1)
        if vector["node_depth"] is None:
            # No node layer: z_u is the sum of the node inputs.
            node = joined
        else:
            node = self._mlp("node_mlp", int(vector["node_depth"]), joined)
        children, valid = child_indices(parents, present, indices, slots, self.children)
        child_tokens = _rows(pool_tokens, slot_rows).gather(-1, children)
        assert self.token_table is not None
        child = self._act(self.token_table[child_tokens])
        slot = (
            self._slot_numbers(logits, draft, gumbels, children)
            if vector["slot"]
            else None
        )
        if self.tree is not None and attention:
            first, _ = child_sources(parents, present, slots, self.children)
            own = torch.cat([child, slot], dim=-1) if slot is not None else child
            node_shift, child_shift = self._tree_shifts(
                self._dense("tree.node_in", node),
                self._dense("tree.child_in", own),
                first,
                valid,
                parents,
                depths,
                present,
                tokens,
            )
            node = node + node_shift
            child = child + child_shift
        deltas = self._edge(node, child, slot)
        deltas = torch.where(valid, deltas, torch.zeros_like(deltas))
        node_out = vector["node_out_depth"]
        out = self._dense(
            "output",
            node if node_out is None else self._mlp("node_out", int(node_out), node),
        )
        alpha = torch.clamp(_softplus(out[..., 0]), max=self.cap)
        beta = torch.clamp(_softplus(out[..., 1]), max=self.cap)
        scores = alpha[..., None] * logits + beta[..., None] * gumbels
        scores = scores.scatter_add(-1, children, deltas)
        return scores - torch.logsumexp(scores, dim=-1, keepdim=True), None, None

    # The incremental path. The local part of a slot reads only its own drafter row, its own pool
    # and its own children, so a round computes it for the slots it expands and nothing else, and
    # the prune reuses what the rounds kept. Same function as local_scores and prune_scores.

    def _draft_rows(self, draft_hidden: torch.Tensor) -> torch.Tensor:
        """The drafter-row term of z_u for a slot of each depth, [bs, depth, hidden]."""
        normed = _layer_norm(
            draft_hidden.float(),
            self.weights["draft.norm.scale"],
            self.weights["draft.norm.shift"],
            self.eps,
        )
        return self._act(self._dense("draft", normed))

    def step_constants(
        self,
        pool_logits: torch.Tensor,
        pool_gumbels: torch.Tensor,
        pool_tokens: torch.Tensor,
        draft_hidden: torch.Tensor,
    ) -> StepConstants:
        """What every slot of the step reads from the drafter, once per step."""
        rows = None
        if "draft" in self.vector["inputs"]:
            with _full_fp32():
                rows = self._draft_rows_fn(draft_hidden)
        return StepConstants(
            draft_rows=rows,
            pool_logits=pool_logits.float(),
            pool_gumbels=pool_gumbels.float(),
            pool_tokens=pool_tokens,
        )

    def _slot_parts(
        self,
        constants: StepConstants,
        slot_logits: torch.Tensor,
        slot_depths: torch.Tensor,
        child_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """z_u, the children's token vectors and slot numbers of some slots, and the Gumbel row
        of each; a slot at the depth cap reads the last pool row, as in _outputs."""
        rows = slot_depths.clamp(0, constants.pool_logits.shape[1] - 1)
        logits = slot_logits.float()
        draft = _rows(constants.pool_logits, rows)
        gumbels = _rows(constants.pool_gumbels, rows)
        terms = []
        if constants.draft_rows is not None:
            terms.append(_rows(constants.draft_rows, rows))
        if self.vector["pool_inputs"]:
            terms.append(self._act(self._pool_summary(logits, draft, gumbels)))
        if self.vector["node_sum"]:
            joined = functools.reduce(operator.add, terms)
        else:
            joined = torch.cat(terms, dim=-1)
        depth = self.vector["node_depth"]
        nodes = joined if depth is None else self._mlp("node_mlp", int(depth), joined)
        assert self.token_table is not None
        tokens_of_children = _rows(constants.pool_tokens, rows).gather(
            -1, child_indices
        )
        vectors = self._act(self.token_table[tokens_of_children])
        numbers = (
            self._slot_numbers(logits, draft, gumbels, child_indices)
            if self.vector["slot"]
            else None
        )
        return nodes, vectors, numbers, gumbels

    def _pool_log_q(
        self,
        nodes: torch.Tensor,
        vectors: torch.Tensor,
        numbers: torch.Tensor | None,
        slot_logits: torch.Tensor,
        gumbels: torch.Tensor,
        child_indices: torch.Tensor,
        child_real: torch.Tensor,
    ) -> torch.Tensor:
        """log q over the pool of some slots, [bs, slots, pool], from z_u and the children."""
        edges = self._edge(nodes, vectors, numbers).masked_fill(~child_real, 0.0)
        depth = self.vector["node_out_depth"]
        head_in = nodes if depth is None else self._mlp("node_out", int(depth), nodes)
        alpha, beta = torch.clamp(
            _softplus(self._dense("output", head_in)[..., :2]), max=self.cap
        ).unbind(dim=-1)
        pooled = (
            alpha.unsqueeze(-1) * slot_logits.float() + beta.unsqueeze(-1) * gumbels
        )
        pooled = pooled.scatter_add(-1, child_indices, edges)
        return pooled - pooled.logsumexp(dim=-1, keepdim=True)

    def expand_slots(
        self,
        constants: StepConstants,
        slot_logits: torch.Tensor,
        slot_depths: torch.Tensor,
        child_indices: torch.Tensor,
        child_real: torch.Tensor,
        slot_log_r: torch.Tensor,
    ) -> SlotExpansion:
        """The local part on the slots one round expanded: slot_logits [bs, width, pool], their
        depths and log r [bs, width], their children's pool indices and which are real
        [bs, width, C]. A child's log r is its slot's plus min(log q, 0), summed one edge at a
        time as training's log_reach does."""
        with _full_fp32():
            nodes, vectors, numbers, gumbels = self._slot_parts(
                constants, slot_logits, slot_depths, child_indices
            )
            pool = self._pool_log_q(
                nodes, vectors, numbers, slot_logits, gumbels, child_indices, child_real
            )
            log_q = pool.gather(-1, child_indices)
            log_r = slot_log_r.unsqueeze(-1) + log_q.clamp(max=0.0)
            log_r = log_r.masked_fill(~child_real, -torch.inf)
        return SlotExpansion(
            nodes=nodes, children=vectors, slots=numbers, log_q=log_q, log_r=log_r
        )

    def _prune_expanded(
        self,
        slot_logits: torch.Tensor,
        pool_gumbels: torch.Tensor,
        nodes: torch.Tensor,
        vectors: torch.Tensor,
        numbers: torch.Tensor | None,
        child_indices: torch.Tensor,
        child_real: torch.Tensor,
        child_positions: torch.Tensor,
        tokens: torch.Tensor,
        parents: torch.Tensor,
        depths: torch.Tensor,
        present: torch.Tensor,
    ) -> torch.Tensor:
        slots = slot_logits.shape[1]
        sources = parents.shape[1]
        rows = depths[:, :slots].clamp(0, pool_gumbels.shape[1] - 1)
        gumbels = _rows(pool_gumbels, rows)
        if self.tree is not None:
            own = vectors if numbers is None else torch.cat([vectors, numbers], dim=-1)
            node_shift, child_shift = self._tree_shifts(
                self._dense("tree.node_in", nodes),
                self._dense("tree.child_in", own),
                child_positions,
                child_real,
                parents,
                depths,
                present,
                tokens,
            )
            nodes, vectors = nodes + node_shift, vectors + child_shift
        lq = self._pool_log_q(
            nodes, vectors, numbers, slot_logits, gumbels, child_indices, child_real
        )
        # Every real child's pool index at its source; the root and absent sources read 0.
        targets = torch.where(
            child_real, child_positions, torch.full_like(child_positions, sources)
        )
        indices = child_indices.new_zeros((parents.shape[0], sources + 1)).scatter(
            1, targets.flatten(1), child_indices.flatten(1)
        )[:, :sources]
        jumps = _jumps(parents[:, :slots], self.jump_steps)
        return self._reach(lq, parents, present, indices, jumps)

    def prune_expanded(
        self, tree: ExpandedTree, constants: StepConstants
    ) -> torch.Tensor:
        """log r of every source by the full head, [bs, sources], from the local parts the rounds
        kept: no slot's pool or drafter row is read again. The same as prune_scores on the
        same tree; -inf for absent sources."""
        with _full_fp32():
            return self._prune_expanded_fn(
                tree.slot_logits,
                constants.pool_gumbels,
                tree.nodes,
                tree.children,
                tree.slots,
                tree.child_indices,
                tree.child_real,
                tree.child_positions,
                tree.tokens,
                tree.parents,
                tree.depths,
                tree.present,
            )
