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
from sglang.srt.speculative.dflash_tfm import WeaverTree
from sglang.srt.speculative.uzu_tree import (
    UzuConstructedTree,
    UzuTreeShape,
    check_uzu_tree_shape,
    construct_uzu_tree,
)

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


class UzuTreeDumper:
    def __init__(self, worker, dump_dir: str, shapes_spec: str | None) -> None:
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
        }
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
        """Grow the served shape and every extra shape on this step's inputs; hold until verify."""
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

    def record_accept(self, accept_tokens: torch.Tensor) -> None:
        """Attach this step's accept tokens (drafts plus bonus) and write the step."""
        pending = self._pending
        if pending is None:
            raise RuntimeError("The uzu tree dump saw a verify without a recorded tree build.")
        self._pending = None
        pending.arrays["accept_tokens"] = _to_numpy(accept_tokens, "tokens")
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
