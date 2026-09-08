from __future__ import annotations

import argparse

import torch
import triton

from sglang.srt.speculative.dflash_tfm import (
    _target_only_verify_target_probs,
    _traversal_verify_target_probs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, choices=(49, 64, 128, 129), required=True)
    parser.add_argument("--vocab-size", type=int, default=248_320)
    args = parser.parse_args()

    num_nodes = args.nodes
    candidates = torch.zeros((1, num_nodes), dtype=torch.int64, device="cuda")
    retrieve_index = torch.arange(num_nodes, dtype=torch.int64, device="cuda")[None]
    next_token = torch.arange(1, num_nodes + 1, dtype=torch.int64, device="cuda")[None]
    next_token[0, -1] = -1
    next_sibling = torch.full_like(next_token, -1)
    parents = torch.arange(-1, num_nodes - 1, dtype=torch.int64, device="cuda")[None]
    depths = torch.arange(num_nodes, dtype=torch.int64, device="cuda")[None]
    node_mask = torch.ones((1, num_nodes), dtype=torch.bool, device="cuda")
    draft_logprobs = torch.zeros((1, num_nodes), dtype=torch.float32, device="cuda")
    sibling_keys = torch.zeros_like(draft_logprobs)
    target_probs = torch.zeros(
        (1, num_nodes, args.vocab_size), dtype=torch.float32, device="cuda"
    )
    target_probs[:, :, 0] = 1.0
    uniforms = torch.full((1, num_nodes), 0.5, dtype=torch.float32, device="cuda")
    bonus_uniforms = torch.full((1,), 0.5, dtype=torch.float32, device="cuda")

    def target_only():
        return _target_only_verify_target_probs(
            candidates=candidates,
            retrieve_index=retrieve_index,
            retrieve_next_token=next_token,
            retrieve_next_sibling=next_sibling,
            target_probs=target_probs,
            uniform_samples=uniforms,
            bonus_uniforms=bonus_uniforms,
        )

    def traversal():
        return _traversal_verify_target_probs(
            candidates=candidates,
            parent_indices=parents,
            depths=depths,
            node_mask=node_mask,
            draft_logprobs=draft_logprobs,
            target_probs=target_probs,
            sibling_keys=sibling_keys,
            uniform_samples=uniforms,
            bonus_uniforms=bonus_uniforms,
        )

    target_only_output = target_only()
    traversal_output = traversal()
    for target_only_tensor, traversal_tensor in zip(
        target_only_output[:3], traversal_output[:3], strict=True
    ):
        assert torch.equal(target_only_tensor, traversal_tensor)

    target_only_ms = triton.testing.do_bench(target_only)
    traversal_ms = triton.testing.do_bench(traversal)
    print(
        f"nodes={num_nodes} vocab={args.vocab_size} "
        f"target_only_us={target_only_ms * 1000:.2f} "
        f"traversal_us={traversal_ms * 1000:.2f} "
        f"target_only_speedup={traversal_ms / target_only_ms:.2f}"
    )


if __name__ == "__main__":
    main()
