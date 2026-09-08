import argparse

import torch
import triton

from sgl_kernel import tree_speculative_sampling_target_only
from sglang.srt.speculative.dflash_tfm import _target_only_verify_target_probs


def legacy_target_only(
    candidates,
    retrieve_index,
    retrieve_next_token,
    retrieve_next_sibling,
    target_probs,
    uniforms,
    bonus_uniforms,
):
    batch_size, num_nodes = candidates.shape
    normalized = target_probs.to(torch.float32).contiguous()
    normalized /= normalized.sum(dim=-1, keepdim=True)
    predicts = torch.full(
        (batch_size * num_nodes,), -1, dtype=torch.int32, device="cuda"
    )
    accept_index = torch.full(
        (batch_size, num_nodes), -1, dtype=torch.int32, device="cuda"
    )
    accept_token_num = torch.empty((batch_size,), dtype=torch.int32, device="cuda")
    tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrieve_index,
        retrive_next_token=retrieve_next_token,
        retrive_next_sibling=retrieve_next_sibling,
        uniform_samples=uniforms,
        uniform_samples_for_final_sampling=bonus_uniforms,
        target_probs=normalized,
        draft_probs=torch.zeros_like(normalized),
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )
    return predicts, accept_index, accept_token_num


def capture(fn):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = fn()
    torch.cuda.synchronize()
    return graph, outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, choices=(49, 64, 128, 129), required=True)
    parser.add_argument("--vocab-size", type=int, default=248320)
    args = parser.parse_args()

    num_nodes = args.nodes
    candidates = torch.zeros((1, num_nodes), dtype=torch.int64, device="cuda")
    retrieve_index = torch.arange(num_nodes, dtype=torch.int64, device="cuda")[None]
    retrieve_next_token = torch.arange(
        1, num_nodes + 1, dtype=torch.int64, device="cuda"
    )[None]
    retrieve_next_token[0, -1] = -1
    retrieve_next_sibling = torch.full_like(retrieve_next_token, -1)
    target_probs = torch.zeros(
        (1, num_nodes, args.vocab_size), dtype=torch.float32, device="cuda"
    )
    target_probs[:, :, 0] = 1.0
    uniforms = torch.full((1, num_nodes), 0.5, dtype=torch.float32, device="cuda")
    bonus_uniforms = torch.full((1,), 0.5, dtype=torch.float32, device="cuda")

    common = (
        candidates,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        target_probs,
        uniforms,
        bonus_uniforms,
    )
    legacy_graph, legacy_outputs = capture(lambda: legacy_target_only(*common))
    fast_graph, fast_outputs = capture(
        lambda: _target_only_verify_target_probs(
            candidates=candidates,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            target_probs=target_probs,
            uniform_samples=uniforms,
            bonus_uniforms=bonus_uniforms,
        )
    )
    for legacy, fast in zip(legacy_outputs, fast_outputs, strict=True):
        assert torch.equal(legacy, fast)

    legacy_ms = triton.testing.do_bench(legacy_graph.replay)
    fast_ms = triton.testing.do_bench(fast_graph.replay)
    scratch_mib = target_probs.numel() * target_probs.element_size() / 2**20
    print(
        f"nodes={num_nodes} vocab={args.vocab_size} "
        f"legacy={legacy_ms * 1000:.2f}us fast={fast_ms * 1000:.2f}us "
        f"speedup={legacy_ms / fast_ms:.2f}x removed_scratch={scratch_mib:.1f}MiB"
    )


if __name__ == "__main__":
    main()
