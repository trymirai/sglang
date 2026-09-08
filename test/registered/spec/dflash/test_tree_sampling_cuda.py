import pytest
import torch

from sglang.srt.speculative.dflash_tfm import (
    _filter_tree_target_probs,
    _target_only_verify_target_probs,
    _tree_sampling_uniforms,
    _traversal_verify_target_probs,
)
from sglang.srt.speculative.tree_sampling import traversal_tree_verify
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Weaver tree sampling kernels require CUDA",
)

_BATCH_SIZE = 20_000
_TARGET = torch.tensor([0.55, 0.25, 0.20], dtype=torch.float32)


def _tree_batch():
    device = torch.device("cuda")
    candidates = torch.tensor([0, 0, 1], device=device).repeat(_BATCH_SIZE, 1)
    parents = torch.tensor([-1, 0, 0], device=device).repeat(_BATCH_SIZE, 1)
    depths = torch.tensor([0, 1, 1], device=device).repeat(_BATCH_SIZE, 1)
    node_mask = torch.ones_like(candidates, dtype=torch.bool)
    draft_logprobs = torch.log(torch.tensor([1.0, 0.7, 0.3], device=device)).repeat(
        _BATCH_SIZE, 1
    )
    target_probs = torch.tensor(
        [
            [0.55, 0.25, 0.20],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.bfloat16,
        device=device,
    ).repeat(_BATCH_SIZE, 1, 1)
    return (
        candidates,
        parents,
        depths,
        node_mask,
        draft_logprobs,
        target_probs,
    )


def _assert_first_token_distribution(predict):
    root_indices = torch.arange(
        0, _BATCH_SIZE * 3, 3, dtype=torch.long, device=predict.device
    )
    counts = torch.bincount(predict[root_indices].to(torch.long), minlength=3)
    observed = counts.float().cpu() / _BATCH_SIZE
    assert torch.allclose(observed, _TARGET, atol=0.015), (observed, _TARGET)


def test_deterministic_target_only_tree_is_lossless():
    candidates, _, _, _, _, target_probs = _tree_batch()
    row_offsets = torch.arange(0, _BATCH_SIZE * 3, 3, dtype=torch.long, device="cuda")[
        :, None
    ]
    retrieve_index = row_offsets + torch.arange(3, device="cuda")[None, :]
    retrieve_next_token = torch.tensor([1, -1, -1], device="cuda").repeat(
        _BATCH_SIZE, 1
    )
    retrieve_next_sibling = torch.tensor([-1, 2, -1], device="cuda").repeat(
        _BATCH_SIZE, 1
    )
    predict, _, _ = _target_only_verify_target_probs(
        candidates=candidates,
        retrieve_index=retrieve_index,
        retrieve_next_token=retrieve_next_token,
        retrieve_next_sibling=retrieve_next_sibling,
        target_probs=target_probs,
        uniform_samples=torch.rand(
            (_BATCH_SIZE, 7), dtype=torch.float32, device="cuda"
        )[:, :3].clamp_(
            min=torch.finfo(torch.float32).tiny,
            max=1.0 - torch.finfo(torch.float32).eps,
        ),
        bonus_uniforms=torch.rand(
            (_BATCH_SIZE,), dtype=torch.float32, device="cuda"
        ).clamp_(
            min=torch.finfo(torch.float32).tiny,
            max=1.0 - torch.finfo(torch.float32).eps,
        ),
    )
    _assert_first_token_distribution(predict)


def test_target_only_fast_path_matches_legacy_on_branched_tree():
    from sgl_kernel import tree_speculative_sampling_target_only

    batch_size = 4096
    num_nodes = 5
    candidates = torch.tensor([4, 0, 1, 2, 3], device="cuda").repeat(batch_size, 1)
    row_offsets = torch.arange(
        0, batch_size * num_nodes, num_nodes, dtype=torch.int64, device="cuda"
    )[:, None]
    retrieve_index = row_offsets + torch.arange(num_nodes, device="cuda")[None]
    retrieve_next_token = torch.tensor([1, 3, -1, -1, -1], device="cuda").repeat(
        batch_size, 1
    )
    retrieve_next_sibling = torch.tensor([-1, 2, -1, 4, -1], device="cuda").repeat(
        batch_size, 1
    )
    target_probs = torch.tensor(
        [
            [0.4, 0.3, 0.2, 0.1, 0.0],
            [0.1, 0.1, 0.2, 0.6, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        device="cuda",
    ).repeat(batch_size, 1, 1)
    uniforms = torch.rand(
        (batch_size, num_nodes), dtype=torch.float32, device="cuda"
    ).clamp_(
        min=torch.finfo(torch.float32).tiny,
        max=1.0 - torch.finfo(torch.float32).eps,
    )
    bonus_uniforms = torch.rand(
        (batch_size,), dtype=torch.float32, device="cuda"
    ).clamp_(
        min=torch.finfo(torch.float32).tiny,
        max=1.0 - torch.finfo(torch.float32).eps,
    )

    legacy_predict = torch.full(
        (batch_size * num_nodes,), -1, dtype=torch.int32, device="cuda"
    )
    legacy_accept_index = torch.full(
        (batch_size, num_nodes), -1, dtype=torch.int32, device="cuda"
    )
    legacy_num_correct = torch.empty((batch_size,), dtype=torch.int32, device="cuda")
    tree_speculative_sampling_target_only(
        predicts=legacy_predict,
        accept_index=legacy_accept_index,
        accept_token_num=legacy_num_correct,
        candidates=candidates,
        retrive_index=retrieve_index,
        retrive_next_token=retrieve_next_token,
        retrive_next_sibling=retrieve_next_sibling,
        uniform_samples=uniforms,
        uniform_samples_for_final_sampling=bonus_uniforms,
        target_probs=target_probs,
        draft_probs=torch.zeros_like(target_probs),
        deterministic=True,
    )
    fast_predict, fast_accept_index, fast_num_correct = (
        _target_only_verify_target_probs(
            candidates=candidates,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            target_probs=target_probs.clone(),
            uniform_samples=uniforms,
            bonus_uniforms=bonus_uniforms,
        )
    )

    assert torch.equal(legacy_num_correct, fast_num_correct)
    depths = torch.arange(num_nodes, device="cuda")[None]
    accepted = depths <= legacy_num_correct[:, None]
    assert torch.equal(legacy_accept_index[accepted], fast_accept_index[accepted])
    accepted_indices = legacy_accept_index[accepted].to(torch.long)
    assert torch.equal(legacy_predict[accepted_indices], fast_predict[accepted_indices])


def test_without_replacement_traversal_tree_is_lossless():
    (
        candidates,
        parents,
        depths,
        node_mask,
        draft_logprobs,
        target_probs,
    ) = _tree_batch()
    uniforms = torch.rand_like(draft_logprobs).clamp_(
        min=torch.finfo(torch.float32).tiny,
        max=1.0 - 1.0e-7,
    )
    sibling_keys = draft_logprobs - torch.log(-torch.log(uniforms))
    predict, _, _, _ = _traversal_verify_target_probs(
        candidates=candidates,
        parent_indices=parents,
        depths=depths,
        node_mask=node_mask,
        draft_logprobs=draft_logprobs,
        target_probs=target_probs,
        sibling_keys=sibling_keys,
        uniform_samples=torch.rand(
            (_BATCH_SIZE, 7), dtype=torch.float32, device="cuda"
        )[:, :3],
        bonus_uniforms=torch.rand((_BATCH_SIZE,), dtype=torch.float32, device="cuda"),
    )
    _assert_first_token_distribution(predict)


def test_traversal_fast_path_matches_cpu_reference_on_multistep_tree():
    batch_size = 512
    num_nodes = 5
    candidates = torch.tensor([99, 0, 1, 0, 2], dtype=torch.long).repeat(batch_size, 1)
    parents = torch.tensor([-1, 0, 0, 1, 1], dtype=torch.long).repeat(batch_size, 1)
    depths = torch.tensor([0, 1, 1, 2, 2], dtype=torch.long).repeat(batch_size, 1)
    node_mask = torch.ones((batch_size, num_nodes), dtype=torch.bool)
    draft_logprobs = torch.log(
        torch.tensor([1.0, 0.65, 0.35, 0.30, 0.70], dtype=torch.float32)
    ).repeat(batch_size, 1)
    target_probs = torch.tensor(
        [
            [0.45, 0.35, 0.20],
            [0.25, 0.15, 0.60],
            [0.20, 0.50, 0.30],
            [0.10, 0.30, 0.60],
            [0.55, 0.25, 0.20],
        ],
        dtype=torch.bfloat16,
    ).repeat(batch_size, 1, 1)

    generator = torch.Generator().manual_seed(20260724)
    sibling_uniforms = torch.rand(
        (batch_size, num_nodes), generator=generator, dtype=torch.float32
    ).clamp_(1.0e-7, 1.0 - 1.0e-7)
    sibling_keys = draft_logprobs - torch.log(-torch.log(sibling_uniforms))
    accept_uniforms = torch.rand(
        (batch_size, num_nodes), generator=generator, dtype=torch.float32
    )
    bonus_uniforms = torch.rand((batch_size,), generator=generator, dtype=torch.float32)

    predict, accept_index, num_correct, accept_leaf = _traversal_verify_target_probs(
        candidates=candidates.cuda(),
        parent_indices=parents.cuda(),
        depths=depths.cuda(),
        node_mask=node_mask.cuda(),
        draft_logprobs=draft_logprobs.cuda(),
        target_probs=target_probs.cuda(),
        sibling_keys=sibling_keys.cuda(),
        uniform_samples=accept_uniforms.cuda(),
        bonus_uniforms=bonus_uniforms.cuda(),
    )
    predict = predict.cpu()
    accept_index = accept_index.cpu()
    num_correct = num_correct.cpu()
    accept_leaf = accept_leaf.cpu()

    for row in range(batch_size):
        expected = traversal_tree_verify(
            tokens=candidates[row],
            parents=parents[row],
            target_probs=target_probs[row],
            draft_logprobs=draft_logprobs[row],
            sibling_keys=sibling_keys[row],
            accept_uniforms=accept_uniforms[row],
            bonus_uniform=float(bonus_uniforms[row]),
        )
        accepted_count = int(num_correct[row]) + 1
        row_base = row * num_nodes
        actual_nodes = tuple(
            int(node - row_base) for node in accept_index[row, :accepted_count]
        )
        assert actual_nodes == expected.accepted_nodes
        assert int(accept_leaf[row]) == expected.accepted_nodes[-1]
        for parent, child in zip(actual_nodes, actual_nodes[1:]):
            assert int(predict[row_base + parent]) == int(candidates[row, child])
        assert int(predict[row_base + int(accept_leaf[row])]) == expected.bonus_token


def test_request_seed_uniforms_are_stable_under_batch_reordering():
    seeds = torch.tensor([11, 29], dtype=torch.long, device="cuda")
    positions = torch.tensor([101, 307], dtype=torch.long, device="cuda")
    first = _tree_sampling_uniforms(sampling_seed=seeds, positions=positions, count=9)
    second = _tree_sampling_uniforms(
        sampling_seed=seeds.flip(0), positions=positions.flip(0), count=9
    )
    assert torch.equal(first, second.flip(0))
    assert bool(torch.all((first > 0.0) & (first < 1.0)))


def test_target_filters_match_normal_sampler_orders():
    probs = torch.tensor([[0.60, 0.25, 0.15]], device="cuda")
    params = {
        "top_ks": torch.tensor([2], dtype=torch.int32, device="cuda"),
        "top_ps": torch.tensor([0.70], device="cuda"),
        "min_ps": torch.tensor([0.0], device="cuda"),
        "need_top_k": True,
        "need_top_p": True,
    }
    joint = _filter_tree_target_probs(probs, **params, sequential=False)
    sequential = _filter_tree_target_probs(probs, **params, sequential=True)
    assert torch.allclose(
        joint.cpu(), torch.tensor([[12 / 17, 5 / 17, 0.0]]), atol=1.0e-6
    )
    assert torch.equal(sequential.cpu(), torch.tensor([[1.0, 0.0, 0.0]]))
