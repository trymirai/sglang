"""Position-stable Gumbel field shared by Uzu proposals and target sampling.

The Philox key is the request seed; its counter is (vocabulary token ID,
absolute parent-token position). Pruning and CUDA Graph replay must reuse this
field, independent of tree packing and the number of accepted tokens.
"""

import torch
import triton
import triton.language as tl


def update_tree_context(worker, sampling_info, prefix_positions):
    """Update persistent inputs separately for each captured batch size."""
    bs = prefix_positions.numel()
    if not hasattr(worker, "_gumbel_tree_inputs"):
        worker._gumbel_tree_inputs = {}
    if bs not in worker._gumbel_tree_inputs:
        worker._gumbel_tree_inputs[bs] = (
            torch.empty_like(sampling_info.sampling_seed),
            torch.empty_like(prefix_positions),
            torch.empty_like(sampling_info.temperatures.reshape(-1)),
            torch.empty((), device=prefix_positions.device),
        )
    seeds, positions, temperatures, noise_scale = worker._gumbel_tree_inputs[bs]
    seeds.copy_(sampling_info.sampling_seed)
    positions.copy_(prefix_positions)
    temperatures.copy_(sampling_info.temperatures.reshape(-1))
    noise_scale.fill_(0.0 if sampling_info.is_all_greedy else 1.0)
    return seeds, positions, temperatures, noise_scale


@triton.jit
def gumbel_at(seed, position, token_id):
    # Philox4x32-10. Four distinct counter words avoid flattening overflow when
    # position * vocabulary_size exceeds 32 bits.
    k0 = seed.to(tl.uint64).to(tl.uint32)
    k1 = (seed.to(tl.uint64) >> 32).to(tl.uint32)
    c0 = token_id.to(tl.uint32)
    c1 = position.to(tl.uint32)
    c2 = tl.full(c0.shape, 0, tl.uint32)
    c3 = tl.full(c0.shape, 0, tl.uint32)
    for _ in tl.static_range(10):
        p0 = c0.to(tl.uint64) * 0xD2511F53
        p1 = c2.to(tl.uint64) * 0xCD9E8D57
        c0, c1, c2, c3 = (
            (p1 >> 32).to(tl.uint32) ^ c1 ^ k0,
            p1.to(tl.uint32),
            (p0 >> 32).to(tl.uint32) ^ c3 ^ k1,
            p0.to(tl.uint32),
        )
        k0 += 0x9E3779B9
        k1 += 0xBB67AE85
    # Exactly representable open-interval FP32 uniform, including both tails.
    # The same conversion is used in the full-vocab and gathered-candidate paths.
    uniform = ((c0 >> 9).to(tl.float32) + 0.5) * (1.0 / 8388608.0)
    return -tl.log(-tl.log(uniform))



@triton.jit
def _argmax(values, seeds, positions, max_values, max_ids, COLS: tl.constexpr,
            TILES: tl.constexpr, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    token = tile * BLOCK + tl.arange(0, BLOCK)
    probs = tl.load(values + row * COLS + token, mask=token < COLS, other=0).to(tl.float32)
    noise = gumbel_at(tl.load(seeds + row), tl.load(positions + row), token)
    score = tl.log(probs) + noise
    score = tl.where(token < COLS, score, -float('inf'))
    best, index = tl.max(score, 0, return_indices=True, return_indices_tie_break_left=True)
    tl.store(max_values + row * TILES + tile, best)
    tl.store(max_ids + row * TILES + tile, tile * BLOCK + index)



def sample_probs(probs, seeds, positions):
    """Sample argmax(log p + G); ties choose the lowest vocabulary token ID."""
    if seeds is None:
        raise ValueError("shared_gumbel requires per-request sampling_seed")
    assert probs.ndim == 2 and seeds.shape == positions.shape == (probs.shape[0],)
    probs = probs.contiguous()
    tiles = triton.cdiv(probs.shape[1], 1024)
    values = torch.empty((probs.shape[0], tiles), device=probs.device, dtype=torch.float32)
    ids = torch.empty_like(values, dtype=torch.int64)
    _argmax[(probs.shape[0], tiles)](
        probs, seeds, positions, values, ids, probs.shape[1], tiles, 1024
    )
    return ids.gather(1, values.argmax(-1, keepdim=True)).squeeze(1)


def filtered_probs(logits, sampling_info, repeats=1):
    """Use the same temperature and joint top-p/top-k policy as tree verification."""
    from sglang.srt.speculative.dflash_tfm import _filter_tree_target_probs
    from sglang.srt.server_args import get_global_server_args
    expand = lambda x: torch.repeat_interleave(x, repeats, dim=0)
    probs = torch.softmax((logits / expand(sampling_info.temperatures)).float(), -1)
    return _filter_tree_target_probs(
        probs, top_ks=expand(sampling_info.top_ks), top_ps=expand(sampling_info.top_ps),
        min_ps=expand(sampling_info.min_ps), need_top_k=sampling_info.need_top_k_sampling,
        need_top_p=sampling_info.need_top_p_sampling,
        sequential=(get_global_server_args().sampling_backend == 'flashinfer'
                    and sampling_info.need_min_p_sampling))
