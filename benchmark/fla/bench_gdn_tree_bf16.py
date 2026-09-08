from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F
import triton

from sglang.srt.layers.attention.fla.gdn_tree_fused import (
    build_tree_structure_static,
)
from sglang.srt.layers.attention.fla.gdn_tree_triton import (
    tree_gdn_triton_verify,
)


def make_parent(kind: str, tokens: int) -> torch.Tensor:
    parent = torch.full((1, tokens), -1, dtype=torch.int64, device="cuda")
    nodes = torch.arange(1, tokens, device="cuda")
    if kind == "chain":
        parent[0, 1:] = nodes - 1
    elif kind == "wide":
        parent[0, 1:] = (nodes - 1) // 4
    elif kind == "mixed":
        parent[0, 1:] = torch.clamp(nodes - 4, min=0)
    else:
        raise AssertionError(kind)
    return parent


def make_inputs(seed: int, parent: torch.Tensor) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    batch, tokens, key_heads, value_heads, dim = 1, parent.shape[1], 16, 48, 128
    q = F.normalize(
        torch.randn(batch, tokens, key_heads, dim, device="cuda"), dim=-1
    ).to(torch.bfloat16)
    k = F.normalize(
        torch.randn(batch, tokens, key_heads, dim, device="cuda"), dim=-1
    ).to(torch.bfloat16)
    v = (0.1 * torch.randn(batch, tokens, value_heads, dim, device="cuda")).to(
        torch.bfloat16
    )
    a = (0.2 * torch.randn(batch, tokens, value_heads, device="cuda")).to(
        torch.bfloat16
    )
    b = torch.randn_like(a)
    a_log = (-3.0 + 0.2 * torch.randn(value_heads, device="cuda")).float()
    dt_bias = (0.2 * torch.randn(value_heads, device="cuda")).float()
    state = (0.05 * torch.randn(1, value_heads, dim, dim, device="cuda")).float()
    state_indices = torch.zeros(batch, device="cuda", dtype=torch.int64)
    return a_log, a, dt_bias, q, k, v, b, state, state_indices


def verify(
    inputs: tuple[torch.Tensor, ...],
    tree,
    *,
    precision: str,
    bf16_mode: str,
):
    a_log, a, dt_bias, q, k, v, b, state, state_indices = inputs
    return tree_gdn_triton_verify(
        a_log,
        a,
        dt_bias,
        1.0,
        20.0,
        q,
        k,
        v,
        b,
        state,
        state_indices,
        tree,
        scale=None,
        use_qk_l2norm_in_kernel=False,
        return_lazy_state=True,
        precision=precision,
        bf16_mode=bf16_mode,
    )


def error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    delta = actual.float() - expected.float()
    return {
        "max_abs": delta.abs().max().item(),
        "mean_abs": delta.abs().mean().item(),
        "rms": delta.square().mean().sqrt().item(),
    }


def capture(function) -> torch.cuda.CUDAGraph:
    for _ in range(10):
        function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        function()
    return graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=65)
    parser.add_argument("--repetitions", type=int, default=1000)
    args = parser.parse_args()

    assert torch.cuda.is_available()
    assert args.tokens > 1
    assert args.repetitions > 0

    correctness = []
    for tree_kind in ("chain", "wide", "mixed"):
        parent = make_parent(tree_kind, args.tokens)
        tree = build_tree_structure_static(parent)
        for seed in (0, 1, 2):
            inputs = make_inputs(seed, parent)
            reference_out, reference_lazy = verify(
                inputs, tree, precision="ieee", bf16_mode="none"
            )
            tf32_out, tf32_lazy = verify(
                inputs, tree, precision="tf32", bf16_mode="none"
            )
            bf16_out, bf16_lazy = verify(
                inputs, tree, precision="tf32", bf16_mode="state"
            )
            torch.cuda.synchronize()
            correctness.append(
                {
                    "tree": tree_kind,
                    "seed": seed,
                    "tf32_vs_ieee": {
                        "output": error(tf32_out, reference_out),
                        "lazy_u": error(tf32_lazy.u, reference_lazy.u),
                        "prefix": error(tf32_lazy.prefix, reference_lazy.prefix),
                    },
                    "selective_bf16_vs_ieee": {
                        "output": error(bf16_out, reference_out),
                        "lazy_u": error(bf16_lazy.u, reference_lazy.u),
                        "prefix": error(bf16_lazy.prefix, reference_lazy.prefix),
                    },
                }
            )

    parent = make_parent("mixed", args.tokens)
    tree = build_tree_structure_static(parent)
    inputs = make_inputs(123, parent)
    latency = {}
    for mode in ("none", "state"):
        graph = capture(
            lambda mode=mode: verify(
                inputs, tree, precision="tf32", bf16_mode=mode
            )
        )
        median, p20, p80 = triton.testing.do_bench(
            graph.replay,
            warmup=200,
            rep=args.repetitions,
            quantiles=[0.5, 0.2, 0.8],
        )
        latency[mode] = {
            "median_us_per_layer": 1000 * median,
            "p20_us_per_layer": 1000 * p20,
            "p80_us_per_layer": 1000 * p80,
        }

    result = {
        "device": torch.cuda.get_device_name(),
        "shape": {
            "batch": 1,
            "tokens": args.tokens,
            "key_heads": 16,
            "value_heads": 48,
            "key_dim": 128,
            "value_dim": 128,
        },
        "accumulator": "fp32",
        "latency": latency,
        "selective_bf16_speedup": (
            latency["none"]["median_us_per_layer"]
            / latency["state"]["median_us_per_layer"]
        ),
        "correctness": correctness,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
