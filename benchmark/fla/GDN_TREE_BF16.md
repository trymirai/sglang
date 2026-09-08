# Selective BF16 Tree-GDN Verification

## Scope

This change tests BF16 tensor-core operands with FP32 accumulation in the
tree-GDN verifier. It does not change Weaver training, tree construction,
sampling, or commit semantics.

The `state` mode uses BF16 operands for:

- key-key and query-key Gram products;
- the committed-state products used by K2 and K3.

It keeps prefix construction, the triangular inverse, forward substitution,
readout against lazy writes, lazy state, and committed state in the existing
TF32/FP32 path. Rejected speculative branches still never enter committed
state.

Enable the measured mode with:

```bash
SGLANG_GDN_TREE_BF16_OPERANDS=state
```

## Fixed-Context Result

The acceptance test used one frozen checkpoint, identical trees, and 192
contexts grouped into eight target-generated trajectories.

| Metric | TF32 | Selective BF16 |
|---|---:|---:|
| Expected accepted draft tokens | 7.54984355 | 7.55122015 |
| Paired difference |  | +0.00137660 |
| Relative difference |  | +0.01823% |
| Top-1 flips |  | 67 / 12,480 (0.53686%) |

The trajectory-cluster bootstrap 95% interval for the paired acceptance
difference is `[-0.00368479, +0.00643563]` tokens. This establishes no
detectable aggregate regression within a predeclared `0.01`-token
non-inferiority margin. It does not establish bitwise equivalence.

Frozen checkpoint SHA256:

```text
a5214d248e7646d3a67f92fba3f4dc5997ab2169ca5358516bafbbdfceca2c54
```

Full analysis artifact SHA256:

```text
b8c98f91a7880c16bad142eb966f14beb8bae8751802bad12dc401d5da9ed9ad
```

## Benchmark

Run on a B200:

```bash
python benchmark/fla/bench_gdn_tree_bf16.py --tokens 65
```

The benchmark reports CUDA-graph latency for the production TF32 and
selective-BF16 paths, plus output, lazy-U, and prefix errors against IEEE dots
for chain, wide, and mixed trees across three seeds.

## Evidence Boundary

The first temperature-1 E2E smoke run is not a valid throughput comparison:
the two numerical paths produced divergent trajectories and output lengths.
Its raw throughput delta must not be quoted.

Release requires the prepared paired E2E gate:

- one physical non-zero B200;
- four independent seed pairs;
- alternating ABBA and BAAB arm order;
- all 80 first-turn MT-Bench prompts at temperature 1 with reasoning enabled;
- identical checkpoint, image, package, prompt, and source manifests;
- no GPU hardware-slowdown flags;
- hierarchical bootstrap lower 95% throughput bound above zero.

Until that gate finishes, this PR claims fixed-context acceptance
non-inferiority and a kernel implementation ready for E2E measurement, not an
E2E throughput win.
