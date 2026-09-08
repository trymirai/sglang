import itertools
from collections import defaultdict
from math import prod

import torch

from sglang.srt.speculative.tree_sampling import (
    filter_target_probs,
    plackett_luce_order,
    target_only_tree_verify,
    traversal_tree_verify,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _emitted_tokens(sample, tokens):
    accepted = tuple(int(tokens[node]) for node in sample.accepted_nodes[1:])
    return accepted + (sample.bonus_token,)


def _target_sequence_distribution(tokens, parents, target_probs):
    distribution = defaultdict(float)

    def visit(node, prefix, mass):
        children = torch.nonzero(parents == node, as_tuple=False).flatten().tolist()
        child_by_token = {int(tokens[child]): child for child in children}
        assert len(child_by_token) == len(children)

        for token, probability in enumerate(target_probs[node].tolist()):
            if probability == 0.0:
                continue
            emitted = prefix + (token,)
            child = child_by_token.get(token)
            if child is None:
                distribution[emitted] += mass * probability
            else:
                visit(child, emitted, mass * probability)

    visit(0, (), 1.0)
    return dict(distribution)


def _categorical_cells(probs):
    probs = probs.to(torch.float64)
    probs = probs / probs.sum()
    cells = []
    lower = 0.0
    for token, probability in enumerate(probs.tolist()):
        if probability == 0.0:
            continue
        cells.append((token, probability, lower + probability / 2.0))
        lower += probability
    assert abs(lower - 1.0) < 1.0e-12
    return cells


def _assert_same_distribution(actual, expected):
    assert actual.keys() == expected.keys()
    assert abs(sum(actual.values()) - 1.0) < 1.0e-12
    assert abs(sum(expected.values()) - 1.0) < 1.0e-12
    for sequence, probability in expected.items():
        assert abs(actual[sequence] - probability) < 1.0e-10, (
            sequence,
            actual[sequence],
            probability,
        )


def _target_only_exact_distribution(tokens, parents, target_probs):
    distribution = defaultdict(float)
    row_cells = [_categorical_cells(row) for row in target_probs]
    for choices in itertools.product(*row_cells):
        result = target_only_tree_verify(
            tokens=tokens,
            parents=parents,
            target_probs=target_probs,
            uniforms=torch.tensor(
                [uniform for _, _, uniform in choices], dtype=torch.float64
            ),
        )
        mass = prod(probability for _, probability, _ in choices)
        distribution[_emitted_tokens(result, tokens)] += mass
    return dict(distribution)


def _sibling_order_options(children_by_parent, draft_logprobs):
    options_by_parent = []
    for children in children_by_parent:
        if len(children) < 2:
            continue
        weights = torch.softmax(draft_logprobs[children].to(torch.float64), dim=0)
        weight_by_child = dict(zip(children, weights.tolist(), strict=True))
        options = []
        for order in itertools.permutations(children):
            probability = 1.0
            for rank, child in enumerate(order):
                remaining = sum(weight_by_child[node] for node in order[rank:])
                probability *= weight_by_child[child] / remaining
            options.append((order, probability))
        options_by_parent.append(options)
    return options_by_parent


def _traversal_exact_distribution(tokens, parents, target_probs, draft_logprobs):
    num_nodes = parents.numel()
    vocab_size = target_probs.shape[1]
    distribution = defaultdict(float)
    children_by_parent = [
        torch.nonzero(parents == parent, as_tuple=False).flatten().tolist()
        for parent in range(num_nodes)
    ]
    order_groups = _sibling_order_options(children_by_parent, draft_logprobs)

    for order_choices in itertools.product(*order_groups):
        sibling_keys = torch.zeros(num_nodes, dtype=torch.float64)
        order_probability = 1.0
        for order, probability in order_choices:
            order_probability *= probability
            for rank, child in enumerate(order):
                sibling_keys[child] = len(order) - rank

        active = torch.ones(num_nodes, dtype=torch.bool)
        target = [row.to(torch.float64).clone() for row in target_probs]
        draft = [torch.zeros(vocab_size, dtype=torch.float64) for _ in range(num_nodes)]
        for parent, children in enumerate(children_by_parent):
            if not children:
                continue
            probabilities = torch.softmax(
                draft_logprobs[children].to(torch.float64), dim=0
            )
            draft[parent][tokens[children]] = probabilities

        acceptance = torch.zeros(num_nodes, dtype=torch.float64)
        acceptance[0] = 1.0

        def refresh_descendants(
            parent, active_state, target_state, draft_state, acceptance_state
        ):
            for child in children_by_parent[parent]:
                if not active_state[child]:
                    continue
                token = int(tokens[child])
                proposal = float(draft_state[parent][token])
                assert proposal > 0.0
                acceptance_state[child] = min(
                    float(acceptance_state[parent])
                    * float(target_state[parent][token])
                    / proposal,
                    1.0,
                )
                refresh_descendants(
                    child,
                    active_state,
                    target_state,
                    draft_state,
                    acceptance_state,
                )

        refresh_descendants(0, active, target, draft, acceptance)

        def enumerate_decisions(
            active_state,
            target_state,
            draft_state,
            acceptance_state,
            accept_uniforms,
            branch_probability,
        ):
            current = 0
            while True:
                children = [
                    child
                    for child in children_by_parent[current]
                    if active_state[child]
                ]
                if not children:
                    leaf = current
                    break
                current = max(children, key=lambda child: float(sibling_keys[child]))

            accept_probability = 1.0 if leaf == 0 else float(acceptance_state[leaf])
            # Ratios that are mathematically 0 or 1 can miss the endpoint by
            # one FP64 ulp and create an impossible zero-residual branch.
            if accept_probability < 1.0e-15:
                accept_probability = 0.0
            elif accept_probability > 1.0 - 1.0e-15:
                accept_probability = 1.0
            if accept_probability > 0.0:
                path = []
                current = leaf
                while current >= 0:
                    path.append(current)
                    current = int(parents[current])
                path.reverse()

                witness_uniforms = accept_uniforms + [accept_probability / 2.0]
                witness_uniforms += [0.5] * (num_nodes - len(witness_uniforms))
                for bonus, bonus_probability, bonus_uniform in _categorical_cells(
                    target_state[leaf]
                ):
                    result = traversal_tree_verify(
                        tokens=tokens,
                        parents=parents,
                        target_probs=target_probs,
                        draft_logprobs=draft_logprobs,
                        sibling_keys=sibling_keys,
                        accept_uniforms=torch.tensor(
                            witness_uniforms, dtype=torch.float64
                        ),
                        bonus_uniform=bonus_uniform,
                    )
                    expected = tuple(int(tokens[node]) for node in path[1:]) + (bonus,)
                    assert _emitted_tokens(result, tokens) == expected
                    distribution[expected] += (
                        order_probability
                        * branch_probability
                        * accept_probability
                        * bonus_probability
                    )

            reject_probability = 1.0 - accept_probability
            if reject_probability == 0.0:
                return

            next_active = active_state.clone()
            next_target = [row.clone() for row in target_state]
            next_draft = [row.clone() for row in draft_state]
            next_acceptance = acceptance_state.clone()
            parent = int(parents[leaf])
            parent_acceptance = float(next_acceptance[parent])
            residual = torch.clamp(
                parent_acceptance * next_target[parent] - next_draft[parent],
                min=0.0,
            )
            residual_mass = float(residual.sum())
            next_target[parent] = (
                residual / residual_mass
                if residual_mass > 0.0
                else torch.zeros_like(residual)
            )

            rejected_token = int(tokens[leaf])
            next_draft[parent][rejected_token] = 0.0
            remaining_mass = float(next_draft[parent].sum())
            if remaining_mass > 0.0:
                next_draft[parent] /= remaining_mass

            denominator = residual_mass + 1.0 - parent_acceptance
            next_acceptance[parent] = (
                residual_mass / denominator if denominator > 0.0 else 0.0
            )
            next_active[leaf] = False
            refresh_descendants(
                parent,
                next_active,
                next_target,
                next_draft,
                next_acceptance,
            )
            enumerate_decisions(
                next_active,
                next_target,
                next_draft,
                next_acceptance,
                accept_uniforms + [(accept_probability + 1.0) / 2.0],
                branch_probability * reject_probability,
            )

        enumerate_decisions(active, target, draft, acceptance, [], 1.0)

    return dict(distribution)


def _exact_tree_cases():
    return (
        (
            torch.tensor([99, 0, 1, 0, 2], dtype=torch.long),
            torch.tensor([-1, 0, 0, 1, 1], dtype=torch.long),
            torch.tensor(
                [
                    [0.45, 0.35, 0.20],
                    [0.25, 0.15, 0.60],
                    [0.20, 0.50, 0.30],
                    [0.10, 0.30, 0.60],
                    [0.55, 0.25, 0.20],
                ],
                dtype=torch.float64,
            ),
            torch.log(torch.tensor([1.0, 0.65, 0.35, 0.30, 0.70], dtype=torch.float64)),
        ),
        (
            torch.tensor([99, 0, 2, 1, 0], dtype=torch.long),
            torch.tensor([-1, 0, 0, 2, 2], dtype=torch.long),
            torch.tensor(
                [
                    [0.50, 0.00, 0.50],
                    [0.00, 0.40, 0.60],
                    [0.25, 0.65, 0.10],
                    [0.70, 0.00, 0.30],
                    [0.20, 0.20, 0.60],
                ],
                dtype=torch.float64,
            ),
            torch.log(torch.tensor([1.0, 0.20, 0.80, 0.70, 0.30], dtype=torch.float64)),
        ),
        (
            torch.tensor([99, 0, 1, 2], dtype=torch.long),
            torch.tensor([-1, 0, 0, 0], dtype=torch.long),
            torch.tensor(
                [
                    [0.20, 0.30, 0.50],
                    [0.10, 0.20, 0.70],
                    [0.60, 0.25, 0.15],
                    [0.35, 0.55, 0.10],
                ],
                dtype=torch.float64,
            ),
            torch.log(torch.tensor([1.0, 0.60, 0.30, 0.10], dtype=torch.float64)),
        ),
        (
            torch.tensor([99, 1, 3, 0], dtype=torch.long),
            torch.tensor([-1, 0, 0, 1], dtype=torch.long),
            torch.tensor(
                [
                    [0.10, 0.40, 0.20, 0.30],
                    [0.55, 0.05, 0.25, 0.15],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.05, 0.15, 0.30, 0.50],
                ],
                dtype=torch.float64,
            ),
            torch.log(torch.tensor([1.0, 0.70, 0.30, 1.0], dtype=torch.float64)),
        ),
    )


def test_target_only_matches_exact_multistep_target_distribution():
    for tokens, parents, target_probs, _ in _exact_tree_cases():
        expected = _target_sequence_distribution(tokens, parents, target_probs)
        actual = _target_only_exact_distribution(tokens, parents, target_probs)
        _assert_same_distribution(actual, expected)


def test_traversal_matches_exact_multistep_target_distribution():
    for tokens, parents, target_probs, draft_logprobs in _exact_tree_cases():
        expected = _target_sequence_distribution(tokens, parents, target_probs)
        actual = _traversal_exact_distribution(
            tokens, parents, target_probs, draft_logprobs
        )
        _assert_same_distribution(actual, expected)


def test_verifiers_reject_duplicate_sibling_tokens():
    tokens = torch.tensor([99, 0, 0], dtype=torch.long)
    parents = torch.tensor([-1, 0, 0], dtype=torch.long)
    target_probs = torch.tensor(
        [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=torch.float64
    )

    try:
        target_only_tree_verify(
            tokens=tokens,
            parents=parents,
            target_probs=target_probs,
            uniforms=torch.full((3,), 0.25, dtype=torch.float64),
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("target-only accepted duplicate sibling tokens")

    try:
        traversal_tree_verify(
            tokens=tokens,
            parents=parents,
            target_probs=target_probs,
            draft_logprobs=torch.zeros(3, dtype=torch.float64),
            sibling_keys=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
            accept_uniforms=torch.full((3,), 0.5, dtype=torch.float64),
            bonus_uniform=0.5,
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("traversal accepted duplicate sibling tokens")


def test_plackett_luce_order_uses_weighted_without_replacement_keys():
    probs = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64)
    uniforms = torch.tensor([0.5, 0.9, 0.1], dtype=torch.float64)
    order = plackett_luce_order(probs, uniforms)
    assert sorted(order.tolist()) == [0, 1, 2]
    assert order.tolist() == [1, 0, 2]


def test_target_only_tree_emits_the_target_sample_on_a_miss():
    tokens = torch.tensor([99, 0, 1], dtype=torch.long)
    parents = torch.tensor([-1, 0, 0], dtype=torch.long)
    target = torch.tensor(
        [
            [0.2, 0.3, 0.5],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    uniforms = torch.tensor([0.9, 0.1, 0.1], dtype=torch.float64)
    result = target_only_tree_verify(
        tokens=tokens,
        parents=parents,
        target_probs=target,
        uniforms=uniforms,
    )
    assert result.accepted_nodes == (0,)
    assert result.bonus_token == 2


def test_target_only_never_emits_a_zero_probability_token_at_zero_uniform():
    result = target_only_tree_verify(
        tokens=torch.tensor([99]),
        parents=torch.tensor([-1]),
        target_probs=torch.tensor([[0.0, 1.0]], dtype=torch.float64),
        uniforms=torch.tensor([0.0], dtype=torch.float64),
    )
    assert result.bonus_token == 1


def test_target_filters_match_joint_and_sequential_sampler_orders():
    probs = torch.tensor([[0.60, 0.25, 0.15]], dtype=torch.float64)
    params = {
        "top_ks": torch.tensor([2]),
        "top_ps": torch.tensor([0.70]),
        "min_ps": torch.tensor([0.0]),
    }
    joint = filter_target_probs(probs, **params, sequential=False)
    sequential = filter_target_probs(probs, **params, sequential=True)
    assert torch.allclose(
        joint, torch.tensor([[12 / 17, 5 / 17, 0.0]], dtype=torch.float64)
    )
    assert torch.equal(sequential, torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64))

    min_p = filter_target_probs(
        probs,
        top_ks=torch.tensor([3]),
        top_ps=torch.tensor([1.0]),
        min_ps=torch.tensor([0.5]),
        sequential=False,
    )
    assert torch.equal(min_p, torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64))


def test_target_only_tree_first_token_matches_target_distribution():
    tokens = torch.tensor([99, 0, 2], dtype=torch.long)
    parents = torch.tensor([-1, 0, 0], dtype=torch.long)
    target = torch.tensor(
        [
            [0.2, 0.3, 0.5],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    counts = torch.zeros(3, dtype=torch.int64)
    samples = 10_000
    for index in range(samples):
        uniform = (index + 0.5) / samples
        result = target_only_tree_verify(
            tokens=tokens,
            parents=parents,
            target_probs=target,
            uniforms=torch.tensor([uniform, 0.5, 0.5]),
        )
        first_token = (
            int(tokens[result.accepted_nodes[1]])
            if len(result.accepted_nodes) > 1
            else result.bonus_token
        )
        counts[first_token] += 1
    assert torch.equal(counts, torch.tensor([2000, 3000, 5000]))


def test_traversal_matches_target_for_all_two_token_sibling_orders():
    tokens = torch.tensor([99, 0, 1], dtype=torch.long)
    parents = torch.tensor([-1, 0, 0], dtype=torch.long)
    target = torch.tensor(
        [
            [0.55, 0.25, 0.20],
            [0.2, 0.3, 0.5],
            [0.4, 0.4, 0.2],
        ],
        dtype=torch.float64,
    )
    draft_logprobs = torch.log(torch.tensor([1.0, 0.7, 0.3]))

    first_token_counts = torch.zeros(3, dtype=torch.int64)
    samples_per_order = 10_000
    orders = list(itertools.permutations([1, 2]))
    order_probabilities = [0.7, 0.3]
    for order, order_probability in zip(orders, order_probabilities, strict=True):
        sibling_keys = torch.zeros(3, dtype=torch.float64)
        sibling_keys[order[0]] = 1.0
        sibling_keys[order[1]] = 0.0
        weighted_samples = round(samples_per_order * order_probability)
        for index in range(weighted_samples):
            u0 = (index + 0.5) / weighted_samples
            result = traversal_tree_verify(
                tokens=tokens,
                parents=parents,
                target_probs=target,
                draft_logprobs=draft_logprobs,
                sibling_keys=sibling_keys,
                accept_uniforms=torch.tensor([u0, 0.5, 0.5]),
                bonus_uniform=(index * 0.61803398875) % 1.0,
            )
            first_token = (
                int(tokens[result.accepted_nodes[1]])
                if len(result.accepted_nodes) > 1
                else result.bonus_token
            )
            first_token_counts[first_token] += 1

    observed = first_token_counts.to(torch.float64) / first_token_counts.sum()
    assert torch.allclose(observed, target[0], atol=0.015)


def test_traversal_uses_residual_target_for_later_siblings_and_bonus():
    tokens = torch.tensor([99, 0, 1], dtype=torch.long)
    parents = torch.tensor([-1, 0, 0], dtype=torch.long)
    target = torch.tensor(
        [
            [0.2, 0.3, 0.5],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    result = traversal_tree_verify(
        tokens=tokens,
        parents=parents,
        target_probs=target,
        draft_logprobs=torch.log(torch.tensor([1.0, 0.8, 0.2])),
        sibling_keys=torch.tensor([0.0, 1.0, 0.0]),
        # Reject token 0. Its residual gives token 1 probability 1/6, so 0.2
        # must also reject token 1. Reading the original target would accept it.
        accept_uniforms=torch.tensor([0.9, 0.2, 0.5]),
        bonus_uniform=0.5,
    )
    assert result.accepted_nodes == (0,)
    assert result.bonus_token == 2


def test_traversal_drains_a_zero_residual_subtree():
    result = traversal_tree_verify(
        tokens=torch.tensor([99, 0, 0, 1]),
        parents=torch.tensor([-1, 0, 1, 1]),
        target_probs=torch.tensor(
            [
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        ),
        draft_logprobs=torch.log(torch.tensor([1.0, 1.0, 0.5, 0.5])),
        sibling_keys=torch.tensor([0.0, 1.0, 1.0, 0.0]),
        accept_uniforms=torch.tensor([0.9, 0.5, 0.5, 0.5]),
        bonus_uniform=0.5,
    )
    assert result.accepted_nodes == (0,)
    assert result.bonus_token == 2
