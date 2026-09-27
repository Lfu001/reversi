import pytest

PURPOSES = (
    "branch_position",
    "action_sampling",
    "sibling_rollout",
    "evaluation",
)


def test_rng_streams_reproduce_independently():
    from lfu_tree_self_play.rng import create_rng_streams

    first = create_rng_streams(42)
    second = create_rng_streams(42)

    for purpose in PURPOSES:
        assert [getattr(first, purpose).random() for _ in range(3)] == [
            getattr(second, purpose).random() for _ in range(3)
        ]


def test_consuming_one_stream_does_not_advance_another():
    from lfu_tree_self_play.rng import create_rng_streams

    first = create_rng_streams(42)
    second = create_rng_streams(42)
    for _ in range(100):
        first.branch_position.random()

    assert first.action_sampling.random() == second.action_sampling.random()


def test_derived_seeds_differ_by_purpose_and_run():
    from lfu_tree_self_play.rng import derive_seed

    seeds = {derive_seed(42, purpose) for purpose in PURPOSES}

    assert len(seeds) == 4
    assert derive_seed(42, "evaluation") != derive_seed(43, "evaluation")


@pytest.mark.parametrize("run_seed", [-1, True, 1.5])
def test_rng_rejects_invalid_run_seed(run_seed):
    from lfu_tree_self_play.rng import create_rng_streams

    with pytest.raises(ValueError):
        create_rng_streams(run_seed)


def test_derive_seed_rejects_unknown_purpose():
    from lfu_tree_self_play.rng import derive_seed

    with pytest.raises(ValueError):
        derive_seed(42, "unknown")
