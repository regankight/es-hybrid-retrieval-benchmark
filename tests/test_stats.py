import pytest

from stats import Interval, bootstrap_mean, bootstrap_median, paired_difference

FAST = dict(resamples=2000, seed=1)


def test_a_constant_difference_has_a_zero_width_interval():
    a = {f"q{i}": 0.5 for i in range(20)}
    b = {q: 0.6 for q in a}
    got = paired_difference(a, b, **FAST)
    assert (got.estimate, got.low, got.high) == pytest.approx((0.1, 0.1, 0.1))


def test_identical_scores_give_exactly_zero_and_do_not_exclude_zero():
    a = {f"q{i}": i / 10 for i in range(10)}
    got = paired_difference(a, dict(a), **FAST)
    assert (got.estimate, got.low, got.high) == (0.0, 0.0, 0.0)
    assert not got.excludes_zero()


def test_same_seed_is_reproducible_and_a_different_seed_is_not_identical():
    # continuous values: with a few binary ones every bootstrap mean is a multiple of 1/n,
    # so different seeds can land on identical percentiles by coincidence
    values = [0.13, 0.91, 0.42, 0.77, 0.05, 0.63, 0.28, 0.99, 0.51, 0.34, 0.86, 0.19]
    assert bootstrap_mean(values, **FAST) == bootstrap_mean(values, **FAST)
    assert bootstrap_mean(values, resamples=2000, seed=2) != bootstrap_mean(values, **FAST)


def test_the_interval_brackets_the_estimate():
    values = [0.0, 1.0] * 15
    got = bootstrap_mean(values, **FAST)
    assert got.estimate == 0.5 and got.low < 0.5 < got.high


def test_more_data_narrows_the_interval():
    small = bootstrap_mean([0.0, 1.0] * 5, **FAST)
    large = bootstrap_mean([0.0, 1.0] * 200, **FAST)
    assert (large.high - large.low) < (small.high - small.low)


def test_higher_confidence_widens_the_interval():
    values = [0.0, 1.0] * 20
    narrow = bootstrap_mean(values, confidence=0.80, **FAST)
    wide = bootstrap_mean(values, confidence=0.99, **FAST)
    assert (wide.high - wide.low) > (narrow.high - narrow.low)


def test_a_consistent_improvement_excludes_zero_and_noise_does_not():
    base = {f"q{i}": 0.5 for i in range(40)}
    better = {q: 0.5 + (0.1 if i % 4 else 0.05) for i, q in enumerate(base)}
    assert paired_difference(base, better, **FAST).excludes_zero()
    noisy = {q: 0.5 + (0.3 if i % 2 else -0.3) for i, q in enumerate(base)}
    assert not paired_difference(base, noisy, **FAST).excludes_zero()


def test_mismatched_query_sets_are_rejected():
    with pytest.raises(ValueError, match="identical query ids"):
        paired_difference({"a": 1.0}, {"b": 1.0})


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        bootstrap_mean([])
    with pytest.raises(ValueError):
        bootstrap_mean([1.0], confidence=1.0)


def test_interval_helper():
    assert Interval(0.1, 0.02, 0.2).excludes_zero()
    assert Interval(-0.1, -0.2, -0.02).excludes_zero()
    assert not Interval(0.0, -0.1, 0.1).excludes_zero()


def test_the_median_interval_is_robust_to_an_outlier_that_would_wreck_a_mean():
    values = [10.0] * 30 + [10_000.0]
    median = bootstrap_median(values, **FAST)
    assert (median.estimate, median.low, median.high) == (10.0, 10.0, 10.0)
    assert bootstrap_mean(values, **FAST).high > 100  # the mean is dragged far away


def test_the_median_interval_brackets_the_estimate_for_noisy_data():
    values = [float(v) for v in range(1, 40)]
    got = bootstrap_median(values, **FAST)
    assert got.estimate == 20.0 and got.low < 20.0 < got.high
