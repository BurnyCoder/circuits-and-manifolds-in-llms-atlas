"""Tests for atlas.common.stats: Wilson score intervals for rates (Wilson 1927), checked against closed-form values.

Local context: the methods review asked for CIs on every rate-based verdict (refusal, steering). The Wilson interval
is the standard choice for small n and rates near 0 or 1 (Brown, Cai & DasGupta 2001,
https://doi.org/10.1214/ss/1009213286); statsmodels' proportion_confint(method="wilson") implements the same formula.
"""

import numpy as np
import pytest

from atlas.common.stats import rate_str, wilson_ci


def test_wilson_zero_successes_has_closed_form_upper_bound():
    # k = 0: lower bound 0, upper bound z² / (n + z²).
    lo, hi = wilson_ci(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-12)
    assert hi == pytest.approx(1.959964**2 / (10 + 1.959964**2), rel=1e-6)


def test_wilson_matches_hand_computation():
    # 13 / 43 (the calendar-circles causal test in the methods review): ≈ [18.6%, 45.1%].
    lo, hi = wilson_ci(13, 43)
    assert lo == pytest.approx(0.186, abs=1e-3)
    assert hi == pytest.approx(0.451, abs=1e-3)


def test_wilson_is_symmetric_and_bounded():
    lo, hi = wilson_ci(7, 7)
    assert hi == pytest.approx(1.0) and 0 < lo < 1
    lo2, hi2 = wilson_ci(0, 7)
    assert lo2 == pytest.approx(1 - hi) and hi2 == pytest.approx(1 - lo)


def test_rate_str_formats_percent_with_interval():
    assert rate_str(13, 43) == "30% [19, 45]"
    assert wilson_ci(0, 0) == (0.0, 1.0)  # no data: the uninformative interval


def test_group_bootstrap_keeps_groups_together():
    from atlas.common.stats import group_bootstrap

    # Two groups with constant values 0 and 1: resampling whole groups gives means in {0, 0.5, 1} only.
    groups = np.r_[np.zeros(50), np.ones(50)]
    vals = groups.copy()
    lo, hi = group_bootstrap(groups, lambda idx: vals[idx].mean(), n_boot=500)
    assert lo == 0.0 and hi == 1.0


def test_ci_rule_and_perm_pvalue():
    from atlas.common.stats import ci_rule, perm_pvalue

    assert ci_rule(0.72, 0.80, 0.7) == (True, True)  # whole CI above: pass
    assert ci_rule(0.65, 0.75, 0.7) == (False, True)  # crosses: partial (inconclusive)
    assert ci_rule(0.50, 0.69, 0.7) == (False, False)  # whole CI below: fail
    assert perm_pvalue(5.0, [1.0, 2.0, 3.0]) == 0.25  # (1 + 0) / (1 + 3)
