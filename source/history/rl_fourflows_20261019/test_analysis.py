import math
import pytest

from analyze import intervals, length, quantile


def test_integer_euc_cycle_includes_closure():
    points = [[0,0],[3,0],[3,4],[0,4]]
    assert length(points,[0,1,2,3]) == 14
    assert length(points,[0,1,3,2]) == 16
    with pytest.raises(ValueError):
        length(points,[0,1,1,3])


def test_interpolated_quantile_has_correct_endpoints():
    assert quantile([3,0,2,1],0) == 0
    assert quantile([3,0,2,1],1) == 3
    assert quantile([3,0,2,1],.5) == 1.5


def test_bootstrap_preserves_map_strata_weights():
    # All maps within a stratum agree: no bootstrap draw may mix weights.
    maps=[{'stratum':s,'value':v} for s,v in [('u',-3),('c',0),('p',6)] for _ in range(4)]
    result=intervals(maps,'value',replicates=1000)
    assert result['mean'] == 1
    assert result['percentile95'] == [1,1]
    assert result['bonferroni_three_comparisons'] == [1,1]


def test_multiplicity_interval_encloses_individual_interval():
    maps=[{'stratum':s,'value':float(i)} for s in ('u','c','p') for i in range(4)]
    result=intervals(maps,'value',replicates=2000)
    low,high=result['percentile95']
    low_adj,high_adj=result['bonferroni_three_comparisons']
    assert low_adj <= low <= result['mean'] <= high <= high_adj
