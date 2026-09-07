import importlib.util
import pytest


def api():
    assert importlib.util.find_spec('scripts.analyze_r03_review_diagnostics') is not None
    from scripts import analyze_r03_review_diagnostics
    return analyze_r03_review_diagnostics


def row(group, cutoff, y, pairs=2, prob=.25):
    return dict(airport='KAGC', group=group, cutoff_us=cutoff*1000000,
                scene_id=f'{group}-{cutoff}', horizon=30, y=y, risk=int(y>=1),
                threshold=1, pair_opportunity=pairs, prob_B=prob, count_B=.5,
                count_A=.7, pair_baseline=.6, prevalence_baseline=.3,
                max_report_age_s=10., n_aircraft=3)


def test_local_baseline_uses_only_matured_prior_groups():
    rows = [row('a',0,2),row('b',100,0),row('c',200,1)]
    result=api().local_baselines(rows)
    assert len(result)==2
    assert result[0]['local_prevalence']==1.
    assert result[0]['local_pair_count']==2.
    assert result[1]['local_prevalence']==.5
    assert result[1]['local_pair_count']==1.
    rows[-1]['y']=900
    assert api().local_baselines(rows)[0]['local_pair_count']==2.


def test_unmatured_group_cannot_calibrate_next_group():
    assert api().local_baselines([row('a',0,2),row('b',20,0)])==[]


def test_group_weighting_not_scene_weighting():
    rows=[row('a',0,0),row('a',1,0),row('b',100,3)]
    for r in rows: r['count_B']=0
    result=api().score(rows,'count_B','y',probability=False)
    assert result['equal_group_mean']==1.5
    assert result['n_groups']==2 and result['n_scenes']==3


def test_reliability_includes_probability_one_in_last_bin():
    rows=[row('a',0,0,prob=0),row('a',1,1,prob=1)]
    bins=api().reliability(rows)
    assert bins[0]['n']==1 and bins[-1]['n']==1
    assert bins[0]['observed_rate']==0 and bins[-1]['observed_rate']==1


def test_local_thresholds_cannot_change_with_group():
    rows=[row('a',0,2),row('b',100,0)]
    rows[1]['threshold']=3
    with pytest.raises(ValueError): api().local_baselines(rows)
