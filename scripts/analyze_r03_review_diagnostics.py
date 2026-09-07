"""Descriptive calibration, freshness and temporally honest local references."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def score(rows, prediction, truth, *, probability=False):
    grouped = defaultdict(list)
    for r in rows:
        error = float(r[prediction]) - float(r[truth])
        grouped[r['group']].append(error ** 2 if probability else abs(error))
    by_group = {g: statistics.fmean(v) for g, v in sorted(grouped.items())}
    return dict(equal_group_mean=statistics.fmean(by_group.values()) if by_group else None,
                by_group=by_group, n_groups=len(by_group), n_scenes=len(rows),
                n_positive=sum(int(r['risk']) for r in rows))


def local_baselines(rows):
    output=[]
    for horizon in sorted({r['horizon'] for r in rows}):
        selected=[r for r in rows if r['horizon']==horizon and r['airport']=='KAGC']
        if not selected: continue
        if len({r['threshold'] for r in selected})!=1:
            raise ValueError('event threshold must remain frozen across groups')
        by_group=defaultdict(list)
        for r in selected: by_group[r['group']].append(r)
        starts={g:min(r['cutoff_us'] for r in v) for g,v in by_group.items()}
        ends={g:max(r['cutoff_us'] for r in v)+horizon*1000000 for g,v in by_group.items()}
        for group in sorted(by_group, key=lambda g:(starts[g],g)):
            eligible=sorted(g for g in by_group if ends[g]<starts[group])
            past=[r for g in eligible for r in by_group[g]]
            if not past: continue
            opportunity=sum(r['pair_opportunity'] for r in past)
            if opportunity<=0: raise ValueError('positive historical pair opportunity required')
            rate=sum(r['y'] for r in past)/opportunity
            prevalence=statistics.fmean(r['risk'] for r in past)
            for r in by_group[group]:
                output.append({**r, 'calibration_groups':eligible,
                    'calibration_latest_label_us':max(ends[g] for g in eligible),
                    'local_pair_count':rate*r['pair_opportunity'],
                    'local_mean_count':statistics.fmean(x['y'] for x in past),
                    'local_median_count':statistics.median(x['y'] for x in past),
                    'local_prevalence':prevalence})
    return output


def reliability(rows, bins=10):
    output=[]
    for i in range(bins):
        lower,upper=i/bins,(i+1)/bins
        chosen=[r for r in rows if lower<=r['prob_B'] and (r['prob_B']<upper or (i==bins-1 and r['prob_B']==1))]
        output.append(dict(lower=lower,upper=upper,n=len(chosen),
            n_groups=len({r['group'] for r in chosen}),
            mean_probability=statistics.fmean(r['prob_B'] for r in chosen) if chosen else None,
            observed_rate=statistics.fmean(r['risk'] for r in chosen) if chosen else None))
    return output


def diagnostics(rows):
    output={}
    for airport in ('KBTP','KAGC'):
        output[airport]={}
        for h in (30,120,300):
            chosen=[r for r in rows if r['airport']==airport and r['horizon']==h]
            strata={}
            for name, subset in [('fresh',[r for r in chosen if r['max_report_age_s']<=10]),
                                 ('older',[r for r in chosen if r['max_report_age_s']>10])]:
                strata[name]={
                    'count_B':score(subset,'count_B','y'),
                    'count_A':score(subset,'count_A','y'),
                    'prob_B':score(subset,'prob_B','risk',probability=True),
                    'prevalence':score(subset,'prevalence_baseline','risk',probability=True),
                }
            output[airport][str(h)]={'reliability_pooled':reliability(chosen),'freshness':strata}
    local=local_baselines(rows)
    local_results={}
    for h in (30,120,300):
        selected=[r for r in local if r['horizon']==h]
        local_results[str(h)]={
            'groups':sorted({r['group'] for r in selected}),
            'n_scenes':len(selected),
            'count':{key:score(selected,key,'y') for key in ('count_B','count_A','pair_baseline','local_pair_count','local_mean_count','local_median_count')},
            'probability':{key:score(selected,key,'risk',probability=True) for key in ('prob_B','prevalence_baseline','local_prevalence')},
            'calibration_groups':{r['group']:r['calibration_groups'] for r in selected},
        }
    return dict(by_airport=output,local_chronological_KAGC=local_results),local


def main():
    root=ROOT/'outputs/experiments/r03_review_supplement_2026-09-04'
    path=root/'predictions.jsonl'
    with path.open(encoding='utf-8') as stream: rows=[json.loads(line) for line in stream]
    report,local=diagnostics(rows)
    report['prediction_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    report['analysis_status']='Post-review descriptive analysis, not new blind confirmation; local references use past KAGC labels, not pure zero-shot.'
    report['calibration_weighting']='Pooled reliability bins; source-group-equal errors in freshness and local comparisons.'
    report['limitations']='Age strata can differ in traffic and event prevalence; observational differences do not identify a causal freshness effect. Local first-group warmup is omitted from every compared method.'
    output=ROOT/'reports/references/r03_review_diagnostics_2026-09-04.json'
    output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    with (root/'local_baseline_predictions.jsonl').open('w',encoding='utf-8') as stream:
        for r in local: stream.write(json.dumps(r)+'\n')
    for h,r in report['local_chronological_KAGC'].items():
        print(h, 'groups',len(r['groups']),'n',r['n_scenes'])
        print('count', {k:round(v['equal_group_mean'],6) for k,v in r['count'].items()})
        print('probability', {k:round(v['equal_group_mean'],6) for k,v in r['probability'].items()})
    print(output)


if __name__=='__main__': main()
