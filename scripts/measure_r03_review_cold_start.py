"""Five fresh-process startup measurements; no training or data changes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/experiments/r03_review_supplement_2026-09-04'

def worker():
    start=time.perf_counter()
    sys.path.insert(0,str(ROOT))
    sys.path.insert(0,str(ROOT/'src'))
    import joblib
    import numpy as np
    from scripts.run_r03_future_onset_confirmation import AIRPORT_CENTERS
    from airspace_complexity.r03_future_onset import build_feature_ladder
    imported=time.perf_counter()
    model=joblib.load(OUT/'frozen_models.joblib')
    scenes=ROOT/'outputs/experiments/r03_future_onset_confirmation/20260903T075352Z-online-complexity-pilot/pilot_scenes.jsonl'
    with scenes.open(encoding='utf-8') as f: row=json.loads(next(f))
    loaded=time.perf_counter()
    center=AIRPORT_CENTERS[row['airport_id']]
    values=build_feature_ladder(row,airport_lat_deg=center[0],airport_lon_deg=center[1])['B']
    x=np.asarray([[values[n] for n in model['feature_names']['B']]])
    featured=time.perf_counter()
    predictions=[]
    for h,m in model['models'].items():
        y=m['B'].predict(x)
        y=np.maximum(np.expm1(y),0.) if m['model_name']=='ridge_log1p' else np.maximum(y,0.)
        p=m['prob_B'].predict_proba(x)[:,1]
        predictions.append({'horizon':h,'count':float(y[0]),'probability':float(p[0])})
    end=time.perf_counter()
    assert len(predictions)==3 and all(np.isfinite([p['count'],p['probability']]).all() for p in predictions)
    return {'import_ms':(imported-start)*1000,'load_model_and_one_scene_ms':(loaded-imported)*1000,
        'feature_ms':(featured-loaded)*1000,'inference_ms':(end-featured)*1000,
        'first_compute_ms':(end-loaded)*1000,'worker_elapsed_ms':(end-start)*1000,
        'scene_id':row['scene_id'],'predictions':predictions}

def main():
    samples=[]
    for i in range(5):
        start=time.perf_counter()
        result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker'],cwd=ROOT,capture_output=True,text=True,check=True)
        elapsed=(time.perf_counter()-start)*1000
        r=json.loads(result.stdout)
        r['process_wall_ms']=elapsed
        samples.append(r)
    import numpy as np
    report={'n':5,'scope':'Fresh Python process, imports, frozen joblib load, first cached scene read, feature extraction and six outputs, and process exit.',
        'exclusions':'No training, no network, no live ingest, no history accumulation. OS disk cache is not cleared; process cold is not machine/cache cold.',
        'sampling':'Same first KBTP confirmation scene on each new process; descriptive five-run timing, not a population latency bound.',
        'samples':samples,'summary':{k:{'p50':float(np.quantile([r[k] for r in samples],.5)),'p95':float(np.quantile([r[k] for r in samples],.95))} for k in samples[0] if k.endswith('_ms')}}
    target=OUT/'cold_start_report.json'
    with target.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps(report['summary']))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.worker: print(json.dumps(worker()))
    else: main()
