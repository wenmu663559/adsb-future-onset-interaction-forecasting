"""Post-review, group-level uncertainty; no model fitting or result selection."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def paired_summary(differences, *, repetitions=20000, seed=170104):
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('finite nonempty group differences required')
    if repetitions < 1:
        raise ValueError('positive bootstrap repetition count required')
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, (repetitions, len(values)), replace=True).mean(axis=1)
    wins, losses = int(np.sum(values < 0)), int(np.sum(values > 0))
    n = wins + losses
    p = min(1., 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2 ** n) if n else 1.
    return {
        'group_count': len(values), 'mean_difference': float(values.mean()),
        'ci95': np.quantile(samples, [.025, .975]).tolist(),
        'wins': wins, 'losses': losses, 'ties': len(values) - n,
        'sign_p_two_sided': p,
    }


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('p values must lie in [0,1]')
    order = np.argsort(p, kind='stable')
    adjusted = np.zeros(len(p))
    running = 0.
    for rank, index in enumerate(order):
        running = max(running, min(1., (len(p) - rank) * p[index]))
        adjusted[index] = running
    return adjusted.tolist()


def main():
    protocol_path = ROOT / 'configs/r03_review_supplement_protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    records, hashes = [], {}
    for airport, filename, pair_key, probability_key in (
        ('KBTP', 'r03_future_onset_confirmation_2026-09-03.json', 'B_vs_pair_exposure_count_MAE', 'B_logistic_vs_prevalence_Brier'),
        ('KAGC', 'r03_kagc_external_validation_2026-09-03.json', 'B_vs_KBTP_pair_exposure_count_MAE', 'B_logistic_vs_KBTP_prevalence_Brier'),
    ):
        path = ROOT / 'reports/references' / filename
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        result = json.loads(path.read_text(encoding='utf-8'))
        for horizon in ('30', '120', '300'):
            for name, key in [('count_B_vs_A', 'B_vs_A_count_MAE'), ('count_B_vs_pair_rate', pair_key), ('prob_B_vs_prevalence', probability_key)]:
                comparison = result['horizons'][horizon][key]
                by_group = comparison['candidate_minus_baseline_by_day']
                summary = paired_summary(
                    [by_group[g] for g in sorted(by_group)],
                    repetitions=protocol['group_bootstrap']['repetitions'],
                    seed=protocol['group_bootstrap']['seed'],
                )
                baseline = comparison['baseline_equal_day_mean']
                summary.update(airport=airport, horizon=int(horizon), comparison=name,
                    source_groups=sorted(by_group), baseline_mean=baseline,
                    candidate_mean=comparison['candidate_equal_day_mean'],
                    relative_improvement_percent=-100 * summary['mean_difference'] / baseline if baseline else None)
                records.append(summary)
    for row, p in zip(records, holm([r['sign_p_two_sided'] for r in records]), strict=True):
        row['sign_p_holm_18'] = p
    output = ROOT / 'reports/references/r03_review_group_uncertainty_2026-09-04.json'
    output.write_text(json.dumps({
        'analysis_status': 'post-review supplementary; not new blind confirmation',
        'input_sha256': hashes,
        'protocol_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        'ci_interpretation': 'Pointwise percentile intervals for equal-source-group mean differences; not simultaneous; small samples and between-group dependence limit inference.',
        'test_interpretation': 'Exact two-sided sign tests concern the direction distribution, not the mean difference; Holm family is all 18 comparisons.',
        'records': records,
    }, indent=2) + '\n', encoding='utf-8')
    for r in records:
        print(f"{r['airport']} {r['horizon']:3} {r['comparison']:25} delta={r['mean_difference']:.6f} CI={r['ci95']} p={r['sign_p_two_sided']:.4f} Holm={r['sign_p_holm_18']:.4f}")
    print(output)


if __name__ == '__main__':
    main()
