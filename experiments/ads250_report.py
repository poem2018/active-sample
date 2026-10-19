"""Standalone offline report: ADS-alg/RF-prior vs saved 250->500 baselines.

Reads saved outputs only (all baselines read-only). Writes into
extensions/ads250_v1/report/. Never touches the other agent's HTML.
Aggregation: per-image values, images equally weighted; missing cells kept.
"""
import json
import time
from pathlib import Path

import numpy as np

from . import ads_celeba128_report as base_report
from .ads250_rf import BUDGETS, FINAL, INITIAL, NAME, OUT, RUN, WINDOW, case_dir, protocol, read

ARMS = ('UU', 'UW', 'WU', 'WW', 'SCRATCH', 'RANDOM', 'ADS')
LABELS = dict(UU='UU', UW='UW', WU='WU', WW='WW', SCRATCH='From-scratch', RANDOM='Random + From-scratch', ADS=NAME)
VISUAL = [250, 251, 260, 300, 350, 400, 450, 500]
SPARSE = [n for n in BUDGETS if n > INITIAL]


def baseline_dirs(case):
    b = RUN / 'groups' / 'b500_i250' / 'cases' / case['key']
    r = RUN / 'extensions/random250_v1/cases' / case['key'] / 'RANDOM'
    return dict(U=b / 'U', W=b / 'W', SCRATCH=b / 'SCRATCH', RANDOM=r), b / 'initialization'


def baseline_rounds(folder, init):
    rows = {}
    for n in range(INITIAL, FINAL + 1):
        f = init / f'round_{n:03d}' if n == INITIAL else folder / f'round_{n:03d}'
        if not (f / 'complete.json').exists():
            break
        state = read(f / 'state.json')
        assert state['count'] == n and len(set(state['history'])) == n
        rows[n] = dict(folder=f, metrics=read(f / 'metrics.json'), state=state)
    return rows


def summary(curve):
    """curve: {n: metric dict}. Exact for consecutive data; sparse grid shared with ADS."""
    have = {n for n, m in curve.items() if m is not None}
    complete_exact = all(n in have for n in range(INITIAL, FINAL + 1))
    complete_sparse = all(n in have for n in [INITIAL] + SPARSE)
    out = dict(last_count=max(have) if have else None, complete_exact=complete_exact, complete_sparse=complete_sparse)
    if INITIAL in have:
        out.update(initial_mse=curve[INITIAL]['reconstruction_mse'], initial_psnr=curve[INITIAL]['psnr'], initial_ssim=curve[INITIAL]['ssim'])
    if FINAL in have:
        f = curve[FINAL]
        out.update(final_mse=f['reconstruction_mse'], final_psnr=f['psnr'], final_ssim=f['ssim'],
                   final_unobserved_mse=f['unobserved_mse'], final_diversity=f['pairwise_diversity_rmse'],
                   final_variance=f['raw_candidate_variance'], final_observed_rmse=f['observed_mean_rmse'],
                   final_initial_rmse=f['initial_history_mean_rmse'])
    if complete_exact:
        out['update_mean_mse_exact'] = float(np.mean([curve[n]['reconstruction_mse'] for n in range(INITIAL + 1, FINAL + 1)]))
    if complete_sparse:
        ys = [curve[n]['reconstruction_mse'] for n in SPARSE]
        out['update_mean_mse_sparse34'] = float(np.mean(ys))
        grid = np.arange(INITIAL + 1, FINAL + 1)
        out['update_mean_mse_interpolated'] = float(np.interp(grid, SPARSE, ys).mean())
    return out


class Loader:
    def __init__(self, assets, audit):
        self.assets, self.audit = assets, audit

    def case(self, case):
        truth = np.load(RUN / case['truth']).astype(np.float64)
        with np.load(RUN / case['sensor']) as f:
            sensor = {k: f[k] for k in f.files}
        dirs, init = baseline_dirs(case)
        rounds = {p: baseline_rounds(d, init) for p, d in dirs.items()}
        curves, histories, frames = {}, {}, {}
        for arm in ARMS[:-1]:
            p = arm if arm in ('SCRATCH', 'RANDOM') else arm[0]
            key = 'weighted' if arm in ('UW', 'WW') else 'uniform'
            curves[arm] = {n: r['metrics'][key] for n, r in rounds[p].items()}
        for p, rr in rounds.items():
            histories[p] = rr[max(rr)]['state']['history'] if rr else []
        ads_metrics = case_dir(case) / 'metrics.json'
        if ads_metrics.exists():
            m = read(ads_metrics)
            curves['ADS'] = {int(n): v for n, v in m['metrics'].items() if v is not None}
            tweedie = {int(n): v for n, v in m['tweedie_intermediate_mse'].items()}
            hist = read(case_dir(case) / 'main/history.json')
            histories['ADS'] = hist['history']; ads_log = hist['log']
        else:
            curves['ADS'], tweedie, ads_log = {}, {}, []
        # Visual frames: aligned budgets present for each arm; MSE replayed from raw arrays.
        for n in VISUAL:
            row = {}
            for arm in ARMS:
                cand, w, hist = self.candidates(arm, n, rounds, case)
                if cand is None:
                    continue
                mean = np.einsum('i,ichw->chw', w, cand)
                saved = curves[arm][n]['reconstruction_mse']
                err = abs(float(np.square(mean - truth).mean()) - saved)
                self.audit['mse_checks'] += 1; self.audit['max_mse_replay_error'] = max(self.audit['max_mse_replay_error'], err)
                assert err < 2e-8, (case['key'], arm, n, err)
                obs = np.full((3, 128, 128), -.55); ids = sensor['indices'][hist].ravel()
                obs.flat[ids] = sensor['readings'][hist].ravel()
                np.testing.assert_array_equal(obs.ravel()[ids], sensor['readings'][hist].ravel())
                row[arm] = dict(mean=self.assets.add(base_report.pixels(mean)), input=self.assets.add(base_report.pixels(obs)),
                                latest=sensor['block_top_left'][hist[-1]].tolist() if n > INITIAL else None,
                                mse=saved, psnr=curves[arm][n]['psnr'])
                if arm == 'ADS':
                    row[arm]['candidates'] = self.assets.add(base_report.atlas(cand))
            frames[str(n)] = row
        tl = sensor['block_top_left']
        paths = {p: [dict(id=int(b), row=int(tl[b][0]), col=int(tl[b][1])) for b in h[INITIAL:]] for p, h in histories.items()}
        overlap = {}
        if histories.get('ADS'):
            a = set(histories['ADS'][INITIAL:])
            for p, h in histories.items():
                if p != 'ADS' and len(h) == FINAL:
                    overlap[p] = len(a & set(h[INITIAL:]))
        costs = self.costs(case, rounds)
        series = {arm: [[n, v['reconstruction_mse'], v['psnr'], v['ssim'], v['unobserved_mse'],
                         v['pairwise_diversity_rmse'], v['raw_candidate_variance'], v['observed_mean_rmse'],
                         v['initial_history_mean_rmse'], v.get('old_mean_rmse'), v.get('new_mean_rmse')]
                        for n, v in sorted(c.items())] for arm, c in curves.items()}
        return dict(key=case['key'], image_id=case['image_id'], gt=self.assets.add(base_report.pixels(truth)),
            summary={arm: summary(curves[arm]) for arm in ARMS}, series=series, frames=frames, paths=paths,
            acquisition_overlap_with_ads=overlap, tweedie=sorted([[n, v] for n, v in tweedie.items()]),
            ads_log=[dict(step=r['step'], n=r['n'], error=r['error'], grad_l2=r['grad_l2'], selected=r['selected'],
                          entropy=r.get('entropy_max')) for r in ads_log], costs=costs,
            ads_evolution=self.evolution(case))

    def candidates(self, arm, n, rounds, case):
        if arm == 'ADS':
            f = case_dir(case) / 'completions' / f'n_{n:03d}'
            if not (f / 'complete.json').exists():
                return None, None, None
            return (np.load(f / 'candidates.npy').astype(np.float64), np.full(16, 1 / 16), read(f / 'state.json')['history'])
        p = arm if arm in ('SCRATCH', 'RANDOM') else arm[0]
        r = rounds[p].get(n)
        if r is None:
            return None, None, None
        w = np.exp(np.asarray(r['state']['log_weights'], np.float64)) if arm in ('UW', 'WW') else np.full(16, 1 / 16)
        return np.load(r['folder'] / 'candidates.npy').astype(np.float64), w, r['state']['history']

    def evolution(self, case):
        out = []
        for n in VISUAL:
            f = case_dir(case) / 'completions' / f'n_{n:03d}' / 'candidates.npy'
            if f.exists():
                out.append([n, self.assets.add(base_report.atlas(np.load(f)))])
        return out

    def costs(self, case, rounds):
        rows = {}
        for p, rr in rounds.items():
            states = [r['state'] for n, r in rr.items() if n > INITIAL]
            rows[p] = dict(updates=len(states), generation_s=sum(s.get('generation_s', 0) for s in states),
                           nfe=sum(s.get('nfe', 0) for s in states),
                           candidate_model_evaluations=sum(s.get('candidate_model_evaluations', 16 * s.get('nfe', 0)) for s in states),
                           peak_allocated_gib=max((s.get('peak_allocated_bytes', 0) for s in states), default=0) / 2**30)
        main = case_dir(case) / 'main/complete.json'
        if main.exists():
            c = read(main)['costs']
            comp = [read(f)['costs'] for f in sorted((case_dir(case) / 'completions').glob('n_*/state.json')) if 'costs' in read(f)]
            rows['ADS'] = dict(main_guided_s=c['guided_s'], main_selection_s=c['selection_s'], main_io_s=c['io_s'],
                main_batched_forward=c['forward_calls'], main_vjp=c['vjp_calls'],
                completion_runs=len(comp), completion_guided_s=sum(x['guided_s'] for x in comp),
                completion_batched_forward=sum(x['forward_calls'] for x in comp),
                completion_vjp=sum(x['vjp_calls'] for x in comp),
                peak_allocated_gib=max([c['peak_allocated_bytes']] + [x['peak_allocated_bytes'] for x in comp]) / 2**30,
                evaluation_s=None)
        return rows


def aggregate(cases):
    out = dict(methods={}, contrasts=[])
    fields = ('initial_mse', 'final_mse', 'final_psnr', 'final_ssim', 'final_unobserved_mse', 'final_diversity',
              'update_mean_mse_exact', 'update_mean_mse_sparse34', 'update_mean_mse_interpolated')
    for arm in ARMS:
        row = {}
        for f in fields:
            vals = [c['summary'][arm].get(f) for c in cases]
            ok = [v for v in vals if v is not None]
            row[f] = dict(mean=float(np.mean(ok)) if len(ok) == len(cases) else None, n=len(ok),
                          partial_mean=float(np.mean(ok)) if ok else None)
        out['methods'][arm] = row
    for b in ('SCRATCH', 'RANDOM', 'UU', 'WW', 'WU', 'UW'):
        for f in ('final_mse', 'update_mean_mse_sparse34'):
            pairs = [(c['image_id'], c['summary']['ADS'].get(f), c['summary'][b].get(f)) for c in cases]
            diffs = [(i, a - x) for i, a, x in pairs if a is not None and x is not None]
            out['contrasts'].append(dict(label=f'ADS − {LABELS[b]}', field=f, images=len(diffs),
                mean_difference=float(np.mean([d for _, d in diffs])) if len(diffs) == len(cases) else None,
                partial_mean=float(np.mean([d for _, d in diffs])) if diffs else None,
                ads_lower_images=sum(d < 0 for _, d in diffs), per_image=diffs))
    curves = {}
    for arm in ARMS:
        pts = []
        for n in [INITIAL] + SPARSE:
            vals = [dict((r[0], r[1]) for r in c['series'][arm]).get(n) for c in cases]
            if all(v is not None for v in vals):
                pts.append([n, float(np.mean(vals))])
        curves[arm] = pts
    out['curves'] = curves
    return out


def build():
    base, group = protocol()
    ads_protocol = read(OUT / 'protocol.json')
    assets = base_report.Assets()
    audit = dict(mse_checks=0, max_mse_replay_error=0.)
    loader = Loader(assets, audit)
    cases = [loader.case(c) for c in base['cases']]
    agg = aggregate(cases)
    status = {c['image_id']: dict(ads_final=c['summary']['ADS'].get('final_mse') is not None,
              ads_sparse_complete=c['summary']['ADS']['complete_sparse'],
              baselines_complete={a: c['summary'][a]['complete_exact'] for a in ARMS[:-1]}) for c in cases}
    results = dict(generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), method=NAME,
        is_original_ads=False, protocol=ads_protocol, availability=read(OUT / 'model_availability.json'),
        status=status, aggregate=agg,
        cases=[{k: c[k] for k in ('key', 'image_id', 'summary', 'costs', 'acquisition_overlap_with_ads')} | dict(
            ads_history=c['paths'].get('ADS')) for c in cases], audit=audit)
    report_dir = OUT / 'report'; report_dir.mkdir(parents=True, exist_ok=True)
    base_report.dump(report_dir / 'ads250_results.json', results)
    data = dict(results=results, cases=cases, assets=assets.values, arms=ARMS, labels=LABELS, visual=VISUAL,
                sparse=SPARSE, window=list(WINDOW))
    template = Path(__file__).with_name('ads250_viewer.html').read_text()
    css = base_report.TEMPLATE.read_text().split('<style>')[1].split('</style>')[0]
    encoded = json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('<', r'<')
    out = report_dir / 'ads250_comparison.html'
    tmp = out.with_suffix('.tmp'); tmp.write_text(template.replace('__CSS__', css).replace('__DATA__', encoded)); tmp.replace(out)
    audit.update(html_sha256=base_report.sha(out), html_bytes=out.stat().st_size, assets=len(assets.values),
                 asset_pixel_roundtrips=assets.checks)
    base_report.dump(report_dir / 'report_manifest.json', audit)
    print(json.dumps(dict(report=str(out), **audit)))


if __name__ == '__main__':
    build()
