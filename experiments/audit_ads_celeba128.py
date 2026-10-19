"""Stream and independently audit committed CelebA128 experiment outputs.

Usage: python -m experiments.audit_ads_celeba128 [--partial]
Full success requires all 15 initializations and all 6,000 updates in 45
trajectories. This tool never loads the prior or generates candidates.
"""
import argparse
import io
import json
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.special import logsumexp
from skimage.metrics import structural_similarity

from .common import ROOT, sha256, write_json


RUN = ROOT / 'runs/celeba128_ads_blocks_v1'
POLICIES = ('U', 'W', 'SCRATCH')
K = 16
UNIFORM = np.full(K, 1 / K)
SSIM = dict(data_range=1., channel_axis=0, gaussian_weights=True, sigma=1.5,
            use_sample_covariance=False, win_size=11, K1=.01, K2=.03)


def read(path):
    return json.loads(Path(path).read_text())


def normalized(logits, beta):
    scaled = beta * (np.asarray(logits, np.float64) - np.max(logits))
    return scaled - logsumexp(scaled)


def ess(logw):
    return float(np.exp(-logsumexp(2 * logw)))


def reweight(previous, measurement, proximity):
    scores = previous - measurement - proximity
    raw = normalized(scores, 1.)
    if ess(raw) >= 8:
        return raw, 1., ess(raw)
    low, high = 0., 1.
    for _ in range(90):
        mid = (low + high) / 2
        if mid == low or mid == high:
            break
        if ess(normalized(scores, mid)) >= 8:
            low = mid
        else:
            high = mid
    return normalized(scores, low), low, ess(raw)


def expected_seeds(image_id, seed, initial, count):
    base = 610100000000 + image_id * 100003 + seed * 1009 + initial * 10000019 + count * 101
    return dict(gaussian=base, recombination=base + 3000000019,
                sensor=2026101703 + image_id * 1009 + seed * 101)


def pool(native):
    return native.reshape(K, 3, 128, 2, 128, 2).mean(axis=(3, 5))


def read_native_snapshot(path, successor, *, permanent, expected_count,
                         partial, retry_events, timeout_s=30., interval_s=1.):
    """Allow bounded NFS visibility lag, never an unproved missing checkpoint."""
    started = time.monotonic()
    event = None
    try:
        while True:
            try:
                # An opened file remains readable if its directory entry is retired.
                with path.open('rb') as stream:
                    raw = stream.read()
            except FileNotFoundError:
                assert not permanent, ('missing permanent native checkpoint', str(path))
                try:
                    committed = read(successor)
                except FileNotFoundError:
                    if event is None:
                        event = dict(native_path=str(path), successor=str(successor),
                                     missing_checks=0, wait_s=0., resolution='unresolved')
                        retry_events.append(event)
                    event['missing_checks'] += 1
                    elapsed = time.monotonic() - started
                    assert partial and elapsed < timeout_s, (
                        'latest rolling native missing without committed successor',
                        str(path), str(successor), elapsed)
                    time.sleep(min(interval_s, timeout_s - elapsed))
                    continue
                assert committed.get('status') == 'complete' and committed.get('count') == expected_count, (
                    'invalid successor commit', str(successor), committed)
                raw, resolution = None, 'successor_verified'
            else:
                resolution = 'native_opened'
            if event is not None:
                event.update(wait_s=time.monotonic() - started, resolution=resolution)
            return raw
    except BaseException:
        if event is not None:
            event.update(wait_s=time.monotonic() - started, resolution='failed')
        raise


class Auditor:
    def __init__(self, partial):
        self.partial = partial
        self.protocol = read(RUN / 'protocol.json')
        self.digest = sha256(RUN / 'protocol.json')
        self.start = time.perf_counter()
        self.audit_id = str(time.time_ns())
        self.where = 'protocol'
        self.result = dict(status='running', mode='partial' if partial else 'full',
            protocol_sha256=self.digest, audit_source_sha256=sha256(Path(__file__)),
            initializations=0, committed_updates=0,
            completed_trajectories=0, expected_initializations=15,
            expected_updates=6000, expected_trajectories=45, checked_hashes=0,
            metric_comparisons=0, acquisition_replays=0, score_replays=0,
            native_checkpoints=0, native_proximity_energy_checks=0,
            native_measurement_energy_checks=0, retired_native_skips=0,
            native_visibility_retries=[],
            endpoint_quality_checks=0,
            max_errors={}, trajectories=[], quality=[], missing_initializations=[],
            failures=[], no_generation_or_gpu_used=True,
            note='Native rolling files are deliberately retired after successor commit; permanent checkpoints and available latest rolling states are checked. Partial audits count only consecutive committed rounds.')

    def close(self):
        self.result['wall_s'] = time.perf_counter() - self.start
        self.result['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        # Keep every attempt, including failures, when a later audit replaces
        # the concise current-status file.
        write_json(RUN / 'report/audit_history' / f'numerical_{self.audit_id}.json', self.result)
        write_json(RUN / 'report/numerical_audit.json', self.result)

    def compare(self, name, actual, expected, atol=1e-11, rtol=1e-11):
        a, b = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
        assert a.shape == b.shape, (self.where, name, a.shape, b.shape)
        assert np.isfinite(a).all() and np.isfinite(b).all(), (self.where, name, 'nonfinite')
        error = float(np.max(np.abs(a - b))) if a.size else 0.
        self.result['max_errors'][name] = max(self.result['max_errors'].get(name, 0.), error)
        assert np.allclose(a, b, atol=atol, rtol=rtol), (self.where, name, error, atol, rtol)
        self.result['metric_comparisons'] += 1

    def check_hash(self, path, expected):
        assert sha256(path) == expected, (self.where, 'hash mismatch', str(path))
        self.result['checked_hashes'] += 1

    def load_round(self, folder, count):
        self.where = str(folder.relative_to(RUN))
        marker = read(folder / 'complete.json')
        assert marker['status'] == 'complete' and marker['count'] == count, self.where
        for name in ('state', 'metrics', 'candidates'):
            suffix = '.npy' if name == 'candidates' else '.json'
            self.check_hash(folder / (name + suffix), marker[name + '_sha256'])
        state, metrics = read(folder / 'state.json'), read(folder / 'metrics.json')
        x = np.load(folder / 'candidates.npy', allow_pickle=False)
        assert x.shape == (K, 3, 128, 128) and np.isfinite(x).all(), self.where
        permanent = bool(marker['native_retained'])
        path = folder / ('native.npy' if permanent else 'resume_native.npy')
        native = None
        successor = folder.parent / f'round_{count + 1:03d}' / 'complete.json'
        raw = read_native_snapshot(path, successor, permanent=permanent,
            expected_count=count + 1, partial=self.partial,
            retry_events=self.result['native_visibility_retries'])
        if raw is None:
            self.result['retired_native_skips'] += 1
        else:
            import hashlib
            assert hashlib.sha256(raw).hexdigest() == marker['native_sha256'], (self.where, 'native hash mismatch')
            self.result['checked_hashes'] += 1
            native = np.load(io.BytesIO(raw), allow_pickle=False)
            assert native.shape == (K, 3, 256, 256) and np.isfinite(native).all(), self.where
            np.testing.assert_array_equal(pool(native), x)
            self.result['native_checkpoints'] += 1
        return state, metrics, x, native

    def metrics(self, x, previous, truth, indices, readings, initial_count,
                state, metrics, endpoint=False):
        x = x.astype(np.float64)
        flat = x.reshape(K, -1)
        gt = truth.astype(np.float64)
        ids, y = indices.ravel(), readings.astype(np.float64).ravel()
        init_ids, init_y = indices[:initial_count].ravel(), readings[:initial_count].astype(np.float64).ravel()
        unseen = np.ones(gt.size, bool); unseen[ids] = False
        logw = np.asarray(state['log_weights'], np.float64)
        assert logw.shape == (K,) and np.isfinite(logw).all(), self.where
        self.compare('logweight_normalization', logsumexp(logw), 0., atol=1e-10)
        def rmse(z):
            return float(np.sqrt(np.square(z).mean()))
        for key, w in [('uniform', UNIFORM), ('weighted', np.exp(logw))]:
            saved = metrics[key]
            mean = np.einsum('i,ichw->chw', w, x)
            mf = mean.ravel(); error = mf - gt.ravel()
            mse = float(np.square(error).mean())
            actual = dict(reconstruction_mse=mse,
                unobserved_mse=float(np.square(error[unseen]).mean()),
                observed_mean_rmse=rmse(mf[ids] - y),
                observed_candidate_rmse=rmse(flat[:, ids] - y),
                initial_history_mean_rmse=rmse(mf[init_ids] - init_y),
                initial_history_rmse=rmse(flat[:, init_ids] - init_y),
                weighted_candidate_variance=float(np.sum(w * np.square(flat - mf).mean(1))),
                ess=float(1 / np.square(w).sum()), maximum_weight=float(w.max()))
            if previous is not None:
                prior = previous.astype(np.float64).reshape(K, -1)
                old_ids, old_y = indices[:-1].ravel(), readings[:-1].astype(np.float64).ravel()
                new_ids, new_y = indices[-1].ravel(), readings[-1].astype(np.float64).ravel()
                actual.update(old_rmse=rmse(flat[:, old_ids] - old_y),
                    old_mean_rmse=rmse(mf[old_ids] - old_y),
                    new_before_rmse=rmse(prior[:, new_ids] - new_y),
                    new_after_rmse=rmse(flat[:, new_ids] - new_y),
                    new_mean_rmse=rmse(mf[new_ids] - new_y),
                    candidate_change_rmse=rmse(flat - prior))
            for name, value in actual.items():
                self.compare(name, saved[name], value)
            if endpoint:
                psnr = float(10 * np.log10(4 / mse))
                ssim = float(structural_similarity((gt + 1) / 2, (mean + 1) / 2, **SSIM))
                self.compare('psnr', saved['psnr'], psnr)
                self.compare('ssim', saved['ssim'], ssim)
                self.result['endpoint_quality_checks'] += 1
                self.result['quality'].append(dict(checkpoint=self.where, view=key,
                    mse=mse, psnr=psnr, ssim=ssim))

    def acquisition(self, previous, previous_state, state, policy, indices):
        prev_logw = np.asarray(previous_state['log_weights'], np.float64)
        self.compare('previous_log_weights', state['previous_log_weights'], prev_logw, atol=0., rtol=0.)
        w = np.exp(prev_logw) if policy == 'W' else UNIFORM
        self.compare('acquisition_weights', state['acquisition_weights'], w, atol=0., rtol=0.)
        x = previous.astype(np.float64)
        mean = np.einsum('i,ichw->chw', w, x)
        variance = np.einsum('i,ichw->chw', w, np.square(x - mean))
        scores = variance.ravel()[indices].sum(1)
        self.compare('acquisition_scores', state['acquisition_scores'], scores, atol=1e-12)
        seen = np.zeros(1024, bool); seen[previous_state['history']] = True
        available = np.flatnonzero(~seen)
        chosen = int(available[np.argmax(scores[available])])
        assert chosen == state['selected'] == state['history'][-1], (self.where, 'selection replay mismatch')
        self.result['acquisition_replays'] += 1

    def scoring(self, previous_state, state, native, previous_native, ids, readings, policy):
        previous = np.asarray(previous_state['log_weights'], np.float64)
        current = np.asarray(state['log_weights'], np.float64)
        if policy == 'SCRATCH':
            self.compare('scratch_uniform_logweights', current, np.log(UNIFORM), atol=0., rtol=0.)
            assert state['score'] == {} and state['beta'] == 0., self.where
            return
        score = state['score']
        me = np.asarray(score['measurement_energy'], np.float64)
        pe = np.asarray(score['proximity_energy'], np.float64)
        assert me.shape == pe.shape == (K,) and np.isfinite(me).all() and np.isfinite(pe).all(), self.where
        assert (me >= 0).all() and (pe >= 0).all(), self.where
        logw, beta, raw_ess = reweight(previous, me, pe)
        self.compare('recursive_logweights', current, logw, atol=1e-9, rtol=1e-10)
        self.compare('recursive_beta', state['beta'], beta, atol=1e-14, rtol=1e-10)
        self.compare('ess_untempered', score['ess_untempered'], raw_ess)
        self.compare('scoring_ess', state['scoring_ess'], ess(current))
        self.compare('scoring_maximum_weight', state['scoring_maximum_weight'], np.exp(current).max())
        assert ess(current) >= 8 - 1e-10 and 0 <= state['beta'] <= 1, self.where
        if native is not None:
            high = native.astype(np.float64)
            pred = pool(high).reshape(K, -1)[:, ids[-1]]
            exact_me = np.square(pred - readings[-1].astype(np.float64)).sum(1) / (2 * .05**2)
            self.compare('native_measurement_energy', me, exact_me, atol=1e-10, rtol=1e-10)
            self.result['native_measurement_energy_checks'] += 1
            if previous_native is not None:
                exact_pe = np.square(high - previous_native.astype(np.float64)).reshape(K, -1).sum(1) / 2
                self.compare('native_proximity_energy', pe, exact_pe, atol=1e-8, rtol=1e-11)
                self.result['native_proximity_energy_checks'] += 1
        self.result['score_replays'] += 1

    def run(self):
        protocol = self.protocol
        assert len(protocol['cases']) == 5 and len(protocol['groups']) == 3
        assert len(protocol['tasks']) == 45 and protocol['expected_updates'] == 6000
        assert [(g['initial'], g['final']) for g in protocol['groups']] == [(50, 100), (100, 200), (250, 500)]
        assert protocol['K'] == 16 and protocol['steps'] == 128 and protocol['sigma'] == .05
        assert protocol['tau_native'] == 1. and protocol['lambda_w'] == 1. and protocol['target_ess'] == 8.
        for name, digest in protocol['source_sha256'].items():
            self.check_hash(ROOT / name, digest)
        for case in protocol['cases']:
            parent = (RUN / case['truth']).parent
            for name, digest in case['input_sha256'].items():
                self.check_hash(parent / name, digest)
        for group in protocol['groups']:
            initial, final = group['initial'], group['final']
            for case in protocol['cases']:
                base = RUN / 'groups' / group['key'] / 'cases' / case['key']
                init_folder = base / 'initialization' / f'round_{initial:03d}'
                if not (init_folder / 'complete.json').exists():
                    self.result['missing_initializations'].append(str(init_folder.relative_to(RUN)))
                    continue
                init_state, init_metric, init_x, init_native = self.load_round(init_folder, initial)
                assert init_state['history'] == group['initial_block_ids'] and init_state['policy'] == 'INITIAL'
                assert init_state['count'] == initial and init_state['selected'] is None
                assert init_state['seeds'] == expected_seeds(case['image_id'], case['seed'], initial, initial)
                assert init_state['nfe'] == 128 and init_state['guidance_blocks'] == initial
                assert init_state['guidance_lipschitz'] == 100.
                self.compare('initial_uniform_logweights', init_state['log_weights'], np.log(UNIFORM), atol=0., rtol=0.)
                truth = np.load(RUN / case['truth'], allow_pickle=False)
                with np.load(RUN / case['sensor'], allow_pickle=False) as bank:
                    indices, readings = bank['indices'], bank['readings']
                assert indices.shape == readings.shape == (1024, 48)
                assert len(np.unique(indices)) == 3 * 128**2
                history = init_state['history']
                assert len(history) == len(set(history)) == initial
                self.metrics(init_x, None, truth, indices[history], readings[history], initial,
                             init_state, init_metric, endpoint=True)
                self.result['initializations'] += 1
                for policy in POLICIES:
                    previous_state, previous, previous_native = init_state, init_x, init_native
                    folder = base / policy
                    last = initial
                    for count in range(initial + 1, final + 1):
                        out = folder / f'round_{count:03d}'
                        if not (out / 'complete.json').exists():
                            break
                        state, metrics, x, native = self.load_round(out, count)
                        history = state['history']
                        assert state['count'] == count and state['policy'] == policy, self.where
                        assert len(history) == len(set(history)) == count, self.where
                        assert min(history) >= 0 and max(history) < 1024, self.where
                        assert history[:initial] == group['initial_block_ids'], self.where
                        assert history[:-1] == previous_state['history'], self.where
                        assert state['nfe'] == 128 and state['guidance_blocks'] == count, self.where
                        assert state['candidate_model_evaluations'] == 16 * 128, self.where
                        assert state['seeds'] == expected_seeds(case['image_id'], case['seed'], initial, count), self.where
                        expected_lipschitz = 100. if policy == 'SCRATCH' else 101.
                        assert state['guidance_lipschitz'] == expected_lipschitz, self.where
                        assert state['guidance_step'] == .5 / expected_lipschitz, self.where
                        self.acquisition(previous, previous_state, state, policy, indices)
                        self.scoring(previous_state, state, native, previous_native,
                                     indices[history], readings[history], policy)
                        self.metrics(x, previous, truth, indices[history], readings[history],
                                     initial, state, metrics, endpoint=count == final)
                        self.result['committed_updates'] += 1
                        previous_state, previous, previous_native = state, x, native
                        last = count
                    complete = False
                    completion = folder / 'completion.json'
                    if completion.exists():
                        info = read(completion)
                        assert info['status'] == 'complete' and info['policy'] == policy
                        assert info['initial'] == initial and info['final'] == final
                        assert info['update_nfe'] == (final - initial) * 128
                        assert info['protocol_sha256'] == self.digest
                        if last == final:
                            assert info['history'] == previous_state['history']
                            complete = True
                            self.result['completed_trajectories'] += 1
                        else:
                            # In partial mode the worker may commit its final
                            # round after our consecutive snapshot stopped.
                            assert self.partial, (str(folder), 'completion marker beyond audited consecutive history')
                    self.result['trajectories'].append(dict(group=group['key'], case=case['key'],
                        policy=policy, initial=initial, final=final, last_audited=last,
                        updates=last - initial, completed=complete))
                    print(json.dumps(dict(group=group['key'], image_id=case['image_id'],
                        policy=policy, last=last, audited_updates=self.result['committed_updates'])), flush=True)
        assert sha256(RUN / 'protocol.json') == self.digest, 'Protocol changed during audit'
        full = (self.result['initializations'] == 15 and self.result['completed_trajectories'] == 45
                and self.result['committed_updates'] == 6000)
        self.result['all_expected_outputs_complete'] = full
        self.result['status'] = 'passed' if full else 'partial_passed' if self.partial else 'incomplete'
        self.close()
        return 0 if full or self.partial else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial', action='store_true', help='Audit available committed rounds without requiring completion')
    args = parser.parse_args()
    audit = Auditor(args.partial)
    try:
        code = audit.run()
    except Exception:
        audit.result['status'] = 'failed'
        audit.result['failures'].append(dict(where=audit.where, traceback=traceback.format_exc()))
        audit.close()
        raise
    print(json.dumps({k: audit.result[k] for k in ('status', 'initializations',
        'committed_updates', 'completed_trajectories', 'wall_s')}, indent=2), flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
