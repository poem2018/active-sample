"""ADS algorithm (Nolan et al., TMLR 2025) on the frozen CelebA-HQ256 RF prior.

This is a separately named ADAPTATION ("ADS-alg/RF-prior"), not original ADS:
the paper's CelebA128 Keras DDIM checkpoint is unpublished. Retained from the
official implementation (artifacts/ads_official, rev 64f2889):
  * one guided reverse trajectory; Tweedie estimates of all Np=16 particles
    select the next action inside the same trajectory (ddim.active_sampling);
  * DPS update next = s' x0 + n' eps - grad_z(omega * ||y - A x0(z)||_2),
    the norm taken jointly over all particles (guidance/pixel_guidance.py);
  * entropy criterion sum_i log sum_j exp(sum_{l in a}(yi_l-yj_l)^2/(2 sy^2)),
    paper Eq.10, sy=1; already-measured actions excluded;
  * 600 steps for |S|=500, window [10, n_acq+10], one action per step;
  * the returned reconstruction is the last-step Tweedie estimate.
Adapted: RF schedule (signal=s, noise=1-s) replaces the cosine DDIM schedule;
actions are the protocol's 1024 aligned 4x4 RGB blocks; the 250 frozen initial
blocks are in the mask from step 0; fixed noisy readings replace noiseless ones;
A=M D2 because the prior is native256. Clean per-budget candidates come from
deterministic completion branches (mask frozen, no further acquisition).
GT is read ONLY by the `evaluate` stage.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from .common import ROOT, sha256, sync, write_json

RUN = ROOT / 'runs/celeba128_ads_blocks_v1'
OUT = Path(os.environ.get('ADS250_OUT', RUN / 'extensions/ads250_v1'))
GROUP_KEY = 'b500_i250'
NAME = 'ADS-alg/RF-prior'
K = 16
STEPS = 600
INITIAL, FINAL = 250, 500
N_ACQ = FINAL - INITIAL
WINDOW = (10, N_ACQ + 10)
INTERVAL = max((WINDOW[1] - WINDOW[0]) // N_ACQ, 1)
OMEGA = 1.
SIGMA_Y = 1.
NATIVE_EPS = .001
BUDGETS = [250, *range(251, 261), *range(270, 501, 10)]
ROLL = 25
SOURCES = ['experiments/ads250_rf.py', 'experiments/ads250_rf.slurm',
           'experiments/official_face_prior.py', 'experiments/ads_celeba128.py',
           'experiments/common.py']


def read(path):
    return json.loads(Path(path).read_text())


def protocol():
    base = read(RUN / 'protocol.json')
    group = next(g for g in base['groups'] if g['key'] == GROUP_KEY)
    assert group['initial'] == INITIAL and group['final'] == FINAL
    assert len(group['initial_block_ids']) == INITIAL == len(set(group['initial_block_ids']))
    return base, group


def case_dir(case):
    return OUT / 'cases' / case['key']


def seed_for(image_id):
    digest = hashlib.sha256(f'ads250_v1|image={image_id}|experiment_seed=6101'.encode()).digest()
    return int.from_bytes(digest[:8], 'little') & (2**63 - 1)


def times():
    h = (1 - NATIVE_EPS) / STEPS
    s = [NATIVE_EPS + k * h for k in range(STEPS)]
    return s, s[1:] + [1.]


def branch_step(n):
    """First step whose guidance uses exactly n blocks; selection then adds n+1."""
    return WINDOW[0] + (n - INITIAL) * INTERVAL


@contextlib.contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def save_npy(path, value):
    tmp = path.with_suffix('.tmp.npy'); np.save(tmp, value); tmp.replace(path)


class Sensor:
    """Readings are released block-by-block, only after the block is chosen."""

    def __init__(self, case):
        path = RUN / case['sensor']
        assert sha256(path) == case['input_sha256']['sensor_bank.npz']
        with np.load(path) as bank:
            self._indices = bank['indices'].copy(); self._readings = bank['readings'].copy()
            top_left = bank['block_top_left']
        b = np.arange(1024)
        assert self._indices.shape == (1024, 48) and (top_left[:, 0] == b // 32 * 4).all()
        assert (top_left[:, 1] == b % 32 * 4).all()
        self.indices = torch.as_tensor(self._indices, device='cuda')
        self.read_order = []

    def read(self, block):
        assert block not in self.read_order, f'repeat measurement {block}'
        self.read_order.append(int(block))
        return self._readings[block]


def entropy(particles128, indices, observed):
    """Paper Eq.10 per aligned block (48 RGB scalars), float64 logsumexp."""
    y = particles128.flatten(1)[:, indices].double()  # K,1024,48
    d2 = (y[:, None] - y[None]).square().sum(-1)        # K,K,1024
    score = torch.logsumexp(d2 / (2 * SIGMA_Y**2), dim=1).sum(0)
    score[observed] = -torch.inf
    return score


class Guided:
    """Official DPS step, exact chunked gradient of the batch-joint L2 norm."""

    def __init__(self, prior, chunk):
        self.prior, self.chunk = prior, chunk
        self.forward_calls = self.vjp_calls = 0

    def step(self, z, s, snext, flat_ids, y):
        clean, noise, grad = [], [], torch.zeros_like(z)
        square = torch.zeros((), device=z.device, dtype=torch.float64)
        for a in range(0, len(z), self.chunk):
            part = z[a:a + self.chunk].detach().requires_grad_(True)
            v = self.prior(part, s)
            x0 = part + (1 - s) * v
            r = y - F.avg_pool2d(x0, 2).flatten(1)[:, flat_ids]
            half = .5 * r.square().sum()
            grad[a:a + len(part)] = torch.autograd.grad(half, part)[0]
            square += 2 * half.detach().double()
            clean.append(x0.detach()); noise.append((part - s * v).detach())
            self.forward_calls += 1; self.vjp_calls += 1
        norm = square.sqrt()
        # d||R||/dz = d(.5 sum r^2)/dz / ||R||  (official grad of omega*L2(R)).
        grad = torch.nan_to_num(OMEGA * grad / norm.float())
        clean, noise = torch.cat(clean), torch.cat(noise)
        nxt = snext * clean + (1 - snext) * noise - grad
        return nxt, clean, dict(error=float(OMEGA * norm), grad_l2=float(grad.flatten(1).norm(dim=1).mean()))


def load_prior():
    from .official_face_prior import CHECKPOINT_SHA256, load_official_prior
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    tic = time.perf_counter(); prior, metadata = load_official_prior(); load_s = time.perf_counter() - tic
    assert metadata['checkpoint_sha256'] == CHECKPOINT_SHA256
    return prior, dict(checkpoint_sha256=CHECKPOINT_SHA256, training_step=metadata['training_step'],
        parameters=metadata['parameters'], load_s=load_s, gpu=torch.cuda.get_device_name(),
        node=os.environ.get('SLURMD_NODENAME'), job_id=os.environ.get('SLURM_JOB_ID'),
        torch=str(torch.__version__), numpy=np.__version__,
        numerics='TF32 matmul + cudnn.benchmark as existing RF runs; deterministic sampler, no bitwise-across-GPU claim')


def history_tensors(sensor, history, readings):
    ids = sensor.indices[history].flatten()
    y = torch.as_tensor(np.stack(readings), device='cuda').flatten()
    return ids, y


def trajectory(prior, case, chunk, deadline):
    """Main ADS trajectory. Writes branch states and the n=500 output."""
    _, group = protocol()
    folder = case_dir(case) / 'main'; folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        return True
    sensor = Sensor(case)
    s_grid, snext_grid = times()
    roll = folder / 'resume.json'
    if roll.exists():
        meta = read(roll); z = torch.as_tensor(np.load(folder / 'resume_z.npy'), device='cuda')
        assert sha256(folder / 'resume_z.npy') == meta['z_sha256']
        start, history, log = meta['step'], meta['history'], meta['log']
        readings = [sensor.read(b) for b in history]
        costs = meta['costs']
    else:
        seed = seed_for(case['image_id'])
        z = torch.randn(K, 3, 256, 256, generator=torch.Generator().manual_seed(seed)).to('cuda')
        save_npy(folder / 'initial_noise.npy', z.cpu().numpy())
        history = list(group['initial_block_ids']); readings = [sensor.read(b) for b in history]
        start, log = 0, []
        costs = dict(guided_s=0., selection_s=0., io_s=0., forward_calls=0, vjp_calls=0,
                     peak_allocated_bytes=0, peak_reserved_bytes=0, sessions=[])
        write_json(folder / 'seed.json', dict(seed=seed, mechanism='torch.Generator(cpu).manual_seed(seed); torch.randn(16,3,256,256) float32 -> cuda',
            derivation="sha256('ads250_v1|image=<id>|experiment_seed=6101')[:8] little-endian & (2^63-1)",
            note='Own draw for this ADS adaptation; not paired with any RF baseline noise.'))
    guided = Guided(prior, chunk)
    torch.cuda.reset_peak_memory_stats()
    session = dict(job=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('SLURMD_NODENAME'), start_step=start)
    tweedie_dir = folder / 'tweedie'; tweedie_dir.mkdir(exist_ok=True)
    for k in range(start, STEPS):
        n = len(history)
        if WINDOW[0] <= k < WINDOW[1]:
            assert n == INITIAL + (k - WINDOW[0]) // INTERVAL
        if n in BUDGETS and n < FINAL and k == branch_step(n):
            tic = time.perf_counter(); bdir = case_dir(case) / 'branches' / f'n_{n:03d}'; bdir.mkdir(parents=True, exist_ok=True)
            save_npy(bdir / 'z.npy', z.cpu().numpy())
            write_json(bdir / 'branch.json', dict(n=n, step=k, history=history, z_sha256=sha256(bdir / 'z.npy')))
            costs['io_s'] += time.perf_counter() - tic
        ids, y = history_tensors(sensor, history, readings)
        sync('cuda'); tic = time.perf_counter()
        z, clean, info = guided.step(z, s_grid[k], snext_grid[k], ids, y)
        sync('cuda'); costs['guided_s'] += time.perf_counter() - tic
        record = dict(step=k, s=s_grid[k], n=n, **info, selected=None)
        acquire = WINDOW[0] <= k < WINDOW[1] and k % INTERVAL == 0 and n < FINAL
        if acquire or (n in BUDGETS and k == branch_step(n)) or k == STEPS - 1:
            pooled = F.avg_pool2d(clean, 2)
        if acquire:
            tic = time.perf_counter()
            observed = torch.zeros(1024, dtype=torch.bool, device='cuda'); observed[history] = True
            score = entropy(pooled, sensor.indices, observed)
            b = int(torch.argmax(score)); top = torch.topk(score, 5)
            assert not observed[b]
            sync('cuda'); costs['selection_s'] += time.perf_counter() - tic
            np.save(tweedie_dir / f'entropy_step_{k:03d}.npy', score.float().cpu().numpy())
            readings.append(sensor.read(b)); history.append(b)
            record.update(selected=b, entropy_max=float(top.values[0]), top5=top.indices.tolist(),
                          top5_values=top.values.tolist())
        if WINDOW[0] <= k < WINDOW[1] or k == STEPS - 1:
            np.save(tweedie_dir / f'mean_step_{k:03d}.npy', pooled.mean(0).cpu().numpy())
        if n in BUDGETS and k == branch_step(n):
            np.save(tweedie_dir / f'particles_n_{n:03d}.npy', pooled.cpu().numpy())
        log.append(record)
        if not torch.isfinite(z).all():
            raise FloatingPointError(f'nonfinite state at step {k}')
        last = k == STEPS - 1
        if last or (k + 1) % ROLL == 0 or time.time() > deadline:
            tic = time.perf_counter()
            costs['forward_calls'] += guided.forward_calls; costs['vjp_calls'] += guided.vjp_calls
            guided.forward_calls = guided.vjp_calls = 0
            costs['peak_allocated_bytes'] = max(costs['peak_allocated_bytes'], torch.cuda.max_memory_allocated())
            costs['peak_reserved_bytes'] = max(costs['peak_reserved_bytes'], torch.cuda.max_memory_reserved())
            if last:
                final = case_dir(case) / 'completions' / f'n_{FINAL:03d}'; final.mkdir(parents=True, exist_ok=True)
                save_npy(final / 'native.npy', clean.cpu().numpy()); save_npy(final / 'candidates.npy', pooled.cpu().numpy())
                write_json(final / 'state.json', dict(n=FINAL, history=history, source='main trajectory last-step Tweedie estimate'))
                write_json(final / 'complete.json', dict(status='complete', n=FINAL,
                    candidates_sha256=sha256(final / 'candidates.npy'), native_sha256=sha256(final / 'native.npy')))
            save_npy(folder / 'resume_z.npy', z.cpu().numpy())
            costs['io_s'] += time.perf_counter() - tic
            costs['sessions'].append(dict(session, end_step=k + 1))
            write_json(roll, dict(step=k + 1, history=history, log=log, costs=costs, z_sha256=sha256(folder / 'resume_z.npy')))
            session = dict(session, start_step=k + 1)
            if not last and time.time() > deadline:
                return False
    assert len(history) == FINAL == len(set(history)) and sensor.read_order == history
    assert history[:INITIAL] == group['initial_block_ids']
    write_json(folder / 'history.json', dict(history=history, acquired=history[INITIAL:], log=log))
    write_json(folder / 'complete.json', dict(status='complete', steps=STEPS, costs=costs,
        history_sha256=sha256(folder / 'history.json')))
    (folder / 'resume_z.npy').unlink()
    return True


def completion(prior, case, n, chunk, deadline):
    out = case_dir(case) / 'completions' / f'n_{n:03d}'
    if (out / 'complete.json').exists():
        return True
    branch = case_dir(case) / 'branches' / f'n_{n:03d}'
    meta = read(branch / 'branch.json'); assert sha256(branch / 'z.npy') == meta['z_sha256']
    history = meta['history']; assert len(history) == n
    sensor = Sensor(case); readings = [sensor.read(b) for b in history]
    ids, y = history_tensors(sensor, history, readings)
    s_grid, snext_grid = times(); out.mkdir(parents=True, exist_ok=True)
    roll = out / 'resume.json'
    if roll.exists():
        r = read(roll); start, costs = r['step'], r['costs']
        assert sha256(out / 'resume_z.npy') == r['z_sha256']
        z = torch.as_tensor(np.load(out / 'resume_z.npy'), device='cuda')
    else:
        start, costs = meta['step'], dict(guided_s=0., forward_calls=0, vjp_calls=0, peak_allocated_bytes=0)
        z = torch.as_tensor(np.load(branch / 'z.npy'), device='cuda')
    guided = Guided(prior, chunk); torch.cuda.reset_peak_memory_stats()
    for k in range(start, STEPS):
        sync('cuda'); tic = time.perf_counter()
        z, clean, _ = guided.step(z, s_grid[k], snext_grid[k], ids, y)
        sync('cuda'); costs['guided_s'] += time.perf_counter() - tic
        last = k == STEPS - 1
        if last or (k + 1) % 50 == 0 or time.time() > deadline:
            costs['forward_calls'] += guided.forward_calls; costs['vjp_calls'] += guided.vjp_calls
            guided.forward_calls = guided.vjp_calls = 0
            costs['peak_allocated_bytes'] = max(costs['peak_allocated_bytes'], torch.cuda.max_memory_allocated())
            if last:
                pooled = F.avg_pool2d(clean, 2)
                save_npy(out / 'candidates.npy', pooled.cpu().numpy())
                if n == INITIAL:
                    save_npy(out / 'native.npy', clean.cpu().numpy())
                write_json(out / 'state.json', dict(n=n, history=history, branch_step=meta['step'], steps_run=STEPS - meta['step'],
                    costs=costs, source='deterministic completion, mask frozen at n, no acquisition'))
                write_json(out / 'complete.json', dict(status='complete', n=n, candidates_sha256=sha256(out / 'candidates.npy')))
                for f in ('resume.json', 'resume_z.npy'):
                    (out / f).unlink(missing_ok=True)
                return True
            save_npy(out / 'resume_z.npy', z.cpu().numpy())
            write_json(roll, dict(step=k + 1, costs=costs, z_sha256=sha256(out / 'resume_z.npy')))
            if time.time() > deadline:
                return False
    return True


def evaluate_case(case):
    """GT-only stage. Uses the existing evaluator unchanged."""
    from .ads_celeba128 import evaluate
    folder = case_dir(case)
    assert sha256(RUN / case['truth']) == case['input_sha256']['truth.npy']
    truth = np.load(RUN / case['truth'])
    with np.load(RUN / case['sensor']) as bank:
        I, Y = bank['indices'], bank['readings']
    history = read(folder / 'main/history.json')['history']
    rows, previous = {}, {}
    for n in BUDGETS:
        out = folder / 'completions' / f'n_{n:03d}'
        if not (out / 'complete.json').exists():
            rows[n] = None; continue
        cand = np.load(out / 'candidates.npy'); assert sha256(out / 'candidates.npy') == read(out / 'complete.json')['candidates_sha256']
        assert read(out / 'state.json')['history'] == history[:n]
        h = history[:n]
        m = evaluate(cand, truth, I[h], Y[h], INITIAL, previous=previous.get(n - 1))
        rows[n] = m; previous[n] = cand
    tweedie = {}
    for k in range(WINDOW[0], WINDOW[1]):
        mean = np.load(folder / 'main/tweedie' / f'mean_step_{k:03d}.npy').astype(np.float64)
        tweedie[INITIAL + k - WINDOW[0]] = float(np.square(mean - truth).mean())
    write_json(folder / 'metrics.json', dict(method=NAME, image_id=case['image_id'], budgets=BUDGETS,
        metrics={str(n): v for n, v in rows.items()},
        tweedie_intermediate_mse={str(n): v for n, v in tweedie.items()},
        tweedie_note='Noisy-time Tweedie posterior-mean estimates at the selection step; diagnostic only, NOT final reconstructions.'))
    return rows


def prepare():
    base, group = protocol()
    sources = {s: sha256(ROOT / s) for s in SOURCES if (ROOT / s).exists()}
    official = ROOT / 'artifacts/ads_official'
    config = dict(name=NAME, status='locked', created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        is_original_ads=False,
        user_confirmation='User selected "ADS-alg + frozen RF", sparse completions, omega=1 a priori (2026-10-18).',
        base_protocol_sha256=sha256(RUN / 'protocol.json'), group=GROUP_KEY,
        image_ids=[c['image_id'] for c in base['cases']], K=K, steps=STEPS, window=list(WINDOW),
        interval=INTERVAL, acquisitions=N_ACQ, omega=OMEGA, sigma_y=SIGMA_Y, native_eps=NATIVE_EPS,
        budgets_evaluated=BUDGETS, rolling_checkpoint_steps=ROLL,
        operator='A = M D2 (2x2 average pool native256 -> task128, then 48 RGB scalars per aligned 4x4 block)',
        scale='[-1,1] for prior, readings and metrics; no rescaling needed. sigma=0.05 sensor noise is in the fixed readings; DPS uses omega*||R||_2 without sigma (official form).',
        prior='Frozen official CelebA-HQ256 pixel RF, checkpoint sha256 890198b7...; same prior as UU/UW/WU/WW/SCRATCH/RANDOM',
        official_reference=dict(repository='https://github.com/active-diffusion-subsampling/ads',
            revision='64f2889b192cee2280ec7bf1405752b81ec02bc1',
            files={f: sha256(official / f) for f in ('ActiveSampler.py', 'ddim.py', 'guidance/pixel_guidance.py',
                   'guidance/utils.py', 'inference_active_sampler.py', 'measurements.py')}),
        faithful=['Single guided trajectory with in-trajectory acquisition from per-particle Tweedie estimates (ddim.active_sampling)',
            'Deterministic remix next=s\'*x0+n\'*eps minus gradient of omega*L2(y-A x0), L2 over all particles jointly',
            'Entropy criterion paper Eq.10 with sigma_y=1, Np=16, observed actions excluded, argmax',
            'Paper CelebA schedule: 600 steps for |S|=500; window [10, n_acq+10]; interval=(stop-start)//n_acq',
            'Reconstruction = mean of last-step Tweedie estimates (paper reports posterior mean)'],
        adapted=['Prior: frozen RF CelebA-HQ256 instead of unpublished CelebA128 Keras DDIM; schedule signal=s, noise=1-s',
            'A=M D2 (native256 prior) instead of A=M at 128',
            'Actions: 1024 aligned non-overlapping 4x4 RGB blocks (48 scalars), not free box placement',
            '250 frozen initial blocks in the mask from step 0; window length = 250 acquisitions',
            'Fixed position-indexed noisy readings (sigma=.05) instead of noiseless official measurements',
            'logsumexp in float64 (mathematically identical to official log(sum(exp)))',
            'Exact chunked computation of the batch-joint norm gradient',
            'Per-budget clean candidates from deterministic completion branches (extra cost, not part of ADS)',
            'omega=1 from official image-domain configs (CelebA omega unpublished); not tuned'],
        not_used=['Old-candidate proximity guidance', 'recursive weights', 'ESS tempering', 'SMC corrections', 'stochastic recombination'],
        gt_isolation='trajectory/completion stages never open truth.npy; readings released per block via Sensor.read after selection',
        sources=sources, python=sys.version, torch=str(torch.__version__), numpy=np.__version__,
        environment=str(ROOT / 'envs/ads250'))
    if (OUT / 'protocol.json').exists():
        old = read(OUT / 'protocol.json')
        keep = {k: old[k] for k in ('created_utc',)}
        changed = {k for k in config if k not in ('created_utc', 'sources') and old.get(k) != config[k]}
        if changed:
            raise RuntimeError(f'Locked protocol differs in {sorted(changed)}')
        config.update(keep)
    write_json(OUT / 'protocol.json', config)
    print(json.dumps(dict(protocol=str(OUT / 'protocol.json'), sources=sources)))


def guard_truth():
    original = np.load
    def guarded(path, *a, **k):
        assert not str(path).endswith('truth.npy'), 'GT access outside evaluation'
        return original(path, *a, **k)
    np.load = guarded


def worker(case_indices, chunk, minutes, budgets=None):
    base, _ = protocol()
    deadline = time.time() + 60 * minutes
    guard_truth()
    prior, env = load_prior()
    write_json(OUT / 'logs' / f'env_{os.environ.get("SLURM_JOB_ID", "local")}_{int(time.time())}.json', env)
    for i in case_indices:
        case = base['cases'][i]
        # One controller per image; a second one fails immediately instead of taking over.
        with exclusive(case_dir(case) / '.controller.lock'):
            if not trajectory(prior, case, chunk, deadline):
                return False
            for n in budgets or BUDGETS:
                if n == FINAL:
                    continue
                if time.time() > deadline:
                    return False
                if not completion(prior, case, n, chunk, deadline):
                    return False
                print(json.dumps(dict(case=case['key'], completed=n, t=time.time())), flush=True)
    return True


def validate(chunk):
    """Implementation checks on case 0 in a scratch OUT: scale, init, first picks, speed."""
    assert 'ADS250_OUT' in os.environ, 'validation must not write into the run directory'
    base, group = protocol(); case = base['cases'][0]
    guard_truth()
    prior, env = load_prior()
    report = dict(env=env, case=case['key'])
    z = torch.randn(4, 3, 256, 256, generator=torch.Generator().manual_seed(1)).to('cuda')
    with torch.no_grad():
        for s in (.001, .5, .9):
            v = prior(z, s); x0 = z + (1 - s) * v
            report[f'unguided_x0_s{s}'] = dict(mean=float(x0.mean()), std=float(x0.std()), min=float(x0.min()), max=float(x0.max()))
    for c in sorted({chunk, 8, 4}, reverse=True):
        g = Guided(prior, c); zz = torch.randn(K, 3, 256, 256, device='cuda')
        sensor = Sensor(case); h = group['initial_block_ids']
        ids, y = history_tensors(sensor, h, [sensor._readings[b] for b in h])
        torch.cuda.reset_peak_memory_stats()
        try:
            g.step(zz, .5, .5 + 1 / STEPS, ids, y); sync('cuda'); tic = time.perf_counter()
            for _ in range(3):
                g.step(zz, .5, .5 + 1 / STEPS, ids, y)
            sync('cuda')
            report[f'chunk_{c}'] = dict(step_s=(time.perf_counter() - tic) / 3, peak_gb=torch.cuda.max_memory_allocated() / 2**30)
        except torch.OutOfMemoryError:
            report[f'chunk_{c}'] = 'oom'
        torch.cuda.empty_cache()
    # Short trajectory replica (30 steps, 20 acquisitions) with independent Eq.10 recomputation.
    s_grid, snext_grid = times()
    sensor = Sensor(case); history = list(group['initial_block_ids']); readings = [sensor.read(b) for b in history]
    z = torch.randn(K, 3, 256, 256, generator=torch.Generator().manual_seed(seed_for(case['image_id']))).to('cuda')
    g = Guided(prior, chunk); picks = []; errs = []
    for k in range(30):
        ids, y = history_tensors(sensor, history, readings)
        z, clean, info = g.step(z, s_grid[k], snext_grid[k], ids, y); errs.append(info)
        if WINDOW[0] <= k < WINDOW[1]:
            pooled = F.avg_pool2d(clean, 2)
            observed = torch.zeros(1024, dtype=torch.bool, device='cuda'); observed[history] = True
            score = entropy(pooled, sensor.indices, observed); b = int(torch.argmax(score))
            # Independent recomputation of Eq.10 for this pick (float64 numpy).
            yy = pooled.flatten(1)[:, sensor.indices[b]].double().cpu().numpy()
            d2 = np.square(yy[:, None] - yy[None]).sum(-1) / 2
            ref = float(np.log(np.exp(d2 - d2.max(1, keepdims=True)).sum(1)).sum() + d2.max(1).sum())
            picks.append(dict(step=k, block=b, entropy=float(score[b]), numpy_recompute=ref,
                              observed_rmse=float((pooled.flatten(1)[:, ids] - y).square().mean().sqrt())))
            readings.append(sensor.read(b)); history.append(b)
    assert len(set(history)) == len(history) and sensor.read_order == history
    report.update(first_picks=picks, guidance=errs, pooled_range=[float(pooled.min()), float(pooled.max())],
                  unique_history=len(set(history)), new_in_initial=any(p['block'] in group['initial_block_ids'] for p in picks))
    try:
        np.load(RUN / case['truth']); report['gt_guard'] = 'FAILED'
    except AssertionError:
        report['gt_guard'] = 'blocked'
    write_json(OUT / 'validation.json', report)
    print(json.dumps(report, indent=1)[:4000])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'worker', 'evaluate', 'validate'])
    p.add_argument('--cases', default='0')
    p.add_argument('--chunk', type=int, default=16)
    p.add_argument('--minutes', type=float, default=105)
    p.add_argument('--budgets', default=None)
    a = p.parse_args()
    cases = [int(x) for x in a.cases.split(',')]
    if a.command == 'prepare':
        prepare()
    elif a.command == 'validate':
        validate(a.chunk)
    elif a.command == 'worker':
        budgets = [int(x) for x in a.budgets.split(',')] if a.budgets else None
        done = worker(cases, a.chunk, a.minutes, budgets)
        print(json.dumps(dict(done=done)), flush=True)
    else:
        base, _ = protocol()
        for i in cases:
            rows = evaluate_case(base['cases'][i])
            print(json.dumps({n: (None if r is None else round(r['reconstruction_mse'], 6)) for n, r in rows.items()}))


if __name__ == '__main__':
    main()
