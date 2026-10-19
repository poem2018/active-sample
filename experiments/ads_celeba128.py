"""Three independently initialized CelebA128 active-block comparisons.

The task is y=M x+noise on an ordinary CelebA128 image. The existing frozen
native256 RF induces x=D2 g; the native guidance operator is therefore M D2.
Only new files are written. Old studies and frozen sampler sources are retained.
"""
import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch
from skimage.metrics import structural_similarity

from .active_blocks import block_grid, select_block
from .adaptive_candidate_weights import recursive_update
from .common import ROOT, sha256, sync, write_json
from .official_face_prior import CHECKPOINT_SHA256, load_official_prior
from .official_face_reuse import downsample
from .sequential_flowdps import CandidateEnergy, recursive_score, sample

RUN = ROOT / 'runs/celeba128_ads_blocks_v1'
DATA = ROOT / 'data/celeba128_ads_v1'
K = 16
UNIFORM = np.full(K, 1 / K)
POLICIES = ('U', 'W', 'SCRATCH')
SSIM_ARGS = dict(data_range=1., channel_axis=0, gaussian_weights=True,
                 sigma=1.5, use_sample_covariance=False, win_size=11,
                 K1=.01, K2=.03)
SOURCES = ['experiments/ads_celeba128.py', 'experiments/ads_celeba128.slurm', 'experiments/sequential_flowdps.py',
           'experiments/flowdps_pixel.py', 'experiments/official_face_reuse.py',
           'experiments/official_face_prior.py', 'experiments/active_blocks.py',
           'experiments/adaptive_candidate_weights.py', 'experiments/common.py',
           'experiments/prepare_ads_celeba128.py']


def read(path):
    return json.loads(Path(path).read_text())


def seeds_for(image_id, seed, initial, count):
    # No policy-dependent seed: Gaussian and all recombination draws are paired.
    base = 610100000000 + image_id * 100003 + seed * 1009 + initial * 10000019 + count * 101
    return dict(gaussian=base, recombination=base + 3000000019,
                sensor=2026101703 + image_id * 1009 + seed * 101)


def spatial_order():
    """Nested maximin layouts; uses only grid geometry, never image content."""
    coordinates, _ = block_grid(128, 4)
    priority = np.random.default_rng(2026101702).permutation(len(coordinates))
    distance = np.full(len(coordinates), np.inf)
    seen = np.zeros(len(coordinates), bool)
    chosen = 15 * 32 + 15
    order = []
    for _ in range(250):
        order.append(int(chosen)); seen[chosen] = True
        distance = np.minimum(distance, np.square(coordinates - coordinates[chosen]).sum(1))
        eligible = priority[~seen[priority]]
        if len(eligible):
            chosen = int(eligible[np.argmax(distance[eligible])])
    return order


def pool(x):
    x = np.asarray(x)
    return x.reshape(*x.shape[:-2], 128, 2, 128, 2).mean(axis=(-3, -1))


def case_root(group, case):
    return RUN / 'groups' / group['key'] / 'cases' / case['key']


def round_path(folder, count):
    return folder / f'round_{count:03d}'


def native_path(folder):
    return folder / ('native.npy' if (folder / 'native.npy').exists() else 'resume_native.npy')


@contextlib.contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def evaluate(candidates, truth, indices, readings, initial_count, previous=None,
             weights=UNIFORM):
    """Unclipped task128 metrics. Only this evaluator receives GT."""
    x = np.asarray(candidates, np.float64)
    gt = np.asarray(truth, np.float64)
    w = np.asarray(weights, np.float64)
    assert x.shape == (K, 3, 128, 128) and gt.shape == (3, 128, 128)
    assert np.isfinite(x).all() and np.isclose(w.sum(), 1.)
    flat = x.reshape(K, -1)
    mean = np.einsum('i,ichw->chw', w, x)
    mf = mean.ravel()
    ix = np.asarray(indices).ravel(); y = np.asarray(readings, np.float64).ravel()
    unseen = np.ones(gt.size, bool); unseen[ix] = False
    rmse = lambda z: float(np.sqrt(np.square(z).mean()))
    err = mf - gt.ravel()
    mse = float(np.square(err).mean())
    gram = flat @ flat.T / flat.shape[1]
    pair_d2 = np.maximum(0., np.diag(gram)[:, None] + np.diag(gram)[None, :] - 2 * gram)
    init_ix = indices[:initial_count].ravel()
    init_y = np.asarray(readings[:initial_count], np.float64).ravel()
    variance = float(np.sum(w * np.square(flat - mf).mean(1)))
    result = dict(reconstruction_mse=mse, psnr=float(10 * np.log10(4 / mse)),
        ssim=float(structural_similarity((gt + 1) / 2, (mean + 1) / 2, **SSIM_ARGS)),
        unobserved_mse=float(np.square(err[unseen]).mean()),
        observed_mean_rmse=rmse(mf[ix] - y), observed_candidate_rmse=rmse(flat[:, ix] - y),
        initial_history_mean_rmse=rmse(mf[init_ix] - init_y),
        initial_history_rmse=rmse(flat[:, init_ix] - init_y),
        pairwise_diversity_rmse=float(np.sqrt(pair_d2[np.tril_indices(K, -1)]).mean()),
        raw_candidate_variance=float(np.square(flat - flat.mean(0)).mean()),
        weighted_candidate_variance=variance, ess=float(1 / np.square(w).sum()),
        maximum_weight=float(w.max()), candidate_gt_mse=float(np.square(flat - gt.ravel()).mean()),
        per_candidate_mse=np.square(flat - gt.ravel()).mean(1).tolist(),
        raw_min=float(x.min()), raw_max=float(x.max()),
        outside_model_range_fraction=float((np.abs(x) > 1).mean()),
        old_rmse=None, old_mean_rmse=None, new_before_rmse=None,
        new_after_rmse=None, new_mean_rmse=None, candidate_change_rmse=None)
    if previous is not None:
        old_ix, new_ix = indices[:-1].ravel(), indices[-1].ravel()
        old_y, new_y = np.asarray(readings[:-1], np.float64).ravel(), np.asarray(readings[-1], np.float64).ravel()
        prev = np.asarray(previous, np.float64).reshape(K, -1)
        result.update(old_rmse=rmse(flat[:, old_ix] - old_y), old_mean_rmse=rmse(mf[old_ix] - old_y),
            new_before_rmse=rmse(prev[:, new_ix] - new_y), new_after_rmse=rmse(flat[:, new_ix] - new_y),
            new_mean_rmse=rmse(mf[new_ix] - new_y), candidate_change_rmse=rmse(flat - prev))
    return result


def prepare():
    if (RUN / 'protocol.json').exists():
        raise FileExistsError('Protocol already locked; use initialize/infer/report to resume.')
    dm = read(DATA / 'manifest.json')
    assert len(dm['cases']) == 5 and len(set(dm['image_ids'])) == 5
    order = spatial_order()
    groups = [dict(key=f'b{end}_i{start:03d}', initial=start, final=end,
                   initial_block_ids=order[:start], initial_coverage=start / 1024,
                   final_coverage=end / 1024) for start, end in [(50, 100), (100, 200), (250, 500)]]
    coords, indices = block_grid(128, 4)
    cases = []
    for entry in dm['cases']:
        iid = int(entry['image_id']); seed = 6101
        key = f'image_{iid}_seed_{seed}'
        inp = RUN / 'inputs' / key; inp.mkdir(parents=True, exist_ok=True)
        source = ROOT / entry['truth_path']
        truth = np.load(source)
        assert truth.shape == (3, 128, 128) and truth.dtype == np.float32
        shutil.copyfile(source, inp / 'truth.npy')
        noise_seed = seeds_for(iid, seed, 50, 50)['sensor']
        noise = np.random.default_rng(noise_seed).normal(0, .05, indices.shape).astype(np.float32)
        np.savez_compressed(inp / 'sensor_bank.npz', indices=indices,
            readings=truth.ravel()[indices] + noise, sensor_noise=noise, block_top_left=coords)
        cases.append(dict(key=key, image_id=iid, seed=seed,
            truth=str((inp / 'truth.npy').relative_to(RUN)),
            sensor=str((inp / 'sensor_bank.npz').relative_to(RUN)), sensor_seed=noise_seed,
            input_sha256={p.name: sha256(p) for p in inp.iterdir() if p.is_file()}))
    initializations = [dict(case_key=c['key'], group_key=g['key']) for g in groups for c in cases]
    tasks = []
    for index in np.random.default_rng(2026101704).permutation(len(initializations)):
        item = initializations[int(index)]
        for policy in np.random.default_rng(2026101705 + int(index)).permutation(POLICIES):
            tasks.append(dict(rank=len(tasks), **item, policy=str(policy)))
    updates = 5 * 3 * sum(g['final'] - g['initial'] for g in groups)
    protocol = dict(status='locked', created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        groups=groups, cases=cases, initializations=initializations, tasks=tasks,
        data_manifest_sha256=sha256(DATA / 'manifest.json'), dataset='ordinary CelebA official test partition',
        dataset_metadata=dm, task_resolution=128, native_resolution=256, block_size=4,
        task_operator='y=M x+eta; direct 4x4 RGB pixels of task128 image, no complete low-resolution base',
        prior_adapter='x=D2 g, frozen official native256 RF; native guidance A=M D2, D2=2x2 average pooling',
        prior_training_independence='Unknown released prior training-image IDs; no held-out-from-prior claim',
        geometry='1024 aligned nonoverlapping blocks; initial50/100/250 are nested spatial maximin layouts',
        initial_geometry_seed=2026101702, sigma=.05, tau_native=1., lambda_w=1., target_ess=8.,
        K=K, steps=128, model_scale='[-1,1]; all metrics/acquisition/guidance/weights use unclipped arrays',
        sampler='Unchanged sequential_flowdps.sample, factor2; fresh Gaussian every update, original clean3GD and stochastic recombination',
        guidance='ALL acquired measurements + corresponding old native candidate for U/W; no old candidate for SCRATCH',
        scoring='s=log(w_previous)-E_new-E_prox; w=softmax(beta*s), largest beta in[0,1] with ESS>=8; beta applies to ENTIRE score',
        scoring_energies='E_new=sum newest48scalar residuals squared/(2*.05^2); E_prox=native256 summed squared old-new/(2*1^2)',
        roles='U/W choose maximum uniform/weighted candidate disagreement before reading the selected block; both maintain recursive scores independently of reconstruction; UU/UW share U, WU/WW share W',
        initialization='Separate shared all-initial-measurement posterior for EACH group/image; uniform initial weights; no inherited trajectory between budgets',
        pairing='Position-indexed sensor noise shared across all groups/policies per image; Gaussian and recombination seeds shared across policies within group/count',
        restrictions='No retraining, warm starts, new-only guidance, resampling, density correction, tuning on these5 images, or rerunning paused50-image study',
        primary='MSE of UNIFORM candidate MEAN in task128. Per-image values averaged across5images; candidates/rounds not independent replicates.',
        secondary='UW/WW weighted means, PSNR/SSIM, unobserved MSE, initial/old/new residuals, diversity, ESS, runtime. Different active masks imply different unobserved regions.',
        trajectory_averaging='Each group averages updates initial+1 through final, excluding initialization; report final and complete curves',
        ssim=SSIM_ARGS, expected_updates=updates, expected_initializations=15,
        expected_nfe=(updates + 15) * 128, expected_gpu_hours=[25., 30.], max_concurrent_gpus=9,
        storage='All task128 candidates/metrics/histories; native initial/quartile/final checkpoints plus one transactional rolling state. About20-25GiB.',
        ads_differences=['Our stipulated half-budget initialization; ADS acquires during one diffusion trajectory',
            'Existing RF/FlowDPS prior and sampler, not ADS diffusion checkpoint',
            'Retained Gaussian sensor noise sigma=.05', 'Aligned nonoverlapping grid; ADS box placement may overlap',
            'Documented center-crop/bilinear128 preprocessing; published CelebA preprocessing details unavailable'],
        checkpoint_sha256=CHECKPOINT_SHA256,
        source_sha256={p: sha256(ROOT / p) for p in SOURCES},
        numpy=np.__version__, torch=str(torch.__version__))
    write_json(RUN / 'protocol.json', protocol)
    shutil.copyfile(DATA / 'manifest.json', RUN / 'data_manifest.json')
    for p in SOURCES:
        out = RUN / 'source_snapshot' / p; out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / p, out)
    print(json.dumps(dict(status='prepared', groups=groups, image_ids=dm['image_ids'],
                          updates=updates, nfe=protocol['expected_nfe'])), flush=True)


def verify_protocol():
    protocol = read(RUN / 'protocol.json')
    for name, digest in protocol['source_sha256'].items():
        assert sha256(ROOT / name) == digest, f'Frozen source changed: {name}'
    return protocol


def verify_inputs(case):
    folder = (RUN / case['truth']).parent
    for name, digest in case['input_sha256'].items():
        assert sha256(folder / name) == digest, f'Input changed: {name}'


def resolve(protocol, item):
    group = next(g for g in protocol['groups'] if g['key'] == item['group_key'])
    case = next(c for c in protocol['cases'] if c['key'] == item['case_key'])
    return group, case


def load_prior():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    tic = time.perf_counter(); prior, metadata = load_official_prior()
    load_s = time.perf_counter() - tic
    tic = time.perf_counter()
    with torch.no_grad():
        for _ in range(3):
            prior(torch.zeros(K, 3, 256, 256, device='cuda'), .5)
    sync('cuda')
    return prior, dict(metadata=metadata, load_s=load_s, warmup_s=time.perf_counter() - tic,
        warmup_nfe=3, node=os.environ.get('SLURMD_NODENAME'), job_id=os.environ.get('SLURM_JOB_ID'),
        gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), numpy=np.__version__,
        numerics='Existing TF32/cudnn.benchmark settings; paired random inputs, no bitwise-across-GPU claim')


@torch.no_grad()
def generate(prior, group, case, count, indices, readings, old):
    seeds = seeds_for(case['image_id'], case['seed'], group['initial'], count)
    epsilon = torch.randn(K, 3, 256, 256,
        generator=torch.Generator().manual_seed(seeds['gaussian'])).to('cuda')
    rng = torch.Generator(device='cuda').manual_seed(seeds['recombination'])
    ids, y = torch.as_tensor(indices, device='cuda'), torch.as_tensor(readings, device='cuda')
    parent = None if old is None else torch.as_tensor(old, device='cuda')
    calls = 0
    def counted(z, t):
        nonlocal calls
        calls += 1
        return prior(z, t)
    sync('cuda'); torch.cuda.reset_peak_memory_stats(); tic = time.perf_counter()
    result, trace, states = sample(counted, epsilon, ids, y, old=parent, sigma=.05,
        tau=1., steps=128, factor=2, generator=rng)
    sync('cuda'); seconds = time.perf_counter() - tic
    assert calls == 128
    assert np.all(trace['proximity_l2'] == 0) if old is None else np.all(trace['proximity_l2'] > 0)
    cost = dict(generation_s=seconds, nfe=calls, candidate_model_evaluations=K * calls,
        backward_evaluations=0, trace_evaluations=0,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), guidance_blocks=len(indices),
        guidance_lipschitz=100. if old is None else 101.,
        guidance_step=.5 / (100. if old is None else 101.))
    # Discard internal image snapshots, retain numerical guidance diagnostics.
    trace = {k: v for k, v in trace.items() if k not in ('states', 'clean_before', 'clean_guided')}
    trace['extra_noise_rng_states'] = states
    return result.cpu().numpy(), trace, cost


def persist(folder, count, native, task, state, metrics, diagnostics, keep_native):
    out = round_path(folder, count)
    if out.exists():
        if (out / 'complete.json').exists():
            raise FileExistsError(out)
        archive = folder / 'incomplete_attempts'; archive.mkdir(parents=True, exist_ok=True)
        shutil.move(str(out), str(archive / (out.name + '_' + str(time.time_ns()))))
    out.mkdir(parents=True)
    tic = time.perf_counter()
    np.save(out / ('native.npy' if keep_native else 'resume_native.npy'), native)
    np.save(out / 'candidates.npy', task)
    np.savez_compressed(out / 'diagnostics.npz', **diagnostics)
    state['io_s'] = time.perf_counter() - tic
    write_json(out / 'state.json', state); write_json(out / 'metrics.json', metrics)
    write_json(out / 'complete.json', dict(status='complete', count=count,
        native_retained=bool(keep_native), native_sha256=sha256(native_path(out)),
        candidates_sha256=sha256(out / 'candidates.npy'), state_sha256=sha256(out / 'state.json'),
        metrics_sha256=sha256(out / 'metrics.json')))
    # A preceding rolling state becomes disposable only AFTER this commit.
    previous = round_path(folder, count - 1) / 'resume_native.npy'
    if previous.exists():
        previous.unlink()
    write_json(folder / 'progress.json', dict(count=count, metrics=metrics, state=state))


def load_checkpoint(folder, count):
    out = round_path(folder, count); committed = read(out / 'complete.json')
    assert sha256(out / 'state.json') == committed['state_sha256']
    assert sha256(out / 'candidates.npy') == committed['candidates_sha256']
    assert sha256(native_path(out)) == committed['native_sha256']
    return read(out / 'state.json'), np.load(native_path(out)), np.load(out / 'candidates.npy')


def initialize_one(prior, group, case, execution):
    folder = case_root(group, case) / 'initialization'
    with exclusive(folder / '.worker.lock'):
        count = group['initial']; out = round_path(folder, count)
        if (out / 'complete.json').exists():
            load_checkpoint(folder, count); return
        verify_inputs(case)
        with np.load(RUN / case['sensor']) as sensor:
            history = group['initial_block_ids']
            ids, y = sensor['indices'][history], sensor['readings'][history]
        native, diagnostics, cost = generate(prior, group, case, count, ids, y, None)
        task = pool(native)
        tic = time.perf_counter()
        metrics = evaluate(task, np.load(RUN / case['truth']), ids, y, count)
        state = dict(count=count, history=history, selected=None, log_weights=np.log(UNIFORM).tolist(),
            previous_log_weights=np.log(UNIFORM).tolist(), beta=0., acquisition_weights=UNIFORM.tolist(),
            acquisition_scores=None, seeds=seeds_for(case['image_id'], case['seed'], count, count),
            evaluation_s=time.perf_counter() - tic, acquisition_s=0., scoring_s=0.,
            score={}, policy='INITIAL', **cost)
        persist(folder, count, native, task, state, dict(uniform=metrics, weighted=metrics), diagnostics, True)
        write_json(folder / 'execution.json', execution)
        print(json.dumps(dict(stage='initialization', group=group['key'], case=case['key'],
            count=count, mse=metrics['reconstruction_mse'], seconds=cost['generation_s'])), flush=True)


def initialize(limit=None):
    protocol = verify_protocol()
    prior, execution = load_prior()
    items = protocol['initializations'] if limit is None else protocol['initializations'][:limit]
    for item in items:
        if (RUN / 'PAUSED_BY_USER.json').exists():
            return
        initialize_one(prior, *resolve(protocol, item), execution)
    if limit is None:
        write_json(RUN / 'initialization_complete.json', dict(status='complete', count=len(items),
            protocol_sha256=sha256(RUN / 'protocol.json'), execution=execution))


def trajectory(rank, limit=None):
    protocol = verify_protocol(); item = protocol['tasks'][rank]
    group, case = resolve(protocol, item); policy = item['policy']
    folder = case_root(group, case) / policy
    with exclusive(folder / '.worker.lock'):
        if (folder / 'completion.json').exists():
            return
        verify_inputs(case)
        last = group['initial']
        for count in range(group['initial'] + 1, group['final'] + 1):
            if not (round_path(folder, count) / 'complete.json').exists():
                break
            last = count
        if last == group['initial']:
            state, parent, previous = load_checkpoint(case_root(group, case) / 'initialization', last)
        else:
            state, parent, previous = load_checkpoint(folder, last)
        history = state['history']; logw = np.asarray(state['log_weights'], np.float64)
        assert len(history) == len(set(history)) == last
        with np.load(RUN / case['sensor']) as data:
            sensor = {k: data[k] for k in data.files}
        truth = np.load(RUN / case['truth'])  # used ONLY by evaluate below
        prior, execution = load_prior(); started = time.perf_counter()
        write_json(folder / f'execution_{time.time_ns()}.json', execution)
        end = group['final'] if limit is None else min(group['final'], last + limit)
        keep = {group['initial'] + (group['final'] - group['initial']) * j // 4 for j in range(1, 5)}
        for count in range(last + 1, end + 1):
            if (RUN / 'PAUSED_BY_USER.json').exists():
                return
            before_logw = logw.copy()
            acq_w = np.exp(before_logw) if policy == 'W' else UNIFORM.copy()
            seen = np.zeros(1024, bool); seen[history] = True
            tic = time.perf_counter(); chosen, scores = select_block(previous, acq_w, seen)
            acquisition_s = time.perf_counter() - tic
            assert chosen not in history
            # Selection has finished before the simulator exposes the next reading.
            history = history + [chosen]
            ids, y = sensor['indices'][history], sensor['readings'][history]
            native, diagnostics, cost = generate(prior, group, case, count, ids, y,
                                                None if policy == 'SCRATCH' else parent)
            task = pool(native); score = {}; beta = 0.
            tic = time.perf_counter()
            if policy in ('U', 'W'):
                energies = recursive_score(before_logw, native, parent, ids[-1], y[-1], .05, 1., factor=2)
                score = {k: energies[k].tolist() for k in ('measurement_energy', 'proximity_energy')}
                update = recursive_update(before_logw, score['measurement_energy'], score['proximity_energy'],
                                          lambda_w=1., target_ess=8.)
                logw = update['log_weights']; beta = float(update['beta'])
                score.update(ess_untempered=update['ess_untempered'], beta_upper=update['beta_upper'])
                assert update['ess'] >= 8 - 1e-10
            scoring_s = time.perf_counter() - tic
            tic = time.perf_counter()
            primary = evaluate(task, truth, ids, y, group['initial'], previous)
            weighted = primary if policy == 'SCRATCH' else evaluate(task, truth, ids, y,
                group['initial'], previous, np.exp(logw))
            evaluation_s = time.perf_counter() - tic
            state = dict(count=count, history=history, selected=chosen, policy=policy,
                previous_log_weights=before_logw.tolist(), log_weights=logw.tolist(), beta=beta,
                acquisition_weights=acq_w.tolist(), acquisition_scores=scores.tolist(), score=score,
                scoring_ess=float(1 / np.exp(2 * logw).sum()), scoring_maximum_weight=float(np.exp(logw).max()),
                seeds=seeds_for(case['image_id'], case['seed'], group['initial'], count),
                acquisition_s=acquisition_s, scoring_s=scoring_s, evaluation_s=evaluation_s, **cost)
            persist(folder, count, native, task, state, dict(uniform=primary, weighted=weighted),
                    diagnostics, count in keep)
            parent, previous = native, task
            print(json.dumps(dict(group=group['key'], case=case['key'], policy=policy, count=count,
                mse=primary['reconstruction_mse'], psnr=primary['psnr'], ess=state['scoring_ess'],
                seconds=cost['generation_s'])), flush=True)
        if end == group['final']:
            write_json(folder / 'completion.json', dict(status='complete', group=group['key'],
                case=case['key'], policy=policy, initial=group['initial'], final=end, history=history,
                update_nfe=(end - group['initial']) * 128, invocation_wall_s=time.perf_counter() - started,
                protocol_sha256=sha256(RUN / 'protocol.json')))


def preflight():
    """Independent operator, sampler-control, geometry and weighting checks."""
    from .flowdps_pixel import sample as scratch_sample
    from scipy.special import logsumexp
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(22)
    x = torch.randn(2, 3, 16, 16, dtype=torch.float64, generator=generator, requires_grad=True)
    ids = torch.tensor([[0, 1, 10, 99], [100, 101, 120, 160]])
    y = torch.randn(ids.shape, dtype=torch.float64, generator=generator)
    old = torch.randn(x.shape, dtype=x.dtype, generator=generator)
    energy = CandidateEnergy(x, ids.ravel(), y.ravel(), .05, old, 1., 2)
    objective = ((downsample(x, 2).flatten(1)[:, ids.ravel()] - y.ravel()) ** 2).sum() / (2 * .05**2)
    objective = objective + ((x - old) ** 2).sum() / 2
    actual = torch.autograd.grad(objective, x)[0]
    grad_error = float((actual - energy.total_gradient(x)).abs().max())
    assert grad_error < 1e-10
    epsilon = x.detach()
    prior = lambda z, t: .1 * z + .02 * t
    a, _, ra = sample(prior, epsilon, ids, y, None, steps=8, factor=2,
                     generator=torch.Generator().manual_seed(123), diagnostics=False)
    b, _, rb = scratch_sample(prior, epsilon, ids, y, steps=8, factor=2,
                     generator=torch.Generator().manual_seed(123), diagnostics=False)
    sampler_error = float((a - b).abs().max())
    assert sampler_error == 0 and np.array_equal(ra, rb)
    order = spatial_order(); coordinates, indices = block_grid(128, 4)
    assert len(order) == len(set(order)) == 250
    assert len(np.unique(indices)) == 3 * 128**2
    for count in (50, 100, 250):
        assert len(np.unique(indices[order[:count]])) == count * 48
    rng = np.random.default_rng(12)
    candidates = rng.normal(0, .7, (K, 3, 128, 128)).astype(np.float32)
    truth = rng.normal(0, .7, (3, 128, 128)).astype(np.float32)
    ys = truth.ravel()[indices[order[:50]]]
    met = evaluate(candidates, truth, indices[order[:50]], ys, 50)
    exact = float(np.square(candidates.astype(np.float64).mean(0) - truth.astype(np.float64)).mean())
    assert abs(met['reconstruction_mse'] - exact) < 1e-12
    me, pe = rng.exponential(5, K), rng.exponential(100, K)
    logw = rng.normal(size=K); logw -= logsumexp(logw)
    result = recursive_update(logw, me, pe, lambda_w=1., target_ess=8.)
    expected = result['beta'] * (logw - me - pe); expected -= logsumexp(expected)
    assert np.max(np.abs(expected - result['log_weights'])) < 1e-12
    assert result['ess'] >= 8 - 1e-10
    seen = np.zeros(1024, bool); seen[order[:50]] = True
    chosen, scores = select_block(candidates, UNIFORM, seen)
    variance = np.square(candidates.astype(np.float64) - candidates.astype(np.float64).mean(0)).mean(0)
    assert np.max(abs(scores - variance.ravel()[indices].sum(1))) < 1e-10
    assert not seen[chosen]
    record = dict(status='passed', measurement_plus_proximity_gradient_max_error=grad_error,
        measurement_only_original_sampler_max_error=sampler_error, rng_states_equal=True,
        mse_replay_error=abs(met['reconstruction_mse'] - exact), geometry_unique=True,
        whole_score_temperature_validated=True, acquisition_variance_validated=True,
        old_50_study_still_paused=(ROOT / 'runs/celebahq_local_refinement_v1/PAUSED_BY_USER.json').exists(),
        note='CPU tractable checks; actual-prior pilot and saved-output audits are separate.')
    write_json(RUN / 'preflight.json', record)
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'preflight', 'initialize', 'infer'])
    parser.add_argument('--rank', type=int)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    try:
        if args.stage == 'prepare': prepare()
        elif args.stage == 'preflight': preflight()
        elif args.stage == 'initialize': initialize(args.limit)
        elif args.rank is not None: trajectory(args.rank, args.limit)
        else: parser.error('--rank required for infer')
    except Exception:
        write_json(RUN / 'failures' / f'{args.stage}_{args.rank}_{time.time_ns()}.json',
            dict(stage=args.stage, rank=args.rank, traceback=traceback.format_exc(),
                 job_id=os.environ.get('SLURM_JOB_ID'), time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
        raise


if __name__ == '__main__':
    main()
