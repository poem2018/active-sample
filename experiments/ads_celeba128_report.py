"""Offline report for the five-image ADS-inspired direct-block experiments.

This module reads committed saved trajectories only. It never runs generation,
selects measurements, or changes the experiment configuration.
"""
import argparse
import base64
import fcntl
import hashlib
import io
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'runs/celeba128_ads_blocks_v1'
POLICIES = ('U', 'W', 'SCRATCH')
ARMS = ('UU', 'UW', 'WU', 'WW', 'SCRATCH')
TEMPLATE = Path(__file__).with_name('ads_celeba128_viewer.html')
TIME_FIELDS = ('generation_s', 'acquisition_s', 'scoring_s', 'evaluation_s', 'io_s')
CALL_FIELDS = ('nfe', 'candidate_model_evaluations', 'backward_evaluations', 'trace_evaluations')
MEMORY_FIELDS = ('peak_allocated_bytes', 'peak_reserved_bytes')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def pixels(value):
    value = np.asarray(value)
    return np.rint(np.clip((value + 1) / 2, 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)


def atlas(value):
    value = np.asarray(value)
    if value.shape != (16, 3, 128, 128):
        raise ValueError(f'Expected K16 task128 candidates, got {value.shape}')
    return np.concatenate([np.concatenate([pixels(value[4*r+c]) for c in range(4)], axis=1)
                           for r in range(4)], axis=0)


class Assets:
    """Lossless deduplicated data URLs, with round-trip pixel verification."""
    def __init__(self):
        self.values = []
        self.index = {}
        self.checks = 0

    def add(self, array):
        array = np.asarray(array, dtype=np.uint8)
        key = hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()
        if key in self.index:
            return self.index[key]
        output = io.BytesIO()
        Image.fromarray(array).save(output, format='WEBP', lossless=True, method=2)
        encoded = output.getvalue()
        np.testing.assert_array_equal(np.array(Image.open(io.BytesIO(encoded))), array)
        self.checks += 1
        result = len(self.values)
        self.values.append('data:image/webp;base64,' + base64.b64encode(encoded).decode('ascii'))
        self.index[key] = result
        return result


def metric_view(metric):
    return dict(mse=metric.get('reconstruction_mse'), psnr_db=metric.get('psnr', metric.get('psnr_db')),
                ssim=metric.get('ssim'), unobserved_mse=metric.get('unobserved_mse'),
                observed_rmse=metric.get('observed_mean_rmse'),
                initial_rmse=metric.get('initial_history_mean_rmse'),
                old_rmse=metric.get('old_mean_rmse'), new_rmse=metric.get('new_mean_rmse'))


def folder_at(base, policy, count, initial):
    return base / ('initialization' if count == initial else policy) / f'round_{count:03d}'


def committed(folder):
    return (folder / 'complete.json').exists()


def load_metadata(base, policy, initial, final):
    result = {}
    # Only consecutive committed rounds count as a complete trajectory. A file
    # from a crashed or currently-writing round is never displayed.
    for count in range(initial, final + 1):
        folder = folder_at(base, policy, count, initial)
        if not committed(folder):
            break
        metrics, state = read(folder / 'metrics.json'), read(folder / 'state.json')
        if state['count'] != count:
            raise ValueError(f'Checkpoint count mismatch: {folder}')
        history = state['history']
        if len(history) != count or len(set(history)) != count:
            raise ValueError(f'Invalid block history: {folder}')
        result[count] = dict(metrics=metrics, state=state, folder=folder)
    return result


def curve(metadata, weighted):
    result = []
    generation = 0.
    for count, data in sorted(metadata.items()):
        state = data['state']
        m = data['metrics']['weighted' if weighted else 'uniform']
        if count != min(metadata):
            generation += float(state.get('generation_s', 0.))
        logw = np.asarray(state['log_weights'], np.float64)
        ess = float(1 / np.exp(2 * logw).sum())
        result.append(dict(count=count, mse=m['reconstruction_mse'],
            psnr_db=m.get('psnr', m.get('psnr_db')), ssim=m.get('ssim'),
            unobserved_mse=m.get('unobserved_mse'), observed_rmse=m.get('observed_mean_rmse'),
            initial_mean_rmse=m.get('initial_history_mean_rmse'), old_mean_rmse=m.get('old_mean_rmse'),
            new_mean_rmse=m.get('new_mean_rmse'), new_candidates_before_rmse=m.get('new_before_rmse'),
            new_candidates_after_rmse=m.get('new_after_rmse'), candidate_change_rmse=m.get('candidate_change_rmse'),
            ess=ess, diversity=m.get('pairwise_diversity_rmse'),
            scoring_maximum_weight=float(np.exp(logw).max()), scoring_beta=float(state.get('beta', 0.)),
            weighted_variance=m.get('weighted_candidate_variance'), generation_s=generation))
    return result


def arm_summary(metadata, weighted, initial, final):
    if not metadata:
        return dict(complete=False, last_count=None)
    key = 'weighted' if weighted else 'uniform'
    init = metadata[initial]['metrics'][key]
    rows = [v for c, v in sorted(metadata.items()) if c > initial]
    complete = final in metadata and len(metadata) == final - initial + 1
    final_metric = metadata[final]['metrics'][key] if complete else {}
    return dict(complete=complete, last_count=max(metadata),
        initial_mse=init['reconstruction_mse'], initial_psnr_db=init.get('psnr', init.get('psnr_db')),
        initial_ssim=init.get('ssim'), final_mse=final_metric.get('reconstruction_mse'),
        psnr_db=final_metric.get('psnr', final_metric.get('psnr_db')), ssim=final_metric.get('ssim'),
        unobserved_mse=final_metric.get('unobserved_mse'),
        trajectory_mse=float(np.mean([r['metrics'][key]['reconstruction_mse'] for r in rows])) if complete else None,
        initial_generation_s=float(metadata[initial]['state'].get('generation_s', 0.)),
        initial_nfe=int(metadata[initial]['state'].get('nfe', 0)),
        generation_s=sum(float(r['state'].get('generation_s', 0.)) for r in rows),
        nfe=sum(int(r['state'].get('nfe', 0)) for r in rows))


def round_costs(records):
    """Count each committed generation once; memory is a maximum, never a sum."""
    states = [r['state'] for r in records]
    result = dict(committed_generations=len(states))
    for field in TIME_FIELDS + CALL_FIELDS:
        result[field] = sum(s.get(field, 0) for s in states)
    for field in MEMORY_FIELDS:
        result[field] = max((s.get(field, 0) for s in states), default=0)
    result['accounted_s'] = sum(result[f] for f in TIME_FIELDS)
    return result


def execution_costs(paths):
    """Loads/warmups are invocation overhead, not candidate generation steps."""
    records = [read(p) for p in paths]
    return dict(invocations=len(records), load_s=sum(r.get('load_s', 0) for r in records),
                warmup_s=sum(r.get('warmup_s', 0) for r in records),
                warmup_nfe=sum(r.get('warmup_nfe', 0) for r in records))


def case_costs(base, metadata, initial):
    # Shared initialization is listed once, followed by the three real updates.
    init = metadata['U'].get(initial)
    result = dict(initialization=round_costs([init] if init else []), updates={})
    for policy in POLICIES:
        records = [r for c, r in metadata[policy].items() if c > initial]
        result['updates'][policy] = round_costs(records)
        result['updates'][policy]['execution'] = execution_costs(sorted((base / policy).glob('execution_*.json')))
    return result


def visual_counts(initial, last, final, frames):
    if last < initial:
        return []
    counts = {initial, last}
    if last > initial:
        counts.add(initial + 1)
    counts.update(int(round(x)) for x in np.linspace(initial, final, max(2, frames - 1)) if x <= last)
    return sorted(counts)


def make_case(run, group, case, assets, audit, frames):
    initial, final = int(group['initial']), int(group['final'])
    base = run / 'groups' / group['key'] / 'cases' / case['key']
    truth = np.load(run / case['truth'])
    if truth.shape != (3, 128, 128):
        raise ValueError(f'Task GT must be128x128: {case["truth"]}, {truth.shape}')
    with np.load(run / case['sensor']) as f:
        sensor = {name: f[name] for name in f.files}
    if sensor['indices'].shape != (1024, 48):
        raise ValueError('Expected1024 nonoverlapping4x4 RGB blocks')
    metadata = {p: load_metadata(base, p, initial, final) for p in POLICIES}
    # Visual comparisons never mix current rounds between methods. Initialization
    # is shared and remains visible before any active trajectory completes.
    last = min(max(m) if m else initial - 1 for m in metadata.values())
    counts = visual_counts(initial, last, final, frames)
    summaries, curves = {}, {}
    for arm in ARMS:
        p = 'SCRATCH' if arm == 'SCRATCH' else arm[0]
        weighted = arm in ('UW', 'WW')
        summaries[arm] = arm_summary(metadata[p], weighted, initial, final)
        curves[arm] = curve(metadata[p], weighted)
    result = dict(key=case['key'], image_id=case['image_id'], seed=case['seed'],
        label=f'#{case["image_id"]} · seed {case["seed"]}', initial=initial, final=final,
        gt=assets.add(pixels(truth)), visual_counts=counts, rounds={}, summary=summaries, curves=curves,
        costs=case_costs(base, metadata, initial),
        note=f'共享初始{initial}块，之后每轮主动获取1块，目标{final}块。可视化显示三条轨迹共同完成至{last if last >= initial else "尚无初始化"}块。')
    result['acquisition_paths'] = {}
    for policy, records in metadata.items():
        if not records:
            result['acquisition_paths'][policy] = dict(last_count=None, initial=[], acquired=[])
            continue
        last_count = max(records)
        history = records[last_count]['state']['history']
        path = [dict(block_id=int(block), row=int(sensor['block_top_left'][block][0]),
                     col=int(sensor['block_top_left'][block][1]), count=index + 1,
                     acquisition_index=index + 1 - initial if index >= initial else None)
                for index, block in enumerate(history)]
        result['acquisition_paths'][policy] = dict(last_count=last_count,
            initial=path[:initial], acquired=path[initial:])
    for count in counts:
        round_record = dict(count=count, policies={})
        for policy in POLICIES:
            record = metadata[policy][count]
            state, metric, folder = record['state'], record['metrics'], record['folder']
            x = np.load(folder / 'candidates.npy').astype(np.float64)
            prev_count = initial if count == initial else count - 1
            old = np.load(folder_at(base, policy, prev_count, initial) / 'candidates.npy').astype(np.float64)
            if x.shape != (16, 3, 128, 128) or not np.isfinite(x).all():
                raise ValueError(f'Invalid raw task candidates: {folder}')
            logw = np.asarray(state['log_weights'], np.float64)
            w = np.exp(logw)
            if w.shape != (16,) or not np.isfinite(logw).all() or abs(w.sum() - 1) > 1e-8:
                raise ValueError(f'Invalid scoring log weights: {folder}')
            mean = x.mean(0)
            weighted_mean = np.einsum('i,ichw->chw', w, x)
            history = np.asarray(state['history'], dtype=np.int64)
            ids = sensor['indices'][history].ravel()
            values = sensor['readings'][history].ravel()
            observation = np.full((3, 128, 128), -.55, dtype=np.float64)
            observation.flat[ids] = values
            # Input verification is independent of the image encoder: assigned
            # pixels must equal sensor values and all other pixels remain gray.
            seen = np.zeros(3 * 128 * 128, bool)
            seen[ids] = True
            np.testing.assert_array_equal(observation.ravel()[ids], values)
            assert np.all(observation.ravel()[~seen] == -.55)
            for key, reconstruction in [('uniform', mean), ('weighted', weighted_mean)]:
                saved = metric[key]['reconstruction_mse']
                replay = float(np.square(reconstruction - truth.astype(np.float64)).mean())
                error = abs(saved - replay)
                audit['mse_checks'] += 1
                audit['max_mse_replay_error'] = max(audit['max_mse_replay_error'], error)
                if error > 2e-8:
                    raise ValueError(f'Raw mean/MSE mismatch {folder} {key}: {saved} vs {replay}')
            difference = np.abs(x - old).mean(1)
            gray = np.rint(np.clip(difference / .25, 0, 1) * 255).astype(np.uint8)
            difference_atlas = np.concatenate([np.concatenate([gray[4*r+c] for c in range(4)], axis=1)
                                               for r in range(4)], axis=0)
            # RGB encoding makes decoder/browser behavior uniform across assets.
            difference_atlas = np.repeat(difference_atlas[..., None], 3, axis=-1)
            selected = state.get('selected')
            latest = None if selected is None else sensor['block_top_left'][int(selected)].tolist()
            round_record['policies'][policy] = dict(input=assets.add(pixels(observation)),
                candidates=assets.add(atlas(x)), previous=assets.add(atlas(old)),
                difference=assets.add(difference_atlas), uniform_mean=assets.add(pixels(mean)),
                weighted_mean=assets.add(pixels(weighted_mean)), history=history.tolist(), latest=latest,
                weights=w.tolist(), log_weights=logw.tolist(), beta=float(state.get('beta', 0.)),
                ess=float(1 / np.square(w).sum()), maximum_weight=float(w.max()),
                change_rmse=float(np.sqrt(np.square(x - old).mean())),
                metrics=dict(primary=metric_view(metric['uniform']), weighted=metric_view(metric['weighted'])))
            audit['input_checks'] += 1
        result['rounds'][str(count)] = round_record
    return result


def group_findings(aggregate_result, initial, final):
    """Descriptive fixed comparisons only; never used for inference or tuning."""
    if aggregate_result['complete_images'] != 5:
        return [f'本组尚有未完成图像（共同完成{aggregate_result["complete_images"]}/5）；五图结果解释将在全部完成后生成。当前配对表仅供查看已保存结果。']
    contrasts = {row['label']: row for row in aggregate_result['contrasts']}
    def comparison(label):
        c = contrasts[label]
        return (f'{label}：最终ΔMSE={c["final_difference"]:+.6g}（{c["improved_images"]}/5图降低），'
                f'更新过程平均ΔMSE={c["trajectory_difference"]:+.6g}（{c["trajectory_improved_images"]}/5图降低）')
    curves = aggregate_result['curves']
    ww_minus_uu = [a[1] - b[1] for a, b in zip(curves['WW'][1:], curves['UU'][1:])]
    better_rounds = sum(d < 0 for d in ww_minus_uu)
    scratch = contrasts['UU − SCRATCH']
    tradeoff = ('最终与过程均值的差异方向相反，不能只看最终一轮。'
                if scratch['final_difference'] * scratch['trajectory_difference'] < 0 else '')
    return [
        '重建加权（完全相同候选与选点路径）：' + comparison('UW − UU') + '；' + comparison('WW − WU') + '。',
        '选点加权（包含后续候选更新的反馈）：' + comparison('WU − UU') + '；' + comparison('WW − UW') + '。',
        '组合方法：' + comparison('WW − UU') + f'。在{initial+1}–{final}的{len(ww_minus_uu)}个更新预算中，WW的五图平均MSE有{better_rounds}个预算低于UU；这些轮次不是独立图像样本。',
        '旧候选方法对照：' + comparison('UU − SCRATCH') + '。' + tradeoff + '这是两种主动采集方法的整体比较；路径会随候选改变，不能解释为固定历史下旧候选guidance的单独因果效应。',
        '以上仅描述预先固定的5张图和本组预算。负ΔMSE表示前者更低，正值表示更高；没有据此选择参数或替换图像，也不据此声称普遍有效。'
    ]


def cross_budget_findings(groups):
    if not all(g['aggregate']['complete_images'] == 5 for g in groups):
        return ['三组预算尚未全部完成5张图；跨预算结论待全部完成后生成，不把部分完成的预算当作最终证据。']
    descriptions = []
    final_improvements = trajectory_improvements = 0
    for group in groups:
        c = next(c for c in group['aggregate']['contrasts'] if c['label'] == 'WW − UU')
        descriptions.append(f'{group["label"]}：最终ΔMSE={c["final_difference"]:+.6g}，更新过程平均ΔMSE={c["trajectory_difference"]:+.6g}')
        final_improvements += c['final_difference'] < 0
        trajectory_improvements += c['trajectory_difference'] < 0
    return ['WW相对UU的三个独立预算设置：' + '；'.join(descriptions) + '。',
            f'WW的最终平均MSE在{final_improvements}/3组降低，更新过程平均MSE在{trajectory_improvements}/3组降低。三组使用同样5张图，不能把它们当成15张独立图像；各组详细解释见下方。']


def aggregate(cases, initial, final):
    matched = [case for case in cases if all(case['summary'][arm].get('complete') for arm in ARMS)]
    n = len(matched)
    result = dict(complete_images=n, matched_image_ids=[c['image_id'] for c in matched], methods={}, curves={}, contrasts=[],
        note=f'{n}/5张图的U、W、From-scratch均完整到{final}块。汇总只使用这些共同完成图像；未完成图不被静默删除或用不同分母比较。更新过程平均只含{initial+1}–{final}块，不含初始化。五张图为小规模试验，不能当作充分的总体效果证据。')
    for arm in ARMS:
        if not n:
            result['methods'][arm] = {}
            result['curves'][arm] = []
            continue
        result['methods'][arm] = {field: float(np.mean([c['summary'][arm][field] for c in matched]))
            for field in ('initial_mse', 'initial_psnr_db', 'initial_ssim', 'final_mse',
                          'trajectory_mse', 'psnr_db', 'ssim', 'unobserved_mse')}
        result['curves'][arm] = [[count, float(np.mean([c['curves'][arm][count-initial]['mse'] for c in matched]))]
                                for count in range(initial, final + 1)]
    for a, b, role in [('UW', 'UU', '相同U候选：重建加权'), ('WW', 'WU', '相同W候选：重建加权'),
                       ('WU', 'UU', '均匀重建：主动选点权重'), ('WW', 'UW', '加权重建：主动选点权重'),
                       ('WW', 'UU', '选点与重建加权的组合'),
                       ('UU', 'SCRATCH', '旧候选guidance及主动路径反馈'),
                       ('WU', 'SCRATCH', '旧候选guidance与选点加权及路径反馈')]:
        differences = [c['summary'][a]['final_mse'] - c['summary'][b]['final_mse'] for c in matched]
        trajectory_differences = [c['summary'][a]['trajectory_mse'] - c['summary'][b]['trajectory_mse'] for c in matched]
        pairs = []
        for c in cases:
            paired = c['summary'][a].get('complete') and c['summary'][b].get('complete')
            pairs.append(dict(image_id=c['image_id'], complete=bool(paired),
                in_matched_summary=c in matched,
                final_difference=c['summary'][a]['final_mse'] - c['summary'][b]['final_mse'] if paired else None,
                trajectory_difference=c['summary'][a]['trajectory_mse'] - c['summary'][b]['trajectory_mse'] if paired else None))
        result['contrasts'].append(dict(label=f'{a} − {b}', role=role,
            final_difference=float(np.mean(differences)) if n else None,
            trajectory_difference=float(np.mean(trajectory_differences)) if n else None,
            improved_images=sum(x < 0 for x in differences),
            trajectory_improved_images=sum(x < 0 for x in trajectory_differences),
            per_image_differences=differences, per_image_pairs=pairs))
    result['findings'] = group_findings(result, initial, final)
    return result


def build(run=DEFAULT_RUN, frames=11):
    run = Path(run).resolve()
    protocol = read(run / 'protocol.json')
    if len(protocol['cases']) != 5:
        raise ValueError('This report requires exactly five predeclared cases')
    assets = Assets()
    audit = dict(status='passed', mse_checks=0, max_mse_replay_error=0., input_checks=0,
                 source_protocol_sha256=sha(run / 'protocol.json'))
    groups = []
    for group in protocol['groups']:
        cases = [make_case(run, group, case, assets, audit, frames) for case in protocol['cases']]
        groups.append(dict(key=group['key'], label=f'{group["initial"]} → {group["final"]} 块',
                           initial=group['initial'], final=group['final'], cases=cases,
                           aggregate=aggregate(cases, int(group['initial']), int(group['final']))))
    complete = sum(g['aggregate']['complete_images'] for g in groups)
    generation_progress = dict(
        initializations=sum(c['costs']['initialization']['committed_generations']
                            for g in groups for c in g['cases']),
        expected_initializations=sum(len(g['cases']) for g in groups),
        trajectories=sum(bool(c['summary'][arm].get('complete'))
                         for g in groups for c in g['cases'] for arm in ('UU', 'WU', 'SCRATCH')),
        expected_trajectories=sum(len(g['cases']) * len(POLICIES) for g in groups),
        updates=sum(c['costs']['updates'][policy]['committed_generations']
                    for g in groups for c in g['cases'] for policy in POLICIES),
        expected_updates=sum(len(g['cases']) * len(POLICIES) * (g['final'] - g['initial'])
                             for g in groups))
    init_execution = run / 'initialization_complete.json'
    shared_initialization_execution = None
    if init_execution.exists():
        record = read(init_execution)['execution']
        shared_initialization_execution = {k: record[k] for k in ('load_s', 'warmup_s', 'warmup_nfe')}
    validation = {}
    numerical = run / 'report/numerical_audit.json'
    if numerical.exists():
        record = read(numerical)
        validation['numerical'] = {k: record.get(k) for k in ('status', 'mode', 'initializations',
            'committed_updates', 'completed_trajectories', 'acquisition_replays', 'score_replays',
            'all_expected_outputs_complete', 'updated_utc')}
    native_pair = run / 'report/native_pair_audit/result.json'
    if native_pair.exists():
        validation['native_pair'] = read(native_pair)
    reader_incident = run / 'report/audit_visibility_fix/change.json'
    if reader_incident.exists():
        record = read(reader_incident)
        validation['resolved_audit_reader_incident'] = {k: record[k] for k in (
            'status', 'summary', 'scope', 'verification', 'failure_log', 'preserved_failure')}
    launch_recovery = run / 'report/launch_failure_recovery/summary.json'
    if launch_recovery.exists():
        validation['infrastructure_launch_recovery'] = read(launch_recovery)
    data = dict(generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), protocol=protocol,
        groups=groups, assets=assets.values, shared_initialization_execution=shared_initialization_execution,
        validation=validation, generation_progress=generation_progress,
        cross_budget_findings=cross_budget_findings(groups),
        status=(f'进度快照：已完成{generation_progress["updates"]}/{generation_progress["expected_updates"]}轮更新'
                f'（{generation_progress["updates"] / generation_progress["expected_updates"]:.1%}）；'
                f'{generation_progress["trajectories"]}/{generation_progress["expected_trajectories"]}条生成轨迹完整。'
                f'{complete}/15个图像×预算组已完成全部三条轨迹；三组更新长度不同，这一比例不代表总更新进度。'
                '共有5张图、3个独立预算组，所有已保存结果保留。'
                + ('全部组已完成。' if complete == 15 else '当前为未完成报告；最终结论需等待相应组完成。')),
        protocol_note='本次使用官方CelebA测试集图像，任务图为128×128。直接测量不重叠4×4 RGB块，共1024个可选块；没有完整低分辨率底图。初始50/100/250块空间分散，分别主动增加到100/200/500块。冻结prior在native256运行，候选以D₂平均池化映射到128任务图，所以任务算子是M，native算子为M D₂。相对ADS论文，这里保留我们自己的prior、FlowDPS风格采样器、σ=0.05噪声与初始半预算方案，并采用不重叠网格；不是ADS算法或论文数值的完整复现。',
        formula='生成：全部历史 Σ_j ||M_j D₂ f − y_j||²/(2σ²) + ||f − old_i||²_native/(2τ²)\nFrom-scratch去掉old项，其余生成设置相同。σ=0.05，τ=1，K=16，128步。\n评分：s_i = log(w_previous,i) − E_new,i − E_prox,i；w_i = softmax(β s_i)\nE_new仅使用最新块；E_prox为native256全图平方差之和/(2τ²)，λ_w=1。\nβ取[0,1]中保证ESS≥8的最大值，作用于整个分数。每个预算组初始权重均为1/16。\n选点：对每个未观测块，求像素与通道上的加权候选方差之和，选最大者。\nMSE在[-1,1]模型尺度计算；PSNR=10 log10(4/MSE)，逐图计算后平均。', audit=audit)
    output = run / 'report/ads_celeba128.html'
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('<', r'\u003c')
    html = TEMPLATE.read_text().replace('__DATA__', encoded)
    if output.exists():
        # Preserve the pre-change report once, without archiving every refresh.
        history = output.parent / 'build_history'
        history.mkdir(exist_ok=True)
        previous = history / 'before_cost_and_paired_report.html'
        if not previous.exists():
            previous.write_bytes(output.read_bytes())
    tmp = output.with_suffix('.tmp')
    tmp.write_text(html)
    tmp.replace(output)
    audit['asset_pixel_roundtrips'] = assets.checks
    audit['assets'] = len(assets.values)
    audit['html_sha256'] = sha(output)
    audit['html_bytes'] = output.stat().st_size
    audit['complete_group_image_cases'] = complete
    audit['generation_progress'] = generation_progress
    audit['visual_counts'] = {g['key']: {c['key']: c['visual_counts'] for c in g['cases']} for g in groups}
    dump(run / 'report/report_manifest.json', audit)
    dump(run / 'analysis/ads_celeba128_summary.json', dict(generated_utc=data['generated_utc'],
        generation_progress=generation_progress,
        protocol_sha256=audit['source_protocol_sha256'], groups=[dict(key=g['key'], aggregate=g['aggregate'],
            cases=[dict(key=c['key'], image_id=c['image_id'], summary=c['summary'], costs=c['costs']) for c in g['cases']]) for g in groups],
        shared_initialization_execution=shared_initialization_execution))
    print(json.dumps(dict(report=str(output), complete_group_image_cases=complete,
                         size_mb=output.stat().st_size/1e6, assets=len(assets.values), audit=audit), ensure_ascii=False))
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--frames', type=int, default=11, help='Approximate visual checkpoints per group; all numeric rounds are retained')
    args = parser.parse_args()
    if args.frames < 2:
        parser.error('--frames must be at least2')
    lock_path = args.run / 'report/.build.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        build(args.run, args.frames)
