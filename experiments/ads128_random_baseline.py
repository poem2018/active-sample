"""Uniform-random block acquisition control for the frozen250->500study.

Only acquisition changes relative to SCRATCH. Initial candidates, physical
observations, prior, fresh Gaussian/recombination inputs, sampler and evaluator
are imported from the unchanged frozen experiment. No GT enters selection.
"""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from . import ads_celeba128 as base
from .common import ROOT, sha256, write_json

RUN = base.RUN / 'extensions/random250_v1'
SOURCE = ROOT / 'experiments/ads128_random_baseline.py'


def design():
    protocol = base.verify_protocol()
    group = next(g for g in protocol['groups'] if g['initial'] == 250)
    plans = []
    for case in protocol['cases']:
        seed = 2026101706 + case['image_id'] * 1009 + case['seed'] * 101
        remaining = np.setdiff1d(np.arange(1024), group['initial_block_ids'])
        suffix = np.random.default_rng(seed).permutation(remaining)[:250].tolist()
        plans.append(dict(case_key=case['key'], acquisition_seed=seed,
                          added_block_ids=suffix))
    return protocol, group, plans


def prepare():
    if (RUN / 'protocol.json').exists():
        raise FileExistsError('Random baseline already frozen; resume infer.')
    protocol, group, plans = design()
    record = dict(status='locked', group=group, cases=protocol['cases'], plans=plans,
        base_protocol_sha256=sha256(base.RUN / 'protocol.json'),
        source_sha256=sha256(SOURCE), K=16, steps=128, sigma=.05,
        initial=250, final=500, expected_updates=1250, expected_trajectories=5,
        method='RANDOM: uniform random unobserved blocks + From-scratch Flow + uniform mean',
        selection='Prespecified uniform permutation of the774unobserved aligned blocks per image; first250used; no GT/candidate input.',
        generation='Exact frozen base.generate(...,old=None); all measurements, same paired Gaussian and recombination seeds by image/budget/count.',
        initialization='Load existing corresponding250-block initial posterior; no regeneration.',
        measurements='Same sensor_bank.npz, initial250blocks, sigma=.05, task128 native256 D2.',
        estimation='Same unclipped base.evaluate; primary candidate-MEAN MSE; update mean excludes initialization.',
        initial_commit_sha256={c['key']:sha256(base.case_root(group,c) / 'initialization/round_250/complete.json') for c in protocol['cases']},
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    write_json(RUN / 'protocol.json', record)
    print(json.dumps(dict(status='prepared', images=[c['image_id'] for c in protocol['cases']], updates=1250)))


def verify():
    base.verify_protocol()
    p = base.read(RUN / 'protocol.json')
    assert sha256(base.RUN / 'protocol.json') == p['base_protocol_sha256']
    assert sha256(SOURCE) == p['source_sha256']
    for c,plan in zip(p['cases'],p['plans']):
        assert c['key'] == plan['case_key']
        history = p['group']['initial_block_ids'] + plan['added_block_ids']
        assert len(history) == len(set(history)) == 500
        assert all(0 <= b < 1024 for b in history)
        initial = base.case_root(p['group'],c) / 'initialization/round_250/complete.json'
        assert sha256(initial) == p['initial_commit_sha256'][c['key']]
    return p


def infer(index, limit=None):
    p = verify(); group = p['group']; case = p['cases'][index]; plan = p['plans'][index]
    folder = RUN / 'cases' / case['key'] / 'RANDOM'
    with base.exclusive(folder / '.worker.lock'):
        if (folder / 'completion.json').exists(): return
        base.verify_inputs(case)
        last = 250
        for count in range(251,501):
            if not (base.round_path(folder,count) / 'complete.json').exists(): break
            last = count
        source = base.case_root(group,case) / 'initialization' if last == 250 else folder
        state, parent, previous = base.load_checkpoint(source,last)
        full_history = group['initial_block_ids'] + plan['added_block_ids']
        assert state['history'] == full_history[:last]
        if last == 500:
            write_json(folder / 'completion.json',dict(status='complete', case=case['key'], policy='RANDOM',initial=250,final=500,history=full_history,update_nfe=250*128))
            return
        with np.load(base.RUN / case['sensor']) as data:
            sensor = {k:data[k] for k in data.files}
        truth = np.load(base.RUN / case['truth'])  # evaluation only
        prior, execution = base.load_prior()
        write_json(folder / f'execution_{time.time_ns()}.json',execution)
        end = 500 if limit is None else min(500,last+limit)
        keep = {250+250*j//4 for j in range(1,5)}
        for count in range(last+1,end+1):
            tic=time.perf_counter()
            chosen = plan['added_block_ids'][count-251]
            assert chosen not in state['history']
            history = state['history'] + [chosen]
            acquisition_s=time.perf_counter()-tic
            ids,y=sensor['indices'][history],sensor['readings'][history]
            native,diagnostics,cost=base.generate(prior,group,case,count,ids,y,None)
            task=base.pool(native)
            tic=time.perf_counter()
            metric=base.evaluate(task,truth,ids,y,250,previous)
            evaluation_s=time.perf_counter()-tic
            state=dict(count=count,history=history,selected=chosen,policy='RANDOM',
                previous_log_weights=np.log(base.UNIFORM).tolist(),log_weights=np.log(base.UNIFORM).tolist(),
                beta=0.,acquisition_weights=base.UNIFORM.tolist(),acquisition_scores=None,
                acquisition_seed=plan['acquisition_seed'],acquisition_probability=1/(1024-(count-1)),
                score={},scoring_ess=16.,scoring_maximum_weight=1/16,
                seeds=base.seeds_for(case['image_id'],case['seed'],250,count),
                acquisition_s=acquisition_s,scoring_s=0.,evaluation_s=evaluation_s,**cost)
            base.persist(folder,count,native,task,state,dict(uniform=metric,weighted=metric),diagnostics,count in keep)
            previous=task
            print(json.dumps(dict(case=case['key'],policy='RANDOM',count=count,mse=metric['reconstruction_mse'],seconds=cost['generation_s'])),flush=True)
        if end==500:
            write_json(folder / 'completion.json',dict(status='complete',case=case['key'],policy='RANDOM',initial=250,final=500,history=history,update_nfe=250*128,protocol_sha256=sha256(RUN/'protocol.json')))


def preflight():
    p=verify(); _,_,again=design()
    assert again==p['plans']
    for case in p['cases']:
        initial=base.case_root(p['group'],case)/'initialization'
        state,native,task=base.load_checkpoint(initial,250)
        assert state['history']==p['group']['initial_block_ids']
        assert np.array_equal(base.pool(native),task)
    write_json(RUN/'preflight.json',dict(status='passed',fixed_roster=5,unique_blocks=500,
        seeds_replay=True,initial_candidates_reused=True,native_task_pool_exact=True,
        sampler='Imported unchanged base.generate with old=None; actual GPU smoke run still required.'))
    print('random baseline geometry, seed replay, initialization hashes and operator preflight passed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['prepare','preflight','infer'])
    parser.add_argument('--case-index',type=int); parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    try:
        if args.stage=='prepare': prepare()
        elif args.stage=='preflight': preflight()
        elif args.case_index is not None: infer(args.case_index,args.limit)
        else: parser.error('--case-index required')
    except Exception:
        write_json(RUN/'failures'/f'{os.environ.get("SLURM_JOB_ID","local")}_{time.time_ns()}.json',dict(traceback=traceback.format_exc(),stage=args.stage,case_index=args.case_index))
        raise
