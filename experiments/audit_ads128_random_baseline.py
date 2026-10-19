"""Replay committed RANDOM outputs without sampling; usable as a job gate."""
import argparse
import json
import time
import traceback

import numpy as np

from . import ads_celeba128 as base
from . import ads128_random_baseline as random_run
from .common import sha256, write_json


def audit(index):
    p=random_run.verify(); case=p['cases'][index]; group=p['group']; plan=p['plans'][index]
    base.verify_inputs(case)
    folder=random_run.RUN/'cases'/case['key']/'RANDOM'
    initial=base.case_root(group,case)/'initialization'
    parent,_,previous=base.load_checkpoint(initial,250)
    expected=group['initial_block_ids']+plan['added_block_ids']
    with np.load(base.RUN/case['sensor']) as f:
        sensor={k:f[k] for k in f.files}
    gt=np.load(base.RUN/case['truth'])
    checks=0; max_error=0.; native_checks=0; last=250
    for count in range(251,501):
        path=base.round_path(folder,count)
        if not (path/'complete.json').exists(): break
        marker=base.read(path/'complete.json')
        assert marker['count']==count
        for name,key in [('state.json','state_sha256'),('metrics.json','metrics_sha256'),('candidates.npy','candidates_sha256')]:
            assert sha256(path/name)==marker[key], str(path/name)
        state=base.read(path/'state.json'); values=base.read(path/'metrics.json')
        assert state['history']==expected[:count]
        assert parent['history']==state['history'][:-1]
        assert state['selected']==plan['added_block_ids'][count-251]
        assert state['selected'] not in parent['history']
        assert state['seeds']==base.seeds_for(case['image_id'],case['seed'],250,count)
        assert state['nfe']==128 and state['candidate_model_evaluations']==2048
        assert state['guidance_blocks']==count and state['guidance_lipschitz']==100.
        assert np.array_equal(state['log_weights'],np.log(base.UNIFORM))
        assert np.array_equal(state['previous_log_weights'],parent['log_weights'])
        assert state['acquisition_seed']==plan['acquisition_seed']
        assert state['acquisition_probability']==1/(1024-(count-1))
        assert values['weighted']==values['uniform']
        task=np.load(path/'candidates.npy')
        history=state['history']; ids=sensor['indices'][history]; y=sensor['readings'][history]
        actual=base.evaluate(task,gt,ids,y,250,previous)
        for name,value in actual.items():
            saved=values['uniform'][name]
            if value is None: assert saved is None; continue
            err=float(np.max(np.abs(np.asarray(value)-np.asarray(saved))))
            assert err<1e-10, (count,name,err)
            max_error=max(max_error,err); checks+=1
        native=base.native_path(path)
        if not native.exists():
            assert not marker['native_retained']
            assert (base.round_path(folder,count+1)/'complete.json').exists()
        if native.exists():
            assert sha256(native)==marker['native_sha256']
            assert np.array_equal(base.pool(np.load(native)),task)
            native_checks+=1
        with np.load(path/'diagnostics.npz') as d:
            assert np.all(d['proximity_l2']==0)
        parent,previous,last=state,task,count
    assert last>250, 'No actual update available for GPU smoke gate'
    if last==500:
        complete=base.read(folder/'completion.json')
        assert complete['status']=='complete' and complete['update_nfe']==250*128
        assert complete['history']==expected
    record=dict(status='passed',case=case['key'],last=last,updates=last-250,
        complete=(last==500),metric_comparisons=checks,max_metric_error=max_error,
        available_native_checks=native_checks,initialization_reused=True,
        random_history_replay=True,paired_generation_seeds=True,uniform_weights=True,
        no_proximity_guidance=True,initialization_excluded_from_update_average=True,
        protocol_sha256=sha256(random_run.RUN/'protocol.json'),
        updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    write_json(random_run.RUN/'audits'/f'{case["key"]}.json',record)
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case-index',type=int,required=True)
    args=parser.parse_args()
    try: audit(args.case_index)
    except Exception:
        write_json(random_run.RUN/'audits'/f'failure_{args.case_index}_{time.time_ns()}.json',dict(status='failed',traceback=traceback.format_exc()))
        raise
