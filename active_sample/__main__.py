"""Run from the repository checkout: python -m active_sample --help."""
import argparse
import gc
import json
from pathlib import Path

GROUPS = {'50-100': 'b100_i050', '100-200': 'b200_i100', '250-500': 'b500_i250'}
IMAGES = (187000, 183258, 200793, 189467, 193037)


def select_tasks(protocol, group='all', image=None, policy='all'):
    return [t for t in protocol['tasks']
            if (group == 'all' or t['group_key'] == GROUPS[group])
            and (image is None or t['case_key'] == f'image_{image}_seed_6101')
            and (policy == 'all' or t['policy'] == policy)]


def run_selected(base, tasks, limit):
    """Initialize each selected image/budget once, then resume its policies."""
    protocol = base.verify_protocol()
    pending = {}
    for task in tasks:
        group, case = base.resolve(protocol, task)
        folder = base.case_root(group, case) / 'initialization'
        if not (base.round_path(folder, group['initial']) / 'complete.json').exists():
            pending[(group['key'], case['key'])] = (group, case)
    if pending:
        import torch
        prior, execution = base.load_prior()
        for group, case in pending.values():
            base.initialize_one(prior, group, case, execution)
        del prior
        gc.collect()
        torch.cuda.empty_cache()
    for task in tasks:
        base.trajectory(task['rank'], limit)


def main():
    parser = argparse.ArgumentParser(description='CelebA128 active block measurements; frozen K=16, 128-step sampler.')
    commands = parser.add_subparsers(dest='command', required=True)
    for cmd in ('prepare', 'preflight', 'status', 'tasks'):
        commands.add_parser(cmd)
    run = commands.add_parser('run', help='Initialize if missing and resume selected trajectories.')
    run.add_argument('--group', choices=['all', *GROUPS], default='50-100')
    run.add_argument('--image', type=int, choices=IMAGES, help='Omit for all five images.')
    run.add_argument('--policy', choices=['all', 'U', 'W', 'SCRATCH'], default='all')
    run.add_argument('--max-updates', type=int, help='Additional rounds per selected trajectory this invocation.')
    report = commands.add_parser('report', help='Build the self-contained offline HTML from available results.')
    report.add_argument('--frames', type=int, default=11)
    args = parser.parse_args()
    if args.command == 'report':
        if args.frames < 2:
            parser.error('--frames must be at least 2')
        from experiments.ads_celeba128_report import build
        build(frames=args.frames)
        return
    from experiments import ads_celeba128 as base
    if args.command == 'prepare':
        if (base.RUN / 'protocol.json').exists():
            base.verify_protocol()
            print('Existing protocol verified; no inputs or initializations overwritten.')
        else:
            base.prepare()
    elif args.command == 'preflight':
        base.preflight()
    else:
        protocol = base.verify_protocol()
        if args.command == 'tasks':
            print(json.dumps(protocol['tasks'], indent=2))
        elif args.command == 'run':
            if args.max_updates is not None and args.max_updates < 1:
                parser.error('--max-updates must be positive')
            run_selected(base, select_tasks(protocol, args.group, args.image, args.policy), args.max_updates)
        else:
            rows = []
            for task in protocol['tasks']:
                group, case = base.resolve(protocol, task)
                path = base.case_root(group, case)
                initialized = (path / 'initialization' / f"round_{group['initial']:03d}" / 'complete.json').exists()
                last = group['initial'] if initialized else None
                for count in range(group['initial'] + 1, group['final'] + 1):
                    if not (path / task['policy'] / f'round_{count:03d}' / 'complete.json').exists():
                        break
                    last = count
                rows.append(dict(task, observed_blocks=last, final=group['final'], complete=last == group['final']))
            print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
