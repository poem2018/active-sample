import hashlib
import json
from pathlib import Path

import numpy as np

from active_sample.__main__ import select_tasks
from experiments.ads_celeba128 import spatial_order, seeds_for

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_algorithm_files_match_original_experiment():
    record = json.loads((ROOT / 'configs/release_provenance.json').read_text())
    for name, expected in record['unchanged_files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name


def test_initial_layouts_match_saved_protocol():
    protocol = json.loads((ROOT / 'configs/experiment_reference.json').read_text())
    order = spatial_order()
    for group in protocol['groups']:
        assert order[:group['initial']] == group['initial_block_ids']


def test_cli_selects_policies_not_reconstruction_views():
    tasks = [dict(rank=i, group_key='b100_i050', case_key='image_187000_seed_6101', policy=p)
             for i, p in enumerate(['U', 'W', 'SCRATCH'])]
    assert select_tasks({'tasks': tasks}, '50-100', 187000) == tasks
    assert select_tasks({'tasks': tasks}, '50-100', 187000, 'W') == [tasks[1]]
    assert not select_tasks({'tasks': tasks}, '250-500', 187000)


def test_sensor_noise_is_position_fixed_across_budgets():
    a = seeds_for(187000, 6101, 50, 51)
    b = seeds_for(187000, 6101, 250, 251)
    assert a['sensor'] == b['sensor']
    assert a['gaussian'] != b['gaussian']
