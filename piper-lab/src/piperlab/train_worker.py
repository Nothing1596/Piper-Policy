"""Narrow compatibility shim around the pinned official LeRobot trainer.

Compute normalization from training episodes only. On Windows write a pointer
file instead of requiring administrator rights to create a last-checkpoint link.
"""
import os
import runpy
from pathlib import Path


def main():
    import numpy as np
    import pyarrow.dataset as arrow
    from lerobot.datasets import factory
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.utils.utils import unflatten_dict

    original = factory.make_dataset
    def make_dataset(cfg):
        dataset = original(cfg)
        if not cfg.dataset.episodes:
            raise ValueError('An explicit training episode split is required')
        stats = []
        rows = arrow.dataset(str(Path(cfg.dataset.root) / 'meta' / 'episodes'), format='parquet').to_table().to_pylist()
        rows = {int(row['episode_index']): row for row in rows}
        for index in cfg.dataset.episodes:
            row = rows[index]
            stats.append(unflatten_dict({k: np.asarray(v) for k, v in row.items()
                                         if k.startswith('stats/')} )['stats'])
        dataset.meta.stats = aggregate_stats(stats)
        return dataset
    factory.make_dataset = make_dataset
    if os.name == 'nt':
        from lerobot.common import train_utils
        def update_last(checkpoint):
            (checkpoint.parent / 'last.txt').write_text(checkpoint.name, encoding='utf-8')
        train_utils.update_last_checkpoint = update_last
    runpy.run_module('lerobot.scripts.lerobot_train', run_name='__main__')


if __name__ == '__main__':
    main()
