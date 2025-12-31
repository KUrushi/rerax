from typing import Sequence

import grain.python as grain


def build_dataloader(
    data_source: grain.RandomAccessDataSource,
    batch_size: int,
    seed=0,
    drop_remainder=True,
    *,
    shuffle: bool = False,
    transformations: Sequence[grain.Transformation] = (),
    worker_count: int = 0,
    num_epochs: int | None = None,
) -> grain.DataLoader:
    sampler = grain.IndexSampler(
        num_records=len(data_source), shuffle=shuffle, seed=seed, num_epochs=num_epochs
    )

    operations = []
    operations.extend(transformations)
    operations.append(grain.Batch(batch_size=batch_size, drop_remainder=drop_remainder))

    loader = grain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=operations,
        worker_count=worker_count,
        shard_options=grain.ShardByJaxProcess(),
        read_options=grain.ReadOptions(num_threads=worker_count),
    )
    return loader
