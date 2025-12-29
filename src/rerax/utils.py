import math
from typing import NamedTuple 

class TrainingSchedule(NamedTuple):
    total_steps: int
    steps_per_epoch: int
    warmup_steps: int

def calculate_training_schedule(
        dataset_size: int,
        global_batch_size: int,
        num_epochs: int,
        drop_remainder: bool = True,
        warmup_ratio: float = 0.0
) -> TrainingSchedule:
    """
    データセットサイズとエポック数から、必要な総ステップを計算します。
    Args:
        dataset_size: データセットの総サンプル数
        global_batch_size: 全デバイス合計のバッチサイズ
        num_epochs: 学習したいエポック数
        drop_remainder: 最後のバッチを捨てるか (Grain/JAXではTrueが一般的)
        warmup_ratio: ウォームアップに使うステップの割合
    Returns:
        TrainingSchedule: total_steps, steps_per_epoch, warmup_steps を含むNamedTuple
    """
    if drop_remainder:
        steps_per_epoch = dataset_size // global_batch_size
    else:
        steps_per_epoch = math.ceil(dataset_size / global_batch_size)

    if steps_per_epoch == 0:
        raise ValueError(f"Global batch size ({global_batch_size}) i s larger than dataset size ({dataset_size}).")

    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    return TrainingSchedule(
        total_steps=total_steps,
        steps_per_epoch=steps_per_epoch,
        warmup_steps=warmup_steps
    )

