import numpy as np


def process_sequence(input_ids, max_len):
    final_inputs = np.zeros((max_len,), dtype=int)
    if len(input_ids) < max_len:
        final_inputs[:len(input_ids)] = input_ids
    elif len(input_ids) > max_len:
        final_inputs[:] = input_ids[-max_len:]
    else:
        final_inputs = input_ids
    final_labels = np.zeros((max_len,), dtype=int)
    # シーケンス長の最後の場合、次の予測値は存在しないので0にする
    final_labels[:max_len-1] = final_inputs[1:]
    mask = final_labels != 0
    return final_inputs, final_labels, mask
