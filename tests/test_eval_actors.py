from __future__ import annotations

import numpy as np

from eval.actors import mask_metrics


def _stack(frames: int = 4) -> np.ndarray:
    truth = np.zeros((frames, 20, 20), np.uint8)
    truth[:, 2:10, 2:6] = 1
    truth[:, 2:10, 12:16] = 2
    return truth


def test_perfect_masks_with_other_label_numbers() -> None:
    truth = _stack()
    pred = np.where(truth == 1, 7, np.where(truth == 2, 3, 0)).astype(np.uint8)
    m = mask_metrics(pred, truth)
    assert m.mean_iou == 1.0
    assert m.id_switches == 0
    assert m.false_positives_per_frame == 0.0


def test_id_switch_and_miss_and_false_positive() -> None:
    truth = _stack()
    pred = truth.copy()
    pred[2:][truth[2:] == 1] = 5  # actor 1 changes predicted id at frame 2 -> one switch
    pred[3][truth[3] == 2] = 0  # actor 2 missed in the last frame
    pred[0, 15:18, 15:18] = 9  # a spurious track in frame 0
    m = mask_metrics(pred, truth)
    assert m.id_switches == 1
    assert m.mean_iou == 7 / 8  # 8 actor-frames, one of them missed
    assert m.false_positives_per_frame == 1 / 4


def test_low_overlap_is_not_a_match() -> None:
    truth = _stack(1)
    pred = np.zeros_like(truth)
    pred[0, 2:10, 5:9] = 1  # overlaps actor 1 by 1 column of 4 -> IoU 1/7
    m = mask_metrics(pred, truth)
    assert m.mean_iou == 0.0
    assert m.false_positives_per_frame == 1.0
