from __future__ import annotations

import numpy as np

from stages.actors.sam2_stage import Box, box_iou, link_detections, merge_logits


def test_box_iou() -> None:
    assert box_iou((0, 0, 2, 2), (0, 0, 2, 2)) == 1.0
    assert box_iou((0, 0, 2, 2), (1, 0, 3, 2)) == 1 / 3
    assert box_iou((0, 0, 1, 1), (5, 5, 6, 6)) == 0.0


def test_link_two_walkers_and_drop_one_off_false_positive() -> None:
    dets: dict[int, list[Box]] = {
        f: [(10.0 + f, 10.0, 30.0 + f, 60.0), (100.0 - f, 10.0, 120.0 - f, 60.0)]
        for f in range(0, 40, 5)
    }
    dets[15].append((200.0, 200.0, 210.0, 220.0))  # seen once -> dropped (min_hits 2)
    tracks = link_detections(dets)
    assert len(tracks) == 2
    assert all(len(t.boxes) == 8 for t in tracks)
    assert tracks[0].first == (0, (10.0, 10, 30, 60))


def test_track_ends_after_gap_and_new_one_starts() -> None:
    dets: dict[int, list[Box]] = {
        f: [(0.0, 0.0, 10.0, 10.0)] if f < 10 or f >= 40 else [] for f in range(0, 60, 5)
    }
    tracks = link_detections(dets, max_gap=3)
    assert [sorted(t.boxes) for t in tracks] == [[0, 5], [40, 45, 50, 55]]


def test_merge_logits_highest_positive_wins() -> None:
    a = np.array([[2.0, -1.0], [0.5, -3.0]], np.float32)
    b = np.array([[1.0, 3.0], [0.9, -0.1]], np.float32)
    labels = merge_logits({1: a, 2: b}, (2, 2))
    assert labels.tolist() == [[1, 2], [2, 0]]
    assert merge_logits({}, (2, 2)).sum() == 0
