from __future__ import annotations

from pathlib import Path

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


def test_duplicate_objects_on_one_person_are_merged() -> None:
    from stages.actors.sam2_stage import duplicate_map, relabel

    body = np.zeros((10, 10), bool)
    body[2:9, 3:6] = True
    other = np.zeros((10, 10), bool)
    other[2:9, 7:9] = True
    torso = body.copy()
    torso[6:] = False  # object 3 covers part of person 1 (track restarted on the same body)
    masks = {
        1: {f: body for f in range(6)},
        2: {f: other for f in range(6)},
        3: {f: torso for f in range(2, 6)},
        4: {f: body for f in range(8, 10)},  # same place, never co-visible: kept separate
    }
    mapping = duplicate_map(masks)
    assert mapping == {1: 1, 2: 2, 3: 1, 4: 4}
    labels = np.array([[0, 1, 2, 3, 4]], np.uint8)
    assert relabel(labels, mapping).tolist() == [[0, 1, 2, 1, 3]]


def test_two_people_are_not_chained_through_a_box_covering_both() -> None:
    from stages.actors.sam2_stage import duplicate_map

    x = np.zeros((10, 10), bool)
    x[2:8, 1:4] = True
    y = np.zeros((10, 10), bool)
    y[2:8, 6:9] = True
    both = x | y  # IoU 0.5 with each person
    masks = {
        1: {f: x for f in range(5)},
        2: {f: y for f in range(5)},
        3: {f: both for f in range(5)},
    }
    mapping = duplicate_map(masks)
    assert mapping[1] != mapping[2]


def test_objects_round_trip(tmp_path: Path) -> None:
    from stages.actors.sam2_stage import load_objects, save_objects

    m = np.eye(4, dtype=bool)
    small = {1: {3: m, 7: ~m}, 2: {}}
    save_objects(tmp_path / "o.npz", small)
    back = load_objects(tmp_path / "o.npz")
    assert sorted(back) == [1, 2] and sorted(back[1]) == [3, 7] and back[2] == {}
    assert (back[1][7] == ~m).all()
