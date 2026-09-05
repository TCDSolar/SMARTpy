"""Tests for `smart.tracking` -- the frame-to-frame matcher and the driver.

These use hand-drawn integer label arrays (and, for the driver, lightweight
stand-in maps) so the association semantics are pinned down without any network
fixtures.
"""

import numpy as np
import pytest

import astropy.units as u
from astropy.time import Time

from smart.tracking import (
    APPEARANCE,
    CONTINUATION,
    DISAPPEARANCE,
    MERGE,
    MERGE_END,
    SPLIT,
    FrameMatch,
    TrackState,
    match_frames,
    persistent_name,
    track,
)


class FakeMap:
    """Minimal stand-in for `~sunpy.map.Map`: the driver only touches ``.date``."""

    def __init__(self, date):
        self.date = Time(date)


@pytest.fixture
def identity_derotate(monkeypatch):
    """Replace `derotate_labels` with a no-op so fake maps can be tracked."""
    monkeypatch.setattr("smart.tracking.derotate_labels", lambda prev, prev_map, ref_map: prev)


def block(shape, *regions):
    """Build a label array; each ``region`` is ``(label, rowslice, colslice)``."""
    arr = np.zeros(shape, dtype=int)
    for label, rows, cols in regions:
        arr[rows, cols] = label
    return arr


# ---------------------------------------------------------------------------
# match_frames
# ---------------------------------------------------------------------------


def test_match_frames_one_to_one():
    prev = block((20, 20), (1, slice(2, 10), slice(2, 10)))
    curr = block((20, 20), (1, slice(3, 11), slice(3, 11)))  # same blob, shifted

    m = match_frames(prev, curr)

    assert isinstance(m, FrameMatch)
    assert m.primary == {1: 1}
    assert m.events == {1: CONTINUATION}
    assert m.appeared == []
    assert m.disappeared == []
    assert m.merged == {}
    assert m.split == {}
    assert m.overlap.shape == (1, 1)


def test_match_frames_appearance_and_disappearance():
    prev = block((20, 20), (1, slice(2, 8), slice(2, 8)))
    curr = block((20, 20), (1, slice(12, 18), slice(12, 18)))  # nowhere near prev-1

    m = match_frames(prev, curr)

    assert m.primary == {1: None}
    assert m.events == {1: APPEARANCE}
    assert m.appeared == [1]
    assert m.disappeared == [1]


def test_match_frames_merge():
    prev = block(
        (20, 20),
        (1, slice(2, 10), slice(2, 9)),
        (2, slice(2, 10), slice(11, 18)),
    )
    curr = block((20, 20), (1, slice(2, 11), slice(3, 17)))  # spans both parents

    m = match_frames(prev, curr)

    assert m.events[1] == MERGE
    assert set(m.parents[1]) == {1, 2}
    # one parent continues as the child, the other terminates into it
    assert m.merged[1] == [2] or m.merged[1] == [1]
    assert m.disappeared == []


def test_match_frames_split_largest_keeps_continuation():
    prev = block((30, 30), (5, slice(2, 12), slice(2, 26)))
    curr = block(
        (30, 30),
        (7, slice(2, 12), slice(2, 8)),  # small child
        (8, slice(2, 12), slice(12, 26)),  # large child
    )

    m = match_frames(prev, curr)

    assert set(m.split[5]) == {7, 8}
    assert m.events[8] == CONTINUATION  # larger fragment
    assert m.events[7] == SPLIT  # smaller fragment
    assert m.primary == {7: 5, 8: 5}


def test_match_frames_min_overlap_drops_thin_touch():
    # prev-1 and curr-1 share exactly two pixels.
    prev = block((20, 20), (1, slice(2, 10), slice(2, 10)))
    curr = block((20, 20), (1, slice(9, 15), slice(2, 4)))

    assert match_frames(prev, curr, min_overlap=1).events == {1: CONTINUATION}
    assert match_frames(prev, curr, min_overlap=5).events == {1: APPEARANCE}


def test_match_frames_tie_break_prefers_larger_parent():
    # curr-1 overlaps prev-1 and prev-2 by the same count; prev-2 is larger.
    prev = block(
        (20, 20),
        (1, slice(2, 4), slice(2, 6)),  # 8 px
        (2, slice(10, 18), slice(2, 10)),  # 64 px
    )
    curr = block(
        (20, 20),
        (1, slice(3, 11), slice(4, 8)),  # 2 px into prev-1 (row 3), 2 px into prev-2 (row 10)
    )

    m = match_frames(prev, curr)
    assert m.overlap.shape == (2, 1)
    assert m.primary[1] == 2  # larger parent wins the tie


def test_match_frames_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same shape"):
        match_frames(np.zeros((10, 10), int), np.zeros((10, 12), int))


def test_match_frames_empty_frames():
    m = match_frames(np.zeros((8, 8), int), np.zeros((8, 8), int))
    assert m.primary == {}
    assert m.appeared == []
    assert m.disappeared == []
    assert m.overlap.shape == (0, 0)


# ---------------------------------------------------------------------------
# persistent_name
# ---------------------------------------------------------------------------


def test_persistent_name_format():
    assert persistent_name(Time("2003-11-25T12:00:00"), 7) == "20031125.MG.07"
    assert persistent_name(Time("2011-02-15T00:00:00"), 13) == "20110215.MG.13"


def test_persistent_name_suffix():
    assert persistent_name(Time("2003-11-25T12:00:00"), 11, suffix="a") == "20031125.MG.11a"
    assert persistent_name(Time("2003-11-25T12:00:00"), 11) == "20031125.MG.11"


# ---------------------------------------------------------------------------
# track
# ---------------------------------------------------------------------------


def test_track_persistent_id_and_name(identity_derotate):
    maps = [FakeMap("2011-01-01T00:00:00"), FakeMap("2011-01-01T06:00:00")]
    f0 = block((20, 20), (1, slice(2, 8), slice(2, 8)), (2, slice(2, 8), slice(12, 18)))
    f1 = block((20, 20), (1, slice(2, 9), slice(3, 9)), (2, slice(2, 9), slice(12, 18)))

    rows, state = track(maps, [f0, f1])

    assert isinstance(state, TrackState)
    frame1 = {r["label"]: r for r in rows if r["frame"] == 1}
    assert frame1[1]["track_id"] == 1
    assert frame1[1]["event"] == CONTINUATION
    assert frame1[2]["track_id"] == 2
    assert frame1[1]["name"] == "20110101.MG.01"
    assert frame1[1]["src"] == 1


def test_track_first_frame_none_is_skipped(identity_derotate):
    maps = [FakeMap("2011-01-01T00:00:00"), FakeMap("2011-01-01T06:00:00")]
    f1 = block((20, 20), (1, slice(2, 8), slice(2, 8)))

    rows, _ = track(maps, [None, f1])

    assert {r["frame"] for r in rows} == {1}
    assert rows[0]["event"] == APPEARANCE


def test_track_appearance_and_disappearance_rows(identity_derotate):
    maps = [FakeMap(f"2011-01-01T{h:02d}:00:00") for h in (0, 6, 12)]
    f0 = block((20, 20), (1, slice(2, 8), slice(2, 8)))
    f1 = block((20, 20), (1, slice(2, 8), slice(2, 8)), (2, slice(12, 18), slice(12, 18)))
    f2 = block((20, 20), (1, slice(2, 8), slice(2, 8)))

    rows, _ = track(maps, [f0, f1, f2])

    new = [r for r in rows if r["frame"] == 1 and r["event"] == APPEARANCE]
    assert len(new) == 1
    assert new[0]["label"] == 2
    gone = [r for r in rows if r["event"] == DISAPPEARANCE]
    assert len(gone) == 1
    assert gone[0]["frame"] == 2
    assert gone[0]["label"] is None
    assert gone[0]["track_id"] == new[0]["track_id"]


def test_track_reset_gap_starts_fresh_track(identity_derotate):
    maps = [FakeMap("2011-01-01T00:00:00"), FakeMap("2011-01-03T00:00:00")]  # 48 h apart
    f = block((20, 20), (1, slice(2, 8), slice(2, 8)))

    rows, _ = track(maps, [f, f.copy()], reset_gap=12 * u.hour)

    frame1 = [r for r in rows if r["frame"] == 1][0]
    assert frame1["event"] == APPEARANCE
    assert frame1["track_id"] == 2  # not carried over from frame 0
    assert frame1["name"].startswith("20110103.MG")


def test_track_state_carry_over(identity_derotate):
    f = block((20, 20), (1, slice(2, 8), slice(2, 8)))
    rows_a, state = track([FakeMap("2011-01-01T00:00:00")], [f])
    rows_b, _ = track([FakeMap("2011-01-01T06:00:00")], [f.copy()], state=state)

    assert rows_b[0]["track_id"] == rows_a[0]["track_id"]
    assert rows_b[0]["event"] == CONTINUATION
    assert rows_b[0]["name"] == rows_a[0]["name"]


def test_track_split_suppress_vs_yafta(identity_derotate):
    maps = [FakeMap("2011-01-01T00:00:00"), FakeMap("2011-01-01T06:00:00")]
    f0 = block((30, 30), (1, slice(2, 12), slice(2, 26)))
    f1 = block(
        (30, 30),
        (1, slice(2, 12), slice(2, 8)),  # small fragment
        (2, slice(2, 12), slice(12, 26)),  # large fragment
    )

    suppressed, _ = track(maps, [f0, f1], split_handling="suppress")
    ids = {r["label"]: r["track_id"] for r in suppressed if r["frame"] == 1}
    assert ids[1] == ids[2] == 1  # both fragments stay on the parent track

    yafta, _ = track(maps, [f0, f1], split_handling="yafta")
    ids = {r["label"]: r["track_id"] for r in yafta if r["frame"] == 1}
    assert ids[2] == 1  # larger fragment keeps the parent id
    assert ids[1] != 1  # smaller fragment gets a new track


def test_track_split_suffix_names_fragments(identity_derotate):
    maps = [
        FakeMap("2011-01-01T00:00:00"),
        FakeMap("2011-01-01T06:00:00"),
        FakeMap("2011-01-01T12:00:00"),
    ]
    f0 = block((30, 30), (1, slice(2, 12), slice(2, 26)))
    f1 = block(
        (30, 30),
        (1, slice(2, 12), slice(2, 8)),  # small fragment
        (2, slice(2, 12), slice(12, 26)),  # large fragment -> keeps parent identity
    )
    # Fragment 1 (now its own track) splits again on frame 2.
    f2 = block(
        (30, 30),
        (3, slice(2, 12), slice(2, 4)),
        (4, slice(2, 12), slice(6, 8)),
    )

    rows, state = track(maps, [f0, f1, f2], split_handling="suffix")

    frame1 = {r["label"]: r for r in rows if r["frame"] == 1}
    assert frame1[2]["track_id"] == 1
    assert frame1[2]["name"] == "20110101.MG.01"  # largest fragment: bare parent name
    assert frame1[1]["track_id"] != 1
    assert frame1[1]["name"] == "20110101.MG.01a"  # extra fragment: lettered suffix
    assert frame1[1]["event"] == SPLIT
    assert frame1[1]["src"] == 1

    fragment_a_tid = frame1[1]["track_id"]
    frame2 = {r["label"]: r for r in rows if r["frame"] == 2}
    # Fragment "a" splits again: its own base name grows an extra letter.
    names2 = {r["name"] for r in frame2.values()}
    assert "20110101.MG.01a" in names2  # larger of the two keeps it
    assert "20110101.MG.01aa" in names2  # smaller gets base + another letter
    assert {r["track_id"] for r in frame2.values()} != {fragment_a_tid}  # a fresh id joined

    # Names are stable in state for anything that continues.
    assert state.names[1] == "20110101.MG.01"


def test_track_split_suffix_unique_across_repeated_splits(identity_derotate):
    # Track "01" (the large half) splits again on the next frame; the letter
    # counter is keyed by name, not by local label, so it must move on to "b"
    # rather than reuse "a" (already handed to the first split's small half).
    maps = [FakeMap(f"2011-01-01T{h:02d}:00:00") for h in (0, 2, 4)]
    f0 = block((20, 20), (1, slice(2, 8), slice(2, 18)))
    f1 = block(
        (20, 20),
        (1, slice(2, 8), slice(2, 6)),  # small -> "01a"
        (2, slice(2, 8), slice(9, 18)),  # large -> keeps "01"
    )
    f2 = block(
        (20, 20),
        (1, slice(2, 8), slice(2, 6)),  # unrelated: "01a" continuing, unsplit
        (2, slice(2, 8), slice(9, 14)),  # "01" splits again: large half keeps "01"
        (3, slice(2, 8), slice(15, 18)),  # "01" splits again: small half -> "01b"
    )

    rows, _ = track(maps, [f0, f1, f2], split_handling="suffix")
    names = [r["name"] for r in rows]
    assert names.count("20110101.MG.01a") == 2  # detected in frames 1 and 2, never reused
    assert "20110101.MG.01b" in names


def test_track_merge_emits_merge_end(identity_derotate):
    maps = [FakeMap("2011-01-01T00:00:00"), FakeMap("2011-01-01T06:00:00")]
    f0 = block(
        (20, 20),
        (1, slice(2, 10), slice(2, 9)),
        (2, slice(2, 10), slice(11, 18)),
    )
    f1 = block((20, 20), (1, slice(2, 11), slice(3, 17)))

    rows, _ = track(maps, [f0, f1])

    merge_row = [r for r in rows if r["frame"] == 1 and r["label"] == 1][0]
    assert merge_row["event"] == MERGE
    assert set(merge_row["parents"]) == {1, 2}

    ended = [r for r in rows if r["event"] == MERGE_END]
    assert len(ended) == 1
    assert ended[0]["merged_into"] == merge_row["track_id"]
    assert ended[0]["track_id"] in {1, 2}


def test_track_length_mismatch_raises(identity_derotate):
    with pytest.raises(ValueError, match="same length"):
        track([FakeMap("2011-01-01T00:00:00")], [None, None])


def test_track_bad_split_handling_raises(identity_derotate):
    f = block((20, 20), (1, slice(2, 8), slice(2, 8)))
    with pytest.raises(ValueError, match="split_handling"):
        track([FakeMap("2011-01-01T00:00:00")], [f], split_handling="nonsense")
