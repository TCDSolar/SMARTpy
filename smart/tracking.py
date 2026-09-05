"""
Frame-to-frame tracking of SMART detections.

This is a Python port of the region-tracking stage of the IDL ``smart_library``
(``ar_track_yafta.pro``), which itself drives the overlap matcher from YAFTA
(``match_features_v01.pro``; Welsch & Longcope 2003) and the SMART-specific
fragmentation suppression (``merge_fragments``).  See Higgins et al. (2011),
Section 3.

The identification half of YAFTA (``rankdown``/``create_features``) is not
reproduced here: SMART feeds its *own* indexed grown masks
(`~smart.indexed_grown_mask.index_and_grow_mask`) straight into the matcher, so
the input to this module is a time-ordered sequence of label arrays plus the
maps they were measured on.

Two entry points:

* `match_frames` - associate the labels of one frame with the (already
  de-rotated) labels of the next, classifying appearance / continuation /
  merge / split / disappearance.  Operates on plain arrays so it is easy to
  test.
* `track` - drive `match_frames` over a whole series, carrying a persistent
  per-region identifier and a ``YYYYMMDD.MG.NN`` catalogue name for the
  lifetime of each region, and returning a flat table of per-frame rows.  Takes
  and returns a `TrackState` so a run can be continued when more magnetograms
  arrive.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import astropy.units as u

from sunpy.coordinates import propagate_with_solar_surface
from sunpy.map import Map

__all__ = [
    "FrameMatch",
    "TrackState",
    "derotate_labels",
    "match_frames",
    "persistent_name",
    "track",
]

# Event labels used in `match_frames` output and the `track` table.
APPEARANCE = "appearance"
CONTINUATION = "continuation"
MERGE = "merge"
SPLIT = "split"
DISAPPEARANCE = "disappearance"
MERGE_END = "merge-end"


@dataclass
class FrameMatch:
    """
    Result of associating two consecutive label frames.

    All ids below are the integer label values as they appear in the two input
    arrays (not persistent track ids - `track` layers those on top).

    Attributes
    ----------
    primary : `dict`
        ``current label -> chosen parent label`` (the previous-frame label of
        greatest pixel overlap), or ``None`` when the current feature has no
        overlap with any previous feature.
    parents : `dict`
        ``current label -> list of every overlapping previous label``, ordered
        by decreasing overlap (then decreasing parent area, then id).  The first
        entry equals ``primary[current label]``.
    children : `dict`
        ``previous label -> list of current labels that chose it as primary``.
    events : `dict`
        ``current label -> one of`` `APPEARANCE`, `CONTINUATION`, `MERGE`,
        `SPLIT`.  ``MERGE`` means the current feature has more than one parent;
        ``SPLIT`` means it shares its primary parent with other current
        features and is not the largest of them.
    appeared : `list`
        Current labels with no parent (`APPEARANCE`).
    disappeared : `list`
        Previous labels that are nobody's primary parent and take part in no
        merge (`DISAPPEARANCE`).
    merged : `dict`
        ``current label -> list of secondary parent labels`` that terminate
        into it (`MERGE_END` for those parents).
    split : `dict`
        ``previous label -> list of current labels`` for parents with more than
        one child.
    overlap : `numpy.ndarray`
        ``(n_prev, n_curr)`` array of pixel-overlap counts.
    prev_ids, curr_ids : `numpy.ndarray`
        Sorted unique non-zero labels of each frame; index the rows/columns of
        ``overlap``.
    """

    primary: dict = field(default_factory=dict)
    parents: dict = field(default_factory=dict)
    children: dict = field(default_factory=dict)
    events: dict = field(default_factory=dict)
    appeared: list = field(default_factory=list)
    disappeared: list = field(default_factory=list)
    merged: dict = field(default_factory=dict)
    split: dict = field(default_factory=dict)
    overlap: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int64))
    prev_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    curr_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))


@dataclass
class TrackState:
    """
    Carry-over state between `track` calls, mirroring ``ar_track_yafta``'s ``state``.

    Attributes
    ----------
    prev_map : `~sunpy.map.Map` or `None`
        The last processed frame that held at least one detection.
    prev_labels : `numpy.ndarray` or `None`
        That frame's label array.
    prev_track_ids : `dict`
        ``label in prev_labels -> persistent track id``.
    max_track_id : `int`
        Highest track id issued so far; the next appearance gets
        ``max_track_id + 1``.
    names : `dict`
        ``track id -> "YYYYMMDD.MG.NN"`` catalogue name, fixed at first detection.
    first_seen : `dict`
        ``track id -> ISO date`` of first detection.
    last_time : `~astropy.time.Time` or `None`
        Observation time of ``prev_map``; used with ``reset_gap`` to decide
        whether the next frame can be matched or must start fresh.
    suffix_counts : `dict`
        ``base catalogue name -> number of lettered fragments spawned from it``
        so far, so ``split_handling="suffix"`` hands out unique letters across
        the whole run (and across resumed runs).
    """

    prev_map: Map | None = None
    prev_labels: np.ndarray | None = None
    prev_track_ids: dict = field(default_factory=dict)
    max_track_id: int = 0
    names: dict = field(default_factory=dict)
    first_seen: dict = field(default_factory=dict)
    last_time: object = None
    suffix_counts: dict = field(default_factory=dict)


def persistent_name(date, label, suffix=""):
    """
    Build the fixed ``YYYYMMDD.MG.NN`` catalogue name (Higgins et al. 2011, Section 3.3).

    Parameters
    ----------
    date : `~astropy.time.Time`
        First-detection time of the region.
    label : `int`
        The region's label index in the frame where it was first detected.
    suffix : `str`, optional
        Lowercase letter(s) appended to tell apart fragments that split off a
        region sharing this base name (Higgins et al. 2011, Section 3.1) --
        ``"a"`` gives ``"20031125.MG.11a"``.  Empty (default) for the primary,
        largest feature.

    Returns
    -------
    `str`
    """
    return f"{date.strftime('%Y%m%d')}.MG.{int(label):02d}{suffix}"


_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def _fragment_suffix(n):
    """Spreadsheet-style letters for the ``n``-th split-off fragment: 1->'a', ..., 27->'aa'."""
    letters = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        letters = _ALPHABET[remainder] + letters
    return letters


def derotate_labels(previous_labels, previous_map: Map, reference_map: Map):
    """
    Reproject a label array onto another frame's grid, following solar rotation.

    The previous frame's labels are differentially rotated to the reference
    frame's observation time (as in `~smart.differential_rotation.diff_rotation`)
    so that pixel overlap with the reference frame is physically meaningful.
    Nearest-neighbour resampling is used so label values are preserved.

    Parameters
    ----------
    previous_labels : `numpy.ndarray`
        Integer label array measured on ``previous_map``.
    previous_map : `~sunpy.map.Map`
        Map the labels were measured on.
    reference_map : `~sunpy.map.Map`
        Map whose grid and observation time to reproject onto.

    Returns
    -------
    `numpy.ndarray`
        ``previous_labels`` on ``reference_map``'s grid, same shape and integer
        dtype; pixels that rotate off disk (or off frame) are ``0``.
    """
    label_map = Map(previous_labels.astype(float), previous_map.meta)
    with propagate_with_solar_surface():
        rotated = label_map.reproject_to(reference_map.wcs, algorithm="interpolation", order="nearest-neighbor")
    return np.rint(np.nan_to_num(rotated.data, nan=0.0)).astype(int)


def _label_areas(labels, ids):
    """Pixel count for each id in ``ids`` (a sorted array of non-zero labels)."""
    counts = np.bincount(labels.ravel())
    return np.array([counts[i] if i < counts.size else 0 for i in ids], dtype=np.int64)


def _overlap_counts(previous_labels, current_labels):
    """
    Pixel-overlap matrix between two label arrays of the same shape.

    Returns ``(overlap, prev_ids, curr_ids)`` where ``overlap[i, j]`` is the
    number of pixels labelled ``prev_ids[i]`` in ``previous_labels`` and
    ``curr_ids[j]`` in ``current_labels``.
    """
    prev_ids = np.unique(previous_labels)
    prev_ids = prev_ids[prev_ids > 0]
    curr_ids = np.unique(current_labels)
    curr_ids = curr_ids[curr_ids > 0]

    overlap = np.zeros((prev_ids.size, curr_ids.size), dtype=np.int64)
    both = (previous_labels > 0) & (current_labels > 0)
    if np.any(both) and prev_ids.size and curr_ids.size:
        rows = np.searchsorted(prev_ids, previous_labels[both])
        cols = np.searchsorted(curr_ids, current_labels[both])
        np.add.at(overlap, (rows, cols), 1)
    return overlap, prev_ids, curr_ids


def match_frames(previous_labels, current_labels, *, min_overlap=1):
    """
    Associate the labels of one frame with those of the next.

    Implements the greedy, overlap-driven matching of YAFTA's
    ``match_features_v01`` (Welsch & Longcope 2003) as used by SMART: every
    current feature inherits the previous-frame label it overlaps most, with
    appearances, disappearances, merges and splits classified from the overlap
    matrix.  ``previous_labels`` is expected to already sit on ``current_labels``'
    grid - use `derotate_labels` first.

    Ties on overlap are broken towards the larger previous feature, then the
    lower label value, so the result is deterministic.

    Parameters
    ----------
    previous_labels : `numpy.ndarray`
        Integer label array for the earlier frame, on the same grid as
        ``current_labels`` (0 = background).
    current_labels : `numpy.ndarray`
        Integer label array for the later frame.
    min_overlap : `int`, optional
        Minimum number of shared pixels for an overlap to count as an
        association (default 1).

    Returns
    -------
    `FrameMatch`
    """
    if previous_labels.shape != current_labels.shape:
        raise ValueError(
            f"label arrays must have the same shape, got {previous_labels.shape} "
            f"and {current_labels.shape}; de-rotate with `derotate_labels` first."
        )

    overlap, prev_ids, curr_ids = _overlap_counts(previous_labels, current_labels)
    prev_areas = _label_areas(previous_labels, prev_ids)
    curr_areas = _label_areas(current_labels, curr_ids)
    prev_area = dict(zip(prev_ids.tolist(), prev_areas.tolist()))
    curr_area = dict(zip(curr_ids.tolist(), curr_areas.tolist()))

    match = FrameMatch(overlap=overlap, prev_ids=prev_ids, curr_ids=curr_ids)

    # Ordered parents for every current feature.
    for j, cid in enumerate(curr_ids.tolist()):
        col = overlap[:, j]
        cand = [int(prev_ids[i]) for i in np.nonzero(col >= min_overlap)[0]]
        cand.sort(key=lambda pid, col=col: (-int(col[np.searchsorted(prev_ids, pid)]), -prev_area[pid], pid))
        match.parents[cid] = cand
        match.primary[cid] = cand[0] if cand else None
        if not cand:
            match.appeared.append(cid)

    # Children of every previous feature (by primary parent only).
    for cid, pid in match.primary.items():
        if pid is not None:
            match.children.setdefault(pid, []).append(cid)

    # Splits: a previous feature claimed by more than one current feature.
    for pid, kids in match.children.items():
        if len(kids) > 1:
            match.split[pid] = list(kids)

    # Merges: a current feature with more than one parent; the extra parents
    # terminate into it.
    for cid, cand in match.parents.items():
        if len(cand) > 1:
            match.merged[cid] = cand[1:]

    # Per-current-feature event label.
    for cid in curr_ids.tolist():
        cand = match.parents[cid]
        if not cand:
            match.events[cid] = APPEARANCE
            continue
        pid = cand[0]
        siblings = match.split.get(pid, [])
        if len(cand) > 1:
            match.events[cid] = MERGE
        elif siblings and cid != max(siblings, key=lambda k: curr_area[k]):
            match.events[cid] = SPLIT
        else:
            match.events[cid] = CONTINUATION

    # Disappearances: previous features that continue nowhere and merge into nothing.
    claimed = {p for p in match.primary.values() if p is not None}
    claimed.update(p for extra in match.merged.values() for p in extra)
    match.disappeared = [int(p) for p in prev_ids.tolist() if int(p) not in claimed]

    return match


def track(
    maps,
    labels,
    *,
    min_overlap=1,
    reset_gap: u.Quantity = 12 * u.hour,
    split_handling="suppress",
    state: TrackState | None = None,
):
    """
    Track SMART detections across a series of magnetograms.

    Drives `match_frames` over consecutive frames, giving every region a
    persistent integer ``track_id`` and a fixed ``YYYYMMDD.MG.NN`` name
    (Higgins et al. 2011, Section 3), and recording merge / split / emergence /
    disappearance events.  This mirrors ``ar_track_yafta.pro``.

    Parameters
    ----------
    maps : sequence of `~sunpy.map.Map`
        Processed magnetograms in time order (e.g. the ``threshold_map`` from
        `~smart.processing.smart_prep`).
    labels : sequence of `numpy.ndarray` or `None`
        The indexed grown mask for each map (from
        `~smart.indexed_grown_mask.index_and_grow_mask`), aligned one-to-one
        with ``maps``.  Use ``None`` to skip a frame that has no detection mask
        - in particular the first frame, which has no earlier frame to remove
        transients against.
    min_overlap : `int`, optional
        Minimum shared-pixel count for an association (passed to
        `match_frames`; default 1).
    reset_gap : `~astropy.units.Quantity`, optional
        If the time since the last frame with detections exceeds this, matching
        is skipped and every current feature starts a new track (default 12 h,
        the ``tlastfoundthresh`` of ``ar_track_yafta``).
    split_handling : {"suppress", "yafta", "suffix"}, optional
        What to do when one region splits.

        * ``"suppress"`` (default) - SMART's ``merge_fragments`` behaviour:
          every fragment stays on the parent's track id and name, so an active
          region remains a single track.
        * ``"suffix"`` - Higgins et al. (2011), Section 3.1: the largest
          fragment keeps the parent's name, and every other fragment gets its
          own track id with the parent's name plus a distinguishing lowercase
          letter (``20031125.MG.11`` -> ``20031125.MG.11a``, ``...11b``).
        * ``"yafta"`` - raw YAFTA: the largest fragment keeps the parent id,
          the rest start brand-new tracks with fresh ``YYYYMMDD.MG.NN`` names.
    state : `TrackState`, optional
        Carry-over state from a previous `track` call, to continue tracking as
        new magnetograms arrive.

    Returns
    -------
    rows : `list` of `dict`
        One row per detection per frame, plus a row for each disappearance and
        each merge-end.  Keys:

        ``frame``
            Index into ``maps``.
        ``time``
            Observation time, ISO 8601.
        ``label``
            The region's label in that frame's mask (``None`` for
            disappearance / merge-end rows).
        ``track_id``
            Persistent identifier for the region's lifetime.
        ``name``
            ``YYYYMMDD.MG.NN`` catalogue name, fixed at first detection --
            with a lettered suffix for a non-primary split fragment under
            ``split_handling="suffix"``.
        ``event``
            One of ``"appearance"``, ``"continuation"``, ``"merge"``,
            ``"split"``, ``"disappearance"``, ``"merge-end"``.
        ``src``
            Track id this identity continues from (``None`` for appearances).
        ``parents``
            All contributing parent track ids (length > 1 for a merge).
        ``merged_into``
            For ``"merge-end"`` rows, the track id the region merged into.
    state : `TrackState`
        Updated state; pass back into `track` to continue.
    """
    if len(maps) != len(labels):
        raise ValueError(f"`maps` and `labels` must be the same length, got {len(maps)} and {len(labels)}.")
    if split_handling not in ("suppress", "yafta", "suffix"):
        raise ValueError(f"`split_handling` must be 'suppress', 'yafta' or 'suffix', got {split_handling!r}.")

    st = state if state is not None else TrackState()
    reset_gap = reset_gap.to(u.s)
    rows = []

    def new_id():
        st.max_track_id += 1
        return st.max_track_id

    def register(track_id, date, label, name=None):
        if track_id not in st.names:
            st.names[track_id] = name if name is not None else persistent_name(date, label)
            st.first_seen[track_id] = date.strftime("%Y-%m-%d")

    def fragment_name(base_name):
        """Next unused lettered name off ``base_name`` (``split_handling="suffix"``)."""
        st.suffix_counts[base_name] = st.suffix_counts.get(base_name, 0) + 1
        return base_name + _fragment_suffix(st.suffix_counts[base_name])

    for frame, (this_map, this_labels) in enumerate(zip(maps, labels)):
        if this_labels is None:
            continue
        curr_ids = [int(v) for v in np.unique(this_labels) if v > 0]
        matchable = (
            st.prev_labels is not None
            and st.last_time is not None
            and (this_map.date - st.last_time).to(u.s) <= reset_gap
        )

        assigned: dict[int, int] = {}
        src: dict[int, int | None] = {}
        events: dict[int, str] = {}
        name_override: dict[int, str] = {}
        fm = None

        if not matchable:
            for cid in curr_ids:
                assigned[cid] = new_id()
                src[cid] = None
                events[cid] = APPEARANCE
        else:
            prev_derotated = derotate_labels(st.prev_labels, st.prev_map, this_map)
            fm = match_frames(prev_derotated, this_labels, min_overlap=min_overlap)
            for cid in curr_ids:
                parent = fm.primary[cid]
                if parent is None:
                    assigned[cid] = new_id()
                    src[cid] = None
                    events[cid] = APPEARANCE
                    continue

                parent_tid = st.prev_track_ids.get(int(parent))
                if parent_tid is None:
                    # Shouldn't normally happen: `parent` came from last frame's
                    # tracked labels. Fall back to minting (and naming) it now,
                    # so a "suffix" fragment below still has a base name to work from.
                    parent_tid = new_id()
                    register(parent_tid, this_map.date, cid)
                siblings = fm.split.get(int(parent), [])
                is_largest = not siblings or cid == max(siblings, key=lambda k: int(np.count_nonzero(this_labels == k)))

                if siblings and not is_largest and split_handling in ("yafta", "suffix"):
                    assigned[cid] = new_id()
                    if split_handling == "suffix":
                        name_override[assigned[cid]] = fragment_name(st.names[parent_tid])
                else:
                    assigned[cid] = parent_tid
                src[cid] = parent_tid
                events[cid] = fm.events[cid]

        for cid in curr_ids:
            register(assigned[cid], this_map.date, cid, name=name_override.get(assigned[cid]))
            rows.append(
                {
                    "frame": frame,
                    "time": this_map.date.isot,
                    "label": cid,
                    "track_id": assigned[cid],
                    "name": st.names[assigned[cid]],
                    "event": events[cid],
                    "src": src[cid],
                    "parents": ([st.prev_track_ids.get(int(p)) for p in fm.parents[cid]] if fm is not None else []),
                    "merged_into": None,
                }
            )

        if fm is not None:
            for pid in fm.disappeared:
                tid = st.prev_track_ids.get(int(pid))
                rows.append(
                    {
                        "frame": frame,
                        "time": this_map.date.isot,
                        "label": None,
                        "track_id": tid,
                        "name": st.names.get(tid),
                        "event": DISAPPEARANCE,
                        "src": tid,
                        "parents": [],
                        "merged_into": None,
                    }
                )
            for cid, extra_parents in fm.merged.items():
                for pid in extra_parents:
                    tid = st.prev_track_ids.get(int(pid))
                    rows.append(
                        {
                            "frame": frame,
                            "time": this_map.date.isot,
                            "label": None,
                            "track_id": tid,
                            "name": st.names.get(tid),
                            "event": MERGE_END,
                            "src": tid,
                            "parents": [],
                            "merged_into": assigned[cid],
                        }
                    )

        st.prev_map = this_map
        st.prev_labels = this_labels
        st.prev_track_ids = {cid: assigned[cid] for cid in curr_ids}
        st.last_time = this_map.date

    return rows, st
