"""
csv_velocity_extractor_standalone.py

Same purpose as csv_velocity_extractor.py (extract distance + velocity per
object from CSV label files, using a window of frames around a target
.mat file) -- but configured entirely by editing the variables below,
instead of command-line arguments. Just edit the CONFIG block and hit Run.

Workflow:
  1. You point MAT_FILE at the .mat radar cube you're currently analyzing.
  2. The script reads the frame number out of that filename (it does NOT
     open or use the .mat file's contents -- only its name).
  3. It looks in CSV_DIR for CSV files within +/- WINDOW frames of that
     frame number, builds position tracks per object, and computes
     distance (range) and velocity for every object present in the
     .mat file's exact frame.
"""

import csv as csv_module
import glob
import os
import re
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

# ============================================================================
# CONFIG -- edit these and just run the script
# ============================================================================
MAT_FILE = r"C:\Users\gokul\Downloads\Automotive\Automotive\2019_05_09_mlms003\000823.mat"
CSV_DIR  = r"C:\Users\gokul\Downloads\Automotive\Automotive\2019_05_09_mlms003\text_labels"
WINDOW   = 8          # +/- this many frames of CSVs around the .mat file's frame
FPS      = 30.0       # camera / label frame rate
OUTPUT   = r"frame_velocity.csv"
# ============================================================================

CLASS_MAP = {0: "person", 1: "motorbike", 2: "car", 3: "motorbike",
             5: "bus", 7: "truck", 80: "cyclist"}

FRAME_NUM_RE = re.compile(r"(\d+)(?=\.\w+$)")


def frame_number_from_path(path):
    """Extract the trailing integer frame number from a filename, e.g.
    '000823.mat' -> 823, '0000000823.csv' -> 823,
    '1784879250796_0000000823.csv' -> 823 (any prefix is ignored)."""
    m = FRAME_NUM_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Could not parse a frame number from {path!r}")
    return int(m.group(1))


def load_csv_window(csv_dir, center_frame, window):
    """Find CSV files within [center_frame-window, center_frame+window] and
    return {frame_number: [(uid, cls, px, py, wid, length), ...]}."""
    frames = {}
    for path in glob.glob(os.path.join(csv_dir, "*.csv")):
        try:
            fnum = frame_number_from_path(path)
        except ValueError:
            continue
        if abs(fnum - center_frame) > window:
            continue
        rows = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                uid, cls, px, py, wid, length = (float(x) for x in line.split(","))
                rows.append((int(uid), int(cls), px, py, wid, length))
        frames[fnum] = rows
    if not frames:
        raise FileNotFoundError(
            f"No CSVs found within +/-{window} frames of {center_frame} in {csv_dir}")
    return frames


@dataclass
class Track:
    uid: int
    cls: int
    points: list = field(default_factory=list)  # (frame, px, py)


def build_tracks(frames_dict, max_match_dist=2.0):
    """Split duplicate uids (same uid, multiple simultaneous physical
    objects) into separate tracks by frame-to-frame nearest-neighbour
    (Hungarian) matching on position."""
    tracks_by_uid = {}
    for fnum in sorted(frames_dict):
        by_uid = {}
        for uid, cls, px, py, wid, length in frames_dict[fnum]:
            by_uid.setdefault(uid, []).append((cls, px, py))

        for uid, dets in by_uid.items():
            active = tracks_by_uid.setdefault(uid, [])
            open_tracks = [t for t in active if t.points and t.points[-1][0] != fnum]

            if not open_tracks:
                for cls, px, py in dets:
                    active.append(Track(uid=uid, cls=cls, points=[(fnum, px, py)]))
                continue

            cost = np.zeros((len(dets), len(open_tracks)))
            for i, (cls, px, py) in enumerate(dets):
                for j, t in enumerate(open_tracks):
                    _, lpx, lpy = t.points[-1]
                    cost[i, j] = np.hypot(px - lpx, py - lpy)
            row_ind, col_ind = linear_sum_assignment(cost)

            matched_dets, matched_tracks = set(), set()
            for i, j in zip(row_ind, col_ind):
                if cost[i, j] <= max_match_dist:
                    cls, px, py = dets[i]
                    open_tracks[j].points.append((fnum, px, py))
                    open_tracks[j].cls = cls
                    matched_dets.add(i)
                    matched_tracks.add(j)

            for i, (cls, px, py) in enumerate(dets):
                if i not in matched_dets:
                    active.append(Track(uid=uid, cls=cls, points=[(fnum, px, py)]))

    out = []
    for uid, sub_tracks in tracks_by_uid.items():
        for k, t in enumerate(sub_tracks):
            t.sub_id = f"{uid}" if len(sub_tracks) == 1 else f"{uid}.{k}"
            out.append(t)
    return out


def track_velocity(track, fps):
    """Least-squares velocity fit (vx, vy) in m/s from a Track's points."""
    pts = sorted(set(track.points))
    frames = np.array([p[0] for p in pts], dtype=float)
    px = np.array([p[1] for p in pts])
    py = np.array([p[2] for p in pts])
    t = (frames - frames.min()) / fps
    if np.ptp(t) == 0 or len(t) < 2:
        return 0.0, 0.0, len(pts), np.ptp(t)
    vx, _ = np.polyfit(t, px, 1)
    vy, _ = np.polyfit(t, py, 1)
    return vx, vy, len(pts), np.ptp(t)


def extract_frame_velocities(csv_dir, center_frame, window=8, fps=30.0):
    """Returns a list of dicts, one per object present in the CSV at
    center_frame: uid, sub_id, cls, px, py, range_m, vx, vy, speed,
    v_radial, n_track_pts, track_span_s.

    v_radial is the radial (line-of-sight) component of velocity: positive
    means receding (range increasing), negative means approaching -- the
    quantity directly comparable to a radar's Doppler velocity."""
    frames_dict = load_csv_window(csv_dir, center_frame, window)
    if center_frame not in frames_dict:
        raise ValueError(
            f"Frame {center_frame} has no matching CSV in {csv_dir}; "
            f"cannot report positions/velocities for it.")
    tracks = build_tracks(frames_dict)
    center_dets = frames_dict[center_frame]

    results = []
    for uid, cls, px, py, wid, length in center_dets:
        best_track, best_dist = None, np.inf
        for t in tracks:
            if t.uid != uid:
                continue
            for f, tpx, tpy in t.points:
                if f == center_frame:
                    d = np.hypot(px - tpx, py - tpy)
                    if d < best_dist:
                        best_dist, best_track = d, t
        if best_track is None:
            continue

        vx, vy, n_pts, span_s = track_velocity(best_track, fps)
        rng = float(np.hypot(px, py))
        v_radial = (vx * px + vy * py) / rng if rng > 0 else 0.0

        results.append(dict(
            uid=uid, sub_id=best_track.sub_id, cls=CLASS_MAP.get(cls, str(cls)),
            px=px, py=py, range_m=rng,
            vx=vx, vy=vy, speed=float(np.hypot(vx, vy)), v_radial=v_radial,
            n_track_pts=n_pts, track_span_s=span_s,
        ))
    return results


# --------------------------------------------------------------------------
# Reporting / output
# --------------------------------------------------------------------------
FIELDNAMES = ["sub_id", "cls", "px", "py", "range_m",
              "vx", "vy", "speed", "v_radial", "n_track_pts", "track_span_s"]


def print_report(results, center_frame):
    hdr = (f"{'uid':>7} {'class':<10} {'px':>7} {'py':>7} {'range(m)':>9} "
           f"{'vx':>7} {'vy':>7} {'speed':>7} {'v_radial':>9} {'pts/span':>9}")
    print(f"Frame {center_frame} -- CSV-derived distance & velocity")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['sub_id']:>7} {r['cls']:<10} {r['px']:7.2f} {r['py']:7.2f} "
              f"{r['range_m']:9.2f} {r['vx']:7.2f} {r['vy']:7.2f} {r['speed']:7.2f} "
              f"{r['v_radial']:9.2f} {r['n_track_pts']:3d}/{r['track_span_s']:.2f}s")


def write_csv(results, path):
    with open(path, "w", newline="") as fh:
        writer = csv_module.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in FIELDNAMES})


def write_txt(results, path, center_frame):
    with open(path, "w") as fh:
        hdr = (f"{'uid':>7} {'class':<10} {'px':>7} {'py':>7} {'range(m)':>9} "
               f"{'vx':>7} {'vy':>7} {'speed':>7} {'v_radial':>9} {'pts/span':>9}\n")
        fh.write(f"Frame {center_frame} -- CSV-derived distance & velocity\n")
        fh.write(hdr)
        fh.write("-" * len(hdr) + "\n")
        for r in results:
            fh.write(f"{r['sub_id']:>7} {r['cls']:<10} {r['px']:7.2f} {r['py']:7.2f} "
                      f"{r['range_m']:9.2f} {r['vx']:7.2f} {r['vy']:7.2f} {r['speed']:7.2f} "
                      f"{r['v_radial']:9.2f} {r['n_track_pts']:3d}/{r['track_span_s']:.2f}s\n")


def write_output(results, path, center_frame):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        write_csv(results, path)
    elif ext == ".txt":
        write_txt(results, path, center_frame)
    else:
        raise ValueError(f"Unsupported output extension {ext!r}; use .csv or .txt")


if __name__ == "__main__":
    center_frame = frame_number_from_path(MAT_FILE)
    print(f"Selected .mat file: {MAT_FILE}")
    print(f"-> frame number: {center_frame}")
    print(f"Looking for CSVs in: {CSV_DIR}  (window +/-{WINDOW} frames)\n")

    results = extract_frame_velocities(CSV_DIR, center_frame, window=WINDOW, fps=FPS)
    print_report(results, center_frame)

    if OUTPUT:
        write_output(results, OUTPUT, center_frame)
        print(f"\nWrote results to {OUTPUT}")