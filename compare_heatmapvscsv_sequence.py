"""
compare_heatmap_vs_csv_sequence.py

Standalone, sequence-level version of compare_heatmap_vs_csv.py: loops over
every .mat file in a folder, runs the same range-Doppler-vs-CSV comparison
on each one, and aggregates the results into a single CSV (one row per
object per frame) plus a summary report (per-class error statistics,
how often objects are flagged significant, how often range bins collide).

This file is self-contained -- it does not import compare_heatmap_vs_csv.py,
so you can run either script independently depending on whether you want
a single frame (compare_heatmap_vs_csv.py) or a whole sequence (this file).
It does import csv_velocity_extractor_standalone.py (for CSV ground truth)
and your actual config.py / radar_processing.py (for the real DSP chain) --
same dependencies as the single-frame script.

Setup
-----
Put this file next to csv_velocity_extractor_standalone.py, and point
PIPELINE_DIR at the folder containing your config.py and radar_processing.py.
Edit MAT_DIR to the folder containing all the .mat files you want to
process (it will process every .mat file in that folder that also has a
matching CSV window available -- frames without CSV coverage are skipped
with a warning, not a crash).

Run:
    python compare_heatmap_vs_csv_sequence.py
"""

import os
import sys
import glob
import csv as csv_module
import statistics

import numpy as np

# ============================================================================
# CONFIG -- edit these and just run the script
# ============================================================================
MAT_DIR  = r"C:\Users\gokul\Downloads\Automotive\Automotive\2019_05_09_mlms003\radar_raw_frame"
CSV_DIR  = r"C:\Users\gokul\Downloads\Automotive\Automotive\2019_05_09_mlms003\text_labels"
WINDOW   = 8          # +/- this many frames of CSVs around each .mat file's frame
FPS      = 30.0       # camera / label frame rate

RANGE_BIN_TOL  = 1     # look +/- this many range bins around the matched bin
CLUTTER_BINS   = 2     # exclude +/- this many bins around 0 m/s (static clutter)
MIN_PEAK_SIGMA = 5.0   # a non-clutter peak must be this many std devs above the
                       # noise floor to count as a real detection

# Folder containing YOUR config.py and radar_processing.py
PIPELINE_DIR = r"C:\Users\gokul\Downloads\project\python"
# Folder containing THIS script and csv_velocity_extractor_standalone.py
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

DETAIL_OUTPUT  = r"all_frames_comparison.csv"    # one row per object per frame
SUMMARY_OUTPUT = r"all_frames_summary.txt"       # aggregated statistics; .txt or .csv

SKIP_ON_ERROR = True   # if True, a frame that fails to process is skipped with
                        # a warning instead of stopping the whole sequence run
# ============================================================================

sys.path.insert(0, THIS_DIR)
sys.path.insert(0, PIPELINE_DIR)

from csv_velocity_extractor import (
    extract_frame_velocities, frame_number_from_path,
)
import config as user_cfg
import radar_processing as rp


# --------------------------------------------------------------------------
# Radar chain (same as compare_heatmap_vs_csv.py; duplicated here so this
# file has no dependency on it, per its "standalone" design)
# --------------------------------------------------------------------------
def get_range_doppler_heatmap(mat_file):
    """Runs YOUR pipeline (radar_processing.py) with YOUR config.py settings
    on mat_file. Returns (rd_map, range_axis, vel_axis) with rd_map shape
    (range_bins, doppler_bins)."""
    raw_frame = rp.load_frame(mat_file, user_cfg.FRAME_MAT_KEY)

    chirp_matrix = rp.select_channel(
        raw_frame,
        tx=user_cfg.SELECTED_TX, rx=user_cfg.SELECTED_RX,
        num_tx=user_cfg.NUM_TX, num_rx=user_cfg.NUM_RX,
        num_chirps=user_cfg.NUM_CHIRPS, num_samples=user_cfg.NUM_ADC_SAMPLES,
    )
    dc_removed = rp.remove_dc_offset(chirp_matrix, enable=user_cfg.DC_REMOVAL_ENABLE)
    windowed = rp.apply_window(
        dc_removed, window_type=user_cfg.RANGE_WINDOW_TYPE, axis=user_cfg.RANGE_FFT_AXIS)
    rfft = rp.range_fft(
        windowed, fft_size=user_cfg.RANGE_FFT_SIZE, axis=user_cfg.RANGE_FFT_AXIS)
    zero_padded = rp.zero_pad_chirps(
        rfft, pad_from=user_cfg.DOPPLER_ZERO_PAD_FROM,
        pad_to=user_cfg.DOPPLER_ZERO_PAD_TO, axis=user_cfg.DOPPLER_FFT_AXIS)
    dfft = rp.doppler_fft(
        zero_padded, fft_size=user_cfg.DOPPLER_FFT_SIZE,
        axis=user_cfg.DOPPLER_FFT_AXIS, window_type=user_cfg.DOPPLER_WINDOW_TYPE)
    shifted = rp.fft_shift(
        dfft, axis=user_cfg.DOPPLER_FFT_AXIS, enable=user_cfg.DOPPLER_FFT_SHIFT)

    magnitude = rp.compute_magnitude(shifted)   # (doppler_bins, range_bins)
    rd_map = magnitude.T                        # -> (range_bins, doppler_bins)

    range_res = user_cfg.LIGHT_SPEED / (
        2.0 * user_cfg.CHIRP_SLOPE_HZ_PER_S * user_cfg.NUM_ADC_SAMPLES / user_cfg.SAMPLE_RATE_HZ)
    range_axis = np.arange(user_cfg.RANGE_FFT_SIZE) * range_res

    wavelength = user_cfg.LIGHT_SPEED / user_cfg.CENTER_FREQ_HZ
    doppler_freqs = np.fft.fftshift(
        np.fft.fftfreq(user_cfg.DOPPLER_FFT_SIZE, d=user_cfg.CHIRP_TIME_S))
    vel_axis = doppler_freqs * wavelength / 2.0

    return rd_map, range_axis, vel_axis


def radar_peak_at_range(rd_map, range_axis, vel_axis, target_range,
                         range_bin_tol=1, clutter_bins=2, min_peak_sigma=5.0):
    """Returns (radar_range_m, radar_velocity, range_bin, is_significant,
    peak_sigma). See compare_heatmap_vs_csv.py for full rationale on the
    significance check (guards against reporting noise-floor fluctuations
    as fake velocity for genuinely stationary objects)."""
    range_res = range_axis[1] - range_axis[0]
    rbin = int(round(target_range / range_res))
    rbin = min(max(rbin, 0), rd_map.shape[0] - 1)
    lo, hi = max(0, rbin - range_bin_tol), min(rd_map.shape[0], rbin + range_bin_tol + 1)

    profile = rd_map[lo:hi, :].sum(axis=0)
    center = len(vel_axis) // 2
    mask = np.ones_like(profile, dtype=bool)
    mask[max(0, center - clutter_bins):center + clutter_bins + 1] = False

    non_clutter = profile[mask]
    noise_mean, noise_std = non_clutter.mean(), non_clutter.std()

    peak_idx = int(np.argmax(profile * mask))
    peak_mag = profile[peak_idx]
    peak_sigma = (peak_mag - noise_mean) / noise_std if noise_std > 0 else 0.0

    is_significant = peak_sigma >= min_peak_sigma
    velocity = float(vel_axis[peak_idx]) if is_significant else 0.0

    return float(range_axis[rbin]), velocity, rbin, is_significant, float(peak_sigma)


def compare_frame(mat_file, csv_dir, window=8, fps=30.0,
                   range_bin_tol=1, clutter_bins=2, min_peak_sigma=5.0):
    """Compares one .mat frame against CSV ground truth. Returns
    (center_frame, rows). Raises if no CSV data covers the frame -- caller
    decides whether to skip or propagate."""
    center_frame = frame_number_from_path(mat_file)

    rd_map, range_axis, vel_axis = get_range_doppler_heatmap(mat_file)
    csv_results = extract_frame_velocities(csv_dir, center_frame, window=window, fps=fps)

    bins_seen = {}
    rows = []
    for r in csv_results:
        radar_range, radar_vel, rbin, is_sig, peak_sigma = radar_peak_at_range(
            rd_map, range_axis, vel_axis, r["range_m"],
            range_bin_tol=range_bin_tol, clutter_bins=clutter_bins,
            min_peak_sigma=min_peak_sigma)
        bins_seen.setdefault(rbin, []).append(r["sub_id"])

        rows.append(dict(
            frame=center_frame,
            sub_id=r["sub_id"], cls=r["cls"],
            csv_range_m=r["range_m"], radar_range_m=radar_range,
            range_bin=rbin, range_error_m=radar_range - r["range_m"],
            csv_v_radial=r["v_radial"], radar_v_radial=radar_vel,
            velocity_error=radar_vel - r["v_radial"],
            radar_significant=is_sig, peak_sigma=peak_sigma,
            n_track_pts=r["n_track_pts"], track_span_s=r["track_span_s"],
        ))

    for row in rows:
        collisions = bins_seen[row["range_bin"]]
        row["bin_collision"] = len(collisions) > 1

    return center_frame, rows


# --------------------------------------------------------------------------
# Sequence loop
# --------------------------------------------------------------------------
def run_sequence(mat_dir, csv_dir, window=8, fps=30.0,
                  range_bin_tol=1, clutter_bins=2, min_peak_sigma=5.0,
                  skip_on_error=True):
    """Runs compare_frame() over every .mat file in mat_dir. Returns a flat
    list of row dicts (each tagged with 'frame'), across all frames."""
    mat_files = sorted(glob.glob(os.path.join(mat_dir, "*.mat")),
                        key=lambda p: frame_number_from_path(p))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {mat_dir}")

    print(f"Found {len(mat_files)} .mat files in {mat_dir}\n")

    all_rows = []
    n_ok, n_skipped = 0, 0
    for mat_file in mat_files:
        frame_num = frame_number_from_path(mat_file)
        try:
            _, rows = compare_frame(
                mat_file, csv_dir, window=window, fps=fps,
                range_bin_tol=range_bin_tol, clutter_bins=clutter_bins,
                min_peak_sigma=min_peak_sigma)
            all_rows.extend(rows)
            n_ok += 1
            print(f"  frame {frame_num:>6d}: {len(rows)} object(s) compared")
        except Exception as e:
            n_skipped += 1
            msg = f"  frame {frame_num:>6d}: SKIPPED ({e})"
            if skip_on_error:
                print(msg)
                continue
            else:
                raise

    print(f"\nProcessed {n_ok} frames, skipped {n_skipped}.")
    return all_rows


# --------------------------------------------------------------------------
# Summary statistics
# --------------------------------------------------------------------------
def summarize(all_rows):
    """Builds a per-class summary: count, mean/median/stdev of range and
    velocity error, fraction flagged significant, fraction with bin
    collisions."""
    by_class = {}
    for r in all_rows:
        by_class.setdefault(r["cls"], []).append(r)

    summary = []
    for cls, rows in sorted(by_class.items()):
        range_errs = [r["range_error_m"] for r in rows]
        vel_errs = [r["velocity_error"] for r in rows]
        n = len(rows)
        n_sig = sum(1 for r in rows if r["radar_significant"])
        n_collision = sum(1 for r in rows if r["bin_collision"])

        summary.append(dict(
            cls=cls, n=n,
            range_error_mean=statistics.mean(range_errs),
            range_error_stdev=statistics.stdev(range_errs) if n > 1 else 0.0,
            vel_error_mean=statistics.mean(vel_errs),
            vel_error_stdev=statistics.stdev(vel_errs) if n > 1 else 0.0,
            vel_error_median=statistics.median(vel_errs),
            frac_significant=n_sig / n,
            frac_bin_collision=n_collision / n,
        ))
    return summary


def print_summary(summary, n_frames):
    print(f"\n{'='*90}")
    print(f"SUMMARY across {n_frames} frame(s)")
    print(f"{'='*90}")
    hdr = (f"{'class':<10} {'n':>5} {'rng err mean':>13} {'rng err std':>12} "
           f"{'vel err mean':>13} {'vel err std':>12} {'vel err med':>12} "
           f"{'%signif':>8} {'%collide':>9}")
    print(hdr)
    print("-" * len(hdr))
    for s in summary:
        print(f"{s['cls']:<10} {s['n']:5d} {s['range_error_mean']:13.3f} "
              f"{s['range_error_stdev']:12.3f} {s['vel_error_mean']:13.3f} "
              f"{s['vel_error_stdev']:12.3f} {s['vel_error_median']:12.3f} "
              f"{s['frac_significant']*100:7.1f}% {s['frac_bin_collision']*100:8.1f}%")


def write_summary_txt(summary, n_frames, path):
    with open(path, "w") as fh:
        fh.write(f"SUMMARY across {n_frames} frame(s)\n")
        hdr = (f"{'class':<10} {'n':>5} {'rng err mean':>13} {'rng err std':>12} "
               f"{'vel err mean':>13} {'vel err std':>12} {'vel err med':>12} "
               f"{'%signif':>8} {'%collide':>9}\n")
        fh.write(hdr)
        fh.write("-" * len(hdr) + "\n")
        for s in summary:
            fh.write(f"{s['cls']:<10} {s['n']:5d} {s['range_error_mean']:13.3f} "
                      f"{s['range_error_stdev']:12.3f} {s['vel_error_mean']:13.3f} "
                      f"{s['vel_error_stdev']:12.3f} {s['vel_error_median']:12.3f} "
                      f"{s['frac_significant']*100:7.1f}% {s['frac_bin_collision']*100:8.1f}%\n")


def write_summary_csv(summary, path):
    fieldnames = ["cls", "n", "range_error_mean", "range_error_stdev",
                  "vel_error_mean", "vel_error_stdev", "vel_error_median",
                  "frac_significant", "frac_bin_collision"]
    with open(path, "w", newline="") as fh:
        writer = csv_module.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for s in summary:
            writer.writerow({k: s[k] for k in fieldnames})


def write_summary(summary, n_frames, path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        write_summary_csv(summary, path)
    elif ext == ".txt":
        write_summary_txt(summary, n_frames, path)
    else:
        raise ValueError(f"Unsupported summary output extension {ext!r}; use .csv or .txt")


# --------------------------------------------------------------------------
# Detail (per-object-per-frame) output
# --------------------------------------------------------------------------
DETAIL_FIELDNAMES = ["frame", "sub_id", "cls", "csv_range_m", "radar_range_m",
                     "range_bin", "range_error_m", "csv_v_radial", "radar_v_radial",
                     "velocity_error", "radar_significant", "peak_sigma",
                     "bin_collision", "n_track_pts", "track_span_s"]


def write_detail_csv(all_rows, path):
    with open(path, "w", newline="") as fh:
        writer = csv_module.DictWriter(fh, fieldnames=DETAIL_FIELDNAMES)
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: r[k] for k in DETAIL_FIELDNAMES})


if __name__ == "__main__":
    all_rows = run_sequence(
        MAT_DIR, CSV_DIR, window=WINDOW, fps=FPS,
        range_bin_tol=RANGE_BIN_TOL, clutter_bins=CLUTTER_BINS,
        min_peak_sigma=MIN_PEAK_SIGMA, skip_on_error=SKIP_ON_ERROR)

    if not all_rows:
        print("No rows produced -- check MAT_DIR/CSV_DIR paths and CSV coverage.")
        sys.exit(1)

    n_frames = len(set(r["frame"] for r in all_rows))
    summary = summarize(all_rows)
    print_summary(summary, n_frames)

    if DETAIL_OUTPUT:
        write_detail_csv(all_rows, DETAIL_OUTPUT)
        print(f"\nWrote per-object detail ({len(all_rows)} rows) to {DETAIL_OUTPUT}")

    if SUMMARY_OUTPUT:
        write_summary(summary, n_frames, SUMMARY_OUTPUT)
        print(f"Wrote summary to {SUMMARY_OUTPUT}")