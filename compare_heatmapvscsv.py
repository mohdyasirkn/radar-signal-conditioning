"""
compare_heatmap_vs_csv.py

Verifies your ADC -> range-Doppler heatmap signal processing chain
(config.py / radar_processing.py) against ground-truth distance & velocity
derived purely from CSV position labels (csv_velocity_extractor_standalone.py).

This calls your ACTUAL pipeline functions from radar_processing.py, using
your ACTUAL config.py settings (channel selection, windowing, zero-padding,
etc) -- it does not reimplement or approximate your DSP chain. It only adds:
  * physically-correct range/velocity axes (derived from config.py's chirp
    constants, using ALL range bins since your ADC data is complex I/Q --
    see note below)
  * a lookup that matches each CSV-labeled object to a range bin and pulls
    the radar's Doppler peak there
  * a side-by-side report of CSV ground truth vs radar-derived values

Note on range bins
-------------------
Your pipeline's range_fft() correctly does not discard any bins (all 128
are computed and returned). Only visualization.py's plotting function cuts
to the first half, which is a real-signal assumption that doesn't apply
here since adcData is complex128 (I/Q already combined) -- for complex
sampling all N range bins are physically valid, not just N/2. This script
uses all 128, matching what your actual FFT already computes; it's only
visualization.py's *plot* that needs a separate fix if you want the plotted
heatmap to show the full range too.

Matching key = RANGE BIN. For each object reported by the CSV extractor at
the target frame:
  1. Convert its CSV range (m) to the nearest range bin index.
  2. Look at the Doppler profile in the heatmap at that range bin (+/- a
     small tolerance), exclude a few bins around 0 m/s (static clutter),
     and take the strongest remaining peak as the radar's velocity
     estimate for that object.
  3. Report both sets of numbers side by side, plus the differences.

Setup
-----
Put this file, csv_velocity_extractor_standalone.py, config.py, and
radar_processing.py so that PIPELINE_DIR (below) points at the folder
containing config.py / radar_processing.py (they can be a different
folder than this script and the CSV extractor, if needed).

Run frame by frame for now (edit MAT_FILE, run). See LOOPING LATER at the
bottom for extending this to a whole sequence.
"""

import os
import sys
import csv as csv_module

import numpy as np

# ============================================================================
# CONFIG -- edit these and just run the script
# ============================================================================
MAT_FILE = r"C:\Users\gokul\Downloads\Automotive\Automotive\2019_05_09_mlms003\radar_raw_frame\000823.mat"
CSV_DIR  = r"C:\Users\gokul\Downloads\Automotive\Automotive\2019_05_09_mlms003\text_labels"
WINDOW   = 8          # +/- this many frames of CSVs around the .mat file's frame
FPS      = 30.0       # camera / label frame rate

RANGE_BIN_TOL = 1     # look +/- this many range bins around the matched bin
CLUTTER_BINS  = 2     # exclude +/- this many bins around 0 m/s (static clutter)
MIN_PEAK_SIGMA = 5.0  # a non-clutter peak must be this many std devs above the
                      # noise floor to count as a real detection (otherwise
                      # treated as "no significant motion" / effectively 0 m/s)

# Folder containing YOUR config.py and radar_processing.py (the actual
# Golden Reference Model files) -- can be the same folder as this script.
PIPELINE_DIR = r"C:\Users\gokul\Downloads\project\python"
# Folder containing THIS script and csv_velocity_extractor_standalone.py
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT = r"heatmap_vs_csv_frame.csv"   # set to None to skip writing a file; .csv or .txt
# ============================================================================

sys.path.insert(0, THIS_DIR)
sys.path.insert(0, PIPELINE_DIR)

from csv_velocity_extractor import (
    extract_frame_velocities, frame_number_from_path,
)
import config as user_cfg
import radar_processing as rp


# --------------------------------------------------------------------------
# Your actual radar processing chain (config.py + radar_processing.py),
# called stage-by-stage exactly like main.py does
# --------------------------------------------------------------------------
def get_range_doppler_heatmap(mat_file):
    """
    Runs YOUR pipeline (radar_processing.py) with YOUR config.py settings
    on mat_file, and returns (rd_map, range_axis, vel_axis):

        rd_map     : 2D array, shape (range_bins, doppler_bins) -- note this
                     is the TRANSPOSE of your pipeline's native
                     (doppler_bins, range_bins) shape, done here only so
                     this comparison script can index it as [range, doppler].
        range_axis : 1D array, length range_bins, meters
        vel_axis   : 1D array, length doppler_bins, m/s
    """
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

    # Use linear magnitude (not dB power) for peak-picking here, regardless
    # of your COMPUTE_POWER/POWER_IN_DB export settings -- summing a small
    # window of adjacent range bins is only physically meaningful in the
    # linear domain, not in dB.
    magnitude = rp.compute_magnitude(shifted)   # (doppler_bins, range_bins)
    rd_map = magnitude.T                        # -> (range_bins, doppler_bins)

    # Physical axes, derived from config.py's (now-correct) chirp constants.
    # All RANGE_FFT_SIZE bins are used -- valid because adcData is complex
    # I/Q, not a real signal (see module docstring).
    range_res = user_cfg.LIGHT_SPEED / (
        2.0 * user_cfg.CHIRP_SLOPE_HZ_PER_S * user_cfg.NUM_ADC_SAMPLES / user_cfg.SAMPLE_RATE_HZ)
    range_axis = np.arange(user_cfg.RANGE_FFT_SIZE) * range_res

    wavelength = user_cfg.LIGHT_SPEED / user_cfg.CENTER_FREQ_HZ
    doppler_freqs = np.fft.fftshift(
        np.fft.fftfreq(user_cfg.DOPPLER_FFT_SIZE, d=user_cfg.CHIRP_TIME_S))
    vel_axis = doppler_freqs * wavelength / 2.0

    return rd_map, range_axis, vel_axis


# --------------------------------------------------------------------------
# Heatmap peak lookup at a given range (matching key = range bin)
# --------------------------------------------------------------------------
def radar_peak_at_range(rd_map, range_axis, vel_axis, target_range,
                         range_bin_tol=1, clutter_bins=2, min_peak_sigma=5.0):
    """Returns (radar_range_m, radar_velocity, range_bin, is_significant,
    peak_sigma) for the strongest non-zero-Doppler peak at the range bin
    nearest target_range.

    min_peak_sigma guards against a specific failure mode: for a genuinely
    stationary object, its true signal sits inside the excluded clutter
    zone, so once that's masked out there's nothing real left outside it --
    argmax will then just return the tallest random noise fluctuation,
    which always looks like a plausible-but-fake velocity. To catch this,
    the "peak" found outside the clutter zone is compared against the
    noise floor statistics of that same (masked) region: if it doesn't
    clear min_peak_sigma standard deviations above the noise floor mean,
    it's flagged is_significant=False and the reported velocity is treated
    as ~0 (no significant non-clutter motion detected) instead of
    whatever noise bin happened to be tallest.

    With ~250 independent Doppler bins, pure noise's own random max
    typically lands around 3-4 sigma just from extreme-value statistics
    (verified empirically on this dataset) -- so min_peak_sigma=5.0 is a
    safety margin above that, not an arbitrary guess.
    """
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


# --------------------------------------------------------------------------
# Put it together
# --------------------------------------------------------------------------
def compare_frame(mat_file, csv_dir, window=8, fps=30.0,
                   range_bin_tol=1, clutter_bins=2, min_peak_sigma=5.0):
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
# Reporting / output
# --------------------------------------------------------------------------
FIELDNAMES = ["sub_id", "cls", "csv_range_m", "radar_range_m", "range_bin",
              "range_error_m", "csv_v_radial", "radar_v_radial",
              "velocity_error", "radar_significant", "peak_sigma",
              "bin_collision", "n_track_pts", "track_span_s"]


def print_report(center_frame, rows):
    hdr = (f"{'uid':>7} {'class':<10} {'CSV rng':>8} {'radar rng':>10} {'bin':>4} "
           f"{'rng err':>8} {'CSV v_rad':>10} {'radar v_rad':>12} {'v err':>7} "
           f"{'sig?':>5} {'sigma':>6} {'coll?':>6}")
    print(f"Frame {center_frame} -- heatmap vs CSV verification")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        coll_flag = "YES" if r["bin_collision"] else ""
        sig_flag = "yes" if r["radar_significant"] else "no"
        print(f"{r['sub_id']:>7} {r['cls']:<10} {r['csv_range_m']:8.2f} "
              f"{r['radar_range_m']:10.2f} {r['range_bin']:4d} {r['range_error_m']:8.2f} "
              f"{r['csv_v_radial']:10.2f} {r['radar_v_radial']:12.2f} "
              f"{r['velocity_error']:7.2f} {sig_flag:>5} {r['peak_sigma']:6.2f} {coll_flag:>6}")
    if any(not r["radar_significant"] for r in rows):
        print("\nNote: 'sig?'=no means no non-clutter peak cleared the significance "
              "threshold -- reported as ~0 m/s (no significant motion detected) rather "
              "than the noise floor's tallest random bump.")
    if any(r["bin_collision"] for r in rows):
        print("\nNote: 'coll?'=YES means two or more CSV objects mapped to the same "
              "range bin -- the radar can't distinguish them by range alone at that "
              "resolution, so the reported radar values may reflect a mix of both.")


def write_csv(rows, path):
    with open(path, "w", newline="") as fh:
        writer = csv_module.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in FIELDNAMES})


def write_txt(rows, path, center_frame):
    with open(path, "w") as fh:
        hdr = (f"{'uid':>7} {'class':<10} {'CSV rng':>8} {'radar rng':>10} {'bin':>4} "
               f"{'rng err':>8} {'CSV v_rad':>10} {'radar v_rad':>12} {'v err':>7} "
               f"{'sig?':>5} {'sigma':>6} {'coll?':>6}\n")
        fh.write(f"Frame {center_frame} -- heatmap vs CSV verification\n")
        fh.write(hdr)
        fh.write("-" * len(hdr) + "\n")
        for r in rows:
            coll_flag = "YES" if r["bin_collision"] else ""
            sig_flag = "yes" if r["radar_significant"] else "no"
            fh.write(f"{r['sub_id']:>7} {r['cls']:<10} {r['csv_range_m']:8.2f} "
                      f"{r['radar_range_m']:10.2f} {r['range_bin']:4d} {r['range_error_m']:8.2f} "
                      f"{r['csv_v_radial']:10.2f} {r['radar_v_radial']:12.2f} "
                      f"{r['velocity_error']:7.2f} {sig_flag:>5} {r['peak_sigma']:6.2f} {coll_flag:>6}\n")


def write_output(rows, path, center_frame):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        write_csv(rows, path)
    elif ext == ".txt":
        write_txt(rows, path, center_frame)
    else:
        raise ValueError(f"Unsupported output extension {ext!r}; use .csv or .txt")


if __name__ == "__main__":
    print(f"Selected .mat file: {MAT_FILE}")
    center_frame, rows = compare_frame(
        MAT_FILE, CSV_DIR, window=WINDOW, fps=FPS,
        range_bin_tol=RANGE_BIN_TOL, clutter_bins=CLUTTER_BINS,
        min_peak_sigma=MIN_PEAK_SIGMA)
    print_report(center_frame, rows)

    if OUTPUT:
        write_output(rows, OUTPUT, center_frame)
        print(f"\nWrote results to {OUTPUT}")

    # -------------------------------------------------------------------
    # LOOPING LATER: once single-frame results look right, wrap this in
    # a loop over multiple .mat files, e.g.:
    #
    #   all_rows = []
    #   for mat_file in glob.glob(os.path.join(MAT_DIR, "*.mat")):
    #       frame, rows = compare_frame(mat_file, CSV_DIR, window=WINDOW, fps=FPS)
    #       for r in rows:
    #           r["frame"] = frame
    #       all_rows.extend(rows)
    #   write_csv(all_rows, "all_frames_comparison.csv")  # add "frame" to FIELDNAMES
    # -------------------------------------------------------------------