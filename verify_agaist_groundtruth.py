"""
verify_against_groundtruth.py

Runs the golden reference receive-chain on one or more synthetic ADC
frames and checks the detected peaks against the actual nuScenes
ground-truth radar points for that same sample token.

This is the "does the whole simulate -> RTL-golden-model -> detect" loop
actually recover what nuScenes says was there?" sanity check.

Requires (only on the machine that has the real dataset):
    pip install nuscenes-devkit numpy --break-system-packages

Usage:
    python verify_against_groundtruth.py \
        --dataroot C:\\Users\\gokul\\Downloads\\archive\\v1.0-trainval \
        --version v1.0-trainval \
        --sensor RADAR_FRONT \
        --adc_dir C:\\Users\\gokul\\Downloads\\archive\\v1.0-trainval\\synthetic_adc \
        file1_adc.npy file2_adc.npy

    (omit the .npy filenames to check every *_adc.npy in --adc_dir)

For each frame:
  1. token = filename minus "_adc.npy"
  2. pulls ground-truth targets for that token via
     nuscenes_to_raw_adc.load_targets_for_sample()
  3. runs golden_reference_model.process_frame + find_peaks on the ADC data
  4. greedily matches each ground-truth target to its nearest *unused*
     detected peak (nearest in range_m/velocity_mps, normalized)
  5. reports, per target: range error (m, and range-FFT bins), velocity
     error (m/s, and Doppler-FFT bins), and whether a match was found
     within a configurable tolerance
  6. optionally saves an overlay PNG (ground truth vs detected) using the
     updated plot_range_doppler()
"""

import argparse
import glob
import os
import sys

import numpy as np

import nuscenes_ref_model as grm
from nuscenes_to_raw_adc import load_targets_for_sample, CONFIG as GEN_CFG

# Bin pitch (used to convert physical error back into "how many bins off"),
# derived the same way the golden model derives them.
RANGE_BIN_M = grm.range_bin_to_m(1, grm.CFG) - grm.range_bin_to_m(0, grm.CFG)
DOPPLER_BIN_MPS = grm.doppler_bin_to_mps(grm.CFG["num_chirps"] // 2 + 1, grm.CFG) - \
                  grm.doppler_bin_to_mps(grm.CFG["num_chirps"] // 2, grm.CFG)


def match_targets_to_peaks(targets, peaks, range_tol_m=None, vel_tol_mps=None):
    """Greedy nearest-neighbor match, one peak per target, no reuse.

    range_tol_m / vel_tol_mps: if given, a match farther than this in either
    axis is reported as "no match" rather than forced onto the nearest peak.
    Defaults to 3 range bins / 3 Doppler bins worth of physical distance.
    """
    if range_tol_m is None:
        range_tol_m = 3 * RANGE_BIN_M
    if vel_tol_mps is None:
        vel_tol_mps = 3 * DOPPLER_BIN_MPS

    remaining = list(range(len(peaks)))
    results = []
    for t in targets:
        best_j, best_d = None, None
        for j in remaining:
            p = peaks[j]
            # normalize each axis by its bin size so neither dominates the
            # nearest-neighbor distance just because of unit scale
            dr = (p["range_m"] - t["range_m"]) / RANGE_BIN_M
            dv = (p["velocity_mps"] - t["radial_vel_mps"]) / DOPPLER_BIN_MPS
            d = dr * dr + dv * dv
            if best_d is None or d < best_d:
                best_d, best_j = d, j
        if best_j is None:
            results.append({"target": t, "peak": None})
            continue
        p = peaks[best_j]
        range_err_m = p["range_m"] - t["range_m"]
        vel_err_mps = p["velocity_mps"] - t["radial_vel_mps"]
        within_tol = (abs(range_err_m) <= range_tol_m) and (abs(vel_err_mps) <= vel_tol_mps)
        if within_tol:
            remaining.remove(best_j)
        results.append({
            "target": t,
            "peak": p if within_tol else None,
            "range_err_m": range_err_m,
            "range_err_bins": range_err_m / RANGE_BIN_M,
            "vel_err_mps": vel_err_mps,
            "vel_err_bins": vel_err_mps / DOPPLER_BIN_MPS,
            "within_tol": within_tol,
        })
    return results


def verify_one(nusc, token, adc_path, sensor, n_peaks, save_png_dir=None):
    targets = load_targets_for_sample(nusc, nusc.get("sample", token), sensor=sensor)
    iq = np.load(adc_path)
    power_rd = grm.process_frame(iq, grm.CFG)
    peaks = grm.find_peaks(power_rd, grm.CFG, n_peaks=max(n_peaks, len(targets)))

    matches = match_targets_to_peaks(targets, peaks)

    print(f"\n=== token {token} ({os.path.basename(adc_path)}) ===")
    print(f"ground-truth targets: {len(targets)}   detected peaks searched: {len(peaks)}")
    print(f"{'gt_range_m':>10} {'gt_vel_mps':>10} | {'match':>5} "
          f"{'range_err_m':>11} {'range_err_bin':>13} "
          f"{'vel_err_mps':>11} {'vel_err_bin':>11}")

    n_matched = 0
    for m in matches:
        t = m["target"]
        if m["peak"] is None:
            print(f"{t['range_m']:>10.2f} {t['radial_vel_mps']:>10.2f} | {'NO':>5} "
                  f"{'--':>11} {'--':>13} {'--':>11} {'--':>11}")
            continue
        n_matched += 1
        print(f"{t['range_m']:>10.2f} {t['radial_vel_mps']:>10.2f} | {'yes':>5} "
              f"{m['range_err_m']:>11.2f} {m['range_err_bins']:>13.2f} "
              f"{m['vel_err_mps']:>11.2f} {m['vel_err_bins']:>11.2f}")

    print(f"matched {n_matched}/{len(targets)} ground-truth targets within tolerance "
          f"(<= 3 range bins / <= 3 Doppler bins)")

    if save_png_dir:
        os.makedirs(save_png_dir, exist_ok=True)
        out_png = os.path.join(save_png_dir, f"{token}_overlay.png")
        grm.plot_range_doppler(power_rd, grm.CFG, path=out_png,
                                title=f"{token} - detected vs ground truth",
                                ground_truth=targets, detected_peaks=peaks)
        print(f"  -> saved {out_png}")

    return matches


def main():
    ap = argparse.ArgumentParser(description="Verify golden-model detections against nuScenes ground truth")
    ap.add_argument("--dataroot", required=True,
                     help="Folder containing samples/, sweeps/, maps/, and the version metadata subfolder")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--sensor", default="RADAR_FRONT")
    ap.add_argument("--adc_dir", required=True,
                     help="Directory containing *_adc.npy files from nuscenes_to_raw_adc.py")
    ap.add_argument("adc_files", nargs="*",
                     help="Specific *_adc.npy filenames (relative to --adc_dir). "
                          "If omitted, every *_adc.npy in --adc_dir is checked.")
    ap.add_argument("--n_peaks", type=int, default=8,
                     help="Number of peaks to search for per frame (default: 8)")
    ap.add_argument("--save_png_dir", default=None,
                     help="If set, saves a ground-truth-vs-detected overlay PNG per frame here")
    args = ap.parse_args()

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    if args.adc_files:
        files = [os.path.join(args.adc_dir, f) for f in args.adc_files]
    else:
        files = sorted(glob.glob(os.path.join(args.adc_dir, "*_adc.npy")))

    if not files:
        print("No ADC files found - check --adc_dir / filenames.")
        sys.exit(1)

    total_targets, total_matched = 0, 0
    for f in files:
        base = os.path.basename(f)
        if not base.endswith("_adc.npy"):
            print(f"skipping {base}: doesn't match *_adc.npy naming convention")
            continue
        token = base[: -len("_adc.npy")]
        matches = verify_one(nusc, token, f, args.sensor, args.n_peaks, args.save_png_dir)
        total_targets += len(matches)
        total_matched += sum(1 for m in matches if m["peak"] is not None)

    if total_targets:
        print(f"\n=== overall: matched {total_matched}/{total_targets} "
              f"({100.0 * total_matched / total_targets:.1f}%) ground-truth targets across "
              f"{len(files)} frame(s) ===")


if __name__ == "__main__":
    main()