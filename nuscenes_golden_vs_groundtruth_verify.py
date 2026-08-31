"""
verify_against_groundtruth.py

Verifies that golden_reference_model.py's range-Doppler heatmap correctly
places energy at the locations the nuScenes ground truth says it should,
for each synthetic ADC frame produced by nuscenes_to_raw_adc.py.

METHODOLOGY (see conversation for rationale):
  This is a LOCAL, ground-truth-guided check, not blind detection. For each
  known target (range_m, radial_vel_mps) from nuScenes, we look only at a
  small window around its *expected* bin location in the heatmap and find
  the local peak there, plus a local SNR estimate from the surrounding
  annulus. This directly tests heatmap correctness (right place, right
  shape, right magnitude) without inheriting detection-algorithm artifacts
  like CFAR target-masking in dense scenes - that's the downstream CFAR
  team's problem to solve on the heatmap you deliver them, not something
  this verification step needs to solve itself.

  A secondary, OPTIONAL global CFAR pass (--cfar_diagnostic) is included
  purely to flag energy that shows up somewhere NOT explained by any
  ground-truth target - useful for catching generator bugs (leakage,
  artifacts) but not the primary correctness metric.

Run this on the machine that has nuscenes-devkit + the dataset installed.

Usage:
    python verify_against_groundtruth.py \
        --dataroot "C:\\Users\\gokul\\Downloads\\archive\\v1.0-trainval" \
        --version v1.0-trainval \
        --adc_dir "C:\\Users\\gokul\\Downloads\\archive\\v1.0-trainval\\synthetic_adc" \
        --outdir  "C:\\Users\\gokul\\Downloads\\project\\python\\radar_golden_model\\verify_out"

Expects nuscenes_to_raw_adc.py and golden_reference_model.py importable
(same folder or on PYTHONPATH).
"""

import argparse
import glob
import os

import numpy as np

import nuscenes_ref_model as grm
import nuscenes_to_raw_adc as gen


def verify_one(token, npy_path, nusc, sensor="RADAR_FRONT", cfg=grm.CFG,
                snr_min_db=6.0, outdir=None, cfar_diagnostic=False, save_output=False):
    sample = nusc.get("sample", token)
    targets = gen.load_targets_for_sample(nusc, sample, sensor=sensor)

    output = grm.run_pipeline(npy_path, cfg, frame_id=token)
    power_rd = output.power
    clip_frac = output.metadata["adc_saturation_fraction"]

    # ---- PRIMARY: local ground-truth-guided verification ----
    results = grm.verify_targets_local(power_rd, targets, cfg)

    print(f"\n=== {token} ({npy_path}) ===")
    print(f"ADC saturation: {clip_frac*100:.2f}%   |  ground-truth targets: {len(targets)}")
    print(f"{'gt_range_m':>10} {'gt_vel_mps':>10} | {'det_range_m':>11} {'det_vel_mps':>11} "
          f"| {'range_err_m':>11} {'vel_err_mps':>11} | {'snr_db':>7}")

    weak = []
    for r in results:
        if not r["found"]:
            print(f"{r['gt_range_m']:>10.2f} {r['gt_vel_mps']:>10.2f} | "
                  f"(target expected location outside valid FFT span - {r['reason']})")
            continue
        flag = "  <- LOW SNR" if r["snr_db"] < snr_min_db else ""
        print(f"{r['gt_range_m']:>10.2f} {r['gt_vel_mps']:>10.2f} | "
              f"{r['det_range_m']:>11.3f} {r['det_vel_mps']:>11.3f} | "
              f"{r['range_err_m']:>11.4f} {r['vel_err_mps']:>11.4f} | "
              f"{r['snr_db']:>7.1f}{flag}")
        if r["snr_db"] < snr_min_db:
            weak.append(r)

    if weak:
        print(f"{len(weak)}/{len(results)} targets have local SNR below {snr_min_db} dB "
              f"(present in the map, but weak relative to their local neighborhood)")

    # ---- SECONDARY (optional): global CFAR diagnostic for unexplained energy ----
    if cfar_diagnostic:
        cfar_peaks = grm.find_peaks(power_rd, cfg)
        unexplained = []
        for p in cfar_peaks:
            near_gt = any(abs(p["range_m"] - t["range_m"]) < 1.0
                           and abs(p["velocity_mps"] - t["radial_vel_mps"]) < 1.0
                           for t in targets)
            if not near_gt:
                unexplained.append(p)
        if unexplained:
            print(f"CFAR diagnostic: {len(unexplained)} detections not explained by any "
                  f"ground-truth target (possible generator artifact) - "
                  f"{[(round(p['range_m'],2), round(p['velocity_mps'],2)) for p in unexplained[:15]]}"
                  f"{' ...' if len(unexplained) > 15 else ''}")

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        png = os.path.join(outdir, f"{token}_rdmap.png")
        gt_points = [(t["range_m"], t["radial_vel_mps"]) for t in targets]
        grm.plot_range_doppler(power_rd, cfg, path=png, title=token, gt_points=gt_points)
        if save_output:
            npz = os.path.join(outdir, f"{token}_output.npz")
            grm.save_frame_output(output, npz)

    found = [r for r in results if r["found"]]
    return {
        "token": token,
        "n_targets": len(targets),
        "n_found": len(found),
        "n_outside_span": len(results) - len(found),
        "n_low_snr": len(weak),
        "range_errs": [r["range_err_m"] for r in found],
        "vel_errs": [r["vel_err_mps"] for r in found],
        "snr_dbs": [r["snr_db"] for r in found],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--sensor", default="RADAR_FRONT")
    ap.add_argument("--adc_dir", required=True, help="Folder containing the *_adc.npy files")
    ap.add_argument("--outdir", default=None, help="Folder to save per-sample RD map PNGs")
    ap.add_argument("--snr_min_db", type=float, default=6.0,
                     help="Local SNR (dB) below which a found target is flagged as weak")
    ap.add_argument("--cfar_diagnostic", action="store_true",
                     help="Also run a global CFAR pass to flag energy not explained by any ground truth")
    ap.add_argument("--save_output", action="store_true",
                     help="Also save the full structured output (range FFT, Doppler FFT, power, "
                          "metadata) as a .npz per token, for handoff to the CFAR detection stage")
    args = ap.parse_args()

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    npy_files = sorted(glob.glob(os.path.join(args.adc_dir, "*_adc.npy")))
    if not npy_files:
        print(f"No *_adc.npy files found in {args.adc_dir}")
        return

    results = []
    for f in npy_files:
        token = os.path.basename(f).replace("_adc.npy", "")
        try:
            results.append(verify_one(token, f, nusc, args.sensor,
                                       snr_min_db=args.snr_min_db,
                                       outdir=args.outdir,
                                       cfar_diagnostic=args.cfar_diagnostic,
                                       save_output=args.save_output))
        except Exception as e:
            print(f"\n=== {token} ({f}) ===\nERROR: {e}")

    all_range_errs = [e for r in results for e in r["range_errs"]]
    all_vel_errs = [e for r in results for e in r["vel_errs"]]
    all_snrs = [s for r in results for s in r["snr_dbs"]]
    total_targets = sum(r["n_targets"] for r in results)
    total_found = sum(r["n_found"] for r in results)
    total_outside = sum(r["n_outside_span"] for r in results)
    total_low_snr = sum(r["n_low_snr"] for r in results)

    print("\n" + "=" * 70)
    print("SUMMARY (primary metric: heatmap correctness at known target locations)")
    print("=" * 70)
    print(f"Frames processed:            {len(results)}")
    print(f"Ground-truth targets:        {total_targets}")
    print(f"Within valid FFT span:       {total_found}  ({100*total_found/max(total_targets,1):.1f}%)")
    print(f"Outside valid span (n/a):    {total_outside}")
    print(f"Low local SNR (< {args.snr_min_db} dB):   {total_low_snr}")
    if all_range_errs:
        print(f"Range error  (m):   mean={np.mean(all_range_errs):.4f}  "
              f"std={np.std(all_range_errs):.4f}  max={np.max(all_range_errs):.4f}")
    if all_vel_errs:
        print(f"Velocity err (m/s): mean={np.mean(all_vel_errs):.4f}  "
              f"std={np.std(all_vel_errs):.4f}  max={np.max(all_vel_errs):.4f}")
    if all_snrs:
        print(f"Local SNR (dB):     mean={np.mean(all_snrs):.1f}  min={np.min(all_snrs):.1f}")

    N, M = grm.CFG["range_fft_len"], grm.CFG["num_chirps"]
    range_bin_m = grm.range_bin_to_m(1, grm.CFG) - grm.range_bin_to_m(0, grm.CFG)
    vel_bin_mps = grm.doppler_bin_to_mps(1, grm.CFG) - grm.doppler_bin_to_mps(0, grm.CFG)
    print(f"\n(For reference: 1 range bin = {range_bin_m:.3f} m, 1 Doppler bin = {vel_bin_mps:.3f} m/s. "
          f"Mean error should sit well under one bin for a correct pipeline.)")


if __name__ == "__main__":
    main()