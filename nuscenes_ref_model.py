"""
golden_reference_model.py

Golden reference receive-chain for the synthetic nuScenes->raw-ADC frames
produced by nuscenes_to_raw_adc.py.

Matches the project's documented I/O contract:

    INPUT:
      - Digitized radar data (raw ADC I/Q)
      - Fixed-point I/Q samples (Q1.(adc_bits-1) format, signed integer codes)
      - Streaming or DMA-based input from DDR (a flat binary buffer with a
        small header, as load_iq_stream()/save_iq_ddr_buffer() read/write)

    OUTPUT (see RadarFrameOutput / save_frame_output()):
      - Range FFT output (frequency-domain data)
      - Doppler FFT output (range-Doppler map)
      - Magnitude / power values per bin
      - Metadata for the detection stage (frame markers, scaling info)

Pipeline (mirrors radar_dc_remove -> radar_window -> radar_fft_range ->
radar_fft_doppler -> radar_mag_power in the RTL spec):

    raw ADC I/Q [chirps, range_samples, rx]  (fixed-point, DDR-sourced)
        -> DC offset removal (per-chirp mean subtraction)
        -> window (Hann, symmetric by default)
        -> range FFT (per chirp, per rx)                 -> Range FFT output
        -> Doppler FFT (across chirps, fftshift'd)        -> Doppler FFT output
        -> power = Re^2 + Im^2                            -> magnitude/power output
        -> non-coherent RX combine (sum power across antennas) -> the heatmap
    Everything above, plus frame metadata, is packaged by run_pipeline()
    into a RadarFrameOutput ready to hand to the CFAR detection stage.

Usage:
    python golden_reference_model.py file1_adc.npy [file2_adc.npy ...]

Ground-truth cross-check (run locally, where nuscenes-devkit + the actual
dataset are available):

    from nuscenes.nuscenes import NuScenes
    from nuscenes_to_raw_adc import load_targets_for_sample
    nusc = NuScenes(version='v1.0-trainval', dataroot=r"...", verbose=False)
    token = "7ff1f84a3c184434a366282773ca9572"   # filename minus "_adc.npy"
    sample = nusc.get('sample', token)
    targets = load_targets_for_sample(nusc, sample, sensor="RADAR_FRONT")
    for t in targets:
        print(t)   # compare range_m / radial_vel_mps against detected peaks below
"""

import sys
import glob
import json
import struct
import time
from dataclasses import dataclass, field, asdict
import numpy as np

# --------------------------------------------------------------------------
# Config - MUST match the CONFIG dict used to generate the ADC frames
# --------------------------------------------------------------------------
CFG = {
    "range_fft_len": 256,
    "num_chirps": 128,
    "num_rx": 4,
    "window_type": "hann",       # "hann" or "hamming"
    "window_periodic": False,    # False = matches generator's own reference_range_doppler()
                                  # True  = matches the RTL golden model's LUT-friendly window
    "doppler_window": True,      # apply Hann on the Doppler axis too (see process_frame)
    "fc_hz": 77.0e9,
    "bandwidth_hz": 500.0e6,
    "chirp_duration_s": 25.6e-6,
    "adc_bits": 16,
}
C = 299_792_458.0

# --------------------------------------------------------------------------
# INPUT: fixed-point I/Q samples, digitized radar data, DDR/streaming buffer
# --------------------------------------------------------------------------
# Fixed-point convention: signed Q1.(bits-1), i.e. an integer ADC code in
# [-(2^(bits-1)-1), 2^(bits-1)-1] represents a normalized value in [-1, 1).
# This matches quantize_adc() in nuscenes_to_raw_adc.py.

def fixed_to_float(code, bits=16):
    """Fixed-point signed integer code -> normalized float in [-1, 1)."""
    full_scale = 2 ** (bits - 1) - 1
    return np.asarray(code).astype(np.float64) / full_scale


def float_to_fixed(x, bits=16):
    """Normalized float in [-1, 1) -> fixed-point signed integer code (clipped)."""
    full_scale = 2 ** (bits - 1) - 1
    xc = np.clip(np.asarray(x), -1.0, 1.0)
    return np.round(xc * full_scale).astype(np.int32)


_DDR_MAGIC = b"RADQIQ01"  # 8-byte magic + version, identifies the buffer format


def save_iq_ddr_buffer(iq_frame, path, cfg=CFG, frame_id=0):
    """
    Serialize a complex I/Q frame to a flat binary buffer shaped like a
    DMA burst read from DDR: fixed header (dimensions, bit width, frame id)
    followed by interleaved int16 I,Q samples in row-major
    [chirp, range_sample, rx] order - the same order a streaming AXI/DMA
    front-end would deliver samples to the signal-conditioning pipeline in.
    """
    M, N, R = iq_frame.shape
    bits = cfg["adc_bits"]
    codes = np.round(np.stack([iq_frame.real, iq_frame.imag], axis=-1)).astype(np.int16)
    with open(path, "wb") as f:
        f.write(_DDR_MAGIC)
        f.write(struct.pack("<IIIII", M, N, R, bits, frame_id))
        f.write(codes.tobytes(order="C"))
    return path


def load_iq_ddr_buffer(path):
    """Inverse of save_iq_ddr_buffer(): DDR-style binary buffer -> (complex iq_frame, frame_id)."""
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != _DDR_MAGIC:
            raise ValueError(f"{path}: not a recognized DDR I/Q buffer (bad magic)")
        M, N, R, bits, frame_id = struct.unpack("<IIIII", f.read(20))
        raw = np.frombuffer(f.read(), dtype=np.int16)
    codes = raw.reshape(M, N, R, 2)
    iq_frame = codes[..., 0].astype(np.float64) + 1j * codes[..., 1].astype(np.float64)
    return iq_frame, frame_id


def load_iq_stream(source, cfg=CFG):
    """
    Unified loader for 'digitized radar data' input: accepts either a
    convenience .npy array (dev/test) or the DDR-style .bin buffer format
    (production-shaped streaming/DMA input). Returns (iq_frame, frame_id).
    """
    if isinstance(source, str) and source.endswith(".bin"):
        return load_iq_ddr_buffer(source)
    iq_frame = np.load(source) if isinstance(source, str) else source
    return iq_frame, None


def _window(n, kind, periodic):
    if kind == "hann":
        return np.hanning(n + 1)[:-1] if periodic else np.hanning(n)
    else:
        return np.hamming(n + 1)[:-1] if periodic else np.hamming(n)


def _pipeline_stages(iq_frame, cfg=CFG):
    """Shared core: DC removal -> window -> range FFT -> Doppler window -> Doppler FFT -> power.
    Returns (range_fft, range_doppler_fft, power), each [chirps, range_bins, rx] except
    power is real-valued. Used by both process_frame() (backward-compat) and run_pipeline()."""
    x = iq_frame.astype(np.complex128)

    # 1. DC offset removal: per-chirp mean subtraction along fast-time (range) axis
    x = x - x.mean(axis=1, keepdims=True)

    # 2. Windowing (fast-time / range axis)
    win_r = _window(cfg["range_fft_len"], cfg["window_type"], cfg["window_periodic"])
    x = x * win_r[None, :, None]

    # 3. Range FFT (per chirp, per rx) -> "Range FFT output (frequency-domain data)"
    range_fft = np.fft.fft(x, n=cfg["range_fft_len"], axis=1)

    # 4. Doppler-axis windowing (slow-time). Without this, a rectangular window
    #    produces a slowly-decaying sinc sidelobe comb along the Doppler axis for
    #    every target - which, for near-zero-velocity targets in particular, can
    #    masquerade as extra "detections" at regular Doppler-bin-spaced offsets.
    #    NOTE: your RTL spec only windows the range axis (Doppler FFT is applied
    #    directly, no ROM window stage) - set cfg["doppler_window"]=False for a
    #    bit-exact comparison against the RTL/ASIC golden model.
    range_fft_for_doppler = range_fft
    if cfg.get("doppler_window", True):
        win_d = _window(cfg["num_chirps"], cfg["window_type"], cfg["window_periodic"])
        range_fft_for_doppler = range_fft * win_d[:, None, None]

    # 5. Doppler FFT (across chirps), fftshift to center zero-Doppler
    #    -> "Doppler FFT output (range-Doppler map)"
    range_doppler = np.fft.fftshift(np.fft.fft(range_fft_for_doppler, n=cfg["num_chirps"], axis=0), axes=0)

    # 6. Magnitude / power -> "Magnitude / power values per bin"
    power = range_doppler.real**2 + range_doppler.imag**2
    return range_fft, range_doppler, power


def process_frame(iq_frame, cfg=CFG):
    """Backward-compatible entry point: iq_frame -> power RD map [chirps, range_bins, rx]."""
    _, _, power = _pipeline_stages(iq_frame, cfg)
    return power


@dataclass
class RadarFrameOutput:
    """The four documented pipeline outputs, packaged for handoff to the CFAR detection stage."""
    range_fft: np.ndarray               # complex [chirps, range_bins, rx] - frequency-domain data
    range_doppler_fft: np.ndarray       # complex [chirps, range_bins, rx] - range-Doppler map (pre-magnitude)
    power: np.ndarray                   # real    [chirps, range_bins, rx] - magnitude/power per bin, per rx
    power_combined: np.ndarray          # real    [chirps, range_bins]     - non-coherent RX-combined heatmap
    metadata: dict = field(default_factory=dict)  # frame markers + scaling info

    def to_dict_for_save(self):
        d = asdict(self)
        d["metadata"] = json.dumps(self.metadata)
        return d


def run_pipeline(source, cfg=CFG, frame_id=None):
    """
    Full documented pipeline: digitized/fixed-point I/Q input -> RadarFrameOutput.

    `source` may be a path to a .npy file, a path to a DDR-style .bin buffer
    (see save_iq_ddr_buffer/load_iq_ddr_buffer), or an already-loaded complex
    ndarray. Fixed-point ADC codes are used directly (their integer scale
    cancels out through the linear FFT/window stages, so absolute power is
    reported in raw fixed-point-code^2 units - see metadata for how to
    rescale to physical units if needed).
    """
    iq_frame, loaded_frame_id = load_iq_stream(source, cfg)
    if frame_id is None:
        frame_id = loaded_frame_id

    clip_frac = float(np.mean((np.abs(iq_frame.real) >= 2 ** (cfg["adc_bits"] - 1) - 1) |
                               (np.abs(iq_frame.imag) >= 2 ** (cfg["adc_bits"] - 1) - 1)))

    range_fft, range_doppler_fft, power = _pipeline_stages(iq_frame, cfg)
    power_combined = power.sum(axis=2)

    range_bin_m = range_bin_to_m(1, cfg) - range_bin_to_m(0, cfg)
    vel_bin_mps = doppler_bin_to_mps(1, cfg) - doppler_bin_to_mps(0, cfg)

    metadata = {
        "frame_id": frame_id,                       # frame marker
        "timestamp_unix": time.time(),               # frame marker
        "num_chirps": cfg["num_chirps"],
        "range_fft_len": cfg["range_fft_len"],
        "num_rx": cfg["num_rx"],
        "adc_bits": cfg["adc_bits"],                 # scaling info: fixed-point format = Q1.(adc_bits-1)
        "fixed_point_format": f"Q1.{cfg['adc_bits'] - 1}",
        "window_type": cfg["window_type"],
        "window_periodic": cfg["window_periodic"],
        "doppler_window_applied": bool(cfg.get("doppler_window", True)),
        "range_bin_size_m": range_bin_m,              # scaling info: bin -> physical units
        "velocity_bin_size_mps": vel_bin_mps,          # scaling info: bin -> physical units
        "max_unambiguous_range_m": range_bin_to_m(cfg["range_fft_len"], cfg),
        "adc_saturation_fraction": clip_frac,          # scaling info: how much headroom was used
    }

    return RadarFrameOutput(
        range_fft=range_fft,
        range_doppler_fft=range_doppler_fft,
        power=power,
        power_combined=power_combined,
        metadata=metadata,
    )


def save_frame_output(output: RadarFrameOutput, path):
    """Serialize the four pipeline outputs + metadata to a single .npz for handoff to the CFAR team."""
    np.savez_compressed(
        path,
        range_fft=output.range_fft,
        range_doppler_fft=output.range_doppler_fft,
        power=output.power,
        power_combined=output.power_combined,
        metadata_json=json.dumps(output.metadata),
    )
    return path


def load_frame_output(path) -> RadarFrameOutput:
    """Inverse of save_frame_output()."""
    with np.load(path, allow_pickle=False) as z:
        return RadarFrameOutput(
            range_fft=z["range_fft"],
            range_doppler_fft=z["range_doppler_fft"],
            power=z["power"],
            power_combined=z["power_combined"],
            metadata=json.loads(str(z["metadata_json"])),
        )


def range_bin_to_m(k, cfg=CFG):
    """k: range FFT bin index (0..N-1). All N bins are valid here (this is complex
    I/Q data, not a real-valued signal, so there's no Nyquist-mirroring to discard -
    aliasing instead happens beyond bin N, i.e. beyond ~fb=Fs)."""
    N = cfg["range_fft_len"]
    Fs = N / cfg["chirp_duration_s"]
    S = cfg["bandwidth_hz"] / cfg["chirp_duration_s"]
    fb = k * Fs / N
    return fb * C / (2 * S)


def doppler_bin_to_mps(kd, cfg=CFG):
    """kd: Doppler FFT bin index AFTER fftshift, centered so kd=M/2 is zero-Doppler."""
    M = cfg["num_chirps"]
    fc = cfg["fc_hz"]
    Tc = cfg["chirp_duration_s"]
    kd_centered = kd - M // 2
    f_d_norm = kd_centered / M          # cycles per chirp
    return f_d_norm * C / (2 * fc * Tc)


def range_m_to_bin(r_m, cfg=CFG):
    """Inverse of range_bin_to_m: physical range -> nearest range FFT bin (float, unrounded)."""
    N = cfg["range_fft_len"]
    Fs = N / cfg["chirp_duration_s"]
    S = cfg["bandwidth_hz"] / cfg["chirp_duration_s"]
    fb = r_m * 2 * S / C
    return fb * N / Fs


def mps_to_doppler_bin(v_mps, cfg=CFG):
    """Inverse of doppler_bin_to_mps: physical velocity -> nearest Doppler FFT bin (float, unrounded, post-fftshift)."""
    M = cfg["num_chirps"]
    fc = cfg["fc_hz"]
    Tc = cfg["chirp_duration_s"]
    f_d_norm = v_mps * 2 * fc * Tc / C
    return f_d_norm * M + M // 2


def verify_targets_local(power_rd, targets, cfg=CFG, search_bins=(2, 2), noise_ring=2,
                          range_key="range_m", vel_key="radial_vel_mps"):
    """
    Ground-truth-guided verification: for each known target, look ONLY at a
    small window around its expected (range_bin, doppler_bin) and find the
    local peak there - rather than asking a detector to blindly rediscover
    all targets in the map (which is what CFAR does, and what breaks down
    in dense scenes due to target masking). This directly answers "is the
    heatmap's energy correctly placed and shaped at each known location?"
    without inheriting detection-algorithm artifacts.

    search_bins: (doppler, range) half-width of the search window around the
                 expected bin, to absorb FFT bin-center quantization.
    noise_ring:  extra bins beyond search_bins used to estimate a local noise
                 floor (for a local SNR estimate), without any global CFAR pass.

    Returns one result dict per target with expected/found range & velocity,
    errors, and local SNR in dB.
    """
    combined = power_rd.sum(axis=2)  # [chirps, range_bins]
    M, N = combined.shape
    valid_r = N  # full range axis is valid for complex I/Q data (see range_bin_to_m)
    sd, sr = search_bins

    results = []
    for t in targets:
        r_gt = t[range_key]
        v_gt = t[vel_key]
        kr_f = range_m_to_bin(r_gt, cfg)
        kd_f = mps_to_doppler_bin(v_gt, cfg)
        kr0, kd0 = int(round(kr_f)), int(round(kd_f))

        r0s, r1s = max(0, kr0 - sr), min(valid_r, kr0 + sr + 1)
        d0s, d1s = max(0, kd0 - sd), min(M, kd0 + sd + 1)
        r0n, r1n = max(0, kr0 - sr - noise_ring), min(valid_r, kr0 + sr + noise_ring + 1)
        d0n, d1n = max(0, kd0 - sd - noise_ring), min(M, kd0 + sd + noise_ring + 1)

        if r1s <= r0s or d1s <= d0s or not (0 <= kr0 < valid_r) or not (0 <= kd0 < M):
            results.append({
                "gt_range_m": r_gt, "gt_vel_mps": v_gt, "found": False,
                "reason": "expected location outside valid FFT range/Doppler span",
            })
            continue

        window = combined[d0s:d1s, r0s:r1s]
        wkd, wkr = np.unravel_index(np.argmax(window), window.shape)
        peak_val = window[wkd, wkr]
        found_kd, found_kr = d0s + wkd, r0s + wkr

        # local noise floor: annulus around the search window, excluding the window itself
        outer = combined[d0n:d1n, r0n:r1n]
        mask = np.ones_like(outer, dtype=bool)
        rel_r0, rel_r1 = r0s - r0n, r1s - r0n
        rel_d0, rel_d1 = d0s - d0n, d1s - d0n
        mask[rel_d0:rel_d1, rel_r0:rel_r1] = False
        noise_floor = outer[mask].mean() if mask.any() else 1e-12

        snr_db = 10 * np.log10((peak_val + 1e-12) / (noise_floor + 1e-12))
        r_found = range_bin_to_m(found_kr, cfg)
        v_found = doppler_bin_to_mps(found_kd, cfg)

        results.append({
            "gt_range_m": r_gt, "gt_vel_mps": v_gt, "found": True,
            "det_range_m": r_found, "det_vel_mps": v_found,
            "range_err_m": abs(r_found - r_gt), "vel_err_mps": abs(v_found - v_gt),
            "snr_db": snr_db,
        })
    return results


def find_peaks(power_rd, cfg=CFG, min_range_bin=2,
                guard_cells=(2, 2), train_cells=(4, 8), pfa=1e-4, max_peaks=200):
    """CA-CFAR (cell-averaging constant false alarm rate) detector over the
    non-coherent RX-combined range-Doppler map. This replaces a fixed
    top-N/exclusion-band scheme, which breaks down badly in dense scenes
    (many targets) and wrongly discards real near-zero-velocity targets
    (common in ego-motion-compensated automotive radar data) while letting
    Doppler-axis leakage sidelobes from strong targets masquerade as peaks.

    guard_cells / train_cells: (doppler, range) half-widths around the CUT.
    pfa: target false-alarm probability; sets the CFAR threshold multiplier.
    """
    N = cfg["range_fft_len"]
    M = cfg["num_chirps"]
    combined = power_rd.sum(axis=2)  # [chirps, range_bins]

    valid_r = N  # full range axis is valid for complex I/Q data (see range_bin_to_m)
    rd = combined[:, min_range_bin:valid_r]

    gd, gr = guard_cells
    td, tr = train_cells
    n_train = (2 * (gd + td) + 1) * (2 * (gr + tr) + 1) - (2 * gd + 1) * (2 * gr + 1)
    alpha = n_train * (pfa ** (-1.0 / n_train) - 1)  # CA-CFAR threshold scale factor

    # box-sum via cumulative sum for speed (padded)
    pad = np.pad(rd, ((gd + td, gd + td), (gr + tr, gr + tr)), mode="edge")
    csum = np.cumsum(np.cumsum(pad, axis=0), axis=1)
    csum = np.pad(csum, ((1, 0), (1, 0)))

    def box_sum(cs, r0, r1, c0, c1):
        return cs[r1, c1] - cs[r0, c1] - cs[r1, c0] + cs[r0, c0]

    Md, Nr = rd.shape
    detections = []
    for kd in range(Md):
        for kr in range(Nr):
            cut = rd[kd, kr]
            # big window (train+guard) minus inner window (guard only) = training cells only
            r0b, r1b = kd, kd + 2 * (gd + td) + 1
            c0b, c1b = kr, kr + 2 * (gr + tr) + 1
            r0i, r1i = kd + td, kd + td + 2 * gd + 1
            c0i, c1i = kr + tr, kr + tr + 2 * gr + 1
            big = box_sum(csum, r0b, r1b, c0b, c1b)
            inner = box_sum(csum, r0i, r1i, c0i, c1i)
            train_sum = big - inner
            noise_est = train_sum / n_train
            threshold = alpha * noise_est
            if cut > threshold and cut > 0:
                detections.append((kd, kr, cut))

    detections.sort(key=lambda d: -d[2])

    # local non-maximum suppression so one target's few above-threshold
    # neighboring cells don't get reported as separate detections
    peaks = []
    taken = np.zeros_like(rd, dtype=bool)
    for kd, kr, val in detections:
        if len(peaks) >= max_peaks:
            break
        r0, r1 = max(0, kd - gd), min(Md, kd + gd + 1)
        c0, c1 = max(0, kr - gr), min(Nr, kr + gr + 1)
        if taken[r0:r1, c0:c1].any():
            continue
        taken[r0:r1, c0:c1] = True
        kr_full = kr + min_range_bin
        peaks.append({
            "range_bin": int(kr_full), "doppler_bin": int(kd),
            "range_m": range_bin_to_m(kr_full, cfg),
            "velocity_mps": doppler_bin_to_mps(kd, cfg),
            "power_db": 10 * np.log10(val + 1e-12),
        })
    return peaks


def report(source, cfg=CFG):
    iq, _ = load_iq_stream(source, cfg)
    full_scale = 2 ** (cfg["adc_bits"] - 1) - 1
    clip_frac = np.mean((np.abs(iq.real) >= full_scale) | (np.abs(iq.imag) >= full_scale))
    power_rd = process_frame(iq, cfg)
    peaks = find_peaks(power_rd, cfg)

    label = source if isinstance(source, str) else "<in-memory frame>"
    print(f"\n=== {label.split('/')[-1] if isinstance(label, str) else label} ===")
    print(f"ADC saturation (fraction of I/Q samples at full-scale): {clip_frac*100:.1f}%")
    print(f"{'range_bin':>9} {'doppler_bin':>11} {'range_m':>9} {'velocity_mps':>13} {'power_dB':>9}")
    for p in peaks:
        print(f"{p['range_bin']:>9} {p['doppler_bin']:>11} {p['range_m']:>9.2f} "
              f"{p['velocity_mps']:>13.2f} {p['power_db']:>9.1f}")
    return peaks


def plot_range_doppler(power_rd, cfg=CFG, path=None, title=None, dpi=150, gt_points=None):
    """Non-coherent RX-combined range-Doppler heatmap in dB, physical axes.
    gt_points: optional list of (range_m, velocity_mps) ground-truth targets to overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N, M = cfg["range_fft_len"], cfg["num_chirps"]
    combined_db = 10 * np.log10(power_rd.sum(axis=2) + 1e-12)  # [chirps, range_bins]

    # Full range axis is valid for complex I/Q data (see range_bin_to_m)
    range_axis = np.array([range_bin_to_m(k, cfg) for k in range(N)])
    vel_axis = np.array([doppler_bin_to_mps(k, cfg) for k in range(M)])

    fig, ax = plt.subplots(figsize=(8, 6))
    vmax = combined_db.max()
    im = ax.pcolormesh(range_axis, vel_axis, combined_db,
                        shading="auto", cmap="viridis", vmin=vmax - 60, vmax=vmax)
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title(title or "Range-Doppler Map")
    fig.colorbar(im, ax=ax, label="Power (dB)")

    if gt_points:
        gr = [p[0] for p in gt_points]
        gv = [p[1] for p in gt_points]
        ax.scatter(gr, gv, s=80, facecolors="none", edgecolors="red",
                   linewidths=1.5, marker="o", label="nuScenes ground truth")
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()

    if path:
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        return path
    return fig


if __name__ == "__main__":
    import argparse
    import os

    ap = argparse.ArgumentParser(description="Golden reference range-Doppler pipeline")
    ap.add_argument("input_files", nargs="*",
                     help="ADC input file(s): .npy (dev/test) or .bin (DDR-style streaming buffer, "
                          "see save_iq_ddr_buffer). If omitted, globs *.npy in the current dir.")
    ap.add_argument("--outdir", default=None,
                     help="Directory to save outputs into. Default: same folder as each input file.")
    ap.add_argument("--save_output", action="store_true",
                     help="Also save the full structured output (range FFT, Doppler FFT, power, "
                          "metadata) as a .npz file for handoff to the CFAR detection stage.")
    args = ap.parse_args()

    files = args.input_files if args.input_files else sorted(glob.glob("*.npy"))
    if not files:
        print("No input files found - check your path or pass filenames explicitly.")
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)

    for f in files:
        peaks = report(f)
        output = run_pipeline(f, CFG, frame_id=os.path.basename(f))
        base = os.path.basename(f).rsplit(".", 1)[0]
        outdir = args.outdir or os.path.dirname(f)

        png_path = os.path.join(outdir, base + "_rdmap.png")
        plot_range_doppler(output.power, CFG, path=png_path, title=os.path.basename(f))
        print(f"  -> saved {png_path}")

        if args.save_output:
            npz_path = os.path.join(outdir, base + "_output.npz")
            save_frame_output(output, npz_path)
            print(f"  -> saved {npz_path}  "
                  f"(range_fft, range_doppler_fft, power, power_combined, metadata_json)")
            print(f"     metadata: {output.metadata}")