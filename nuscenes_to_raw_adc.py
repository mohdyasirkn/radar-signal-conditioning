"""
nuscenes_to_raw_adc.py

Forward-simulates synthetic raw FMCW radar ADC (I/Q) data from nuScenes'
already-processed radar point clouds / annotations.

nuScenes does NOT contain true raw ADC samples (only clustered/filtered
radar points from the Continental ARS408). This script treats each
nuScenes radar point (range, radial velocity, RCS) as ground-truth target
state and forward-simulates the raw time-domain beat signal that an FMCW
radar would have produced for that target, then quantizes it to emulate
an ADC. The output is shaped to match the radar_if.v / signal-conditioning
front end described in the project spec (Raw ADC I/Q -> DC removal ->
window -> Range FFT -> Doppler FFT -> magnitude -> CFAR).

Usage:
    python nuscenes_to_raw_adc.py --dataroot /path/to/nuscenes \
        --version v1.0-mini --sensor RADAR_FRONT --out ./synthetic_adc

Requires: nuscenes-devkit, numpy
    pip install nuscenes-devkit numpy --break-system-packages
"""

import argparse
import os
import numpy as np

# --------------------------------------------------------------------------
# 1. RADAR WAVEFORM / SYSTEM CONFIG
# --------------------------------------------------------------------------
# Parameters marked [SPEC] come directly from the project document.
# Parameters marked [ASSUMED] are not specified there and are set to
# ARS408-like automotive radar values (the sensor nuScenes radar data
# actually came from) so the simulated beat frequencies stay physically
# consistent with the dataset's range/velocity values. Override as needed.

CONFIG = {
    # --- [SPEC] Range FFT / signal conditioning ---
    "range_fft_len": 256,        # [SPEC] radix-2, 256 or 512 in doc; must be pow2
    "window_type": "hann",       # [SPEC] Hann or Hamming
    "adc_bits": 16,              # [ASSUMED] doc says "fixed-point, parameterized"
                                  #           but gives no bit width
    "dc_offset_lsb": 0,          # simulated ADC DC bias to be removed downstream
                                  # (0 = clean; set >0 to exercise radar_dc_remove.v)

    # --- [ASSUMED] FMCW waveform, ARS408-like (matches nuScenes radar sensor) ---
    "fc_hz": 77.0e9,              # carrier frequency
    "bandwidth_hz": 500.0e6,      # sweep bandwidth -> range resolution
    "chirp_duration_s": 25.6e-6,  # active sweep time per chirp
    "num_chirps": 128,            # [ASSUMED] Doppler FFT size, independent of
                                   #           the range-FFT length in the doc
    "num_rx": 4,                  # [ASSUMED] receive channels (typical automotive)
    # NOTE: max_unambiguous_range_m is NOT set here as a fixed number anymore.
    # It used to be hardcoded to 250.0 m, which didn't match what the waveform
    # (Fs, N, bandwidth, chirp duration) can actually support unambiguously
    # (~76.7 m for the params above) - any target beyond that limit had its
    # beat frequency alias past Nyquist, landing in the wrong range bin in the
    # synthetic ADC output. It's now computed below from the waveform params
    # themselves (with a small safety margin) so this can't drift out of sync
    # again if you change range_fft_len / bandwidth_hz / chirp_duration_s.

    # --- Noise / dynamic range ---
    "snr_db_at_1rcs_10m": 40.0,   # calibration point for the radar-eqn amplitude
    "noise_floor_std": 0.02,      # relative to full-scale ADC
}

C = 299_792_458.0

# Derive the true unambiguous range from the waveform parameters (see
# range_bin_to_m / find_peaks discussion: for complex I/Q data, aliasing
# happens beyond beat frequency fb = Fs, i.e. beyond R = Fs*C/(2*S)).
_Fs = CONFIG["range_fft_len"] / CONFIG["chirp_duration_s"]
_S = CONFIG["bandwidth_hz"] / CONFIG["chirp_duration_s"]
_R_max_physical = _Fs * C / (2 * _S)
CONFIG["max_unambiguous_range_m"] = 0.98 * _R_max_physical  # small margin off the exact Nyquist edge




# --------------------------------------------------------------------------
# 2. LOAD TARGETS FROM NUSCENES
# --------------------------------------------------------------------------
def load_targets_for_sample(nusc, sample, sensor="RADAR_FRONT"):
    """Return list of dicts: {range_m, radial_vel_mps, rcs_dbsm} for one frame."""
    from nuscenes.utils.data_classes import RadarPointCloud

    sd = nusc.get("sample_data", sample["data"][sensor])
    pc = RadarPointCloud.from_file(nusc.get_sample_data_path(sd["token"]))
    # pc.points rows: x, y, z, dyn_prop, id, rcs, vx, vy, vx_comp, vy_comp, ...
    x, y = pc.points[0, :], pc.points[1, :]
    vx, vy = pc.points[8, :], pc.points[9, :]  # compensated (ego-motion removed)
    rcs = pc.points[5, :]

    targets = []
    for i in range(pc.points.shape[1]):
        r = float(np.hypot(x[i], y[i]))
        if r < 1.0 or r > CONFIG["max_unambiguous_range_m"]:
            continue
        # radial velocity = velocity component along the range vector
        if r > 0:
            v_r = float((vx[i] * x[i] + vy[i] * y[i]) / r)
        else:
            v_r = 0.0
        targets.append({"range_m": r, "radial_vel_mps": v_r, "rcs_dbsm": float(rcs[i])})
    return targets


# --------------------------------------------------------------------------
# 3. FORWARD FMCW SIMULATION -> RAW I/Q
# --------------------------------------------------------------------------
def simulate_raw_adc_frame(targets, cfg=CONFIG, rng=None):
    """
    Returns complex I/Q cube of shape [num_chirps, range_fft_len, num_rx],
    representing the raw (pre-conditioning) ADC stream for one frame.
    """
    if rng is None:
        rng = np.random.default_rng()

    N = cfg["range_fft_len"]           # samples per chirp
    M = cfg["num_chirps"]
    Fs = N / cfg["chirp_duration_s"]   # ADC sample rate implied by chirp/N
    S = cfg["bandwidth_hz"] / cfg["chirp_duration_s"]  # slope Hz/s
    fc = cfg["fc_hz"]
    Tc = cfg["chirp_duration_s"]

    t = np.arange(N) / Fs              # fast-time samples within a chirp
    m_idx = np.arange(M)               # slow-time (chirp) index

    iq = np.zeros((M, N, cfg["num_rx"]), dtype=np.complex128)

    for tgt in targets:
        R, v, rcs_dbsm = tgt["range_m"], tgt["radial_vel_mps"], tgt["rcs_dbsm"]

        fb = 2 * S * R / C                       # beat frequency from range
        doppler_phase_per_chirp = 4 * np.pi * fc * v * Tc / C

        # Amplitude from a simplified radar equation, calibrated so a
        # 1 dBsm target at 10 m sits at the configured reference SNR.
        rcs_lin = 10 ** (rcs_dbsm / 10.0)
        ref_r, ref_rcs = 10.0, 1.0
        amp = np.sqrt(rcs_lin / ref_rcs) * (ref_r / max(R, 1.0)) ** 2
        amp *= 10 ** (cfg["snr_db_at_1rcs_10m"] / 20.0) * cfg["noise_floor_std"]

        fast_time_phase = 2 * np.pi * fb * t     # [N]
        slow_time_phase = doppler_phase_per_chirp * m_idx  # [M]

        # random per-RX phase offset (antenna spacing -> angle info; not used
        # for detection here, just keeps channels non-identical)
        rx_phase = rng.uniform(0, 2 * np.pi, cfg["num_rx"])

        phase = (fast_time_phase[None, :, None]
                 + slow_time_phase[:, None, None]
                 + rx_phase[None, None, :])
        iq += amp * np.exp(1j * phase)

    # Additive thermal noise
    noise = (rng.normal(0, cfg["noise_floor_std"], iq.shape)
             + 1j * rng.normal(0, cfg["noise_floor_std"], iq.shape))
    iq += noise

    # --- AGC: normalize per-frame peak to a fixed headroom before quantizing ---
    # Without this, amp (above) commonly exceeds full scale even for a single
    # moderate-RCS target, causing near-total ADC clipping and corrupting both
    # range and Doppler estimates downstream.
    headroom = 0.7  # fraction of full scale to normalize peak amplitude to
    peak = max(np.abs(iq.real).max(), np.abs(iq.imag).max())
    if peak > 0:
        iq *= headroom / peak

    # Simulated DC bias (to exercise radar_dc_remove.v downstream)
    iq += cfg["dc_offset_lsb"] / (2 ** (cfg["adc_bits"] - 1))

    return quantize_adc(iq, cfg["adc_bits"])


def quantize_adc(iq, bits):
    """Clip to [-1, 1] full scale and quantize to a signed integer ADC code."""
    full_scale = 2 ** (bits - 1) - 1
    i = np.clip(np.real(iq), -1, 1)
    q = np.clip(np.imag(iq), -1, 1)
    i_code = np.round(i * full_scale).astype(np.int32)
    q_code = np.round(q * full_scale).astype(np.int32)
    return i_code + 1j * q_code


# --------------------------------------------------------------------------
# 4. (OPTIONAL) REFERENCE RECEIVE CHAIN, FOR SELF-VALIDATION
#    Mirrors radar_dc_remove -> radar_window -> radar_fft_range/doppler ->
#    radar_mag_power, so you can sanity-check the synthetic data recovers
#    the original nuScenes range/velocity before it ever hits RTL.
# --------------------------------------------------------------------------
def reference_range_doppler(iq_frame, cfg=CONFIG):
    x = iq_frame.astype(np.complex128)

    # DC offset removal (mean subtraction per chirp)
    x = x - x.mean(axis=1, keepdims=True)

    # Windowing
    if cfg["window_type"] == "hann":
        win = np.hanning(cfg["range_fft_len"])
    else:
        win = np.hamming(cfg["range_fft_len"])
    x = x * win[None, :, None]

    # Range FFT (per chirp, per rx)
    range_fft = np.fft.fft(x, n=cfg["range_fft_len"], axis=1)

    # Doppler FFT (across chirps)
    range_doppler = np.fft.fftshift(np.fft.fft(range_fft, axis=0), axes=0)

    power = np.abs(range_doppler) ** 2  # matches radar_mag_power.v: Re^2+Im^2
    return power  # [chirps, range_bins, rx]


# --------------------------------------------------------------------------
# 5. DRIVER
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default=r"C:\Users\gokul\Downloads\archive\v1.0-trainval",
                     help="Folder containing samples/, sweeps/, maps/, and the "
                          "v1.0-trainval metadata subfolder")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--sensor", default="RADAR_FRONT")
    ap.add_argument("--out", default=r"C:\Users\gokul\Downloads\archive\v1.0-trainval\synthetic_adc")
    ap.add_argument("--max_samples", type=int, default=None,
                     help="Limit number of frames for a quick test run")
    args = ap.parse_args()

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    os.makedirs(args.out, exist_ok=True)

    rng = np.random.default_rng(42)
    count = 0
    for sample in nusc.sample:
        if args.max_samples and count >= args.max_samples:
            break
        targets = load_targets_for_sample(nusc, sample, args.sensor)
        if not targets:
            continue
        iq_frame = simulate_raw_adc_frame(targets, CONFIG, rng)

        out_path = os.path.join(args.out, f"{sample['token']}_adc.npy")
        np.save(out_path, iq_frame)
        count += 1

    print(f"Wrote {count} synthetic raw ADC frames to {args.out}")
    print(f"Frame shape: [num_chirps={CONFIG['num_chirps']}, "
          f"range_fft_len={CONFIG['range_fft_len']}, num_rx={CONFIG['num_rx']}]")


if __name__ == "__main__":
    main()