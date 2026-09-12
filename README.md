# Radar Signal Conditioning & FFT Processing — Golden Reference Model

Python golden-reference pipeline for an ASIC-ready FMCW radar signal-conditioning and FFT IP. This repo is the **verification/reference side** of the project: it forward-simulates raw radar ADC data from the nuScenes dataset, runs it through a DC-removal → window → range-FFT → Doppler-FFT → power pipeline (mirroring the planned RTL: `radar_dc_remove` → `radar_window` → `radar_fft_range` → `radar_fft_doppler` → `radar_mag_power`), and checks the resulting range-Doppler heatmap against nuScenes ground truth.

RTL implementation is a separate, not-yet-published part of the project — this repo currently contains only the Python reference model and its verification harness.

**Files**
| File | Role |
|---|---|
| `nuscenes_to_raw_adc.py` | Forward-simulates synthetic raw FMCW ADC I/Q data from nuScenes radar point clouds and quantizes it to fixed-point |
| `nuscenes_ref_model.py` | Golden reference receive chain (DC removal → window → range FFT → Doppler FFT → power) + I/O helpers (`.npy`/DDR-style `.bin`/`.npz`) + CFAR/local peak finding + heatmap plotting |
| `nuscenes_golden_vs_groundtruth_verify.py` | Runs the golden model on synthetic frames and checks the heatmap against nuScenes ground truth, per-frame and in aggregate |

---

## Software Required

- Python 3.x
- `numpy`
- `matplotlib` (only needed if generating heatmap PNGs)
- [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit) + a local copy of the **nuScenes `v1.0-trainval`** dataset (`samples/`, `sweeps/`, `maps/`, and the metadata folder)

```
pip install nuscenes-devkit numpy matplotlib --break-system-packages
```

---

## How to Execute

Run in this order — each script's output feeds the next.

**1. Generate synthetic raw ADC data from nuScenes**
```
python nuscenes_to_raw_adc.py --dataroot /path/to/v1.0-trainval --version v1.0-trainval --sensor RADAR_FRONT --out ./synthetic_adc
```
Optional: `--max_samples N` to limit the run to N frames for a quick test.
This writes one `<sample_token>_adc.npy` file per frame to `--out`.

**2. Run the golden reference pipeline standalone** *(optional — mainly for spot-checking one/several frames)*
```
python nuscenes_ref_model.py ./synthetic_adc/*.npy --save_output
```
For each input `.npy`, this prints detected range/Doppler/power peaks, saves a `*_rdmap.png` heatmap, and (with `--save_output`) a `*_output.npz` containing the range FFT, Doppler FFT, power map, and metadata.

**3. Verify the golden model against nuScenes ground truth**
```
python nuscenes_golden_vs_groundtruth_verify.py --dataroot /path/to/v1.0-trainval --version v1.0-trainval --adc_dir ./synthetic_adc --outdir ./verify_out
```
Optional flags: `--snr_min_db` (default 6.0) to flag weak-but-present targets, `--cfar_diagnostic` to additionally run a global CFAR pass and flag any detections not explained by ground truth, `--save_output` to also dump `.npz` outputs per frame.

`nuscenes_to_raw_adc.py` and `nuscenes_ref_model.py` must be importable (same folder or on `PYTHONPATH`) when running step 3.

---

## Input Format

- Source data: nuScenes **RADAR_FRONT** point-cloud annotations (`range`, ego-motion-compensated radial velocity, RCS) per sample — nuScenes does not contain true raw ADC samples, so each point is treated as a ground-truth target and forward-simulated into a raw time-domain beat signal.
- Synthetic ADC output: complex I/Q cube, shape `[num_chirps, range_fft_len, num_rx]`, **signed fixed-point** ADC codes (Q1.(adc_bits−1), default 16-bit).
- Two on-disk formats are supported downstream:
  - `.npy` — plain numpy array, for development/testing
  - `.bin` — DDR/streaming-style buffer: 8-byte magic (`RADQIQ01`) + a fixed header (chirps, range samples, rx, bit width, frame id) + interleaved `int16` I/Q samples in `[chirp, range_sample, rx]` order, produced/read by `save_iq_ddr_buffer()` / `load_iq_ddr_buffer()` in `nuscenes_ref_model.py`.

## Expected Output

- **Synthetic ADC frames**: `<token>_adc.npy` per nuScenes sample (from step 1).
- **Range-Doppler outputs** (from step 2, and internally in step 3): a `RadarFrameOutput` with `range_fft`, `range_doppler_fft`, `power` (per-RX, `Re²+Im²`), `power_combined` (non-coherent RX sum — the heatmap), and metadata (bin-to-meters/mps scaling, ADC saturation fraction, frame id). Saved as `.npz` when `--save_output` is passed.
- **Heatmap PNGs**: `<token>_rdmap.png`, range (m) vs. velocity (m/s) in dB, with nuScenes ground-truth points overlaid when produced via the verify script.
- **Verification report** (step 3, console): per-frame table of ground-truth range/velocity vs. detected range/velocity, range/velocity error, and local SNR, plus an aggregate summary (targets found within valid FFT span, low-SNR count, mean/std/max range and velocity error, mean/min SNR). Verification uses a **local, ground-truth-guided check** (small search window + noise annulus around each expected target bin) rather than blind CFAR, since CFAR's own false-alarm/miss behavior would confound a heatmap-correctness check in dense scenes; CFAR is available only as an optional diagnostic for unexplained energy.

## Important Parameters

Config lives in the `CONFIG` dict (`nuscenes_to_raw_adc.py`) / `CFG` dict (`nuscenes_ref_model.py`) — **the two must be kept in sync** when changed, since the golden model has no way to detect a mismatch against the generator.

| Parameter | Default | Notes |
|---|---|---|
| `range_fft_len` | 256 | Range FFT size (samples per chirp); must be a power of 2 |
| `num_chirps` | 128 | Doppler FFT size |
| `num_rx` | 4 | Receive channels |
| `adc_bits` | 16 | ADC width; fixed-point format is Q1.(adc_bits−1) |
| `window_type` | `hann` | `hann` or `hamming` |
| `window_periodic` | `False` | `True` = LUT-friendly (matches RTL windowing convention); `False` = matches the generator's own reference chain |
| `doppler_window` | `True` | Applies a window on the Doppler axis too — **set `False` for a bit-exact comparison against RTL**, since the RTL spec only windows the range axis |
| `fc_hz` | 77.0 GHz | Carrier frequency (ARS408-like, matches the nuScenes radar sensor) |
| `bandwidth_hz` | 500 MHz | Sweep bandwidth |
| `chirp_duration_s` | 25.6 µs | Active sweep time per chirp |
| `max_unambiguous_range_m` | *derived* | Computed from `range_fft_len`, `bandwidth_hz`, `chirp_duration_s` (≈0.98× the Nyquist-limited physical max) — not hardcoded, so it can't drift out of sync if waveform params change |
| `snr_db_at_1rcs_10m` / `noise_floor_std` | 40 dB / 0.02 | Amplitude calibration and noise floor for the synthetic signal (generator only) |
| `--snr_min_db` (verify script) | 6.0 | Local SNR below which a found target is flagged "weak" |

---


