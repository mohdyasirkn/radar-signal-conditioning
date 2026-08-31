"""
main.py
=======
Top-level orchestration script for the Radar Range-Doppler Golden Reference
Model. This is the file you run.

Pipeline stages (mirrors the RTL datapath 1:1, in order):
    1. Load raw ADC frame                 (dataset/frame_0001.mat)
    2. Select TX/RX channel               -> (chirps, samples) complex matrix
    3. Per-chirp DC offset removal
    4. Hann window (fast-time / range axis)
    5. 128-point Range FFT
    6. Zero-pad 255 -> 256 chirps
    7. 256-point Doppler FFT
    8. FFT shift (Doppler axis)
    9. Magnitude / Power computation
   10. Save every intermediate stage as .npy
   11. Export RTL-compatible vectors (.txt)
   12. Plot intermediate stages + final Range-Doppler heatmap

Every stage's output is saved before moving to the next, so each step can be
diffed directly against an RTL simulation waveform/memory dump.

Depends on:
    config.py             - all parameters
    radar_processing.py   - load_frame, select_channel, remove_dc_offset,
                             apply_window, range_fft, zero_pad_chirps,
                             doppler_fft, fft_shift, compute_magnitude,
                             compute_power
    export_vectors.py     - export_stage
    visualization.py      - plot_pipeline_stages, plot_range_doppler_heatmap
"""

import os
import numpy as np

import config as cfg
import radar_processing as rp
import export_vectors as ev
import visualization as viz


def save_stage(name, array):
    """Persist an intermediate stage as .npy for regression / RTL comparison."""
    path = os.path.join(cfg.NPY_DIR, f"{name}.npy")
    np.save(path, array)
    print(f"[SAVE] {name:<20s} shape={str(array.shape):<15s} -> {path}")
    return array


def main():
    print("=" * 60)
    print("Radar Golden Reference Model")
    print(f"Frame     : {cfg.FRAME_FILE}")
    print(f"Channel   : TX{cfg.SELECTED_TX} / RX{cfg.SELECTED_RX}")
    print("=" * 60)

    # ------------------------------------------------------------
    # 1. Load raw frame
    # ------------------------------------------------------------
    raw_frame = rp.load_frame(cfg.FRAME_FILE, cfg.FRAME_MAT_KEY)
    save_stage("00_raw_frame", raw_frame)

    # ------------------------------------------------------------
    # 2. Select RX/TX channel -> (chirps, samples) complex matrix
    # ------------------------------------------------------------
    chirp_matrix = rp.select_channel(
        raw_frame,
        tx=cfg.SELECTED_TX,
        rx=cfg.SELECTED_RX,
        num_tx=cfg.NUM_TX,
        num_rx=cfg.NUM_RX,
        num_chirps=cfg.NUM_CHIRPS,
        num_samples=cfg.NUM_ADC_SAMPLES,
    )
    save_stage("01_selected_channel", chirp_matrix)

    # ------------------------------------------------------------
    # 3. Per-chirp DC offset removal
    # ------------------------------------------------------------
    dc_removed = rp.remove_dc_offset(chirp_matrix, enable=cfg.DC_REMOVAL_ENABLE)
    save_stage("02_dc_removed", dc_removed)

    # ------------------------------------------------------------
    # 4. Hann window (range / fast-time axis)
    # ------------------------------------------------------------
    windowed = rp.apply_window(
        dc_removed,
        window_type=cfg.RANGE_WINDOW_TYPE,
        axis=cfg.RANGE_FFT_AXIS,
    )
    save_stage("03_range_windowed", windowed)

    # ------------------------------------------------------------
    # 5. Range FFT (128-point)
    # ------------------------------------------------------------
    range_fft = rp.range_fft(
        windowed,
        fft_size=cfg.RANGE_FFT_SIZE,
        axis=cfg.RANGE_FFT_AXIS,
    )
    save_stage("04_range_fft", range_fft)

    # ------------------------------------------------------------
    # 6. Zero-pad chirps (255 -> 256)
    # ------------------------------------------------------------
    zero_padded = rp.zero_pad_chirps(
        range_fft,
        pad_from=cfg.DOPPLER_ZERO_PAD_FROM,
        pad_to=cfg.DOPPLER_ZERO_PAD_TO,
        axis=cfg.DOPPLER_FFT_AXIS,
    )
    save_stage("05_zero_padded", zero_padded)

    # ------------------------------------------------------------
    # 7. Doppler FFT (256-point)
    # ------------------------------------------------------------
    doppler_fft = rp.doppler_fft(
        zero_padded,
        fft_size=cfg.DOPPLER_FFT_SIZE,
        axis=cfg.DOPPLER_FFT_AXIS,
        window_type=cfg.DOPPLER_WINDOW_TYPE,
    )
    save_stage("06_doppler_fft", doppler_fft)

    # ------------------------------------------------------------
    # 8. FFT shift (Doppler axis, zero-velocity to center)
    # ------------------------------------------------------------
    shifted = rp.fft_shift(
        doppler_fft,
        axis=cfg.DOPPLER_FFT_AXIS,
        enable=cfg.DOPPLER_FFT_SHIFT,
    )
    save_stage("07_fft_shifted", shifted)

    # ------------------------------------------------------------
    # 9. Magnitude / Power
    # ------------------------------------------------------------
    magnitude = None
    power = None

    if cfg.COMPUTE_MAGNITUDE:
        magnitude = rp.compute_magnitude(shifted)
        save_stage("08_magnitude", magnitude)

    if cfg.COMPUTE_POWER:
        power = rp.compute_power(
            shifted,
            in_db=cfg.POWER_IN_DB,
            ref=cfg.POWER_DB_REF,
            floor_db=cfg.POWER_DB_FLOOR,
        )
        save_stage("09_power", power)

    # ------------------------------------------------------------
    # 10. Export RTL-compatible vectors (.txt)
    # ------------------------------------------------------------
    stages_to_export = {
        "chirp_matrix":   chirp_matrix,
        "dc_removed":     dc_removed,
        "range_windowed": windowed,
        "range_fft":      range_fft,
        "zero_padded":    zero_padded,
        "doppler_fft":    doppler_fft,
        "fft_shifted":    shifted,
    }
    if magnitude is not None:
        stages_to_export["magnitude"] = magnitude
    if power is not None:
        stages_to_export["power"] = power

    for stage_name, array in stages_to_export.items():
        ev.export_stage(
            array,
            stage_name=stage_name,
            output_dir=cfg.TXT_DIR,
            int_bits=cfg.EXPORT_INT_BITS,
            frac_bits=cfg.EXPORT_FRAC_BITS,
            fmt=cfg.EXPORT_FORMAT,
            signed=cfg.EXPORT_SIGNED,
        )

    # ------------------------------------------------------------
    # 11. Visualization
    # ------------------------------------------------------------
    if cfg.SAVE_FIGURES or cfg.SHOW_FIGURES:
        viz.plot_pipeline_stages(
            raw=chirp_matrix,
            dc_removed=dc_removed,
            windowed=windowed,
            range_fft=range_fft,
            doppler_fft=shifted,
        )

        final_power = power if power is not None else rp.compute_power(shifted)
        viz.plot_range_doppler_heatmap(final_power, cfg=cfg)

    print("=" * 60)
    print("Pipeline complete.")
    print(f"  npy     -> {cfg.NPY_DIR}")
    print(f"  txt     -> {cfg.TXT_DIR}")
    print(f"  figures -> {cfg.FIG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
