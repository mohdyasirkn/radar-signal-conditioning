"""
visualization.py
=================
Plotting utilities for the Radar Range-Doppler Golden Reference Model.

Two entry points, called from main.py:

    plot_pipeline_stages(...)        - one figure per intermediate stage,
                                        useful for eyeballing sanity/bugs
                                        before comparing numerically to RTL.
    plot_range_doppler_heatmap(...)  - the final Range-Doppler map with
                                        physical range (m) / velocity (m/s)
                                        axes.

Figures are saved under cfg.FIG_DIR and/or shown interactively, controlled
by cfg.SAVE_FIGURES / cfg.SHOW_FIGURES. Nothing here computes DSP — it only
visualizes arrays produced by radar_processing.py.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import config as cfg


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------
def _finish(fig, filename):
    """Save/show a figure per config flags, then close it to free memory."""
    if cfg.SAVE_FIGURES:
        path = os.path.join(cfg.FIG_DIR, filename)
        fig.savefig(path, dpi=cfg.FIGURE_DPI, bbox_inches="tight")
        print(f"[FIG]  {filename:<30s} -> {path}")

    if cfg.SHOW_FIGURES:
        plt.show()

    plt.close(fig)


def _mag_db(complex_or_real, floor_db=-120.0):
    """Convert a complex/real array to dB magnitude for display."""
    mag = np.abs(complex_or_real)
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(mag + 1e-20)
    return np.maximum(db, floor_db)


def _range_axis_m(num_range_bins):
    """
    Physical range axis (meters) for a real FMCW chirp, one-sided
    (bins 0 .. num_range_bins/2 correspond to positive beat frequencies).
    """
    beat_freqs = np.fft.fftfreq(num_range_bins, d=1.0 / cfg.SAMPLE_RATE_HZ)
    ranges = beat_freqs * cfg.LIGHT_SPEED / (2.0 * cfg.CHIRP_SLOPE_HZ_PER_S)
    return ranges


def _velocity_axis_mps(num_doppler_bins):
    """
    Physical velocity axis (m/s), centered on zero after fftshift.
    """
    wavelength = cfg.LIGHT_SPEED / cfg.CENTER_FREQ_HZ
    doppler_freqs = np.fft.fftshift(
        np.fft.fftfreq(num_doppler_bins, d=cfg.CHIRP_TIME_S)
    )
    velocities = doppler_freqs * wavelength / 2.0
    return velocities


# ----------------------------------------------------------------------
# Pipeline stage plots
# ----------------------------------------------------------------------
def plot_pipeline_stages(raw, dc_removed, windowed, range_fft, doppler_fft):
    """
    Plot a snapshot of each intermediate stage for visual sanity-checking.

    raw, dc_removed, windowed : (chirps, samples), time-domain, complex
    range_fft                 : (chirps, range_bins), complex
    doppler_fft               : (doppler_bins, range_bins), complex,
                                 already zero-padded / Doppler-FFT'd /
                                 shifted (i.e. the final complex map)
    """
    _plot_time_domain_stage(raw, "00_raw_first_chirp.png",
                             title="Raw ADC data (chirp 0)")
    _plot_time_domain_stage(dc_removed, "01_dc_removed_first_chirp.png",
                             title="DC-Removed (chirp 0)")
    _plot_time_domain_stage(windowed, "02_windowed_first_chirp.png",
                             title="Hann-Windowed (chirp 0)")

    _plot_range_fft_stage(range_fft)
    _plot_doppler_map_stage(doppler_fft, "04_doppler_fft_magnitude_db.png",
                             title="Doppler FFT magnitude (dB, pre-heatmap)")


def _plot_time_domain_stage(chirp_matrix, filename, title, chirp_index=0):
    """Plot I/Q of a single chirp, time-domain."""
    chirp = chirp_matrix[chirp_index, :]
    samples = np.arange(chirp.shape[0])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(samples, chirp.real, label="I (real)")
    ax.plot(samples, chirp.imag, label="Q (imag)")
    ax.set_title(title)
    ax.set_xlabel("Sample index (fast time)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    _finish(fig, filename)


def _plot_range_fft_stage(range_fft_matrix):
    """Plot range-FFT magnitude for a single chirp and the full matrix."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    mag_db_single = _mag_db(range_fft_matrix[0, :])
    axes[0].plot(mag_db_single)
    axes[0].set_title("Range FFT magnitude, chirp 0")
    axes[0].set_xlabel("Range bin")
    axes[0].set_ylabel("Magnitude (dB)")
    axes[0].grid(True, alpha=0.3)

    mag_db_all = _mag_db(range_fft_matrix)
    im = axes[1].imshow(
        mag_db_all, aspect="auto", origin="lower", cmap="viridis"
    )
    axes[1].set_title("Range FFT magnitude, all chirps")
    axes[1].set_xlabel("Range bin")
    axes[1].set_ylabel("Chirp index")
    fig.colorbar(im, ax=axes[1], label="dB")

    _finish(fig, "03_range_fft_magnitude_db.png")


def _plot_doppler_map_stage(doppler_fft_matrix, filename, title):
    """Plot the Doppler-FFT (post-shift) magnitude in raw bin coordinates."""
    mag_db = _mag_db(doppler_fft_matrix)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mag_db, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Range bin")
    ax.set_ylabel("Doppler bin (shifted)")
    fig.colorbar(im, ax=ax, label="dB")

    _finish(fig, filename)


# ----------------------------------------------------------------------
# Final Range-Doppler heatmap
# ----------------------------------------------------------------------
def plot_range_doppler_heatmap(power_matrix, cfg=cfg):
    """
    Plot the final Range-Doppler map with physical axes.

    power_matrix : (doppler_bins, range_bins), already fftshift'd on the
                   Doppler axis. Values may be linear power or dB
                   (cfg.POWER_IN_DB controls the colorbar label only).
    """
    num_doppler_bins, num_range_bins = power_matrix.shape

    ranges = _range_axis_m(num_range_bins)
    velocities = _velocity_axis_mps(num_doppler_bins)

    # Only the positive-range half of the range FFT is physically meaningful
    # for a real FMCW beat signal; show 0 .. Nyquist.
    half = num_range_bins // 2
    ranges_half = ranges[:half]
    power_half = power_matrix[:, :half]

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.pcolormesh(
        ranges_half, velocities, power_half,
        shading="auto", cmap="jet",
    )
    ax.set_title("Range-Doppler Heatmap")
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Velocity (m/s)")

    cbar_label = "Power (dB)" if cfg.POWER_IN_DB else "Power (linear)"
    fig.colorbar(im, ax=ax, label=cbar_label)

    _finish(fig, "10_range_doppler_heatmap.png")
