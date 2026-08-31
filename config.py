"""
config.py
=========
Central configuration for the Radar Range-Doppler Golden Reference Model.

This file is the single source of truth for every parameter that must stay
bit/architecture-consistent with the RTL implementation (frame geometry,
FFT sizes, windowing, fixed-point export format, etc). Treat changes here
as changes to the "spec" the RTL is being verified against.
"""

import os

# ------------------------------------------------------------------
# Directory structure
# ------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = r"C:\Users\gokul\Downloads\Automotive\Automotive\2019_05_09_mlms003\radar_raw_frame"
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs")
NPY_DIR      = os.path.join(OUTPUT_DIR, "npy")
TXT_DIR      = os.path.join(OUTPUT_DIR, "txt")
FIG_DIR      = os.path.join(OUTPUT_DIR, "figures")

for _d in (NPY_DIR, TXT_DIR, FIG_DIR):
    os.makedirs(_d, exist_ok=True)

FRAME_FILE    = os.path.join(DATASET_DIR, "000823.mat")
FRAME_MAT_KEY = "adcData"   # <-- set to the actual variable name inside the .mat file

# ------------------------------------------------------------------
# Radar frame geometry (must match RTL AXI framing / memory map)
# ------------------------------------------------------------------
NUM_TX = 2
NUM_RX = 4
NUM_CHIRPS = 255          # chirps captured per frame (slow time)
NUM_ADC_SAMPLES = 128     # samples per chirp (fast time)

# Channel selected for this run (0-indexed). Change these or expose them
# as CLI args later if you need to sweep every TX/RX pair.
SELECTED_TX = 0
SELECTED_RX = 0

# ------------------------------------------------------------------
# ADC / fixed point of the RAW data
# ------------------------------------------------------------------
ADC_BITS = 16
ADC_SIGNED = True

# ------------------------------------------------------------------
# DC Offset Removal
# ------------------------------------------------------------------
DC_REMOVAL_ENABLE = True
DC_REMOVAL_AXIS = "per_chirp"   # subtract the mean along fast-time, per chirp

# ------------------------------------------------------------------
# Windowing
# ------------------------------------------------------------------
RANGE_WINDOW_TYPE = "hann"      # applied along fast-time, before the range FFT
DOPPLER_WINDOW_TYPE = "hann"      # set to "hann" if the RTL also windows Doppler

# ------------------------------------------------------------------
# Range FFT
# ------------------------------------------------------------------
RANGE_FFT_SIZE = 128
RANGE_FFT_AXIS = 1              # fast-time axis in a (chirps, samples) matrix

# ------------------------------------------------------------------
# Doppler processing
# ------------------------------------------------------------------
DOPPLER_ZERO_PAD_FROM = 255     # chirps present after range FFT
DOPPLER_ZERO_PAD_TO   = 256     # chirps after zero-padding
DOPPLER_FFT_SIZE = 256
DOPPLER_FFT_AXIS = 0            # slow-time axis
DOPPLER_FFT_SHIFT = True        # move zero-velocity bin to the center

# ------------------------------------------------------------------
# Post-processing
# ------------------------------------------------------------------
COMPUTE_MAGNITUDE = True
COMPUTE_POWER = True
POWER_IN_DB = True
POWER_DB_REF = 1.0
POWER_DB_FLOOR = -120.0         # dB floor to avoid log(0) on empty bins

# ------------------------------------------------------------------
# RTL-compatible export
# ------------------------------------------------------------------
EXPORT_FIXED_POINT = True
EXPORT_INT_BITS = 16
EXPORT_FRAC_BITS = 0            # set > 0 if exporting Qm.n fixed point
EXPORT_FORMAT = "hex"           # "hex" or "dec"
EXPORT_SIGNED = True

# ------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------
SAVE_FIGURES = True
SHOW_FIGURES = False
FIGURE_DPI = 150

# Physical constants used only for axis labeling (range/velocity units).
# Adjust to match your actual radar's chirp configuration.
SAMPLE_RATE_HZ = 4e6
CHIRP_SLOPE_HZ_PER_S = 21e12
CHIRP_TIME_S = 120e-6
CENTER_FREQ_HZ = 77e9
LIGHT_SPEED = 3e8
