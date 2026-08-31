"""
radar_processing.py
====================
DSP core of the Radar Range-Doppler Golden Reference Model.

Every function here corresponds to exactly one RTL pipeline stage. Keep that
1:1 mapping intact — if a stage gets split or merged in the RTL, mirror the
same split/merge here so waveform-vs-.npy comparisons stay meaningful.

Data convention used throughout this module:
    A single-channel frame is a 2-D complex array shaped
        (chirps, samples)   i.e. (slow_time, fast_time)
    unless noted otherwise. This matches DOPPLER_FFT_AXIS=0 / RANGE_FFT_AXIS=1
    in config.py.
"""

import numpy as np
from scipy.io import loadmat


# ----------------------------------------------------------------------
# 1. Load raw frame
# ----------------------------------------------------------------------
def load_frame(mat_path, mat_key):
    """
    Load a raw radar frame from a .mat file.

    Expected on-disk layout (adjust to your actual capture format):
        raw[mat_key] : complex or real array of shape
                        (num_tx, num_rx, num_chirps, num_samples)
                     or (num_tx * num_rx, num_chirps, num_samples)
                     or interleaved I/Q real data with a trailing dim of 2.

    Returns
    -------
    np.ndarray
        The raw array exactly as stored (dtype/shape untouched). Reshaping
        and channel selection happen in select_channel(), not here, so this
        stage can be diffed 1:1 against whatever the RTL testbench reads
        off of the ADC/AXI interface.
    """
    mat = loadmat(mat_path)
    if mat_key not in mat:
        available = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(
            f"'{mat_key}' not found in {mat_path}. "
            f"Available variables: {available}"
        )
    raw = mat[mat_key]

    # If I/Q was stored as a trailing real dimension of size 2, fold it into
    # a proper complex array: [..., 0] = I, [..., 1] = Q.
    if np.isrealobj(raw) and raw.shape[-1] == 2:
        raw = raw[..., 0] + 1j * raw[..., 1]

    return raw


# ----------------------------------------------------------------------
# 2. Select TX/RX channel
# ----------------------------------------------------------------------
def select_channel(raw_frame, tx, rx, num_tx, num_rx, num_chirps, num_samples):
    """
    Extract a single TX/RX channel as a (chirps, samples) complex matrix.

    Handles two common raw layouts:
        (num_tx, num_rx, num_chirps, num_samples)   - already separated
        (num_tx * num_rx, num_chirps, num_samples)  - channels flattened,
                                                        TX-major (tx*num_rx + rx)

    If your capture format differs, this is the only function that needs
    to change — everything downstream just consumes a clean 2-D matrix.
    """
    if raw_frame.ndim == 4:
        expected = (num_samples, num_chirps, num_rx, num_tx)
        if raw_frame.shape != expected:
            raise ValueError(
                f"raw_frame shape {raw_frame.shape} does not match "
                f"expected {expected}"
            )
        channel = raw_frame[:, :, rx, tx]
        channel = channel.T

    elif raw_frame.ndim == 3:
        expected_chan = num_tx * num_rx
        if raw_frame.shape[0] != expected_chan:
            raise ValueError(
                f"raw_frame first dim {raw_frame.shape[0]} does not match "
                f"num_tx*num_rx={expected_chan}"
            )
        chan_idx = tx * num_rx + rx
        channel = raw_frame[chan_idx, :, :]

    else:
        raise ValueError(
            f"Unsupported raw_frame.ndim={raw_frame.ndim}; expected 3 or 4."
        )

    if channel.shape != (num_chirps, num_samples):
        raise ValueError(
            f"Selected channel shape {channel.shape} != "
            f"expected (num_chirps, num_samples)=({num_chirps}, {num_samples})"
        )

    return channel.astype(np.complex128)


# ----------------------------------------------------------------------
# 3. Per-chirp DC offset removal
# ----------------------------------------------------------------------
def remove_dc_offset(chirp_matrix, enable=True):
    """
    Subtract the mean of each chirp (row) independently.

    RTL equivalent: an accumulate-then-subtract stage running once per
    chirp over the fast-time samples, resetting the accumulator each chirp.
    """
    if not enable:
        return chirp_matrix.copy()

    dc = np.mean(chirp_matrix, axis=1, keepdims=True)
    return chirp_matrix - dc


# ----------------------------------------------------------------------
# 4. Windowing
# ----------------------------------------------------------------------
def _get_window(window_type, length):
    if window_type is None:
        return np.ones(length)
    window_type = window_type.lower()
    if window_type == "hann":
        # periodic Hann, matches most RTL LUT-based implementations
        # (symmetric Hann would divide by N-1 instead of N)
        return np.hanning(length + 1)[:-1]
    if window_type == "hamming":
        return np.hamming(length)
    if window_type == "blackman":
        return np.blackman(length)
    raise ValueError(f"Unsupported window_type: {window_type}")


def apply_window(chirp_matrix, window_type, axis):
    """
    Apply a window function along the given axis.

    axis=1 -> fast-time (range) windowing, one window per chirp
    axis=0 -> slow-time (Doppler) windowing, one window per range bin
    """
    if window_type is None:
        return chirp_matrix.copy()

    length = chirp_matrix.shape[axis]
    win = _get_window(window_type, length)

    shape = [1, 1]
    shape[axis] = length
    win = win.reshape(shape)

    return chirp_matrix * win


# ----------------------------------------------------------------------
# 5. Range FFT
# ----------------------------------------------------------------------
def range_fft(chirp_matrix, fft_size, axis):
    """
    N-point FFT along the fast-time (range) axis.

    No fftshift here — range bins stay in natural FFT order (0 .. N-1),
    matching a typical RTL range-FFT core output before any reordering.
    """
    return np.fft.fft(chirp_matrix, n=fft_size, axis=axis)


# ----------------------------------------------------------------------
# 6. Zero-pad chirps (slow-time)
# ----------------------------------------------------------------------
def zero_pad_chirps(range_fft_matrix, pad_from, pad_to, axis):
    """
    Zero-pad the slow-time axis from pad_from to pad_to chirps.

    e.g. 255 captured chirps -> 256 chirps, appending (pad_to - pad_from)
    all-zero rows so the Doppler FFT can run as a power-of-two transform.
    """
    current = range_fft_matrix.shape[axis]
    if current != pad_from:
        raise ValueError(
            f"zero_pad_chirps expected input size {pad_from} along axis "
            f"{axis}, got {current}"
        )

    pad_amount = pad_to - pad_from
    if pad_amount < 0:
        raise ValueError(f"pad_to ({pad_to}) must be >= pad_from ({pad_from})")

    pad_width = [(0, 0)] * range_fft_matrix.ndim
    pad_width[axis] = (0, pad_amount)

    return np.pad(range_fft_matrix, pad_width, mode="constant", constant_values=0)


# ----------------------------------------------------------------------
# 7. Doppler FFT
# ----------------------------------------------------------------------
def doppler_fft(zero_padded_matrix, fft_size, axis, window_type=None):
    """
    N-point FFT along the slow-time (Doppler) axis.

    An optional window (window_type) can be applied along the same axis
    immediately before the transform, if the RTL Doppler stage windows too.
    """
    data = zero_padded_matrix
    if window_type is not None:
        data = apply_window(data, window_type=window_type, axis=axis)

    return np.fft.fft(data, n=fft_size, axis=axis)


# ----------------------------------------------------------------------
# 8. FFT shift
# ----------------------------------------------------------------------
def fft_shift(doppler_fft_matrix, axis, enable=True):
    """
    Shift the zero-Doppler (zero-velocity) bin to the center of the axis.

    RTL equivalent: an address-remap / circular-buffer reorder stage on the
    Doppler output, not an actual computation.
    """
    if not enable:
        return doppler_fft_matrix.copy()
    return np.fft.fftshift(doppler_fft_matrix, axes=axis)


# ----------------------------------------------------------------------
# 9. Magnitude / Power
# ----------------------------------------------------------------------
def compute_magnitude(complex_matrix):
    """|z| = sqrt(I^2 + Q^2), same as an RTL CORDIC magnitude stage."""
    return np.abs(complex_matrix)


def compute_power(complex_matrix, in_db=True, ref=1.0, floor_db=-120.0):
    """
    Power = |z|^2, optionally converted to dB: 10*log10(power / ref^2).

    floor_db clamps the output so that empty/zero bins (e.g. the zero-padded
    Doppler rows before windowing, or unused range bins) don't produce -inf.
    """
    power_lin = np.abs(complex_matrix) ** 2

    if not in_db:
        return power_lin

    with np.errstate(divide="ignore"):
        power_db = 10.0 * np.log10(power_lin / (ref ** 2))

    power_db = np.maximum(power_db, floor_db)
    return power_db
