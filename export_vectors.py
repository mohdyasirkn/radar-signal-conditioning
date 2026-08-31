"""
export_vectors.py
==================
RTL-compatible vector export for the Radar Range-Doppler Golden Reference
Model.

Every stage in the pipeline (radar_processing.py) produces either a real
array (e.g. magnitude, power) or a complex array (e.g. raw samples, FFT
outputs). This module converts those arrays into fixed-point, one-value-
per-line text files that a Verilog/VHDL testbench can load directly with
$readmemh / $readmemb (or an equivalent VHDL textio read).

Conventions
-----------
* Fixed point is Q(int_bits-frac_bits).(frac_bits), i.e.:
      int_bits  = TOTAL word width in bits (includes the sign bit)
      frac_bits = number of fractional bits within that word
  Example: int_bits=16, frac_bits=0  -> signed Q16.0 (plain 16-bit int)
           int_bits=16, frac_bits=15 -> signed Q1.15

* Complex arrays are split into two independent files, one for the real
  (I) part and one for the imaginary (Q) part, since that mirrors how a
  complex sample is almost always stored in RTL: two parallel words /
  two separate BRAMs, never an interleaved Python-style complex type.

* Multi-dimensional arrays are flattened in row-major (C) order, i.e. the
  last axis varies fastest. This matches the natural order an RTL memory
  gets written in when iterating with nested for-loops (outer loop over
  axis 0, inner loop over axis 1). If your RTL memory map iterates in a
  different order, flip the array (e.g. .T) before calling export_stage.

* Every output file starts with a header block of `//`-prefixed comment
  lines (Verilog $readmemh/$readmemb and standard VHDL textio both treat
  these as whitespace), documenting shape, format, and scaling so nothing
  gets lost between "the array" and "the file".

Public API
----------
    export_stage(array, stage_name, output_dir, int_bits, frac_bits,
                 fmt="hex", signed=True)

        Single entry point used by main.py. Dispatches internally based on
        whether `array` is real or complex.
"""

import os
import numpy as np


# ----------------------------------------------------------------------
# Fixed-point quantization
# ----------------------------------------------------------------------
def quantize_fixed_point(values, int_bits, frac_bits, signed=True):
    """
    Quantize a real-valued numpy array to fixed-point integer codes.

    Parameters
    ----------
    values : np.ndarray (real, float)
    int_bits : int
        Total word width in bits (includes the sign bit if signed=True).
    frac_bits : int
        Number of fractional bits within the word (Q(int_bits-frac_bits).frac_bits).
    signed : bool

    Returns
    -------
    np.ndarray of dtype int64
        Quantized integer codes, rounded-to-nearest and saturated to the
        representable range of the target word width. Saturation (rather
        than wraparound) is used deliberately: golden-model vectors should
        never silently overflow, since that would hide a scaling bug that
        the RTL would otherwise expose as a real overflow/saturation event.
    """
    scale = 2 ** frac_bits
    scaled = np.round(np.asarray(values, dtype=np.float64) * scale)

    if signed:
        code_min = -(2 ** (int_bits - 1))
        code_max = (2 ** (int_bits - 1)) - 1
    else:
        code_min = 0
        code_max = (2 ** int_bits) - 1

    n_sat_hi = np.count_nonzero(scaled > code_max)
    n_sat_lo = np.count_nonzero(scaled < code_min)
    if n_sat_hi or n_sat_lo:
        print(
            f"[WARN] quantize_fixed_point: saturated {n_sat_hi + n_sat_lo} "
            f"of {scaled.size} values (int_bits={int_bits}, "
            f"frac_bits={frac_bits}, signed={signed})"
        )

    clipped = np.clip(scaled, code_min, code_max)
    return clipped.astype(np.int64)


# ----------------------------------------------------------------------
# Integer code -> text formatting
# ----------------------------------------------------------------------
def _to_hex_string(code, int_bits):
    """Two's-complement hex string, zero-padded to ceil(int_bits/4) digits."""
    n_digits = (int_bits + 3) // 4
    mask = (1 << int_bits) - 1
    unsigned_code = int(code) & mask
    return format(unsigned_code, f"0{n_digits}x")


def _to_bin_string(code, int_bits):
    """Two's-complement binary string, zero-padded to int_bits digits."""
    mask = (1 << int_bits) - 1
    unsigned_code = int(code) & mask
    return format(unsigned_code, f"0{int_bits}b")


def _to_dec_string(code, int_bits):
    """Plain signed/unsigned decimal string (no padding, no two's complement)."""
    return str(int(code))


_FORMATTERS = {
    "hex": _to_hex_string,
    "bin": _to_bin_string,
    "dec": _to_dec_string,
}


def format_codes(codes, int_bits, fmt):
    """Vectorized-ish formatting of an int array to a list of text lines."""
    if fmt not in _FORMATTERS:
        raise ValueError(
            f"Unsupported fmt='{fmt}'. Supported formats: {list(_FORMATTERS)}"
        )
    formatter = _FORMATTERS[fmt]
    return [formatter(c, int_bits) for c in codes.flatten(order="C")]


# ----------------------------------------------------------------------
# File writing
# ----------------------------------------------------------------------
def _write_lines_with_header(path, lines, header_lines):
    with open(path, "w") as f:
        for h in header_lines:
            f.write(f"// {h}\n")
        f.write("//\n")
        for line in lines:
            f.write(line + "\n")


def _build_header(stage_name, array, part_label, int_bits, frac_bits, fmt, signed):
    scale = 2 ** frac_bits
    return [
        f"Stage        : {stage_name}{part_label}",
        f"Shape        : {array.shape} (flattened row-major / C order)",
        f"Num elements : {array.size}",
        f"Format       : {fmt}",
        f"Word width   : {int_bits} bits",
        f"Frac bits    : {frac_bits} (scale = 2^{frac_bits} = {scale})",
        f"Signed       : {signed}",
        "Fixed point  : Q(int_bits-frac_bits).(frac_bits) -> "
        f"Q{int_bits - frac_bits}.{frac_bits}",
        "Load in Verilog with $readmemh / $readmemb as appropriate.",
    ]


# ----------------------------------------------------------------------
# Per-part export (real-valued array -> one file)
# ----------------------------------------------------------------------
def _export_real_part(array, stage_name, part_label, output_dir,
                       int_bits, frac_bits, fmt, signed):
    """
    Quantize and write a single real-valued array to
    <output_dir>/<stage_name><part_label>.txt

    part_label is "" for a genuinely real stage, or "_I" / "_Q" for the
    real/imag parts of a complex stage.
    """
    codes = quantize_fixed_point(array, int_bits, frac_bits, signed=signed)
    lines = format_codes(codes, int_bits, fmt)
    header = _build_header(stage_name, array, part_label,
                            int_bits, frac_bits, fmt, signed)

    ext = "txt"
    filename = f"{stage_name}{part_label}.{ext}"
    path = os.path.join(output_dir, filename)

    _write_lines_with_header(path, lines, header)
    print(f"[EXPORT] {filename:<30s} n={array.size:<8d} -> {path}")
    return path


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------
def export_stage(array, stage_name, output_dir, int_bits, frac_bits,
                  fmt="hex", signed=True):
    """
    Export one pipeline-stage array as RTL-compatible text file(s).

    Real arrays  (dc-removed... no wait, magnitude/power/etc.)
        -> <output_dir>/<stage_name>.txt

    Complex arrays (raw samples, windowed data, range FFT, Doppler FFT, ...)
        -> <output_dir>/<stage_name>_I.txt   (real part)
        -> <output_dir>/<stage_name>_Q.txt   (imag part)

    Parameters
    ----------
    array : np.ndarray
        Real or complex array from any radar_processing.py stage.
    stage_name : str
        Base filename (no extension), e.g. "04_range_fft".
    output_dir : str
        Directory to write into (typically cfg.TXT_DIR).
    int_bits : int
        Total fixed-point word width in bits.
    frac_bits : int
        Fractional bit count for the Qm.n fixed-point representation.
    fmt : str
        "hex", "bin", or "dec".
    signed : bool
        Whether to use a signed (two's complement) representation.

    Returns
    -------
    list[str]
        Paths of the file(s) written.
    """
    os.makedirs(output_dir, exist_ok=True)

    if np.iscomplexobj(array):
        i_path = _export_real_part(
            array.real, stage_name, "_I", output_dir,
            int_bits, frac_bits, fmt, signed,
        )
        q_path = _export_real_part(
            array.imag, stage_name, "_Q", output_dir,
            int_bits, frac_bits, fmt, signed,
        )
        return [i_path, q_path]

    path = _export_real_part(
        array, stage_name, "", output_dir,
        int_bits, frac_bits, fmt, signed,
    )
    return [path]


# ----------------------------------------------------------------------
# Convenience: export several stages in one call (optional helper, not
# required by main.py's current loop, but handy for ad-hoc scripts/tests)
# ----------------------------------------------------------------------
def export_all_stages(stages_dict, output_dir, int_bits, frac_bits,
                       fmt="hex", signed=True):
    """
    stages_dict : dict[str, np.ndarray]
        e.g. {"01_selected_channel": chirp_matrix, "04_range_fft": range_fft, ...}

    Returns
    -------
    dict[str, list[str]]
        stage_name -> list of file paths written for that stage.
    """
    written = {}
    for stage_name, array in stages_dict.items():
        written[stage_name] = export_stage(
            array, stage_name, output_dir, int_bits, frac_bits, fmt, signed
        )
    return written


if __name__ == "__main__":
    # Quick self-test / usage example, not part of the pipeline.
    demo_dir = "outputs/txt_demo"
    real_stage = np.array([[0.5, -0.25], [1.0, -1.0]])
    complex_stage = np.array([[1 + 2j, -3 - 4j], [0.5 - 0.5j, -0.1 + 0.1j]])

    export_stage(real_stage, "demo_power", demo_dir,
                 int_bits=16, frac_bits=8, fmt="hex", signed=True)
    export_stage(complex_stage, "demo_range_fft", demo_dir,
                 int_bits=16, frac_bits=8, fmt="hex", signed=True)
