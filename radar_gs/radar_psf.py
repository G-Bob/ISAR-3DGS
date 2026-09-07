from __future__ import annotations

from functools import lru_cache

import numpy as np
from matplotlib import colormaps
from scipy.ndimage import binary_dilation
from scipy.signal import fftconvolve


@lru_cache(maxsize=1)
def _hot_lut() -> np.ndarray:
    values = np.linspace(0.0, 1.0, 256)
    return np.round(colormaps["hot"](values)[:, :3] * 255.0).astype(np.uint8)


@lru_cache(maxsize=1)
def _packed_hot_lookup() -> np.ndarray:
    """Dense 24-bit RGB lookup; -1 marks colours outside the exact LUT."""
    lut = _hot_lut()
    packed_lut = (
        (lut[:, 0].astype(np.uint32) << 16)
        | (lut[:, 1].astype(np.uint32) << 8)
        | lut[:, 2].astype(np.uint32)
    )
    lookup = np.full(1 << 24, -1, dtype=np.int16)
    lookup[packed_lut] = np.arange(256, dtype=np.int16)
    return lookup


def decode_hot_index_rgb(rgb: np.ndarray) -> np.ndarray:
    """Invert the exact 256-level matplotlib hot map to its scalar LUT index."""
    lut = _hot_lut()
    packed = (
        (rgb[..., 0].astype(np.uint32) << 16)
        | (rgb[..., 1].astype(np.uint32) << 8)
        | rgb[..., 2].astype(np.uint32)
    )
    indices = _packed_hot_lookup()[packed]
    missing = indices < 0
    if missing.any():
        values, inverse = np.unique(packed[missing], return_inverse=True)
        colours = np.stack(
            ((values >> 16) & 255, (values >> 8) & 255, values & 255), axis=1
        ).astype(np.int32)
        distance = ((colours[:, None, :] - lut[None].astype(np.int32)) ** 2).sum(axis=2)
        indices = indices.copy()
        indices[missing] = distance.argmin(axis=1).astype(np.int16)[inverse]
    return indices.astype(np.uint8)


def decode_hot_normalized_db_rgb(rgb: np.ndarray) -> np.ndarray:
    """Recover the scalar value that was mapped from [-60, 0] dB to [0, 1]."""
    return decode_hot_index_rgb(rgb).astype(np.float32) / 255.0


def decode_hot_db_rgb(rgb: np.ndarray, floor_db: float = -60.0) -> np.ndarray:
    """Recover linear amplitude from the exact hot-mapped [-60, 0] dB image."""
    indices = decode_hot_index_rgb(rgb)
    normalized_db = indices.astype(np.float32) / 255.0
    amplitude = np.power(10.0, (floor_db * (1.0 - normalized_db)) / 20.0)
    amplitude[indices == 0] = 0.0
    return amplitude.astype(np.float32)


def encode_loader_amplitude(amplitude: np.ndarray, floor_db: float = -60.0) -> np.ndarray:
    """Reproduce dB clipping, hot mapping, then the legacy RGB-max loader."""
    amplitude = np.clip(amplitude.astype(np.float32), 0.0, 1.0)
    db = 20.0 * np.log10(np.maximum(amplitude, 10.0 ** (floor_db / 20.0)))
    normalized = np.clip((db - floor_db) / -floor_db, 0.0, 1.0)
    indices = np.round(normalized * 255.0).astype(np.uint8)
    rgb = _hot_lut()[indices]
    encoded = rgb.max(axis=2).astype(np.float32) / 255.0
    encoded[amplitude <= 0.0] = 0.0
    return encoded


@lru_cache(maxsize=16)
def amplitude_psf(height: int, width: int, beta_range: float, beta_doppler: float) -> np.ndarray:
    """Amplitude envelope of the exact separable Kaiser + two-IFFT processing chain."""
    window = np.kaiser(height, beta_range)[:, None] * np.kaiser(width, beta_doppler)[None, :]
    response = np.fft.fftshift(np.fft.ifft2(window))
    envelope = np.abs(response)
    return (envelope / envelope.max()).astype(np.float32)


def _first_null(profile: np.ndarray) -> int:
    centre = len(profile) // 2
    right = profile[centre:]
    for index in range(1, min(32, len(right) - 1)):
        if right[index] <= right[index - 1] and right[index] <= right[index + 1]:
            return index
    return 2


def psf_parts(psf: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    row_radius = _first_null(psf[:, psf.shape[1] // 2])
    col_radius = _first_null(psf[psf.shape[0] // 2, :])
    rr, cc = np.ogrid[: psf.shape[0], : psf.shape[1]]
    centre_r, centre_c = psf.shape[0] // 2, psf.shape[1] // 2
    main_mask = (
        ((rr - centre_r) / max(row_radius, 1)) ** 2
        + ((cc - centre_c) / max(col_radius, 1)) ** 2
        <= 1.0
    )
    sidelobe = psf.copy()
    sidelobe[main_mask] = 0.0
    return main_mask.astype(np.float32), sidelobe, (row_radius, col_radius)


def sidelobe_confidence_weight(
    amplitude: np.ndarray,
    psf: np.ndarray,
    source_threshold: float = 0.2,
    sidelobe_floor_db: float = -50.0,
    minimum_weight: float = 0.25,
) -> tuple[np.ndarray, dict]:
    """Conservative mask: downweight only sidelobes strongly explained by isolated bright peaks."""
    main, sidelobe, radii = psf_parts(psf)
    source_mask = amplitude >= source_threshold
    sources = np.where(source_mask, amplitude, 0.0)
    predicted = fftconvolve(sources, sidelobe, mode="same")
    floor = 10.0 ** (sidelobe_floor_db / 20.0)
    protected_mainlobe = binary_dilation(source_mask, iterations=max(radii))
    explained = (
        (predicted >= floor)
        & (predicted >= 0.5 * amplitude)
        & ~protected_mainlobe
    )
    weight = np.ones_like(amplitude, dtype=np.float32)
    weight[explained] = float(minimum_weight)
    return weight, {
        "masked_fraction": float(explained.mean()),
        "source_count": int(np.count_nonzero(sources)),
        "mainlobe_radii": [int(radii[0]), int(radii[1])],
    }


def conservative_sidelobe_clean(
    amplitude: np.ndarray,
    psf: np.ndarray,
    source_threshold: float = 0.2,
    gain: float = 0.1,
    maximum_removed_energy_fraction: float = 0.10,
) -> tuple[np.ndarray, dict]:
    """Nonnegative, energy-limited subtraction of only the measured PSF sidelobe envelope."""
    _, sidelobe, radii = psf_parts(psf)
    source_mask = amplitude >= source_threshold
    sources = np.where(source_mask, amplitude, 0.0)
    estimate = float(gain) * fftconvolve(sources, sidelobe, mode="same")
    total_energy = float(np.square(amplitude).sum())
    removed = np.minimum(amplitude, estimate)
    removed_energy = float(np.square(removed).sum())
    limit = maximum_removed_energy_fraction * total_energy
    scale = 1.0 if removed_energy <= limit or removed_energy == 0.0 else (limit / removed_energy) ** 0.5
    cleaned = np.maximum(amplitude - scale * estimate, 0.0)
    actual_removed = float(np.square(amplitude - cleaned).sum())
    return cleaned.astype(np.float32), {
        "source_count": int(np.count_nonzero(sources)),
        "mainlobe_radii": [int(radii[0]), int(radii[1])],
        "removed_energy_fraction": actual_removed / max(total_energy, 1.0e-12),
        "subtraction_scale": float(scale),
    }
