from __future__ import annotations

import torch
import torch.nn.functional as F


def compress_amplitude(signal: torch.Tensor, gamma: float) -> torch.Tensor:
    """Fixed, monotonic amplitude mapping without per-render normalization."""
    return -torch.expm1(-float(gamma) * signal.clamp_min(0.0))


def inverse_compress_amplitude(image: torch.Tensor, gamma: float) -> torch.Tensor:
    """Map a normalized image amplitude back to the renderer's signal domain."""
    return -torch.log1p(-image.clamp(0.0, 1.0 - 1.0e-6)) / max(
        float(gamma), 1.0e-6
    )


def display_amplitude(image: torch.Tensor) -> torch.Tensor:
    """Clamp an already calibrated image for PNG output only."""
    return image.detach().clamp(0.0, 1.0)


def ssim(prediction: torch.Tensor, target: torch.Tensor, window: int = 7) -> torch.Tensor:
    prediction = prediction[None, None]
    target = target[None, None]
    padding = window // 2
    mean_x = F.avg_pool2d(prediction, window, stride=1, padding=padding)
    mean_y = F.avg_pool2d(target, window, stride=1, padding=padding)
    variance_x = F.avg_pool2d(prediction.square(), window, 1, padding) - mean_x.square()
    variance_y = F.avg_pool2d(target.square(), window, 1, padding) - mean_y.square()
    covariance = F.avg_pool2d(prediction * target, window, 1, padding) - mean_x * mean_y
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mean_x * mean_y + c1) * (2 * covariance + c2)
    denominator = (mean_x.square() + mean_y.square() + c1) * (
        variance_x + variance_y + c2
    )
    return (numerator / denominator.clamp_min(1.0e-8)).mean()


def charbonnier(
    residual: torch.Tensor, epsilon: float, weight: torch.Tensor | None = None
) -> torch.Tensor:
    value = torch.sqrt(residual.square() + float(epsilon) ** 2)
    if weight is None:
        return value.mean()
    weight = weight.to(device=value.device, dtype=value.dtype).clamp(0.0, 1.0)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lambda_ssim: float,
    lambda_log: float = 0.2,
    charbonnier_epsilon: float = 1.0e-3,
    log_gain: float = 20.0,
    weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Robust amplitude loss that preserves the absolute q-signal scale."""
    prediction = prediction.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    robust = charbonnier(prediction - target, charbonnier_epsilon, weight)
    log_prediction = torch.log1p(float(log_gain) * prediction)
    log_target = torch.log1p(float(log_gain) * target)
    log_robust = charbonnier(log_prediction - log_target, charbonnier_epsilon, weight)
    ssim_prediction = prediction
    if weight is not None:
        weight = weight.to(device=prediction.device, dtype=prediction.dtype).clamp(0.0, 1.0)
        ssim_prediction = weight * prediction + (1.0 - weight) * target.detach()
    ssim_value = ssim(ssim_prediction, target)
    loss = robust + float(lambda_log) * log_robust + float(lambda_ssim) * (1.0 - ssim_value)
    return loss, {
        "charbonnier": robust,
        "log_charbonnier": log_robust,
        "ssim": ssim_value,
    }


def support_entropy(probability: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    probability = probability.clamp(eps, 1.0 - eps)
    return -(
        probability * probability.log()
        + (1.0 - probability) * (1.0 - probability).log()
    ).mean()
