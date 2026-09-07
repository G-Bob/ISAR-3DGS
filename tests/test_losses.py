import torch

from radar_gs.losses import (
    compress_amplitude,
    inverse_compress_amplitude,
    reconstruction_loss,
)


def test_prediction_scale_is_not_normalized_away():
    target = torch.full((16, 16), 0.8)
    good = torch.full((16, 16), 0.8)
    weak = torch.full((16, 16), 0.08)
    good_loss, _ = reconstruction_loss(good, target, lambda_ssim=0.0)
    weak_loss, _ = reconstruction_loss(weak, target, lambda_ssim=0.0)
    assert weak_loss > good_loss


def test_fixed_amplitude_compression_is_bounded():
    values = compress_amplitude(torch.tensor([0.0, 1.0, 100.0]), gamma=1.0)
    assert torch.all(values >= 0.0)
    assert torch.all(values <= 1.0)
    assert values[0] < values[1] < values[2]


def test_amplitude_compression_round_trip():
    image = torch.tensor([0.0, 0.1, 0.5, 0.9])
    signal = inverse_compress_amplitude(image, gamma=1.7)
    restored = compress_amplitude(signal, gamma=1.7)
    assert torch.allclose(restored, image, atol=1.0e-6)
