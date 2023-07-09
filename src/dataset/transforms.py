"""Some extra transforms for video"""

import torch
import random
import numpy as np


def to_tensor(clip):
    """
    Cast tensor type to float, then permute dimensions from TxHxWxC to CxTxHxW, and finally divide by 255

    Parameters
    ----------
    clip : torch.tensor
        video clip
    """
    return clip.float().permute(3, 0, 1, 2) / 255.0


def normalize(clip, mean, std):
    """
    Normalise clip by subtracting mean and dividing by standard deviation

    Parameters
    ----------
    clip : torch.tensor
        video clip
    mean : tuple
        Tuple of mean values for each channel
    std : tuple
        Tuple of standard deviation values for each channel
    """
    clip = clip.clone()
    mean = torch.as_tensor(mean, dtype=clip.dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=clip.dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
    return clip


class NormalizeVideo:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, clip):
        return normalize(clip, self.mean, self.std)


class ToTensorVideo:
    def __init__(self):
        pass

    def __call__(self, clip):
        return to_tensor(clip)


class AddNoise(object):
    """Add SNR noise [-1, 1]
    """

    def __init__(self, noise, snr_levels=[-5, 0, 5, 10, 15, 20, 9999]):
        assert noise.dtype in [np.float32, np.float64], "noise only supports float data type"
        
        self.noise = noise
        self.snr_levels = snr_levels

    def get_power(self, clip):
        clip2 = clip.copy()
        clip2 = clip2 **2
        return np.sum(clip2) / (len(clip2) * 1.0)

    def __call__(self, signal):
        assert signal.dtype in [np.float32, np.float64], "signal only supports float32 data type"
        snr_target = random.choice(self.snr_levels)
        if snr_target == 9999:
            return signal
        else:
            # -- get noise
            start_idx = random.randint(0, len(self.noise)-len(signal))
            noise_clip = self.noise[start_idx:start_idx+len(signal)]

            sig_power = self.get_power(signal)
            noise_clip_power = self.get_power(noise_clip)
            factor = (sig_power / noise_clip_power ) / (10**(snr_target / 10.0))
            desired_signal = (signal + noise_clip*np.sqrt(factor)).astype(np.float32)
            return desired_signal


class NormalizeUtterance():
    """Normalize per raw audio by removing the mean and divided by the standard deviation
    """
    def __call__(self, signal):
        if (signal_std := np.std(signal)) == 0:
            return np.zeros_like(signal)

        signal_mean = np.mean(signal)
        return (signal - signal_mean) / signal_std
