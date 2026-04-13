import torch
import math


def _schedule_progress(step, total_steps):
    """Return clipped progress in [0, 1] over valid indices [0, total_steps - 1]."""
    step = torch.as_tensor(step, dtype=torch.float32)
    total_steps = torch.as_tensor(total_steps, dtype=torch.float32, device=step.device)
    denom = torch.clamp(total_steps - 1.0, min=1.0)
    return torch.clamp(step / denom, 0.0, 1.0)


def get_linear_schedule(total_steps, min=0.01):
    def linear_noise_schedule(step):
        progress = _schedule_progress(step, total_steps)
        return 1.0 + (min - 1.0) * progress

    return linear_noise_schedule


def get_cosine_schedule(total_steps, min=0.01, s=0.008, pow=2):
    def cosine_schedule(step):
        progress = _schedule_progress(step, total_steps)
        t = 1.0 - progress
        offset = 1 + s
        return (1. - min) * torch.cos(0.5 * math.pi * (offset - t) / offset) ** pow + min

    return cosine_schedule


def get_constant_schedule():
    def constant_schedule(step):
        return torch.tensor(1.)

    return constant_schedule
