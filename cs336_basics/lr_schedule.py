import math

def learning_rate_schedule(
    t: int,
    lr_max: float,
    lr_min: float,
    warmup_steps: int,
    cos_steps: int
) -> float:
    if t < warmup_steps:
        return t / warmup_steps * lr_max
    if t <= cos_steps:
        ratio = (t - warmup_steps) / (cos_steps - warmup_steps)
        cos = 1 + math.cos(ratio * math.pi)
        return lr_min + 0.5 * cos * (lr_max - lr_min)
    return lr_min
