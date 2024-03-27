import torch


class CustomSchedule:
    """
    Scheduler used in the paper Attention Is All You Nedd (https://arxiv.org/abs/1706.03762)
    """

    def __init__(self, d_model, warmup_steps=4000):
        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = torch.tensor(step, dtype=torch.float32)
        d_model = torch.tensor(self.d_model, dtype=torch.float32).float()
        arg1 = torch.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return torch.rsqrt(d_model) * torch.minimum(arg1, arg2)
