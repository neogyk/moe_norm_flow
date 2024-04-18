import torch


class MobiusFlow(torch.nn.Module):
    def __init__(self, c_init_value: torch.tensor) -> None:
        super().__init__(*args, **kwargs)
        self.w = torch.nn.Parameter(c_init_value, requires_grad=False)

    def forward(self, x: torch.tensor) -> torch.tensor:

        result = ((1 - torch.abs(self.w) ** 2) / torch.abs(x - self.w)) * (
            x - self.w
        ) - self.w
        return result

    def inverse(self, z: torch.tensor) -> torch.tensor:

        result = ((1 - torch.abs(self.w) ** 2) / torch.abs(z + self.w)) * (
            z + self.w
        ) + self.w
        return result
