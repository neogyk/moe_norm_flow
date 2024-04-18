from turtle import forward
from uu import Error
import torch


class NormalizingFlow(torch.nn.Module):
    def __init__(
        self, n_flows: int, in_dim: int, out_dim: int, prior_dist: None
    ) -> None:
        super().__init__()

        self.n_flows: int = n_flows
        self.in_dim: int = in_dim
        self.out_dim: int = out_dim
        self.linear = torch.nn.Linear(in_features=in_dim, out_features=out_dim)
        self.prior_dist = prior_dist

        return

    def set_objective(
        self,
    ):
        return

    def get_likelihood():
        if len(self.jacobian_list) == self.n_flows:
            likelihood_score = torch.log(self.prior.dist) + torch.sum(
                torch.tensor(self.jacobian_list)
            )
        else:
            raise Error("Please run the forward pass")
        return likelihood_score

    def jacobian(self, func, x):
        jacobian = torch.autograd.functional.jacobian(func, x)
        # TODO add the decomposition if it's necessary

        return torch.det(jacobian)

    def tranformation(self, x):
        z = x + self.linear(x)

        return z

    def forward(self, x):
        self.jacobian_list = []

        for i in range(self.n_flows):
            x = self.tranformation(x)
            self.jacobian_list.append(self.jacobian(x))

        return x

    def inverse(self, x):
        z = x
        return z

