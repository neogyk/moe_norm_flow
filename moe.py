from turtle import forward
import torch


class CommunicationMixterOfExperts(torch.nn.Module):
    def __init__(self):
        return

    def forward(x):
        return x


class SharedMoE(torch.nn.Module):
    def __init__(self):
        return

    def forward(self, x):
        return x


class Expert(torch.nn.Module):
    """ """

    # self.___global__ = None
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(in_features=in_dim, out_features=out_dim)
        self.___globals__ = None

    def forward(self, x):
        out = self.layer(x)
        out = torch.nn.functional.sigmoid(out)
        return out


class Routing(torch.nn.Module):
    """ """

    def __init__(self, in_dim: int, n_experts: int, k: int, bs: int) -> None:
        super().__init__()
        self.bs = bs
        self.W_g = torch.nn.Linear(in_features=in_dim, out_features=n_experts, bias=0)
        self.W_noise = torch.nn.Linear(
            in_features=in_dim, out_features=n_experts, bias=0
        )
        self.k: int = k

        return

    def forward(self, x: torch.tensor) -> torch.tensor:
        # the index of expert in the list where to pass input data;
        normal_dist = torch.normal(0.0, std=torch.tensor([1.0 for i in range(self.bs)]))
        out = torch.nn.functional.softmax(
            self.W_g(x) + torch.nn.functional.softplus(self.W_noise(x))
        )

        topk = torch.topk(out, k=self.k)
        return topk


class Aggregation(torch.nn.Module):
    """ """

    def __init__(self, agg_type="mean") -> None:
        super().__init__()
        self.agg_type = agg_type

    def forward(self, x: torch.tensor) -> torch.tensor:
        """ """
        if self.agg_type == "sum":
            out = torch.sum(x, dim=0)
        if self.agg_type == "mean":
            out = torch.mean(x, dim=0)
        if self.agg_type == "var":
            out = torch.sum(x, dim=0) * 1 / torch.sqrt(x.size(-1))
        return out


class MoE(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n: int, k: int, bs: int) -> None:
        super().__init__()
        self.k = k
        self.n = n
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bs: int = bs
        self.experts = [
            Expert(in_dim=self.in_dim, out_dim=self.out_dim) for i in range(self.n)
        ]
        self.routing = Routing(
            in_dim=self.in_dim, n_experts=self.n, k=self.k, bs=self.bs
        )
        self.aggregation = Aggregation()
        return

    def forward(self, x):
        topk = self.routing(x)
        values, indices = topk.values, topk.indices
        res = []
        # ToDo Selectively Get the x-token according the indices
        for idx in range(self.bs):
            selected = filter(
                lambda i: self.experts.index(i) in indices[idx], self.experts
            )
            _x = values[idx] @ torch.stack([e(x[idx]) for e in list(selected)])
            res.append(_x)
        x = torch.stack(res, dim=0)
        # each token passed to the corresponding expert defined by router]
        x = self.aggregation(x)
        return x


class InveseMOE(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.routing_matrix = routing_matrix
        self.experts = experts
        return

    def forward():
        return


class DepthMOE(torch.nn.Module):
    # Paper: https://arxiv.org/pdf/2404.02258.pdf
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        return

    def forward(self, x):
        return


if __name__ == "__main__":
    x = torch.rand(32, 4)
    moe = MoE(in_dim=4, out_dim=10, n=10, k=4, bs=32)
    result = moe(x)
    print(result)
