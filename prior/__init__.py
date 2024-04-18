import torch

class PriorDistribution(torch.distributions.distribution.Distribution):

    def __init__(self, in_dim:int) -> None:
        super().__init__()
        self.init_distribution = torch.random(in_dim)
    
    def sample(self, in_dim:int):
        samples = self.init_distribution.sample()
        return samples

    def get_statistics(self, x):
        return
    
    def get_likelihood(self, x):
        return 