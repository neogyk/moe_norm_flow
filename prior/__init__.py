import torch

class PriorDistribution(torch.nn.Module):

    def __init__(self, in_dim) -> None:
        super().__init__()
        self.init_distribution = torch.random(in_dim)
    
    def sample(self, in_dim):
        samples = self.init_distribution
        return sampled

    def get_statistics(self, x):
        
    def get_likelihood(self, x):
        return 