import torch


class DQN(torch.nn.Module):
    def __init__(self, dims, device):
        super().__init__()
        self.device = device
        self.dims = dims

        self.l1 = torch.nn.Linear(in_features=dims[0], out_features=dims[1]).to(device)
        self.l2 = torch.nn.Linear(in_features=dims[1], out_features=dims[2]).to(device)
        self.l3 = torch.nn.Linear(in_features=dims[2], out_features=dims[3]).to(device)

    def forward(self, obs):
        obs=obs.to(self.device)
        i1 = torch.nn.functional.relu(self.l1(obs))
        i2 = torch.nn.functional.relu(self.l2(i1))
        q = self.l3(i2)
        return q


def get_dims_from_weights(weights) -> tuple:
    dims_in, dims_out  = [], []
    for key, value in weights.items():
        if key.endswith('weight'):
            dims_in.append(value.size(1))
            dims_out.append(value.size(0))
    dims_in.append(dims_out[-1])
    return tuple(dims_in)