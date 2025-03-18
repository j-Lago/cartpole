from game import GAMESTATE, Game
import torch
import numpy as np


class ptGame(Game):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.player_key = 'p2'
        self.steps_count = 0

    def get_state(self):
        model = self.players[self.player_key].model
        state = np.array([model.y[0][0], model.y[0][1],
                          model.y[1][0], model.y[1][1],
                          model.y[2][0], model.y[2][1],
                          model.y[3][0], model.y[3][1]])
        return torch.Tensor(torch.tensor(np.array([state]), dtype=torch.float))

    def reset_system(self):
        self.steps_count = 0
        self.reset()
        self.state = GAMESTATE.RUN
        return self.get_state()

    def simulate_system(self, action, verbose=0):
        self.steps_count += 1
        axis_input = (0., 1., -1., 0.5, -0.5)[action]
        done = self.ia_step({'p2': axis_input})
        reward = self.players[self.player_key].reward
        score = self.players[self.player_key].score
        if done and verbose == 2:
            print(f'{self.state}, score: {score}')

        if verbose == 1:
            print(f'{self.steps_count=}, {axis_input=} -> {reward=}')

        reward = torch.tensor([reward], dtype=torch.float)
        next_state = self.get_state()
        return next_state, reward, done



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


def get_dims_from_weights(weights) -> tuple[int]:
        dims_in, dims_out  = [], []
        for key, value in weights.items():
            if key.endswith('weight'):
                dims_in.append(value.size(1))
                dims_out.append(value.size(0))
        dims_in.append(dims_out[-1])
        return tuple(dims_in)