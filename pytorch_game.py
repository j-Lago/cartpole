from game import GAMESTATE, Game
import torch
import numpy as np
import assets
from collections import namedtuple, deque
import random


class ptGame(Game):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.player_key = 'p2'
        self.steps_count = 0
        self.actions_size = 5
        self.states_size = 9

    def get_state(self):
        player = self.players[self.player_key]
        model = player.model
        state = np.array([model.y[0][0], model.y[0][1],
                          model.y[1][0], model.y[1][1],
                          model.y[2][0], model.y[2][1],
                          model.y[3][0], model.y[3][1], player.fuel])
        return torch.Tensor(torch.tensor(np.array([state]), dtype=torch.float))

    def reset_system(self):
        self.steps_count = 0
        self.reset()
        self.state = GAMESTATE.RUN
        return self.get_state()

    def simulate_system(self, action, verbose=0):
        self.steps_count += 1
        axis_input = (0., 1., -1., 0.5, -0.5)[action]
        done = self.ia_step({self.player_key: axis_input})
        reward = self.players[self.player_key].reward - abs(axis_input)*0.1
        if done:
            reward -= 100
        score = self.players[self.player_key].score
        if done and verbose == 2:
            print(f'{self.state}, {score=}')

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


class ReplayMemory():
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) > batch_size


class Agent():
    def __init__(self, e_start, e_end, e_decay, num_actions, device):
        self.current_step = 0
        self.current_rate = e_start
        self.num_actions = num_actions
        self.device = device
        self.e_start = e_start
        self.e_end = e_end
        self.e_decay = 1-e_decay

    def select_action(self, state, police_net):
        if self.current_rate > self.e_end:
            self.current_rate *= self.e_decay
        else:
            self.current_rate = self.e_end
        #self.current_rate = self.e_end + (self.e_start - self.e_end) * np.exp(-1. * self.current_step * self.e_decay)
        #self.current_step += 1

        if self.current_rate > random.random():
            action = torch.tensor([random.randrange(self.num_actions)]).to(self.device)
        else:
            with torch.no_grad():
                action = police_net(state).argmax(dim=1).to(self.device)

        return action


Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)


def extract_tensors(experiences):
    batch = Experience(*zip(*experiences))
    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)
    return (t1, t2, t3, t4)


class Qvalues():
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = device

    @staticmethod
    def get_current(police_net, states, actions):
        return police_net(states).gather(dim=1, index=actions.unsqueeze(-1))
        #return police_net(states).gather(dim=1, index=actions.unsqueeze(-1)).squeeze()


    @staticmethod
    def get_next(target_net, next_states, device):
        final_state_locations = next_states.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values


def get_dims_from_weights(weights) -> tuple:
    dims_in, dims_out  = [], []
    for key, value in weights.items():
        if key.endswith('weight'):
            dims_in.append(value.size(1))
            dims_out.append(value.size(0))
    dims_in.append(dims_out[-1])
    return tuple(dims_in)


def create_game(render=False) -> ptGame:
    return ptGame(name='CartPole',
                  window_size=(1600, 900),
                  fps=60,
                  sounds=assets.sounds,
                  fonts=assets.fonts,
                  images=assets.images,
                  game_duration=45,
                  max_power=18,
                  DO_NOT_RENDER=not render,
                  STEP_BY_STEP=True,
                  training_mode=True,
                  )