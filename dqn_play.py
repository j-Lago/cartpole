from itertools import count
import assets
import torch
from pytorch_game import ptGame, DQN, get_dims_from_weights


def load_and_play(weights_path):
    game = ptGame(name='CartPole',
                  window_size=(1600, 900),
                  fps=60,
                  sounds=assets.sounds,
                  fonts=assets.fonts,
                  images=assets.images,
                  game_duration=5,
                  max_power=18,
                  DO_NOT_RENDER=True,
                  STEP_BY_STEP=True
                  )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    weights = torch.load(weights_path, weights_only=True)
    dims = get_dims_from_weights(weights)
    print(f'weights loaded from "{weights_path}": net dims: {dims}')

    police_net = DQN(dims, device)
    police_net.load_state_dict(weights)

    episode = 0
    while True:
        state = game.reset_system()
        print(f'{episode=}')
        done = False
        while not done:
            action = police_net(state).argmax(dim=1).to(device)
            state, reward, done = game.simulate_system(action.item())
        episode += 1


if __name__ == '__main__':
    load_and_play('meta/pendulo_trained_12.pth')
