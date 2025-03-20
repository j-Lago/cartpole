import torch
from pytorch_game import DQN, get_dims_from_weights, create_game


def load_and_play(weights_path):
    game = create_game(render=True)
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
    load_and_play('meta/wb2_best_score.pth')
