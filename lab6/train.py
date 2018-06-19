import argparse
import gym
import torch
from tqdm import trange

from models import DQN
from summary import Summary

game = 'CartPole-v0'

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=int, default=1000,
                    help='number of episode')
    parser.add_argument('--max_step', type=int, default=300,
                    help='maximum number of steps in 1 episode')
    parser.add_argument('--update_interval', type=int, default=50,
                    help='update target network interval')
    parser.add_argument('--test', type=int, default=100,
                    help='iterations for each test')
    parser.add_argument('--test_interval', type=int, default=0,
                    help='test interval')
    parser.add_argument('--replay_size', type=int, default=5000,
                    help='size of replay buffer')
    parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                    help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.95,
                    help='gamma')

    parser.add_argument('--log', type=str, default=None, help='path to csv log')

    return parser

def parse_opt(args=None):
    parser = get_parser()
    opt = parser.parse_args(args)
    return opt

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 1)
        m.bias.data.fill_(0)

def main():
    opt = parse_opt()
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    env = gym.make(game)

    agent = DQN(env, opt, device=device)
    agent.network.apply(weights_init)
    agent.sync_weight()
    
    progress = trange(opt.episode, ascii=True)
    summary = Summary()
    last_rewards = 0

    for episode in progress:
        # Training
        state = env.reset()
        for s in range(opt.max_step):
            # use epsilon-greedy in training
            action = agent.egreedy_action(state)
            next_state, reward, done, _ = env.step(action)
            loss = agent.perceive(state, action, reward, next_state, done)
            state = next_state
            if done:
                break

        summary.add(episode, 'loss', loss)

        # Testing
        if opt.test_interval > 0 and (episode+1) % opt.test_interval == 0:
            rewards = 0
            for t in trange(opt.test, ascii=True, leave=False):
                state = env.reset()
                for s in range(opt.max_step):
                    action = agent.action(state)
                    next_state, reward, done, _ = env.step(action)
                    state = next_state
                    rewards += reward
                    if done:
                        break

            if opt.test > 0:
                rewards /= opt.test

            last_rewards = rewards
            summary.add(episode, 'reward', rewards)

        progress.set_description('Loss: {:.4f} | Reward: {:2}'.format(loss, last_rewards))

    if opt.log:
        summary.write(opt.log)

if __name__ == '__main__':
    main()
