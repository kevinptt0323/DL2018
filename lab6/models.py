import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

class ReplayMemory(object):
    """
    A cyclic list to store experience replay
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def append(self, data):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = data 
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class MLP(nn.Module):
    """
    Core network for DQN
    """
    def __init__(self, input_size, output_size, bias=True):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 32, bias=bias)
        self.fc2 = nn.Linear(32, output_size, bias=bias)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out

class DQN():
    def __init__(self, env, opt, device=torch.device('cpu')):
        self.env = env
        self.device = device
        self.memory = ReplayMemory(opt.replay_size)
        self.epsilon = 1
        self.gamma = opt.gamma
        self.update_interval = opt.update_interval
        self.batch_size = opt.batch_size
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.network = MLP(self.state_dim, self.action_dim).to(self.device)
        self.network_fixed = MLP(self.state_dim, self.action_dim).to(self.device)
        self.network_fixed.eval()
        self.optimizer = optim.Adam(self.network.parameters(), lr=opt.learning_rate)
        self.criterion = nn.MSELoss()
        self.train_counter = 0

    def sync_weight(self):
        self.network_fixed.load_state_dict(self.network.state_dict())

    def action(self, state):
        with torch.no_grad():
            input = torch.tensor([state], device=self.device, dtype=torch.float)
            output = self.network_fixed(input)
            result = output.argmax().item()
        return result

    def egreedy_action(self, state):
        if random.random() <= self.epsilon:
            result = random.randint(0, self.action_dim - 1)
        else:
            result = self.action(state)

        # epsilon decays
        self.epsilon = max(self.epsilon * 0.995, 0.1)

        return result

    def perceive(self, state, action, reward, next_state, done):
        # add to experience replay memory
        self.memory.append((state, action, reward, next_state, done))
        loss = 0

        if len(self.memory) >= self.batch_size:
            loss = self.train_network()

        return loss

    def train_network(self):
        minibatch = self.memory.sample(self.batch_size)
        state, action, reward, next_state, done = zip(*minibatch)

        with torch.no_grad():
            action = torch.tensor(list(action), device=self.device, dtype=torch.long)
            reward = torch.tensor(list(reward), device=self.device)
            done = torch.tensor(list(done), device=self.device, dtype=torch.float)

            input = torch.tensor(list(next_state), device=self.device, dtype=torch.float)
            targets = self.network_fixed(input)
            # target adds q-value if done == 0
            targets = reward + (1 - done) * targets.max(dim=1)[0] * self.gamma

        input = torch.tensor(list(state), device=self.device, dtype=torch.float)
        output = self.network(input)
        output = output.gather(1, action.unsqueeze(1)).squeeze(1)

        loss = self.criterion(output, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_counter += 1
        # update fixed Q-target
        if self.update_interval > 0 and self.train_counter % self.update_interval == 0:
            self.sync_weight()

        return loss.item()
