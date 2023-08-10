import torch
import torch.nn.functional as f
import numpy as np
import torch.optim as optim


class DoubleDQN:
    def __init__(self, target_q_net, q_net, learning_rate, epsilon, gamma, device, target_update):
        """ DoubleDQN算法类

        :param target_q_net: 目标网络
        :param q_net: Q网络
        :param learning_rate: Q网络的学习率
        :param epsilon: epsilon-Greedy策略选择动作
        :param gamma: 折扣因子
        :param device: 设备
        :param target_update: 目标网络更新频率
        """

        self.target_q_net = target_q_net
        self.q_net = q_net

        self.action_dim = self.q_net.action_dim
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.device = device
        self.target_update = target_update

        self.count = 0
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        self.return_list = []

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.action_dim)
        else:
            state = torch.tensor(np.array(state), dtype=torch.float32).to(device=self.device)
            action = torch.argmax(self.q_net(state)).item()

        return action

    def update(self, transition_dict):
        self.count += 1

        states = torch.tensor(transition_dict["states"], dtype=torch.float32).to(device=self.device)
        actions = torch.tensor(transition_dict["actions"], dtype=torch.int64).reshape(-1, 1).to(device=self.device)
        rewards = torch.tensor(transition_dict["rewards"], dtype=torch.float32).reshape(-1, 1).to(device=self.device)
        next_states = torch.tensor(transition_dict["next_states"], dtype=torch.float32).to(device=self.device)
        dones = torch.tensor(transition_dict["dones"], dtype=torch.float32).reshape(-1, 1).to(device=self.device)

        q_values = self.q_net(states).gather(1, actions)

        best_actions = self.q_net(next_states).max(1)[1].view(-1, 1)

        q_targets = rewards + self.gamma * self.target_q_net(next_states).gather(1, best_actions) * (1 - dones)

        dqn_loss = f.mse_loss(q_values, q_targets)

        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
