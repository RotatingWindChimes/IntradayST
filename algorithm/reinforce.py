import torch
import numpy as np
import torch.optim as optim


class REINFORCE:
    def __init__(self, policy_net, learning_rate, gamma, device):
        """ REINFORCE算法类

        :param policy_net: 策略网络
        :param learning_rate: 学习率
        :param gamma: 折扣因子
        :param device: 设备
        """
        self.gamma = gamma
        self.device = device
        self.policy_net = policy_net

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    def take_action(self, state):
        """ 根据PolicyNet的分布采样动作

        :param state: 状态
        :return: 动作
        """
        state = torch.tensor(state, dtype=torch.float32).to(device=self.device)

        probs = self.policy_net(state).detach().cpu().numpy().reshape(-1)

        actions = np.arange(len(probs))

        action = np.random.choice(a=actions, size=1, p=probs)

        return action.item()

    """
    def take_action(self, state):

        state = torch.tensor(state, dtype=torch.float).to(device=self.device)
        probs = self.policy_net(state).detach().cpu().numpy().reshape(-1)
        max_prob = np.max(probs)

        if max_prob >= 0.7:
            return np.argmax(probs)
        else:
            return (len(probs) - 1) / 2
    """

    def update(self, transition_dict):
        """ 根据一条轨迹更新一次策略网络

        :param transition_dict: 一条轨迹序列
        :return: None
        """
        states = transition_dict["states"]
        rewards = transition_dict["rewards"]
        actions = transition_dict["actions"]

        self.optimizer.zero_grad()
        psi = 0

        for i in reversed(range(len(rewards))):
            state = torch.tensor(states[i], dtype=torch.float32).to(device=self.device)
            action = torch.tensor(actions[i], dtype=torch.int64).view(-1, 1).to(device=self.device)
            reward = rewards[i]

            psi = self.gamma * psi + reward

            log_prob = torch.log(self.policy_net(state).gather(1, action)).view(-1)

            loss = -log_prob * psi

            loss.backward()
        self.optimizer.step()
