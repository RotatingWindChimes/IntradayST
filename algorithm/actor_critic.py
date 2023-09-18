import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as f


class ActorCritic:
    def __init__(self, policy_net, value_net, actor_lr, critic_lr, gamma, device):
        self.policy_net = policy_net
        self.value_net = value_net
        self.gamma = gamma
        self.device = device

        self.policy_net_optimizer = optim.Adam(self.policy_net.parameters(), lr=actor_lr)
        self.value_net_optimizer = optim.Adam(self.value_net.parameters(), lr=critic_lr)

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
        state = torch.tensor(state, dtype=torch.float32).to(device=self.device)

        probs = self.policy_net(state).detach().cpu().numpy().reshape(-1)

        max_prob = np.max(probs)

        if max_prob >= 0.7:
            return np.argmax()
        else:
            return (len(probs) - 1) / 2
    """

    def update(self, transition_dict):
        states = torch.tensor(transition_dict["states"], dtype=torch.float32).to(self.device)
        actions = torch.tensor(transition_dict["actions"], dtype=torch.int64).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict["rewards"], dtype=torch.float32).view(-1, 1).to(self.device)
        dones = torch.tensor(transition_dict["dones"], dtype=torch.float32).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict["next_states"], dtype=torch.float32).to(self.device)

        td_target = rewards + self.gamma * self.value_net(next_states) * (1-dones)
        td_error = td_target - self.value_net(states)

        log_probs = torch.log(self.policy_net(states).gather(1, actions))

        actor_loss = torch.mean(-log_probs * td_error.detach())
        critic_loss = f.mse_loss(self.value_net(states), td_target.detach())

        self.policy_net_optimizer.zero_grad()
        self.value_net_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.policy_net_optimizer.step()
        self.value_net_optimizer.step()
