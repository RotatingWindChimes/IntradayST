import sys
sys.path.append("..")
from utils import config
from algorithm.dqn import DoubleDQN
from utils.net import EncoderNet, QNet
from utils.functions import xavier_init_weights


class IntradaySTSingleDQN:
    def __init__(self, hidden_dim, state_dim, device, prob):
        """ DQN direction + PPO 连续数量日内交易模型

        :param hidden_dim: 隐状态大小
        :param state_dim: 状态大小
        :param device: 设备
        """
        self.device = device
        self.hidden_dim = hidden_dim

        self.state_dim = state_dim  # 状态大小

        # 编码网络
        self.encoder = EncoderNet(state_dim=self.state_dim, hidden_dim=self.hidden_dim, prob=prob).to(self.device)

        # DoubleDQN 网络
        self.double_q_net = QNet(encoder=self.encoder, hidden_dim=self.hidden_dim, action_dim=201).to(self.device)
        self.double_q_target = QNet(encoder=self.encoder, hidden_dim=self.hidden_dim, action_dim=201).to(self.device)

        self.encoder.apply(xavier_init_weights)
        self.double_q_target.apply(xavier_init_weights)
        self.double_q_net.apply(xavier_init_weights)

        # DoubleDQN判断涨跌
        self.dqn = DoubleDQN(q_net=self.double_q_net, target_q_net=self.double_q_target,
                             **config.__dict__["DQN_PARAMS"])

    def take_action(self, state):
        # 获得动作
        action = self.dqn.take_action(state)

        return action

    def update(self, transition_dict):
        self.dqn.update(transition_dict=transition_dict)
