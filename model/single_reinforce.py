import sys
sys.path.append("..")
from utils import config
from algorithm.reinforce import REINFORCE
from utils.net import EncoderNet, PolicyNet
from utils.functions import xavier_init_weights


class IntradaySTSingleREIN:
    def __init__(self, hidden_dim, state_dim, device, action_mid, prob):
        """ REINFORCE日内交易模型 (201分类)

        :param hidden_dim: 隐状态大小
        :param action_mid: 动作大小
        :param state_dim: 状态大小
        :param device: 设备
        :param prob: Encoder参数
        """
        self.device = device
        self.hidden_dim = hidden_dim

        self.state_dim = state_dim  # 状态大小
        self.action_mid = action_mid

        # 编码网络
        self.encoder = EncoderNet(state_dim=self.state_dim, hidden_dim=self.hidden_dim, prob=prob).to(self.device)

        # Policy网络
        self.policy_net = PolicyNet(encoder=self.encoder, hidden_dim=self.hidden_dim,
                                    action_dim=2*self.action_mid+1).to(self.device)

        self.encoder.apply(xavier_init_weights)
        self.policy_net.apply(xavier_init_weights)

        # REINFORCE
        self.reinforce = REINFORCE(policy_net=self.policy_net, **config.__dict__["REINFORCE_PARAMS"])

    def take_action(self, state):
        # 获得动作
        action = self.reinforce.take_action(state)

        return action

    def update(self, transition_dict):
        self.reinforce.update(transition_dict=transition_dict)
