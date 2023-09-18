import sys
sys.path.append("..")
from utils import config
from algorithm.actor_critic import ActorCritic
from utils.functions import xavier_init_weights
from utils.net import EncoderNet, PolicyNet, ValueNet


class IntradaySTActorCritic:
    def __init__(self, hidden_dim, state_dim, action_mid, device, prob):
        """ ActorCritic日内交易模型 (201分类)

        :param hidden_dim: 隐状态大小
        :param state_dim: 状态大小
        :param action_mid: 动作大小
        :param device: 设备
        :param prob: Encoder参数
        """
        self.device = device
        self.hidden_dim = hidden_dim
        self.action_mid = action_mid

        self.state_dim = state_dim  # 状态大小

        # 编码网络
        self.encoder = EncoderNet(state_dim=self.state_dim, hidden_dim=self.hidden_dim, prob=prob).to(self.device)

        # Actor-Critic 网络
        self.actor = PolicyNet(encoder=self.encoder, hidden_dim=self.hidden_dim,
                               action_dim=2*self.action_mid+1).to(self.device)
        self.critic = ValueNet(encoder=self.encoder, hidden_dim=self.hidden_dim).to(self.device)

        self.encoder.apply(xavier_init_weights)
        self.actor.apply(xavier_init_weights)
        self.critic.apply(xavier_init_weights)

        # Actor-Critic
        self.actor_critic = ActorCritic(policy_net=self.actor, value_net=self.critic, **config.__dict__["AC_PARAMS"])

    def take_action(self, state):
        # 获得动作
        action = self.actor_critic.take_action(state)

        return action

    def update(self, transition_dict):
        self.actor_critic.update(transition_dict=transition_dict)
