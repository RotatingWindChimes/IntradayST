import torch
import torch.nn as nn
import torch.nn.functional as f


class EncoderNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, prob):
        """ 编码网络, 将原始状态压缩为隐状态

        :param state_dim: 原始状态大小
        :param hidden_dim: 隐状态大小
        """
        super(EncoderNet, self).__init__()
        self.encoder1 = nn.Linear(state_dim, hidden_dim)
        self.encoder2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=prob)

    def forward(self, x):
        # 一维变两维
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, dim=0)

        x = self.dropout(torch.tanh(self.encoder1(x)))
        return self.encoder2(x)


class QNet(nn.Module):
    def __init__(self, encoder, hidden_dim, action_dim):
        """ Q网络

        :param encoder: 编码网络
        :param hidden_dim: 隐状态大小
        :param action_dim: 动作空间大小
        """
        super(QNet, self).__init__()
        self.encoder = encoder
        self.action_dim = action_dim
        self.fc = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # 一维变两维
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, dim=0)

        x = f.relu(self.encoder(x))
        return self.fc(x)
