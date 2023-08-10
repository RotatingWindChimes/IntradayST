import torch

# 环境超参数
# 参数用在create_envs.py文件
ENV_PARAMS = {
    "buy_cost_pct": 1e-4,
    "sell_cost_pct": 1.5e-4,
    "bound": 100,
    "stock_base": 100,
    "initial_amount": 1e6,
    "delta": 0.1
}

# DQN模型超参数, DQN用作判断涨跌, 也可能用作生成 [0, 100] 的数量
# 参数用在model文件夹
DQN_PARAMS = {
    "learning_rate": 1e-3,
    "gamma": 0.98,
    "epsilon": 0.01,
    "target_update": 20,
    "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
}

# 经验池大小, 参数用在learn文件夹
BUFFER_PARAMS = {
    "capacity": 100000,
}

# 模型超参数
# 参数用在learn文件
MODEL_PARAMS = {
    "hidden_dim": 128,
    "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    "prob": 0.5
}

# 经验回放阈值以及批量大小
# 参数用在learn文件夹
TRAIN_PARAMS = {
    "minimal_size": 1000,
    "batch_size": 64
}
