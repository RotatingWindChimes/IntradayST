import os
import sys
import torch
import pprint
from tqdm import tqdm
sys.path.append("..")
from utils import config
from daily import daily_test
from utils.create_envs import DiscreteEnvBuffer
from model.single_dqn import IntradaySTSingleDQN


class TraderSingleDQN:
    def __init__(self):
        self.test_env_buffer = DiscreteEnvBuffer(data_type="test")                           # 测试环境Buffer

        print("Create environments using all testing data.")
        self.test_env_buffer.create()                                                        # 创建训练环境

        self.agent = IntradaySTSingleDQN(state_dim=self.test_env_buffer.state_dim(),
                                         **config.__dict__["MODEL_PARAMS"])                  # SingleDQN模型

        # 日总资产变化
        self.daily_value_list = []

        # 每天的夏普率
        self.sharpe_ratio_dict = {}

        if not os.path.exists(os.path.join("..", "Images")):
            os.makedirs(os.path.join("..", "Images"))
        self.backtest_images = os.path.join("..", "Images")

    def load_params(self):
        """ 加载判断方向和数量的网络参数

        :return: None
        """
        params_path = os.path.join("..", "params", "best_single_dqn_params", "q_net.params")

        self.agent.double_q_net.load_state_dict(torch.load(params_path))

    def backtest(self):
        test_envs, test_dates = self.test_env_buffer.sample(date_size=len(self.test_env_buffer.buffer))

        for test_env, test_date in tqdm(zip(test_envs, test_dates)):
            # 测试日交易时刻
            tick_list = range(len(test_env.daily_data))

            # 测试日不做交易的总资产变化
            origin_shares_values = test_env.holdings_forever_value()

            # 初始化环境
            state = test_env.reset()
            done = False

            self.daily_value_list = []
            while not done:
                # 交易时刻的买卖方向和数量
                action = self.agent.take_action(state=state)

                # 执行买卖
                next_state, reward, done, info = test_env.step(action)

                # 保存总资产
                self.daily_value_list.append(info)

                state = next_state

            self.sharpe_ratio_dict[test_date] = daily_test(daily_value_list=self.daily_value_list, tick_list=tick_list,
                                                           origin_shares_values=origin_shares_values,
                                                           save_path=os.path.join(self.backtest_images,
                                                                                  str(test_date) + ".jpg"))
        pprint.pprint(self.sharpe_ratio_dict)


if __name__ == "__main__":
    trader = TraderSingleDQN()

    trader.load_params()

    trader.backtest()
