import os
import sys
import torch
from tqdm import tqdm
from argparse import ArgumentParser
sys.path.append("..")
from utils import config
from utils.create_envs import DiscreteEnvBuffer
from model.single_reinforce import IntradaySTSingleREIN
from trader.daily import daily_test, plot_return, print_action


class SingleReinTrainer:
    def __init__(self, time_steps):
        """ 训练 reinforce 模型

        :param time_steps: 迭代训练次数
        """
        self.time_steps = time_steps                                                         # 迭代次数
        self.train_env_buffer = DiscreteEnvBuffer(data_type="train", is_train=True)          # 训练环境Buffer
        self.test_buffer = DiscreteEnvBuffer(data_type="test", is_train=False)               # 测试环境Buffer
        self.train_buffer = DiscreteEnvBuffer(data_type="train", is_train=False)

        print("Create environments using all training data.")
        self.train_env_buffer.create()                                                       # 创建训练环境
        self.train_envs, self.train_dates = self.train_env_buffer.sample_all()

        print("Create environments using all testing data.")                                 # 创建测试环境
        self.test_buffer.create()
        self.envs_test, self.dates_test = self.test_buffer.sample(date_size=10)              # 10天验证
        self.train_buffer.create()
        self.envs_train, self.dates_train = self.train_buffer.sample(date_size=5)

        # reinforce 模型
        self.agent = IntradaySTSingleREIN(state_dim=self.train_env_buffer.state_dim(),
                                          **config.__dict__["MODEL_PARAMS"])

        # 回报
        self.return_list = []

        # 参数路径
        self.params_path = os.path.join("..", "params", "single_reinforce")
        if not os.path.exists(self.params_path):
            os.makedirs(self.params_path)

    def train(self):
        print("Start Train.")
        for i in tqdm(range(self.time_steps)):
            # 初始化本轮奖励
            episode_return = 0

            # 本轮迭代中用到的环境, 抽样5个环境, 生成序列
            # envs, random_dates = self.train_env_buffer.sample(date_size=20)
            envs, random_dates = self.train_envs, self.train_dates

            length = len(envs)

            # 每个环境生成一条序列
            for env in tqdm(envs):

                # print(f"Using {random_date}'s data for simulation.")

                # 初始化环境
                state = env.reset()
                done = False

                # 记录本轮序列的轨迹
                transition_dict = {"states": [], "rewards": [], "actions": [], "dones": [], "next_states": []}

                # 得到本轮序列轨迹
                while not done:
                    action = self.agent.take_action(state=state)
                    next_state, reward, done, info, volume = env.step(action)

                    transition_dict["states"].append(state)
                    transition_dict["rewards"].append(reward)
                    transition_dict["actions"].append(action)
                    transition_dict["dones"].append(done)
                    transition_dict["next_states"].append(next_state)

                    state = next_state

                    episode_return += reward

                # 利用本轮序列学习
                self.agent.update(transition_dict=transition_dict)

            self.return_list.append(float(episode_return) / length)

            print("Episode {}: reward {}".format(i, self.return_list[-1]))

            model_path = os.path.join(self.params_path, str(i + 1) + "-Iterations")
            if not os.path.exists(model_path):
                os.makedirs(model_path)

            self.save_model(model_path=model_path)
            self.test_model(model_path=model_path)

            if (i + 1) % 10 == 0:
                plot_return(self.return_list, save_path=os.path.join(self.params_path, str(i + 1) + "-Returns.jpg"))

    def save_model(self, model_path):
        torch.save(self.agent.policy_net.state_dict(), os.path.join(model_path, "policy_net.params"))

    def test_model(self, model_path):
        for data_type, dates, envs in [("train", self.dates_train, self.envs_train),
                                       ("test", self.dates_test, self.envs_test)]:

            if not os.path.exists(os.path.join(model_path, data_type)):
                os.makedirs(os.path.join(model_path, data_type))

            for date, env in zip(dates, envs):
                # 动作列表
                action_list = []
                # 实际成交量列表
                volume_list = []
                # 实际奖励
                reward_list = []
                # 仓位列表
                holdings_list = []
                # 账户总资产变化
                daily_value_list = []

                # 初始化环境
                state = env.reset()
                done = False
                # daily_value_list.append(origin_shares_values[0])

                while not done:
                    # 交易时刻的买卖方向和数量
                    action = self.agent.take_action(state=state)
                    # 执行买卖
                    next_state, reward, done, info, volume = env.step(action)

                    # 记录本轮买卖的初始仓位、理论成交量、实际成交量、奖励
                    holdings_list.append(state[1] * env.initial_stock_num)

                    action_list.append(action)  # 时刻 t 动作
                    volume_list.append(volume)  # 时刻 t 成交量
                    reward_list.append(reward)  # 动作 t 奖励

                    # 保存总资产
                    daily_value_list.append(info)  # 动作 t 后资产 (时刻 t+1)

                    # 状态改变
                    state = next_state

                # 不做交易的总资产变化
                origin_shares_values = env.holdings_forever_value()

                # 交易时刻
                tick_list = range(1, len(origin_shares_values))
                print_action(action_list=action_list, volume_list=volume_list, reward_list=reward_list,
                             holdings_list=holdings_list, env=env,
                             save_path=os.path.join(model_path, data_type, str(date) + ".csv"))

                daily_test(daily_value_list=daily_value_list, tick_list=tick_list,
                           origin_shares_values=origin_shares_values, initial_share_value=1e6,
                           save_path=os.path.join(model_path, data_type, str(date) + "-images.jpg"))


if __name__ == "__main__":
    # 迭代次数
    args = ArgumentParser(description="Iteration Timesteps")
    args.add_argument("--time_steps", "-time", default=20, type=int)
    options = args.parse_args()

    trainer = SingleReinTrainer(time_steps=options.time_steps)

    trainer.train()
