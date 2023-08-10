import os
import torch
import sys
import shutil
from tqdm import tqdm
from argparse import ArgumentParser
sys.path.append("..")
from utils import config
from utils.buffer import Buffer
from trader.daily import daily_test
from utils.create_envs import DiscreteEnvBuffer
from model.single_dqn import IntradaySTSingleDQN


class DqnDqnTrainer:
    def __init__(self, time_steps):
        """ 训练 dqn_dqn 模型

        :param time_steps: 迭代训练次数
        """
        self.time_steps = time_steps                                                         # 迭代次数
        self.train_env_buffer = DiscreteEnvBuffer(data_type="train")                         # 训练环境Buffer
        self.test_env_buffer = DiscreteEnvBuffer(data_type="test")                           # 测试环境Buffer

        print("Create environments using all training data.")
        self.train_env_buffer.create()                                                       # 创建训练环境

        print("Create single-day trader environments.")                                      # 创建单天测试环境
        self.test_env = self.test_env_buffer.create_one()

        # dqn_dqn 模型
        self.agent = IntradaySTSingleDQN(state_dim=self.train_env_buffer.state_dim(), **config.__dict__["MODEL_PARAMS"])

        # 训练中的轨迹buffer
        self.exp_buffer = Buffer(**config.__dict__["BUFFER_PARAMS"])

        # 回报
        self.return_list = []

        # 日总资产变化
        self.daily_value_list = []

        # 不同迭代次数的夏普率
        self.sharpe_ratio_dict = {}

        # 参数路径
        self.params_path = os.path.join("..", "params", "dqn_dqn")
        if not os.path.exists(self.params_path):
            os.makedirs(self.params_path)

    def train(self, minimal_size, batch_size):
        print("Start Train.")
        for i in tqdm(range(self.time_steps)):
            # 初始化本轮奖励
            episode_return = 0

            # 本轮迭代中用到的环境, 抽样5个环境, 生成序列
            envs, random_dates = self.train_env_buffer.sample(date_size=5)

            # 每个环境生成一条序列
            for env in envs:

                # print(f"Using {random_date}'s data for simulation.")

                # 初始化环境
                state = env.reset()
                done = False

                while not done:
                    action = self.agent.take_action(state=state)

                    next_state, reward, done, info = env.step(action)

                    self.exp_buffer.add(state, action, reward, next_state, done)

                    state = next_state

                    episode_return += reward

                    if self.exp_buffer.size() >= minimal_size:
                        states, actions, rewards, next_states, dones = self.exp_buffer.sample(batch_size)
                        transitions = {"states": states, "actions": actions, "rewards": rewards,
                                       "next_states": next_states, "dones": dones}

                        self.agent.update(transition_dict=transitions)

            self.return_list.append(float(episode_return) / 5)

            if (i+1) % 2 == 0:
                print("Episode {}: reward {}".format(i, self.return_list[-1]))

                model_path = os.path.join(self.params_path, str(i+1) + "-Iterations")
                if not os.path.exists(model_path):
                    os.makedirs(model_path)

                self.save_model(model_path=model_path)
                self.sharpe_ratio_dict[str(i+1)] = self.test_model(model_path=model_path)

    def save_model(self, model_path):
        torch.save(self.agent.double_q_net.state_dict(), os.path.join(model_path, "q_net.params"))

    def test_model(self, model_path):
        # 测试日交易时刻
        tick_list = range(len(self.test_env.daily_data))

        # 测试日不做交易的总资产变化
        origin_shares_values = self.test_env.holdings_forever_value()

        # 初始化环境
        state = self.test_env.reset()
        done = False

        self.daily_value_list = []
        while not done:
            # 交易时刻的动作
            action = self.agent.take_action(state=state)

            # 执行买卖
            next_state, reward, done, info = self.test_env.step(action=action)

            # 保存总资产
            self.daily_value_list.append(info)

            state = next_state

        return daily_test(daily_value_list=self.daily_value_list, tick_list=tick_list,
                          origin_shares_values=origin_shares_values, save_path=os.path.join(model_path, "images.jpg"))

    def select_model(self):
        best_sharpe_ration = sorted(self.sharpe_ratio_dict.items(), key=lambda x: x[1], reverse=True)[0]
        best_iterations = best_sharpe_ration[0]

        # 复制最佳网络参数
        best_folder = os.path.join("..", "params", "best_single_dqn_params")
        if not os.path.exists(best_folder):
            os.makedirs(best_folder)

        source_folder = os.path.join(self.params_path, str(best_iterations) + "-Iterations")
        file_list = os.listdir(source_folder)

        for filename in file_list:
            source_path = os.path.join(source_folder, filename)
            target_path = os.path.join(best_folder, filename)
            shutil.copy(source_path, target_path)

        print("Best Model Selected.")


if __name__ == "__main__":
    # 迭代次数
    args = ArgumentParser(description="Iteration Timesteps")
    args.add_argument("--time_steps", "-time", default=2000, type=int)
    options = args.parse_args()

    trainer = DqnDqnTrainer(time_steps=options.time_steps)

    trainer.train(**config.__dict__["TRAIN_PARAMS"])

    trainer.select_model()
