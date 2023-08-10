import os
import random
import pandas as pd
from tqdm import tqdm
from collections import deque
from utils import config
from utils.env import IntradayEnvDiscrete


class DiscreteEnvBuffer:
    def __init__(self, data_type="train"):
        """ 根据指定类型创建离散环境buffer

        :param data_type: 类型, 可以为 train 或者 test
        """
        self.data_path = "../dataset/" + data_type
        self.file_list = [file_name for file_name in sorted(os.listdir(self.data_path + "_lob"))
                          if file_name.endswith(".csv")]   # 所有的csv文件
        self.buffer = deque(maxlen=len(self.file_list))

    def create(self):
        """ 根据所有csv文件 (features, lob data) 创建buffer

        :return: buffer
        """
        for file_name in tqdm(self.file_list):
            daily_data = pd.read_csv(self.data_path + "_processed/" + file_name, header=0, index_col=0)
            lob_data = pd.read_csv(self.data_path + "_lob/" + file_name, header=0, index_col=0)
            env = IntradayEnvDiscrete(daily_data=daily_data, lob_data=lob_data, **config.__dict__["ENV_PARAMS"])

            self.buffer.append(env)

    def create_one(self):
        """ 创建单个环境日

        :return: 单个环境
        """
        daily_data = pd.read_csv(self.data_path + "_processed/" + self.file_list[0], header=0, index_col=0)
        lob_data = pd.read_csv(self.data_path + "_lob/" + self.file_list[0], header=0, index_col=0)
        env = IntradayEnvDiscrete(daily_data=daily_data, lob_data=lob_data, **config.__dict__["ENV_PARAMS"])

        return env

    def sample(self, date_size):
        """ 采样

        :param date_size: 采样的日期大小
        :return: 返回采样到的环境列表, 对应的日期列表
        """
        random_indices = random.sample(range(len(self.buffer)), date_size)

        random_dates = [self.file_list[random_index].partition(".")[0] for random_index in random_indices]
        envs = [self.buffer[random_index] for random_index in random_indices]

        return envs, random_dates

    def state_dim(self):
        return self.buffer[0].state_dim
