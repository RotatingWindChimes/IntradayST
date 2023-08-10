import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_mean_std(data_column):
    """ 计算data_column的均值, 只考虑非nan值的均值

    :param data_column: 一个数组
    :return: 均值
    """
    data_column = np.array(data_column)
    mean = data_column[~np.isnan(data_column)].mean()
    std = data_column[~np.isnan(data_column)].std()

    return mean, std


def total_mean_std(label):
    file_list = sorted([file_name for file_name in os.listdir("../dataset/" + label) if file_name.endswith(".csv")])

    print("Sample 80% data to calculate feature mean and feature std.")

    random.shuffle(file_list)

    sample_size = int(len(file_list) * 0.8)

    data = pd.DataFrame([])
    for file_name in tqdm(file_list[:sample_size]):
        temp_data = pd.read_csv("../dataset/" + label + "/" + file_name, header=0, index_col=0).drop(["bidprice",
                                                                                                      "askprice",
                                                                                                      "bidvolume",
                                                                                                      "askvolume"],
                                                                                                     axis=1)
        data = pd.concat([data, temp_data], axis=0)

    data_mean = []
    data_std = []

    cols = data.columns
    for column in cols:
        mean, std = get_mean_std(data[column])
        data_mean.append(mean)
        data_std.append(std)

    data_mean = np.array(data_mean).reshape(1, -1)
    data_std = np.array(data_mean).reshape(1, -1)

    return data_mean, data_std


def normalize(data_mean, data_std, file_name=None):
    """ 逐列累积标准化一张表, 只标准化非nan值, 标准化后nan值填0

    :param data_std: 每一列的标准差
    :param data_mean: 每一列的均值
    :param file_name: 文件名
    :return: 累积标准后的数据
    """
    data = pd.read_csv(file_name, header=0, index_col=0).drop(["bidprice", "askprice", "bidvolume", "askvolume"],
                                                              axis=1)

    data_copy = data.to_numpy()
    data_copy = (data_copy - data_mean) / data_std

    data = pd.DataFrame(data_copy, index=data.index, columns=data.columns)
    data = data.fillna(0)

    return data


def norm_data(label):
    """ 逐列累积标准化所有表

    :param label: 训练 or 测试
    :return: None
    """
    if not os.path.exists(os.path.join("../dataset", label.partition('_')[0] + "_processed")):
        os.makedirs(os.path.join("../dataset", label.partition('_')[0] + "_processed"))

    path = os.path.join("../dataset", label.partition('_')[0] + "_processed")

    data_mean, data_std = total_mean_std(label="train_data")

    print("Norm-Scaler the " + label.partition('_')[0] + " features.")
    file_list = sorted([file_name for file_name in os.listdir("../dataset/" + label) if file_name.endswith(".csv")])

    for file_name in tqdm(file_list):
        data = normalize(file_name=os.path.join("../dataset", label, file_name), data_mean=data_mean, data_std=data_std)
        data = data.reset_index()
        data.to_csv(os.path.join(path, file_name), index=False)

    print("Done!")
