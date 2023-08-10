import os
import pandas as pd
from tqdm import tqdm


def create_lob(file_name=None):
    """ 逐列累积归一化一张表

    :param file_name: 文件名, 格式为: train_data/20220104.csv
    :return: lob 数据
    """
    data = pd.read_csv("../dataset/" + file_name, header=0, index_col=0)[["bidprice", "askprice", "bidvolume",
                                                                          "askvolume"]]

    return data


def lob_data(label):
    """ 逐列累积标准化所有表

    :param label: 训练 or 测试
    :return: None
    """
    if not os.path.exists(os.path.join("../dataset", label.partition('_')[0] + "_lob")):
        os.makedirs(os.path.join("../dataset", label.partition('_')[0] + "_lob"))

    path = os.path.join("../dataset", label.partition('_')[0] + "_lob")

    print("Extract " + label.partition('_')[0] + " lob data.")
    file_list = sorted([file_name for file_name in os.listdir("../dataset/" + label) if file_name.endswith(".csv")])

    for file_name in tqdm(file_list):
        data = create_lob(file_name=os.path.join("../dataset", label, file_name))
        data = data.reset_index()
        data.to_csv(os.path.join(path, file_name), index=False)
    print("Done!")
