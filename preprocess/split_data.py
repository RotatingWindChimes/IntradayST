import os
from tqdm import tqdm
from pull_daily_data import pull_data


# 找到在所有日期中的未停牌的股票
def get_stock(feature_name="features23"):
    """ 找到对于给定的因子组, 所有天数都未曾停牌的股票

    :param feature_name: 因子组名字
    :return: 第一支股票
    """
    feature_path = "/data/research/zlalgor/ProcessedData2/" + feature_name + "/group0"

    stock_list = []

    for idx, date in enumerate(os.listdir(feature_path)):
        if idx == 0:
            stock_list = os.listdir(os.path.join(feature_path, date))
        else:
            stock_list = list(set(stock_list).intersection(set(os.listdir(os.path.join(feature_path, date)))))

    return stock_list[0]


# 提取股票的所有数据, 并划分训练集和测试集, 取前 80% 天的数据用于训练每日的强化学习智能体
def split_data(stock_name, feature_name="features23"):

    # 特征路径
    feature_path = "/data/research/zlalgor/ProcessedData2/" + feature_name + "/group0"

    # 日期列表
    date_list = sorted(os.listdir(feature_path))

    # 训练集大小
    train_size = int(len(date_list) * 0.8)

    train_date = date_list[:train_size]
    test_date = date_list[train_size:]

    if not os.path.exists("../dataset/train_data"):
        os.makedirs(os.path.join("..", "dataset", "train_data"))
    if not os.path.exists("../dataset/test_data"):
        os.makedirs(os.path.join("..", "dataset", "test_data"))

    print("Store train data.")
    for trd in tqdm(train_date):
        data_path = os.path.join("..", "dataset", "train_data", trd+".csv")
        pull_data(trd, stock_name=stock_name).to_csv(data_path, index=False)
    print("Done!")

    print("Store trader data.")
    for ted in tqdm(test_date):
        data_path = os.path.join("..", "dataset", "test_data", ted+".csv")
        pull_data(ted, stock_name=stock_name).to_csv(data_path, index=False)
    print("Done!")
