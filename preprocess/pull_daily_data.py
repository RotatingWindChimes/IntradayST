import os
import numpy as np
import pandas as pd


def pull_group0_data(file_path):
    """ 提取 features23下特定日期的 group0 数据

    :param file_path: 路径, 精确到是 features23下哪只股哪一天
    :return: nTime DataFrame
    """
    date_data = np.fromfile(file_path, dtype=np.uint64).reshape(-1, 2)

    return pd.DataFrame(date_data, columns=["nTime", "lTime"])[["nTime"]]


def pull_group1_data(file_path):
    """提取 features23下特定日期的 group1 数据

    :param file_path: 路径, 精确到是 features23下哪只股哪一天
    :return: bid、ask、vol_bid、vol_ask DataFrame
    """
    lob_data = np.fromfile(file_path, dtype=np.float32).reshape(-1, 5)
    return pd.DataFrame(lob_data, columns=["bidprice", "askprice", "info", "bidvolume", "askvolume"]).drop("info",
                                                                                                           axis=1)


def pull_group2_data(file_path):
    """提取 features23下特定日期的 group2 数据

    :param file_path: 路径，精确到 features23下哪只股哪一天
    :return: features DataFrame
    """
    feature_data = np.fromfile(file_path, dtype=np.float32).reshape(-1, 178)
    return pd.DataFrame(feature_data, columns=["feature{}".format(i) for i in range(1, 179)]).drop(["feature156",
                                                                                                    "feature160",
                                                                                                    "feature177",
                                                                                                    "feature10",
                                                                                                    "feature91",
                                                                                                    "feature115",
                                                                                                    "feature116"],
                                                                                                   axis=1)


def uint64_to_time_format(origin_time):
    """ 修改时间的显示形式, 原始数据为 【小时、分钟、秒、毫秒】

    :param origin_time: 原始时间
    :return: 修改后时间
    """
    # 时间显式模板
    template = "{}:{}:{}.{}"

    # 字符串化原始时间
    uint64_str = str(origin_time)

    # b补足为 9 位
    uint64_str = uint64_str.zfill(9)

    # 提取小时、分钟、秒和毫秒
    hour = uint64_str[:2]
    minute = uint64_str[2:4]
    second = uint64_str[4:6]
    millisecond = uint64_str[6:]

    return template.format(hour, minute, second, millisecond)


def pull_data(date, stock_name):
    """ 提取 features23下某一天的数据

    :param date: 日期 str, 形如 20201201
    :param stock_name: 股票名
    :return: DataFrame
    """
    file_path_0 = os.path.join("/data/research/zlalgor/ProcessedData2/features23/group0", date, stock_name)
    file_path_1 = os.path.join("/data/research/zlalgor/ProcessedData2/features23/group1", date, stock_name)
    file_path_2 = os.path.join("/data/research/zlalgor/ProcessedData2/features23/group2", date, stock_name)

    daily_data = pd.concat([pull_group0_data(file_path_0),
                            pull_group1_data(file_path_1),
                            pull_group2_data(file_path_2)],
                           axis=1)

    daily_data["nTime"] = daily_data["nTime"].apply(uint64_to_time_format)

    return daily_data
