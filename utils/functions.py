import numpy as np
import torch.nn as nn


def xavier_init_weights(m):
    """ 网络初始化

    :param m: 网络
    :return: None
    """
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)


def moving_average(arr, window_size=9):
    """ 滑动窗口平均

    :param arr: 初始数组
    :param window_size: 窗口大小
    :return: 平均后数组
    """
    arr_cumsum = np.cumsum(np.insert(arr, 0, 0))

    r = np.arange(1, window_size - 1, 2)

    # 前半段
    begin = np.cumsum(arr[:window_size-1][::2]) / r

    # 中间部分, 整个序列所有大小为 window_size 的滑动窗口内均值
    middle = (arr_cumsum[window_size:] - arr_cumsum[:-window_size]) / window_size

    # 后半段
    end = (np.cumsum(arr[-(window_size-1):][::-1][::2]) / r)[::-1]

    return np.concatenate((begin, middle, end))