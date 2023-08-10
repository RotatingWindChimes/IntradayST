import numpy as np
import matplotlib.pyplot as plt


def daily_test(daily_value_list, tick_list, origin_shares_values, save_path):
    """ 每天的交易测试

    :param daily_value_list: 当天的总资产价值变化
    :param tick_list: 当天的交易时刻
    :param origin_shares_values: 当天始终持仓总价值
    :param save_path: 图形保存路径
    :return: None
    """
    if len(daily_value_list) < len(tick_list):
        daily_value_list.extend([daily_value_list[-1]] * (len(tick_list) - len(daily_value_list)))

    total_return_list = np.array([(daily_value - origin_shares_values[0]) / origin_shares_values[0] for
                                  daily_value in daily_value_list], dtype=np.float64)

    figures, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    axes[0].plot(tick_list, daily_value_list, c="r", label="RL Policy")
    axes[0].plot(tick_list, origin_shares_values, c="g", label="Holding Policy")
    axes[0].legend()

    axes[1].plot(tick_list, total_return_list, c="k", label="Total Return")
    axes[1].legend()

    plt.savefig(save_path)

    sharpe_ratio = total_return_list.mean() / total_return_list.std()

    return sharpe_ratio
