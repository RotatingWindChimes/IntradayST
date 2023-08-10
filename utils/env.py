import numpy as np


class IntradayEnvDiscrete:
    def __init__(self, daily_data, lob_data, stock_base, buy_cost_pct, sell_cost_pct, initial_amount, bound, delta):
        """ 日间交易强化学习智能体离散环境

        :param daily_data: 一日的特征数据, DataFrame, 没有 nTime 列
        :param lob_data: 一日的lob数据, DataFrame, 没有 nTime 列
        :param stock_base: 股票交易基数=100
        :param buy_cost_pct: 做多手续费
        :param sell_cost_pct: 做空手续费
        :param initial_amount: 初始资金=1e6
        :param bound: 股票基数界限, 不超过100基
        :param delta: 远视奖励系数
        """
        self.daily_data = daily_data
        self.lob_data = lob_data
        self.stock_base = stock_base
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.initial_amount = initial_amount
        self.bound = bound
        self.delta = delta

        self.tick_index = 0                                          # 日内交易起始点索引
        self.state_dim = 2 + len(self.daily_data.columns)            # 现金/初始现金、底仓/10000、因子特征
        self.total_assets = 0                                        # 当前模拟序列中随时间变化的总资产
        self.asset_states = np.array([])                             # 当前模拟序列中随时间变化的资产状态

    def get_tick_data(self, tick_id):
        """ 获得第 tick_id 时刻的特征数据

        :param tick_id: 交易时刻
        :return: NumPy array
        """
        return self.daily_data.iloc[tick_id].to_numpy()

    def reset(self):
        # 初始化日内交易起始点
        self.tick_index = 0

        # 初始化当前状态
        initial_state = np.concatenate((np.array([1.0, 0.0]), self.get_tick_data(self.tick_index)))

        # 初始化本轮序列中的总资产、状态
        self.total_assets = self.initial_amount
        self.asset_states = initial_state

        return np.array(self.asset_states, dtype=np.float64)

    def get_transaction(self, action):
        # action范围是[0, 201], 实际股数是 |action-100| * 100, 负号表示买入, 正号表示卖出
        direction = 0 if action > 100 else 1
        volume = abs(direction - 100) * self.stock_base

        # 截断为盘口量
        if direction == 0:         # 0表示做空, 卖出, 不能超过自己的持仓 & bidvolume
            volume = min(volume, self.asset_states[1] * self.stock_base * self.bound,
                         self.lob_data.iloc[self.tick_index]["bidvolume"])
        else:                      # 1表示做多, 买入, 不能超过 askvolume
            volume = min(volume, self.lob_data.iloc[self.tick_index]["askvolume"])

        return direction, volume

    def cash(self):
        """ 获得交易时刻的现金

        :return: 现金
        """
        return self.initial_amount * self.asset_states[0]

    def holdings_forever_value(self):
        initial_shares = self.initial_amount / self.lob_data.iloc[0]["bidprice"]
        origin_shares_values = [initial_shares * self.lob_data.iloc[j]["bidprice"] for j in range(len(self.lob_data))]

        return origin_shares_values

    def get_future_price(self):
        """ 计算交易日内, 交易时刻往后的平均报价

        :return: float
        """
        return (self.lob_data[self.tick_index:]["bidprice"]).mean()

    def step(self, action):

        if self.tick_index == len(self.daily_data) - 1:  # 最后一个交易点不进行交易, 没有奖励
            return np.array(self.asset_states, dtype=np.float64), 0, True, self.total_assets
        else:
            # 现金
            begin_cash = self.cash()

            # 获得实际交易量
            direction, volume = self.get_transaction(action=action)

            # 判断买卖方向
            if direction == 0:  # 做空
                sell_interest = volume * self.lob_data.iloc[self.tick_index]["bidprice"]
                sell_cost = sell_interest * self.sell_cost_pct

                if begin_cash + sell_interest < sell_cost:  # 卖出的成本费太高
                    return np.array(self.asset_states, dtype=np.float64), -100, True, self.total_assets

                # 卖出后现金
                coh = begin_cash + sell_interest - sell_cost
                # 卖出后仓位
                holdings = self.asset_states[1] * self.stock_base * self.bound - volume

            else:  # 做多
                buy_interest = volume * self.lob_data.iloc[self.tick_index]["askprice"]
                buy_cost = buy_interest * self.buy_cost_pct

                if begin_cash < buy_cost + buy_interest:   # 买不起
                    return np.array(self.asset_states, dtype=np.float64), -100, True, self.total_assets

                # 买入后现金
                coh = begin_cash - buy_cost - buy_interest
                # 买入后仓位
                holdings = self.asset_states[1] * self.stock_base * self.bound + volume

            # 前进到下一个交易点
            self.tick_index += 1

            # 长期持资未来平均价值与此刻价值
            future_price = self.get_future_price()
            current_price = self.lob_data.iloc[self.tick_index]["bidprice"]

            # 当前资产总价值
            new_total_assets = holdings * current_price + coh

            # 计算奖励: 相比前一个时刻当前资产损益 + 长时间持有资产的潜在收益
            reward = new_total_assets - self.total_assets + self.delta * (future_price - current_price)

            # 更新总资产, 状态
            self.total_assets = new_total_assets
            self.asset_states = np.concatenate((np.array([coh/self.initial_amount]),
                                                np.array([holdings/(self.stock_base * self.bound)]),
                                                self.get_tick_data(self.tick_index)))

            return np.array(self.asset_states, dtype=np.float64), reward, False, self.total_assets
