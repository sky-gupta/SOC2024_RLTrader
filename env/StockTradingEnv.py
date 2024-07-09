import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 10000


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""

    def __init__(self, df, render_mode='human'):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)
        self.render_mode = render_mode 
        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6, 6), dtype=np.float16)

    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        frame = np.array([
            self.df.loc[self.current_step: self.current_step +
                        5, 'Open'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        5, 'High'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        5, 'Low'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        5, 'Close'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        5, 'Volume'].values / MAX_NUM_SHARES
        ])

        # Append additional data and scale each value to between 0-1
        # print(self.current_step)
        # print(frame.shape)
        obs1 = np.append(frame, [[
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis / MAX_SHARE_PRICE,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        ]], axis=0)
        obs =obs1
        return obs

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        current_price = random.uniform(
            self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "Close"])

        action_type = action[0]
        amount = action[1]

        if action_type < 1:
            # Buy amount % of balance in shares
            total_possible = int(self.balance / current_price)
            shares_bought = int(total_possible * amount)
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            self.cost_basis = (
                prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought

        elif action_type < 2:
            # Sell amount % of shares held
            shares_sold = int(self.shares_held * amount)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        if self.current_step > len(self.df.loc[:, 'Open'].values) - 6:
            self.current_step = 0

        delay_modifier = (self.current_step / MAX_STEPS)

        reward = self.balance * delay_modifier
        done = self.net_worth <= 0

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(
            0, len(self.df.loc[:, 'Open'].values) - 6)

        return self._next_observation()

    # def render(self, mode='human', close=False):
        # Render the environment to the screen
        # if mode == 'human':
        #     profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        #     print(f'Step: {self.current_step}')
        #     print(f'Balance: {self.balance}')
        #     print(
        #         f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        #     print(
        #         f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        #     print(
        #         f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        #     print(f'Profit: {profit}')
        # elif mode == 'rgb_array':
        #     # Implement image rendering logic here, if applicable
        #     pass
        # else:
        #     super().render(mode=mode)  # Just to ensure compatibility with other render modes
        def render(self, savefig=False, filename='myfig'):
            if self._first_render:
                self._f, (self._ax, self._ay, self._az, self._at) = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=False, squeeze=True,
                            gridspec_kw={'height_ratios': [4, 1, 1, 0]},)



                self._ax = [self._ax]
                self._ay = [self._ay]
                self._az = [self._az]
                self._at = [self._at]
                self._f.set_size_inches(12, 6)
                self._first_render = False
                self._f.canvas.mpl_connect('close_event', self._handle_close)


            #  price
            ask, bid, mid, rsi, cci = self._tick_buy, self._tick_sell,self.tick_mid, self.tick_rsi_14, self.tick_cci_14

            self._ax[-1].plot([self._iteration, self._iteration + 1], [mid, mid], color='white')
            self._ay[-1].plot([self._iteration, self._iteration + 1], [cci, cci], color='green')
            self._az[-1].plot([self._iteration, self._iteration + 1], [rsi, rsi], color='blue')
            self._ay[0].set_ylabel('CCI')
            self._az[0].set_ylabel('RSI')

            ymin, ymax = self._ax[-1].get_ylim()
            yrange = ymax - ymin
            if self.Sell_render:
                self._ax[-1].scatter(self._iteration + 0.5, bid + 0.03 *
                                    yrange, color='orangered', marker='v')
            elif self.Buy_render:
                self._ax[-1].scatter(self._iteration + 0.5, ask - 0.03 *
                                    yrange, color='lawngreen', marker='^')
            if self.TP_render:
                self._ax[-1].scatter(self._iteration + 0.5, bid + 0.03 *
                                    yrange, color='gold', marker='.')
            elif self.SL_render:
                self._ax[-1].scatter(self._iteration + 0.5, ask - 0.03 *
                                    yrange, color='maroon', marker='.')


            self.TP_render=self.SL_render=False
            self.Buy_render=self.Sell_render=False

            plt.suptitle('Total Reward: ' + "%.2f" % self._total_reward +
                        '  Total PnL: ' + "%.2f" % self._total_pnl +
                        '  Unrealized Return: ' + "%.2f" % (self.unrl_pnl*100)  + "% "+
                        '  Pstn: ' + ['flat', 'long', 'short'][list(self._position).index(1)] +
                        '  Action: ' + ['flat', 'long', 'short'][list(self._action).index(1)] +
                        '  Tick:' + "%.2f" % self._iteration)
            self._f.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

            plt.xticks(range(self._iteration)[::5])
            plt.xlim([max(0, self._iteration - 80.5), self._iteration + 0.5])

            plt.subplots_adjust(top=0.85)
            plt.pause(0.00001) # 0.01
            if savefig:
                plt.savefig(filename)