import numpy as np
import pandas as pd
import gym

from gym import spaces

from stable_baselines3.common import logger

#matplotlib.use('Agg')

class CryptoTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                df,
                state_space,
                action_space,
                acc_balance,
                tech_indicator_list,
                reward_scaling,
                max_trades,
                ):
        self.df = df
        self.action_space = spaces.Box(low = -1, high = 1, shape = (action_space,)) 
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape = (state_space,))
        self.acc_balance = acc_balance
        self.tech_indicator_list = tech_indicator_list
        self.reward_scaling = reward_scaling
        self.max_trades = max_trades

        self.print_verbosity = 1

        # initalize state
        self.day = 0
        self.terminal = False
        self.data = self.df.iloc[self.day]
        self.state = self._initiate_state()
        
        # initialize reward
        self.reward = 0
        self.trades = 0
        self.episode = 0

        # memorize all the total balance change
        self.acc_balance_memory = [self.acc_balance]
        self.asset_value_memory = [0]
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]

    def step(self, action):
        self.terminal = self.day >= len(self.df.index.unique())-1
        if self.terminal: 
            initial_total_value = self.acc_balance_memory[0] + self.asset_value_memory[0]
            end_total_value = self.state[0] + (self.state[1]*self.state[2])
            total_reward = end_total_value - initial_total_value

            # Add outputs to logger interface
            logger.record("environment/portfolio_value", end_total_value)
            logger.record("environment/total_reward", total_reward)
            logger.record("environment/total_reward_pct", (total_reward/(end_total_value-total_reward))*100)
            logger.record("environment/total_trades", self.trades)

            return self.state, self.reward, self.terminal, {}

        else:
            action = action.astype(float)
            begin_total_value = self.state[0] + (self.state[1]*self.state[2])

            self._take_action(action)

            self.day += 1
            self.data = self.df.iloc[self.day]
            self.state = self._update_state()
                           
            end_total_value = self.state[0] + (self.state[1]*self.state[2])

            reward = end_total_value - begin_total_value

            self.acc_balance_memory.append(self.state[0])
            self.asset_value_memory.append(self.state[1]*self.state[2])
            self.rewards_memory.append(reward)
            self.date_memory.append(self._get_date())

            self.reward += reward * self.reward_scaling     # Test

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.reward = 0
        self.trades = 0
        self.episode = 0

        self.day = 0
        self.terminal = False
        self.data = self.df.iloc[self.day]
        self.state = self._initiate_state()

        self.acc_balance_memory = [self.acc_balance]
        self.asset_value_memory = [0]
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        
        self.episode += 1

        return self.state
    
    def render(self):
        return self.state

    def _take_action(self, action):
        if action > 0:
            self._buy(action)
        if action < 0:
            action = abs(action)
            self._sell(action)
        if action == 0:
            self.actions_memory.append(0.)


    def _sell(self, action):
        if self.state[1] > 0: # Sell only if asset has value
            if self.state[2] > 0: # Sell only if we have assets to sell
                maximum_sell = self.state[2]
                sell_number = maximum_sell * action
                sell_asset_value = self.state[1] * sell_number
                #update balance
                self.state[0] += sell_asset_value
                self.state[2] -= sell_number
                self.actions_memory.append(-sell_number[0])
                self.trades += 1
            else:
                self.actions_memory.append(0.)
        else:
            self.actions_memory.append(0.)
    
    def _buy(self, action):
        if self.state[1] > 0: # Buy only if asset has value
            #if Only invest if you're below your investment cap
            maximum_buy = self.state[0] / self.state[1]
            buy_number = maximum_buy * action
            buy_asset_value = buy_number * self.state[1]
            #update balance
            self.state[0] -= buy_asset_value
            self.state[2] += buy_number
            self.actions_memory.append(buy_number[0])
            self.trades += 1
        else:
            self.actions_memory.append(0.)

    def _initiate_state(self):
        state = [
            self.acc_balance,
            self.data.close,
            0,
            self.data.turbulence,
        ] + [self.data[ti] for ti in self.tech_indicator_list]
        return np.array(state)

    def _update_state(self):
        state = [
                self.state[0],
                self.data.close,
                self.state[2],
                self.data.turbulence,
        ] + [self.data[ti] for ti in self.tech_indicator_list]
        return np.array(state)

    def _get_date(self):
        date = self.data.date
        return date

    def save_account_value(self):
        date_memory = self.date_memory
        account_value = [x + y for x, y in zip(self.acc_balance_memory, self.asset_value_memory)]
        df_account_value = pd.DataFrame({'date':date_memory,'account_value':account_value})
        return df_account_value

    def save_actions(self):
        date_memory = self.date_memory[:-1]
        assert sum(self.actions_memory) == self.state[2]      # Residual assets not accounted for in sum
        action_memory = self.actions_memory
        df_actions = pd.DataFrame({'date':date_memory,'actions':action_memory})
        return df_actions