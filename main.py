import warnings
import os

warnings.filterwarnings('ignore')
#pd.set_option('display.max_columns', None)

# Credit to FinRL
from finrl import config
from finrl.utils import load_data, split_data, backtest_stats, tear_sheet, ti_graphs, plot_actions, plot_portfolio
from finrl.preprocessing import FeatureEngineer
from finrl.cryptotradingenv import CryptoTradingEnv

from stable_baselines3 import A2C, PPO
from stable_baselines3 import DDPG, TD3, SAC    # Redundant for single asset trading

from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_checker import check_env

def train():

    models = [A2C, PPO]

    df = load_data(config.TIC, config.START_DATE, config.END_DATE)
    print(df.head())

    fe = FeatureEngineer(
                    use_technical_indicator = True,
                    tech_indicator_list = config.TECHNICAL_INDICATORS_LIST,
                    use_turbulence = True,              # Calculates turbulence
                    use_psar = True,                    # Calculates parabolic stop and reverse
                    use_obv = True,                     # Calculates on balance volume
                    user_defined_feature = False)

    processed = fe.preprocess_data(df)
    print(processed.head())
    processed.to_csv('./results/df_processed.csv')
    processed.tail(5).to_html('./results/df_processed.html')

    train = split_data(processed, config.START_DATE, config.START_TRADE_DATE)
    trade = split_data(processed, config.START_TRADE_DATE, config.END_DATE)

    ti_graphs(trade)    # Generates graphs for each technical indicator used

    # Account Balance / Close / No. of Assets / Turbluence / Number of Technical Indicators
    state_space = 4 + len(config.TI_LIST)

    env_kwargs = {
        "state_space": state_space,
        "action_space": 1,
        "acc_balance": 100000,
        "tech_indicator_list": config.TI_LIST, 
        "reward_scaling": 1e-3,
        "max_trades": 12,
    }

    for i, model_name in enumerate(config.MODEL_LIST):
    
        env = CryptoTradingEnv(df=train, **env_kwargs)
        check_env(env)

        # Continues training from wherever it left off.
        if os.path.exists(f'./models/{model_name}.zip'):
            model = models[i].load(f'./models/{model_name}.zip')
            # model.save(f'./models/{model_name}_backup') # Saves a backup model in case of error or poor performance
            # model.set_env(env)
            # model.learn(total_timesteps=config.TIMESTEPS)
            # model.save(f'./models/{model_name}')
        else:
            model = models[i]("MlpPolicy", env, verbose=1)
            model.learn(total_timesteps=config.TIMESTEPS)
            model.save(f'./models/{model_name}')

        trade_env = CryptoTradingEnv(df=trade, **env_kwargs)
        check_env(trade_env)

        obs = trade_env.reset()
        for i in range(len(trade.index.unique())):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = trade_env.step(action)
            if dones:
                print(f"[ Completed {model_name.upper()} Predictions ]")
                break

        df_account_value = trade_env.save_account_value()
        df_actions = trade_env.save_actions()

        df_account_value.to_csv(f'./results/{model_name}/df_account_value.csv')
        df_actions.to_csv(f'./results/{model_name}/df_actions.csv')

        # Backtesting and producing performance metrics
        performance_metrics = backtest_stats(df_account_value)
        performance_metrics.to_csv(f'./results/{model_name}/perf_metrics.csv')
        performance_metrics.to_html(f'./results/{model_name}/perf_metrics.html')

        tear_sheet(df_account_value, model_name)

        plot_actions(trade, df_actions, model_name)

        plot_portfolio(df_actions, df_account_value, model_name)

if __name__ == '__main__':
    train()