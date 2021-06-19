import pandas as pd
import numpy as np
import pyfolio as pf  # zipline.assets not found
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import mplfinance as mpf

from finrl import config
from pyfolio import timeseries
from bokeh.plotting import figure, show, output_file

def load_data(tic, start_date, end_date):
    df = pd.read_csv(f'./datasets/{tic}.csv')
    #df = df.iloc[242:]
    df.dropna().reset_index()
    df.drop(['SNo', 'Name', 'Symbol', 'Marketcap'], axis=1, inplace=True)
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    df['tic'] = 'BTC'
    df['date'] = df['date'].apply(lambda x: x[:10])
    df['date'] = pd.to_datetime(df['date'])
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    df = df.loc[mask]
    return df

def split_data(df, start, end):
    data = df[(df.date >= start) & (df.date <= end)]
    data = data.sort_values(["date", "tic"], ignore_index=True)
    data.index = data.date.factorize()[0]
    return data

def get_daily_return(df_account_value):
    df = df_account_value.copy()
    df["daily_return"] = df["account_value"].pct_change(1)
    df["daily_return"][0] = 0
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True, drop=True)
    df.index = df.index.tz_localize("UTC")
    return pd.Series(df["daily_return"].values, index=df.index)

def backtest_stats(df_account_value):
    daily_return = get_daily_return(df_account_value)
    perf_stats = timeseries.perf_stats(
        returns=daily_return,
        positions=None,
        transactions=None,
        turnover_denom="AGB",
    )
    print(perf_stats)
    return pd.DataFrame(perf_stats)

def tear_sheet(df_account_value, model_name):
    returns = get_daily_return(df_account_value)
    pf.create_simple_tear_sheet(returns)
    plt.margins(0,0)
    plt.suptitle(f'{model_name.upper()} Tear Sheet', fontsize=32, y=0.05)
    plt.savefig(f'./results/{model_name}/tearsheet.png', bbox_inches='tight', pad_inches=0)
    plt.clf()

def ti_graphs(df):

    sns.set_theme(style="darkgrid")
    df.index = pd.to_datetime(df.index)

    # mpf.plot(df, type='candle', volume=True, style='yahoo')
    # plt.show()

    y1 = np.array(df['boll_lb'].values)
    y2 = np.array(df['boll_ub'].values)
    plt.figure(figsize=(1500/config.PPI, 500/config.PPI), dpi=config.PPI)
    plt.title("boll bands")
    plt.plot(df['date'], df['boll_ub'], label="boll_ub", color=config.COLOUR_LIST[1])
    plt.plot(df['date'], df['boll_lb'], label="boll_lb", color=config.COLOUR_LIST[2])
    plt.plot(df['date'], df['close'], label="close")
    plt.fill_between(df['date'], y1, y2, color=config.COLOUR_LIST[1], alpha=0.2)
    plt.legend(loc='upper left')
    plt.savefig('./results/technical analysis/boll.png')
    plt.clf()

    plt.figure(figsize=(1500/config.PPI, 500/config.PPI), dpi=config.PPI)
    plt.title("p sar")
    plt.plot(df['date'], df['close'], label="close")
    plt.plot(df['date'], df['psar'], label="psar", color=config.COLOUR_LIST[-1])
    plt.legend(loc='upper left')
    plt.savefig('./results/technical analysis/psar.png')
    plt.clf()

    for i, ti in enumerate(config.TI_LIST):
        plt.figure(figsize=(1500/config.PPI, 500/config.PPI), dpi=config.PPI)
        plt.title(f"{ti}")

        if ti in config.PASS_LIST:
            plt.clf()
            continue
        elif ti in config.SINGLE_TI:
            plt.plot(df['date'], df[ti], label=ti, color=config.COLOUR_LIST[i])
        else:
            plt.plot(df['date'], df['close'], label="close")
            plt.plot(df['date'], df[ti], label=ti, color=config.COLOUR_LIST[i])
            plt.legend(loc='upper left')

        plt.savefig(f'./results/technical analysis/{ti}.png')
        plt.clf()

def plot_actions(df, df_actions, model_name):
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(1500/config.PPI, 500/config.PPI), dpi=config.PPI)
    plt.plot(df['date'], df['close'])

    df_actions.loc[365] = 0

    df.index = range(len(df.index))
    df_actions.index = range(len(df_actions.index))

    for i, day in df.iterrows():
        
        action = df_actions.iloc[i]['actions']

        if action > 0:
            plt.scatter(day['date'], day['close'], color='green')
        elif action < 0:
            plt.scatter(day['date'], day['close'], color='red')

    plt.title(f'{model_name.upper()} Actions')
    plt.savefig(f'./results/{model_name}/actions.png')
    plt.clf()

def plot_portfolio(df_actions, df_account_value, model_name):
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(1500/config.PPI, 500/config.PPI), dpi=config.PPI)

    df_actions.loc[365] = 0

    acc_val = pd.Series(df_account_value['account_value'])
    actions = pd.Series(df_actions['actions'])
    holding = actions.cumsum()

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Day')
    ax1.set_ylabel('Portfolio')
    ax1.bar(holding.index, holding, color='palevioletred', linewidth=0)

    ax2 = ax1.twinx()

    ax2.set_ylabel('Total Asset Value')
    ax2.plot(acc_val, color='royalblue')

    fig.tight_layout()

    plt.title(f'{model_name.upper()} Portfolio & Total Asset Value')
    plt.savefig(f'./results/{model_name}/portfolio.png', bbox_inches='tight')
    plt.clf()