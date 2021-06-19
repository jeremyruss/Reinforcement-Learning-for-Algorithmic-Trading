TIC = 'btc'

TURBULENCE_FACTOR = 1.2
TIMESTEPS = 10000   # ~ 500 timesteps per second | 3600 seconds in an hour
PPI = 137.68

COLOUR_LIST = [ 
    "lightcoral",
    "lightskyblue",
    "lightskyblue",
    "sandybrown",
    "limegreen",
    "deepskyblue",
    "plum",
    "turquoise",
    "crimson",
    "mediumpurple",
    "hotpink"
]

SINGLE_TI = ["macd", "rsi_30", "cci_30", "dx_30", "adx", "obv"]
PASS_LIST = ["boll_ub", "boll_lb", "close_30_sma"]

MODEL_LIST = ["a2c", "ppo"] # ["ddpg", "td3", "sac"]

START_DATE = "2014-06-01"
END_DATE = "2020-05-30"

START_TRADE_DATE = "2019-06-01"

# Stockstats list
TECHNICAL_INDICATORS_LIST = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
    "adx",
]

# My TI list
TI_LIST = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
    "adx",
    "psar",
    "obv"
]