import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf
from finrl import config

class FeatureEngineer:

    def __init__(
        self,
        use_technical_indicator=True,
        tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
        use_turbulence=False,
        use_psar=False,
        use_obv=False,
        user_defined_feature=False,
    ):
        self.use_technical_indicator = use_technical_indicator
        self.tech_indicator_list = tech_indicator_list
        self.use_turbulence = use_turbulence
        self.use_psar = use_psar
        self.use_obv = use_obv
        self.user_defined_feature = user_defined_feature

    def preprocess_data(self, df):
        
        # add technical indicators using stockstats
        if self.use_technical_indicator == True:
            df = self.add_technical_indicator(df)
            print("[ Successfully Added Technical Indicators ]\n")

        # add turbulence index
        if self.use_turbulence == True:
            df = self.add_turbulence(df)
            print("[ Successfully Added Turbulence Index ]\n")

        # add parabolic sar
        if self.use_psar == True:
            df = self.add_parabolic_sar(df)
            print("[ Successfully Added Parabolic SAR ]\n")

        # add on balance volume
        if self.use_obv == True:
            df = self.add_on_balance_volume(df)
            print("[ Successfully Added On Balance Volume ]\n")

        # fill the missing values at the beginning and the end
        df = df.fillna(method="bfill").fillna(method="ffill")
        return df

    def add_technical_indicator(self, data):
        df = data.copy()
        df = df.sort_values(by=['tic','date'])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in self.tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator['tic'] = unique_ticker[i]
                    temp_indicator['date'] = df[df.tic == unique_ticker[i]]['date'].to_list()
                    indicator_df = indicator_df.append(
                        temp_indicator, ignore_index=True
                    )
                except Exception as e:
                    print(e)
            df = df.merge(indicator_df[['tic','date',indicator]],on=['tic','date'],how='left')
        df = df.sort_values(by=['date','tic'])
        return df

    def add_turbulence(self, data):
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df)
        df = df.merge(turbulence_index, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    def add_parabolic_sar(self, data):
        df = data.copy()
        psar = df['low'][0]
        ep = df['high'][0]
        a = 0.02
        amax = 0.2
        uptrend = True
        values = []
        values.append(psar)

        for i, day in df.iloc[1:].iterrows():

            psar = psar + a * (ep - psar)
            values.append(psar)

            change_trend = False
            if psar < df['high'][i-1]:
                if not uptrend:
                    change_trend = True
                uptrend = True
            elif psar > df['low'][i-1]:
                if uptrend:
                    change_trend = True
                uptrend = False
            
            if uptrend:
                if (day['high'] > ep):
                    ep = day['high']
                    if change_trend:
                        a = 0.02
                    else:
                        if round(a, 2) < amax:
                            a = round(a + 0.02, 2)
            else:
                if (day['low'] < ep):
                    ep = day['low']
                    if change_trend:
                        a = 0.02
                    else:
                        if round(a, 2) < amax:
                            a = round(a + 0.02, 2)
        
        df['psar'] = pd.Series(values, index=df.index)
        return df.copy()

    def add_on_balance_volume(self, data):
        df = data.copy()
        values = [0]

        for i in range(1, len(df.index.unique())):
            if df['close'][i] > df['close'][i-1]:
                obv = values[i-1] + df['volume'][i]
            elif df['close'][i] < df['close'][i-1]:
                obv = values[i-1] - df['volume'][i]
            else:
                obv = values[i-1]
            values.append(obv)

        df['obv'] = pd.Series(values, index=df.index)
        return df.copy()

    def calculate_turbulence(self, data):
        # can add other market assets
        df = data.copy()
        df_price_pivot = df.pivot(index="date", columns="tic", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.date.unique()
        # start after a year
        start = 252
        turbulence_index = [0] * start
        # turbulence_index = [0]
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            # use one year rolling window to calcualte covariance
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - 252])
            ]
            # Drop tickers which has number missing values more than the "oldest" ticker
            filtered_hist_price = hist_price.iloc[hist_price.isna().sum().min():].dropna(axis=1)

            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(filtered_hist_price, axis=0)
            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    # avoid large outlier because of the calculation just begins
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)

        turbulence_index = pd.DataFrame(
            {"date": df_price_pivot.index, "turbulence": turbulence_index}
        )
        return turbulence_index