import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm


# select FAANG: FB, AAPL, AMZN, NVDA(replacing for NFLX), GOOGL and MSFT
faang = ['FB', 'AAPL', 'AMZN', 'NVDA', 'GOOGL', 'MSFT']
# casino stocks: WYNN, LVS, MGM
casino = ['WYNN', 'LVS', 'MGM']
tickers = yf.Tickers(faang + casino)
wynn = tickers.tickers['WYNN'].history(period="5y")
lvs = tickers.tickers['LVS'].history(period="5y")
mgm = tickers.tickers['MGM'].history(period="5y")
aapl = tickers.tickers['AAPL'].history(period="5y")
googl = tickers.tickers['GOOGL'].history(period="5y")
print(aapl.describe()) # stock split of 4 means 4 for 1 split (1 stock becomes 4 stocks, price goes down by 4x)
# check for missing values
for ticker in [wynn, lvs, mgm, aapl, googl]:
    print(ticker.isna().sum()) # no missing values for all

# split into train validation and test sets 
# there is no need for a validation set in this case, but we will keep it since we may want to use a different model later
def train_test_val_split(data, train_size=0.7, val_size=0.2):
    train_end = int(len(data) * train_size)
    val_end = train_end + int(len(data) * val_size)
    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]
    return train, val, test

wynn_train, wynn_val, wynn_test = train_test_val_split(wynn)
lvs_train, lvs_val, lvs_test = train_test_val_split(lvs)
mgm_train, mgm_val, mgm_test = train_test_val_split(mgm)
aapl_train, aapl_val, aapl_test = train_test_val_split(aapl)
googl_train, googl_val, googl_test = train_test_val_split(googl)
# pairs to look at: # AAPL vs GOOGL, WYNN vs LVS, WYNN vs MGM, LVS vs MGM
# conduct a engle-granger test for each pair to identify cointegration
def engle_granger_test(series1, series2):
    # run OLS regression
    X = sm.add_constant(series1)
    y = series2
    model = sm.OLS(y, X)
    res = model.fit()
    print(res.summary())
    # calculate residuals
    residuals = res.resid
    # perform ADF test on residuals
    adf_result = sm.tsa.adfuller(residuals)
    print(f'ADF Statistic: {adf_result[0]}')
    print(f'p-value: {adf_result[1]}') # reject null hypothesis if p-value < 0.05
    if adf_result[1] < 0.05:
        print("The series are cointegrated.")
    else:
        print("The series are not cointegrated.")

# test pairs
engle_granger_test(aapl_train['Close'], googl_train['Close'])
engle_granger_test(wynn_train['Close'], lvs_train['Close'])
engle_granger_test(wynn_train['Close'], mgm_train['Close'])
engle_granger_test(lvs_train['Close'], mgm_train['Close'])
# create a spread for the cointegrated pairs
def create_spread(series1, series2):
    X = sm.add_constant(series1)
    model = sm.OLS(series2, X)
    res = model.fit()
    spread = res.resid
    return spread
def calculate_z_score(spread):
    return (spread - spread.mean()) / spread.std()
# create spreads and calculate z-scores
spread_lvs_mgm = create_spread(lvs_train['Close'], mgm_train['Close'])
spread_wynn_mgm = create_spread(wynn_train['Close'], mgm_train['Close'])
z_score_lvs_mgm = calculate_z_score(spread_lvs_mgm)
z_score_wynn_mgm = calculate_z_score(spread_wynn_mgm)
# plot spreads and z-scores
plt.figure(figsize=(14, 9))
plt.subplot(2, 1, 1)
plt.plot(spread_lvs_mgm, label='LVS vs MGM Spread')
plt.plot(spread_wynn_mgm, label='Wynn vs MGM Spread')
plt.title('Spreads of Cointegrated Pairs')
plt.xlabel('Date')
plt.ylabel('Spread')
plt.subplot(2, 1, 2)
plt.plot(z_score_lvs_mgm, label='LVS vs MGM Z-Score')
plt.plot(z_score_wynn_mgm, label='Wynn vs MGM Z-Score')
plt.axhline(1, color='red', linestyle='--', label='Upper Threshold (1)')
plt.axhline(-1, color='red', linestyle='--', label='Lower Threshold (-1)')
plt.axhline(0, color='black', linestyle='--', label='Mean (0)')
plt.title('Z-Scores of Cointegrated Pairs')
plt.xlabel('Date')
plt.ylabel('Z-Score')
plt.legend()
plt.show()

