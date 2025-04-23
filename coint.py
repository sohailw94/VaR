import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns
portfolio=['AAPL',"TSLA","MSFT","GOOG","NVDA"]

data=yf.download(portfolio,start='2020-03-31',end='2025-04-21')['Close']
data=data.dropna()

def engle_granger_test(y,x):
    x=sm.add_constant(x)
    model=sm.OLS(y,x).fit()
    resid=model.resid
    adf_res=adfuller(resid)
    return adf_res[1],resid

from itertools import combinations

results=[]
for s1 , s2 in combinations(portfolio,2):
    p_value ,resid=engle_granger_test(data[s1],data[s2])
    results.append((s1,s2,p_value,resid))

results_sorted=sorted(results,key=lambda x:x[2])

best_pair = results_sorted[0]

spread=best_pair[3]
mean_spread=spread.mean()
std_spread=spread.std()
zscore=(spread-mean_spread)/std_spread

print(f"Best cointegrated pair: {best_pair[0]} & {best_pair[1]} | p-value: {best_pair[2]:.4f}")
plt.figure(figsize=(14,6))
plt.plot(zscore, label='Z-score of Residual Spread', color='blue')
plt.axhline(0, color='black', linestyle='-')
plt.axhline(1, color='green', linestyle='--', label='Short Signal')
plt.axhline(-1, color='red', linestyle='--', label='Long Signal')
plt.axhline(2, color='green', linestyle=':', label='Extreme Short')
plt.axhline(-2, color='red', linestyle=':', label='Extreme Long')
plt.title(f'Z-Score Signals for Spread between {best_pair[0]} and {best_pair[1]}')
plt.legend()
plt.grid(True)
plt.show(block=False)

stk_1,stk_2=best_pair[0],best_pair[1]
y=data[stk_1]
x=data[stk_2]

x_const=sm.add_constant(x)
model=sm.OLS(y,x_const).fit()
beta=model.params[1]
spread=model.resid

zscore=(spread-spread.mean())/spread.std()

signals=pd.DataFrame(index=spread.index)
signals['spread']=spread
signals['zscore']=zscore
signals['long']=zscore < -1
signals['short']=zscore > 1
signals['exit']=(zscore*zscore.shift(1)<0)

signals['position']=0
signals.loc[signals['long'],'position']=1
signals.loc[signals['short'],'position']=-1
signals.loc[signals['exit'],'position']=0
signals['position']=signals['position'].ffill().fillna(0)

returns=pd.DataFrame(index=data.index)
returns['y_ret'] = y.pct_change()
returns['x_ret'] = x.pct_change()
returns['spread_ret'] = returns['y_ret'] - beta * returns['x_ret']

signals['daily_pnl'] = signals['position'].shift(1) * returns['spread_ret']
signals['cum_pnl'] = (1 + signals['daily_pnl'].fillna(0)).cumprod()

plt.figure(figsize=(14,6))
plt.plot(signals['cum_pnl'], label='Equity Curve', color='blue')
plt.title(f'Equity Curve for {stk_1} - {beta:.2f}  {stk_2}')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.legend()
plt.show(block=False)

last_signal=signals.iloc[-1]

if last_signal['long']:
    print(f" Go LONG the spread between {stk_1} and {stk_2}")
elif last_signal['short']:
    print(f"Go SHORT the spread between {stk_1} and {stk_2}")
elif last_signal['exit']:
    print(f"EXIT position in spread between {stk_1} and {stk_2}")
else:
    print(f"No trade signal. Stay NEUTRAL on {stk_1}-{stk_2} spread")

