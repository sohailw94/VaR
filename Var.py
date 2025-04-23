import yfinance as yf
import pandas as pd
import numpy as np

portfolio=['AAPL',"TSLA","MSFT","GOOG","NVDA"]

data=yf.download(portfolio,start='2020-03-31',end='2025-04-01')['Close']


df=data.interpolate(method='linear',limit_direction='both')

log_ret=np.log(df/df.shift(1)).dropna()

# print(log_ret.head(5))



mu=log_ret.mean().values
cov=log_ret.cov().values

L=np.linalg.cholesky(cov)

# print("Covariance matrix:\n", cov)
# print("\nCholesky matrix:\n", L)

T=20
n_sim=10000
dt=1
n_assets=len(log_ret.columns)
strt_prcs=df.iloc[-1].values
sigma=np.sqrt(np.diag(cov))

drift=(mu-0.5*sigma**2)*dt

sim_path=np.zeros((T,n_assets,n_sim))
sim_path[0]=np.tile(strt_prcs.reshape(-1,1),(1,n_sim))

Z=np.random.normal(size=(T,n_assets,n_sim))
corr_Z=np.einsum('ij,tjk->tik',L,Z)

drift=drift[:,np.newaxis]
sigma=sigma[:,np.newaxis]

drift=np.tile(drift,(1,n_sim))
sigma=np.tile(sigma,(1,n_sim))

drift=np.tile(drift,(T,1,1)).transpose(0,1,2)
sigma=np.tile(sigma,(T,1,1)).transpose(0,1,2)

log_ret_sim=drift+sigma*corr_Z

price_paths = np.zeros((T, n_assets, n_sim))
price_paths[0] = np.tile(strt_prcs.reshape(-1, 1), (1, n_sim))

# Cumulative return in log space
cum_returns = np.cumsum(log_ret_sim, axis=0)
price_paths[1:] = np.exp(cum_returns[1:]) * price_paths[0]
import matplotlib.pyplot as plt

# for i, ticker in enumerate(log_ret.columns):
#     plt.plot(price_paths[:, i, 0], label=ticker)
# plt.legend()
# plt.title("1st Simulation Path of Each Asset")
# plt.show()

weights=np.array([0.2,0.2,0.2,0.2,0.2])

init_val=np.dot(strt_prcs,weights)

nrmlised_prc=price_paths/price_paths[0]

port_ret=np.einsum('taj,a->tj',nrmlised_prc,weights)

port_val=init_val*port_ret

fnl_val=port_val[-1,:]

port_pnl=fnl_val-init_val

plt.hist(port_pnl, bins=100, color='skyblue', edgecolor='black')
plt.axvline(np.percentile(port_pnl, 5), color='red', linestyle='--', label='5% VaR')
plt.axvline(np.percentile(port_pnl, 1), color='orange', linestyle='--', label='1% VaR')
plt.title("Simulated Portfolio P&L Distribution")
plt.xlabel("P&L")
plt.ylabel("Frequency")
plt.legend()
plt.show()