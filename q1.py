# FINANCE

# QUESTION 1
'''Taking rate of return for 2 securities,

finding the return, risk, covariance

randomly generating 1,00,000 weights using uniform, random distribution and simulating thousands of possible portfolios (different weight combos)

computing return and risk for each

plotting the efficient frontier

identifying minimum-variance portfolio'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

# 1. Load data
'''df_HDFC = pd.read_csv("/content/HDFC.csv")
df_KOTAK = pd.read_csv("/content/kotak.csv")

df_HDFC['Month'] = pd.to_datetime(df_HDFC['Month'], format='%b-%y', errors='coerce')
df_KOTAK['Month'] = pd.to_datetime(df_KOTAK['Month'], format='%b-%y', errors='coerce')'''

df_HDFC = yf.download('HDFCBANK.NS', start='2020-01-01', end='2025-10-01', interval='1mo')
df_Kotak = yf.download('KOTAKBANK.NS', start='2020-01-01', end='2025-10-01', interval='1mo')

df_HDFC.reset_index(inplace=True)
df_Kotak.reset_index(inplace=True)

df_HDFC.rename(columns={'Date': 'Month'}, inplace=True)
df_Kotak.rename(columns={'Date': 'Month'}, inplace=True)

df_HDFC = df_HDFC.sort_values(by='Month')
df_KOTAK = df_Kotak.sort_values(by='Month')

# 2. Calculate rate of return
df_HDFC['Rate_of_Return'] = (df_HDFC['Close'] - df_HDFC['Open']) / df_HDFC['Open']
df_KOTAK['Rate_of_Return'] = (df_KOTAK['Close'] - df_KOTAK['Open']) / df_KOTAK['Open']

hdfc_returns = df_HDFC['Rate_of_Return'].dropna().values
kotak_returns = df_KOTAK['Rate_of_Return'].dropna().values

n = len(hdfc_returns)

# ---- Manual Covariance Calculation ----
mean_hdfc = np.sum(hdfc_returns) / n
mean_kotak = np.sum(kotak_returns) / n

cov_hdfc = np.sum((hdfc_returns - mean_hdfc) ** 2) / (n - 1)  # Var(HDFC)
cov_kotak = np.sum((kotak_returns - mean_kotak) ** 2) / (n - 1)  # Var(KOTAK)
cov_cross = np.sum((hdfc_returns - mean_hdfc) * (kotak_returns - mean_kotak)) / (n - 1)  # Cov(HDFC,KOTAK)

# Covariance Matrix C
C = np.array([
    [cov_hdfc, cov_cross],
    [cov_cross, cov_kotak]
])

# Expected returns
expected_return_hdfc = mean_hdfc
expected_return_kotak = mean_kotak
expected_risk_hdfc = np.sqrt(cov_hdfc)
expected_risk_kotak = np.sqrt(cov_kotak)

# ---- Minimum Variance Weights ----
U = np.array([1, 1])
C_inv = np.linalg.inv(C)

numerator = U @ C_inv
denominator = U @ C_inv @ U.T
weights_min_var = numerator / denominator

print("Minimum Variance Weights:")
print(f"HDFC Weight:  {weights_min_var[0]:.4f}")
print(f"KOTAK Weight: {weights_min_var[1]:.4f}")

# 3. Efficient Frontier
num_portfolios = 100000
weights_uniform = np.random.uniform(0, 1, num_portfolios)
weights_normal = np.clip(np.random.normal(0.5, 0.2, num_portfolios), 0, 1)

def portfolio_metrics(weights):
    returns = []
    risks = []
    for w1 in weights:
        w2 = 1 - w1
        port_return = w1 * expected_return_hdfc + w2 * expected_return_kotak
        port_var = (w1**2 * cov_hdfc) + (w2**2 * cov_kotak) + 2 * w1 * w2 * cov_cross
        port_risk = np.sqrt(port_var)
        returns.append(port_return)
        risks.append(port_risk)
    return np.array(risks), np.array(returns)

risks_uniform, returns_uniform = portfolio_metrics(weights_uniform)
risks_normal, returns_normal = portfolio_metrics(weights_normal)

min_var_uniform_idx = np.argmin(risks_uniform)
min_var_normal_idx = np.argmin(risks_normal)

# Plot Uniform Frontier
plt.figure(figsize=(10, 6))
plt.scatter(risks_uniform, returns_uniform, s=5, alpha=0.4)
plt.scatter(risks_uniform[min_var_uniform_idx], returns_uniform[min_var_uniform_idx], marker='x', s=60, color='red')
plt.xlabel('Portfolio Risk (σᵥ)')
plt.ylabel('Portfolio Return (μᵥ)')
plt.title('Efficient Frontier - Uniform Weights')
plt.grid(True)
plt.show()

# Plot Normal Frontier
plt.figure(figsize=(10, 6))
plt.scatter(risks_normal, returns_normal, s=5, alpha=0.4)
plt.scatter(risks_normal[min_var_normal_idx], returns_normal[min_var_normal_idx], marker='x', s=60, color='red')
plt.xlabel('Portfolio Risk (σᵥ)')
plt.ylabel('Portfolio Return (μᵥ)')
plt.title('Efficient Frontier - Normal Weights')
plt.grid(True)
plt.show()

print(f"Min Variance Uniform: Risk = {risks_uniform[min_var_uniform_idx]:.5f}, Return = {returns_uniform[min_var_uniform_idx]:.5f}")
print(f"Min Variance Normal: Risk = {risks_normal[min_var_normal_idx]:.5f}, Return = {returns_normal[min_var_normal_idx]:.5f}")



# QUESTION 2
# For multiple securities, compute weights and find return and minimum variance

import pandas as pd
import numpy as np
import yfinance as yf

tickers = ['HDFCBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS']

data = yf.download(tickers, start='2020-01-01', end='2025-10-01', interval='1mo', auto_adjust=True)

if isinstance(data.columns, pd.MultiIndex):
    data = data['Close'][tickers]
else:
    data = data[tickers]

data = data.dropna()

returns = data.pct_change().dropna().to_numpy()
T, n_assets = returns.shape

mu = np.sum(returns, axis=0) / T
sigma = np.sqrt(np.sum((returns - mu)**2, axis=0) / (T - 1))

cov_matrix = np.zeros((n_assets, n_assets))
for i in range(n_assets):
    for j in range(n_assets):
        cov_matrix[i, j] = np.sum((returns[:, i] - mu[i]) * (returns[:, j] - mu[j])) / (T - 1)

corr_matrix = np.zeros((n_assets, n_assets))
for i in range(n_assets):
    for j in range(n_assets):
        corr_matrix[i, j] = cov_matrix[i, j] / (sigma[i] * sigma[j])

ones = np.ones(n_assets)
inv_cov = np.linalg.inv(cov_matrix)
w_minvar = inv_cov @ ones / (ones.T @ inv_cov @ ones)

mu_v = np.dot(w_minvar, mu)
sigma_v2 = np.dot(w_minvar.T, np.dot(cov_matrix, w_minvar))
sigma_v = np.sqrt(sigma_v2)

print("=== Individual Asset Statistics ===")
for i, t in enumerate(tickers):
    print(f"{t}: Mean={mu[i]:.6f}, SD={sigma[i]:.6f}")

print("\n=== Covariance Matrix (3x3) ===")
for i in range(n_assets):
    print([round(float(x), 3) for x in cov_matrix[i]])

print("\n=== Minimum Variance Portfolio ===")
for t, w in zip(tickers, w_minvar):
    print(f"Weight of {t}: {w:.4f}")

print(f"\nPortfolio Expected Return (μv): {mu_v:.6f}")
print(f"Portfolio Variance (σv²): {sigma_v2:.6f}")
print(f"Portfolio Std. Deviation (σv): {sigma_v:.6f}")

# multiple securities

import pandas as pd
import numpy as np
import yfinance as yf

# Download data
df_HDFC = yf.download('HDFCBANK.NS', start='2020-01-01', end='2025-10-01', interval='1mo')
df_Kotak = yf.download('KOTAKBANK.NS', start='2020-01-01', end='2025-10-01', interval='1mo')
df_SBI = yf.download('SBIN.NS', start='2020-01-01', end='2025-10-01', interval='1mo')

# Prepare dataframes
df_HDFC.reset_index(inplace=True)
df_Kotak.reset_index(inplace=True)
df_SBI.reset_index(inplace=True)

df_HDFC.rename(columns={'Date': 'Month'}, inplace=True)
df_Kotak.rename(columns={'Date': 'Month'}, inplace=True)
df_SBI.rename(columns={'Date': 'Month'}, inplace=True)

df_HDFC = df_HDFC.sort_values(by='Month')
df_KOTAK = df_Kotak.sort_values(by='Month')
df_SBI = df_SBI.sort_values(by='Month')

# 1. Calculate monthly returns
df_HDFC['Rate_of_Return'] = (df_HDFC['Close'] - df_HDFC['Open']) / df_HDFC['Open']
df_KOTAK['Rate_of_Return'] = (df_KOTAK['Close'] - df_KOTAK['Open']) / df_KOTAK['Open']
df_SBI['Rate_of_Return'] = (df_SBI['Close'] - df_SBI['Open']) / df_SBI['Open']

# 2. Merge all returns into one DataFrame
returns_df = pd.DataFrame({
    'HDFC': df_HDFC['Rate_of_Return'].values,
    'KOTAK': df_KOTAK['Rate_of_Return'].values,
    'SBI': df_SBI['Rate_of_Return'].values
}).dropna()

# 3. Convert to numpy matrix
R = returns_df.values   # shape (T, n)
T, n = R.shape

# 4. Mean vector
mean_returns = np.mean(R, axis=0)

# 5. Manual covariance matrix
cov_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        cov_matrix[i, j] = np.sum((R[:, i] - mean_returns[i]) * (R[:, j] - mean_returns[j])) / (T - 1)

# Regularization (to avoid singular matrix)
cov_matrix += np.eye(n) * 1e-6

# 6. Minimum Variance Portfolio
U = np.ones(n)
C_inv = np.linalg.inv(cov_matrix)

numerator = U @ C_inv
denominator = U @ C_inv @ U.T
weights_min_var = numerator / denominator

print("Minimum Variance Portfolio Weights:")
for name, w in zip(returns_df.columns, weights_min_var):
    print(f"{name}: {w:.4f}")

# 7. Portfolio statistics
port_return = np.dot(weights_min_var, mean_returns)
port_variance = weights_min_var @ cov_matrix @ weights_min_var.T
port_risk = np.sqrt(port_variance)

print(f"\nExpected Portfolio Return (mu_p): {port_return:.5f}")
print(f"Portfolio Risk (sigma_p): {port_risk:.5f}")



# QUESTION 3
# Market vs Stock

import numpy as np

# Example returns (% converted to decimals)
R_xyz = np.array([0.02, 0.05, -0.01, 0.04, 0.03])
R_mkt = np.array([0.015, 0.04, -0.005, 0.035, 0.025])
prob = np.array([0.3, 0.25, 0.2, 0.2, 0.05])
Rf = 0.01  # 1% risk-free rate

T = len(R_xyz)

# mean
mu_xyz = np.sum(prob * R_xyz)
mu_mkt = np.sum(prob * R_mkt)

# Variances
sigma2_xyz = np.sum(prob * (R_xyz - mu_xyz) ** 2)
sigma2_mkt = np.sum(prob * (R_mkt - mu_mkt) ** 2)

# Covariance
cov_im = np.sum(prob * (R_xyz - mu_xyz) * (R_mkt - mu_mkt))

# Beta
beta_xyz = cov_im / sigma2_mkt

# Fair expected return (CAPM)
mu_v = Rf + beta_xyz * (mu_mkt - Rf)

print(f"Expected Return (μ_xyz): {mu_xyz:.4f}")
print(f"Market Return (μ_m): {mu_mkt:.4f}")
print(f"Variance of XYZ (σ²_xyz): {sigma2_xyz:.6f}")
print(f"Beta (β): {beta_xyz:.4f}")
print(f"Fair Return (μv): {mu_v:.4f}")

if mu_xyz > mu_v:
    print("Investment is GOOD (undervalued).")
else:
    print("Investment is NOT good (overvalued).")
