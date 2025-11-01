#ARMA
import numpy as np
import matplotlib.pyplot as plt

def generate_arma(phi, theta, n=200, c=0, sigma=1):
    """
    Generate ARMA(p,q) process.
    phi   : list of AR coefficients (phi_1 ... phi_p)
    theta : list of MA coefficients (theta_1 ... theta_q)
    n     : length of series
    c     : constant term
    sigma : noise std dev
    """
    p = len(phi)
    q = len(theta)
    X = np.zeros(n)
    eps = np.random.normal(0, sigma, n)

    for t in range(max(p,q), n):
        ar_part = sum(phi[i] * X[t-i-1] for i in range(p))
        ma_part = sum(theta[j] * eps[t-j-1] for j in range(q))
        X[t] = c + ar_part + eps[t] + ma_part
    return X

# Example: ARMA(2,1) process
phi = [0.6, -0.3]   # AR coefficients
theta = [0.5]       # MA coefficients
X = generate_arma(phi, theta, n=300)

plt.figure(figsize=(10,4))
plt.plot(X, label="ARMA(2,1) Simulated Series")
plt.legend()
plt.show()


# -------------------------------
# Step 2: Manual Forecasting
# -------------------------------
def arma_forecast(X, phi, theta, steps=10):
    """
    Forecast future values given ARMA coefficients.
    """
    p = len(phi)
    q = len(theta)
    n = len(X)
    eps = np.zeros(n + steps)   # assume future eps = 0 for forecast
    forecast = np.zeros(n + steps)
    forecast[:n] = X

    for t in range(n, n+steps):
        ar_part = sum(phi[i] * forecast[t-i-1] for i in range(p))
        ma_part = sum(theta[j] * eps[t-j-1] for j in range(q))
        forecast[t] = ar_part + ma_part  # no new noise term
    return forecast[n:]

# Forecast next 20 points
forecast_vals = arma_forecast(X, phi, theta, steps=20)

plt.figure(figsize=(10,4))
plt.plot(X, label="Observed")
plt.plot(range(len(X), len(X)+20), forecast_vals, "r--", label="Forecast")
plt.legend()
plt.show()

#ARIMA
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Simulate ARIMA(p,d,q) process
# -------------------------------
def generate_arima(phi, theta, d=1, n=200, c=0, sigma=1):
    """
    Generate ARIMA(p,d,q) process
    phi   : AR coefficients
    theta : MA coefficients
    d     : order of differencing
    n     : length of series
    c     : constant
    sigma : noise std
    """
    # First generate ARMA(p,q)
    p = len(phi)
    q = len(theta)
    X = np.zeros(n)
    eps = np.random.normal(0, sigma, n)

    for t in range(max(p,q), n):
        ar_part = sum(phi[i] * X[t-i-1] for i in range(p))
        ma_part = sum(theta[j] * eps[t-j-1] for j in range(q))
        X[t] = c + ar_part + eps[t] + ma_part

    # Integrate (reverse differencing) to make it ARIMA
    Y = X.copy()
    for _ in range(d):
        Y = np.cumsum(Y)   # integrate d times
    return Y

# Example: ARIMA(2,1,1)
phi = [0.6, -0.3]
theta = [0.5]
series = generate_arima(phi, theta, d=1, n=300)

plt.figure(figsize=(10,4))
plt.plot(series, label="Simulated ARIMA(2,1,1)")
plt.legend()
plt.show()

# -------------------------------
# Step 2: Differencing
# -------------------------------
def difference(series, d=1):
    """
    Difference a series d times
    """
    diff = series.copy()
    for _ in range(d):
        diff = np.diff(diff)
    return diff

d = 1
diff_series = difference(series, d)

plt.figure(figsize=(10,4))
plt.plot(diff_series, label="Differenced Series (Stationary)")
plt.legend()
plt.show()

# -------------------------------
# Step 3: Forecasting ARIMA
# -------------------------------
def arima_forecast(series, phi, theta, d=1, steps=10):
    """
    Forecast ARIMA(p,d,q) process with known params.
    """
    # Step A: Difference series
    diff_series = difference(series, d)

    # Step B: Forecast ARMA on differenced series
    p = len(phi)
    q = len(theta)
    n = len(diff_series)
    eps = np.zeros(n + steps)
    forecast = np.zeros(n + steps)
    forecast[:n] = diff_series

    for t in range(n, n+steps):
        ar_part = sum(phi[i] * forecast[t-i-1] for i in range(p))
        ma_part = sum(theta[j] * eps[t-j-1] for j in range(q))
        forecast[t] = ar_part + ma_part

    # Step C: Integrate back (reverse differencing)
    forecast_vals = forecast[n:]
    last_obs = series[-1]
    arima_forecast_vals = np.cumsum(forecast_vals) + last_obs
    return arima_forecast_vals

# Forecast next 20 values
forecast_vals = arima_forecast(series, phi, theta, d=1, steps=20)

plt.figure(figsize=(10,4))
plt.plot(series, label="Observed ARIMA Series")
plt.plot(range(len(series), len(series)+20), forecast_vals, "r--", label="Forecast")
plt.legend()
plt.show()

def needleman_wunsch(seq1, seq2,match=1, mismatch=-1,gap=-2):
   m,n=len(seq1),len(seq2)
   T=[[0]*(n+1) for _ in range(m+1)]
   for i in range(m+1):
      T[i][0]=i*gap
   for j in range(n+1):
      T[0][j]=j*gap
   for i in range(1,m+1):
      for j in range(1,n+1):
        T[i][j]=max(T[i-1][j-1]+(match if seq1[i-1]==seq2[j-1] else mismatch),
                    T[i-1][j]+gap,
                    T[i][j-1]+gap)
   print(T)
   align1,align2='',''
   i,j=m,n
   path=[(i,j)]
   while i>0 or j>0:
    if i>0 and j>0 and T[i][j]==T[i-1][j-1]+(match if seq1[i-1]==seq2[j-1] else mismatch):
      align1,align2=seq1[i-1]+align1,seq2[j-1]+align2
      i,j=i-1,j-1
    elif i>0 and T[i][j]==T[i-1][j]+gap:
      align1,align2=seq1[i-1]+align1,'-'+align2
      i-=1
    else:
      align1,align2='-'+align1,seq2[j-1]+align2
      j-=1
    path.append((i,j))
   path=np.array(path)
   return align1,align2,T,path


seq1, seq2 = "AGTCG", "ATCG"
a1, a2, T, path = needleman_wunsch(seq1, seq2)

df = pd.DataFrame(T,
                  index=["-"] + list(seq1),
                  columns=["-"] + list(seq2))
print(df)

line = "".join("|" if a1[i]==a2[i] else " " if "-" in (a1[i],a2[i]) else "." for i in range(len(a1)))
print("\nAlignment:\n" + a1 + "\n" + line + "\n" + a2)
plt.imshow(T, cmap='Blues', origin='upper')
plt.colorbar(label='Score')
plt.xticks(np.arange(len(seq2)+1),['-'] + list(seq2))
plt.yticks(np.arange(len(seq1)+1), ['-']+ list(seq1))
plt.plot(path[:,1], path[:,0], color='red', linewidth=2, marker='o')
plt.title("Needleman-Wunsch Alignment Matrix")
plt.xlabel("Sequence 1")
plt.ylabel("Sequence 2")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
