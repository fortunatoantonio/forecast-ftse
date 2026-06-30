import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from pandas.tseries.frequencies import to_offset
from scipy import stats
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from prophet import Prophet

# --------------------------
# Caricamento e preprocessing
# --------------------------
dataset = pd.read_excel("FTSE_MIB_Futures_con_stagionalita.xlsx")
dataset["date"] = pd.to_datetime(dataset["date"])
dataset.set_index("date", inplace=True)
dataset = dataset.sort_index()

# Assegna la frequenza all'indice (infer_freq fallirà per le festività, usiamo 'B' come fallback)
if dataset.index.freq is None:
    dataset.index = pd.DatetimeIndex(dataset.index.values)
    dataset.index._data._freq = to_offset(pd.infer_freq(dataset.index) or 'B')

serie = dataset["closed"].copy()
serie = serie.replace([np.inf, -np.inf], np.nan).dropna()
serie_ma = serie.rolling(window=5, min_periods=5).mean().dropna()
serie_for_plot = serie.loc[serie_ma.index]

stat_originale = serie.describe()
stat_smussata = serie_ma.describe()

print("Statistiche serie originale:")
print(stat_originale)
print("\nStatistiche serie smussata:")
print(stat_smussata)

plt.figure(figsize=(10, 4))
plt.plot(serie_for_plot.index, serie_for_plot.values, label="Serie originale", linewidth=1.2)
plt.plot(serie_ma.index, serie_ma.values, label="Media mobile (5)", linewidth=2)
plt.title("Serie originale vs Media mobile (5)")
plt.xlabel("Data")
plt.ylabel("Prezzo di chiusura")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("serie_vs_media_mobile5.png", dpi=150, bbox_inches="tight")
plt.close()

# =============================================================================
# Figure didattiche per i sottoparagrafi 2.1 (Fondamenti Prophet: trend, stagionalità, ecc.)
# =============================================================================
print("\n📐 Generazione figure didattiche per Sez. 2.1 (trend, stagionalità, regressori)...")
t = np.linspace(0, 800, 801)  # 801 giorni (~2.2 anni) per asse tempo
P_yearly = 365.25
np.random.seed(42)

# --- 2.1.1 Architettura additiva: y = g(t) + s(t) + h(t) + epsilon ---
# Trend sintetico (piecewise linear semplificato)
g_t = 100 + 0.08 * t
g_t[t > 300] = 100 + 0.08 * 300 + 0.02 * (t[t > 300] - 300)
g_t[t > 500] = 100 + 0.08 * 300 + 0.02 * 200 - 0.03 * (t[t > 500] - 500)
# Stagionalità Fourier (solo primi 3 termini per chiarezza)
N_fourier = 3
s_t = np.zeros_like(t)
for n in range(1, N_fourier + 1):
    s_t += 2.0 / n * np.cos(2 * np.pi * n * t / P_yearly) + 1.5 / n * np.sin(2 * np.pi * n * t / P_yearly)
# Regressore sintetico (contributo lineare)
h_t = 0.5 * np.sin(t / 50) * 3  # oscillazione lenta come placeholder per regressori
# Serie osservata = g + s + h + rumore
y_t = g_t + s_t + h_t + np.random.normal(0, 2, size=len(t))

fig_211, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
axes[0].plot(t, g_t, 'b-', linewidth=2, label='g(t) trend')
axes[0].set_ylabel('g(t)')
axes[0].set_title('2.1.1 Architettura additiva: y = g(t) + s(t) + h(t) + ε')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)
axes[1].plot(t, s_t, 'green', linewidth=1.5, label='s(t) stagionalità')
axes[1].set_ylabel('s(t)')
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)
axes[2].plot(t, h_t, 'orange', linewidth=1.5, label='h(t) regressori')
axes[2].set_ylabel('h(t)')
axes[2].legend(loc='upper right')
axes[2].grid(True, alpha=0.3)
axes[3].plot(t, g_t + s_t + h_t, 'purple', linewidth=1.5, label='g(t)+s(t)+h(t)')
axes[3].set_ylabel('Segnale')
axes[3].legend(loc='upper right')
axes[3].grid(True, alpha=0.3)
axes[4].plot(t, y_t, 'gray', linewidth=0.8, alpha=0.8, label='y(t) osservato')
axes[4].set_xlabel('Tempo (giorni)')
axes[4].set_ylabel('y(t)')
axes[4].legend(loc='upper right')
axes[4].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("prophet_2_1_1_additive_decomposition.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ prophet_2_1_1_additive_decomposition.png")

# --- 2.1.2 Componente di trend: piecewise linear con changepoints ---
# Simulazione trend a tratti: pendenza k che cambia in 3 punti
cp = [200, 400, 600]  # changepoint (indici)
k0, d1, d2, d3 = 0.06, -0.04, 0.02, -0.03  # crescita base + variazioni
g_piece = np.zeros_like(t)
g_piece[:cp[0]] = 50 + k0 * t[:cp[0]]
g_piece[cp[0]:cp[1]] = 50 + k0 * cp[0] + (k0 + d1) * (t[cp[0]:cp[1]] - cp[0])
g_piece[cp[1]:cp[2]] = g_piece[cp[1]-1] + (k0 + d1 + d2) * (t[cp[1]:cp[2]] - t[cp[1]])
g_piece[cp[2]:] = g_piece[cp[2]-1] + (k0 + d1 + d2 + d3) * (t[cp[2]:] - t[cp[2]])

fig_212, ax = plt.subplots(figsize=(10, 4))
ax.plot(t, g_piece, 'b-', linewidth=2, label='g(t) trend lineare a tratti')
for i, c in enumerate(cp):
    ax.axvline(t[c], color='red', linestyle='--', alpha=0.7, label='Changepoint' if i == 0 else None)
ax.set_xlabel('Tempo (giorni)')
ax.set_ylabel('g(t)')
ax.set_title('2.1.2 La componente di trend: piecewise linear con changepoints')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("prophet_2_1_2_trend_piecewise_linear.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ prophet_2_1_2_trend_piecewise_linear.png")

# --- 2.1.3 Componente stagionale: Fourier s(t) = sum a_n cos(2πnt/P) + b_n sin(2πnt/P) ---
# Figura didattica: confronto 1ª armonica (onda liscia) vs somma di N=10 armoniche
N = 10
np.random.seed(123)
a_n = np.random.randn(N) * 0.4
b_n = np.random.randn(N) * 0.4
t_one_year = np.linspace(0, P_yearly, 366)
# Prima armonica sola (n=1): un'onda con un ciclo in P giorni
solo_armonica1 = a_n[0] * np.cos(2 * np.pi * 1 * t_one_year / P_yearly) + b_n[0] * np.sin(2 * np.pi * 1 * t_one_year / P_yearly)
# Somma di tutte e 10 le armoniche
s_full = np.zeros_like(t_one_year)
for n in range(1, N + 1):
    s_full += a_n[n-1] * np.cos(2 * np.pi * n * t_one_year / P_yearly) + b_n[n-1] * np.sin(2 * np.pi * n * t_one_year / P_yearly)
# Asse x in mesi (0–12) per leggibilità
mesi = t_one_year / (P_yearly / 12)
mesi_labels = ['Gen', 'Feb', 'Mar', 'Apr', 'Mag', 'Giu', 'Lug', 'Ago', 'Set', 'Ott', 'Nov', 'Dic']
mesi_ticks = np.arange(0, 13) - 0.5  # centri approssimativi dei mesi

fig_213, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
# Pannello 1: confronto 1 armonica vs 10 armoniche
ax1.plot(mesi, solo_armonica1, 'steelblue', linewidth=2.5, label='Solo 1ª armonica (n=1): un\'onda liscia all\'anno')
ax1.plot(mesi, s_full, 'green', linewidth=1.8, label='Somma di N=10 armoniche: pattern più articolato')
ax1.set_ylabel('s(t)')
ax1.set_xlim(0, 12)
ax1.set_xticks(np.arange(12) + 0.5)
ax1.set_xticklabels(mesi_labels)
ax1.set_title('Come si costruisce la stagionalità: una sola onda vs somma di 10 onde (Fourier)')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
# Pannello 2: stagionalità completa su un anno (stessa curva s_full, evidenziata)
ax2.plot(mesi, s_full, 'green', linewidth=2.2, label='s(t) = somma di 10 armoniche (formula 2.4)')
ax2.set_xlabel('Mesi dell\'anno')
ax2.set_ylabel('s(t)')
ax2.set_xlim(0, 12)
ax2.set_xticks(np.arange(12) + 0.5)
ax2.set_xticklabels(mesi_labels)
ax2.set_title('Andamento della componente stagionale su un anno (P = 365.25 giorni)')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("prophet_2_1_3_seasonal_fourier.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ prophet_2_1_3_seasonal_fourier.png")

# --- 2.1.4 Regressori esogeni: h(t) = sum beta_j * x_j(t) ---
# Due regressori fittizi (tipo momentum) e loro contributo additivo
x1 = np.cumsum(np.random.randn(len(t)) * 0.5)  # tipo momentum 1
x2 = np.cumsum(np.random.randn(len(t)) * 0.3)  # tipo momentum 3
beta1, beta2 = 0.8, 0.4
h_reg = beta1 * x1 + beta2 * x2
fig_214, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
axes[0].plot(t, x1, 'b-', alpha=0.8, label='x₁(t) (es. momentum 1)')
axes[0].set_ylabel('x₁(t)')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)
axes[1].plot(t, x2, 'orange', alpha=0.8, label='x₂(t) (es. momentum 3)')
axes[1].set_ylabel('x₂(t)')
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)
axes[2].plot(t, h_reg, 'purple', linewidth=1.2, label=f'h(t) = β₁x₁ + β₂x₂  (β₁={beta1}, β₂={beta2})')
axes[2].set_xlabel('Tempo (giorni)')
axes[2].set_ylabel('h(t)')
axes[2].set_title('2.1.4 Regressori esogeni: contributo additivo h(t) = Σ βⱼ xⱼ(t)')
axes[2].legend(loc='upper right')
axes[2].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("prophet_2_1_4_regressori_esogeni.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ prophet_2_1_4_regressori_esogeni.png")

# --- 2.1.5 seasonality_prior_scale: effetto della regolarizzazione sulla stagionalità ---
# Stagionalità "forte" (prior alto → coefficienti più grandi) vs "debole" (prior basso → più liscia)
scale_alto = 1.0   # prior più permissivo → oscillazioni ampie
scale_basso = 0.1  # prior restrittivo → oscillazioni smorzate
a_alto = np.random.randn(N) * scale_alto
b_alto = np.random.randn(N) * scale_alto
a_basso = np.random.randn(N) * scale_basso
b_basso = np.random.randn(N) * scale_basso
s_alto = np.zeros_like(t_one_year)
s_basso = np.zeros_like(t_one_year)
for n in range(1, N + 1):
    s_alto += a_alto[n-1] * np.cos(2 * np.pi * n * t_one_year / P_yearly) + b_alto[n-1] * np.sin(2 * np.pi * n * t_one_year / P_yearly)
    s_basso += a_basso[n-1] * np.cos(2 * np.pi * n * t_one_year / P_yearly) + b_basso[n-1] * np.sin(2 * np.pi * n * t_one_year / P_yearly)

fig_215, (ax_high, ax_low) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax_high.plot(t_one_year, s_alto, 'darkgreen', linewidth=1.5, label='seasonality_prior_scale alto (es. 10): stagionalità flessibile')
ax_high.set_ylabel('s(t)')
ax_high.legend(loc='upper right')
ax_high.grid(True, alpha=0.3)
ax_low.plot(t_one_year, s_basso, 'teal', linewidth=1.5, label='seasonality_prior_scale basso (es. 0.1): stagionalità regolarizzata')
ax_low.set_xlabel('Giorni (un anno)')
ax_low.set_ylabel('s(t)')
ax_low.set_title('2.1.5 Il ruolo di seasonality_prior_scale: flessibilità vs regolarizzazione')
ax_low.legend(loc='upper right')
ax_low.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("prophet_2_1_5_seasonality_prior_scale.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ prophet_2_1_5_seasonality_prior_scale.png")

# Configurazione del back-test: training lungo tutta la serie tranne gli ultimi 10 punti,
# che vengono usati come verifica out-of-sample.
test_size = 10  # Ultime 10 osservazioni come test
train_size = len(serie_ma) - test_size  # Tutto il resto come training

serie_ma_train = serie_ma.iloc[:train_size]
serie_ma_test = serie_ma.iloc[train_size:train_size + test_size]

print(f"\nOsservazioni train: {len(serie_ma_train)}")
print(f"Osservazioni test: {len(serie_ma_test)}")

# --------------------------
# Calcolo Momentum per regressori esogeni
# --------------------------
print("\n📊 Calcolo momentum 1, 3, 5, 10...")
momentum_1 = serie_ma.diff(1).dropna()
momentum_3 = serie_ma.diff(3).dropna()
momentum_5 = serie_ma.diff(5).dropna()
momentum_10 = serie_ma.diff(10).dropna()

# Allinea tutti i momentum (intersezione degli indici)
common_idx = momentum_1.index.intersection(momentum_3.index).intersection(momentum_5.index).intersection(momentum_10.index)
serie_ma_aligned = serie_ma.loc[common_idx]
momentum_1_aligned = momentum_1.loc[common_idx]
momentum_3_aligned = momentum_3.loc[common_idx]
momentum_5_aligned = momentum_5.loc[common_idx]
momentum_10_aligned = momentum_10.loc[common_idx]

# Ricalcola train/test allineati
train_size_aligned = len(serie_ma_aligned) - test_size
serie_ma_train_aligned = serie_ma_aligned.iloc[:train_size_aligned]
serie_ma_test_aligned = serie_ma_aligned.iloc[train_size_aligned:train_size_aligned + test_size]

momentum_1_train = momentum_1_aligned.iloc[:train_size_aligned]
momentum_1_test = momentum_1_aligned.iloc[train_size_aligned:train_size_aligned + test_size]
momentum_3_train = momentum_3_aligned.iloc[:train_size_aligned]
momentum_3_test = momentum_3_aligned.iloc[train_size_aligned:train_size_aligned + test_size]
momentum_5_train = momentum_5_aligned.iloc[:train_size_aligned]
momentum_5_test = momentum_5_aligned.iloc[train_size_aligned:train_size_aligned + test_size]
momentum_10_train = momentum_10_aligned.iloc[:train_size_aligned]
momentum_10_test = momentum_10_aligned.iloc[train_size_aligned:train_size_aligned + test_size]

print(f"Dati allineati - Train: {len(serie_ma_train_aligned)}, Test: {len(serie_ma_test_aligned)}")

# --------------------------
# MODELLO 1: Solo annuale + seasonality 0.1 + Momentum 1,3,5,10
# --------------------------
print("\n" + "="*70)
print("🤖 MODELLO 1: Solo annuale + seasonality 0.1 + Momentum 1,3,5,10")
print("="*70)

df_train_1 = pd.DataFrame({
    'ds': serie_ma_train_aligned.index,
    'y': serie_ma_train_aligned.values,
    'momentum_1': momentum_1_train.values,
    'momentum_3': momentum_3_train.values,
    'momentum_5': momentum_5_train.values,
    'momentum_10': momentum_10_train.values
})

future_1 = pd.DataFrame({
    'ds': serie_ma_test_aligned.index,
    'momentum_1': momentum_1_test.values,
    'momentum_3': momentum_3_test.values,
    'momentum_5': momentum_5_test.values,
    'momentum_10': momentum_10_test.values
})

model_1 = Prophet(
    daily_seasonality=False,
    weekly_seasonality=False,
    yearly_seasonality=True,
    seasonality_mode='additive',
    seasonality_prior_scale=0.1
)

model_1.add_regressor('momentum_1')
model_1.add_regressor('momentum_3')
model_1.add_regressor('momentum_5')
model_1.add_regressor('momentum_10')

print("Addestramento Modello 1...")
model_1.fit(df_train_1)

# Previsioni
forecast_1_test = model_1.predict(future_1)
forecast_1_train = model_1.predict(df_train_1)

y_pred_1_test = forecast_1_test['yhat'].values
y_true_1_test = serie_ma_test_aligned.values
y_pred_1_train = forecast_1_train['yhat'].values
y_true_1_train = df_train_1['y'].values

# Metriche Modello 1
mae_1_test = np.mean(np.abs(y_true_1_test - y_pred_1_test))
rmse_1_test = np.sqrt(np.mean((y_true_1_test - y_pred_1_test) ** 2))
mape_1_test = mean_absolute_percentage_error(y_true_1_test, y_pred_1_test)
r2_1_test = r2_score(y_true_1_test, y_pred_1_test)

mae_1_train = np.mean(np.abs(y_true_1_train - y_pred_1_train))
rmse_1_train = np.sqrt(np.mean((y_true_1_train - y_pred_1_train) ** 2))
mape_1_train = mean_absolute_percentage_error(y_true_1_train, y_pred_1_train)
r2_1_train = r2_score(y_true_1_train, y_pred_1_train)

# Previsioni su tutto il dataset
df_full_1 = pd.DataFrame({
    'ds': serie_ma_aligned.index,
    'momentum_1': momentum_1_aligned.values,
    'momentum_3': momentum_3_aligned.values,
    'momentum_5': momentum_5_aligned.values,
    'momentum_10': momentum_10_aligned.values
})
forecast_1_full = model_1.predict(df_full_1)
y_pred_1_full = forecast_1_full['yhat'].values
y_true_1_full = serie_ma_aligned.values

# Metriche su tutto il dataset - Modello 1
mae_1_full = np.mean(np.abs(y_true_1_full - y_pred_1_full))
rmse_1_full = np.sqrt(np.mean((y_true_1_full - y_pred_1_full) ** 2))
mape_1_full = mean_absolute_percentage_error(y_true_1_full, y_pred_1_full)
r2_1_full = r2_score(y_true_1_full, y_pred_1_full)

print(f"\n📊 METRICHE MODELLO 1:")
print(f"  TEST  - R²: {r2_1_test:.4f} | MAPE: {mape_1_test*100:.2f}% | MAE: {mae_1_test:.4f} | RMSE: {rmse_1_test:.4f}")
print(f"  TRAIN - R²: {r2_1_train:.4f} | MAPE: {mape_1_train*100:.2f}% | MAE: {mae_1_train:.4f} | RMSE: {rmse_1_train:.4f}")
print(f"  FULL  - R²: {r2_1_full:.4f} | MAPE: {mape_1_full*100:.2f}% | MAE: {mae_1_full:.4f} | RMSE: {rmse_1_full:.4f}")

# --------------------------
# MODELLO 2: Solo annuale + seasonality 0.15 + Momentum 3,10
# --------------------------
print("\n" + "="*70)
print("🤖 MODELLO 2: Solo annuale + seasonality 0.15 + Momentum 3,10")
print("="*70)

df_train_2 = pd.DataFrame({
    'ds': serie_ma_train_aligned.index,
    'y': serie_ma_train_aligned.values,
    'momentum_3': momentum_3_train.values,
    'momentum_10': momentum_10_train.values
})

future_2 = pd.DataFrame({
    'ds': serie_ma_test_aligned.index,
    'momentum_3': momentum_3_test.values,
    'momentum_10': momentum_10_test.values
})

model_2 = Prophet(
    daily_seasonality=False,
    weekly_seasonality=False,
    yearly_seasonality=True,
    seasonality_mode='additive',
    seasonality_prior_scale=0.15
)

model_2.add_regressor('momentum_3')
model_2.add_regressor('momentum_10')

print("Addestramento Modello 2...")
model_2.fit(df_train_2)

# Previsioni
forecast_2_test = model_2.predict(future_2)
forecast_2_train = model_2.predict(df_train_2)

y_pred_2_test = forecast_2_test['yhat'].values
y_true_2_test = serie_ma_test_aligned.values
y_pred_2_train = forecast_2_train['yhat'].values
y_true_2_train = df_train_2['y'].values

# Metriche Modello 2
mae_2_test = np.mean(np.abs(y_true_2_test - y_pred_2_test))
rmse_2_test = np.sqrt(np.mean((y_true_2_test - y_pred_2_test) ** 2))
mape_2_test = mean_absolute_percentage_error(y_true_2_test, y_pred_2_test)
r2_2_test = r2_score(y_true_2_test, y_pred_2_test)

mae_2_train = np.mean(np.abs(y_true_2_train - y_pred_2_train))
rmse_2_train = np.sqrt(np.mean((y_true_2_train - y_pred_2_train) ** 2))
mape_2_train = mean_absolute_percentage_error(y_true_2_train, y_pred_2_train)
r2_2_train = r2_score(y_true_2_train, y_pred_2_train)

# Previsioni su tutto il dataset
df_full_2 = pd.DataFrame({
    'ds': serie_ma_aligned.index,
    'momentum_3': momentum_3_aligned.values,
    'momentum_10': momentum_10_aligned.values
})
forecast_2_full = model_2.predict(df_full_2)
y_pred_2_full = forecast_2_full['yhat'].values
y_true_2_full = serie_ma_aligned.values

# Metriche su tutto il dataset - Modello 2
mae_2_full = np.mean(np.abs(y_true_2_full - y_pred_2_full))
rmse_2_full = np.sqrt(np.mean((y_true_2_full - y_pred_2_full) ** 2))
mape_2_full = mean_absolute_percentage_error(y_true_2_full, y_pred_2_full)
r2_2_full = r2_score(y_true_2_full, y_pred_2_full)

print(f"\n📊 METRICHE MODELLO 2:")
print(f"  TEST  - R²: {r2_2_test:.4f} | MAPE: {mape_2_test*100:.2f}% | MAE: {mae_2_test:.4f} | RMSE: {rmse_2_test:.4f}")
print(f"  TRAIN - R²: {r2_2_train:.4f} | MAPE: {mape_2_train*100:.2f}% | MAE: {mae_2_train:.4f} | RMSE: {rmse_2_train:.4f}")
print(f"  FULL  - R²: {r2_2_full:.4f} | MAPE: {mape_2_full*100:.2f}% | MAE: {mae_2_full:.4f} | RMSE: {rmse_2_full:.4f}")

# --------------------------
# RIEPILOGO FINALE - METRICHE FONDAMENTALI
# --------------------------
print("\n" + "="*70)
print("🎯 RIEPILOGO FINALE - METRICHE FONDAMENTALI")
print("="*70)

print(f"\n{'METRICA':<25} {'MODELLO 1':<50} {'MODELLO 2':<50}")
print("-"*125)
print(f"{'':<25} {'TRAIN':<16} {'TEST':<16} {'FULL':<16} {'TRAIN':<16} {'TEST':<16} {'FULL':<16}")
print("-"*125)

r2_1_train_str = f"{r2_1_train:.4f}".ljust(16)
r2_1_test_str = f"{r2_1_test:.4f}".ljust(16)
r2_1_full_str = f"{r2_1_full:.4f}".ljust(16)
r2_2_train_str = f"{r2_2_train:.4f}".ljust(16)
r2_2_test_str = f"{r2_2_test:.4f}".ljust(16)
r2_2_full_str = f"{r2_2_full:.4f}".ljust(16)

mape_1_train_str = f"{mape_1_train*100:.2f}".ljust(16)
mape_1_test_str = f"{mape_1_test*100:.2f}".ljust(16)
mape_1_full_str = f"{mape_1_full*100:.2f}".ljust(16)
mape_2_train_str = f"{mape_2_train*100:.2f}".ljust(16)
mape_2_test_str = f"{mape_2_test*100:.2f}".ljust(16)
mape_2_full_str = f"{mape_2_full*100:.2f}".ljust(16)

mae_1_train_str = f"{mae_1_train:.4f}".ljust(16)
mae_1_test_str = f"{mae_1_test:.4f}".ljust(16)
mae_1_full_str = f"{mae_1_full:.4f}".ljust(16)
mae_2_train_str = f"{mae_2_train:.4f}".ljust(16)
mae_2_test_str = f"{mae_2_test:.4f}".ljust(16)
mae_2_full_str = f"{mae_2_full:.4f}".ljust(16)

rmse_1_train_str = f"{rmse_1_train:.4f}".ljust(16)
rmse_1_test_str = f"{rmse_1_test:.4f}".ljust(16)
rmse_1_full_str = f"{rmse_1_full:.4f}".ljust(16)
rmse_2_train_str = f"{rmse_2_train:.4f}".ljust(16)
rmse_2_test_str = f"{rmse_2_test:.4f}".ljust(16)
rmse_2_full_str = f"{rmse_2_full:.4f}".ljust(16)

print(f"{'R² Score':<25} {r2_1_train_str} {r2_1_test_str} {r2_1_full_str} {r2_2_train_str} {r2_2_test_str} {r2_2_full_str}")
print(f"{'MAPE (%)':<25} {mape_1_train_str} {mape_1_test_str} {mape_1_full_str} {mape_2_train_str} {mape_2_test_str} {mape_2_full_str}")
print(f"{'MAE':<25} {mae_1_train_str} {mae_1_test_str} {mae_1_full_str} {mae_2_train_str} {mae_2_test_str} {mae_2_full_str}")
print(f"{'RMSE':<25} {rmse_1_train_str} {rmse_1_test_str} {rmse_1_full_str} {rmse_2_train_str} {rmse_2_test_str} {rmse_2_full_str}")

print("\n" + "="*70)
print("🏆 MIGLIOR MODELLO")
print("="*70)
if r2_1_test > r2_2_test:
    print("Modello 1: Solo annuale + seasonality 0.1 + Momentum 1,3,5,10")
    print(f"  R² Test: {r2_1_test:.4f}")
    print(f"  MAPE Test: {mape_1_test*100:.2f}%")
    best_model = model_1
    best_forecast_test = forecast_1_test
    best_forecast_train = forecast_1_train
    best_forecast_full = forecast_1_full
    best_y_pred_test = y_pred_1_test
    best_y_true_test = y_true_1_test
    best_y_pred_train = y_pred_1_train
    best_y_true_train = y_true_1_train
    best_y_pred_full = y_pred_1_full
    best_y_true_full = y_true_1_full
    best_name = "Modello 1: Solo annuale + seasonality 0.1 + Momentum 1,3,5,10"
    best_r2_test = r2_1_test
    best_mape_test = mape_1_test
    best_r2_train = r2_1_train
    best_mape_train = mape_1_train
    best_r2_full = r2_1_full
    best_mape_full = mape_1_full
else:
    print("Modello 2: Solo annuale + seasonality 0.15 + Momentum 3,10")
    print(f"  R² Test: {r2_2_test:.4f}")
    print(f"  MAPE Test: {mape_2_test*100:.2f}%")
    best_model = model_2
    best_forecast_test = forecast_2_test
    best_forecast_train = forecast_2_train
    best_forecast_full = forecast_2_full
    best_y_pred_test = y_pred_2_test
    best_y_true_test = y_true_2_test
    best_y_pred_train = y_pred_2_train
    best_y_true_train = y_true_2_train
    best_y_pred_full = y_pred_2_full
    best_y_true_full = y_true_2_full
    best_name = "Modello 2: Solo annuale + seasonality 0.15 + Momentum 3,10"
    best_r2_test = r2_2_test
    best_mape_test = mape_2_test
    best_r2_train = r2_2_train
    best_mape_train = mape_2_train
    best_r2_full = r2_2_full
    best_mape_full = mape_2_full

print("="*70)

# --------------------------
# Grafici aggiuntivi per il capitolo di tesi
# --------------------------

# Grafico momentum
print("\n📈 Creazione grafici aggiuntivi...")
fig_mom, axes_mom = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
for ax_m, (mom, label) in zip(axes_mom, [
    (momentum_1_aligned, 'Momentum 1 (variazione 1 giorno)'),
    (momentum_3_aligned, 'Momentum 3 (variazione 3 giorni)'),
    (momentum_5_aligned, 'Momentum 5 (variazione 5 giorni)'),
    (momentum_10_aligned, 'Momentum 10 (variazione 10 giorni)')]):
    ax_m.plot(mom.index, mom.values, linewidth=0.7, alpha=0.8)
    ax_m.axhline(y=0, color='red', linestyle='--', linewidth=0.8)
    ax_m.set_ylabel('Punti')
    ax_m.set_title(label, fontsize=10)
    ax_m.grid(True, alpha=0.3)
axes_mom[-1].set_xlabel('Data')
fig_mom.suptitle('Regressori di Momentum utilizzati nel modello Prophet', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("prophet_momentum_regressors.png", dpi=150, bbox_inches="tight")
plt.close()

# Decomposizione componenti Prophet (Modello 1 - migliore)
fig_comp = best_model.plot_components(best_forecast_full)
fig_comp.suptitle(f'Decomposizione componenti - {best_name}', fontsize=12, fontweight='bold', y=1.02)
fig_comp.savefig("prophet_decomposition.png", dpi=150, bbox_inches="tight")
plt.close(fig_comp)

# Residui Modello 1
residui_1_train = y_true_1_train - y_pred_1_train
fig_res1, axes_res1 = plt.subplots(2, 2, figsize=(12, 8))
axes_res1[0, 0].plot(df_train_1['ds'], residui_1_train, linewidth=0.7, color="tab:blue")
axes_res1[0, 0].axhline(y=0, color="red", linestyle="--", linewidth=1)
axes_res1[0, 0].set_title("Residui nel Tempo - Modello 1")
axes_res1[0, 0].set_xlabel("Data")
axes_res1[0, 0].set_ylabel("Residuo")
axes_res1[0, 0].grid(True, alpha=0.3)

axes_res1[0, 1].hist(residui_1_train, bins=40, density=True, alpha=0.7, color="tab:blue", edgecolor="black")
mu_r1, sigma_r1 = np.mean(residui_1_train), np.std(residui_1_train)
x_r1 = np.linspace(residui_1_train.min(), residui_1_train.max(), 200)
if sigma_r1 > 0:
    axes_res1[0, 1].plot(x_r1, stats.norm.pdf(x_r1, mu_r1, sigma_r1), "r-", linewidth=2, label="Normale teorica")
axes_res1[0, 1].set_title("Distribuzione Residui")
axes_res1[0, 1].legend()
axes_res1[0, 1].grid(True, alpha=0.3)

stats.probplot(residui_1_train, dist="norm", plot=axes_res1[1, 0])
axes_res1[1, 0].set_title("Q-Q Plot Residui")
axes_res1[1, 0].grid(True, alpha=0.3)

plot_acf(residui_1_train, lags=min(40, len(residui_1_train) // 4), ax=axes_res1[1, 1], alpha=0.05)
axes_res1[1, 1].set_title("ACF Residui")
axes_res1[1, 1].grid(True, alpha=0.3)

fig_res1.suptitle("Diagnostica Residui - Modello 1 Prophet", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("prophet_residual_analysis_modello1.png", dpi=150, bbox_inches="tight")
plt.close()

# Residui Modello 2
residui_2_train = y_true_2_train - y_pred_2_train
fig_res2, axes_res2 = plt.subplots(2, 2, figsize=(12, 8))
axes_res2[0, 0].plot(df_train_2['ds'], residui_2_train, linewidth=0.7, color="tab:blue")
axes_res2[0, 0].axhline(y=0, color="red", linestyle="--", linewidth=1)
axes_res2[0, 0].set_title("Residui nel Tempo - Modello 2")
axes_res2[0, 0].set_xlabel("Data")
axes_res2[0, 0].set_ylabel("Residuo")
axes_res2[0, 0].grid(True, alpha=0.3)

axes_res2[0, 1].hist(residui_2_train, bins=40, density=True, alpha=0.7, color="tab:blue", edgecolor="black")
mu_r2, sigma_r2 = np.mean(residui_2_train), np.std(residui_2_train)
x_r2 = np.linspace(residui_2_train.min(), residui_2_train.max(), 200)
if sigma_r2 > 0:
    axes_res2[0, 1].plot(x_r2, stats.norm.pdf(x_r2, mu_r2, sigma_r2), "r-", linewidth=2, label="Normale teorica")
axes_res2[0, 1].set_title("Distribuzione Residui")
axes_res2[0, 1].legend()
axes_res2[0, 1].grid(True, alpha=0.3)

stats.probplot(residui_2_train, dist="norm", plot=axes_res2[1, 0])
axes_res2[1, 0].set_title("Q-Q Plot Residui")
axes_res2[1, 0].grid(True, alpha=0.3)

plot_acf(residui_2_train, lags=min(40, len(residui_2_train) // 4), ax=axes_res2[1, 1], alpha=0.05)
axes_res2[1, 1].set_title("ACF Residui")
axes_res2[1, 1].grid(True, alpha=0.3)

fig_res2.suptitle("Diagnostica Residui - Modello 2 Prophet", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("prophet_residual_analysis_modello2.png", dpi=150, bbox_inches="tight")
plt.close()

# Statistiche residui
print("\nAnalisi residui Modello 1 (training):")
print(pd.Series(residui_1_train).describe())
print("\nAnalisi residui Modello 2 (training):")
print(pd.Series(residui_2_train).describe())

# Confronto metriche R²/MAPE
fig_metr, axes_metr = plt.subplots(1, 2, figsize=(12, 4))
x_labels = ["Train", "Test", "Full"]
axes_metr[0].bar(x_labels, [r2_1_train * 100, r2_1_test * 100, r2_1_full * 100], color=["tab:blue", "tab:orange", "tab:green"])
axes_metr[0].set_title("Confronto R² Train vs Test vs Full - Modello 1")
axes_metr[0].set_ylabel("R² (%)")
axes_metr[0].grid(True, axis="y", alpha=0.3)
for idx_b, val_b in enumerate([r2_1_train * 100, r2_1_test * 100, r2_1_full * 100]):
    axes_metr[0].text(idx_b, val_b + 1, f"{val_b:.2f}%", ha="center", fontweight="bold")

axes_metr[1].bar(x_labels, [mape_1_train * 100, mape_1_test * 100, mape_1_full * 100], color=["tab:blue", "tab:orange", "tab:green"])
axes_metr[1].set_title("Confronto MAPE Train vs Test vs Full - Modello 1")
axes_metr[1].set_ylabel("MAPE (%)")
axes_metr[1].grid(True, axis="y", alpha=0.3)
for idx_b, val_b in enumerate([mape_1_train * 100, mape_1_test * 100, mape_1_full * 100]):
    axes_metr[1].text(idx_b, val_b + 0.02, f"{val_b:.3f}%", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("prophet_confronto_metriche_mod1.png", dpi=150, bbox_inches="tight")
plt.close()

fig_metr2, axes_metr2 = plt.subplots(1, 2, figsize=(12, 4))
axes_metr2[0].bar(x_labels, [r2_2_train * 100, r2_2_test * 100, r2_2_full * 100], color=["tab:blue", "tab:orange", "tab:green"])
axes_metr2[0].set_title("Confronto R² Train vs Test vs Full - Modello 2")
axes_metr2[0].set_ylabel("R² (%)")
axes_metr2[0].grid(True, axis="y", alpha=0.3)
for idx_b, val_b in enumerate([r2_2_train * 100, r2_2_test * 100, r2_2_full * 100]):
    axes_metr2[0].text(idx_b, val_b + 1, f"{val_b:.2f}%", ha="center", fontweight="bold")

axes_metr2[1].bar(x_labels, [mape_2_train * 100, mape_2_test * 100, mape_2_full * 100], color=["tab:blue", "tab:orange", "tab:green"])
axes_metr2[1].set_title("Confronto MAPE Train vs Test vs Full - Modello 2")
axes_metr2[1].set_ylabel("MAPE (%)")
axes_metr2[1].grid(True, axis="y", alpha=0.3)
for idx_b, val_b in enumerate([mape_2_train * 100, mape_2_test * 100, mape_2_full * 100]):
    axes_metr2[1].text(idx_b, val_b + 0.02, f"{val_b:.3f}%", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("prophet_confronto_metriche_mod2.png", dpi=150, bbox_inches="tight")
plt.close()

# Focus test: confronto modelli
focus_min = min(np.min(y_true_1_test), np.min(y_pred_1_test), np.min(y_pred_2_test))
focus_max = max(np.max(y_true_1_test), np.max(y_pred_1_test), np.max(y_pred_2_test))
focus_range = focus_max - focus_min
padding = focus_range * 0.15

plt.figure(figsize=(10, 5))
plt.plot(serie_ma_test_aligned.index, y_true_1_test, marker="o", linewidth=2, label="Test (MA5)", color="tab:blue")
plt.plot(serie_ma_test_aligned.index, y_pred_1_test, marker="x", linestyle="--", linewidth=2, label="Modello 1 (Mom 1,3,5,10)", color="tab:orange")
plt.plot(serie_ma_test_aligned.index, y_pred_2_test, marker="s", linestyle="--", linewidth=2, label="Modello 2 (Mom 3,10)", color="tab:green")
plt.title("Focus Test: Confronto Modelli Prophet")
plt.xlabel("Data")
plt.ylabel("Prezzo smussato")
ax_f = plt.gca()
ax_f.set_ylim(focus_min - padding, focus_max + padding)
ax_f.grid(True, alpha=0.3)
plt.legend(loc="best")
plt.text(0.02, 0.95,
    f"Modello 1: R²={r2_1_test:.3f}, MAPE={mape_1_test:.2%}\nModello 2: R²={r2_2_test:.3f}, MAPE={mape_2_test:.2%}",
    transform=plt.gca().transAxes, verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7), fontsize=9)
plt.tight_layout()
plt.savefig("prophet_focus_test_forecast.png", dpi=150, bbox_inches="tight")
plt.close()

# Errori di previsione Modello 1
error_1 = y_true_1_test - y_pred_1_test
fig_err1, (ax_e1_top, ax_e1_bot) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax_e1_top.plot(serie_ma_test_aligned.index, y_true_1_test, marker="o", linewidth=2, label="Osservato", color="tab:blue")
ax_e1_top.plot(serie_ma_test_aligned.index, y_pred_1_test, marker="s", linestyle="--", linewidth=2, label="Previsione", color="tab:orange")
ax_e1_top.set_title("Errori di previsione - Modello 1 Prophet")
ax_e1_top.set_ylabel("Prezzo smussato")
ax_e1_top.grid(True, alpha=0.3)
ax_e1_top.legend()
ax_e1_bot.bar(serie_ma_test_aligned.index, error_1, color=["lightgreen" if e >= 0 else "lightcoral" for e in error_1], edgecolor="black", linewidth=0.5)
ax_e1_bot.axhline(y=0, color="red", linestyle="--", linewidth=1)
ax_e1_bot.set_title("Errori (Actual - Forecast)")
ax_e1_bot.set_xlabel("Data")
ax_e1_bot.set_ylabel("Errore")
ax_e1_bot.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("prophet_forecast_error_mod1.png", dpi=150, bbox_inches="tight")
plt.close()

# Previsioni puntuali stampa
print("\nPrevisioni Prophet Modello 1 sui 10 punti test:")
for i in range(test_size):
    dt = serie_ma_test_aligned.index[i].strftime('%Y-%m-%d')
    pred = y_pred_1_test[i]
    actual = y_true_1_test[i]
    err = actual - pred
    err_pct = err / actual * 100
    print(f"  {dt}: Previsto={pred:.2f}, Effettivo={actual:.2f}, Errore={err:.2f} ({err_pct:+.2f}%)")

print("\nPrevisioni Prophet Modello 2 sui 10 punti test:")
for i in range(test_size):
    dt = serie_ma_test_aligned.index[i].strftime('%Y-%m-%d')
    pred = y_pred_2_test[i]
    actual = y_true_2_test[i]
    err = actual - pred
    err_pct = err / actual * 100
    print(f"  {dt}: Previsto={pred:.2f}, Effettivo={actual:.2f}, Errore={err:.2f} ({err_pct:+.2f}%)")

# --------------------------
# Visualizzazione risultati
# --------------------------
print("\n📈 Creazione visualizzazione finale...")

# Grafico finale: Training, Test e Full Dataset
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14))

# Subplot 1: Training
ax1.plot(df_train_1['ds'], best_y_true_train, 'o-', label='Valori reali', linewidth=1.5, markersize=4, alpha=0.7)
ax1.plot(df_train_1['ds'], best_y_pred_train, '-', label='Previsioni Prophet', linewidth=1.5, alpha=0.8)
ax1.set_title(f'Training Set - {best_name}', fontsize=12, fontweight='bold')
ax1.set_ylabel('Prezzo di chiusura (media mobile 5)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.text(0.02, 0.98, f'R²: {best_r2_train:.4f}\nMAPE: {best_mape_train*100:.2f}%', 
         transform=ax1.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
         fontsize=10, fontweight='bold')

# Subplot 2: Test
ax2.plot(future_1['ds'], best_y_true_test, 'o-', label='Valori reali', linewidth=2, markersize=8)
ax2.plot(future_1['ds'], best_y_pred_test, 's-', label='Previsioni Prophet', linewidth=2, markersize=8)

# Zoom sulla scala per test
y_min = min(np.min(best_y_true_test), np.min(best_y_pred_test))
y_max = max(np.max(best_y_true_test), np.max(best_y_pred_test))
y_range = y_max - y_min
ax2.set_ylim(y_min - 0.01 * y_range, y_max + 0.01 * y_range)

ax2.set_title(f'Test Set - {best_name}', fontsize=12, fontweight='bold')
ax2.set_ylabel('Prezzo di chiusura (media mobile 5)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.text(0.02, 0.98, f'R²: {best_r2_test:.4f}\nMAPE: {best_mape_test*100:.2f}%', 
         transform=ax2.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
         fontsize=10, fontweight='bold')

# Subplot 3: Full Dataset
ax3.plot(df_full_1['ds'], best_y_true_full, 'o-', label='Valori reali', linewidth=1.2, markersize=3, alpha=0.6)
ax3.plot(df_full_1['ds'], best_y_pred_full, '-', label='Previsioni Prophet', linewidth=1.2, alpha=0.7)
ax3.set_title(f'Full Dataset - {best_name}', fontsize=12, fontweight='bold')
ax3.set_xlabel('Data')
ax3.set_ylabel('Prezzo di chiusura (media mobile 5)')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.text(0.02, 0.98, f'R²: {best_r2_full:.4f}\nMAPE: {best_mape_full*100:.2f}%', 
         transform=ax3.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
         fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig("prophet_final_best_model.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Grafico finale salvato in: prophet_final_best_model.png")

print("\n" + "="*70)
print("✅ COMPLETATO")
print("="*70)
