from typing import cast

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from scipy import stats
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# --------------------------
# Caricamento e preprocessing
# --------------------------
dataset = pd.read_excel("FTSE_MIB_Futures_con_stagionalita.xlsx")
dataset["date"] = pd.to_datetime(dataset["date"])
dataset.set_index("date", inplace=True)
dataset.sort_index(inplace=True)
dataset.index._data._freq = BDay()  # Giorni lavorativi (Business days)

serie = dataset["closed"].copy()
serie = serie.replace([np.inf, -np.inf], np.nan).dropna()
serie_ma = serie.rolling(window=5, min_periods=5).mean().dropna()

stat_originale = serie.describe()
stat_smussata = serie_ma.describe()

print("Statistiche serie originale:")
print(stat_originale)
print("\nStatistiche serie smussata:")
print(stat_smussata)

# Grafico della serie originale
plt.figure(figsize=(10, 4))
plt.plot(serie.index, serie.values, linewidth=1.2, color="tab:blue")
plt.title("Serie storica FTSE MIB Futures - Prezzi di chiusura")
plt.xlabel("Data")
plt.ylabel("Prezzo di chiusura")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("serie_originale.png", dpi=150, bbox_inches="tight")
plt.close()

# Grafico serie originale vs media mobile
plt.figure(figsize=(10, 4))
plt.plot(serie.index, serie.values, label="Serie originale", linewidth=1.2)
plt.plot(serie_ma.index, serie_ma.values, label="Media mobile (5)", linewidth=2)
plt.title("Serie originale vs Media mobile (5)")
plt.xlabel("Data")
plt.ylabel("Prezzo di chiusura")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("serie_vs_media_mobile5.png", dpi=150, bbox_inches="tight")
plt.close()

# -------------------------------
# Analisi ACF/PACF della serie differenziata
# -------------------------------
# I modelli ARIMA utilizzano d=1 (differenziazione non stagionale), senza stagionalità.
# Analizziamo la serie differenziata per identificare i pattern di autocorrelazione.

# Differenziazione non stagionale (d=1) e rimozione NaN
serie_ma_diff_completa = serie_ma.diff().dropna()

nlags = 40

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

plot_acf(serie_ma_diff_completa, lags=nlags, ax=ax1, alpha=0.05, title='Autocorrelation Function (ACF) - Serie smussata (MA5) differenziata (d=1)')
ax1.grid(True, alpha=0.3)

plot_pacf(serie_ma_diff_completa, lags=nlags, ax=ax2, alpha=0.05, title='Partial Autocorrelation Function (PACF) - Serie smussata (MA5) differenziata (d=1)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("acf_pacf_serie_ma5.png", dpi=150, bbox_inches="tight")
plt.close()

# Calcolo numerico ACF/PACF
# Quando alpha è specificato, acf/pacf restituiscono (values, confint).
# cast() dichiara il tipo atteso al type-checker (Pylance/Pyright).
acf_values, acf_confint = cast(
    tuple[np.ndarray, np.ndarray],
    acf(serie_ma_diff_completa, nlags=nlags, alpha=0.05),
)
pacf_values, pacf_confint = cast(
    tuple[np.ndarray, np.ndarray],
    pacf(serie_ma_diff_completa, nlags=nlags, alpha=0.05),
)
standard_bound = 1.96 / np.sqrt(len(serie_ma_diff_completa))
acf_conf_bound = np.abs(acf_confint[:, 1] - acf_values) 
pacf_conf_bound = np.abs(pacf_confint[:, 1] - pacf_values) 

print("\n" + "="*80)
print("ACF/PACF - VALORI NUMERICI (Lag 0-10)")
print("="*80)
print(f"{'Lag':<6} {'ACF':<12} {'Sign.':<8} {'PACF':<12} {'Sign.':<8}")
print("-"*80)
for lag in range(11):
    acf_val = acf_values[lag]
    pacf_val = pacf_values[lag]
    acf_bound = acf_conf_bound[lag] 
    pacf_bound = pacf_conf_bound[lag] 
    acf_sig = "SÌ" if abs(acf_val) > acf_bound else "NO"
    pacf_sig = "SÌ" if abs(pacf_val) > pacf_bound else "NO"
    print(f"{lag:<6} {acf_val:>10.4f}  {acf_sig:<8} {pacf_val:>10.4f}  {pacf_sig:<8}")
print("="*80)
print(f"Bound confidenza (95%):")
print(f"  Standard: ±{standard_bound:.4f}")
acf_bound_mean = np.mean(acf_conf_bound[:11]) 
pacf_bound_mean = np.mean(pacf_conf_bound[:11]) 
print(f"  ACF (media lag 0-10): ±{acf_bound_mean:.4f}")
print(f"  PACF (media lag 0-10): ±{pacf_bound_mean:.4f}\n")

# Configurazione del back-test: training lungo tutta la serie tranne gli ultimi 10 punti,
# che vengono usati come verifica out-of-sample.
train_size = 1262
test_size = 10

serie_ma_train = serie_ma.iloc[:train_size]
serie_ma_test = serie_ma.iloc[train_size:train_size + test_size]

focus_start_date = pd.Timestamp("2024-08-30")

print(f"Osservazioni train: {len(serie_ma_train)}")
print(f"Osservazioni test: {len(serie_ma_test)}")

# Addestriamo il modello ARIMA migliore individuato ([1,3],1,0) con drift lineare
# ------------------------
# Modello 1: ARIMA([1,3],1,0) con drift lineare (trend=[0,1,0,0])
# ------------------------
model = SARIMAX(
    serie_ma_train,
    order=([1,3], 1, 0),
    seasonal_order=(0, 0, 0, 0),
    trend=[0,1,0,0],
    measurement_error=False,
    enforce_stationarity=True,
    enforce_invertibility=True,
)
# Usa cov_type='robust_approx' per metodo standard (Hessian) invece di OPG
# Questo elimina il warning e dà standard errors più accurati
results = model.fit(disp=False, cov_type='robust_approx')
print(results.summary())

# Addestriamo il secondo modello ARIMA per confronto
# ------------------------
# Modello 2: ARIMA(0,1,4) con drift lineare (trend=[0,1,0,0])
# Modello MA(4) per confronto con il modello AR selettivo
# ------------------------
model_2 = SARIMAX(
    serie_ma_train,
    order=(0,1,4),
    seasonal_order=(0, 0, 0, 0),
    trend=[0,1,0,0],
    enforce_stationarity=True,
    enforce_invertibility=True,
)
# Usa cov_type='robust_approx' per metodo standard (Hessian) invece di OPG
# Questo elimina il warning e dà standard errors più accurati
results_2 = model_2.fit(disp=False, cov_type='robust_approx')
print(results_2.summary())

# Analisi dei residui del modello su tutto il train
residui_train = results.resid.dropna()
print("\nAnalisi residui Modello 1 (tutta la serie di training):")
print(residui_train.describe())
if not residui_train.empty:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(residui_train.index, residui_train.values, linewidth=0.7, color="tab:blue")
    axes[0, 0].axhline(y=0, color="red", linestyle="--", linewidth=1)
    axes[0, 0].set_title("Residui nel Tempo - Modello 1")
    axes[0, 0].set_xlabel("Data")
    axes[0, 0].set_ylabel("Residuo")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(residui_train, bins=40, density=True, alpha=0.7, color="tab:blue", edgecolor="black")
    mu, sigma = residui_train.mean(), residui_train.std()
    x_vals = np.linspace(residui_train.min(), residui_train.max(), 200)
    if sigma > 0:
        axes[0, 1].plot(x_vals, stats.norm.pdf(x_vals, mu, sigma), "r-", linewidth=2, label="Normale teorica")
    axes[0, 1].set_title("Distribuzione Residui")
    axes[0, 1].set_xlabel("Residuo")
    axes[0, 1].set_ylabel("Densità")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    stats.probplot(residui_train, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot Residui")
    axes[1, 0].grid(True, alpha=0.3)

    plot_acf(residui_train, lags=min(40, len(residui_train) // 4), ax=axes[1, 1], alpha=0.05)
    axes[1, 1].set_title("ACF Residui")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("residual_analysis_modello1.png", dpi=150, bbox_inches="tight")
    plt.close()

# Analisi dei residui del secondo modello
residui_train_2 = results_2.resid.dropna()
print("\nAnalisi residui Modello 2 (tutta la serie di training):")
print(residui_train_2.describe())
if not residui_train_2.empty:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(residui_train_2.index, residui_train_2.values, linewidth=0.7, color="tab:blue")
    axes[0, 0].axhline(y=0, color="red", linestyle="--", linewidth=1)
    axes[0, 0].set_title("Residui nel Tempo - Modello 2")
    axes[0, 0].set_xlabel("Data")
    axes[0, 0].set_ylabel("Residuo")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(residui_train_2, bins=40, density=True, alpha=0.7, color="tab:blue", edgecolor="black")
    mu2, sigma2 = residui_train_2.mean(), residui_train_2.std()
    x_vals2 = np.linspace(residui_train_2.min(), residui_train_2.max(), 200)
    if sigma2 > 0:
        axes[0, 1].plot(x_vals2, stats.norm.pdf(x_vals2, mu2, sigma2), "r-", linewidth=2, label="Normale teorica")
    axes[0, 1].set_title("Distribuzione Residui")
    axes[0, 1].set_xlabel("Residuo")
    axes[0, 1].set_ylabel("Densità")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    stats.probplot(residui_train_2, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot Residui")
    axes[1, 0].grid(True, alpha=0.3)

    plot_acf(residui_train_2, lags=min(40, len(residui_train_2) // 4), ax=axes[1, 1], alpha=0.05)
    axes[1, 1].set_title("ACF Residui")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("residual_analysis_modello2.png", dpi=150, bbox_inches="tight")
    plt.close()


# Previsioni sul test set - Modello 1
forecast_res = results.get_forecast(steps=test_size)
forecast = forecast_res.predicted_mean
conf_int_df = forecast_res.conf_int()

# Allineiamo gli indici con le date effettive del test set
forecast.index = serie_ma_test.index
conf_int_df.index = serie_ma_test.index
conf_int_df.columns = ["lower_ci", "upper_ci"]

print("\nPrevisioni ARIMA Modello 1 (ARIMA([1,3],1,0) con drift lineare) sui prossimi 10 punti:")
print(
    pd.concat(
        [forecast, conf_int_df, serie_ma_test.rename("actual")],
        axis=1,
    )
)

r2_test = r2_score(serie_ma_test, forecast)
mape_test = mean_absolute_percentage_error(serie_ma_test, forecast)

# Metriche training calcolate sull'intero train
fitted_train = results.fittedvalues.loc[serie_ma_train.index]
r2_train_metric = r2_score(serie_ma_train, fitted_train)
mape_train_metric = mean_absolute_percentage_error(serie_ma_train, fitted_train)
mae_train_metric = np.mean(np.abs(serie_ma_train - fitted_train))
rmse_train_metric = np.sqrt(np.mean((serie_ma_train - fitted_train) ** 2))

# Previsioni su tutto il dataset - Modello 1
forecast_full_1 = pd.concat([fitted_train, forecast])
serie_ma_full = pd.concat([serie_ma_train, serie_ma_test])
r2_full_1 = r2_score(serie_ma_full, forecast_full_1)
mape_full_1 = mean_absolute_percentage_error(serie_ma_full, forecast_full_1)
mae_full_1 = np.mean(np.abs(serie_ma_full - forecast_full_1))
rmse_full_1 = np.sqrt(np.mean((serie_ma_full - forecast_full_1) ** 2))

mae_test = np.mean(np.abs(serie_ma_test - forecast))
rmse_test = np.sqrt(np.mean((serie_ma_test - forecast) ** 2))

print(f"\nMetriche Modello 1:")
print(f"R² Train: {r2_train_metric:.4f} ({r2_train_metric*100:.2f}%)")
print(f"R² Test: {r2_test:.4f} ({r2_test*100:.2f}%)")
print(f"R² Full: {r2_full_1:.4f} ({r2_full_1*100:.2f}%)")
print(f"MAPE Train: {mape_train_metric:.6f} ({mape_train_metric*100:.4f}%)")
print(f"MAPE Test: {mape_test:.6f} ({mape_test*100:.4f}%)")
print(f"MAPE Full: {mape_full_1:.6f} ({mape_full_1*100:.4f}%)")
print(f"MAE Train: {mae_train_metric:.4f} | MAE Test: {mae_test:.4f} | MAE Full: {mae_full_1:.4f}")
print(f"RMSE Train: {rmse_train_metric:.4f} | RMSE Test: {rmse_test:.4f} | RMSE Full: {rmse_full_1:.4f}")

# Grafico riassuntivo delle metriche (R² e MAPE) train vs test vs full.
fig_metrics, axes_metrics = plt.subplots(1, 2, figsize=(12, 4))
axes_metrics[0].bar(["Train", "Test", "Full"], [r2_train_metric * 100, r2_test * 100, r2_full_1 * 100], color=["tab:blue", "tab:orange", "tab:green"])
axes_metrics[0].set_title("Confronto R² Train vs Test vs Full - Modello 1")
axes_metrics[0].set_ylabel("R² (%)")
axes_metrics[0].grid(True, axis="y", alpha=0.3)
for idx, val in enumerate([r2_train_metric * 100, r2_test * 100, r2_full_1 * 100]):
    axes_metrics[0].text(idx, val + 1, f"{val:.2f}%", ha="center", fontweight="bold")

axes_metrics[1].bar(["Train", "Test", "Full"], [mape_train_metric * 100, mape_test * 100, mape_full_1 * 100], color=["tab:blue", "tab:orange", "tab:green"])
axes_metrics[1].set_title("Confronto MAPE Train vs Test vs Full - Modello 1")
axes_metrics[1].set_ylabel("MAPE (%)")
axes_metrics[1].grid(True, axis="y", alpha=0.3)
for idx, val in enumerate([mape_train_metric * 100, mape_test * 100, mape_full_1 * 100]):
    axes_metrics[1].text(idx, val + 0.01, f"{val:.3f}%", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("confronto_metriche_train_test.png", dpi=150, bbox_inches="tight")
plt.close()

# Grafico di focus: coda del training (29/08/2024)  + test con osservato vs stimato/previsione
train_focus_mod1 = serie_ma_train.loc[serie_ma_train.index >= focus_start_date]
if train_focus_mod1.empty:
    train_focus_mod1 = serie_ma_train.iloc[-80:]
observed_focus_mod1 = pd.concat([train_focus_mod1, serie_ma_test])
estimated_focus_mod1 = pd.concat(
    [fitted_train.loc[train_focus_mod1.index], forecast]
)

plt.figure(figsize=(10, 5))
plt.plot(
    observed_focus_mod1.index,
    observed_focus_mod1.values,
    marker="o",
    linewidth=1.8,
    color="tab:blue",
    label="Osservato (train+test)",
)
plt.plot(
    estimated_focus_mod1.index,
    estimated_focus_mod1.values,
    marker="s",
    linestyle="--",
    linewidth=1.8,
    color="tab:orange",
    label="Stimato/Previsto",
)
plt.axvline(
    x=serie_ma_test.index[0],
    color="gray",
    linestyle=":",
    linewidth=1,
    label="Inizio test",
)
plt.title("Focus coda training + test - Modello 1")
plt.xlabel("Data")
plt.ylabel("Prezzo smussato")
plt.grid(True, alpha=0.3)
metrics_box_mod1 = (
    f"Train: R²={r2_train_metric:.3f}, MAPE={mape_train_metric:.3%}\n"
    f"Test:  R²={r2_test:.3f}, MAPE={mape_test:.3%}"
)
plt.text(
    0.02,
    0.95,
    metrics_box_mod1,
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.4"),
    fontsize=9,
)
plt.legend(loc="best")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("focus_training_test_modello1.png", dpi=150, bbox_inches="tight")
plt.close()

error_mod1 = (serie_ma_test - forecast).copy()
fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Funzioni di supporto per colorare aree positive/negative tra due curve.
def _fill_segment(ax, x0, x1, upper0, upper1, lower0, lower1, label_state):
    if x1 == x0:
        return
    err0 = upper0 - lower0
    err1 = upper1 - lower1
    if err0 == 0 and err1 == 0:
        return
    if np.sign(err0) == np.sign(err1) or err0 == 0 or err1 == 0:
        sign = err1 if err1 != 0 else err0
        color = "lightgreen" if sign >= 0 else "lightcoral"
        label = None
        if color == "lightgreen" and not label_state["pos"]:
            label = "Errore positivo (Actual > Forecast)"
            label_state["pos"] = True
        elif color == "lightcoral" and not label_state["neg"]:
            label = "Errore negativo (Actual < Forecast)"
            label_state["neg"] = True
        ax.fill_between([x0, x1], [upper0, upper1], [lower0, lower1], color=color, alpha=0.3, label=label)
    else:
        x_mid = x0 + (x1 - x0) / 2
        upper_mid = (upper0 + upper1) / 2
        lower_mid = (lower0 + lower1) / 2
        _fill_segment(ax, x0, x_mid, upper0, upper_mid, lower0, lower_mid, label_state)
        _fill_segment(ax, x_mid, x1, upper_mid, upper1, lower_mid, lower1, label_state)

def fill_signed_area(ax, x_values, upper_values, lower_values):
    label_state = {"pos": False, "neg": False}
    for i in range(len(x_values) - 1):
        _fill_segment(
            ax,
            x_values[i],
            x_values[i + 1],
            upper_values[i],
            upper_values[i + 1],
            lower_values[i],
            lower_values[i + 1],
            label_state,
        )
    return label_state

ax_top.plot(serie_ma_test.index, serie_ma_test.values, marker="o", linewidth=2, label="Osservato (Test)", color="tab:blue")
ax_top.plot(forecast.index, forecast.values, marker="s", linestyle="--", linewidth=2, label="Previsione", color="tab:orange")
fill_signed_area(ax_top, serie_ma_test.index, serie_ma_test.values, forecast.values)
ax_top.set_title("Errori di previsione - Modello 1 (Osservato vs Previsto)")
ax_top.set_ylabel("Prezzo smussato")
ax_top.grid(True, alpha=0.3)
ax_top.legend(loc="best")

ax_bottom.plot(error_mod1.index, error_mod1.values, marker="o", color="tab:purple", linewidth=1.8)
ax_bottom.axhline(y=0, color="red", linestyle="--", linewidth=1)
fill_signed_area(ax_bottom, error_mod1.index, error_mod1.values, np.zeros(len(error_mod1)))
ax_bottom.set_title("Serie degli errori (Actual - Forecast)")
ax_bottom.set_xlabel("Data")
ax_bottom.set_ylabel("Errore")
ax_bottom.grid(True, alpha=0.3)
ax_bottom.legend(loc="best")

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("forecast_error_modello1.png", dpi=150, bbox_inches="tight")
plt.close()

# Previsioni sul test set - Modello 2
forecast_res_2 = results_2.get_forecast(steps=test_size)
forecast_2 = forecast_res_2.predicted_mean
conf_int_df_2 = forecast_res_2.conf_int()

# Allineiamo gli indici con le date effettive del test set
forecast_2.index = serie_ma_test.index
conf_int_df_2.index = serie_ma_test.index
conf_int_df_2.columns = ["lower_ci", "upper_ci"]

print("\nPrevisioni ARIMA Modello 2 (ARIMA(0,1,4) con drift lineare) sui prossimi 10 punti:")
print(
    pd.concat(
        [forecast_2, conf_int_df_2, serie_ma_test.rename("actual")],
        axis=1,
    )
)

r2_test_2 = r2_score(serie_ma_test, forecast_2)
mape_test_2 = mean_absolute_percentage_error(serie_ma_test, forecast_2)
error_mod2 = (serie_ma_test - forecast_2).copy()
fitted_train2 = results_2.fittedvalues.loc[serie_ma_train.index]
r2_train_metric2 = r2_score(serie_ma_train, fitted_train2)
mape_train_metric2 = mean_absolute_percentage_error(serie_ma_train, fitted_train2)
mae_train_metric2 = np.mean(np.abs(serie_ma_train - fitted_train2))
rmse_train_metric2 = np.sqrt(np.mean((serie_ma_train - fitted_train2) ** 2))

# Previsioni su tutto il dataset - Modello 2
forecast_full_2 = pd.concat([fitted_train2, forecast_2])
r2_full_2 = r2_score(serie_ma_full, forecast_full_2)
mape_full_2 = mean_absolute_percentage_error(serie_ma_full, forecast_full_2)
mae_full_2 = np.mean(np.abs(serie_ma_full - forecast_full_2))
rmse_full_2 = np.sqrt(np.mean((serie_ma_full - forecast_full_2) ** 2))

mae_test_2 = np.mean(np.abs(serie_ma_test - forecast_2))
rmse_test_2 = np.sqrt(np.mean((serie_ma_test - forecast_2) ** 2))

print(f"\nMetriche Modello 2:")
print(f"R² Train: {r2_train_metric2:.4f} ({r2_train_metric2*100:.2f}%)")
print(f"R² Test: {r2_test_2:.4f} ({r2_test_2*100:.2f}%)")
print(f"R² Full: {r2_full_2:.4f} ({r2_full_2*100:.2f}%)")
print(f"MAPE Train: {mape_train_metric2:.6f} ({mape_train_metric2*100:.4f}%)")
print(f"MAPE Test: {mape_test_2:.6f} ({mape_test_2*100:.4f}%)")
print(f"MAPE Full: {mape_full_2:.6f} ({mape_full_2*100:.4f}%)")
print(f"MAE Train: {mae_train_metric2:.4f} | MAE Test: {mae_test_2:.4f} | MAE Full: {mae_full_2:.4f}")
print(f"RMSE Train: {rmse_train_metric2:.4f} | RMSE Test: {rmse_test_2:.4f} | RMSE Full: {rmse_full_2:.4f}")

# Grafico delle metriche train vs test vs full anche per il secondo modello.
fig_metrics2, axes_metrics2 = plt.subplots(1, 2, figsize=(12, 4))
axes_metrics2[0].bar(["Train", "Test", "Full"], [r2_train_metric2 * 100, r2_test_2 * 100, r2_full_2 * 100], color=["tab:blue", "tab:orange", "tab:green"])
axes_metrics2[0].set_title("Confronto R² Train vs Test vs Full - Modello 2")
axes_metrics2[0].set_ylabel("R² (%)")
axes_metrics2[0].grid(True, axis="y", alpha=0.3)
for idx, val in enumerate([r2_train_metric2 * 100, r2_test_2 * 100, r2_full_2 * 100]):
    axes_metrics2[0].text(idx, val + 1, f"{val:.2f}%", ha="center", fontweight="bold")

axes_metrics2[1].bar(["Train", "Test", "Full"], [mape_train_metric2 * 100, mape_test_2 * 100, mape_full_2 * 100], color=["tab:blue", "tab:orange", "tab:green"])
axes_metrics2[1].set_title("Confronto MAPE Train vs Test vs Full - Modello 2")
axes_metrics2[1].set_ylabel("MAPE (%)")
axes_metrics2[1].grid(True, axis="y", alpha=0.3)
for idx, val in enumerate([mape_train_metric2 * 100, mape_test_2 * 100, mape_full_2 * 100]):
    axes_metrics2[1].text(idx, val + 0.01, f"{val:.3f}%", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("confronto_metriche_train_test_mod2.png", dpi=150, bbox_inches="tight")
plt.close()

# Grafico di focus: coda del training (da 2024-09-15) + test per il secondo modello
train_focus_mod2 = serie_ma_train.loc[serie_ma_train.index >= focus_start_date]
if train_focus_mod2.empty:
    train_focus_mod2 = serie_ma_train.iloc[-80:]
observed_focus_mod2 = pd.concat([train_focus_mod2, serie_ma_test])
estimated_focus_mod2 = pd.concat(
    [fitted_train2.loc[train_focus_mod2.index], forecast_2]
)

plt.figure(figsize=(10, 5))
plt.plot(
    observed_focus_mod2.index,
    observed_focus_mod2.values,
    marker="o",
    linewidth=1.8,
    color="tab:blue",
    label="Osservato (train+test)",
)
plt.plot(
    estimated_focus_mod2.index,
    estimated_focus_mod2.values,
    marker="s",
    linestyle="--",
    linewidth=1.8,
    color="tab:green",
    label="Stimato/Previsto",
)
plt.axvline(
    x=serie_ma_test.index[0],
    color="gray",
    linestyle=":",
    linewidth=1,
    label="Inizio test",
)
plt.title("Focus coda training + test - Modello 2")
plt.xlabel("Data")
plt.ylabel("Prezzo smussato")
plt.grid(True, alpha=0.3)
metrics_box_mod2 = (
    f"Train: R²={r2_train_metric2:.3f}, MAPE={mape_train_metric2:.3%}\n"
    f"Test:  R²={r2_test_2:.3f}, MAPE={mape_test_2:.3%}"
)
plt.text(
    0.02,
    0.95,
    metrics_box_mod2,
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.4"),
    fontsize=9,
)
plt.legend(loc="best")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("focus_training_test_modello2.png", dpi=150, bbox_inches="tight")
plt.close()

fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

ax_top.plot(serie_ma_test.index, serie_ma_test.values, marker="o", linewidth=2, label="Osservato (Test)", color="tab:blue")
ax_top.plot(forecast_2.index, forecast_2.values, marker="s", linestyle="--", linewidth=2, label="Previsione", color="tab:orange")
fill_signed_area(ax_top, serie_ma_test.index, serie_ma_test.values, forecast_2.values)
ax_top.set_title("Errori di previsione - Modello 2 (Osservato vs Previsto)")
ax_top.set_ylabel("Prezzo smussato")
ax_top.grid(True, alpha=0.3)
ax_top.legend(loc="best")

ax_bottom.plot(error_mod2.index, error_mod2.values, marker="o", color="tab:purple", linewidth=1.8)
ax_bottom.axhline(y=0, color="red", linestyle="--", linewidth=1)
fill_signed_area(ax_bottom, error_mod2.index, error_mod2.values, np.zeros(len(error_mod2)))
ax_bottom.set_title("Serie degli errori (Actual - Forecast)")
ax_bottom.set_xlabel("Data")
ax_bottom.set_ylabel("Errore")
ax_bottom.grid(True, alpha=0.3)
ax_bottom.legend(loc="best")

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("forecast_error_modello2.png", dpi=150, bbox_inches="tight")
plt.close()

# Grafico di focus sulle ultime 5 osservazioni (test vs forecast)
focus_min = min(serie_ma_test.min(), forecast.min(), forecast_2.min())
focus_max = max(serie_ma_test.max(), forecast.max(), forecast_2.max())
focus_range = focus_max - focus_min
# Imposta limiti con piccolo padding per vedere meglio le variazioni
padding = focus_range * 0.1  # 10% di padding per vedere meglio le variazioni

plt.figure(figsize=(10, 5))
plt.plot(serie_ma_test.index, serie_ma_test.values, marker="o", linewidth=2, label="Test (MA5)", color="tab:blue")
plt.plot(forecast.index, forecast.values, marker="x", linestyle="--", linewidth=2, label="Forecast ARIMA([1,3],1,0) drift lineare", color="tab:orange")
plt.plot(forecast_2.index, forecast_2.values, marker="s", linestyle="--", linewidth=2, label="Forecast ARIMA(0,1,4) drift lineare", color="tab:green")
plt.title("Focus ultime 10 osservazioni: Test vs Forecast")
plt.xlabel("Data")
plt.ylabel("Prezzo smussato")
ax_focus = plt.gca()
ax_focus.grid(True, alpha=0.3)
# Imposta limiti con padding per vedere meglio le variazioni
ax_focus.set_ylim(focus_min - padding, focus_max + padding)
plt.legend(loc="best")
plt.text(
    0.02,
    0.95,
    f"Modello 1: R²={r2_test:.3f}, MAPE={mape_test:.2%}\nModello 2: R²={r2_test_2:.3f}, MAPE={mape_test_2:.2%}",
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    fontsize=9,
)
plt.tight_layout()
plt.savefig("focus_test_forecast.png", dpi=150, bbox_inches="tight")
plt.close()

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

r2_1_train_str = f"{r2_train_metric:.4f}".ljust(16)
r2_1_test_str = f"{r2_test:.4f}".ljust(16)
r2_1_full_str = f"{r2_full_1:.4f}".ljust(16)
r2_2_train_str = f"{r2_train_metric2:.4f}".ljust(16)
r2_2_test_str = f"{r2_test_2:.4f}".ljust(16)
r2_2_full_str = f"{r2_full_2:.4f}".ljust(16)

mape_1_train_str = f"{mape_train_metric*100:.2f}".ljust(16)
mape_1_test_str = f"{mape_test*100:.2f}".ljust(16)
mape_1_full_str = f"{mape_full_1*100:.2f}".ljust(16)
mape_2_train_str = f"{mape_train_metric2*100:.2f}".ljust(16)
mape_2_test_str = f"{mape_test_2*100:.2f}".ljust(16)
mape_2_full_str = f"{mape_full_2*100:.2f}".ljust(16)

mae_1_train_str = f"{mae_train_metric:.4f}".ljust(16)
mae_1_test_str = f"{mae_test:.4f}".ljust(16)
mae_1_full_str = f"{mae_full_1:.4f}".ljust(16)
mae_2_train_str = f"{mae_train_metric2:.4f}".ljust(16)
mae_2_test_str = f"{mae_test_2:.4f}".ljust(16)
mae_2_full_str = f"{mae_full_2:.4f}".ljust(16)

rmse_1_train_str = f"{rmse_train_metric:.4f}".ljust(16)
rmse_1_test_str = f"{rmse_test:.4f}".ljust(16)
rmse_1_full_str = f"{rmse_full_1:.4f}".ljust(16)
rmse_2_train_str = f"{rmse_train_metric2:.4f}".ljust(16)
rmse_2_test_str = f"{rmse_test_2:.4f}".ljust(16)
rmse_2_full_str = f"{rmse_full_2:.4f}".ljust(16)

print(f"{'R² Score':<25} {r2_1_train_str} {r2_1_test_str} {r2_1_full_str} {r2_2_train_str} {r2_2_test_str} {r2_2_full_str}")
print(f"{'MAPE (%)':<25} {mape_1_train_str} {mape_1_test_str} {mape_1_full_str} {mape_2_train_str} {mape_2_test_str} {mape_2_full_str}")
print(f"{'MAE':<25} {mae_1_train_str} {mae_1_test_str} {mae_1_full_str} {mae_2_train_str} {mae_2_test_str} {mae_2_full_str}")
print(f"{'RMSE':<25} {rmse_1_train_str} {rmse_1_test_str} {rmse_1_full_str} {rmse_2_train_str} {rmse_2_test_str} {rmse_2_full_str}")

print("\n" + "="*70)
print("🏆 MIGLIOR MODELLO")
print("="*70)
if r2_test > r2_test_2:
    print("Modello 1: ARIMA([1,3],1,0) con drift lineare (trend=[0,1,0,0]) - MIGLIORE")
    print(f"  R² Test: {r2_test:.4f}")
    print(f"  MAPE Test: {mape_test*100:.2f}%")
else:
    print("Modello 2: ARIMA(0,1,4) con drift lineare (trend=[0,1,0,0])")
    print(f"  R² Test: {r2_test_2:.4f}")
    print(f"  MAPE Test: {mape_test_2*100:.2f}%")
print("="*70)

# --------------------------
# Visualizzazione previsioni su tutto il dataset
# --------------------------
print("\n📈 Creazione visualizzazione previsioni su tutto il dataset...")

# Grafico finale: Full Dataset per entrambi i modelli
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Subplot 1: Modello 1 - Full Dataset
ax1.plot(serie_ma_full.index, serie_ma_full.values, 'o-', label='Valori reali', linewidth=1.2, markersize=3, alpha=0.6)
ax1.plot(forecast_full_1.index, forecast_full_1.values, '-', label='Previsioni ARIMA', linewidth=1.2, alpha=0.7)
ax1.axvline(x=serie_ma_test.index[0], color='gray', linestyle=':', linewidth=1, label='Inizio test')
ax1.set_title('Full Dataset - Modello 1: ARIMA([1,3],1,0) con drift lineare (trend=[0,1,0,0])', fontsize=12, fontweight='bold')
ax1.set_ylabel('Prezzo di chiusura (media mobile 5)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.text(0.02, 0.98, f'R²: {r2_full_1:.4f}\nMAPE: {mape_full_1*100:.2f}%', 
         transform=ax1.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
         fontsize=10, fontweight='bold')

# Subplot 2: Modello 2 - Full Dataset
ax2.plot(serie_ma_full.index, serie_ma_full.values, 'o-', label='Valori reali', linewidth=1.2, markersize=3, alpha=0.6)
ax2.plot(forecast_full_2.index, forecast_full_2.values, '-', label='Previsioni ARIMA', linewidth=1.2, alpha=0.7)
ax2.axvline(x=serie_ma_test.index[0], color='gray', linestyle=':', linewidth=1, label='Inizio test')
ax2.set_title('Full Dataset - Modello 2: ARIMA(0,1,4) con drift lineare (trend=[0,1,0,0])', fontsize=12, fontweight='bold')
ax2.set_xlabel('Data')
ax2.set_ylabel('Prezzo di chiusura (media mobile 5)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.text(0.02, 0.98, f'R²: {r2_full_2:.4f}\nMAPE: {mape_full_2*100:.2f}%', 
         transform=ax2.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
         fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig("arima_full_dataset_predictions.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Grafico previsioni su tutto il dataset salvato in: arima_full_dataset_predictions.png")

print("\n" + "="*70)
print("✅ COMPLETATO")
print("="*70)
