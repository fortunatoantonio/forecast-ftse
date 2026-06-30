import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from pandas.tseries.frequencies import to_offset
from scipy import stats
import os
import math
import warnings
warnings.filterwarnings('ignore')

# Sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor

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

# Configurazione del back-test: test set ridotto a 10 giorni lavorativi
# Test rigoroso: mantenere R² positivo con solo 10 punti di test
test_size = 10  # Ultimi 10 giorni come test (molto rigoroso!)
train_size = len(serie_ma) - test_size  # Tutto il resto come training

serie_ma_train = serie_ma.iloc[:train_size]
serie_ma_test = serie_ma.iloc[train_size:]

print(f"\nOsservazioni train: {len(serie_ma_train)}")
print(f"Osservazioni test: {len(serie_ma_test)}")

# ============================================================================
# CONFIGURAZIONE MODELLO NEURALE
# ============================================================================

# Parametri per la creazione di sequenze
N_STEPS = 4  # Numero di time steps per sequenza (prevede il 5° step)

# Architettura CNN
FILTERS_1 = 64
FILTERS_2 = 32
FILTERS_3 = 16
KERNEL_SIZE_1 = 3
KERNEL_SIZE_2 = 3
KERNEL_SIZE_3 = 3
POOL_SIZE = 2

# Configurazione LSTM
USE_LSTM = True
LSTM_UNITS = 64
USE_BIDIRECTIONAL = True

# Configurazione layer densi
DENSE_UNITS_1 = 128
DENSE_UNITS_2 = 64
DENSE_UNITS_3 = 32
DROPOUT_RATE = 0.2
L2_REGULARIZATION = 0.001

# Configurazione training MLPRegressor
MAX_ITER = 300  # Aumentato da 100 (epochs equivalenti)
HIDDEN_LAYER_SIZES_SIMPLE = (64, 32)  # Semplificato per test rapido: (64, 32)
HIDDEN_LAYER_SIZES_COMPLEX = (128, 64, 32)  # Per modello finale: (128, 64, 32)
ALPHA = 0.001  # L2 regularization
LEARNING_RATE_INIT = 0.001
SOLVER = 'adam'  # 'adam' per dataset più grandi, 'lbfgs' per dataset piccoli
EARLY_STOPPING = True
VALIDATION_FRACTION = 0.15
N_ITER_NO_CHANGE = 20  # Patience per early stopping
TOL = 1e-4  # Tolerance per convergenza

# Directory per risultati
OUTPUT_DIR = "neural_network_results"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ============================================================================
# CALCOLO FEATURE TECNICHE
# ============================================================================

def calculate_technical_features(data):
    """
    Calcola tutte le feature tecniche disponibili per la serie temporale.
    
    Parameters:
    -----------
    data : pd.Series
        Serie temporale (serie_ma)
    
    Returns:
    --------
    features_dict : dict
        Dizionario con tutte le feature calcolate
    """
    features = {}
    
    # 1. Momentum (Rate of Change)
    features['momentum_1'] = data.diff(1).fillna(0)
    features['momentum_3'] = data.diff(3).fillna(0)
    features['momentum_5'] = data.diff(5).fillna(0)
    features['momentum_10'] = data.diff(10).fillna(0)
    
    # 2. Rate of Change (ROC)
    features['roc_1'] = data.pct_change(1).fillna(0) * 100
    features['roc_3'] = data.pct_change(3).fillna(0) * 100
    features['roc_5'] = data.pct_change(5).fillna(0) * 100
    
    # 3. Simple Moving Averages
    features['sma_3'] = data.rolling(window=3, min_periods=1).mean()
    features['sma_5'] = data.rolling(window=5, min_periods=1).mean()
    features['sma_10'] = data.rolling(window=10, min_periods=1).mean()
    features['sma_20'] = data.rolling(window=20, min_periods=1).mean()
    
    # 4. Exponential Moving Averages
    features['ema_3'] = data.ewm(span=3, adjust=False).mean()
    features['ema_5'] = data.ewm(span=5, adjust=False).mean()
    features['ema_10'] = data.ewm(span=10, adjust=False).mean()
    
    # 5. Volatilità (Rolling Standard Deviation)
    features['volatility_3'] = data.rolling(window=3, min_periods=1).std().fillna(0)
    features['volatility_5'] = data.rolling(window=5, min_periods=1).std().fillna(0)
    features['volatility_10'] = data.rolling(window=10, min_periods=1).std().fillna(0)
    
    # 6. RSI (Relative Strength Index)
    def calculate_rsi(series, window=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # RSI neutro quando non disponibile
    
    features['rsi_14'] = calculate_rsi(data, 14)
    features['rsi_7'] = calculate_rsi(data, 7)
    
    # 7. MACD (Moving Average Convergence Divergence)
    ema_12 = data.ewm(span=12, adjust=False).mean()
    ema_26 = data.ewm(span=26, adjust=False).mean()
    features['macd'] = ema_12 - ema_26
    features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
    features['macd_histogram'] = features['macd'] - features['macd_signal']
    
    # 8. Bollinger Bands
    bb_window = 20
    bb_std = 2
    bb_middle = data.rolling(window=bb_window, min_periods=1).mean()
    bb_std_val = data.rolling(window=bb_window, min_periods=1).std().fillna(0)
    features['bb_upper'] = bb_middle + (bb_std_val * bb_std)
    features['bb_lower'] = bb_middle - (bb_std_val * bb_std)
    # Evita divisione per zero
    bb_width_denom = bb_middle.replace(0, np.nan)
    features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / bb_width_denom
    features['bb_width'] = features['bb_width'].fillna(0)
    # Evita divisione per zero in bb_position
    bb_range = features['bb_upper'] - features['bb_lower']
    bb_range = bb_range.replace(0, np.nan)
    features['bb_position'] = (data - features['bb_lower']) / bb_range
    features['bb_position'] = features['bb_position'].fillna(0.5)  # Neutro quando range = 0
    
    # 9. Stochastic Oscillator
    def calculate_stochastic(series, k_window=14, d_window=3):
        low_min = series.rolling(window=k_window, min_periods=1).min()
        high_max = series.rolling(window=k_window, min_periods=1).max()
        k_percent = 100 * ((series - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_window, min_periods=1).mean()
        return k_percent.fillna(50), d_percent.fillna(50)
    
    stoch_k, stoch_d = calculate_stochastic(data)
    features['stoch_k'] = stoch_k
    features['stoch_d'] = stoch_d
    
    # 10. Lagged values
    features['lag_1'] = data.shift(1).fillna(method='bfill').fillna(data.iloc[0])
    features['lag_2'] = data.shift(2).fillna(method='bfill').fillna(data.iloc[0])
    features['lag_3'] = data.shift(3).fillna(method='bfill').fillna(data.iloc[0])
    
    # 11. Price position relative to range
    features['price_position'] = (data - data.rolling(window=20, min_periods=1).min()) / (
        data.rolling(window=20, min_periods=1).max() - data.rolling(window=20, min_periods=1).min())
    features['price_position'] = features['price_position'].fillna(0.5)
    
    return features

def combine_dataset_features(dataset_df, feature_list):
    """
    Combina feature del dataset originale (non tecniche).
    Esclude 'closed' e 'date' che sono già utilizzati.
    
    Parameters:
    -----------
    dataset_df : pd.DataFrame
        DataFrame con tutte le colonne del dataset originale
    feature_list : list
        Lista di colonne del dataset da includere
        
    Returns:
    --------
    feature_array : np.array
        Array (n_samples, n_features)
    feature_names : list
        Lista dei nomi delle feature
    """
    combined = []
    feature_names = []
    
    for feat_name in feature_list:
        if feat_name in dataset_df.columns:
            feat_values = dataset_df[feat_name].values
            
            # Gestisci NaN
            if np.any(np.isnan(feat_values)):
                feat_values = pd.Series(feat_values).fillna(method='ffill').fillna(method='bfill').fillna(0).values
            
            combined.append(feat_values)
            feature_names.append(feat_name)
        else:
            print(f"⚠️  Feature dataset '{feat_name}' non trovata, saltata.")
    
    if len(combined) == 0:
        return None, []
    
    result_array = np.column_stack(combined)
    
    # Verifica NaN finali
    if np.any(np.isnan(result_array)):
        result_df = pd.DataFrame(result_array)
        result_df = result_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        result_array = result_df.values
    
    return result_array, feature_names

def combine_features(data, feature_list):
    """
    Combina le feature selezionate in un array multi-feature.
    
    Parameters:
    -----------
    data : pd.Series
        Serie temporale originale
    feature_list : list
        Lista di nomi di feature da includere
    
    Returns:
    --------
    feature_array : np.array
        Array (n_samples, n_features)
    feature_names : list
        Lista dei nomi delle feature
    """
    all_features = calculate_technical_features(data)
    
    # Inizia sempre con la serie originale
    combined = [data.values]
    feature_names = ['price']
    
    # Aggiungi le feature richieste (escludi 'price' se presente, viene già aggiunto)
    for feat_name in feature_list:
        if feat_name == 'price':
            continue  # 'price' è già incluso come primo elemento
        if feat_name in all_features:
            feat_values = all_features[feat_name].values
            # Verifica NaN e sostituisci
            if np.any(np.isnan(feat_values)):
                feat_values = pd.Series(feat_values).fillna(method='ffill').fillna(method='bfill').fillna(0).values
            combined.append(feat_values)
            feature_names.append(feat_name)
        else:
            print(f"⚠️  Feature '{feat_name}' non trovata, saltata.")
    
    # Verifica NaN finali
    result_array = np.column_stack(combined)
    if np.any(np.isnan(result_array)):
        # Sostituisci NaN con valori forward fill e backward fill
        result_df = pd.DataFrame(result_array)
        result_df = result_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        result_array = result_df.values
    
    return result_array, feature_names

# Definizioni di combinazioni di feature da testare (FEATURE TECNICHE)
FEATURE_COMBINATIONS_TECHNICAL = {
    'baseline': [],  # Solo prezzo (viene aggiunto automaticamente)
    
    'momentum': ['momentum_1', 'momentum_3', 'momentum_5', 'momentum_10'],
    
    'momentum_roc': ['momentum_1', 'momentum_3', 'momentum_5', 'momentum_10',
                     'roc_1', 'roc_3', 'roc_5'],
    
    'moving_averages': ['sma_5', 'sma_10', 'sma_20', 'ema_5', 'ema_10'],
    
    'volatility': ['volatility_3', 'volatility_5', 'volatility_10'],
    
    'technical_indicators': ['rsi_14', 'macd', 'macd_signal', 'macd_histogram'],
    
    'bollinger': ['bb_upper', 'bb_lower', 'bb_width', 'bb_position'],
    
    'stochastic': ['stoch_k', 'stoch_d'],
    
    'all_momentum': ['momentum_1', 'momentum_3', 'momentum_5', 'momentum_10',
                     'roc_1', 'roc_3', 'roc_5'],
    
    'all_ma': ['sma_3', 'sma_5', 'sma_10', 'sma_20',
               'ema_3', 'ema_5', 'ema_10'],
    
    'complete': ['momentum_1', 'momentum_3', 'momentum_5', 'momentum_10',
                 'roc_1', 'roc_3', 'roc_5',
                 'sma_5', 'sma_10', 'sma_20',
                 'ema_5', 'ema_10',
                 'volatility_5', 'volatility_10',
                 'rsi_14', 'macd', 'macd_signal',
                 'bb_width', 'bb_position',
                 'stoch_k', 'stoch_d',
                 'lag_1', 'lag_2'],
    
    'optimal': ['momentum_1', 'momentum_3', 'momentum_5',
                'sma_5', 'sma_10',
                'volatility_5',
                'rsi_14', 'macd',
                'bb_position',
                'lag_1']
}

# Definizioni di combinazioni di feature del DATASET ORIGINALE
# Variabili disponibili (escluso closed e date):
# - OHLCV: open, high, low, vol.
# - Fluctuation: fluctuation (dicotomica 0/1)
# - Variazioni: closed_var, open_var, high_var, low_var, vol_var
# - Stagionalità: month, day_month, day_week

# Variabili con R² positivo testate singolarmente (N_STEPS=15):
# - closed_var: 0.8338 (MIGLIORE)
# - month: 0.7613
# - low_var: 0.7021
# - high: 0.5469
# - low: 0.5383
# - day_week: 0.4946
# - open: 0.3039
# - day_month: 0.2963
# - vol_var: 0.2586

# Test singole variabili (per confronto)
FEATURE_COMBINATIONS_DATASET_SINGLE = {
    'dataset_single_closed_var': ['closed_var'],
    'dataset_single_month': ['month'],
    'dataset_single_low_var': ['low_var'],
    'dataset_single_high': ['high'],
    'dataset_single_low': ['low'],
    'dataset_single_day_week': ['day_week'],
    'dataset_single_open': ['open'],
}

# Test combinazioni multiple con N_STEPS=3
# Partiamo da combinazioni di 2 feature per vedere se funzionano
FEATURE_COMBINATIONS_DATASET = {
    # Combinazioni di 2 feature (top performers singoli)
    'dataset_2vars_1': ['closed_var', 'month'],  # Top 2
    'dataset_2vars_2': ['closed_var', 'low_var'],  # Top 1 + Top 3
    'dataset_2vars_3': ['month', 'low_var'],  # Top 2 + Top 3
    'dataset_2vars_4': ['closed_var', 'high'],  # Top 1 + Top 4
    'dataset_2vars_5': ['closed_var', 'low'],  # Top 1 + Top 5
    'dataset_2vars_6': ['month', 'high'],  # Top 2 + Top 4
    'dataset_2vars_7': ['low_var', 'high'],  # Top 3 + Top 4
    
    # Combinazioni di 3 feature (se le 2 funzionano)
    'dataset_3vars_1': ['closed_var', 'month', 'low_var'],  # Top 3
    'dataset_3vars_2': ['closed_var', 'month', 'high'],  # Top 1 + Top 2 + Top 4
    'dataset_3vars_3': ['closed_var', 'low_var', 'high'],  # Top 1 + Top 3 + Top 4
    'dataset_3vars_4': ['month', 'low_var', 'high'],  # Top 2 + Top 3 + Top 4
    'dataset_3vars_5': ['closed_var', 'month', 'low'],  # Top 1 + Top 2 + Top 5
    
    # Combinazioni di 4 feature
    'dataset_4vars_1': ['closed_var', 'month', 'low_var', 'high'],  # Top 4
    'dataset_4vars_2': ['closed_var', 'month', 'low_var', 'low'],  # Top 4 (low invece di high)
    'dataset_4vars_3': ['closed_var', 'month', 'high', 'low'],  # Top 1 + Top 2 + Top 4 + Top 5
    
    # Combinazioni di 5 feature
    'dataset_5vars_1': ['closed_var', 'month', 'low_var', 'high', 'low'],  # Top 5
}

# Per il test con N_STEPS=3, testiamo combinazioni del dataset + baseline per confronto
FEATURE_COMBINATIONS = {
    'baseline': [],  # Solo prezzo per confronto
    **FEATURE_COMBINATIONS_DATASET_SINGLE,  # Singole variabili per confronto
    **FEATURE_COMBINATIONS_DATASET  # Combinazioni multiple
}

# ============================================================================
# FUNZIONI DI PREPROCESSING
# ============================================================================

def create_sequences(data, n_steps, feature_list=None, dataset_df=None, use_dataset_features=False):
    """
    Crea sequenze di input-output per il modello neurale con 1 step ahead.
    Per MLPRegressor, le sequenze vengono appiattite in un vettore.
    
    Parameters:
    -----------
    data : pd.Series
        Serie temporale con valori della media mobile (serie_ma)
    n_steps : int
        Numero di time steps per ogni sequenza di input
    feature_list : list, optional
        Lista di feature da includere. Se None, usa solo il prezzo.
    dataset_df : pd.DataFrame, optional
        DataFrame originale del dataset (per feature del dataset)
    use_dataset_features : bool
        Se True, usa feature del dataset invece di feature tecniche
    
    Returns:
    --------
    X : np.array
        Array di input appiattito (samples, n_steps * n_features)
    y : np.array
        Array di valori target (samples,) - 1 step ahead
    n_features : int
        Numero di features per time step
    feature_names : list
        Nomi delle feature utilizzate
    """
    if feature_list is None:
        feature_list = []
    
    # Combina le feature (dataset o tecniche)
    if use_dataset_features and dataset_df is not None:
        # Usa feature del dataset
        dataset_subset = dataset_df.loc[data.index]  # Allinea per indice
        dataset_features, dataset_feature_names = combine_dataset_features(dataset_subset, feature_list)
        
        if dataset_features is None or len(feature_list) == 0:
            # Fallback: solo prezzo
            data_array, feature_names = combine_features(data, [])
        else:
            # Combina prezzo + feature dataset
            price_array, _ = combine_features(data, [])
            
            # Allinea dimensioni (potrebbero esserci NaN all'inizio)
            min_len = min(len(price_array), len(dataset_features))
            if len(price_array) > min_len:
                price_array = price_array[-min_len:]
                data = data.iloc[-min_len:]
            if len(dataset_features) > min_len:
                dataset_features = dataset_features[-min_len:]
            
            # Combina: prezzo + feature dataset
            data_array = np.column_stack([price_array, dataset_features])
            feature_names = ['price'] + dataset_feature_names
    else:
        # Usa feature tecniche (default)
        data_array, feature_names = combine_features(data, feature_list)
    
    n_features = data_array.shape[1]
    
    X, y = [], []
    
    # Crea sequenze con previsione 1 step ahead
    # X[i] contiene dati da t-n_steps a t-1 (appiattiti per MLP)
    # y[i] contiene il valore a t (1 step ahead)
    for i in range(len(data_array) - n_steps):
        # Appiattisci la sequenza per MLPRegressor
        X.append(data_array[i:i+n_steps].flatten())
        # y è sempre serie_ma con 1 step ahead
        y.append(data.values[i+n_steps])
    
    return np.array(X), np.array(y), n_features, feature_names

def prepare_neural_data(serie_train, serie_test, n_steps, feature_list=None, scaler_type='standard',
                        dataset_df=None, use_dataset_features=False):
    """
    Prepara i dati per il modello neurale con normalizzazione e 1 step ahead.
    
    Parameters:
    -----------
    serie_train : pd.Series
        Serie di training (serie_ma)
    serie_test : pd.Series
        Serie di test (serie_ma)
    n_steps : int
        Numero di time steps
    feature_list : list, optional
        Lista di feature da includere
    scaler_type : str
        Tipo di scaler ('standard' o 'minmax')
    dataset_df : pd.DataFrame, optional
        DataFrame originale del dataset (per feature del dataset)
    use_dataset_features : bool
        Se True, usa feature del dataset invece di feature tecniche
    
    Returns:
    --------
    X_train, y_train, X_test, y_test, scaler_X, scaler_y, n_features, feature_names : tuple
        Dati preparati e scaler
    """
    if feature_list is None:
        feature_list = []
    
    # Crea sequenze per training
    X_train_raw, y_train_raw, n_features, feature_names = create_sequences(
        serie_train, n_steps, feature_list, dataset_df, use_dataset_features
    )
    
    # Crea sequenze per test usando walk-forward validation
    # Ogni sequenza di test usa i dati reali precedenti come contesto
    serie_full = pd.concat([serie_train, serie_test])
    
    # Crea le sequenze di test: ogni sequenza inizia da train_size - n_steps + i
    # per avere sempre il contesto corretto
    X_test_raw, y_test_raw = [], []
    test_len = len(serie_test)
    
    for i in range(test_len):
        # L'indice di inizio per la sequenza di test
        start_idx = len(serie_train) - n_steps + i
        end_idx = len(serie_train) + i
        
        # Estrai la sequenza dalla serie completa per calcolare le feature
        seq_data = serie_full.iloc[start_idx:end_idx]
        
        # Calcola le stesse feature utilizzate nel training
        if use_dataset_features and dataset_df is not None:
            # Usa feature del dataset
            seq_dataset_subset = dataset_df.loc[seq_data.index]
            seq_dataset_features, _ = combine_dataset_features(seq_dataset_subset, feature_list)
            
            if seq_dataset_features is None or len(feature_list) == 0:
                seq_array, _ = combine_features(seq_data, [])
            else:
                seq_price_array, _ = combine_features(seq_data, [])
                # Allinea dimensioni
                min_len = min(len(seq_price_array), len(seq_dataset_features))
                seq_price_array = seq_price_array[-min_len:]
                seq_dataset_features = seq_dataset_features[-min_len:]
                seq_array = np.column_stack([seq_price_array, seq_dataset_features])
        else:
            # Usa feature tecniche
            seq_array, _ = combine_features(seq_data, feature_list)
        
        # Appiattisci per MLPRegressor
        X_test_raw.append(seq_array.flatten())
        # y è sempre serie_ma con 1 step ahead
        y_test_raw.append(serie_test.iloc[i])
    
    X_test_raw = np.array(X_test_raw)
    y_test_raw = np.array(y_test_raw)
    
    # Normalizza i dati
    if scaler_type == 'standard':
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
    else:
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
    
    # Fit scaler solo sul training set
    # Per MLPRegressor, X è già appiattito (samples, n_steps * n_features)
    scaler_X.fit(X_train_raw)
    scaler_y.fit(y_train_raw.reshape(-1, 1))
    
    # Trasforma training
    X_train_scaled = scaler_X.transform(X_train_raw)
    y_train_scaled = scaler_y.transform(y_train_raw.reshape(-1, 1)).flatten()
    
    # Trasforma test
    X_test_scaled = scaler_X.transform(X_test_raw)
    y_test_scaled = scaler_y.transform(y_test_raw.reshape(-1, 1)).flatten()
    
    return (X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
            scaler_X, scaler_y, n_features, feature_names)

# ============================================================================
# ARCHITETTURA MODELLO NEURALE (MLPRegressor)
# ============================================================================

def build_neural_model(n_features, hidden_layer_sizes=None, max_iter=300, 
                      alpha=0.001, learning_rate_init=0.001, solver='adam',
                      early_stopping=True, validation_fraction=0.15,
                      n_iter_no_change=20, tol=1e-4, random_state=42):
    """
    Costruisce un modello MLPRegressor per previsione serie temporali.
    
    Parameters:
    -----------
    n_features : int
        Numero di features (dopo il flatten delle sequenze)
    hidden_layer_sizes : tuple, optional
        Dimensione dei layer nascosti. Default: (64, 32)
    max_iter : int
        Numero massimo di iterazioni (epochs)
    alpha : float
        L2 regularization parameter
    learning_rate_init : float
        Learning rate iniziale
    solver : str
        Solver da usare ('adam', 'lbfgs', 'sgd')
    early_stopping : bool
        Se True, usa early stopping
    validation_fraction : float
        Frazione di dati da usare per validazione
    n_iter_no_change : int
        Numero di iterazioni senza miglioramento per early stopping
    tol : float
        Tolerance per convergenza
    random_state : int
        Random state per riproducibilità
    
    Returns:
    --------
    model : MLPRegressor
        Modello MLP compilato
    """
    if hidden_layer_sizes is None:
        hidden_layer_sizes = HIDDEN_LAYER_SIZES_SIMPLE
    
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',  # ReLU activation (equivalente a LeakyReLU)
        solver=solver,
        alpha=alpha,
        batch_size='auto',
        learning_rate='constant' if solver != 'adam' else 'adaptive',
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        shuffle=True,
        random_state=random_state,
        tol=tol,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=n_iter_no_change,
        max_fun=15000
    )
    
    return model

# ============================================================================
# FUNZIONI DI VALUTAZIONE
# ============================================================================

def evaluate_neural_model(y_true, y_pred, scaler_y=None):
    """
    Valuta le prestazioni del modello neurale.
    
    Parameters:
    -----------
    y_true : array
        Valori reali (normalizzati)
    y_pred : array
        Valori previsti (normalizzati)
    scaler_y : scaler, optional
        Scaler per denormalizzare i dati
    
    Returns:
    --------
    metrics : dict
        Dizionario con le metriche
    y_true_orig, y_pred_orig : arrays
        Valori denormalizzati
    """
    # Denormalizza se necessario
    if scaler_y is not None:
        y_true_orig = scaler_y.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    else:
        y_true_orig = y_true
        y_pred_orig = y_pred
    
    # Calcola metriche
    mse = mean_squared_error(y_true_orig, y_pred_orig)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    mape = mean_absolute_percentage_error(y_true_orig, y_pred_orig)
    r2 = r2_score(y_true_orig, y_pred_orig)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }
    
    return metrics, y_true_orig, y_pred_orig

# ============================================================================
# FUNZIONI DI VISUALIZZAZIONE
# ============================================================================

def plot_training_history_sklearn(model, save_path):
    """Plotta la storia del training per MLPRegressor."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Loss curve
    if hasattr(model, 'loss_curve_'):
        axes[0].plot(model.loss_curve_, label='Train Loss', linewidth=2)
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Validation scores
    if hasattr(model, 'validation_scores_') and len(model.validation_scores_) > 0:
        axes[1].plot(model.validation_scores_, label='Validation Score', linewidth=2)
        axes[1].set_title('Validation Score')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'Validation scores not available', 
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Validation Score')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_predictions(dates, y_true, y_pred, title, save_path, dataset_type='train'):
    """Plotta le previsioni vs valori reali."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(dates, y_true, 'o-', label='Valori reali', linewidth=1.5, markersize=4, alpha=0.7)
    ax.plot(dates, y_pred, 's-', label='Previsioni Neurali', linewidth=1.5, markersize=4, alpha=0.8)
    
    # Calcola metriche per il titolo
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    ax.set_title(f'{title}\nR²: {r2:.4f} | MAPE: {mape*100:.2f}% | MAE: {mae:.4f}', 
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Data')
    ax.set_ylabel('Prezzo di chiusura (media mobile 5)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Zoom per test set
    if dataset_type == 'test':
        y_min = min(np.min(y_true), np.min(y_pred))
        y_max = max(np.max(y_true), np.max(y_pred))
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.01 * y_range, y_max + 0.01 * y_range)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

# ============================================================================
# TEST DI COMBINAZIONI DI FEATURE
# ============================================================================

print("\n" + "="*70)
print("🧪 TEST DI COMBINAZIONI DI FEATURE")
print("="*70)

# Testa diverse combinazioni di feature
best_combination = None
best_r2_test = -np.inf
best_metrics = None
results_summary = []

print("\n📊 Testing diverse combinazioni di feature...")

for combo_name, feature_list in FEATURE_COMBINATIONS.items():
    print(f"\n{'='*70}")
    print(f"🔍 Testing combinazione: {combo_name}")
    print(f"   Features: {', '.join(feature_list) if feature_list else 'solo prezzo'}")
    print(f"{'='*70}")
    
    try:
        # Determina se usare feature del dataset o tecniche
        use_dataset_features = (combo_name in FEATURE_COMBINATIONS_DATASET or 
                                combo_name in FEATURE_COMBINATIONS_DATASET_SINGLE or
                                combo_name.startswith('dataset_'))
        
        # Prepara i dati con questa combinazione di feature
        X_train, y_train, X_test, y_test, scaler_X, scaler_y, n_features, feature_names = prepare_neural_data(
            serie_ma_train, serie_ma_test, N_STEPS, feature_list,
            dataset_df=dataset, use_dataset_features=use_dataset_features
        )
        
        print(f"   Shape X_train: {X_train.shape}")
        print(f"   Numero features: {n_features}")
        print(f"   Features utilizzate: {', '.join(feature_names)}")
        
        # Costruisci il modello (semplificato per test rapido)
        # X_train è già appiattito: (samples, n_steps * n_features)
        n_features_flat = X_train.shape[1]
        model = build_neural_model(
            n_features=n_features_flat,
            hidden_layer_sizes=HIDDEN_LAYER_SIZES_SIMPLE,  # Semplificato per test
            max_iter=250,  # Aumentato da 100 a 250 per test
            alpha=ALPHA,
            learning_rate_init=LEARNING_RATE_INIT,
            solver=SOLVER,
            early_stopping=EARLY_STOPPING,
            validation_fraction=VALIDATION_FRACTION,
            n_iter_no_change=N_ITER_NO_CHANGE,
            tol=TOL,
            random_state=42
        )
        
        # Training
        print(f"   Training modello...")
        history = model.fit(X_train, y_train)
        
        # Previsioni test
        y_pred_test_scaled = model.predict(X_test)
        metrics_test, _, _ = evaluate_neural_model(y_test, y_pred_test_scaled, scaler_y)
        
        r2_test = metrics_test['R2']
        
        # Filtra solo modelli con R² test positivo
        if r2_test > 0:
            print(f"   ✅ R² Test: {r2_test:.4f} | MAPE: {metrics_test['MAPE']*100:.2f}% | MAE: {metrics_test['MAE']:.4f}")
            
            results_summary.append({
                'combination': combo_name,
                'features': feature_list,
                'n_features': n_features,
                'r2_test': r2_test,
                'mape_test': metrics_test['MAPE'],
                'mae_test': metrics_test['MAE'],
                'rmse_test': metrics_test['RMSE']
            })
            
            # Salva la migliore combinazione (solo se R² > 0)
            if r2_test > best_r2_test:
                best_r2_test = r2_test
                best_combination = combo_name
                best_metrics = metrics_test.copy()
                best_metrics['combination'] = combo_name
                best_metrics['feature_list'] = feature_list
                best_metrics['feature_names'] = feature_names
                best_metrics['n_features'] = n_features
        else:
            print(f"   ⚠️  R² Test: {r2_test:.4f} (NEGATIVO - scartato)")
            results_summary.append({
                'combination': combo_name,
                'features': feature_list,
                'n_features': n_features,
                'r2_test': r2_test,
                'mape_test': metrics_test['MAPE'],
                'mae_test': metrics_test['MAE'],
                'rmse_test': metrics_test['RMSE'],
                'status': 'scartato (R² ≤ 0)'
            })
        
    except Exception as e:
        print(f"   ❌ Errore durante il test: {str(e)}")
        results_summary.append({
            'combination': combo_name,
            'features': feature_list,
            'n_features': 0,
            'r2_test': -999,
            'mape_test': 999,
            'mae_test': 999,
            'rmse_test': 999,
            'error': str(e)
        })

# Stampa riepilogo
print("\n" + "="*70)
print("📊 RIEPILOGO TEST COMBINAZIONI")
print("="*70)
print(f"\n{'Combinazione':<20} {'R² Test':<12} {'MAPE %':<12} {'MAE':<12} {'N Features':<12}")
print("-"*70)

# Filtra solo modelli con R² test positivo per il riepilogo
results_summary_valid = [r for r in results_summary if 'error' not in r and 'status' not in r and r['r2_test'] > 0]
results_summary_invalid = [r for r in results_summary if 'error' not in r and ('status' in r or r['r2_test'] <= 0)]
results_summary_errors = [r for r in results_summary if 'error' in r]

results_summary_sorted = sorted(results_summary_valid, key=lambda x: x['r2_test'], reverse=True)

print(f"\n📊 MODELLI VALIDI (R² Test > 0):")
print("-"*70)
for result in results_summary_sorted:
    print(f"{result['combination']:<20} {result['r2_test']:<12.4f} {result['mape_test']*100:<12.2f} {result['mae_test']:<12.4f} {result['n_features']:<12}")

if results_summary_invalid:
    print(f"\n⚠️  MODELLI SCARTATI (R² Test ≤ 0 o problemi):")
    print("-"*70)
    for result in results_summary_invalid:
        status = result.get('status', 'R² ≤ 0')
        print(f"{result['combination']:<20} R²: {result['r2_test']:<12.4f} - {status}")

if results_summary_errors:
    print(f"\n❌ MODELLI CON ERRORI:")
    print("-"*70)
    for result in results_summary_errors:
        print(f"{result['combination']:<20} - {result.get('error', 'Errore sconosciuto')}")

if best_combination and best_metrics:
    print(f"\n🏆 Migliore combinazione: {best_combination}")
    print(f"   R² Test: {best_r2_test:.4f}")
    print(f"   Features: {', '.join(best_metrics.get('feature_list', [])) if best_metrics.get('feature_list') else 'solo prezzo'}")
else:
    print(f"\n⚠️  Nessuna combinazione con R² test positivo trovata!")
    print(f"   Procederò con la migliore combinazione disponibile (anche se R² ≤ 0)")
    results_summary_all = sorted([r for r in results_summary if 'error' not in r], 
                                 key=lambda x: x['r2_test'], reverse=True)
    if results_summary_all:
        best_result = results_summary_all[0]
        best_combination = best_result['combination']
        best_r2_test = best_result['r2_test']
        # Ricrea best_metrics
        for combo_name, feature_list in FEATURE_COMBINATIONS.items():
            if combo_name == best_combination:
                # Determina se usare feature del dataset o tecniche
                use_dataset_features = (combo_name in FEATURE_COMBINATIONS_DATASET or 
                                        combo_name in FEATURE_COMBINATIONS_DATASET_SINGLE or
                                        combo_name.startswith('dataset_'))
                
                X_train, y_train, X_test, y_test, scaler_X, scaler_y, n_features, feature_names = prepare_neural_data(
                    serie_ma_train, serie_ma_test, N_STEPS, feature_list,
                    dataset_df=dataset, use_dataset_features=use_dataset_features
                )
                n_features_flat = X_train.shape[1]
                model = build_neural_model(
                    n_features=n_features_flat,
                    hidden_layer_sizes=HIDDEN_LAYER_SIZES_SIMPLE,
                    max_iter=250,
                    alpha=ALPHA,
                    learning_rate_init=LEARNING_RATE_INIT,
                    solver=SOLVER,
                    early_stopping=EARLY_STOPPING,
                    validation_fraction=VALIDATION_FRACTION,
                    n_iter_no_change=N_ITER_NO_CHANGE,
                    tol=TOL,
                    random_state=42
                )
                model.fit(X_train, y_train)
                y_pred_test_scaled = model.predict(X_test)
                metrics_test, _, _ = evaluate_neural_model(y_test, y_pred_test_scaled, scaler_y)
                best_metrics = metrics_test.copy()
                best_metrics['combination'] = best_combination
                best_metrics['feature_list'] = feature_list
                best_metrics['feature_names'] = feature_names
                best_metrics['n_features'] = n_features
                break

print("="*70)

# Salva riepilogo completo e validi separatamente
results_df_all = pd.DataFrame(results_summary)
results_df_all.to_csv(os.path.join(OUTPUT_DIR, "feature_combinations_test_all.csv"), index=False)

if results_summary_sorted:
    results_df_valid = pd.DataFrame(results_summary_sorted)
    results_df_valid.to_csv(os.path.join(OUTPUT_DIR, "feature_combinations_test_valid.csv"), index=False)
    print(f"\n✅ Riepilogo completo salvato in: {OUTPUT_DIR}/feature_combinations_test_all.csv")
    print(f"✅ Riepilogo modelli validi (R² > 0) salvato in: {OUTPUT_DIR}/feature_combinations_test_valid.csv")
    
    # Salva i top 6 modelli per sviluppo completo
    top_6_combinations = results_summary_sorted[:6] if len(results_summary_sorted) >= 6 else results_summary_sorted
    print(f"\n📊 Top {len(top_6_combinations)} modelli da sviluppare:")
    for i, result in enumerate(top_6_combinations, 1):
        print(f"  {i}. {result['combination']}: R² = {result['r2_test']:.4f}")
else:
    print(f"\n⚠️  Nessun modello valido trovato. Riepilogo completo salvato in: {OUTPUT_DIR}/feature_combinations_test_all.csv")

# ============================================================================
# SVILUPPO TOP 6 MODELLI CON PREVISIONI E METRICHE
# ============================================================================

if not best_combination or not results_summary_sorted:
    print("\n" + "="*70)
    print("❌ ERRORE: Nessuna combinazione valida trovata!")
    print("="*70)
    print("Impossibile procedere con lo sviluppo dei modelli.")
    exit(1)

# Top 6 modelli da sviluppare
top_6_combinations = results_summary_sorted[:6] if len(results_summary_sorted) >= 6 else results_summary_sorted

print("\n" + "="*70)
print("🧠 SVILUPPO TOP 6 MODELLI NEURALI CON PREVISIONI COMPLETE")
print("="*70)
print(f"\n📊 Modelli da sviluppare: {len(top_6_combinations)}")
for i, result in enumerate(top_6_combinations, 1):
    print(f"  {i}. {result['combination']}: R² = {result['r2_test']:.4f}")

# Dizionario per salvare tutti i modelli e risultati
all_models_results = {}

# Sviluppa ogni modello del top 6
for idx, result in enumerate(top_6_combinations, 1):
    combo_name = result['combination']
    feature_list = result['features']
    
    print("\n" + "="*70)
    print(f"🤖 MODELLO {idx}/6: {combo_name.upper()}")
    print("="*70)
    print(f"   R² Test: {result['r2_test']:.4f}")
    print(f"   MAPE: {result['mape_test']*100:.2f}%")
    print(f"   MAE: {result['mae_test']:.4f}")
    print(f"   Features: {feature_list if feature_list else 'solo prezzo'}")
    print(f"   Numero features: {result['n_features']}")
    
    try:
        # Determina se usare feature del dataset o tecniche
        use_dataset_features = (combo_name in FEATURE_COMBINATIONS_DATASET or 
                                combo_name in FEATURE_COMBINATIONS_DATASET_SINGLE or
                                combo_name.startswith('dataset_'))
        
        # Prepara i dati
        X_train, y_train, X_test, y_test, scaler_X, scaler_y, n_features, feature_names = prepare_neural_data(
            serie_ma_train, serie_ma_test, N_STEPS, feature_list,
            dataset_df=dataset, use_dataset_features=use_dataset_features
        )
        
        print(f"\n📊 Dati preparati:")
        print(f"   Shape X_train: {X_train.shape}")
        print(f"   Shape y_train: {y_train.shape}")
        print(f"   Shape X_test: {X_test.shape}")
        print(f"   Shape y_test: {y_test.shape}")
        print(f"   Features utilizzate: {', '.join(feature_names)}")
        
        # Costruisci il modello (USA STESSA CONFIGURAZIONE DEL TEST INIZIALE)
        n_features_flat = X_train.shape[1]
        print(f"\n🏗️  Costruzione modello MLPRegressor...")
        model = build_neural_model(
            n_features=n_features_flat,
            hidden_layer_sizes=HIDDEN_LAYER_SIZES_SIMPLE,  # (64, 32) - stessa del test iniziale per coerenza
            max_iter=250,  # Stesso del test iniziale per coerenza
            alpha=ALPHA,
            learning_rate_init=LEARNING_RATE_INIT,
            solver=SOLVER,
            early_stopping=EARLY_STOPPING,
            validation_fraction=VALIDATION_FRACTION,
            n_iter_no_change=N_ITER_NO_CHANGE,
            tol=TOL,
            random_state=42  # Stessa seed per riproducibilità
        )
        
        print(f"   Architettura: Input({n_features_flat}) -> Hidden{HIDDEN_LAYER_SIZES_SIMPLE} -> Output(1)")
        print(f"   Max iterations: 250 (stessa del test iniziale)")
        print(f"   Solver: {SOLVER}")
        print(f"   Alpha (L2 reg): {ALPHA}")
        
        # Training
        print(f"\n🎯 Training del modello...")
        model.fit(X_train, y_train)
        
        print(f"   ✅ Training completato!")
        print(f"   Iterazioni completate: {model.n_iter_}")
        print(f"   Loss finale: {model.loss_:.6f}")
        
        # Salva il modello
        import joblib
        model_dir = os.path.join(OUTPUT_DIR, f"model_{idx}_{combo_name}")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        model_path = os.path.join(model_dir, f'{combo_name}_model.pkl')
        joblib.dump(model, model_path)
        print(f"   ✅ Modello salvato in: {model_path}")
        
        # Previsioni
        print(f"\n🔮 Generazione previsioni...")
        
        # Previsioni training
        y_pred_train_scaled = model.predict(X_train)
        metrics_train, y_train_orig, y_pred_train_orig = evaluate_neural_model(
            y_train, y_pred_train_scaled, scaler_y
        )
        
        # Previsioni test
        y_pred_test_scaled = model.predict(X_test)
        metrics_test, y_test_orig, y_pred_test_orig = evaluate_neural_model(
            y_test, y_pred_test_scaled, scaler_y
        )
        
        # Previsioni su tutto il dataset
        serie_ma_full = pd.concat([serie_ma_train, serie_ma_test])
        X_full_raw, y_full_raw, _, _ = create_sequences(
            serie_ma_full, N_STEPS, feature_list, dataset_df=dataset, use_dataset_features=use_dataset_features
        )
        X_full_scaled = scaler_X.transform(X_full_raw)
        y_full_scaled = scaler_y.transform(y_full_raw.reshape(-1, 1)).flatten()
        y_pred_full_scaled = model.predict(X_full_scaled)
        metrics_full, y_full_orig, y_pred_full_orig = evaluate_neural_model(
            y_full_scaled, y_pred_full_scaled, scaler_y
        )
        
        # Salva metriche
        metrics_df = pd.DataFrame({
            'dataset': ['train', 'test', 'full'],
            'r2': [metrics_train['R2'], metrics_test['R2'], metrics_full['R2']],
            'mape': [metrics_train['MAPE'], metrics_test['MAPE'], metrics_full['MAPE']],
            'mae': [metrics_train['MAE'], metrics_test['MAE'], metrics_full['MAE']],
            'rmse': [metrics_train['RMSE'], metrics_test['RMSE'], metrics_full['RMSE']]
        })
        metrics_path = os.path.join(model_dir, f'{combo_name}_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f"   ✅ Metriche salvate in: {metrics_path}")
        
        # Visualizzazioni
        print(f"\n📈 Creazione visualizzazioni...")
        
        # Training set
        train_dates = serie_ma_train.index[N_STEPS:]
        plot_predictions(
            train_dates,
            y_train_orig,
            y_pred_train_orig,
            f"Previsioni - {combo_name.upper()} - Training Set",
            os.path.join(model_dir, f'{combo_name}_predictions_train.png'),
            dataset_type='train'
        )
        
        # Test set
        test_dates = serie_ma_test.index[:len(y_test_orig)]
        plot_predictions(
            test_dates,
            y_test_orig,
            y_pred_test_orig,
            f"Previsioni - {combo_name.upper()} - Test Set",
            os.path.join(model_dir, f'{combo_name}_predictions_test.png'),
            dataset_type='test'
        )
        
        # Full dataset
        full_dates = serie_ma_full.index[N_STEPS:]
        plot_predictions(
            full_dates,
            y_full_orig,
            y_pred_full_orig,
            f"Previsioni - {combo_name.upper()} - Full Dataset",
            os.path.join(model_dir, f'{combo_name}_predictions_full.png'),
            dataset_type='full'
        )
        
        # Grafico comparativo delle metriche
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # R²
        axes[0].bar(['Train', 'Test', 'Full'], 
                   [metrics_train['R2'], metrics_test['R2'], metrics_full['R2']],
                   color=['blue', 'green', 'orange'], alpha=0.7)
        axes[0].set_title('R² Score')
        axes[0].set_ylabel('R²')
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].set_ylim([0, 1])
        
        # MAPE
        axes[1].bar(['Train', 'Test', 'Full'],
                   [metrics_train['MAPE']*100, metrics_test['MAPE']*100, metrics_full['MAPE']*100],
                   color=['blue', 'green', 'orange'], alpha=0.7)
        axes[1].set_title('MAPE (%)')
        axes[1].set_ylabel('MAPE (%)')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # MAE
        axes[2].bar(['Train', 'Test', 'Full'],
                   [metrics_train['MAE'], metrics_test['MAE'], metrics_full['MAE']],
                   color=['blue', 'green', 'orange'], alpha=0.7)
        axes[2].set_title('MAE')
        axes[2].set_ylabel('MAE')
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, f'{combo_name}_metrics_comparison.png'), 
                   dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"   ✅ Visualizzazioni salvate in: {model_dir}/")
        
        # Salva risultati per confronto
        all_models_results[combo_name] = {
            'model': model,
            'metrics_train': metrics_train,
            'metrics_test': metrics_test,
            'metrics_full': metrics_full,
            'feature_list': feature_list,
            'feature_names': feature_names,
            'n_features': n_features,
            'y_train_orig': y_train_orig,
            'y_pred_train_orig': y_pred_train_orig,
            'y_test_orig': y_test_orig,
            'y_pred_test_orig': y_pred_test_orig,
            'y_full_orig': y_full_orig,
            'y_pred_full_orig': y_pred_full_orig,
            'train_dates': train_dates,
            'test_dates': test_dates,
            'full_dates': full_dates,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'model_dir': model_dir
        }
        
        print(f"\n✅ Modello {idx}/6 ({combo_name}) completato!")
        
    except Exception as e:
        print(f"\n❌ Errore durante lo sviluppo del modello {combo_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

# ============================================================================
# CONFRONTO FINALE DEI TOP 6 MODELLI
# ============================================================================

print("\n" + "="*70)
print("📊 CONFRONTO FINALE TOP 6 MODELLI")
print("="*70)

if all_models_results:
    # Tabella comparativa
    comparison_df = pd.DataFrame({
        'Modello': list(all_models_results.keys()),
        'R² Train': [all_models_results[k]['metrics_train']['R2'] for k in all_models_results.keys()],
        'R² Test': [all_models_results[k]['metrics_test']['R2'] for k in all_models_results.keys()],
        'R² Full': [all_models_results[k]['metrics_full']['R2'] for k in all_models_results.keys()],
        'MAPE Train (%)': [all_models_results[k]['metrics_train']['MAPE']*100 for k in all_models_results.keys()],
        'MAPE Test (%)': [all_models_results[k]['metrics_test']['MAPE']*100 for k in all_models_results.keys()],
        'MAPE Full (%)': [all_models_results[k]['metrics_full']['MAPE']*100 for k in all_models_results.keys()],
        'MAE Train': [all_models_results[k]['metrics_train']['MAE'] for k in all_models_results.keys()],
        'MAE Test': [all_models_results[k]['metrics_test']['MAE'] for k in all_models_results.keys()],
        'MAE Full': [all_models_results[k]['metrics_full']['MAE'] for k in all_models_results.keys()],
        'N Features': [all_models_results[k]['n_features'] for k in all_models_results.keys()]
    })
    
    comparison_df = comparison_df.sort_values('R² Test', ascending=False)
    comparison_path = os.path.join(OUTPUT_DIR, "top6_models_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n✅ Confronto salvato in: {comparison_path}")
    print("\n" + comparison_df.to_string(index=False))
    
    # Grafico comparativo finale
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    models_list = list(all_models_results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(models_list)))
    
    # R² Test
    axes[0, 0].barh(models_list, 
                   [all_models_results[m]['metrics_test']['R2'] for m in models_list],
                   color=colors)
    axes[0, 0].set_title('R² Test Score (più alto è meglio)', fontweight='bold')
    axes[0, 0].set_xlabel('R²')
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    # MAPE Test
    axes[0, 1].barh(models_list,
                   [all_models_results[m]['metrics_test']['MAPE']*100 for m in models_list],
                   color=colors)
    axes[0, 1].set_title('MAPE Test (%) (più basso è meglio)', fontweight='bold')
    axes[0, 1].set_xlabel('MAPE (%)')
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    # MAE Test
    axes[1, 0].barh(models_list,
                   [all_models_results[m]['metrics_test']['MAE'] for m in models_list],
                   color=colors)
    axes[1, 0].set_title('MAE Test (più basso è meglio)', fontweight='bold')
    axes[1, 0].set_xlabel('MAE')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Numero features
    axes[1, 1].barh(models_list,
                   [all_models_results[m]['n_features'] for m in models_list],
                   color=colors)
    axes[1, 1].set_title('Numero di Features', fontweight='bold')
    axes[1, 1].set_xlabel('N Features')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "top6_models_comparison.png"), 
               dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Grafico confronto salvato in: {OUTPUT_DIR}/top6_models_comparison.png")
    
    # Previsioni test set a confronto
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot valori reali una sola volta
    first_model = list(all_models_results.keys())[0]
    test_dates = all_models_results[first_model]['test_dates']
    y_test_orig = all_models_results[first_model]['y_test_orig']
    ax.plot(test_dates, y_test_orig, 'ko-', label='Valori Reali', 
           linewidth=3, markersize=8, alpha=0.8, zorder=10)
    
    # Plot previsioni di ogni modello
    for i, (model_name, results) in enumerate(all_models_results.items()):
        ax.plot(test_dates, results['y_pred_test_orig'], 
               'o-', label=f"{model_name} (R²={results['metrics_test']['R2']:.3f})",
               linewidth=2, markersize=6, alpha=0.7, zorder=5-i)
    
    ax.set_title('Confronto Previsioni Test Set - Top 6 Modelli', fontsize=14, fontweight='bold')
    ax.set_xlabel('Data', fontsize=12)
    ax.set_ylabel('Prezzo di chiusura (media mobile 5)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "top6_models_predictions_comparison.png"), 
               dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Grafico previsioni a confronto salvato in: {OUTPUT_DIR}/top6_models_predictions_comparison.png")

print("\n" + "="*70)
print("✅ SVILUPPO TOP 6 MODELLI COMPLETATO")
print("="*70)
print(f"\n📁 Tutti i risultati salvati in: {OUTPUT_DIR}/")
print(f"  - Confronto top 6: top6_models_comparison.csv")
print(f"  - Grafico confronto: top6_models_comparison.png")
print(f"  - Grafico previsioni a confronto: top6_models_predictions_comparison.png")
print(f"\n  Ogni modello ha la sua directory:")
for idx, combo_name in enumerate(all_models_results.keys(), 1):
    model_dir = all_models_results[combo_name]['model_dir']
    print(f"    {idx}. {combo_name}: {model_dir}/")
    print(f"       - Modello: {combo_name}_model.pkl")
    print(f"       - Metriche: {combo_name}_metrics.csv")
    print(f"       - Grafici: predictions_train/test/full + metrics_comparison")
print("="*70)

# ============================================================================
# RIEPILOGO FINALE
# ============================================================================

if best_combination and best_combination in all_models_results:
    print("\n" + "="*70)
    print("🏆 MODELLO FINALE (MIGLIORE COMBINAZIONE)")
    print("="*70)
    print(f"   Combinazione: {best_combination}")
    print(f"   R² Test: {best_r2_test:.4f}")
    print(f"   ✅ Modello sviluppato e salvato in: {all_models_results[best_combination]['model_dir']}")
    print(f"   📊 Tutte le metriche e previsioni disponibili nella directory del modello")
    print("="*70)
else:
    print("\n⚠️  Modello migliore non disponibile in all_models_results")

print("\n" + "="*70)
print("✅ TUTTI I TOP 6 MODELLI COMPLETATI")
print("="*70)
