import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas_ta as ta

def prepare_data(data, window_size=30):
    # Calculate technical indicators
    data.ta.rsi(length=14, append=True)
    data.ta.macd(append=True)
    data.ta.sma(length=20, append=True)
    data.ta.sma(length=50, append=True)
    data.ta.atr(length=14, append=True)
    data.ta.bbands(length=20, append=True)
    data.ta.obv(append=True)
    data.ta.adx(length=14, append=True)
    data.ta.cci(length=20, append=True)
    data.ta.mom(length=10, append=True)
    
    # Create features
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI_14', 'MACD_12_26_9', 'SMA_20', 'SMA_50', 'ATRr_14', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'OBV', 'ADX_14', 'CCI_20', 'MOM_10']
    X = data[features].values
    y = data[['Close']].shift(-1).values[:-1]  # Predict next day's close
    
    # Normalize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y)
    
    # Create sequences
    X_sequences = []
    y_sequences = []
    for i in range(len(X_scaled) - window_size):
        X_sequences.append(X_scaled[i:i+window_size])
        y_sequences.append(y_scaled[i+window_size])
    
    return np.array(X_sequences), np.array(y_sequences), scaler

def add_sentiment_features(X, sentiment_scores):
    # Ensure sentiment_scores align with X
    aligned_scores = sentiment_scores[-len(X):]
    
    # Calculate moving averages of sentiment scores
    sentiment_ma_3 = np.convolve(aligned_scores, np.ones(3), 'valid') / 3
    sentiment_ma_7 = np.convolve(aligned_scores, np.ones(7), 'valid') / 7
    
    # Pad the moving averages to match X's length
    pad_3 = np.full(2, np.nan)
    pad_7 = np.full(6, np.nan)
    sentiment_ma_3 = np.concatenate((pad_3, sentiment_ma_3))
    sentiment_ma_7 = np.concatenate((pad_7, sentiment_ma_7))
    
    # Add sentiment features to X
    sentiment_features = np.column_stack((aligned_scores, sentiment_ma_3, sentiment_ma_7))
    X_with_sentiment = np.concatenate((X, sentiment_features.reshape(X.shape[0], X.shape[1], -1)), axis=2)
    
    return X_with_sentiment

def add_fundamental_features(symbol):
    # This is a placeholder function. In a real-world scenario, you would
    # fetch actual fundamental data for the given symbol.
    # For now, we'll return random data as an example.
    num_samples = 100
    num_features = 5
    return np.random.randn(num_samples, num_features)

#the end#

