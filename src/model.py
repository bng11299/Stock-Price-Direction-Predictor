import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

FEATURES = [
    'return',
    'rsi',
    'vol_z',
    'mom_3',
    'mom_10',
    'trend_10',
    'vol_ratio',
    'vol_surge',
    'pos_in_range'
]

# -------------------
# Logistic Regression
# -------------------
def logistic_model():
    import os

    df = pd.read_csv('data/processed/features.csv')

    X = df[FEATURES].copy()
    y = df['target'].values

    # Clean NaNs / inf
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    X = X.values

    print("NaNs remaining:", np.isnan(X).sum())
    print("Shape:", X.shape)

    # Time-based split
    split = int(len(df)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"Logistic Regression Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    os.makedirs('results/metrics', exist_ok=True)
    pd.DataFrame([{'model': 'logistic', 'accuracy': acc}]).to_csv(
        'results/metrics/logistic.csv', index=False
    )

    return acc

# -------------------
# LSTM Model
# -------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

def prepare_sequences(df, features, seq_len=5):
    
    data = df[features].values
    labels = df['target'].values
    
    X, y = [], []
    
    for i in range(seq_len, len(df)):
        X.append(data[i-seq_len:i])
        y.append(labels[i])
    
    X = np.array(X)
    y = np.array(y)
    
    print("LSTM X shape:", X.shape)
    
    return X, y

def train_lstm(seq_len=5, epochs=20, batch_size=32):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler


    print("\nTraining LSTM Model...")

    # Load data
    df = pd.read_csv('data/processed/features.csv')

    # --- Select CLEAN features (no leakage, no garbage columns) ---
    features = FEATURES

    # Drop NaNs safely
    df = df[features + ['target']].dropna().reset_index(drop=True)

    # --- Scale features (VERY important for LSTM) ---
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # --- Create sequences ---
    X, y = [], []
    for i in range(seq_len, len(df)):
        X.append(df[features].iloc[i-seq_len:i].values)
        y.append(df['target'].iloc[i])

    X = np.array(X)
    y = np.array(y)

    print(f"LSTM X shape: {X.shape}, y shape: {y.shape}")

    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # --- Model ---
    class LSTMClassifier(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.lstm = nn.LSTM(input_size, 64, batch_first=True)
            self.fc = nn.Linear(64, 2)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            return self.fc(out)

    model = LSTMClassifier(input_size=X.shape[2])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- Train ---
    losses = []

    for epoch in range(epochs):
        total_loss = 0

        for xb, yb in dataloader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        _, preds = torch.max(outputs, 1)
        acc = (preds == y_tensor).float().mean().item()



    import pandas as pd
    os.makedirs('results/metrics', exist_ok=True)
    pd.DataFrame([{'model': 'lstm', 'accuracy': acc}]).to_csv(
        'results/metrics/lstm.csv', index=False
    )

    print("Saved LSTM metrics to results/metrics/lstm.csv")
    print(f"LSTM Accuracy: {acc:.4f}")
    return acc

def xgboost_model():
    import pandas as pd
    from sklearn.metrics import accuracy_score
    from xgboost import XGBClassifier

    df = pd.read_csv('data/processed/features.csv')

    X = df[FEATURES]
    y = df['target']

    # Time-based split
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"XGBoost Accuracy: {acc:.4f}")

    return acc

def boosting_model():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.ensemble import HistGradientBoostingClassifier

    df = pd.read_csv('data/processed/features.csv')

    X = df[FEATURES]
    y = df['target']

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = HistGradientBoostingClassifier()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Boosting Accuracy: {acc:.4f}")

    return acc

    
if __name__ == "__main__":
    logistic_model()
    train_lstm()