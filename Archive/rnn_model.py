# rnn_model.py

import torch
import torch.nn as nn
import numpy as np
import random
import time
from sklearn.metrics import mean_squared_error

# Seed-Setter
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Dummy-Daten (z. B. 100 Sequenzen à 10 Zeitschritte mit je 1 Feature)
def generate_dummy_data(n_samples=100, seq_len=10, input_size=1):
    X = np.random.rand(n_samples, seq_len, input_size)
    y = np.random.rand(n_samples)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Einfaches RNN-Modell
class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=16):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # letztes Zeitschritt-Output
        return self.fc(out).squeeze()

# Main-Funktion, die von außen aufgerufen wird
def main(seed=42):
    start_time = time.time()
    set_seed(seed)

    # Daten vorbereiten
    X, y = generate_dummy_data()
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Modell, Loss, Optimizer
    model = SimpleRNN()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training
    model.train()
    for epoch in range(20):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy()
        y_test_np = y_test.numpy()
        rmse = np.sqrt(mean_squared_error(y_test_np, predictions))

    end_time = time.time()
    print(f"Seed/Modell   {seed} abgeschlossen in {end_time - start_time:.2f} Sekunden.")

    return rmse