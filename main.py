import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from typing import Tuple

INPUTS_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "day", "month", "year"]
OUTPUT_COLUMNS = ["Close"]
INPUT_SIZE = len(INPUTS_COLUMNS)
OUTPUT_SIZE = len(OUTPUT_COLUMNS)

EPOCHS = 10
STEPS_PER_EPOCH = 100
SEQ_LEN = 16
BATCH_SIZE = 128
LR = 5e-4
DROP_PROB = 0.5

# preprocess given data for training a sequential neural network
# data Columns: Date, Open, High, Low, Close, Adj Close, Volume 
def preprocess_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    # convert date to datetime object
    data["Date"] = pd.to_datetime(data["Date"])
    # add day as 0,1,2,3,4,5,6
    data["day"] = data["Date"].dt.dayofweek
    # maybe sin cos encoding of day
    # data["sin_day"] = np.sin(2 * np.pi * data["day"] / 6)

    # add month, year columns 
    data["month"] = data["Date"].dt.month
    data["year"] = data["Date"].dt.year

    # drop date column
    data = data.drop(columns=["Date"])

    #only up to year 2021
    data = data[data["year"] <= 2021]

    # normalize the data
    scaler = StandardScaler()
    data[["Open", "High", "Low", "Close", "Adj Close", "Volume"]] = scaler.fit_transform(data[["Open", "High", "Low", "Close", "Adj Close", "Volume"]])

    # split into validation and training data
    split_pct = 0.8
    train_data = data.iloc[:int(len(data) * split_pct)]
    val_data = data.iloc[int(len(data) * split_pct):]

    return train_data, val_data, scaler

def data_generator(data: pd.DataFrame, seq_len: int, batch_size: int, device: torch.device):
    data = data.to_numpy()
    n = len(data) - seq_len
    output_indices = [INPUTS_COLUMNS.index(col) for col in OUTPUT_COLUMNS]

    for i in range(0, n, batch_size):
        x = []
        y = []
        for j in range(min(batch_size, n-i)):
            x.append(data[i+j:i+j+seq_len])  
            y.append(data[i+j+seq_len, output_indices])  # Get target for the next timestep after the sequence

        yield torch.tensor(np.array(x), dtype=torch.float32).to(device), \
              torch.tensor(np.array(y), dtype=torch.float32).to(device)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, OUTPUT_SIZE)
        self.dropout = nn.Dropout(DROP_PROB)

    def forward(self, x):
        # Forward pass through the entire sequence
        batch_size, seq_len, input_size = x.shape
        
        # Reshape for processing all timesteps at once
        x_flat = x.reshape(-1, input_size)
        
        # Process through layers
        x_flat = torch.relu(self.fc1(x_flat))
        x_flat = self.dropout(x_flat)
        x_flat = torch.relu(self.fc2(x_flat))
        x_flat = self.dropout(x_flat)
        x_flat = torch.relu(self.fc3(x_flat))
        x_flat = self.dropout(x_flat)
        x_flat = self.fc4(x_flat)
        
        # Reshape back to sequence form
        x_seq = x_flat.view(batch_size, seq_len, OUTPUT_SIZE)
        
        # Only return the last prediction for each sequence
        return x_seq[:, -1, :]

class CONV1D(nn.Module):
    def __init__(self):
        super(CONV1D, self).__init__()
        self.conv1 = nn.Conv1d(INPUT_SIZE, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, 3, padding=1)
        # self.conv3 = nn.Conv1d(96, 64, 3)

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(64, OUTPUT_SIZE)
        self.dropout = nn.Dropout(DROP_PROB)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        # Transpose to [batch_size, input_size, seq_len] for 1D convolution
        x = x.permute(0, 2, 1)
        
        # Apply convolutional layers with ReLU activation and dropout
        x = torch.relu(self.conv1(x))
        x = self.dropout(x)
        x = torch.relu(self.conv2(x))
        x = self.dropout(x)
        # x = torch.relu(self.conv3(x))
        # x = self.dropout(x)
        
        # Global average pooling across the sequence dimension
        x = self.gap(x)  # Shape: [batch_size, 64, 1]
        
        # Remove the last dimension
        x = x.squeeze(-1)  # Shape: [batch_size, 64]
        
        # Final fully connected layer
        x = self.fc(x)  # Shape: [batch_size, OUTPUT_SIZE]
        
        return x


# LSTM
# TRANSFORMER

# Trains the given model with the given data
def train(model, loss_fn, optimizer, trainings_data, validation_data, raw_data, device):
    train_loss_history = []
    val_loss_history = []
    mae_history = []
    mape_history = []

    train_data_gen = data_generator(trainings_data, SEQ_LEN, BATCH_SIZE, device)
    val_data_gen = data_generator(validation_data, SEQ_LEN, BATCH_SIZE, device)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_mae = 0
        train_mape = 0

        for _ in tqdm(range(STEPS_PER_EPOCH)):
            try:
                inputs, labels = next(train_data_gen)
            except StopIteration:
                # Restart the generator if we run out of data
                train_data_gen = data_generator(trainings_data, SEQ_LEN, BATCH_SIZE, device)
                inputs, labels = next(train_data_gen)
                
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # calculate mean absolute error
            mae = torch.mean(torch.abs(outputs - labels)).item()
            train_mae += mae
            train_mape += mae / torch.mean(torch.abs(labels)).item()  # Added abs to prevent division by zero

        train_loss /= STEPS_PER_EPOCH
        train_mae /= STEPS_PER_EPOCH
        train_mape /= STEPS_PER_EPOCH
        mae_history.append(train_mae)
        mape_history.append(train_mape)
        train_loss_history.append(train_loss)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for _ in range(STEPS_PER_EPOCH):
                try:
                    inputs, labels = next(val_data_gen)
                except StopIteration:
                    # Restart the generator if we run out of data
                    val_data_gen = data_generator(validation_data, SEQ_LEN, BATCH_SIZE, device)
                    inputs, labels = next(val_data_gen)
                    
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

        val_loss /= STEPS_PER_EPOCH
        val_loss_history.append(val_loss)

        print(f"Epoch {epoch + 1}/{EPOCHS}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, MAE: {train_mae:.4f}, MAPE: {train_mape:.4f}")

    return train_loss_history, val_loss_history, mae_history, mape_history

def main():
    path = "data/AAPL.csv"
    data = pd.read_csv(path)
    train_data, val_data, scaler = preprocess_data(data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = MLP().to(device)
    model = CONV1D().to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_loss_history, val_loss_history, mae_history, mape_history = train(model, loss_fn, optimizer, train_data, val_data, data, device)

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(mae_history, label="MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Error")
    plt.title("Training MAE")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()