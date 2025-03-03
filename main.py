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
NORMALIZE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
OUTPUT_COLUMNS = ["Close"]
INPUT_SIZE = len(INPUTS_COLUMNS)
OUTPUT_SIZE = len(OUTPUT_COLUMNS)

EPOCHS = 10
STEPS_PER_EPOCH = 100
SEQ_LEN = 32
PRED_LEN = 5
BATCH_SIZE = 128
LR = 5e-4
DROP_PROB = 0.5

# preprocess given data for training a sequential neural network
# data Columns: Date, Open, High, Low, Close, Adj Close, Volume
def preprocess_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    # convert date to datetime object
    data["Date"] = pd.to_datetime(data["Date"])

    # add day as 1 - 31 | 30 | 29 | 28
    data["day"] = data["Date"].dt.day

    # add day as 0,1,2,3,4,5,6
    # data["day"] = data["Date"].dt.dayofweek
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
    data[NORMALIZE_COLUMNS] = scaler.fit_transform(data[NORMALIZE_COLUMNS])

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

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, 64, 2, batch_first=True, dropout=DROP_PROB)
        self.fc = nn.Linear(64, OUTPUT_SIZE)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

class TRANSFORMER(nn.Module):
    def __init__(self):
        super(TRANSFORMER, self).__init__()
        self.transformer = nn.Transformer(d_model=INPUT_SIZE, nhead=3, dropout=DROP_PROB, batch_first=True)
        self.fc = nn.Linear(INPUT_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        x = self.transformer(x, x)
        x = self.fc(x[:, -1, :])
        return x

def predict_future(model: TRANSFORMER | LSTM | CONV1D | MLP, sequence: np.ndarray, device):
    # The sequence the model should Predict the future of.
    # Needs to be len of SEQ_LEN + PRED_LEN
    # Already Normalized.
    # Prediction should predict PRED_LEN into the future with a sequence of SEQ_LENs

    predictions = []

    with torch.no_grad():
        # Predict the next timestep
        for i in range(PRED_LEN):
            # Ensure sequence length never exceeds SEQ_LEN
            sequence = sequence[-SEQ_LEN:]
            # Convert the sequence to a tensor and add a batch dimension
            x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
            # Get the model's prediction
            y_pred = model(x)
            # Get the actual value
            y_pred = y_pred.cpu().numpy()
            # Append the prediction to the sequence
            np.append(sequence, y_pred)
            # Append the prediction to the list of predictions
            predictions.append(y_pred)


    return predictions

# Trains the given model with the given data
def train(model: TRANSFORMER | LSTM | CONV1D | MLP, loss_fn, optimizer, trainings_data, validation_data, sequence, device):
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


    # Make predictions
    predictions = predict_future(model, sequence, device)


    return train_loss_history, val_loss_history, mae_history, mape_history, predictions

def reverse_transform(data: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    data[["Open", "High", "Low", "Close", "Adj Close", "Volume"]] = scaler.inverse_transform(data[["Open", "High", "Low", "Close", "Adj Close", "Volume"]])
    return data

def reverse_transform_predictions(prediction: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    # Reverse the normalization of the predictions with a dummy
    # DataFrame to use the inverse_transform method
    dummy_df = pd.DataFrame(np.zeros((len(prediction), len(NORMALIZE_COLUMNS))), columns=NORMALIZE_COLUMNS)
    dummy_df["Close"] = prediction

    inverse = scaler.inverse_transform(dummy_df[NORMALIZE_COLUMNS])
    prediction = inverse[:, NORMALIZE_COLUMNS.index("Close")]

    return prediction

def save_predictions(sequence, predictions, path: str):
    # Create a Date object from the day, month, year columns (day = 1 | 2 | 3 | 4 | 5 | 6 | 0)
    sequence["Date"] = pd.to_datetime(sequence[["day", "month", "year"]])
    sequence = sequence.drop(columns=["day", "month", "year"])
    
    # Predictions date are the last PRED_LEN values of the sequence data
    predictions = pd.DataFrame(predictions, columns=["Close"])
    predictions["Date"] = sequence["Date"].iloc[-len(predictions):].values


    # Plot the sequence and predictions
    plt.plot(sequence["Date"], sequence["Close"], label="Sequence", linestyle="dashed")
    plt.plot(predictions["Date"], predictions["Close"], label="Predictions")
    plt.tight_layout()
    plt.legend()
    plt.savefig(path + "predictions.png")
    plt.close()

def save_loss(train_loss, val_loss, path: str):
    # Plot the loss and metrics
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.tight_layout()
    plt.legend()
    plt.savefig(path + "loss.png")
    plt.close()

def save_metrics(mae, mape, path: str):
    # Plot the metrics
    plt.plot(mae, label="Mean Absolute Error")
    plt.plot(mape, label="Mean Absolute Percentage Error")
    plt.tight_layout()
    plt.legend()
    plt.savefig(path + "metrics.png")
    plt.close()

def check_folder(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    path = "data/AAPL.csv"
    data = pd.read_csv(path)
    train_data, val_data, scaler = preprocess_data(data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = MLP().to(device)
    # model = CONV1D().to(device)
    model = LSTM().to(device)
    # model = TRANSFORMER().to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # The sequence the model should Predict the future of.
    # Needs to be len of SEQ_LEN + PRED_LEN
    sequence = train_data[-SEQ_LEN:]

    train_loss_history, val_loss_history, mae_history, mape_history, predictions = train(model, loss_fn, optimizer, train_data, val_data, sequence.values, device)

    # Inverse the normalization of sequence and predictions:
    sequence = reverse_transform(sequence, scaler)
    predictions = reverse_transform_predictions(predictions, scaler)

    folder_name = "MLP"

    check_folder(f"results/{folder_name}")
    save_loss(train_loss_history, val_loss_history, f"results/{folder_name}/")
    save_metrics(mae_history, mape_history, f"results/{folder_name}/")
    save_predictions(sequence, predictions, f"results/{folder_name}/")



if __name__ == "__main__":
    main()