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

CHOOSSEN_ATTR = "High"
INPUTS_COLUMNS = ["High", "day", "month", "year"]
NORMALIZE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
OUTPUT_COLUMNS = ["High"]
INPUT_SIZE = len(INPUTS_COLUMNS)
OUTPUT_SIZE = len(OUTPUT_COLUMNS)

EPOCHS = 3
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

    # add month, year columns
    data["month"] = data["Date"].dt.month
    data["year"] = data["Date"].dt.year

    # Keep original date for plotting
    original_dates = data["Date"].copy()

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

    # Store the dates corresponding to the training and validation data
    train_dates = original_dates[:int(len(data) * split_pct)]
    val_dates = original_dates[int(len(data) * split_pct):]

    return train_data, val_data, scaler

def data_generator(data: pd.DataFrame, seq_len: int, batch_size: int, device: torch.device):
    # Extract only the columns we want to use as input
    input_data = data[INPUTS_COLUMNS].to_numpy()
    
    n = len(input_data) - seq_len
    output_indices = [INPUTS_COLUMNS.index(col) for col in OUTPUT_COLUMNS]

    for i in range(0, n, batch_size):
        x = []
        y = []
        for j in range(min(batch_size, n-i)):
            x.append(input_data[i+j:i+j+seq_len])
            y.append(input_data[i+j+seq_len, output_indices])  # Get target for the next timestep after the sequence

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
        self.transformer = nn.Transformer(d_model=INPUT_SIZE, nhead=2, dropout=DROP_PROB, batch_first=True)
        self.fc = nn.Linear(INPUT_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        x = self.transformer(x, x)
        x = self.fc(x[:, -1, :])
        return x

def predict_future(model: TRANSFORMER | LSTM | CONV1D | MLP, sequence: np.ndarray, full_data: pd.DataFrame, device):
    """
    Predict future values based on the given sequence.
    
    Args:
        model: The trained model
        sequence: Input sequence containing only the INPUTS_COLUMNS
        full_data: The full DataFrame containing all columns to copy other attributes
        device: The compute device
        
    Returns:
        Array of predicted values
    """
    current_sequence = sequence.copy()
    predictions = []
    # Get the last row from the full data to use for non-predicted columns
    last_full_row = full_data.iloc[-1].copy()
    
    with torch.no_grad():
        # Predict PRED_LEN steps into the future
        for i in range(PRED_LEN):
            # Use last SEQ_LEN elements of the current sequence
            input_seq = current_sequence[-SEQ_LEN:]
            
            # Convert to tensor and add batch dimension
            x = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Get prediction for Close price
            y_pred = model(x).cpu().numpy()[0]
            
            # Create a new row with the predicted values
            new_row = np.zeros(current_sequence.shape[1])
            
            # Set predicted Close value
            close_idx = INPUTS_COLUMNS.index(CHOOSSEN_ATTR)
            new_row[close_idx] = y_pred[0]
            
            # Set time features (day/month/year)
            day_idx = INPUTS_COLUMNS.index("day")
            month_idx = INPUTS_COLUMNS.index("month")
            year_idx = INPUTS_COLUMNS.index("year")
            
            # Increment the date (simplified approach)
            new_row[day_idx] = (current_sequence[-1, day_idx] + 1) % 31
            if new_row[day_idx] == 0:
                new_row[day_idx] = 1
                new_row[month_idx] = (current_sequence[-1, month_idx] + 1) % 13
                if new_row[month_idx] == 0:
                    new_row[month_idx] = 1
                    new_row[year_idx] = current_sequence[-1, year_idx] + 1
                else:
                    new_row[year_idx] = current_sequence[-1, year_idx]
            else:
                new_row[month_idx] = current_sequence[-1, month_idx]
                new_row[year_idx] = current_sequence[-1, year_idx]
            
            # Append to sequence and save prediction
            current_sequence = np.vstack([current_sequence, new_row])
            predictions.append(y_pred[0])

    return np.array(predictions)

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

    # Get only the input features columns from the sequence
    input_sequence = sequence[INPUTS_COLUMNS].values
    
    # Make predictions
    predictions = predict_future(model, input_sequence, sequence, device)

    return train_loss_history, val_loss_history, mae_history, mape_history, predictions

def reverse_transform(data: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    data[NORMALIZE_COLUMNS] = scaler.inverse_transform(data[NORMALIZE_COLUMNS])
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
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.tight_layout()
    plt.legend()
    plt.savefig(path + "loss.png")
    plt.close()

def save_metrics(mae, mape, path: str):
    # Plot the metrics
    plt.figure(figsize=(10, 6))
    plt.plot(mae, label="Mean Absolute Error")
    plt.plot(mape, label="Mean Absolute Percentage Error")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Training Metrics")
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

    # Select one of the model types
    # model = MLP().to(device)
    # model = CONV1D().to(device)
    model = LSTM().to(device)
    # model = TRANSFORMER().to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # The sequence the model should predict the future of
    sequence = train_data[-SEQ_LEN-PRED_LEN:]

    train_loss_history, val_loss_history, mae_history, mape_history, predictions = train(
        model, loss_fn, optimizer, train_data, val_data, sequence, device
    )

    # Inverse the normalization of sequence and predictions
    sequence = reverse_transform(sequence, scaler)
    predictions = reverse_transform_predictions(predictions, scaler)

    # Change this to the model type you're using
    folder_name = model.__class__.__name__.lower()

    check_folder(f"resultsSingelAttr/{folder_name}")
    save_loss(train_loss_history, val_loss_history, f"resultsSingelAttr/{folder_name}/")
    save_metrics(mae_history, mape_history, f"resultsSingelAttr/{folder_name}/")
    save_predictions(sequence, predictions, f"resultsSingelAttr/{folder_name}/")

if __name__ == "__main__":
    main()