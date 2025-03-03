import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

# These can be modified as needed
# INPUTS_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "day", "month", "year"]
INPUTS_COLUMNS = ["High", "day", "month", "year"]
# NORMALIZE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
NORMALIZE_COLUMNS = ["High"]
OUTPUT_COLUMNS = ["High"]
# Dynamically set based on columns
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
def preprocess_data(data: pd.DataFrame, inputs_columns: List[str], normalize_columns: List[str], 
                    output_columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    processed_data = data.copy()
    
    # convert date to datetime object if it exists
    if "Date" in processed_data.columns:
        processed_data["Date"] = pd.to_datetime(processed_data["Date"])
        
        # Only add date components if they're in the inputs
        if "day" in inputs_columns:
            processed_data["day"] = processed_data["Date"].dt.day
        if "month" in inputs_columns:
            processed_data["month"] = processed_data["Date"].dt.month
        if "year" in inputs_columns:
            processed_data["year"] = processed_data["Date"].dt.year
            
        # Filter by year if year column exists
        if "year" in processed_data.columns:
            processed_data = processed_data[processed_data["year"] <= 2021]
            
        # drop date column after extracting components
        processed_data = processed_data.drop(columns=["Date"])

    # Ensure all required columns exist
    for col in normalize_columns + inputs_columns + output_columns:
        if col not in processed_data.columns and col not in ["day", "month", "year"]:
            raise ValueError(f"Column {col} not found in data")

    # normalize the data
    scaler = StandardScaler()
    # Only normalize columns that exist
    normalize_cols = [col for col in normalize_columns if col in processed_data.columns]
    if normalize_cols:
        processed_data[normalize_cols] = scaler.fit_transform(processed_data[normalize_cols])

    # Only keep the columns we need
    all_needed_columns = list(set(inputs_columns + output_columns))
    processed_data = processed_data[all_needed_columns]

    # split into validation and training data
    split_pct = 0.8
    train_data = processed_data.iloc[:int(len(processed_data) * split_pct)]
    val_data = processed_data.iloc[int(len(processed_data) * split_pct):]

    return train_data, val_data, scaler

def data_generator(data: pd.DataFrame, seq_len: int, batch_size: int, device: torch.device, 
                    inputs_columns: List[str], output_columns: List[str]):
    # Map column names to indices in the numpy array
    col_to_idx = {col: i for i, col in enumerate(data.columns)}
    output_indices = [col_to_idx[col] for col in output_columns]
    
    data = data.to_numpy()
    n = len(data) - seq_len
    
    for i in range(0, n, batch_size):
        x = []
        y = []
        for j in range(min(batch_size, n-i)):
            x.append(data[i+j:i+j+seq_len])
            y.append(data[i+j+seq_len, output_indices])  # Get target for the next timestep after the sequence

        yield torch.tensor(np.array(x), dtype=torch.float32).to(device), \
              torch.tensor(np.array(y), dtype=torch.float32).to(device)

class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)
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
    def __init__(self, input_size: int, output_size: int):
        super(CONV1D, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, 3, padding=1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, output_size)
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
    def __init__(self, input_size: int, output_size: int):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, 128, 3, batch_first=True, dropout=DROP_PROB)
        self.fc = nn.Linear(128, output_size)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

class TRANSFORMER(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(TRANSFORMER, self).__init__()
        # Use max(1, min(input_size, 8)) for n_head to ensure compatibility
        n_head = max(1, min(3, input_size // 4))
        self.transformer = nn.Transformer(d_model=input_size, nhead=n_head, dropout=DROP_PROB, batch_first=True)
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        x = self.transformer(x, x)
        x = self.fc(x[:, -1, :])
        return x

def predict_future(model, sequence: np.ndarray, device, pred_len: int, seq_len: int):
    predictions = []

    with torch.no_grad():
        # Make a copy of the sequence to avoid modifying the original
        current_seq = sequence.copy()
        
        for i in range(pred_len):
            # Ensure sequence length never exceeds seq_len
            current_seq = current_seq[-seq_len:]
            # Convert the sequence to a tensor and add a batch dimension
            x = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0).to(device)
            # Get the model's prediction
            y_pred = model(x)
            # Convert prediction to numpy
            pred = y_pred.cpu().numpy()[0]
            # Append the prediction to the sequence
            # Create a row with the same shape as current_seq rows
            new_row = np.zeros((1, current_seq.shape[1]))
            # Place the prediction values in the right column positions
            # For now, we only update the output columns
            new_row[0, -len(pred):] = pred
            # Append new row to current sequence
            current_seq = np.append(current_seq, new_row, axis=0)
            # Append the prediction to the list of predictions
            predictions.append(pred)

    return np.array(predictions).squeeze()

def train(model, loss_fn, optimizer, trainings_data, validation_data, sequence, device,
          epochs: int, steps_per_epoch: int, seq_len: int, batch_size: int, pred_len: int,
          inputs_columns: List[str], output_columns: List[str]):
    train_loss_history = []
    val_loss_history = []
    mae_history = []
    mape_history = []

    train_data_gen = data_generator(trainings_data, seq_len, batch_size, device, inputs_columns, output_columns)
    val_data_gen = data_generator(validation_data, seq_len, batch_size, device, inputs_columns, output_columns)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_mae = 0
        train_mape = 0

        for _ in tqdm(range(steps_per_epoch)):
            try:
                inputs, labels = next(train_data_gen)
            except StopIteration:
                # Restart the generator if we run out of data
                train_data_gen = data_generator(trainings_data, seq_len, batch_size, device, inputs_columns, output_columns)
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
            train_mape += mae / (torch.mean(torch.abs(labels)).item() + 1e-8)  # Added small epsilon to prevent division by zero

        train_loss /= steps_per_epoch
        train_mae /= steps_per_epoch
        train_mape /= steps_per_epoch
        mae_history.append(train_mae)
        mape_history.append(train_mape)
        train_loss_history.append(train_loss)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for _ in range(steps_per_epoch):
                try:
                    inputs, labels = next(val_data_gen)
                except StopIteration:
                    # Restart the generator if we run out of data
                    val_data_gen = data_generator(validation_data, seq_len, batch_size, device, inputs_columns, output_columns)
                    inputs, labels = next(val_data_gen)

                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

        val_loss /= steps_per_epoch
        val_loss_history.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, MAE: {train_mae:.4f}, MAPE: {train_mape:.4f}")

    # Make predictions
    predictions = predict_future(model, sequence, device, pred_len, seq_len)

    return train_loss_history, val_loss_history, mae_history, mape_history, predictions

def reverse_transform(data: pd.DataFrame, scaler: StandardScaler, normalize_columns: List[str]) -> pd.DataFrame:
    # Only transform columns that exist in both data and normalize_columns
    cols_to_transform = [col for col in normalize_columns if col in data.columns]
    if cols_to_transform:
        data[cols_to_transform] = scaler.inverse_transform(data[cols_to_transform])
    return data

def reverse_transform_predictions(prediction: np.ndarray, scaler: StandardScaler, 
                                   normalize_columns: List[str], output_columns: List[str]) -> np.ndarray:
    # Check if output columns are in normalize columns
    output_in_norm = [col for col in output_columns if col in normalize_columns]
    
    if not output_in_norm:
        # If no output columns are normalized, just return the prediction
        return prediction
    
    # Create a dummy DataFrame with zeros for all normalized columns
    dummy_df = pd.DataFrame(np.zeros((len(prediction), len(normalize_columns))), 
                           columns=normalize_columns)
    
    # Place the predictions in the corresponding columns
    for i, col in enumerate(output_columns):
        if col in normalize_columns:
            norm_idx = normalize_columns.index(col)
            dummy_df.iloc[:, norm_idx] = prediction[:, i] if prediction.ndim > 1 else prediction
    
    # Apply inverse transform
    inverse = scaler.inverse_transform(dummy_df)
    
    # Extract the relevant columns
    result = np.zeros((len(prediction), len(output_columns)))
    for i, col in enumerate(output_columns):
        if col in normalize_columns:
            norm_idx = normalize_columns.index(col)
            result[:, i] = inverse[:, norm_idx]
        else:
            result[:, i] = prediction[:, i] if prediction.ndim > 1 else prediction
    
    return result.squeeze()

def save_predictions(sequence, predictions, output_columns, path: str):
    # Create a copy to avoid modifying the original
    sequence_df = sequence.copy()
    
    # Reconstruct date if day, month, year columns exist
    if all(col in sequence_df.columns for col in ["day", "month", "year"]):
        sequence_df["Date"] = pd.to_datetime(sequence_df[["year", "month", "day"]])
        
        # Create a predictions DataFrame
        predictions_df = pd.DataFrame(predictions, 
                                     columns=output_columns)
        
        # Add dates for predictions (last PRED_LEN days after sequence)
        last_date = sequence_df["Date"].iloc[-1]
        pred_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(predictions_df))
        predictions_df["Date"] = pred_dates
        
        # Plot each output column
        for col in output_columns:
            if col in sequence_df.columns and col in predictions_df.columns:
                plt.figure(figsize=(12, 6))
                plt.title(f"{col} Prediction")
                plt.plot(sequence_df["Date"], sequence_df[col], label="Historical", linestyle="dashed")
                plt.plot(predictions_df["Date"], predictions_df[col], label="Predictions")
                plt.tight_layout()
                plt.legend()
                plt.savefig(f"{path}predictions.png")
                plt.close()
    else:
        # If date components are not available, use simple indices
        plt.figure(figsize=(12, 6))
        for i, col in enumerate(output_columns):
            if col in sequence_df.columns:
                plt.plot(range(len(sequence_df)), sequence_df[col], label=f"Historical {col}", linestyle="dashed")
                plt.plot(range(len(sequence_df), len(sequence_df) + len(predictions)), 
                        predictions[:, i] if predictions.ndim > 1 else predictions, 
                        label=f"Predictions {col}")
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"{path}predictions.png")
        plt.close()

def save_loss(train_loss, val_loss, path: str):
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"{path}loss.png")
    plt.close()

def save_metrics(mae, mape, path: str):
    plt.figure(figsize=(10, 6))
    plt.plot(mae, label="Mean Absolute Error")
    plt.plot(mape, label="Mean Absolute Percentage Error")
    plt.title("Training Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"{path}metrics.png")
    plt.close()

def check_folder(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    # These can be customized at runtime
    inputs_columns = INPUTS_COLUMNS
    normalize_columns = NORMALIZE_COLUMNS
    output_columns = OUTPUT_COLUMNS
    model_type = "TRANSFORMER"  # Options: "MLP", "CONV1D", "LSTM", "TRANSFORMER"
    
    # Dynamically set based on columns
    input_size = len(inputs_columns)
    output_size = len(output_columns)
    
    path = "data/AAPL.csv"
    data = pd.read_csv(path)
    
    # Process data with the specified columns
    train_data, val_data, scaler = preprocess_data(data, inputs_columns, normalize_columns, output_columns)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the model based on type
    if model_type == "MLP":
        model = MLP(input_size, output_size).to(device)
    elif model_type == "CONV1D":
        model = CONV1D(input_size, output_size).to(device)
    elif model_type == "LSTM":
        model = LSTM(input_size, output_size).to(device)
    elif model_type == "TRANSFORMER":
        model = TRANSFORMER(input_size, output_size).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # The sequence the model should Predict the future of
    sequence = train_data.iloc[-SEQ_LEN:].copy()

    train_loss_history, val_loss_history, mae_history, mape_history, predictions = train(
        model, loss_fn, optimizer, train_data, val_data, sequence.values, device,
        EPOCHS, STEPS_PER_EPOCH, SEQ_LEN, BATCH_SIZE, PRED_LEN,
        inputs_columns, output_columns
    )

    # Inverse the normalization of sequence and predictions
    sequence = reverse_transform(sequence, scaler, normalize_columns)
    predictions = reverse_transform_predictions(predictions, scaler, normalize_columns, output_columns)

    folder_name = model_type
    results_path = f"results/{folder_name}/"
    check_folder(results_path)
    
    save_loss(train_loss_history, val_loss_history, results_path)
    save_metrics(mae_history, mape_history, results_path)
    save_predictions(sequence, predictions, output_columns, results_path)
    
    print(f"Training complete. Results saved to {results_path}")

if __name__ == "__main__":
    main()