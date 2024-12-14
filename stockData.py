import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# Constants
DATA_PATH = "./data/AAPL.csv"
SEQ_LEN_PAST = 200
SEQ_LEN_FUTURE = 3
BATCH_SIZE = 256
EPOCHS = 100
STEPS_PER_EPOCH = 100
NUM_INPUTS = 6

# Data Preparation
def read_data(file_path: str) -> pd.DataFrame:
    """Reads the CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)

def prepare_sequence(data: pd.DataFrame, seq_len_past: int, seq_len_future: int):
    """Creates sequences of past and future data for training."""
    i = np.random.randint(seq_len_past, len(data) - seq_len_future)

    input_seq = data.iloc[i - seq_len_past:i].values
    output_seq = data.iloc[i:i + seq_len_future].values

    return input_seq, output_seq

def data_generator(data, batch_size, device):
    """ Generates batches of data for training and/or validataion."""
    while True:

        input, label = prepare_sequence(data, SEQ_LEN_PAST, SEQ_LEN_FUTURE)

        input_tensor = torch.tensor(input, dtype=torch.float32).to(device)
        label_tensor = torch.tensor(label, dtype=torch.float32).to(device)

        yield input_tensor, label_tensor

def test_data_generator(data, device):
    """Returns one batch of data and the label for testing the prediction of the model"""
    # Randome Index inside the data
    sample_idx = np.random.randint(SEQ_LEN_PAST, len(data) - SEQ_LEN_FUTURE)
    sequences = data[sample_idx - SEQ_LEN_PAST:sample_idx]
    labels_with_date = data[sample_idx-SEQ_LEN_PAST:sample_idx + SEQ_LEN_FUTURE]
    sequences = sequences.drop(columns=["Date"])  # Exclude the Date column
    sequences = sequences.apply(pd.to_numeric, errors='coerce').dropna()  # Ensure numeric and drop NaNs
    sequences = sequences.values
    input_batch = torch.tensor(sequences, dtype=torch.float32).to(device)
    return input_batch, labels_with_date


# Plotting
def plot_loss(train_loss, val_loss, save_path):
    """Plots training and validation loss."""
    plt.figure()
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_path, "loss_curve.png"))
    plt.close()

def plot_seq(data_seq: pd.DataFrame, prediction, save_path: str, save_name: str = "plot.png") -> None:
    """Plots the given sequence along with predictions."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #print(data_seq["Close"])

    data_seq.loc[:, "Date"] = pd.to_datetime(data_seq["Date"], errors="coerce")
    data_seq.set_index("Date", inplace=True)

    # Get all possible first days of the month. They don’t need to be the first day of the month, but the first day of the month that is in the data.
    first_month_dates = data_seq.groupby(data_seq.index.to_period("M")).head(1).index
    first_month_dates = first_month_dates.append(pd.Index([data_seq.index[-1]]))

    plt.figure(figsize=(12, 6))
    plt.plot(data_seq.index, data_seq['Close'], label="Ground Truth", color="blue")
    
    # Prediction shape (3, 6), I only need the closing price right now. The values are: Open,High,Low,Close,Adj Close,Volume.
    # It missis the date, so I will just add the date from the data_seq (it always the next possible date (monday-friday))
    last_date = data_seq.index[-1]
    next_three_date = []

    # check if next date is saturday or sunday
    for i in range(1, 4):
        next_date = last_date + pd.DateOffset(days=i)
        if next_date.weekday() == 5:
            next_date = next_date + pd.DateOffset(days=2)
        elif next_date.weekday() == 6:
            next_date = next_date + pd.DateOffset(days=1)
        next_three_date.append(next_date)
    
    # create a df with the date and the prediction
    prediction_df = pd.DataFrame(prediction, columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"])
    prediction_df["Date"] = next_three_date

    #print("\n\n\n\n")
    #print(prediction_df)
    #print("\n\n\n\n")

    # Plot the prediction
    plt.plot(prediction_df["Date"], prediction_df['Close'], label="Prediction", color="red")

    plt.xticks(first_month_dates, first_month_dates.strftime('%Y-%m-%d'), rotation=90)
    
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, save_name))
    plt.close()

# MLP Model
class own_MLP(nn.Module):
    def __init__(self, layer: list[int], dp=0.5):
        super(own_MLP, self).__init__()
        self.layer = layer
        self.dp = dp

        self.fc_layers = nn.ModuleList()

        # Input layer
        self.fc_layers.append(nn.Linear(SEQ_LEN_PAST * NUM_INPUTS, self.layer[0]))

        # Hidden layers
        for i in range(len(self.layer) - 1):
            self.fc_layers.append(nn.Linear(self.layer[i], self.layer[i + 1]))

        # Output layer
        self.fc_layers.append(nn.Linear(self.layer[-1], SEQ_LEN_FUTURE * NUM_INPUTS))
        self.dropout = nn.Dropout(self.dp)

    def forward(self, x):
        for i in range(len(self.fc_layers) - 1):
            x = self.fc_layers[i](x)
            x = torch.relu(x)
            x = self.dropout(x)
        x = self.fc_layers[-1](x)
        return x


# Training
def train(model, loss_fn, optimizer, device, data, val_data, raw_data):
    """Train the model."""
    train_loss_history = []
    val_loss_history = []


    train_data_gen = data_generator(data, BATCH_SIZE, device)
    val_data_gen = data_generator(val_data, BATCH_SIZE, device)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for _ in tqdm(range(STEPS_PER_EPOCH)):
            inputs, labels = next(train_data_gen)
            optimizer.zero_grad()
            print(f"\n\n\n {inputs.shape} \n\n\n")
            output = model(inputs)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss_history.append(train_loss / STEPS_PER_EPOCH)

        #Val_plot
        #plot_seq(label_with_date, prediction, "./plots", f"pred_epoch_{epoch}.png")

        # Validation
        model.eval()
        val_loss = 0
        for _ in range(STEPS_PER_EPOCH):
            inputs, labels = next(val_data_gen)
            output = model(inputs)
            loss = loss_fn(output, labels)
            val_loss += loss.item()
        val_loss_history.append(val_loss / STEPS_PER_EPOCH)

        print(f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}")

        # Plot loss curves
        plot_loss(train_loss_history, val_loss_history, "./plots")



# Main Function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Read and prepare data
    raw_data = read_data(DATA_PATH)
    data = raw_data.drop(columns=["Date"])  # Exclude the Date column
    data = data.apply(pd.to_numeric, errors='coerce').dropna()  # Ensure numeric and drop NaNs

    # split data into train and validation
    split_pct = 0.8
    train_data = data.iloc[:int(len(data) * split_pct)]
    val_data = data.iloc[int(len(data) * split_pct):]


    # Initialize model, loss, and optimizer
    model = own_MLP(layer=[256, 128, 64]).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train(model, loss_fn, optimizer, device, train_data, val_data, raw_data)


if __name__ == "__main__":
    main()
