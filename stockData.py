import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

DATA_PATH = "./data/AAPL.csv"

SEQ_LEN_PAST = 200
SEQ_LEN_FUTURE = 3

BATCH_SIZE = 64
EPOCHS = 50
STEPS_PER_EPOCH = 10

VALIDATION_STEPS = 32

NUM_INPUTS = 6

def read_data(file_path: str) -> pd.DataFrame:
    """Reads the data from the given file path and returns a pandas DataFrame."""
    return pd.read_csv(file_path)

def get_data_points(data: pd.DataFrame, batch_size: int, rnd: bool = True) -> pd.DataFrame:
    """Selects a `batch_size` of randome rows from the given DataFrame."""
    start_idx = SEQ_LEN_PAST
    if not rnd:
        return data.iloc[start_idx:start_idx + batch_size]
    else:
        return data.iloc[start_idx::].sample(batch_size)
    
def create_seq_len(raw_data: pd.DataFrame, idx: int, seq_len_past: int, seq_len_future: int) -> pd.DataFrame:
    """Creates a sequence of `seq_len` past points plus the current index `idx`."""
    return raw_data.iloc[idx - seq_len_past:idx + seq_len_future]

def df_to_tensor(df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    """Converts a pandas DataFrame to a torch Tensor."""
    if 'Date' in df.columns:
        df = df.drop(columns=['Date'])

    # Ensure the data is numeric (if there's any issue with type)
    df = df.apply(pd.to_numeric, errors='coerce')  # This will convert all columns to numeric, setting errors to NaN where it can't
    return torch.from_numpy(df.values).float().to(device)

def data_generator(model, device, val=False):
    """Generates data for training the model."""
    raw_data = read_data(DATA_PATH)
    data = get_data_points(raw_data, BATCH_SIZE)

    for idx in data.index:

        if idx < SEQ_LEN_PAST:
            continue

        seq = create_seq_len(raw_data, idx, SEQ_LEN_PAST, SEQ_LEN_FUTURE)
        
        input = df_to_tensor(seq.iloc[:SEQ_LEN_PAST], device)  # First `SEQ_LEN_PAST` rows for input
        label = df_to_tensor(seq.iloc[SEQ_LEN_PAST:], device)  # Next `SEQ_LEN_FUTURE` rows for label

        print(f"\n\n\n Input shape: {input.shape}, Label shape: {label.shape} \n\n\n")


        yield model.preprocess_input(input), label


def plot_seq(seq: pd.DataFrame, save_path: str, save_name: str = "plot.png") -> None:
    """Plots the given sequence."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    seq.loc[:, "Date"] = pd.to_datetime(seq["Date"], errors="coerce")
    seq.set_index("Date", inplace=True)

    # Get all possible first days of the month. They dont need to be the first day of the month, but the first day of the month that is in the data.
    first_month_dates = seq.groupby(seq.index.to_period("M")).head(1).index
    first_month_dates = first_month_dates.append(pd.Index([seq.index[-1]]))

    plt.plot(seq.index, seq['Close'])
    plt.xticks(first_month_dates, first_month_dates.strftime('%Y-%m-%d'), rotation=90)
    plt.ylabel("Close Price")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, save_name))
    plt.close()

def train(model, loss_fn, optimizer, device: torch.device):
    train_data = data_generator(model, device)
    val_data = data_generator(model, device, val=True)

    train_loss_history = []
    validation_loss_history = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for step in tqdm(range(STEPS_PER_EPOCH)):
            input, label = next(train_data)
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss_history.append(train_loss / STEPS_PER_EPOCH)

        #val_plot_checkpoint(model, SEQ_LEN_PAST, SEQ_LEN_FUTURE, name, epoch)

        model.eval()
        validation_loss = 0
        for step in range(VALIDATION_STEPS):
            input, label = next(val_data)
            output = model(input)
            loss = loss_fn(output, label)
            validation_loss += loss.item()
        validation_loss_history.append(validation_loss / VALIDATION_STEPS)

        print(f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {train_loss_history[-1]} - Validation Loss: {validation_loss_history[-1]}")


        #save_loss_history_as_image_plot(train_loss_history, validation_loss_history, name)


class own_MLP(nn.Module):
    def __init__(self, layer: list[int], dp=0.5):
        super(own_MLP, self).__init__()
        self.layer = layer
        self.dp = dp
        
        self.fc_layers = nn.ModuleList()
        
        self.fc_layers.append(nn.Linear(NUM_INPUTS, self.layer[0]))
        
        for i in range(len(self.layer) - 1):
            self.fc_layers.append(nn.Linear(self.layer[i], self.layer[i + 1]))
        
        self.fc_layers.append(nn.Linear(self.layer[-1], SEQ_LEN_FUTURE))
        self.dropout = nn.Dropout(self.dp)

    def forward(self, x):
        x = self.fc_layers[0](x)
        x = torch.relu(x)
        
        for i in range(1, len(self.fc_layers) - 1):
            x = self.fc_layers[i](x)
            x = torch.relu(x)
            x = self.dropout(x)
        
        # Final output layer
        x = self.fc_layers[-1](x)
        return x
    
    def preprocess_input(self, x):
        return x


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = own_MLP(layer=[256, 128, 64])

    model = model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, loss_fn, optimizer, device)

if __name__ == "__main__":
    main()