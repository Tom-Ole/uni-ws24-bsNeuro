import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


DATA_PATH = "./data/AAPL.csv"
SEQ_LEN_PAST = 64
SEQ_LEN_FUTURE = 3
BATCH_SIZE = 64
EPOCHS = 500
STEPS_PER_EPOCH = 500

NUM_INPUTS = 7 - 6

CHOOSEN_COLUMN = "Close"

LAYERS = [64, 128, 64] 
DP = 0.5
LR = 5e-4

def read_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def prepare_sequence(data: pd.DataFrame, seq_len_past: int, seq_len_future: int):
    i = np.random.randint(seq_len_past, len(data) - seq_len_future)
    input_seq = data.iloc[i - seq_len_past:i].values
    output_seq = data.iloc[i:i + seq_len_future].values
    return input_seq, output_seq

def data_generator(data, batch_size, device):
    while True:
        inputs, labels = zip(*(prepare_sequence(data, SEQ_LEN_PAST, SEQ_LEN_FUTURE) for _ in range(batch_size)))
        input_tensor = torch.tensor(np.array(inputs), dtype=torch.float32).view(batch_size, -1).to(device)
        label_tensor = torch.tensor(np.array(labels), dtype=torch.float32).view(batch_size, -1).to(device)
        yield input_tensor, label_tensor

def plot_loss(train_loss, val_loss, save_path):
    plt.figure()
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.savefig(os.path.join(save_path, "loss_curve.png"))
    plt.close()


def plot_prediction(input_data, ground_truth, prediction, future_dates, save_path, epoch):
    """Plots the prediction and corresponding ground truth dynamically based on random input."""

    if epoch % 10 != 0 and epoch != 1:
        return

    input_data = input_data.copy()
    input_data["Date"] = pd.to_datetime(input_data["Date"], errors="coerce")
    input_data.set_index("Date", inplace=True)

    future_dates = pd.to_datetime(future_dates, errors="coerce")

    plt.figure(figsize=(12, 6))
    # Plot last known values (ground truth up to the random sequence)
    plt.plot(input_data.index, input_data[CHOOSEN_COLUMN], label="Input Data", color="blue", alpha=0.5)
    plt.plot(future_dates, ground_truth, label="Ground Truth", color="green", linestyle="dashed")
    plt.plot(future_dates, prediction[:, 0], label="Prediction", color="red")
    plt.scatter(future_dates, prediction[:, 0], label="Prediction", color="orange")
    # plt.plot(future_dates, prediction[:, 3], label="Prediction", color="red")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title(f"Prediction vs Ground Truth (Epoch {epoch})")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.savefig(os.path.join(save_path, f"prediction_epoch_{epoch}.png"))
    plt.close()

class own_MLP(nn.Module):
    def __init__(self):
        super(own_MLP, self).__init__()
        self.input_layer = nn.Linear(SEQ_LEN_PAST * NUM_INPUTS, LAYERS[0])
        self.hidden_layers = nn.ModuleList(
            nn.Linear(LAYERS[i], LAYERS[i + 1]) for i in range(len(LAYERS) - 1)
        )
        self.output_layer = nn.Linear(LAYERS[-1], SEQ_LEN_FUTURE * NUM_INPUTS)
        self.dropout = nn.Dropout(DP)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = self.dropout(x)
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
            x = self.dropout(x)
        x = self.output_layer(x)
        return x

class own_LSTM(nn.Module):
    def __init__(self):
        super(own_LSTM, self).__init__()
        hidden_size = 256
        self.lstm = nn.LSTM(NUM_INPUTS * SEQ_LEN_PAST, hidden_size, 3, batch_first=True, dropout=DP)
        self.fc = nn.Linear(hidden_size, SEQ_LEN_FUTURE * NUM_INPUTS)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

class own_transformer(nn.Module):
    def __init__(self):
        super(own_transformer, self).__init__()
        #self.position_encoder = nn.Parameter(torch.zeros(BATCH_SIZE, SEQ_LEN_PAST, NUM_INPUTS)) # Shape: (SEQ_LEN_PAST, 1)
        self.transformer = nn.TransformerEncoderLayer(d_model=SEQ_LEN_PAST, nhead=4, dropout=DP, batch_first=True)
        self.fc = nn.Linear(SEQ_LEN_PAST, SEQ_LEN_FUTURE * NUM_INPUTS)

    def forward(self, x):
        #x = x + self.position_encoder
        x = self.transformer(x)
        x = self.fc(x)
        return x

def train(model, loss_fn, optimizer, schedular, device, data, val_data, raw_data):
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
            output = model(inputs)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss_history.append(train_loss / STEPS_PER_EPOCH)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for _ in range(STEPS_PER_EPOCH):
                inputs, labels = next(val_data_gen)
                output = model(inputs)
                loss = loss_fn(output, labels)
                val_loss += loss.item()
        val_loss_history.append(val_loss / STEPS_PER_EPOCH)


        if schedular is not None:
            schedular.step()

        print(f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {train_loss / STEPS_PER_EPOCH:.4f} - Validation Loss: {val_loss / STEPS_PER_EPOCH:.4f}")
        plot_loss(train_loss_history, val_loss_history, f"./losses/{type(model).__name__}")

        with torch.no_grad():
            # Select a random sequence for prediction
            random_index = np.random.randint(SEQ_LEN_PAST, len(raw_data) - SEQ_LEN_FUTURE)
            raw_sample_input = raw_data.iloc[random_index - SEQ_LEN_PAST:random_index]
            sample_input = drop_columns(raw_sample_input).values
            sample_input = torch.tensor(sample_input, dtype=torch.float32).reshape(1, -1).to(device)
            prediction = model(sample_input).cpu().numpy().reshape(SEQ_LEN_FUTURE, -1)

            # Ground truth for plotting
            future_df = raw_data.iloc[random_index:random_index + SEQ_LEN_FUTURE]
            ground_truth = future_df[CHOOSEN_COLUMN].values
            future_dates = future_df["Date"]

            plot_prediction(raw_sample_input, ground_truth, prediction, future_dates, f"./predictions/{type(model).__name__}", epoch + 1)

def drop_columns(data):
    possible_col = ["Date","Open","High","Low", "Close","Adj Close","Volume"]
    possible_col.remove(CHOOSEN_COLUMN)
    return data.drop(columns=possible_col)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    raw_data = read_data(DATA_PATH)
    #print(type(raw_data[CHOOSEN_COLUMN][0]))

    # reduce noise
    raw_data = raw_data[:6000]

    data = drop_columns(raw_data)

    scaler = StandardScaler()
    data[[CHOOSEN_COLUMN]] = scaler.fit_transform(data[[CHOOSEN_COLUMN]])

    split_pct = 0.8
    train_data = data.iloc[:int(len(data) * split_pct)]
    val_data = data.iloc[int(len(data) * split_pct):]

    model = own_MLP().to(device)
    # model = own_LSTM().to(device)
    # model = own_transformer().to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)


    train(model, loss_fn, optimizer, scheduler, device, train_data, val_data, raw_data)

    # MAYBE?: Avg prediction from all models?

if __name__ == "__main__":
    main()
