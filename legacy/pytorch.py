import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

SEQ_LEN_PAST = 30
SEQ_LEN_FUTURE = 3
HIDDEN_SIZE = 128

BATCH_SIZE = 64
EPOCHS = 50
STEPS_PER_EPOCH = 10

VALIDATION_STEPS = 32

NUM_INPUT_PARAMETERS = 1
NUM_OUTPUT_PARAMETERS = 1

def parse_file(file_name):
    with open(file_name, "r") as file:
        lines = file.readlines()
        samples = []
        for line in lines:
            _, y = line.strip().split(' ')
            samples.append((float(y),))
        return samples

def select_data(batch_size, all_files):
    selected_inputs = []
    selected_labels = []

    num = 0
    while num < batch_size:
        idx_file = random.randint(0, len(all_files) - 1)
        samples = parse_file(all_files[idx_file])

        idx_seq = random.randint(SEQ_LEN_PAST, len(samples) - SEQ_LEN_FUTURE)
        sub_seq_input = samples[idx_seq - SEQ_LEN_PAST:idx_seq]
        sub_seq_label = samples[idx_seq:idx_seq + SEQ_LEN_FUTURE]

        selected_inputs.append(sub_seq_input)
        selected_labels.append(sub_seq_label)

        num += 1
    return np.asarray(selected_inputs), np.asarray(selected_labels)

def data_generator(path, batch_size, model, device):
    all_files = sorted(glob.glob(path + '*.txt'))
    print(f"Found {len(all_files)} files in {path}")

    if not all_files:
        raise FileNotFoundError(f"No .txt files found in the directory: {path}")

    while True:
        inputs, labels = select_data(batch_size, all_files)

        mu = 0
        sigma = 0.01
        rnd = np.random.normal(mu, sigma, size=inputs.shape)
        inputs = inputs + rnd 

        # Convert to PyTorch tensors
        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        labels = torch.tensor(labels, dtype=torch.float32).to(device)

        inputs = model.preprocess_input(inputs)

        yield inputs, labels

def save_loss_history_as_image_plot(train_loss_history, validate_loss_history, save_dir):
    save_path = os.path.join(f"./prediction/{save_dir}/loss")
    os.makedirs(save_path, exist_ok=True)


    min_loss = min(min(train_loss_history), min(validate_loss_history))
    max_loss = max(max(train_loss_history), max(validate_loss_history))
    margin = 0.1 * (max_loss - min_loss)  # Add 10% margin
    ylim = (min_loss - margin, max_loss + margin)


    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(validate_loss_history, label='Validation Loss')
    plt.title("Training and Validation Loss History")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.ylim(ylim)
    plot_file = os.path.join(save_path, f"loss.png")
    plt.savefig(plot_file)
    plt.close()

def plot_partial(samples, sub_seq_input, sub_seq_label, sub_seq_pred, epoch, save_dir):
    save_path = os.path.join(f"./prediction/{save_dir}/partial")
    os.makedirs(save_path, exist_ok=True)

    input_samples_y = [point[0] for point in sub_seq_input]
    label_samples_y = [point[0] for point in sub_seq_label]
    pred_samples_y = [point[0] for point in sub_seq_pred]

    input_indices = list(range(len(input_samples_y)))
    label_indices = list(range(len(input_samples_y), len(input_samples_y) + len(label_samples_y)))
    pred_indices = label_indices 

    plt.figure(figsize=(10, 6))
    plt.scatter(input_indices, input_samples_y, label="Input (Past)", color='blue')
    plt.scatter(pred_indices, pred_samples_y, label="Prediction (Future)", color='orange')
    plt.scatter(label_indices, label_samples_y, label="Ground Truth (Future)", color='green')
    plt.title(f"Prediction vs Ground Truth (Epoch: {epoch})")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend(loc="upper left")
    plt.grid(True)
    plot_file = os.path.join(save_path, f"partial_plot_epoch_{epoch}.png")
    plt.savefig(plot_file)
    plt.close()

def plot_unrolled(samples, model, epoch, save_dir):
    save_path = os.path.join(f"./prediction/{save_dir}/unrolled")
    os.makedirs(save_path, exist_ok=True)

    rolling_buffer = samples[0:SEQ_LEN_PAST]
    initial_samples = rolling_buffer.copy()

    num_steps = 300  

    pred_samples = []
    for i in range(num_steps):
        input_seq = np.expand_dims(np.asarray(rolling_buffer), axis=0)        

        input_tensor = torch.tensor(input_seq, dtype=torch.float32).to(next(model.parameters()).device)
        input_tensor = model.preprocess_input(input_tensor)
        preds = model(input_tensor).detach().cpu().numpy()
        preds = preds.astype(np.float32)

        pred_samples.append((preds[0][0],))
        rolling_buffer.append((preds[0][0],))

        rolling_buffer = rolling_buffer[1:]

    y_pred = [pred[0] for pred in pred_samples]
    y_initial = [sample[0] for sample in initial_samples]
    y_samples = [sample[0] for sample in samples]

    tmp1 = list(range(0, len(y_samples)))
    tmp2 = list(range(0, len(y_initial)))
    tmp3 = list(range(len(y_initial), len(y_pred)+len(y_initial))) 

    plt.figure(figsize=(10, 6))
    plt.scatter(tmp1, y_samples, label="Ground Truth", color='blue')
    plt.scatter(tmp2, y_initial, label="Initial Samples", color='green')
    plt.scatter(tmp3, y_pred, label="Predictions", color='orange')
    plt.title(f"Unrolled Prediction (Epoch: {epoch})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc="upper left")
    plt.grid(True)
    plot_file = os.path.join(save_path, f"unrolled_plot_epoch_{epoch}.png")
    plt.savefig(plot_file)
    plt.close()


def val_plot_checkpoint(model, SEQ_LEN_PAST, SEQ_LEN_FUTURE, save_dir, epoch):

    all_files = sorted(glob.glob("./datasets/sin_data/val/*.txt"))

    num_plots = 1

    for i in range(num_plots):
        idx_file = random.randint(0, len(all_files) - 1)
        samples = parse_file(all_files[idx_file])
        idx_seq = random.randint(SEQ_LEN_PAST, len(samples) - SEQ_LEN_FUTURE)

        sub_req_input = samples[idx_seq - SEQ_LEN_PAST:idx_seq]
        sub_seq_label = samples[idx_seq:idx_seq + SEQ_LEN_FUTURE]

        sub_seq_input = np.asarray(sub_req_input)
        sub_seq_label = np.asarray(sub_seq_label)

        input_tensor = torch.tensor(sub_seq_input, dtype=torch.float32).to(next(model.parameters()).device)
        input_tensor = model.preprocess_input(input_tensor)

        preds = model(input_tensor).detach().cpu().numpy().reshape(SEQ_LEN_FUTURE, NUM_OUTPUT_PARAMETERS)
        
        preds = preds.astype(np.float32)

        plot_partial(samples, sub_seq_input, sub_seq_label, preds, epoch, save_dir)
        plot_unrolled(samples, model, epoch, save_dir)



class own_MLP(nn.Module):
    def __init__(self):
        super(own_MLP, self).__init__()
        self.inputLayer = nn.Linear(SEQ_LEN_PAST * NUM_INPUT_PARAMETERS, HIDDEN_SIZE)
        self.layer1 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.layer2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.outputLayout = nn.Linear(HIDDEN_SIZE, SEQ_LEN_FUTURE * NUM_OUTPUT_PARAMETERS)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.inputLayer(x))
        x = self.dropout(x)
        x = torch.relu(self.layer1(x))
        x = self.dropout(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.outputLayout(x)
        return x
    
    def preprocess_input(self, x):
        return x.view(-1, SEQ_LEN_PAST * NUM_INPUT_PARAMETERS) 
    
class own_Conv1d(nn.Module):
    def __init__(self):
        super(own_Conv1d, self).__init__()
        self.inputLayer = nn.Conv1d(NUM_INPUT_PARAMETERS, HIDDEN_SIZE, kernel_size=3)
        self.convLayer1 = nn.Conv1d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=3)
        self.convLayer2 = nn.Conv1d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=3)
        self.pooling = nn.AvgPool1d(kernel_size=2)
        self.Dropout = nn.Dropout(0.3)

        reduced_seq_length = (SEQ_LEN_PAST - 6) // 2
        self.denseLayer = nn.Linear(HIDDEN_SIZE * reduced_seq_length, HIDDEN_SIZE)
        self.outputLayout = nn.Linear(HIDDEN_SIZE, SEQ_LEN_FUTURE * NUM_OUTPUT_PARAMETERS)

    def forward(self, x):
        x = torch.relu(self.inputLayer(x))
        x = self.Dropout(x)
        x = torch.relu(self.convLayer1(x))
        x = self.Dropout(x)
        x = torch.relu(self.convLayer2(x))
        x = self.Dropout(x)

        x = self.pooling(x)
        x = torch.flatten(x, start_dim=1)

        x = torch.relu(self.denseLayer(x))
        x = self.outputLayout(x)
        return x
    
    def preprocess_input(self, x):
        return x.view(-1, NUM_INPUT_PARAMETERS, SEQ_LEN_PAST)

class own_LSTM(nn.Module):
    def __init__(self):
        super(own_LSTM, self).__init__()
        self.inputLayer = nn.LSTM(NUM_INPUT_PARAMETERS, HIDDEN_SIZE, batch_first=True, bias=True)
        self.lstmLayer1 = nn.LSTM(HIDDEN_SIZE, HIDDEN_SIZE, batch_first=True, bias=True)
        self.outputLayer = nn.Linear(HIDDEN_SIZE, SEQ_LEN_FUTURE * NUM_OUTPUT_PARAMETERS)

    def forward(self, x):
        # Pass through the first LSTM layer
        x, _ = self.inputLayer(x)
        print(f"After inputLayer: {x.shape}")  # Debug print
        
        # Pass through the second LSTM layer
        x, _ = self.lstmLayer1(x)
        print(f"After lstmLayer1: {x.shape}")  # Debug print
        
        # Pass through the final linear layer
        # Only use the last output (hidden state at the last time step)
        x = x[:, -1, :]  # Select the last time step output
        print(f"After selecting last time step: {x.shape}")  # Debug print
        
        x = self.outputLayer(x)
        print(f"After outputLayer: {x.shape}")  # Debug print

        # Reshape the output to match the expected label shape
        batch_size = x.size(0)
        x = x.view(batch_size, SEQ_LEN_FUTURE, NUM_OUTPUT_PARAMETERS)
        print(f"Final reshaped output: {x.shape}")  # Debug print

        return x

    def preprocess_input(self, x):
        return x.view(-1, SEQ_LEN_PAST, NUM_INPUT_PARAMETERS)


def train_model(model, loss_fn, optimizer, data_path: str, device):
    train_data = data_generator(data_path + "/train/", BATCH_SIZE, model, device)
    val_data = data_generator(data_path + "/val/", BATCH_SIZE, model, device)

    train_loss_history = []
    validate_loss_history = []

    name = f"{type(model).__name__}_EP{EPOCHS}_ST{STEPS_PER_EPOCH}_BS{BATCH_SIZE}_HS{HIDDEN_SIZE}"

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for step in tqdm(range(STEPS_PER_EPOCH)):
            inputs, labels = next(train_data)
            optimizer.zero_grad()
            outputs = model(inputs)
            print("\n\n")
            print(f"outputs shape: {outputs.shape}")
            print(f"labels shape: {labels.shape}")
            print("\n\n")
            loss = loss_fn(outputs, labels.view(-1, SEQ_LEN_FUTURE, NUM_OUTPUT_PARAMETERS))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss_history.append(train_loss / STEPS_PER_EPOCH)

        val_plot_checkpoint(model, SEQ_LEN_PAST, SEQ_LEN_FUTURE, name, epoch)

        model.eval()
        val_loss = 0
        for step in range(VALIDATION_STEPS):
            inputs, labels = next(val_data)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels.view(-1, SEQ_LEN_FUTURE, NUM_OUTPUT_PARAMETERS))
            val_loss += loss.item()
        validate_loss_history.append(val_loss / VALIDATION_STEPS)

        print(f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {train_loss_history[-1]} - Validation Loss: {validate_loss_history[-1]}")


        save_loss_history_as_image_plot(train_loss_history, validate_loss_history, name)

    return

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_path = "./datasets/sin_data/"


    # model = own_MLP()
    # model = own_Conv1d()
    model = own_LSTM()

    model = model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, loss_fn, optimizer, data_path, device)

if __name__ == "__main__":
    main()