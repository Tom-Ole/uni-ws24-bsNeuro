import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

SEQ_LEN_PAST = 30
SEQ_LEN_FUTURE = 3
HIDDEN_SIZE = 1024

BATCH_SIZE = 128
EPOCHS = 100
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

def data_generator(path, batch_size):
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
        inputs = torch.tensor(inputs, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        inputs = inputs.view(batch_size, SEQ_LEN_PAST * NUM_INPUT_PARAMETERS) 

        yield inputs, labels

def save_loss_history_as_image_plot(train_loss_history, validate_loss_history, name):
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(validate_loss_history, label='Validation Loss')
    plt.title("Training and Validation Loss History")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{name}.png")
    plt.close()

def save_prediction_as_plot(inputs, labels, predictions, name):
    plt.scatter(range(len(inputs)), inputs, label='Input', color='blue', s=10)  # Punkte für Input
    plt.scatter(range(len(inputs), len(inputs) + len(labels)), labels, label='Ground Truth', color='orange', s=10)  # Punkte für Ground Truth
    plt.scatter(range(len(inputs), len(inputs) + len(predictions)), predictions, label='Prediction', color='green', s=10)  # Punkte für Prediction
    plt.title("Prediction")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(f"./prediction/{name}.png")
    plt.close()


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
    
def train_model(model, loss_fn, optimizer, data_path: str):
    train_data = data_generator(data_path + "/train/", BATCH_SIZE)
    val_data = data_generator(data_path + "/val/", BATCH_SIZE)

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
            loss = loss_fn(outputs, labels.view(BATCH_SIZE, SEQ_LEN_FUTURE * NUM_OUTPUT_PARAMETERS))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss_history.append(train_loss / STEPS_PER_EPOCH)

        save_prediction_as_plot(inputs[0].view(SEQ_LEN_PAST).detach().numpy(), labels[0].view(SEQ_LEN_FUTURE).detach().numpy(), outputs[0].view(SEQ_LEN_FUTURE).detach().numpy(), f"prediction_{epoch + 1}_{name}")

        model.eval()
        val_loss = 0
        for step in range(VALIDATION_STEPS):
            inputs, labels = next(val_data)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels.view(BATCH_SIZE, SEQ_LEN_FUTURE * NUM_OUTPUT_PARAMETERS))
            val_loss += loss.item()
        validate_loss_history.append(val_loss / VALIDATION_STEPS)

        print(f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {train_loss_history[-1]} - Validation Loss: {validate_loss_history[-1]}")


        save_loss_history_as_image_plot(train_loss_history, validate_loss_history, f"loss_history_{name}")

    return

def main():

    data_path = "./datasets/sin_data/"

    model = own_MLP()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, loss_fn, optimizer, data_path)

if __name__ == "__main__":
    main()