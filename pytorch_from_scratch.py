import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

SEQ_LEN_PAST = 10
SEQ_LEN_FUTURE = 5
HIDDEN_SIZE = 64

BATCH_SIZE = 128
EPOCHS = 100
STEPS_PER_EPOCH = 20

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
        sigma = 0.1
        rnd = np.random.normal(mu, sigma, size=inputs.shape)
        inputs = inputs + rnd 

        # Convert to PyTorch tensors
        inputs = torch.tensor(inputs, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        inputs = inputs.view(batch_size, SEQ_LEN_PAST * NUM_INPUT_PARAMETERS) 

        yield inputs, labels

class MLP(nn.Module):
    def __init__(self, seq_len_past, num_input_parameters, seq_len_future, num_output_parameters, dropout_prob=0.4):
        super(MLP, self).__init__()

        # TODO: Make it dynamic
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(seq_len_past * num_input_parameters, 1024)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dense2 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.dense3 = nn.Linear(1024, 1024)
        self.dropout3 = nn.Dropout(dropout_prob)
        self.dense4 = nn.Linear(1024, 128)
        self.dropout4 = nn.Dropout(dropout_prob)
        self.dense5 = nn.Linear(128, 128)
        self.dropout5 = nn.Dropout(dropout_prob)
        self.dense6 = nn.Linear(128, seq_len_future * num_output_parameters)
        
        self.seq_len_future = seq_len_future
        self.num_output_parameters = num_output_parameters

    def forward(self, x):
        x = torch.relu(self.dense1(x))
        x = self.dropout1(x)
        x = torch.relu(self.dense2(x))
        x = self.dropout2(x)
        x = torch.relu(self.dense3(x))
        x = self.dropout3(x)
        x = torch.relu(self.dense4(x))
        x = self.dropout4(x)
        x = torch.relu(self.dense5(x))
        x = self.dropout5(x)
        x = self.dense6(x)

        # Reshape to (batch_size, seq_len_future, num_output_parameters)
        x = x.view(-1, self.seq_len_future, self.num_output_parameters)
        return x

def train_model(model, data_path: str, epochs: int, steps_per_epoch: int, batch_size: int, optimizer, criterion):
    train_gen = data_generator(data_path + "/train/", batch_size)
    validation_gen = data_generator(data_path + "/val/", batch_size)

    train_loss_history = []
    validate_loss_history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for step in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}/{epochs}"):
            inputs, labels = next(train_gen)
            optimizer.zero_grad()

            predictions = model(inputs)

            # Reshape labels to match the shape of predictions: (batch_size, seq_len_future, num_output_parameters)
            labels = labels.view(labels.size(0), SEQ_LEN_FUTURE, NUM_OUTPUT_PARAMETERS)

            loss = criterion(predictions, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= steps_per_epoch
        train_loss_history.append(train_loss)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")

        model.eval()

        # validate the model
        validate_loss = 0.0
        for _ in range(steps_per_epoch):
            inputs, labels = next(validation_gen)
            with torch.no_grad():
                predictions = model(inputs)

                # Reshape labels to match the shape of predictions: (batch_size, seq_len_future, num_output_parameters)
                labels = labels.view(labels.size(0), SEQ_LEN_FUTURE, NUM_OUTPUT_PARAMETERS)

                loss = criterion(predictions, labels)
                validate_loss += loss.item()

        validate_loss /= steps_per_epoch
        validate_loss_history.append(validate_loss)
        print(f"Epoch {epoch + 1}, Validation Loss: {validate_loss:.4f}")

        save_history_as_image_plot(train_loss_history, validate_loss_history, f"loss_{type(model).__name__}_{epochs}")


def save_history_as_image_plot(train_loss_history, validate_loss_history, name):
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(validate_loss_history, label='Validation Loss')
    plt.title("Training and Validation Loss History")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{name}.png")
    plt.close()


def main():
    data_path = "./datasets/sin_data/"
    batch_size = BATCH_SIZE
    epochs = EPOCHS
    steps_per_epoch = STEPS_PER_EPOCH


    model_mlp = MLP(SEQ_LEN_PAST, NUM_INPUT_PARAMETERS, SEQ_LEN_FUTURE, NUM_OUTPUT_PARAMETERS)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_mlp.parameters(), lr=0.001)

    train_model(model_mlp, data_path, epochs, steps_per_epoch, batch_size, optimizer, criterion)


if __name__ == "__main__":
    main()
