import os
import random
import numpy as np
import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from matplotlib import pyplot as plt

GPU_STRING = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128
MODEL_NAME = "experiment1"
EPOCHS = 10
STEPS_PER_EPOCH = 10
VALIDATION_STEPS = 32
SEQ_LEN_PAST = 30
SEQ_LEN_FUTURE = 1
NUM_INPUT_PARAMETERS = 1
NUM_OUTPUT_PARAMETERS = 1

LOSS_PATH = ""

class ModelHistory:
    def __init__(self, model_path):
        self.model_path = model_path
        self.loss = []
        self.loss_val = []
        self.mae = []
        self.mae_val = []
        self.mse = []
        self.mse_val = []

    def on_epoch_end(self, epoch, train_loss, val_loss, train_mae, val_mae, train_mse, val_mse):
        self.loss.append(train_loss)
        self.loss_val.append(val_loss)
        self.mae.append(train_mae)
        self.mse.append(train_mse)
        self.mae_val.append(val_mae)
        self.mse_val.append(val_mse)
        self.plot_data()

    def plot_data(self):
        vis_path = os.path.join(self.model_path, 'vis')
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)

        model_name = self.model_path.split('/')[-1]

        plt.clf()
        plt.plot(self.loss)
        plt.plot(self.loss_val)
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        loss_path = os.path.join(vis_path, 'loss.png')
        plt.savefig(loss_path)

def setup_model_mlp():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(SEQ_LEN_PAST * NUM_INPUT_PARAMETERS, 1024),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(1024, 128),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(128, SEQ_LEN_FUTURE * NUM_OUTPUT_PARAMETERS),
        nn.Linear(SEQ_LEN_FUTURE * NUM_OUTPUT_PARAMETERS, SEQ_LEN_FUTURE * NUM_OUTPUT_PARAMETERS)
    )
    return model

def setup_model_conv_1d():
    model = nn.Sequential(
        nn.Conv1d(NUM_INPUT_PARAMETERS, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Conv1d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Conv1d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, SEQ_LEN_FUTURE * NUM_OUTPUT_PARAMETERS),
        nn.Linear(SEQ_LEN_FUTURE * NUM_OUTPUT_PARAMETERS, SEQ_LEN_FUTURE * NUM_OUTPUT_PARAMETERS)
    )
    return model

def setup_model_lstm():
    model = nn.Sequential(
        nn.LSTM(NUM_INPUT_PARAMETERS, 256, batch_first=True),
        nn.LSTM(256, 512, batch_first=True),
        nn.Linear(512, SEQ_LEN_FUTURE * NUM_OUTPUT_PARAMETERS),
        nn.Linear(SEQ_LEN_FUTURE * NUM_OUTPUT_PARAMETERS, SEQ_LEN_FUTURE * NUM_OUTPUT_PARAMETERS)
    )
    return model

def setup_model_transformer():
    # Implementing a simple Transformer model
    return nn.Transformer()

def parse_file(file_name):
    with open(file_name, "r") as file:
        lines = file.readlines()

    samples = []
    for line in lines:
        x, y = line.strip().split(' ')
        samples.append((float(y),))
    
    return samples

def select_data(batch_size, all_files):
    selected_inputs = []
    selected_labels = []
    num = 0
    while num < batch_size:
        idx_file = random.randint(0, len(all_files)-1)
        samples = parse_file(all_files[idx_file])
        idx_seq = random.randint(SEQ_LEN_PAST, len(samples) - SEQ_LEN_FUTURE)
        sub_seq_input = samples[idx_seq-SEQ_LEN_PAST:idx_seq]
        sub_seq_label = samples[idx_seq:idx_seq+SEQ_LEN_FUTURE]
        selected_inputs.append(sub_seq_input)
        selected_labels.append(sub_seq_label)
        num += 1
    return np.asarray(selected_inputs), np.asarray(selected_labels)

class DataGenerator(data.Dataset):
    def __init__(self, path, batch_size):
        self.all_files = sorted(glob.glob(path + '*.txt'))
        self.batch_size = batch_size

    def __len__(self):
        return len(self.all_files) // self.batch_size

    def __getitem__(self, idx):
        inputs, labels = select_data(self.batch_size, self.all_files)
        mu, sigma = 0, 0.1
        rnd = np.random.normal(mu, sigma, size=inputs.shape)
        inputs += rnd
        return torch.Tensor(inputs), torch.Tensor(labels)

def train(data_path, model_path, model, from_checkpoint=False):
    train_gen = DataGenerator(data_path + 'train\\', BATCH_SIZE)
    val_gen = DataGenerator(data_path + 'val\\', BATCH_SIZE)

    model = model.to(GPU_STRING)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model_history = ModelHistory(model_path)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(enumerate(train_gen), total=len(train_gen), desc=f"Epoch {epoch + 1}/{EPOCHS}")


        for i, (inputs, labels) in enumerate(train_gen):
            inputs, labels = inputs.to(GPU_STRING), labels.to(GPU_STRING)

            optimizer.zero_grad()
            outputs = model(inputs)

            labels = labels.squeeze(-1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            progress_bar.set_postfix(loss=f"{running_loss / (i + 1):.4f}")

        model_history.on_epoch_end(epoch, running_loss / len(train_gen), running_loss / len(val_gen), 0, 0, 0, 0)

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss / len(train_gen)}")

def save_loss_png(read_path: str, save_path: str, name: str) -> bool:
    with open(read_path, 'rb') as file:
        data = file.read()
        with open(save_path + name + '.png', 'wb') as file:
            file.write(data)
            return True
    return False

def run():
    print("\n")
    print("GPUs:", torch.cuda.is_available())
    print("\n")

    path = './'
    data_path = path + 'datasets/sin_data/'
    model_path = path + 'models/' + MODEL_NAME + '/'

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model = setup_model_mlp()
    train(data_path, model_path, model)
    save_loss_png("D:/uni/bsNeuro/models/experiment1/vis/loss.png", "D:/uni/bsNeuro/", "loss_mlp_50Epochs")
    print("\n\n\n [INFO]: FINISHED MLP \n [INFO]: Starting Transformer \n\n\n")

    # model = setup_model_transformer()
    # train(data_path, model_path, model)
    # save_loss_png("D:/uni/bsNeuro/models/experiment1/vis/loss.png", "D:/uni/bsNeuro/", "loss_transformer_50Epochs")
    # print("\n\n\n [INFO]: FINISHED Transformer \n [INFO]: Starting LSTM \n\n\n")

    model = setup_model_lstm()
    train(data_path, model_path, model)
    save_loss_png("D:/uni/bsNeuro/models/experiment1/vis/loss.png", "D:/uni/bsNeuro/", "loss_lstm_50Epochs")
    print("\n\n\n [INFO]: FINISHED LSTM \n [INFO]: Starting Conv1D \n\n\n")

    model = setup_model_conv_1d()
    train(data_path, model_path, model)
    save_loss_png("D:/uni/bsNeuro/models/experiment1/vis/loss.png", "D:/uni/bsNeuro/", "loss_conv_1d_50Epochs")
    print("\n\n\n [INFO]: FINISHED Conv1D \n\n\n")

if __name__ == "__main__":
    run()
