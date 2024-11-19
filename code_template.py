import os
import random
import numpy as np
import tensorflow as tf
import glob
from matplotlib import pyplot as plt

GPU_STRING = '/gpu:0'
BATCH_SIZE = 128
MODEL_NAME = "experiment1"
EPOCHS = 50
STEPS_PER_EPOCH = 10
VALIDATION_STEPS = 32
SEQ_LEN_PAST = 30
SEQ_LEN_FUTURE = 1
NUM_INPUT_PARAMETERS = 1
NUM_OUTPUT_PARAMETERS = 1

LOSS_PATH = "";

class ModelHistory(tf.keras.callbacks.Callback):
  def __init__(self, model_path):
    self.model_path = model_path
    self.loss = []
    self.loss_val = []
    self.mae = []
    self.mae_val = []
    self.mse = []
    self.mse_val = []

  def on_epoch_end(self, batch, logs):
    self.loss.append(logs.get('loss'))
    self.loss_val.append(logs.get('val_loss'))
    self.mae.append(logs.get('mean_absolute_error'))
    self.mse.append(logs.get('mean_squared_error'))
    self.mae_val.append(logs.get('val_mean_absolute_error'))
    self.mse_val.append(logs.get('val_mean_squared_error'))
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
    LOSS_PATH = os.path.join(vis_path, 'loss.png')
    plt.savefig(LOSS_PATH)


def setup_model_mlp():
  input = tf.keras.layers.Input(shape=(SEQ_LEN_PAST, NUM_INPUT_PARAMETERS), name='input')

  dp = 0.4
  x = tf.keras.layers.Flatten()(input)
  x = tf.keras.layers.Dense(1024, activation='relu')(x)
  x = tf.keras.layers.Dropout(dp)(x)
  x = tf.keras.layers.Dense(1024, activation='relu')(x)
  x = tf.keras.layers.Dropout(dp)(x)
  x = tf.keras.layers.Dense(1024, activation='relu')(x)
  x = tf.keras.layers.Dropout(dp)(x)
  x = tf.keras.layers.Dense(128, activation='relu')(x)
  x = tf.keras.layers.Dropout(dp)(x)
  x = tf.keras.layers.Dense(128, activation='relu')(x)
  x = tf.keras.layers.Dropout(dp)(x)  
  x = tf.keras.layers.Dense(SEQ_LEN_FUTURE * NUM_OUTPUT_PARAMETERS, activation='linear')(x)
  x = tf.keras.layers.Reshape((SEQ_LEN_FUTURE, NUM_OUTPUT_PARAMETERS))(x)

  model = tf.keras.models.Model(input, x)
  return model


def setup_model_conv_1d():
  input = tf.keras.layers.Input(shape=(SEQ_LEN_PAST, NUM_INPUT_PARAMETERS), name='input')

  dp = 0.4
  x = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(input)
  x = tf.keras.layers.Dropout(dp)(x)
  x = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
  x = tf.keras.layers.Dropout(dp)(x)
  x = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
  x = tf.keras.layers.Dropout(dp)(x)  
  x = tf.keras.layers.GlobalAveragePooling1D()(x)

  x = tf.keras.layers.Dense(128, activation='relu')(x)  
  x = tf.keras.layers.Dense(SEQ_LEN_FUTURE * NUM_OUTPUT_PARAMETERS, activation='linear')(x)
  x = tf.keras.layers.Reshape((SEQ_LEN_FUTURE, NUM_OUTPUT_PARAMETERS))(x)

  model = tf.keras.models.Model(input, x)
  return model  


def setup_model_lstm():
  input = tf.keras.layers.Input(shape=(SEQ_LEN_PAST, NUM_INPUT_PARAMETERS), name='input')

  x = tf.keras.layers.LSTM(256, return_sequences=True)(input)
  x = tf.keras.layers.LSTM(512)(input)
  x = tf.keras.layers.Dense(SEQ_LEN_FUTURE * NUM_OUTPUT_PARAMETERS, activation='linear')(x)
  x = tf.keras.layers.Reshape((SEQ_LEN_FUTURE, NUM_OUTPUT_PARAMETERS))(x)

  model = tf.keras.models.Model(input, x)
  return model  


def transformer_encoder(inputs, dropout=0.25, head_size=256, num_heads=32, ff_dim=4):

    # Normalization and Attention
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = tf.keras.layers.Dropout(dropout)(x)

    res = x + inputs

    # Feed Forward Part
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
    x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def setup_model_transformer(dropout=0.4, num_transformer_blocks=8, mlp_units=[128]):
  inputs = tf.keras.Input(shape=(SEQ_LEN_PAST, NUM_INPUT_PARAMETERS))
  x = inputs

  for _ in range(num_transformer_blocks):
      x = transformer_encoder(x)

  x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
  for dim in mlp_units:
      x = tf.keras.layers.Dense(dim, activation="relu")(x)
      x = tf.keras.layers.Dropout(dropout)(x)

  outputs = tf.keras.layers.Dense(SEQ_LEN_FUTURE * NUM_OUTPUT_PARAMETERS)(x)
  outputs = tf.keras.layers.Reshape((SEQ_LEN_FUTURE, NUM_OUTPUT_PARAMETERS))(outputs)

  return tf.keras.Model(inputs, outputs)


def parse_file(file_name):
  file = open(file_name, "r")
  lines = file.readlines()

  samples = []

  count = 0
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


def data_generator(path, batch_size):
  all_files = sorted(glob.glob(path + '*.txt'))  

  while True:
   inputs, labels = select_data(batch_size, all_files)

   #Data Augmentation
   mu = 0
   sigma = 0.1
   rnd = np.random.normal(mu, sigma, size=inputs.shape)
   inputs += rnd

   yield inputs, labels


def plot_data_partial(samples, sub_seq_input, sub_seq_label, sub_seq_pred, model_path, epoch, count):
    vis_path = os.path.join(model_path, 'vis', 'plots')
    if not os.path.exists(vis_path):
      os.makedirs(vis_path)

    input_samples_y = []
    for i in range(sub_seq_input.shape[1]):
      input_samples_y.append(sub_seq_input[0][i][0])

    label_samples_y = []
    for i in range(sub_seq_label.shape[0]):
      label_samples_y.append(sub_seq_label[i][0])      

    pred_samples_y = []
    for i in range(sub_seq_pred.shape[0]):
      pred_samples_y.append(sub_seq_pred[i][0])      

    plt.clf()
    
    tmp = list(range(0, len(input_samples_y)))
    plt.scatter(tmp, input_samples_y)    
    plt.scatter(len(input_samples_y), pred_samples_y)
    plt.scatter(len(input_samples_y), label_samples_y)
    
    plt.title('Sin Plot Partial')
    plt.xlabel('X')
    plt.ylabel('Y')    
    plt.legend(['Inputs', 'Pred', 'Label'], loc='upper left')
    plt.savefig(vis_path + '\\' + str(epoch).zfill(4) + '_' + str(count).zfill(4) + '_p.png')    


def plot_unroll(samples, model_path, model, epoch, count):
  vis_path = os.path.join(model_path, 'vis', 'plots')
  if not os.path.exists(vis_path):
    os.makedirs(vis_path)

  rolling_buffer = samples[0:SEQ_LEN_PAST]
  initial_samples = rolling_buffer.copy()

  num_steps = 300  

  pred_samples = []
  for i in range(num_steps):
    input_seq = np.expand_dims(np.asarray(rolling_buffer), axis=0)        

    preds = model.predict(input_seq)[0]
    preds = preds.astype(np.float32)

    pred_samples.append((preds[0][0],))
    rolling_buffer.append((preds[0][0],))

    rolling_buffer = rolling_buffer[1:]


  y_pred = []
  for i in range(len(pred_samples)):
    y_pred.append(pred_samples[i][0])   

  y_initial = []
  for i in range(len(initial_samples)):
    y_initial.append(initial_samples[i][0])   

  y_samples = []
  for i in range(len(samples)):
    y_samples.append(samples[i][0])

  plt.clf()
  tmp1 = list(range(0, len(y_samples)))
  tmp2 = list(range(0, len(y_initial)))
  tmp3 = list(range(len(y_initial), len(y_pred)+len(y_initial)))  

  
  plt.scatter(tmp1, y_samples)
  plt.scatter(tmp2, y_initial)
  plt.scatter(tmp3, y_pred)        
  plt.title('Rolling Predictions')
  plt.xlabel('X')
  plt.ylabel('Y')    
  plt.legend(['GT', 'Initial', 'Pred'], loc='upper left')
  plt.savefig(vis_path + '\\' + str(epoch).zfill(4) + '_' + str(count).zfill(4) + '_u.png')   


def val_func(data_path, model_path, model, epoch):
  all_files = sorted(glob.glob(data_path + 'val/*.txt'))

  num_plots = 10

  for i in range(num_plots):
    idx_file = random.randint(0, len(all_files)-1)
    samples = parse_file(all_files[idx_file])

    idx_seq = random.randint(SEQ_LEN_PAST, len(samples) - SEQ_LEN_FUTURE)
    sub_seq_input = samples[idx_seq-SEQ_LEN_PAST:idx_seq]
    sub_seq_label = samples[idx_seq:idx_seq+SEQ_LEN_FUTURE]    

    sub_seq_input = np.asarray(sub_seq_input)
    sub_seq_label = np.asarray(sub_seq_label)

    sub_seq_input = np.expand_dims(sub_seq_input, axis=0)

    preds = model.predict(sub_seq_input)[0]
    preds = preds.astype(np.float32)

    plot_data_partial(samples, sub_seq_input, sub_seq_label, preds, model_path, epoch, i)
    plot_unroll(samples, model_path, model, epoch, i)


def train(data_path, model_path, model, from_checkpoint=False):

  train_gen = data_generator(data_path + 'train\\', BATCH_SIZE)   
  val_gen = data_generator(data_path + 'val\\', BATCH_SIZE)  

  opt = tf.keras.optimizers.Adam(learning_rate=0.001)
  
  model.compile(loss='mse', optimizer=opt, metrics=["mse", "mae"])
  model.summary()

  model_history_callback = ModelHistory(model_path)

  # dont know yet
  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(model_path + 'model-{epoch:03d}.keras', verbose=1, monitor='val_mae', save_best_only=True, mode='auto')
  
  # saves each epoch as a checkpoint
  val_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: val_func(data_path, model_path, model, epoch))

  history = model.fit(train_gen, steps_per_epoch=STEPS_PER_EPOCH, validation_steps=VALIDATION_STEPS, epochs=EPOCHS, 
                      validation_data=val_gen, callbacks=[model_history_callback], 
                      shuffle=True, verbose='auto')

def save_loss_png(read_path: str, save_path: str, name: str) -> bool:
  # save a loss png file to the given Path with the given name
  
  with open(read_path, 'rb') as file:
    data = file.read()
    with open(save_path + name + '.png', 'wb') as file:
      file.write(data)
      return True

  return False



def run():
  physical_devices = tf.config.list_physical_devices('GPU')
  print("\n")
  print("GPUs:", physical_devices)

  print("\n")

  with tf.device(GPU_STRING):
    path = './'
    data_path = path + 'datasets\\sin_data\\'
    model_path = path + 'models\\' + MODEL_NAME + '\\'

    if not os.path.exists(model_path):
      os.makedirs(model_path)


    model = setup_model_mlp()
    train(data_path, model_path, model)
    save_loss_png("D:/uni/bsNeuro/models/experiment1/vis/loss.png", "D:/uni/bsNeuro/", "loss_mlp_50Epochs")
    print("\n\n\n [INFO]: FINISHED MLP \n [INFO]: Starting Transfromer \n\n\n")
  

    model = setup_model_transformer()
    train(data_path, model_path, model)
    save_loss_png("D:/uni/bsNeuro/models/experiment1/vis/loss.png", "D:/uni/bsNeuro/", "loss_transformer_50Epochs")
    print("\n\n\n [INFO]: FINISHED Transfromer \n [INFO]: Starting lstm \n\n\n")

    model = setup_model_lstm()
    train(data_path, model_path, model)
    save_loss_png("D:/uni/bsNeuro/models/experiment1/vis/loss.png", "D:/uni/bsNeuro/", "loss_lstm_50Epochs")
    print("\n\n\n [INFO]: FINISHED Transfromer \n [INFO]: Starting conv_1d \n\n\n")

    model = setup_model_conv_1d()
    train(data_path, model_path, model)
    save_loss_png("D:/uni/bsNeuro/models/experiment1/vis/loss.png", "D:/uni/bsNeuro/", "loss_conv_1d_50Epochs")
    print("\n\n\n [INFO]: FINISHED conv_1d \n\n\n")

    mode = 'train'



if __name__== "__main__":
  run()