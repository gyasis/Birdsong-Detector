# %%
try:
  %load_ext autotime
except:
  print("Console warning-- Autotime is jupyter platform specific")
# %%
import comet_ml
from comet_ml import Experiment
import os
import glob
from tqdm import tqdm
import pandas as pd
import datetime
from datetime import datetime
import time
import matplotlib.pyplot as plt

import icecream as ic
from IPython.display import Audio, display
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
import librosa
import torch
from scipy.io import wavfile as wav
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
torch.cuda._initialized = True
import time
import math, random

# %%
# look at metadatacsv
df = pd.read_csv('/media/gyasis/Drive 2/Data/birdsong/metadata.csv')
df.head()

# %%
print(df.columns)
# %%
df = df[['Recording_ID','Audio_file','Species','Path', 'Length']]
# %%
torch.cuda.empty_cache() 
# %%
def fix_path(x):
    main_path = '/media/gyasis/Drive 2/Data/birdsong/mp3/'
    x = os.path.split(x)
    x = os.path.join(main_path,x[1])
    return x
    
# %%
df['Path'] = df['Path'].apply(lambda x: fix_path(x))
# %%
df.head()
# %%
species = df.Species.unique()
# %%
print(len(species))
# %%

# %%
def get_seconds(x):
    x = datetime.strptime(x,'%M:%S')
    #remove date from data time?
    a_timedelta = x - datetime(1900, 1, 1)
    seconds = a_timedelta.total_seconds()
    return seconds

# %%
df['Total_sec'] = df['Length'].apply(lambda x: get_seconds(x))
# %%
df.head()




# %%

#function histogram for df.Total_sec

def histogram(x):
    plt.hist(x, bins=100, range=(0, 750))
    plt.show()


histogram(df.Total_sec)



# %%
# last, categorical encoding
from sklearn.preprocessing import LabelEncoder
l_e = LabelEncoder()
df['Class'] = l_e.fit_transform(df['Species'])



# %%
def point_path(x):
    wav_path = '/media/gyasis/Drive 2/Data/birdsong/wav/'
    for i,file in enumerate(df.Path):
        x = os.path.split(x)
        x = x[1].replace("mp3", "wav")
        new_path = os.path.join(wav_path, x)
        return new_path
        
        
df['Wav'] = df['Path'].apply(lambda x: point_path(x))

# %%
offset = list()
frame_dur = list()
wavey = list()
classy = list()
target_size = 5 #seconds 
target_frames = target_size * 22050   
print(len(df))

for i, data in enumerate(df.Wav):
  
  metadata = torchaudio.info(df.Wav[i])
  temp_frames = metadata.num_frames
  begin_frame = 0
    
  if temp_frames > target_frames:
    
    while temp_frames > target_frames: 
     
      offset.append(begin_frame)
      wavey.append(df.Wav[i])
      classy.append(df.Class[i])
      
      if temp_frames > temp_frames - target_frames :
        frame_dur.append(target_frames) 
      else :
        frame_dur.append(-1)
        
      temp_frames = temp_frames - target_frames
      begin_frame = begin_frame + target_frames + 1
   
  else:
    # print('no need to splice')
    offset.append(begin_frame)
    frame_dur.append(-1)
    wavey.append(df.Wav[i])
    classy.append(df.Class[i])
      
    temp_frames = temp_frames - target_frames


meta = list(zip(offset, frame_dur))
meta_plus = list(zip(wavey, classy))
data1 = pd.DataFrame(meta, columns = ['Offset', 'Duration'])
data2 = pd.DataFrame(meta_plus, columns = ['Wav','Class'])
df = pd.merge(data1, data2, left_index=True, right_index=True)
def mp3path(x):
    mp3_path = '/media/gyasis/Drive 2/Data/birdsong/mp3/'
    for i,file in enumerate(df.Wav):
        x = os.path.split(x)
        x = x[1].replace("wav", "mp3")
        new_path = os.path.join(mp3_path, x)
        return new_path
df['mp3'] = df['Wav'].apply(lambda x: mp3path(x))
df['Species'] = l_e.inverse_transform(df['Class'])
df.head()

# %%
experiment = Experiment(api_key="xleFjfKO3kcwc56tglgC1d3zU",
                        project_name="Bird_calls",log_code=True)
# %%

max_ms = 5000
n_fft = 1024
win_length = None
hop_length = None
n_mels = 256
n_mfcc = 256
samp_r = 22050
mfcc_transform = T.MFCC(
    sample_rate=samp_r,
    n_mfcc=n_mfcc,
    melkwargs={
      'n_fft': n_fft,
      'n_mels': n_mels,
      'hop_length': hop_length,
      # 'mel_scale': 'htk',
    }
)


def pad_trunc(aud, max_ms = max_ms):
  sr = 22050
  sig = aud
  
  num_rows, sig_len = sig.shape
  yt = samp_r // 1000
  t = yt * max_ms
  
  max_len = samp_r//1000 * max_ms
  
  if (sig_len >= max_len):
    sig = sig[:,:max_len]
    
  elif (sig_len < max_len):
    pad_begin_len = random.randint(0, max_len - sig_len)
    pad_end_len = max_len - sig_len - pad_begin_len
    pad_begin = torch.zeros((num_rows, pad_begin_len))
    pad_end = torch.zeros((num_rows, pad_end_len))
    sig = torch.cat((pad_begin, sig, pad_end), 1)
    
  return(sig)
def aud_open(audio_file, a, b, show_stats=False):
  try:
    sig, sr= torchaudio.backend.sox_io_backend.load(audio_file, 
                                         frame_offset= a, 
                                         num_frames = b,
                                         normalize =False, 
                                         channels_first = True )
    
  except:
    sig, sr= torchaudio.backend.sox_io_backend.load(audio_file, 
                                         frame_offset= a, 
                                         num_frames = -1,
                                         normalize = False, 
                                         channels_first = True )
  
  
  if show_stats==True:
    show_metadata(audio_file)
  
  return (sig, sr)
def time_shift(aud, shift_limit):
  sig = aud
  _, sig_len = sig.shape
  shift_amt = int(random.random() * shift_limit * sig_len)
  return (sig.roll(shift_amt))
def spectro_gram(x, n_mels=128, n_fft=1024, hop_len=None):
  sig = x
  top_db = 80
  sr = 22050

  spec = torchaudio.transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
  spec = torchaudio.transforms.AmplitudeToDB(top_db=top_db)(spec)

  return(spec)
 # %%
def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
  _, n_mels, n_steps = spec.shape

  mask_value = spec.mean()
  aug_spec = spec
  
  freq_mask_param= max_mask_pct * n_mels
  for _ in range(n_freq_masks):
    aug_spec = T.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)
    
  time_mask_param = max_mask_pct * n_steps
  for _ in range(n_freq_masks):
    
    aug_spec = T.TimeMasking(time_mask_param)(aug_spec,mask_value)
  
  
  return aug_spec
def show_metadata(audio_file):
  metadata = torchaudio.info(audio_file)
  print('--------------------')
  print('--------------------')
  
  print("File:            ", audio_file)
  print('--------------------')
  print("sample_rate:     ",metadata.sample_rate)
  print('--------------------')
  print("num_frames:      ",metadata.num_frames)
  print('--------------------')
  print("num_channels:    ",metadata.num_channels)
  print('--------------------')
  print("bits_per_sample: ",metadata.bits_per_sample)
  print('--------------------')
  print("encoding:        ",metadata.encoding)
  
  print('--------------------')
  print('--------------------')
  
  
  num_frames = metadata.num_frames
  # return num_frames,audio_file
def rechannel(aud, new_channel = 1):
  # ic.ic(aud)
  sig = aud
  # print(sig.shape)
  
  if (sig.shape[0]==new_channel):
    # print('hello')
    return aud

  elif(sig.shape[0]== 2): #shift to mono
    sig = sig[:1, :]
    # ic.ic(sig)
    # print('shift to mono')
         

  elif(sig.shape[0] == 1):
    sig = torch.cat([sig, sig])
    # ic.ic(sig)
    # print('shift to stereo')
    
  # print('this is the sig ',sig)
  return sig
def mel(aud, sr):
    mfcc_module = T.MFCC(sr,
               n_mfcc = 128,
               norm = 'ortho')
    
    spec = mfcc_module(aud)
    
    return spec

# mfcc_module = MFCC(sample_rate=sr, n_mfcc=20, melkwargs={"n_fft": 2048, "hop_length": 512, "power": 2})
# torch_mfcc = mfcc_module(torch.tensor(audio))

n_fft = 1024
win_length = None
hop_length = None
n_mels = 64
mel_spectrogram = T.MelSpectrogram(
    sample_rate=samp_r,
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
    norm='slaney',
    onesided=True,
    n_mels=n_mels,
    # mel_scale="htk",
)


# %%
#Log audio files to Comet for debugging
import random

for i, file in enumerate(df.Wav):
    single_file = aud_open(df.Wav[i],df.Offset[i], df.Duration[i], show_stats=False)
    temp,_ = single_file
    temp1 = temp.numpy()
    randomnumber = random.randint(1,99999999)
    experiment.log_audio(temp.T, 
                         sample_rate=22050, 
                         copy_to_tmp=False,
                         file_name = str(randomnumber)+"Original"+"_"+ df.Species[i]+"_"
                        )
    
    temp = rechannel(temp, 1)
    experiment.log_audio(temp.T, 
                         sample_rate=22050, 
                         copy_to_tmp=False,
                         file_name = str(randomnumber)+
                            "Rechannel"+ df.Species[i])
    
    temp = pad_trunc(temp, 7000)
    experiment.log_audio(temp.numpy().T, 
                         sample_rate=22050,
                         copy_to_tmp=False,
                         file_name = str(randomnumber)+"Pad"+ df.Species[i] +"_")        
            
    temp = time_shift(temp, 0.4)
    experiment.log_audio(temp.numpy().T, 
                         sample_rate=22050,
                         copy_to_tmp=False,
                         file_name = str(randomnumber)+"TimeShift"+ df.Species[i] +"_"+str(randomnumber)
                        )
    
    # plot_x  = temp.squeeze()
    # x_= plt.imshow(plot_x,aspect='auto', 
    #                 cmap='hot'
    #                 )
    # experiment.log_figure(x_, figure_name=randomnumber+df.Species[i], figure=None, overwrite=False, step=None)
    
    
    if i < 7:
      break
    
    
#     import IPython.display
#   from scipy.io import wavfile

# rate, s = wavfile.read('h.wav')
# IPython.display.Audio(s, rate=rate)

# %%
class MyDataset(Dataset):
    def __init__(self, dataset):
        print('initializing....')
        self.df = dataset
        ## set to wave or mp3
        self.sound_arr = np.asarray(dataset.Wav)
        # self.sound_arr = np.asarray(dataset.mp3)
        self.class_arr = np.asarray(dataset.Class)
        self.species_arr = np.asarray(dataset.Species)
        self.offset_arr = np.asarray(dataset.Offset)
        self.duration_arr = np.asarray(dataset.Duration)
        self.data_len = len(dataset.index)
        self.duration = 7000
        self.sr = 22050
        self.channel = 1
        self.shift_pct=0.4
    
    def __getitem__(self, index):
        
        def trans_aug(sample):
            class_id = self.df.loc[index, 'Class']
            species_id = self.df.loc[index, 'Species']
            
            #intermittent logging
            randomnumber = random.randint(1,7)
            
            #chain moving and transforming data
            sample,_ = sample
            sample = rechannel(sample, self.channel)
            sample = pad_trunc(sample, self.duration)
            sample = time_shift(sample, self.shift_pct)
            sample = mfcc_transform(sample)
            spec = torchaudio.transforms.AmplitudeToDB(top_db=40)(sample)
            spec = spectro_augment(spec, n_freq_masks=2, n_time_masks=2)
            return spec, class_id
        
        single_file = aud_open(self.sound_arr[index],self.offset_arr[index], self.duration_arr[index], show_stats=False)
        trans, class_id = trans_aug(single_file)
        trans = trans.float()
      
        return trans, class_id
            
    def __len__(self):
      return self.data_len

# %%
datasetter = MyDataset(df)
# %%
set_batchsize = 512

# %%
from torch.utils.data import random_split

num_items = len(datasetter)
num_train = round(num_items * 0.7)
num_val = num_items - num_train
train_ds, val_ds = random_split(datasetter, [num_train, num_val])
train_dl = DataLoader(train_ds, batch_size=set_batchsize,num_workers=4,pin_memory=True, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=set_batchsize, num_workers=4,pin_memory=True, shuffle=False)
# %%
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class AudioClassifier(nn.Module):
    
        
    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Second Convolution Block
        self.conv3 = nn.Conv2d(32, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(128)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Second Convolution Block
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(256)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=256, out_features=50)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)
 
    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)
        
        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
       
        # Linear layer
        x = self.lin(x)
        
        # Final output
        return x

# Create the model and put it on the GPU if available
myModel = AudioClassifier()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
myModel = myModel.to(device)

# Check that it is on Cuda
# next(myModel.parameters()).device

# %%
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# %%
# ----------------------------
# Training Loop
# ----------------------------
def training(model, train_dl, num_epochs):
  # Loss Function, Optimizer and Scheduler
  optimizer_name = torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
  criterion = nn.CrossEntropyLoss()
  optimizer = optimizer_name
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                steps_per_epoch=int(len(train_dl)),
                                                epochs=num_epochs,
                                                anneal_strategy='linear')



  # Repeat for each epoch
  for epoch in tqdm(range(num_epochs), position=0):
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0

    # Repeat for each batch in the training set
    for i, data in tqdm(enumerate(train_dl),total=len(train_dl), position=1):
        # Get the input features and target labels, and put them on the GPU
        inputs, labels = data[0].to(device), data[1].to(device)
        

        # Normalize the inputs
        inputs_m, inputs_s = inputs.mean(), inputs.std()
        inputs = (inputs - inputs_m) / inputs_s

        # Zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Keep stats for Loss and Accuracy
        running_loss += loss.item()
        
        # Get the predicted class with the highest score
        _, prediction = torch.max(outputs,1)
        # Count of predictions that matched the target label
        correct_prediction += (prediction == labels).sum().item()
        total_prediction += prediction.shape[0]
        running_acc = correct_prediction/total_prediction
        writer.add_scalar("Train/train_accuracy",running_acc, epoch)
        writer.add_scalar("Loss/train",loss, epoch)
        experiment.log_metric("Train/train_accuracy",running_acc, epoch)
        experiment.log_metric("Loss/train",loss, epoch)
        writer.flush()
        
        if i % 5 == 0:    # print every 10 mini-batches
          try:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / i))
          except ZeroDivisionError:
            print('division by zero')
            
    # Print stats at the end of the epoch
    num_batches = len(train_dl)
    avg_loss = running_loss / num_batches
    acc = correct_prediction/total_prediction
    print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')
    experiment.log_metric("Accuracy", acc, epoch)
    writer.add_scalar("Train/epoch",epoch, epoch)
    writer.add_scalar("Train/loss",avg_loss, epoch)
    writer.add_scalar("Train/train_accuracy",acc, epoch)
    writer.flush()
  print('Finished Training')
  
num_epochs = 5 # Just for demo, adjust this higher.
training(myModel, train_dl, num_epochs)
experiment.end()

# %%
# ----------------------------
# Inference
# ----------------------------
def inference (model, val_dl):
  correct_prediction = 0
  total_prediction = 0

  # Disable gradient updates
  with torch.no_grad():
    for data in val_dl:
      # Get the input features and target labels, and put them on thresige GPU
      inputs, labels = data[0].to(device), data[1].to(device)

      # Normalize the inputs
      inputs_m, inputs_s = inputs.mean(), inputs.std()
      inputs = (inputs - inputs_m) / inputs_s

      # Get predictions
      outputs = model(inputs)
      # ic.ic(outputs)

      # Get the predicted class with the highest score
      _, prediction = torch.max(outputs,1)
      # Count of predictions that matched the target label
      correct_prediction += (prediction == labels).sum().item()
      total_prediction += prediction.shape[0]
      # ic.ic(prediction)
    
  acc = correct_prediction/total_prediction
  print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')

# Run inference on trained model with the validation set
inference(myModel, val_dl)
 
# %%
end_ = 10
for i,data in enumerate(datasetter):
  
     if i < end_: 
       if i > 3:
     
         print('----------')
         print(i)
         print('----------')
       
         print(data[0].shape)
         x = data[0].squeeze()
         ic.ic(x.shape)
         plt.imshow(x,aspect='auto', 
                    cmap='hot'
                    )
         plot_spectrogram(x)
         plot_waveform(x,22050)
        #  plot_kaldi_pitch(x)

         
     else:
         break
     






# %%
import optuna



#Trial 0 finished with value: 0.9308248370382751 and parameters: {'lr': 0.0002377344895974743, 'optimizer': 'RMSprop', 'num_epochs': 14}. Best is trial 0 with value: 0.9308248370382751.
#Epoch: 13, Loss: 0.24, Accuracy: 0.93


def objective(trial):
  
  # Create the model and put it on the GPU if available
  model = AudioClassifier()
  model = model.to(device)

  # Define the hyperparameters
  criterion = nn.CrossEntropyLoss()
  lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
  optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW','RMSprop', 'Adagrad'])
  num_epochs = trial.suggest_int('num_epochs', 2, 7)
  optimizer = getattr(torch.optim, optimizer_name)(model.parameters(),lr=lr)
  
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, cycle_momentum=False,
                                                steps_per_epoch=int(len(train_dl)),
                                                epochs=num_epochs,
                                                anneal_strategy='linear')
  
  # weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
  
 
  # Repeat for each epoch
  for epoch in tqdm(range(num_epochs)):
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0

    # Repeat for each batch in the training set
    for i, data in enumerate(train_dl):
        # Get the input features and target labels, and put them on the GPU
        inputs, labels = data[0].to(device), data[1].to(device)
        
        # Normalize the inputs
        inputs_m, inputs_s = inputs.mean(), inputs.std()
        inputs = (inputs - inputs_m) / inputs_s

        # Zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Keep stats for Loss and Accuracy
        running_loss += loss.item()
        
        # Get the predicted class with the highest score
        _, prediction = torch.max(outputs,1)
        # Count of predictions that matched the target label
        correct_prediction += (prediction == labels).sum().item()
        total_prediction += prediction.shape[0]
        running_acc = correct_prediction/total_prediction
        writer.add_scalar("Train/train_accuracy",running_acc, epoch)
        writer.add_scalar("Loss/train",loss, epoch)
        experiment.log_metric("Train/train_accuracy",running_acc, epoch)
        experiment.log_metric("Loss/train",loss, epoch)
        writer.flush()
        
        if i % 5 == 0:    # print every 10 mini-batches
          try:
            print('[%d, %5d] loss: %.3f  acc: %.2f' % (epoch + 1, i + 1, running_loss / i, running_acc))
          except ZeroDivisionError:
            print('division by zero')
            
    # Print stats at the end of the epoch
    num_batches = len(train_dl)
    avg_loss = running_loss / num_batches
    acc = correct_prediction/total_prediction
    print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')
    experiment.log_metric("Accuracy", acc, epoch)
    writer.add_scalar("Train/epoch",epoch, epoch)
    writer.add_scalar("Train/loss",avg_loss, epoch)
    writer.add_scalar("Train/train_accuracy",acc, epoch)
    writer.flush()
    print('----------------------------------------------------------------')
    print('Pruning?')
    trial.report(acc, epoch)
    if trial.should_prune():
      raise optuna.exceptions.TrialPruned()
    print('----------------------------------------------------------------')
 
  return acc

sampler = optuna.samplers.TPESampler()
study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='maximize', pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=10)

print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))

print("Best trial:")
trial = study.best_trial

print("  Trial number: ", trial.number)
print("  Loss (trial value): ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))  



# sig, sr= torchaudio.backend.sox_io_backend.load(audio_file, 
#                                          frame_offset= a, 
#                                          num_frames = b,
#                                          normalize =True, 
#                                          channels_first = True )
# %%

# %%
def print_stats(waveform, sample_rate=None, src=None):
  if src:
    print("-" * 10)
    print("Source:", src)
    print("-" * 10)
  if sample_rate:
    print("Sample Rate:", sample_rate)
  print("Shape:", tuple(waveform.shape))
  print("Dtype:", waveform.dtype)
  print(f" - Max:     {waveform.max().item():6.3f}")
  print(f" - Min:     {waveform.min().item():6.3f}")
  print(f" - Mean:    {waveform.mean().item():6.3f}")
  print(f" - Std Dev: {waveform.std().item():6.3f}")
  print()
  print(waveform)
  print()

def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].plot(time_axis, waveform[c], linewidth=1)
    axes[c].grid(True)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
    if ylim:
      axes[c].set_ylim(ylim)
  figure.suptitle(title)
  plt.show(block=False)

def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].specgram(waveform[c], Fs=sample_rate)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
  figure.suptitle(title)
  plt.show(block=False)

def play_audio(waveform, sample_rate):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  if num_channels == 1:
    display(Audio(waveform[0], rate=sample_rate))
  elif num_channels == 2:
    display(Audio((waveform[0], waveform[1]), rate=sample_rate))
  else:
    raise ValueError("Waveform with more than 2 channels are not supported.")

def inspect_file(path):
  print("-" * 10)
  print("Source:", path)
  print("-" * 10)
  print(f" - File size: {os.path.getsize(path)} bytes")
  print(f" - {torchaudio.info(path)}")

def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'Spectrogram (db)')
  axs.set_ylabel(ylabel)
  axs.set_xlabel('frame')
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  fig.colorbar(im, ax=axs)
  plt.show(block=False)

def plot_mel_fbank(fbank, title=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'Filter bank')
  axs.imshow(fbank, aspect='auto')
  axs.set_ylabel('frequency bin')
  axs.set_xlabel('mel bin')
  plt.show(block=False)

def get_spectrogram(
    n_fft = 400,
    win_len = None,
    hop_len = None,
    power = 2.0,
):
  waveform, _ = get_speech_sample()
  spectrogram = T.Spectrogram(
      n_fft=n_fft,
      win_length=win_len,
      hop_length=hop_len,
      center=True,
      pad_mode="reflect",
      power=power,
  )
  return spectrogram(waveform)

def plot_pitch(waveform, sample_rate, pitch):
  figure, axis = plt.subplots(1, 1)
  axis.set_title("Pitch Feature")
  axis.grid(True)

  end_time = waveform.shape[1] / sample_rate
  time_axis = torch.linspace(0, end_time,  waveform.shape[1])
  axis.plot(time_axis, waveform[0], linewidth=1, color='gray', alpha=0.3)

  axis2 = axis.twinx()
  time_axis = torch.linspace(0, end_time, pitch.shape[1])
  ln2 = axis2.plot(
      time_axis, pitch[0], linewidth=2, label='Pitch', color='green')

  axis2.legend(loc=0)
  plt.show(block=False)

def plot_kaldi_pitch(waveform, sample_rate, pitch, nfcc):
  figure, axis = plt.subplots(1, 1)
  axis.set_title("Kaldi Pitch Feature")
  axis.grid(True)

  end_time = waveform.shape[1] / sample_rate
  time_axis = torch.linspace(0, end_time,  waveform.shape[1])
  axis.plot(time_axis, waveform[0], linewidth=1, color='gray', alpha=0.3)

  time_axis = torch.linspace(0, end_time, pitch.shape[1])
  ln1 = axis.plot(time_axis, pitch[0], linewidth=2, label='Pitch', color='green')
  axis.set_ylim((-1.3, 1.3))

  axis2 = axis.twinx()
  time_axis = torch.linspace(0, end_time, nfcc.shape[1])
  ln2 = axis2.plot(
      time_axis, nfcc[0], linewidth=2, label='NFCC', color='blue', linestyle='--')

  lns = ln1 + ln2
  labels = [l.get_label() for l in lns]
  axis.legend(lns, labels, loc=0)
  plt.show(block=False)

DEFAULT_OFFSET = 201
SWEEP_MAX_SAMPLE_RATE = 48000
DEFAULT_LOWPASS_FILTER_WIDTH = 6
DEFAULT_ROLLOFF = 0.99
DEFAULT_RESAMPLING_METHOD = 'sinc_interpolation'

def _get_log_freq(sample_rate, max_sweep_rate, offset):
  """Get freqs evenly spaced out in log-scale, between [0, max_sweep_rate // 2]

  offset is used to avoid negative infinity `log(offset + x)`.

  """
  half = sample_rate // 2
  start, stop = math.log(offset), math.log(offset + max_sweep_rate // 2)
  return torch.exp(torch.linspace(start, stop, sample_rate, dtype=torch.double)) - offset

def _get_inverse_log_freq(freq, sample_rate, offset):
  """Find the time where the given frequency is given by _get_log_freq"""
  half = sample_rate // 2
  return sample_rate * (math.log(1 + freq / offset) / math.log(1 + half / offset))

def _get_freq_ticks(sample_rate, offset, f_max):
  # Given the original sample rate used for generating the sweep,
  # find the x-axis value where the log-scale major frequency values fall in
  time, freq = [], []
  for exp in range(2, 5):
    for v in range(1, 10):
      f = v * 10 ** exp
      if f < sample_rate // 2:
        t = _get_inverse_log_freq(f, sample_rate, offset) / sample_rate
        time.append(t)
        freq.append(f)
  t_max = _get_inverse_log_freq(f_max, sample_rate, offset) / sample_rate
  time.append(t_max)
  freq.append(f_max)
  return time, freq

def plot_sweep(waveform, sample_rate, title, max_sweep_rate=SWEEP_MAX_SAMPLE_RATE, offset=DEFAULT_OFFSET):
  x_ticks = [100, 500, 1000, 5000, 10000, 20000, max_sweep_rate // 2]
  y_ticks = [1000, 5000, 10000, 20000, sample_rate//2]

  time, freq = _get_freq_ticks(max_sweep_rate, offset, sample_rate // 2)
  freq_x = [f if f in x_ticks and f <= max_sweep_rate // 2 else None for f in freq]
  freq_y = [f for f in freq if f >= 1000 and f in y_ticks and f <= sample_rate // 2]

  figure, axis = plt.subplots(1, 1)
  axis.specgram(waveform[0].numpy(), Fs=sample_rate)
  plt.xticks(time, freq_x)
  plt.yticks(freq_y, freq_y)
  axis.set_xlabel('Original Signal Frequency (Hz, log scale)')
  axis.set_ylabel('Waveform Frequency (Hz)')
  axis.xaxis.grid(True, alpha=0.67)
  axis.yaxis.grid(True, alpha=0.67)
  figure.suptitle(f'{title} (sample rate: {sample_rate} Hz)')
  plt.show(block=True)

def get_sine_sweep(sample_rate, offset=DEFAULT_OFFSET):
    max_sweep_rate = sample_rate
    freq = _get_log_freq(sample_rate, max_sweep_rate, offset)
    delta = 2 * math.pi * freq / sample_rate
    cummulative = torch.cumsum(delta, dim=0)
    signal = torch.sin(cummulative).unsqueeze(dim=0)
    return signal

def benchmark_resample(
    method,
    waveform,
    sample_rate,
    resample_rate,
    lowpass_filter_width=DEFAULT_LOWPASS_FILTER_WIDTH,
    rolloff=DEFAULT_ROLLOFF,
    resampling_method=DEFAULT_RESAMPLING_METHOD,
    beta=None,
    librosa_type=None,
    iters=5
):
  if method == "functional":
    begin = time.time()
    for _ in range(iters):
      F.resample(waveform, sample_rate, resample_rate, lowpass_filter_width=lowpass_filter_width,
                 rolloff=rolloff, resampling_method=resampling_method)
    elapsed = time.time() - begin
    return elapsed / iters
  elif method == "transforms":
    resampler = T.Resample(sample_rate, resample_rate, lowpass_filter_width=lowpass_filter_width,
                           rolloff=rolloff, resampling_method=resampling_method, dtype=waveform.dtype)
    begin = time.time()
    for _ in range(iters):
      resampler(waveform)
    elapsed = time.time() - begin
    return elapsed / iters
  elif method == "librosa":
    waveform_np = waveform.squeeze().numpy()
    begin = time.time()
    for _ in range(iters):
      librosa.resample(waveform_np, sample_rate, resample_rate, res_type=librosa_type)
    elapsed = time.time() - begin
    return elapsed / iters
# %%


waveform, sample_rate = aud_open(df.Wav[0],
         df.Offset[0],
         df.Duration[0])
try:
    print_stats(waveform, sample_rate=sample_rate)
except:
    print('short')
plot_waveform(waveform, sample_rate)
plot_specgram(waveform, sample_rate)
play_audio(waveform, sample_rate)
plot_spectrogram(spec[0])

# %%

i=80
waveform, sample_rate = aud_open(df.Wav[i],
         df.Offset[i],
         df.Duration[i])
ic.ic(waveform.shape)
n_fft = 2048
win_length = None
hop_length = 512
n_mels = 128

mel_spectrogram = T.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
    norm='slaney',
    onesided=True,
    n_mels=n_mels,
    # mel_scale="htk",
)
chrome://history/?q=bounding%20box---', sample.shape)      

def spectro_augment_(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
  _, n_mels, n_steps = spec.shape
  ic.ic(n_mels,n_steps)
  mask_value = spec.mean()
  aug_spec = spec
  
  freq_mask_param= max_mask_pct * n_mels
  for _ in range(n_freq_masks):
    aug_spec = T.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)
    
  time_mask_param = max_mask_pct * n_steps
  for _ in range(n_freq_masks):
    
    aug_spec = T.TimeMasking(time_mask_param)(aug_spec,mask_value)         
  return aug_spec

melspec = mel_spectrogram(sample)
ic.ic(melspec.shape)
plot_spectrogram(
    melspec[0], title="MelSpectrogram - torchaudio", ylabel='mel freq')


n_fft = 2048
win_length = None
hop_length = 512
n_mels = 128
n_mfcc = 128

mfcc_transform = T.MFCC(
    sample_rate=sample_rate,
    n_mfcc=n_mfcc,
    melkwargs={
      'n_fft': n_fft,
      'n_mels': n_mels,
      'hop_length': hop_length,
      # 'mel_scale': 'htk',
    }
)

ic.ic(sample.shape)
mfcc = mfcc_transform(sample)

ic.ic(mfcc.shape)
# spec = spectro_gram(mfcc)
spec = torchaudio.transforms.AmplitudeToDB(top_db=40)(mfcc)
ic.ic(spec.shape)

spec = spectro_augment_(spec, n_freq_masks=2, n_time_masks=2)

           
      



print(mfcc.shape)
# print(mfcc)
# print(spec.shape)
plot_spectrogram(spec[0])

# %%
sample = rechannel(sample, self.channel)
sample = pad_trunc(sample, df.Duration[i])
            
sample = time_shift(sample, 0.2)
            
