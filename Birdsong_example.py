
import comet_ml
from comet_ml import Experiment
import os
import glob
from tqdm import tqdm
import pandas as pd
import datetime
from datetime import datetime
import time
import matplotlib.pyplot as plt import icecream as ic
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

# look at metadatacsv
df = pd.read_csv('/media/gyasis/Drive 2/Data/birdsong/metadata.csv')
print(df.columns)
df = df[['Recording_ID','Audio_file','Species','Path', 'Length']]
torch.cuda.empty_cache() 
def fix_path(x):
    main_path = '/media/gyasis/Drive 2/Data/birdsong/mp3/'
    x = os.path.split(x)
    x = os.path.join(main_path,x[1])
    return x
    
df['Path'] = df['Path'].apply(lambda x: fix_path(x))


species = df.Species.unique()
print(len(species))

def get_seconds(x):
    x = datetime.strptime(x,'%M:%S')
    #remove date from data time?
    a_timedelta = x - datetime(1900, 1, 1)
    seconds = a_timedelta.total_seconds()
    return seconds

df['Total_sec'] = df['Length'].apply(lambda x: get_seconds(x))

#function histogram for df.Total_sec
def histogram(x):
    plt.hist(x, bins=100, range=(0, 750))
    plt.show()
histogram(df.Total_sec)


# last, categorical encoding
from sklearn.preprocessing import LabelEncoder
l_e = LabelEncoder()
df['Class'] = l_e.fit_transform(df['Species'])


def point_path(x):
    wav_path = '/media/gyasis/Drive 2/Data/birdsong/wav/'
    for i,file in enumerate(df.Path):
        x = os.path.split(x)
        x = x[1].replace("mp3", "wav")
        new_path = os.path.join(wav_path, x)
        return new_path
        
        
df['Wav'] = df['Path'].apply(lambda x: point_path(x))

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



# checking target column splits and 
def prepare_class_split(dataframe, target="class_name", p_split=0.30, test_target_split=0.50):
  dataframe = dataframe.copy()
  df_len = len(dataframe)
  class_amount = len(dataframe[target].unique())
  df_split = int(df_len * p_split)
  class_list = list(dataframe[target].unique())
  print(class_list)
  proposed_split = df_split/class_amount
  
  class_counts = dataframe[target].value_counts()
  # print(df_len,df_split,proposed_split,class_counts)
  
  outcomes = []
  
  
  print("Total of Test Split is {} and Proposed split is {}".format(df_split,proposed_split))
  
  
  for lable in class_list:
    percent_split = class_counts[lable] / df_len
    proposed_percent_split = class_counts[lable] / df_split
    
    if class_counts[lable] >= proposed_split * 2:
      print("Class {} is OK!!".format(lable))
      outcomes.append("OK!!")
    elif class_counts[lable] < proposed_split * 2 and class_counts[lable] > proposed_split:
      print("Class {} fails equity threshold, look to augment training dataset ".format(lable))
      outcomes.append("Augment??")
    elif class_counts[lable] < proposed_split:
      print("Class {} is {} and Proposed split is {}".format(lable,class_counts[lable],proposed_split))
      print("Class " + lable + " is less than the proposed split")
      print("Class {} is {} and the proposed split is {}".format(lable,class_counts[lable],proposed_split))
      print("Both augmentation and weights may be necessary!!")
      outcomes.append("Weights/Augment/Split!!")
      
def get_class_frequencies(dataframe,target):
  try:
    dataframe = pd.get_dummies(dataframe[target].astype(str))
  except:
    dataframe = pd.get_dummies(dataframe[target])
    
  f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
  sample_array = np.array(dataframe)
  positive_freq = sample_array.sum(axis=0) / sample_array.shape[0]
  negative_freq = np.ones(positive_freq.shape) - positive_freq
  data = pd.DataFrame({"Class": dataframe.columns, "Label": "Positive", "Value": positive_freq})
  data = data.append([{"Class": dataframe.columns[l], "Label": "Negative", "Value": v} for l, v in enumerate(negative_freq)], ignore_index=True)
  plt.xticks(rotation=90)
  sns.barplot(x="Class", y="Value",hue="Label", data=data, ax=ax1)
  pos_weights = negative_freq
  neg_weights = positive_freq
  pos_contribution = positive_freq * pos_weights
  neg_contribution = negative_freq * neg_weights
  # print("Weight to be added:  ",pos_contribution)
  
  data1 = pd.DataFrame({"Class": dataframe.columns, "Label": "Positive", "Value": pos_contribution})
  data1 = data1.append([{"Class": dataframe.columns[l], "Label": "Negative", "Value": v} for l, v in enumerate(neg_contribution)], ignore_index=True)
  ax1.tick_params(axis='x', labelrotation=90)
  ax2.tick_params(axis='x', labelrotation=90)
  sns.barplot(x="Class", y="Value",hue="Label", data=data1, ax=ax2)
  
  return pos_contribution





experiment = Experiment(api_key="xleFjfKO3kcwc56tglgC1d3zU",
                        project_name="Bird_calls",log_code=True)

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

#Log audio files to Comet for debugging
#get random sample of pandas dataframe
def get_random_sample(df, n=1):
    return df.sample(n=n)
df2 = get_random_sample(df, n=50)


df2.reset_index(drop=True, inplace=True)

import random
for i, file in enumerate(df2.Wav):
    print(i)
    single_file = aud_open(df2.Wav[i],df2.Offset[i], df2.Duration[i], show_stats=False)
    temp,_ = single_file
    temp1 = temp.numpy()
    randomnumber = random.randint(1,99999999)

    temp = rechannel(temp, 1)
    experiment.log_audio(temp.T, 
                         sample_rate=22050, 
                         copy_to_tmp=False,
                         file_name = df2.Species[i])
    print("Logging ",df2.Species[i])


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
            
            #chain --> moving and transforming data
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

datasetter = MyDataset(df)
sampleset = MyDataset(df2)

set_batchsize = 10

from torch.utils.data import random_split
num_items = len(datasetter)
num_train = round(num_items * 0.7)
num_val = num_items - num_train
train_ds, val_ds = random_split(datasetter, [num_train, num_val])
train_dl = DataLoader(train_ds, batch_size=set_batchsize,num_workers=4,pin_memory=True, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=set_batchsize, num_workers=4,pin_memory=True, shuffle=False)
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
viz_model = AudioClassifier()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
myModel = myModel.to(device)


# Check that it is on Cuda
# next(myModel.parameters()).device

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from torchviz import make_dot
 
# this uses a batch to interate for the visualizer to work
fake_batch = next(iter(train_dl))
yhat = viz_model(fake_batch[0])
make_dot(yhat, params=dict(list(viz_model.named_parameters()))).render("example_model", format="png")
from IPython.display import Image
Image(filename="example_model.png")

# ----------------------------
# Training Loop
# ----------------------------
def training(model, train_dl, num_epochs):
  # Loss Function, Optimizer and Scheduler
  optimizer_name = torch.optim.Adam(model.parameters(),lr=0.005279173305530663,weight_decay=0.0001)
  criterion = nn.CrossEntropyLoss()
  optimizer = optimizer_name
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.006,
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
        
        # if i % 5 == 0:    # print every 10 mini-batches
        #   try:
        #     print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / i))
        #   except ZeroDivisionError:
        #     print('division by zero')
            
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
  
num_epochs = 18 # Just for demo, adjust this higher.
training(myModel, train_dl, num_epochs)
experiment.end()

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
 
end_ = 10
for i,data in enumerate(sampleset):
  
     if i < end_: 
       if i > 3:
     
         print('----------')
         print(i)
         print('----------')
       
         print(data[0].shape)
         x = data[0].squeeze()
        #  ic.ic(x.shape)
        #  plt.imshow(x,aspect='auto', 
        #             cmap='hot'
        #             )
         plot_spectrogram(x)
        #  plot_waveform(x,22050)
        #  plot_kaldi_pitch(x)
         
     else:
         break
     

torch.save(myModel.state_dict(), 'model.pt')


import optuna

def objective(trial):
  
  # Create the model and put it on the GPU if available
  model = AudioClassifier()
  model = model.to(device)
  # Define the hyperparameters
  criterion = nn.CrossEntropyLoss()
  lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
  optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW','RMSprop', 'Adagrad'])
  num_epochs = trial.suggest_int('num_epochs', 5, 15)
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
    for i, data in enumerate(tqdm(train_dl)):
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


#hyperparameter optimization
sampler = optuna.samplers.TPESampler()
study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='maximize', pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=15)
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



#save trained model with hyperparameters
def save_model(model, optimizer, scheduler, model_name, epochs, loss, best_loss,
               save_dir='./models/'):
  model_dir = os.path.join(save_dir, model_name)
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  model_path = os.path.join(model_dir, 'model.pt')
  optimizer_path = os.path.join(model_dir, 'optimizer.pt')
  scheduler_path = os.path.join(model_dir, 'scheduler.pt')
  loss_path = os.path.join(model_dir, 'loss.pt')
  best_loss_path = os.path.join(model_dir, 'best_loss.pt')
  torch.save(model.state_dict(), model_path)
  torch.save(optimizer.state_dict(), optimizer_path)
  torch.save(scheduler.state_dict(), scheduler_path)
  torch.save(loss, loss_path)
  torch.save(best_loss, best_loss_path)
  print('Saved model to: {}'.format(model_path))
