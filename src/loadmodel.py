import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torchaudio
# import torchaudio.functional as F
import torchaudio.transforms as T

# Model Class
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


# transfroms preprocessing
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
def rechannel(aud, new_channel = 1):
  sig = aud
  if (sig.shape[0]==new_channel):
    return aud

  elif(sig.shape[0]== 2): #shift to mono
    sig = sig[:1, :]
  elif(sig.shape[0] == 1):
    sig = torch.cat([sig, sig])
  
  return sig
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

def preprocess_audio(x):
    x = rechannel(sample, channel)
    x = mfcc_transform(x)
    x = T.AmplitudeToDB(top_db=40)(x)
    x = spectro_augment(x, n_frgment(x, n_freq_masks=2. n_time_masks=2)
    return x 


#actual prediction function
def predict_chunk(sample):
    with torch.no_grad():

        # Create the model and put it on the GPU if available
        myModel = AudioClassifier()
        myModel = load_state_dict(torch.load(PATH))
        myModel.eval()
        myModel = myModel.to(device)


        x = preprocess_audio(sample)
        prediction = myModel(x)
        predicted_class = torch.argmax(prediction)
        
        #predictions should populate a list which then checks the frequency--a poor mans voting classifier
        print(predicted_class)
        q.put(predicted_class)
        return predicted_class

        


# %%%
with torch.no_grad():

# Create the model and put it on the GPU if available
myModel = AudioClassifier()
myModel = load_state_dict(torch.load(PATH))
myModel.eval()
myModel = myModel.to(device)


x = preprocess_audio(sample)
prediction = myModel(x)
predicted_class = torch.argmax(prediction)

#predictions should populate a list which then checks the frequency--a poor mans voting classifier
print(predicted_class)
q.put(predicted_class)
return predicted_class
