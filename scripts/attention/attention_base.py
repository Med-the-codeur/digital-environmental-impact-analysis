
## Standard libraries
import os
import numpy as np
import random
import math
import json
from functools import partial

## Imports for plotting
import matplotlib.pyplot as plt
#plt.set_cmap('cividis')
#%matplotlib inline
#from IPython.display import set_matplotlib_formats
#set_matplotlib_formats('svg', 'pdf') # For export
#from matplotlib.colors import to_rgb
#import matplotlib
#matplotlib.rcParams['lines.linewidth'] = 2.0
#import seaborn as sns
#sns.reset_orig()

## tqdm for loading bars
from tqdm.notebook import tqdm


## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

## Torchvision
#import torchvision
#from torchvision.datasets import CIFAR100
#from torchvision import transforms

#from codecarbon import EmissionsTracker

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
#DATASET_PATH = "../data"
# Path to the folder where the pretrained models are saved
#CHECKPOINT_PATH = "../saved_models/tutorial6"

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Setting the seed
seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

from codecarbon import OfflineEmissionsTracker

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    #tracker = EmissionsTracker()
    #tracker.start()
    with OfflineEmissionsTracker(country_iso_code="TN") as tracker:
        # GPU intensive training code goes here
        #tracker.start()
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        #emissions = tracker.stop()
        #print(f"Emissions: {emissions} kg CO₂")
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
    return values, attention

seq_len, d_k = 300, 200
seed_everything(42)
q = torch.randn(seq_len, d_k)
k = torch.randn(seq_len, d_k)
v = torch.randn(seq_len, d_k)
print("scaled_dot_product() starts")
values, attention = scaled_dot_product(q, k, v)
print("scaled_dot_product() ends")
#print("Q\n", q)
#print("K\n", k)
#print("V\n", v)
#print("Values\n", values)
#print("Attention\n", attention)
