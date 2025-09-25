from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from scipy.signal import butter, lfilter, stft
from torch.utils.data import Dataset, DataLoader
import os
import glob
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from zipfile import ZipFile
import matplotlib.pyplot as plt
import triton
import triton.language as tl

import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torchaudio
from scipy.signal import butter, lfilter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SEQ_LEN = 500   # number of samples per example after downsampling
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")