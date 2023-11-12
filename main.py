import os
import numpy as np

try:
    import tensorflow  # required in Colab to avoid protobuf compatibility issues
except ImportError:
    pass

import torch
import pandas as pd
import whisper
import torchaudio

# print("available: " + str(torch.cuda.is_available()))
from tqdm.notebook import tqdm
# torch.cuda.empty_cache()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = whisper.load_model("medium", device=DEVICE)
result = model.transcribe("test.m4a")
print(result["text"])