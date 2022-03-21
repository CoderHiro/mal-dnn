import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


if __name__ == "__main__": 
    data_path = '/home/cgh/pycode/mal-dnn/data/normal/'
    save_path = '/home/cgh/pycode/mal-dnn/data/embed_normal/'
    print(len(os.listdir(data_path)))
    print(len(os.listdir(save_path)))