from flask import Flask
from controllers import upload
from flask_cors import CORS

import os
import cv2
import pdb
import time
import warnings
import random
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, sampler
from matplotlib import pyplot as plt
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.pytorch import ToTensor
warnings.filterwarnings("ignore")

final = torch.load("/home/rath772k/skyf.pth", map_location=torch.device("cpu"))
state = torch.load('/home/rath772k/model.pth', map_location=torch.device("cpu"))
final.load_state_dict(state["state_dict"])
final.to(torch.device("cpu"))
final.eval()

app = Flask(__name__, static_url_path='/static')

CORS(app)

@app.route('/upload', methods=['POST'])
def handleUpload():
    return upload.Upload()

@app.route('/<path:path>')
def handleStatic(path):
    return app.send_static_file(path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
