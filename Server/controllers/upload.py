from flask import flash, request, jsonify
from werkzeug.utils import secure_filename
from albumentations import Resize
from main import final as model

import requests 
import string 
import random 
  
# initializing size of string  
N = 7
  
# using random.choices() 
# generating random strings  

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

UPLOAD_FOLDER = '/home/rath772k/temp/static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#def generate_angle(mask,gamma,fov):
#  last_i,last_j=0,0
#
#  for i in range(mask.shape[0]):
#    for j in range(mask.shape[1]):
#      if(mask[i][j]==1):
#        last_i,last_j = i,j
#        break
#  angle_from_perpendicular = np.arctan(((last_i-(256))*2 )*0.5/256
#                                       * np.tan(fov*np.pi/180/2))
#  gamma = gamma + angle_from_perpendicular*180/np.pi - 90
#
#  return np.abs(gamma)

def generate_angle(mask,gamma,fov):
    angles = np.zeros((mask.shape[1],))
    for j in range(mask.shape[1]):
        for i in range(mask.shape[0]-1,0,-1):
            if(mask[i][j]!=0):
                angle_from_perpendicular = np.arctan(np.tan(fov/2) * np.abs(i-256)/256)
                if(i>256):
                    angle_from_perpendicular *= -1
                angles[j] = gamma + angle_from_perpendicular*180/np.pi - 90
                break
    return angles

def post_process(mask, min_size):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((512, 512), np.int32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num

def Upload():
    if request.method == 'POST':
        #if 'url' not in request.orgs:
         #   return jsonify({"error": "No image url in the request"}), 400
        url = request.form['url']
        print(url)
        #if url and allowed_file(url):
        # URL of the image to be downloaded is defined as image_url 
        r = requests.get(url) # create HTTP response object 

        # send a HTTP request to the server and save 
        # the HTTP response in a response object called r 
        extension = ''
        if '.jpg' in url:
            extension = '.jpg'
        if '.png' in url:
            extension = '.png'
        if '.jpeg' in url:
            extension = '.jpeg'
            
        res = ''.join(random.choices(string.ascii_uppercase +
                             string.digits, k = N))
        randomStr = str(res)
        with open("/home/rath772k/temp/static/" + randomStr + extension,'wb') as f: 

            # Saving received content as a png file in 
            # binary format 

            # write the contents of the response (r.content) 
            # to a new file in binary mode. 
            f.write(r.content) 

        newName = "/home/rath772k/temp/static/" + randomStr + extension
        filename = newName
        # call the ML model
        im = cv2.imread(newName)
        size = 512
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = Compose([
            Normalize(mean=mean, std=std, p=1),
            Resize(size, size),
            ToTensor(),
        ])
        image_in = transform(image=im)["image"]
        image_in = image_in.reshape(1, 3, 512, 512).to(torch.device("cpu"))
        model.eval()
        with torch.no_grad():
            op = model(image_in)
            op = op.cpu().numpy()
        op = (op > 0) + 0
        op = (op == 1) + 0
        op = op.reshape((512, 512))
        op, _ = post_process(op, 50)
        op = op.astype('uint8')
        op = op * 255
        alpha = 0.6
        im = cv2.resize(im, (512, 512))
        redImg = np.zeros(im.shape, im.dtype)
        bluImg = redImg
        redImg[:, :] = (0, 0, 255)
        redMask = cv2.bitwise_and(redImg, redImg, mask=op)
        final_image = cv2.addWeighted(redMask, alpha, im, 1 - alpha, 0, im)
        cv2.imwrite(filename, final_image)
        
        angleOfElevation = generate_angle(op, 120, 0.78)
        fig, ax = plt.subplots( nrows=1, ncols=1 )
        ax.plot(angleOfElevation)
        angle_plot_image_filename=os.path.join(UPLOAD_FOLDER, "angle_plot_img.jpg")
        fig.savefig(angle_plot_image_filename,bbox_inches='tight')

        responseImgPath = "http://34.69.240.165:3000/" + randomStr + extension
        return jsonify({"segmentedImagePath": responseImgPath, "angle_plot_img_path": "http://34.69.240.165:3000/angle_plot_img.jpg"}), 200