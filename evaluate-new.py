#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import sys
import os
import shutil
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt


def display_rgb_image(im: np.ndarray):
    plt.imshow(im, interpolation='nearest')
    plt.show()


def conv_to_rgba(im):
    ret = np.empty_like(im)
    Y, Co, Cg = im[:, :, 0], im[:,:,1], im[:,:,2]
    ret[:, :, 0] = Co + Y + (1 - Cg >> 1) - (Co >> 1) #R
    ret[:, :, 1] = Y - ((-Cg) >> 1) #G
    ret[:, :, 2] = Y + ((1 - Cg) >> 1) - (Co >> 1) #B
    return ret

output_directory = 'C:/Users/Varun/Desktop/Homeworks/696ds/eval_folder'
if os.path.exists(output_directory):
    shutil.rmtree(output_directory)
os.makedirs(output_directory)

filenames = [filename for filename in os.listdir("kodak_dump/") if "9" in filename]
for filename in filenames:
    data_dump = pd.read_csv("kodak_dump/"+filename, dtype = {'original_ycocg': int, 'srec_prediction_ycocg': int, 'flif_scanline_prediction_ycocg': int})  #Add interlaced
    
    #Get RGB values
    original_rgb = pd.Series(data_dump['original_ycocg']).to_numpy()
    original_rgb = np.transpose(np.reshape(original_rgb,(3, 512, 768)), (1,2,0)).astype("uint8")
    original_rgb = conv_to_rgba(original_rgb)
    display_rgb_image(original_rgb)
    data_dump['original_rgb'] = original_rgb.ravel()
    
    srec_prediction_rgb = pd.Series(data_dump['srec_prediction_ycocg']).to_numpy()
    srec_prediction_rgb = np.transpose(np.reshape(srec_prediction_rgb,(3, 512, 768)), (1,2,0)).astype("uint8")
    srec_prediction_rgb = conv_to_rgba(srec_prediction_rgb)
    data_dump['srec_prediction_rgb'] = srec_prediction_rgb.ravel()
    
    flif_scanline_prediction_rgb = pd.Series(data_dump['flif_scanline_prediction_ycocg']).to_numpy()
    flif_scanline_prediction_rgb = np.transpose(np.reshape(flif_scanline_prediction_rgb,(3, 512, 768)), (1,2,0)).astype("uint8")
    flif_scanline_prediction_rgb = conv_to_rgba(flif_scanline_prediction_rgb)
    data_dump['flif_scanline_prediction_rgb'] =flif_scanline_prediction_rgb.ravel()
    
#     flif_interlaced_prediction_rgb = pd.Series(data_dump['flif_interlaced_prediction_ycocg']).to_numpy()
#     flif_interlaced_prediction_rgb = np.transpose(np.reshape(flif_interlaced_prediction_rgb,(3, 512, 768)), (1,2,0)).astype("uint8")
#     flif_interlaced_prediction_rgb = conv_to_rgba(flif_interlaced_prediction_rgb)
#     data_dump['flif_interlaced_prediction_rgb'] =flif_interlaced_prediction_rgb.ravel()
    
    # Get FLIF_ycocg result
    flif_scanline_se_ycocg = pd.Series((data_dump['original_ycocg'] - data_dump['flif_scanline_prediction_ycocg'])**2).to_numpy()
    flif_scanline_se_ycocg = np.transpose(np.reshape(flif_scanline_se_ycocg,(3, 512, 768)), (1,2,0)).astype("uint8")
    data_dump['flif_scanline_se_ycocg'] = flif_scanline_se_ycocg.ravel()
    
#     flif_interlaced_se_ycocg = pd.Series((data_dump['original_ycocg'] - data_dump['flif_interlaced_prediction_ycocg'])**2).to_numpy()
#     flif_interlaced_se_ycocg = np.transpose(np.reshape(flif_interlaced_se_ycocg,(3, 512, 768)), (1,2,0)).astype("uint8")
#     data_dump['flif_interlaced_se_ycocg'] = flif_interlaced_se_ycocg.ravel()

    #Get SReC_ycocg result
    srec_se_ycocg = pd.Series((data_dump['original_ycocg'] - data_dump['srec_prediction_ycocg'])**2).to_numpy()
    srec_se_ycocg = np.transpose(np.reshape(srec_se_ycocg,(3, 512, 768)), (1,2,0)).astype("uint8")
    data_dump['srec_se_ycocg'] = srec_se_ycocg.ravel()
    
    data_dump['SReC-FLIF_scanline_ycocg'] = data_dump['srec_se_ycocg'] - data_dump['flif_scanline_se_ycocg']
    #data_dump['SReC-FLIF_interlaced_ycocg'] = data_dump['srec_se_ycocg'] - data_dump['flif_interlaced_se_ycocg']
    
    
    # Get FLIF_rgb result
    flif_scanline_se_rgb = pd.Series((data_dump['original_rgb'] - data_dump['flif_scanline_prediction_rgb'])**2).to_numpy()
    flif_scanline_se_rgb = np.transpose(np.reshape(flif_scanline_se_rgb,(3, 512, 768)), (1,2,0)).astype("uint8")
    data_dump['flif_scanline_se_rgb'] = flif_scanline_se_rgb.ravel()
    
#     flif_interlaced_se_rgb = pd.Series((data_dump['original_rgb'] - data_dump['flif_interlaced_prediction_rgb'])**2).to_numpy()
#     flif_interlaced_se_rgb = np.transpose(np.reshape(flif_interlaced_se_rgb,(3, 512, 768)), (1,2,0)).astype("uint8")
#     data_dump['flif_interlaced_se_rgb'] = flif_interlaced_se_rgb.ravel()

    #Get SReC_rgb result
    srec_se_rgb = pd.Series((data_dump['original_rgb'] - data_dump['srec_prediction_rgb'])**2).to_numpy()
    srec_se_rgb = np.transpose(np.reshape(srec_se_rgb,(3, 512, 768)), (1,2,0)).astype("uint8")
    data_dump['srec_se_rgb'] = srec_se_rgb.ravel()
    
    data_dump['SReC-FLIF_scanline_rgb'] = data_dump['srec_se_rgb'] - data_dump['flif_scanline_se_rgb']
#     data_dump['SReC-FLIF_interlaced_rgb'] = data_dump['srec_se_rgb'] - data_dump['flif_interlaced_se_rgb']

    out_path = output_directory + '/' + 'eval_' + filename
    data_dump.to_csv(out_path, index = False)

