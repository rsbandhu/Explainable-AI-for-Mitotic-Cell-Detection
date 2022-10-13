
# Import Libraries
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import cv2 as cv
import sys
import os
from PIL import Image
from torchvision import models
from torchvision import transforms
from torchvision.models import resnet50
import torchvision.models as models
from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np
import streamlit as st


# Device used (CUDA or CPU)
def get_device():
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("Using " + torch.cuda.get_device_name(device))
    device = torch.device("cpu")
    return device

# Load pretrained Model 
PATH = '/Users/yonastesh/Desktop/final_streamlit10922/Resnet50 models/resnet50_all_class.pt'

#@st.cache()
@st.cache(allow_output_mutation=True)
def load_model(file_path):
    model = models.resnet50(pretrained=True)   
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 4)
    device = torch.device('cpu')
    model.load_state_dict(torch.load(PATH, map_location=device))
    return model

# Image Transformation Functions
#@st.cache()
@st.cache(allow_output_mutation=True)
def img_transform(path):
    transform = transforms.Compose([
     transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor()
    ])
    transform_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
     )
    img = Image.open(path)
    transformed_img = transform(img)
    input = transform_normalize(transformed_img)
    input = input.unsqueeze(0)

    return input

# Image transformation 
#@st.cache()
@st.cache(allow_output_mutation=True)
def img_trans(path):
    '''
    Image transform without returning an input
    '''
    transform = transforms.Compose([
     transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor()
    ])
    img = Image.open(path)
    img_r = transform(img)
    return img_r

# Image transforamtion
#@st.cache()
@st.cache(allow_output_mutation=True)
def load_image(img, model, device):

    img_input = img_transform(img)
    model.eval()
    model = model.to(device)
    img_input = img_input.to(device)
    img_cuda_input = img_input.to(device)
    output = model(img_input) 
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)
    pred_label_idx.squeeze_()
    predicted_label = str(pred_label_idx.item())
    print(output)
    print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')
    return img_input, pred_label_idx, predicted_label, prediction_score

    
#@st.cache()
@st.cache(allow_output_mutation=True)
def get_image(path):
    '''
    Open up the image and display it
    '''
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB') 

        
#@st.cache()   
@st.cache(allow_output_mutation=True)   
def get_pil_transform(): 
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])    
    return transf
   
# Image preprocessing
#@st.cache()
@st.cache(allow_output_mutation=True)
def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])     
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])    
    return transf 

# Batch Prediction
#@st.cache()
@st.cache(allow_output_mutation=True)
def batch_predict(images):
    model = load_model(file_path='resnet50_all_class.pt')
    preprocess_transform = get_preprocess_transform()
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)
    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()
