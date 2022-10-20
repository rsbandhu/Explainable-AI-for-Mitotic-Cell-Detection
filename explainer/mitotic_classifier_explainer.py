import attr
from mitosis_phase_classifier.classifier_model import get_model
#from importlib_metadata import re
#import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

import sys
sys.path.append('../../')

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
#from torchvision import models, transforms

from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz



def attribute_image_features(algorithm, input, model, target, **kwargs):
    """
    A generic function that will be used for calling attribute on attribution algorithm 
    defined in input.
    """
    model.zero_grad()
    tensor_attributions = algorithm.attribute(input,
                                              target=target,
                                              **kwargs
                                              )

    return tensor_attributions


def saliency_map(img, model, target):

    saliency = Saliency(model)
    grads = saliency.attribute(img, target=target)
    grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))
    #grads.shape

    return grads


def integrated_gradients(img, model, target):

    ig = IntegratedGradients(model)
    attr_ig, delta = attribute_image_features(
        ig, img, model, target, baselines=img * 0, return_convergence_delta=True)
    attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
    print('Approximation delta: ', abs(delta))

    return attr_ig


def occlusion(img, model, target):
    
    occlusion = Occlusion(model)
    attributions_occ = occlusion.attribute(img,
                        strides=(3, 5, 5), target=target, 
                        sliding_window_shapes=(3, 12, 12), baselines=0)

    attributions_occ = np.transpose(attributions_occ.squeeze(), (1, 2, 0))

    return attributions_occ


def gradient_shap(img, model, target):

    torch.manual_seed(0)
    np.random.seed(0)

    gradient_shap = GradientShap(model)
    input = img

    # Defining baseline distribution of images
    rand_img_dist = torch.cat([input * 0, input * 1])

    attributions_gs = gradient_shap.attribute(input,
                                            n_samples=50,
                                            stdevs=0.0001,
                                            baselines=rand_img_dist,
                                            target=target)
    
    return attributions_gs

def read_image(img_file, data_dir):
    img = Image.open(os.path.join(data_dir, img_file))


    # specify the transforms on the image
    img = img.convert("RGB")

    img_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
    return img_transforms(img)

def classifier_model(model_weights, class_names):

    # get classifier model and set it to eval mode
    classifier_model = models.resnet50(pretrained=True)
    num_features = classifier_model.fc.in_features

    #class_names = ['1_prophase', '2_metaphase', '3_anaphase', '5_background']
    # Change the final classification layer, to the number of classes
    classifier_model.fc = nn.Linear(num_features, len(class_names))

    classifier_model.load_state_dict(torch.load(model_weights))
    
    return classifier_model

def get_explainers(img_file, data_dir, class_names):

    model = get_model('resnet50_all_class.pt')
    model.eval()

    img = read_image(img_file, data_dir)
    img = img.unsqueeze(0)
    output = model(img)
    _, pred = torch.max(output, 1)
    target = pred.cpu().detach()[0]
        
    saliency=saliency_map(img, model, target)
    attr_ig = integrated_gradients(img, model, target)
    attr_gs = gradient_shap(img, model, target)
    attributions_occ = occlusion(img, model, target)

    img_orig = np.transpose(img.squeeze().cpu().detach().numpy(), (1, 2, 0))
    attr_gs = np.transpose(attr_gs.squeeze().cpu().detach().numpy(), (1, 2, 0))

    attributions_occ = attributions_occ.cpu().detach().numpy()

    print("shappe of attr_occ: ", attributions_occ.shape, attr_gs.shape)
    return img_orig, saliency, attr_ig, attr_gs, attributions_occ
