#from dis import dis
from re import A
import matplotlib.image as mpimg

from explainer.mitotic_detections_summary import summarize_detections, detection_bbox_width_height, generate_mitotic_count_10hpf
from explainer.explainer_plots import plot_image_attribution_heatmap, visualize_image_attr, plot_attributions, plot_width_height, plot_count_heatmap

from captum.attr import visualization as viz
from image_utils.plot_detections import plot_all_detections_in_region, plot_all_detections_in_low_mag, get_detection_locations
import argparse
from http import server
import streamlit as st
import pandas as pd
import numpy as np
import time

from pathlib import Path
from multiprocessing import Pool
from functools import partial

import os
import openslide   

import sys
sys.path.append('../')
import torch

#print(repr(sys.path))
from PIL import Image
import plotly.express as px

import streamlit.components.v1 as components

from image_utils.image_utils import calculate_patch_count_in_WSI, image_stats

from yolov7.detect_mitotic_wsi import detect
from mitosis_phase_classifier.detect_mitotic_phases import detect_mitotic_phases 
from explainer.mitotic_classifier_explainer import get_explainers

@st.cache
def upload_image(image_file, detects_dir):
    if image_file is not None:
        # see image details
        file_details = {"FileName": image_file.name,
                        "FileType": image_file.type}

        #Saving image file in a WSI folder
        image_name = image_file.name
        image_folder = os.path.join(detects_dir, image_name.split('.')[0])

        if not os.path.exists(image_folder):
            os.mkdir(image_folder)

        save_img_path = os.path.join(image_folder, image_name)
        with open(save_img_path, "wb") as f:
            f.write(image_file.getbuffer())
    
    return save_img_path, image_name


def display_image(save_img_path):
    # display the image in the lowest resolution
    
    slide_path = save_img_path
    #slide_path = str(os.path.join('WSI', file_name))
    #file_type = file_name.split('.')[-1]
    file_type = 'svs'
    if file_type == 'svs':
        slide = openslide.open_slide(str(slide_path))
        slide_dims = slide.level_dimensions
        st.write('Number of magnification levels = ', slide.level_count)
        for i, x in enumerate(slide_dims):
            st.write('level = ', i, '   Slide dimensions: ', x)
        disp_size = slide_dims[-1] 
        img = slide.read_region((0, 0), level=2, size=disp_size)
        print(f"size of slide: {img.size}")
    elif file_type in ['jpg', 'jpeg', 'png']:
        img = Image.open(slide_path)
        img.load()
        print(f"size of slide: {img.size}")

    print(type(img))

    return img

def save_single_image_patch(save_dir, patch_size, mean_threshold, n1, n2):
    """
    For a patch in a WSI indexed by (n1, n2)
    save the patch image 
    the top left corner of the patch has coordinates: (n1*patch_size, n2*patch_size)

    Inputs:
    save_dir: directory to save the patch image
    patch_size: size of the patch
    n1: X-index of the patch top left corner, units of patch_size
    n2: Y-index of the patch top left corner, units of patch_size

    """

    # define the coordinates of top left corner of the patch in the WSI
    (x_topleft, y_topleft) = (n1*patch_size, n2*patch_size)

    # get the patch as numpy array
    img = np.asarray(slide.read_region((x_topleft, y_topleft), level=0, size=(patch_size, patch_size)))
    #img = wsi_img[x_topleft:x_topleft+patch_size, y_topleft:y_topleft+patch_size, :3]
    img = img[:, :, :3]  # ignore the 4th channel

    # calculate the statistics for each channel
    (img_mean, img_std_dev) = image_stats(img)

    if np.min(img_mean) < mean_threshold:
        # create image patch names and save image patch
        patch_name = f"{filename.split('.')[0]}_{str(n1)}_{str(n2)}.jpeg"
        filename_save = f"{save_dir}/{patch_name}"
        #print(len(labels), patch_name)
        im = Image.fromarray(img)
        im.save(filename_save)
        #print(img.shape)

def save_image_patches(wsi_img, save_dir, patch_size, mean_threshold):
    
    wsi_x = wsi_img.shape[0]
    wsi_y = wsi_img.shape[1]

    for i in range(int(wsi_x/patch_size)):
        for j in range(int(wsi_y/patch_size)):

            (x_topleft, y_topleft) = (i*patch_size, j*patch_size)
            
            img = wsi_img[x_topleft:x_topleft+patch_size, y_topleft:y_topleft+patch_size, :3]
            (img_mean, img_std_dev) = image_stats(img)

            if np.min(img_mean) < mean_threshold:
                
                patch_name = f"{filename.split('.')[0]}_{str(i)}_{str(j)}.jpeg"
                filename_save = f"{save_dir}/{patch_name}"
                #print(len(labels), patch_name)
                im = Image.fromarray(img)
                im.save(filename_save)
        print(i)

@st.cache(allow_output_mutation=True)
def mitotic_detections(opt):

    all_mitotic_detections = detect(opt)
    return all_mitotic_detections

@st.cache
def mitotic_phase_classify(opt, mitotic_detections):


    t0 = time.time()
    model_clsf = 'resnet50_all_class.pt'
    all_mitotic_detections = detect_mitotic_phases( data_clsf_dir,  model_clsf, mitotic_detections)

    print(f"Classification time : {time.time()-t0}")

    return all_mitotic_detections


@st.cache
def store_detections_to_file(detections):
    f = open(f"{detects_dir}/{filename.split('.')[0]}/all_detections.csv", "w")

    # Write all the detections and phase classifications to an output file
    for k, v in all_mitotic_detections.items():
        line = k
        for item in v[:4]:
            line = line + ','+str(int(item))
        line = line + ',' + str(v[4])
        line = line + ',' + str(int(v[5]))
        line = line + ',' + str(int(v[6])) + '\n'
        f.write(line)
    f.close()

#if __name__ == '__main__':

# Define some global parameters and settings
parser = argparse.ArgumentParser()
# file/folder, 0 for webcam
parser.add_argument('--source', type=str, default='inference/images', help='source')
parser.add_argument('--data-detections-dir', type=str, default='inference/images', help='data where image patches are stored')
parser.add_argument('--image-size-detection', type=int, default=320, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
parser.add_argument('--weights', nargs='+', type=str,
                    default='yolov7/pre_trained_models/yolov7_exp_A_10_best.pt', help='model.pt path(s)')
parser.add_argument('--batch-size', type=int, default=64, help='batch szie for object detection')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--img-mean-thres', type=float, default=235, help='image channel mean value threshold')
parser.add_argument('--view-img', action='store_true', help='display results')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--update', action='store_true', help='update all models')
parser.add_argument('--project', default='runs/detect', help='save results to project/name')
parser.add_argument('--name', default='exp', help='save results to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
parser.add_argument('--no-trace', action='store_true', help='don`t trace model')

parser.add_argument('--data-clsf-dir', type=str, default='inference/images', help='source')
parser.add_argument('--image-size-clsf', type=float, default=80, help='image height & width for mitosis phase classification')

opt = parser.parse_args()
opt.classes =  1
print(opt)
t00 = time.time()
x = 10
print(x)
image_size_detection = opt.image_size_detection #320  # square shape, 320 X 320
#image_size_clsf = 80  # size of image for mitotic phase classification
img_mean_threshold = opt.img_mean_thres # 235 # max value of each channel mean

# Define directories
detects_dir = 'data/detections'

st.markdown("## Welcome to Mitosis Detector 🎈")
# Upload the WSI file
with st.sidebar:
    #st.markdown("# Main page 🎈")

    st.subheader("Upload your Image")
    image_file = st.file_uploader("Upload a Image", type=["svs", "jpeg"])
    #file_name = image_file.name
    #st.write(file_name)

#save_img_path, filename = upload_image(image_file, detects_dir)
st.success("WSI image has been uploaded")

# Create the directories where images for detection and classification will be saved

filename = '4eee7b944ad5e46c60ce.svs'
print(os.getcwd())

save_img_path = os.path.join('data', filename)
print(f"save image path : {save_img_path}")
if not os.path.exists(save_img_path):
    os.mkdir(save_img_path)

data_detections_dir = f"{detects_dir}/{filename.split('.')[0]}/filepatch_detections"
opt.data_detections_dir = data_detections_dir
data_clsf_dir = f"{detects_dir}/{filename.split('.')[0]}/mitosis_classifications"
opt.data_clsf_dir = data_clsf_dir

if not os.path.exists(data_detections_dir):
    os.mkdir(data_detections_dir)

if not os.path.exists(data_clsf_dir):
    os.mkdir(data_clsf_dir)


# Display the low resolution image
img = np.asarray(display_image(save_img_path))
img = img[:, :, :3]
#st.write('Image shape: ', img.shape)

fig = px.imshow(img, width=1000, height=1000)
st.plotly_chart(fig)

# open ths slide and get relevant properties
slide_path = os.path.join(save_img_path)
slide = openslide.open_slide(str(slide_path))

(wsi_x, wsi_y) = slide.dimensions  # get X and Y pixel counts of WSI
dims = slide.level_dimensions
mag_ratio = dims[0][0]/dims[-1][0]
#st.write('Max mag ratio = ', mag_ratio)
patch_count_in_wsi = calculate_patch_count_in_WSI(
    wsi_x, wsi_y, image_size_detection)
st.subheader(f"Slide full size dimension: {wsi_x, wsi_y}")
st.subheader(f"number of patches : {patch_count_in_wsi} of size: {image_size_detection}")

print(
    f"Slide full size dimension: {wsi_x, wsi_y}, number of patches : {patch_count_in_wsi} of size: {image_size_detection}")


#print(f"gpu memory used : {torch.cuda.memory_summary()}")
# iterate through the patch centers, use multiprocessing
# First check if the patch images have already been created based on the expected count of patch images for this WSI

print(f"*** Starting creation of image tiles of size {image_size_detection} X {image_size_detection}")

# Check the number of jpeg files that already exist in the patched images directory for this WSI
patch_files = os.listdir(data_detections_dir)
patch_file_count = len([1 for x in patch_files if x.endswith('jpeg')])

if patch_file_count != 15610:
    p = Pool(10)
    n1_n2 = [(x, y) for x in range(int(wsi_x/image_size_detection))
            for y in range(int(wsi_y/image_size_detection))]

    with p:
            patch_ops_save_dir_fn = partial(
                # max value of each channel mean
                save_single_image_patch, data_detections_dir, image_size_detection, img_mean_threshold)
            patch_names = p.starmap(patch_ops_save_dir_fn, n1_n2)
else:
    print(f"Image patches already exists")
print(f"*** Finished creation of image tiles")

# Perform mitotic detections using object detector (e.g. YOLOV7)
# Save results in a dict
# Key = patch image name
# Value = [x1, y1, x2, y2, prob, class]

t0 = time.time()
mitotic_detections = mitotic_detections(opt)

#print(f"gpu memory used : {torch.cuda.memory_summary()}")
print(f"*** detection time : {time.time()-t0}")

# Perform mitotic phase and background classification using resnet50
t0 = time.time()
class_names = ['1_Prophase', '2_Metaphase', '3_Anaphase / Telophase', '5_Background']
all_mitotic_detections = mitotic_phase_classify(opt, mitotic_detections)

print(f"*** Total time taken for mitotic phase classification: {time.time() - t0}")

# Save all detections to file
t0 = time.time()
store_detections_to_file(all_mitotic_detections)
print(f"*** Total time to save all detections: {time.time() - t0}")

# Display all detections on top of WSI at low mag
with st.container():
    st.header("Mitotic Detections in WSI at lowest magnification")
    fig = px.imshow(img, width=1000, height=1000)
    shapes = get_detection_locations(
        all_mitotic_detections, opt.image_size_detection, mag_ratio=16)

    fig.update_layout(shapes=shapes)
    st.plotly_chart(fig)
st.header("Summary of mitotic Detections")

det_summary = summarize_detections(all_mitotic_detections, class_names)

st.write(det_summary)

# Plots distribution on bounding box widths and heights
det_widths, det_heights = detection_bbox_width_height(all_mitotic_detections)
fig = plot_width_height(det_widths, det_heights)
st.write(fig)

detections_10hpf = generate_mitotic_count_10hpf(
    all_mitotic_detections, dims[0], dims[-1], patch_size=opt.image_size_detection, microns_per_pixel=0.25)
#data_clsf_dir

max_count = np.max(detections_10hpf)
if (max_count > 7):
    target_text = f"High mitotic count in 10 HPF: {max_count}, greater than threshold of normal value 7"

else:
    target_text = f"Mitotic Count is normal. Max count in 10 HPF is {max_count}"

print("done creating heatmaps", detections_10hpf.shape)
fig = plot_count_heatmap(img, detections_10hpf)

st.write(fig)

st.markdown(
    f'<h1 style="color:#33ff33;font-size:28px;">{target_text}</h1>', unsafe_allow_html=True)

st.header("Mitotic Phase Classification in zoomed in view")

# Zoom in some region to visualize high resolution image of detection region
# Add bounding box and labels 
with st.sidebar:
    #if st.button('Choose the region '):
    w = 40
    h = 40
    x = st.number_input(f"Insert the x-coordinate of the top left corner. \n Must be between 0 and {slide.level_dimensions[-1][0]}", min_value=0, max_value=slide.level_dimensions[-1][0], value=1408)
    y = st.number_input(f"Insert the y-coordinate of the top left corner. \n Must be between 0 and {slide.level_dimensions[-1][1]}",
                        min_value=0, max_value=slide.level_dimensions[-1][1], value=2270)
    w = st.number_input("Specify the width of the region . Must be between 20 and 60",
                        min_value=20, max_value=60, value=40)
    st.write("Top Left X coordinate is: ", x)
    st.write("Top Left Y coordinate is: ", y)
    h = w
    st.write("Zooming in region")
    st.write("Top left = ", x,y, "Bottom right : ", x+w, y+h)

w = int(w*mag_ratio)
h = w
print(f'w = {w}')
img_label, detections_in_box = plot_all_detections_in_region(
    x, y, w, h, all_mitotic_detections, save_img_path, opt.image_size_detection, class_names)

fig = px.imshow(img_label, width=1000, height=1000)
st.plotly_chart(fig)

#st.write(detections_in_box)

st.header("Visual attributions of Mitotic Phase Classification")
t0 = time.time()

for img_file in detections_in_box:
    img_file = img_file + '.jpeg'
    st.write(img_file)

    img_original, saliency, attr_ig, attr_gradshap, attributions_occ = get_explainers(
        img_file, data_clsf_dir, class_names)
    
    original_image, norm_attr, cmap, vmin, vmax = visualize_image_attr(
        saliency, img_original,  sign="absolute_value",
        show_colorbar=True, title="Overlayed Gradient Magnitudes",
        use_pyplot=False)
    #fig, h = plot_image_attribution_heatmap(original_image, norm_attr)
    #st.write(fig)
    
    original_image, attr_ig, cmap, vmin, vmax = visualize_image_attr(
        attr_ig, img_original,  sign="all",
        show_colorbar=True, title="Overlayed Gradient Magnitudes",
        use_pyplot=False)

    original_image, attr_gradshap, cmap, vmin, vmax = visualize_image_attr(
        attr_gradshap, img_original,  sign="all",
        show_colorbar=True, title="Overlayed Gradient Shap Values",
        use_pyplot=False)
    #fig, h = plot_image_attribution_heatmap(original_image, norm_attr, cmap='PiYG')
    #st.write(fig)

    original_image, attr_occ, cmap, vmin, vmax = visualize_image_attr(
        attributions_occ, img_original,  sign="all",
        show_colorbar=True, title="Overlayed Gradient Magnitudes",
        use_pyplot=False)
    #fig, h = plot_image_attribution_heatmap(original_image, norm_attr, cmap='PiYG')

    attributions = [norm_attr, attr_ig, attr_gradshap, attr_occ]
    cmaps = ["Blues", "PiYG", "PiYG", "PiYG"]
    titles = ["Saliency Map", 'Integrated gradients', 'Gradient Shap', "Image Occlusion"]
    
    fig = plot_attributions(original_image, attributions, cmaps, titles)
    st.write(fig)

    #st.write(img_file, img.shape, saliency.shape, attr_ig.shape, attributions_occ.shape)

print(f"Total time display heatmaps: {time.time() - t0}")

