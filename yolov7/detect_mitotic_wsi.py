import argparse
import time
import os
from pathlib import Path
#from tkinter import Image
from PIL import Image

import cv2
import torch
import torch.backends.cudnn as cudnn

import torchvision
from torchvision import datasets, models, transforms

import numpy as np
from numpy import random

import sys
sys.path.append('./')

from .models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from utils.image_utils import image_stats, get_detection_bbox, get_image_crop_coords
from image_utils.image_utils import save_single_image_patch
#from deployment.mitotic_phase_background_classifier.classifier_model import get_image_dataloader, get_model


def detect(opt, save_img=False):
    #source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    source = opt.source
    weights = opt.weights
    view_img = opt.view_img
    imgsz = opt.image_size_detection
    trace = not opt.no_trace
    batch_size = opt.batch_size
    data_detections_dir = opt.data_detections_dir
    cropped_img_dir = opt.data_clsf_dir
    mean_thrshold = opt.img_mean_thres
    crop_size = opt.image_size_clsf
    save_txt = opt.save_txt

    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    mean_thrshold = opt.img_mean_thres
    crop_size = opt.image_size_clsf

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    print("save dir : ", save_dir)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    detections = {}

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    
    if trace:
        model = TracedModel(model, device, opt.image_size_detection)

    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    
    dataset = LoadImages(data_detections_dir, img_size=imgsz, stride=stride)
    print(f"number of images :{dataset.nf}")

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    names = ['non-M', 'M']
    print(f"names : {names}")
    #print(model.modules)
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    t1 = time.time()
    model.eval()
    # my additions
    img_count = 0
    b_size = 0
    img_id_batch = []

    for path, img, im0s, vid_cap in dataset:
        # more than 90% of the time is spent in NMS calculation , about 3.5msec / image
        # if anything can be done to speed up NMS calculation, it speeds up the entire process
        img_count += 1  # my additions

        # vid_cap is None for Images
        # img shape: (3, imgsz, imgsz)
        # im0s shape: (imgsz, imgsz, 3)

        img_id = path.split('/')[-1].split('.')[0]
        #print(f"\n****** \nimage name :  {img_id}")

        # -------------------------------
        # check mean value of each channel and discard white regions
        #print(img.shape)
        img_mean, img_std = image_stats(img)

        # if the image is too white and we are not at the end of the dataset
        if (np.min(img_mean) > mean_thrshold) and (img_count < dataset.nf):
            detections[img_id] = []
            #print(f"Ignoring patch : {img_id}")
            
        # -------------------------------

        else:
            
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            # my additions
            # ------------------------
            b_size += 1  # my addition
            img_id_batch.append(img_id)

            if b_size % batch_size == 1: # start of a new batch of images
                inputs = img
            else: # accumuluate the new img  to the existing batch of images
                inputs = torch.cat((inputs, img), dim=0)

                if (b_size % batch_size == 0) or (img_count == dataset.nf): # we have a batch of images or reached the end of dataset
                    # perform prediction
                    with torch.no_grad():
                        b_size = 0
                        pred_b = model(inputs, augment=opt.augment)[0]
                    
                        # Apply NMS
                        pred_b = non_max_suppression(
                            pred_b, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    
                        for i, dets in enumerate(pred_b): #iterate over the images in the batch
                            img_name = img_id_batch[i]
                            #print(dets.cpu().numpy())
                            detections[img_name] = []
                            if len(dets): # there are some detection in this image
                                # I am commenting out the next line, since in my case there is no scaling involved
                                #dets[:, :4] = scale_coords(img.shape[2:], dets[:, :4], im0.shape).int()
                            
                                detections[img_name] = dets.cpu().detach().numpy()

                        #reset the batch of image ids to empty list
                        img_id_batch = []
                        # del pred_b, inputs
            #del img
            #print(inputs.shape, b_size)
        if img_count %500 == 0:
            print(img_count, time.time()-t0)
            #print(torch.cuda.memory_summary())
            t0 = time.time()
    print(f"Number of images with detections: {sum([1 for x in detections if len(detections[x]) > 0])}")
    print(f"Total detection time: {time.time() - t1}")
    
    ### Save the images for which there are detections from YOLOV7 to do mitotic phase classification

    # get the dict with key = detection unique id and value = [x1, y1, x2, y2, prob,class]
    all_mitotic_detections = save_cropped_images(
        detections, cropped_img_dir, data_detections_dir, crop_size)

    return all_mitotic_detections


def classify_mitotic_phases(data_dir, model_clsf, all_mitotic_detections):

    ### perform mitotic phase and background classification of the detections that are mitotic

    #data_dir = 'runs/detect/exp8/data_mitotic_detections_80X80'
    # Create dataloader
    dataloader = get_image_dataloader(data_dir)

    # get classifier model and set it to eval mode
    mitotic_clsf = get_model()
    mitotic_clsf.eval()

    # iterate through the cropped images to perform classification
    to = time.time()
    with torch.no_grad():
        for i, inputs in enumerate(dataloader):
            imgs = inputs[0] # batch of image file names
            inputs = inputs[1] #batch of image data

            # get classifier predictions of mitotic phases
            outputs = mitotic_clsf(inputs)
            _, preds = torch.max(outputs, 1)
            
            preds = preds.cpu().numpy()

            for i, img in enumerate(imgs):
                det_uid = img.split('.')[0]
                all_mitotic_detections[det_uid].append(preds[i]) #append the mitotic phase for each detection in dict
            
    print(f"time to classify mitotic detection areas : {time.time() - t0}")

    for k, v in all_mitotic_detections.items():
        print(k, v)

    return all_mitotic_detections

    # --------------------------------------------------------------------------------- 

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")
    #for k,v in detections.items():
    #    print(k, " *** ", v)
    print(f'Done. ({time.time() - t0:.3f}s)')



def save_cropped_images(detections, mitotic_cropped_img_dir, source, patch_size):

    t0 = time.time()
    all_mitotic_detections = {}

    for img_id, detects in detections.items():
        if len(detects):

            # save a patch from the image around the detection for mitotic classification
            filename = f"{img_id}.jpeg"
            original_img = Image.open(os.path.join(source, filename))
            img_w_h = original_img.size

            for i, item in enumerate(reversed(detects)):
                #print(item)
                # each detection is a list of 6 items
                # [x_topleft, y_topleft, x-botright, y_botright, prob, class_id]
                # class_id is 0 for mitotic look alike, and 1 for mitotic

                bbox_xyxy = get_detection_bbox(item)

                # get the coordinates of the cropped region
                (x1, y1, x2, y2) = get_image_crop_coords(
                    img_w_h, bbox_xyxy, patch_size)

                # crop the image
                img_cropped = original_img.crop((x1, y1, x2, y2))
                det_uid = f"{img_id}_{str(i)}"  #unique id of the detection, also identifies the cropped file name
                img_cropped_name = f"{det_uid}.jpeg"
                save_path = str(Path(mitotic_cropped_img_dir) /
                                img_cropped_name)  # img.jpeg

                img_cropped.save(save_path)

                all_mitotic_detections[det_uid] = list(item)

    print(f"time to save cropped detection areas : {time.time() - t0}")

    return all_mitotic_detections

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
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

    parser.add_argument('--mean-thres', type=float,
                        default=225, help='image channel mean value threshold')
    parser.add_argument('--patch-size', type=float,
                        default=80, help='image height & width for mitosis phase classification')

    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
