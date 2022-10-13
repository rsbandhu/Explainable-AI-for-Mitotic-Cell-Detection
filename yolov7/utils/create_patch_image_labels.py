import numpy
import multiprocessing
import time
import os
import json

from PIL import Image
from multiprocessing import Pool
import time
from functools import partial

import openslide
from SlideRunner.dataAccess.database import Database
from SlideRunner.dataAccess.annotations import *

from pathlib import Path
path = Path('./')

import sys
sys.path.append('../')

# you need to create a "utils" folder with the following two files
from image_utils import *
from coco_format_labels_creation import *



def save_patch_image_return_label(save_dir, patch_annotation_dict, patch_size, threshold_mean, n1, n2):

    """
    For a patch in a WSI indexed by (n1, n2)
    save the patch image if the following 2 conditions are satisfied:
        1. there is at least one annotation inside the patch
        2. The min value of the average pixel value is less than the "threshold_mean"

    Inputs:
    save_dir: directory to save the patch image
    patch_size: size of the patch
    threshold_mean: 
    n1: X-index of the patch top left corner, units of patch_size
    n2: Y-index of the patch top left corner, units of patch_size

    Return:
    patch_name: name of the patch, 
    labels: list of labels for each patch
    """
    #patch_size = 640
    #threshold_mean = 235
    
    #x = save_dir
    
    (x_topleft,y_topleft) = (n1*patch_size, n2*patch_size)
    #print(x_topleft, y_topleft)
    
    #print(f"{(n1,n2)}, :: {img_mean[:3]},  std::  {img_std_dev[:3]}")
    # if the statistics passes threshold, then it is considered a valid patch
    
    patch_name = ''
    labels = []
    labels = patch_annotation_dict[(n1,n2)] # find all annotations in this patch
    #print(labels)
    #print(labels)
    if (len(labels) > 0): # if there is an annotation within this patch
        
        # get the patch as numpy array
        img = np.asarray(slide.read_region((x_topleft, y_topleft), level=0, size=(patch_size, patch_size)))
        img = img[:, :, :3]
        
        # calculate the statistics for each channel
        (img_mean, img_std_dev) = image_stats(img)
        #print(n1, n2, img_mean, img_std_dev)
        # 
        if np.min(img_mean) < threshold_mean:
            #valid_patch_count += 1
            #number of annotations within this patch
            #label_count += len(patch_annotation_dict[(n1,n2)])
            
            # create image patch names and save image patch
            patch_name = f"{filename.split('.')[0]}_{str(n1)}_{str(n2)}.jpeg"
            filename_save = f"{save_dir}/{patch_name}"
            #print(len(labels), patch_name)
            im = Image.fromarray(img)
            im.save(filename_save)
            
            # reduce label ID by 1. class label index start from 0
            for label in labels:
                label[0] -= 1
            #print(labels)
        else:
            #print(len(labels), patch_name)
            print("annotation ignored because of patch color", n1, n2, img_mean, img_std_dev, len(labels))
            labels = []
    return (patch_name, labels)

def save_patch_labels(patch_names, labels, save_dir):

    count = 0
    for i, label in enumerate(labels):
        if len(label) > 0:
            filename = patch_names[i]
            count += 1
            #print("filename ", filename)
            label_file = f"{filename.split('.')[0]}.txt"
            #print(label_file, label)
            
            with open(os.path.join(save_dir, label_file), 'w') as f:
                for x in label:
                    write_str = str(x[0])
                    for y in x[1]:
                        write_str += f" {str(y)}"
                    write_str += '\n'
                    f.write(write_str)
                    
                f.close()
    print(count) 

def save_img_list(save_filename, img_list):

    with open(save_filename, 'w') as f:
        for x in img_list:
            f.write(x+"\n")

    f.close()

def train_test_split_wsi(DB, getslides):

    """
    Split the WSI into two groups: train set WSI and test set WSI

    Returns:
    train_set_wsi: list of WSI in the train set
    train_set_wsi: list of WSI in the val set
    """
    ### Define the WSI for train and val set
    test_set_wsi = []
    test_set_wsi.append('4eee7b944ad5e46c60ce.svs')
    test_set_wsi.append('e09512d530d933e436d5.svs')

    test_set_wsi.append('2d56d1902ca533a5b509.svs')
    test_set_wsi.append('13528f1921d4f1f15511.svs')

    test_set_wsi.append('69a02453620ade0edefd.svs')
    test_set_wsi.append('b1bdee8e5e3372174619.svs')

    test_set_wsi.append('022857018aa597374b6c.svs')

    print(f"\nNumber of WSI in test set = {len(test_set_wsi)}")

    train_set_wsi = []

    for item in DB.execute(getslides).fetchall():
        slide = item[1]
        if slide not in test_set_wsi:
            train_set_wsi.append(slide)
            
    print(f"Number of WSI in test set = {len(train_set_wsi)}")

    # Check if there is any overlap between train and test set WSI

    overlap_count = 0
    for x in test_set_wsi:
        if x in train_set_wsi:
            print(f" Image: {x} is present in both train and val sets")
            overlap_count += 1
    print(f"Number of WSI present in both train and val set: {overlap_count}")
    print(f'Finished creating train and test set split of WSI images')
    return train_set_wsi, test_set_wsi

def create_data_dirs(root_dir, patch_size):
    """
    Create all the directories needed for storing the patch images and labels 
    Folder structure is compatible with requirement for YOLOV7 training
    """
    # set up the directory name where everything will be stored
    save_root_dir = os.path.join(root_dir, f"data_{str(patch_size)}X{str(patch_size)}_xywh_patch_cmc_all/")
    print(f'\nData will be saved in directory:\n{save_root_dir}')
    save_image_root_dir = f'{save_root_dir}images'
    save_label_root_dir = f'{save_root_dir}labels'

    # create all the directories to save the data
    def make_dir(new_dir):
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)

    make_dir(save_root_dir)
    make_dir(save_image_root_dir)
    make_dir(save_label_root_dir)

    make_dir(os.path.join(save_image_root_dir, 'train'))
    make_dir(os.path.join(save_image_root_dir, 'val'))

    make_dir(os.path.join(save_label_root_dir, 'train'))
    make_dir(os.path.join(save_label_root_dir, 'val'))
    print(f'Finished creating all directories')

    return save_root_dir, save_image_root_dir, save_label_root_dir

np.set_printoptions(precision=1)


def main(patch_size, threshold_mean, bbox_length, slide_list, data_dirs):

    train_wsi_count =0
    val_wsi_count = 0
    image_count = 0
    
    #list containing image names in train and val set
    train_img_list = []
    val_img_list = []
    save_root_dir, save_image_root_dir, save_label_root_dir = data_dirs

    annot_count = 0
    # iterate through the WSI images
    
    for currslide, filename  in slide_list:
        
        image_count += 1
        if image_count > 3:
            break

        #identify if it is in the train set or val set
        if filename in train_set_wsi:
            print(f"\n{currslide, filename} is in train set")
            train_or_val = 'train'
        elif filename in test_set_wsi:
            print(f"\n{currslide, filename} is in test set")
            train_or_val = 'val'

        save_image_dir = f"{save_image_root_dir}/{train_or_val}"
        save_label_dir = f"{save_label_root_dir}/{train_or_val}"
        
        #load the WSI into memory
        database.loadIntoMemory(currslide)
        annot_count += len(database.annotations.keys())
        print(annot_count)
        
        slide_path = os.path.join(slide_dir, filename)  #basepath + os.sep + filename
        slide = openslide.open_slide(str(slide_path))
        
        (wsi_x, wsi_y) = slide.dimensions # get X and Y pixel counts of WSI
        patch_count_in_wsi = calculate_patch_count_in_WSI(wsi_x, wsi_y, patch_size)
        print(f"Slide dimension: {wsi_x, wsi_y}, number of patches : {patch_count_in_wsi} of size: {patch_size}")
        
        single_img_annotations = database.annotations
        print(f"Number of annotations: {len(single_img_annotations.keys())}")
        # Generate the dictionary of annotations for this WSI
        patch_annotation_dict = get_patch_annotations(patch_size, (wsi_x, wsi_y), bbox_length, single_img_annotations)
        
        print(f"Number of patches with annotations: {len(patch_annotation_dict.keys())}\n")
        
        #print(patch_annotation_dict.keys())
        # iterate through the patch centers

        p = Pool(10)
        n1_n2 = [(x,y) for x in range(int(wsi_x/patch_size)) for y in range(int(wsi_y/patch_size))]
        
        #n1_n2 = [(x,y) for x in range(20, 60) for y in range(20, 40)]
        with p:
            patch_ops_save_dir_fn = partial(save_patch_image_return_label, save_image_dir, patch_annotation_dict, patch_size, threshold_mean)
            (patch_names, labels) = zip(*p.starmap(patch_ops_save_dir_fn, n1_n2))
            #(patch_names, labels) = zip(*p.starmap(patch_ops, n1_n2))
        
        print("Number of patches with labels: ", sum([1 for x in labels if len(x) > 0]))
        print("Number of images with labels: ", sum([1 for x in patch_names if len(x) > 0]))
        
        # save the patch labels to files   
        save_patch_labels(patch_names, labels, save_label_dir)
        
        # Append the image names to the train or Val list of images
        if train_or_val == 'train':
            train_img_list += [f"./images/{train_or_val}/{x}" for x in patch_names if len(x) > 0]
        elif train_or_val == 'val':
            val_img_list += [f"./images/{train_or_val}/{x}" for x in patch_names if len(x) > 0]

    print(f"Number of images in train set: {len(train_img_list)}")
    print(f"Number of images in val set: {len(val_img_list)}")

    # Save the list of images in val set
    save_filename = save_root_dir+'/val.txt'
    save_img_list(save_filename, val_img_list)

    # Save the list of images in train set
    save_filename = save_root_dir+'/train.txt'
    save_img_list(save_filename, train_img_list)

if __name__ == '__main__':

    patch_size = 320   # set image patch size
    bbox_length = 50  # set bounding box length
    threshold_mean = 235  # set pixel threshold mean
    edge_exclusion = 10

    print(f"patch size : {patch_size}")
    print(
        f"patch with pixel mean above threshold value *** {threshold_mean} *** will not be saved")
    print(
        f"annotation that is within distance *** {edge_exclusion} *** from nearest edge within patch will not be saved")
    
    # Specify root dir and database dir
    root_dir = '/media/bony/Ganga_HDD_3TB/Ganges_Backup/Courses/FourthBrain_Cohort_8_June22_2022/capstone'
    database_dir = os.path.join(root_dir, 'MITOS_WSI_CMC/databases')
    #print(root_dir, database_dir)

    #Specify the database SQL file
    db_file = 'MITOS_WSI_CMC_COADEL_TR.sqlite'
    db_path = os.path.join(database_dir, db_file)

    #Specify directory of whole slide images
    slide_dir = os.path.join(root_dir, 'MITOS_WSI_CMC/WSI')

    #database.open(os.path.join(database_dir, database))
    database = Database()
    DB = database.open(db_path)

    getslides = """SELECT uid, filename FROM Slides"""
    #currslide, filename = DB.execute(getslides).fetchall()[slide_num]

    # Create all the directories as necessary for storing the data
    data_dirs = create_data_dirs(root_dir, patch_size)
    save_root_dir, save_image_root_dir, save_label_root_dir = data_dirs
    
    # Create Train and Test set WSI
    train_set_wsi, test_set_wsi = train_test_split_wsi(DB, getslides)

    # get a particular slide
    slide_list = database.listOfSlides()
    
    image_patch_names = []
    image_patch_labels = []

    #main(patch_size, threshold_mean, bbox_radius, slide_list, data_dirs)

    train_wsi_count =0
    val_wsi_count = 0
    image_count = 0
    
    #list containing image names in train and val set
    train_img_list = []
    val_img_list = []   

    annot_count = 0

    # iterate through the WSI images
    for currslide, filename  in slide_list:
        
        image_count += 1
        #if image_count > 3:
        #    break

        #identify if it is in the train set or val set
        if filename in train_set_wsi:
            print(f"\n{currslide, filename} is in train set")
            train_or_val = 'train'
        elif filename in test_set_wsi:
            print(f"\n{currslide, filename} is in test set")
            train_or_val = 'val'

        # set directory based on train or test file
        save_image_dir = f"{save_image_root_dir}/{train_or_val}"
        save_label_dir = f"{save_label_root_dir}/{train_or_val}"
        
        #load the WSI into memory
        database.loadIntoMemory(currslide)
        annot_count += len(database.annotations.keys())
        print(annot_count)
        
        # open ths slide and get relevant properties
        slide_path = os.path.join(slide_dir, filename)  #basepath + os.sep + filename
        slide = openslide.open_slide(str(slide_path))
        
        (wsi_x, wsi_y) = slide.dimensions # get X and Y pixel counts of WSI
        patch_count_in_wsi = calculate_patch_count_in_WSI(wsi_x, wsi_y, patch_size)
        print(f"Slide dimension: {wsi_x, wsi_y}, number of patches : {patch_count_in_wsi} of size: {patch_size}")
        
        #get annotations from the slide from the database
        single_img_annotations = database.annotations
        print(f"Number of original annotations: {len(single_img_annotations.keys())}")

        
        # Generate the dictionary of annotations for this WSI
        # dict, key is identified by (filename_n1_n2) where top left corner of the patch is at location (n1, n2)*patch_size
        # value is list of annotations within this patch
        #   each annotation is a list of 2 elements
        #       first element is the class of the annotation
        #       second element is a list that is the normalized bounding box in XYWH format

        patch_annotation_dict = get_patch_annotations(
            patch_size, (wsi_x, wsi_y), bbox_length, edge_exclusion, single_img_annotations)
        
        print(f"Number of patches that has annotations: {len(patch_annotation_dict.keys())}\n")
        annotation_count = sum([len(x) for x in patch_annotation_dict.values()])
        print(f"annotations that are in these patches :: {annotation_count}")

        #print(patch_annotation_dict.keys())
        # iterate through the patch centers

        p = Pool(10)
        n1_n2 = [(x,y) for x in range(int(wsi_x/patch_size)) for y in range(int(wsi_y/patch_size))]
        
        #n1_n2 = [(x,y) for x in range(20, 60) for y in range(20, 40)]
        with p:
            patch_ops_save_dir_fn = partial(save_patch_image_return_label, save_image_dir, patch_annotation_dict, patch_size, threshold_mean)
            (patch_names, labels) = zip(*p.starmap(patch_ops_save_dir_fn, n1_n2))
            #(patch_names, labels) = zip(*p.starmap(patch_ops, n1_n2))
        
        print("Number of patches with labels: ", sum([1 for x in labels if len(x) > 0]))
        print("Number of images with labels: ", sum([1 for x in patch_names if len(x) > 0]))
        
        # save the patch labels to files   
        save_patch_labels(patch_names, labels, save_label_dir)
        
        # Append the image names to the train or Val list of images
        if train_or_val == 'train':
            train_img_list += [f"./images/{train_or_val}/{x}" for x in patch_names if len(x) > 0]
        elif train_or_val == 'val':
            val_img_list += [f"./images/{train_or_val}/{x}" for x in patch_names if len(x) > 0]

    print(f"Number of images in train set: {len(train_img_list)}")
    print(f"Number of images in val set: {len(val_img_list)}")

    # Save the list of images in val set
    save_filename = save_root_dir+'/val.txt'
    save_img_list(save_filename, val_img_list)

    # Save the list of images in train set
    save_filename = save_root_dir+'/train.txt'
    save_img_list(save_filename, train_img_list)

    #print(DB.execute(getslides).fetchall())
