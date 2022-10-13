from collections import defaultdict
import numpy as np
import os
import openslide
#from SlideRunner.dataAccess.annotations import *




def create_list_by_type_one_image(single_img_annotations):
    annotation_list_by_type = defaultdict(list)

    for k in single_img_annotations.keys():
        annot_class = single_img_annotations[k].agreedClass
        annotation_list_by_type[annot_class].append(k)
        
    len(annotation_list_by_type[1]), len(annotation_list_by_type[2])

    return annotation_list_by_type

def get_slides(DB):
    """
    Get a list of tuples (slide_num, WSI_name) from the database
    """

    getslides = """SELECT uid, filename FROM Slides"""
    return DB.execute(getslides).fetchall()

def calculate_bbox(x, y, patch_size, r=25, m=1):
    """
    Calculates bounding box of an annotation  (normalized in height and width of the image) of an annotation)
    Inputs:
    x: annotation center X coordinate
    y: annotation center Y coordinate
    r: Radius around the annotation center
    m: Microscope mag
    
    Output:
    a list of 4 numbers that defines the bounding box
    (topleft X, topleft y, bottom right x, bottom right y)
    """
    d = 2 * r / m
    
    x_min = max(0, (x - r) / m)
    y_min = max(0, (y - r) / m)
    
    x_max = min(x_min + d, patch_size)
    y_max = min(y_min + d, patch_size)
    
    #bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
    # normalized bounding box
    bbox = [x_min/patch_size, y_min/patch_size, x_max/patch_size, y_max/patch_size]
    
    return bbox


def calculate_bbox_norm_xywh(x, y, patch_size, edge_exclusion, bbox_size, m=1):
    """
    Calculates bounding box (normalized in height and width of the image) of an annotation
    Inputs:
    x: annotation center X coordinate (zero is top left of image, max value is image width)
    y: annotation center Y coordinate (zero is top left of image, max value is image height)
    r: Radius around the annotation center
    m: Microscope mag
    
    Output:
    a list of 4 numbers that defines the bounding box
    (center X, center y, width, height)
    """
    half_width = int(bbox_size/(2*m))
    d = int(bbox_size / m) # nominal width and height of the bbox , nominal= 50

    assert 0 <= x <= patch_size, f'X coordinate value = {x}  out of bounds, must be within {0} and {patch_size}. '
    assert 0 <= y <= patch_size, f'Y coordinate value = {y}  out of bounds, must be within {0} and {patch_size}. '

    dist_from_nearest_edge_x = min(x, patch_size-x)
    dist_from_nearest_edge_y = min(y, patch_size-y)

    min_dist_from_edge = min(dist_from_nearest_edge_x,
                             dist_from_nearest_edge_y)

    #print(f"min distance from nearest edge in both X and Y: {min_dist_from_edge}")

    # calculate un-normalized width, 
    # add a random number to the width and height
    # if the min distance from the edge is smaller than edge_exclusion
    # then there is no bbox, ignore this annotation

    if min_dist_from_edge < edge_exclusion:
        bbox = []
        #print(f"min dist from edge: {min_dist_from_edge}")

    else: # if the annotation center is not in the edge exclusion zone
        #calculate bounding box

        if min_dist_from_edge > (half_width + 10):  
            # add a random number to bounding box 
            # if the center is at least distance = (half_width + 10) away from the nearest edge
            w = d + np.random.randint(high=10, low=-10)
            h = d + np.random.randint(high=10, low=-10)

        else:
            w = min_dist_from_edge
            h = min_dist_from_edge

        # unnormalized bounding box
        #bbox = [x,y,w,h]
        # normalized bounding box
        bbox = [round(x/patch_size, 4), round(y/patch_size, 4),
                round(w/patch_size, 4), round(h/patch_size, 4)]

    return bbox

def create_patch_center_identifier(filename, nx, ny):

    return f"{filename}_{str(nx)}_{str(ny)}.jpg"

def calculate_patch_count_in_WSI(wsi_x, wsi_y, patch_size):
    """
    Calculates the number of patches in a WSI, 
    
    Inputs:
    wsi_x: number of pixels in X
    wsi_y: number of pixels in Y
    patch_size: size of each patch
    
    Returns:
    number of patches in the WSI. 
    """
    nx = int(wsi_x / patch_size)
    ny = int(wsi_y / patch_size)

    return nx*ny

def find_patch_index_within_WSI(x, y, patch_size):
    """
    Calculates the index of the patch where the annotation belongs

    Inputs:
    x: annotation center X coordinate
    y: annotation center X coordinate
    patch_size: size of each patch

    Returns: (nx, ny): index of the patch in units of patch size where the annotation is present
    """
    nx = int(x / patch_size)
    ny = int(y / patch_size)

    return (nx, ny)


def get_patch_annotations(patch_size, image_size, bbox_size, edge_exclusion, single_img_annotations):
    """
    Create a dict where key are tuples of patch center coordinates and
    values are list of bounding boxes of annotations that are within that patch

    Inputs:
    patch_size: size of the patch
    image_size: tuple (x,y) that denotes image size
    bbox_length: size of bounding box. square shaped
    single_img_annotations: annotations for a single image

    Returns:
    A dictionary where
    key = tuple (nx_topleft, ny_topleft) indicating the topleft corner index of the patch within WSI
    value = list of item corresponding to all annotations that lie in the corresponding patch. 
    Each item is an array of 2 elements.
    first element is the class of the annotation
    second element is the normalized bounding box in the format XYWH
    X: top left corner, X,  of the bounding box normalized to the width of the image size
    Y: top left corner, Y,  of the bounding box normalized to the height of the image size
    W: Width of the bounding box normalized to the width of the image size
    H: Height of the bounding box normalized to the height of the image size
    """
    cls_2_count = 0

    # Sample 20 random mitotic cell regions

    annotation_list_by_type = create_list_by_type_one_image(single_img_annotations)

    sample_of_mitotic_cells = np.random.choice(annotation_list_by_type[2], 10)
    #print(f"Sample of annotations:  {sample_of_mitotic_cells}")

    img_x, img_y = image_size
    patch_annotation_dict = defaultdict(list)

    #for id, annotation in database.annotations.items():
    # Loop through all the annotations in the image
    count_edge_exclusion = 0

    for annot_uid in single_img_annotations: #database.annotations.items():
    
        annotation = single_img_annotations[annot_uid]
        if annotation.deleted or annotation.annotationType != AnnotationType.SPOT:
            continue
        else:
            #get X and Y coordinates of the annotation
            x = annotation.x1 
            y = annotation.y1
            agreed_classs = annotation.agreedClass
            annot_uid = annotation.uid #unique ID of the annotation
            
            #print(x, y)
            # Calculate the top left corner of the patch where this annotation is present
            # in units of the patch_size
            x_topleft = int(x/patch_size)
            y_topleft = int(y/patch_size)

            # Calculate position of annotation within the patch
            # (0,0) location is at the top left corner of the patch
            # top left corner coordinates = (x_topleft * patch_size,  y_topleft * patch_size)

            x_patch = x - x_topleft * patch_size
            y_patch = y - y_topleft * patch_size

            
            #print(annotation.annotationType, annotation.agreedClass, x, y)
            #bbox = calculate_bbox(x,y, patch_size, r = bbox_length)
            
            bbox = calculate_bbox_norm_xywh(
                x_patch, y_patch, patch_size, edge_exclusion, bbox_size)
                #print(x, y, bbox)
            #patch_center_x = int(n_x*patch_size + patch_size/2)
            #patch_center_y = int(n_y*patch_size + patch_size/2)
            
            if len(bbox) > 0:
                patch_annotation_dict[(x_topleft, y_topleft)].append(
                    [agreed_classs, bbox])
            else:
                #print(f"annotation :  {agreed_classs} :: too close to one corner {x_patch, y_patch}")
                #print(bbox)
                count_edge_exclusion += 1

            #Assign to the list of annotations for this patch
            #print(bbox)
    print(f"{count_edge_exclusion} out of ::  {len(single_img_annotations.keys())} :: annotations close to edge are ecluded")
    return patch_annotation_dict
            

def create_image_patches(slide_dir, DB, patch_size):
    
    slide_list = get_slides(DB) #list of all slides in the database

    global_patch_count = 0
    coco_image_labels = []
    for n in range(len(slide_list)):
        slide_num = n
        currslide, filename = slide_list[slide_num]
        #print(f"\n\nSlide number: {currslide},  Name: {filename}\n")

        #load the WSI into memory
        #database.loadIntoMemory(currslide)
        
        #img_annotaton = database.annotations
        #img_annotation_list = create_list_by_type_one_image(img_annotaton)
        
        slide_path = os.path.join(slide_dir, filename)  #basepath + os.sep + filename
        slide = openslide.open_slide(str(slide_path))
        
        dx, dy = slide.dimensions
        print(f"\n\nSlide number: {currslide},  Name: {filename}, Slide dimension: {slide.dimensions}\n")

        for i in range(int(dx/patch_size)):
            for j in range(int(dy/patch_size)):
                global_patch_count += 1
                image_dict = {}
                #patch_name = f"{filename}_{str(i)}_{str(j)}_{str(global_patch_count)}"
                patch_name = f"{filename}_{str(i)}_{str(j)}"
                image_dict['license']=1
                image_dict['file_name'] =  f'{patch_name}.jpg',
                image_dict['coco_url'] =  'http://images.cocodataset.org/val2017/000000397133.jpg',
                image_dict['height'] = patch_size
                image_dict['width'] =  patch_size
                image_dict['date_captured'] = '2013-11-14 17:02:52',
                image_dict['flickr_url']= 'http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg'
                image_dict['id']= global_patch_count

                coco_image_labels.append(image_dict)
                if global_patch_count%1000 ==0:    
                    print(global_patch_count, "  ***  ", image_dict)

    return coco_image_labels

def image_stats(img):
    mean = np.mean(img, axis=(0,1))
    std_dev = np.std(img, axis = (0,1))

    return (mean, std_dev)


def patch_ops(save_dir, patch_annotation_dict, img, patch_size, threshold_mean, n1, n2):

    """
    For a patch in a WSI indexed by (n1, n2)
    save the patch image if the following 2 conditions are satisfied:
        1. there is at least one annotation inside the patch
        2. The min value of the average pixel value is less than the "threshold_mean"

    Inputs:
    save_dir: directory to save the patch image
    patch_size: size of the patc
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
    
    if (len(labels) > 0): # if there is an annotation within this patch
        
        # get the patch as numpy array
        #img = np.asarray(slide.read_region((x_topleft, y_topleft), level=0, size=(patch_size, patch_size)))
        #img = img[:, :, :3]
        
        # calculate the statistics for each channel
        (img_mean, img_std_dev) = image_stats(img)
    
        # 
        if np.min(img_mean) < threshold_mean:
            #valid_patch_count += 1
            #number of annotations within this patch
            #label_count += len(patch_annotation_dict[(n1,n2)])
            
            # create image patch names and save image patch
            patch_name = f"{filename.split('.')[0]}_{str(n1)}_{str(n2)}.jpeg"
            filename_save = f"{save_dir}/{patch_name}"
            
            im = Image.fromarray(img)
            im.save(filename_save)
            
            # reduce label ID by 1. class label index start from 0
            for label in labels:
                label[0] -= 1
            #print(labels)
    
    return (patch_name, labels)

def get_detection_bbox(detection):
    
    x_topleft = int(detection[0])
    y_topleft = int(detection[1])

    x_botright = int(detection[2])
    y_botright = int(detection[3])

    return (x_topleft, y_topleft, x_botright, y_botright)

def get_image_crop_coords(img_w_h, bbox_xyxy, patch_size):

    '''
    Given an image, bounding box, get a patch of width and height = patch_size
    centered as much as posible on the bounding box

    Inputs:
    img_w_h: image (width, height)
    bbox_xyxy: bounding box coordinates (x_topleft, y_topleft, x_botright, y_botright)
    patch_size: width and height of the area to be cropped

    Returns:
    cropped image
    '''

    w, h = img_w_h

    assert patch_size < w, "cropped size is bigger than original image along X"
    assert patch_size < h, "cropped size is bigger than original image along Y"

    (x_topleft, y_topleft, x_botright, y_botright) = bbox_xyxy

    x_center = int((x_topleft + x_botright)/2)
    y_center = int((y_topleft + y_botright)/2)

    # Find the X coordinates of the cropped region
    half_w = int(patch_size/2)
    if (x_center > half_w): #check distance from left side
        if (w-x_center) > half_w:  # check distance from left side
            x1 = x_center - half_w 
            x2 = x1 + patch_size
        else:
            x1 = w-patch_size
            x2 = w
    else:
        x1 = 0
        x2 = patch_size
    
    # Find the X coordinates of the cropped region
    half_h = int(patch_size/2)
    if (y_center > half_h):  # check distance from left side
        if (h-y_center) > half_h:  # check distance from left side
            y1 = y_center - half_h
            y2 = y1 + patch_size
        else:
            y1 = h-patch_size
            y2 = h
    else:
        y1 = 0
        y2 = patch_size

    assert x2-x1 == patch_size, "cropped size incorrect along X"
    assert y2-y1 == patch_size, "cropped size incorrect along Y"

    return (x1, y1, x2, y2)




