from collections import defaultdict
import numpy as np
import os

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
    img = np.asarray(slide.read_region((x_topleft, y_topleft),
                                       level=0, size=(patch_size, patch_size)))
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
    
def image_stats(img):
    mean = np.mean(img, axis=(0,1))
    std_dev = np.std(img, axis = (0,1))

    return (mean, std_dev)

