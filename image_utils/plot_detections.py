import openslide
from PIL import Image
import numpy as np
import cv2


def convert_local_to_global_coordinates(filename, detection, patch_size):
    """
    convert detection coordinates that are defined wrt to a patch
    to corodinates of the image from which patches were created

    """
    #print(detection)
    # get the index of the patch in the WSI image at level 0
    f_n1_n2 = filename.split('_')
    n1 = int(f_n1_n2[1])
    n2 = int(f_n1_n2[2])

    # get the top left corner (X, Y) of the patch at the highest mag
    # where there is a detection
    X_tl = n1 * patch_size
    Y_tl = n2 * patch_size

    # coordinate of the bbox top left within the patch
    x1 = int(detection[0])
    y1 = int(detection[1])

    # coordinate of the bbox bot right within the patch
    x2 = int(detection[2])
    y2 = int(detection[3])

    # coordinate of the bbox center within the patch
    x_c = int((x1+x2)/2)
    y_c = int((y1+y2)/2)

    # global coordinate of the top left, bottom right and bbox center at the highest mag
    x1 += X_tl
    y1 += Y_tl

    x2 += X_tl
    y2 += Y_tl

    x_c += X_tl
    y_c += Y_tl

    return (x1, y1, x2, y2, x_c, y_c)


def get_detection_locations(all_detections, patch_size, mag_ratio=16):

    shapes = []
    for filename, detection in all_detections.items():
        (x1, y1, x2, y2, x_c, y_c) = convert_local_to_global_coordinates(
            filename, detection, patch_size)
        
        # get the coordinate of the bbox center in the low mag image
        x_c = int(x_c/mag_ratio)
        y_c = int(y_c/mag_ratio)
        
        # define the top left and bot right of the circle to be drawn
        x1 = x_c -10
        y1 = y_c -10

        x2 = x_c +10 
        y2 = y_c +10

        if detection[6] == 3:
            color = 'blue'
        else:
            color = 'gold'
        shapes.append(dict(type="circle",
                           xref="x", yref="y",
                           x0=x1, y0=y1, x1=x2, y1=y2,
                           line_color='black',
                           fillcolor=color,
                           ))

    return shapes
           

def plot_all_detections_in_low_mag(all_detections, wsi_img_path, patch_size):
    """
    x, y : coordinate of the top left corner of the region of interest in low mag
    """
    #slide_path = os.path.join(wsi_img_path)  #basepath + os.sep + filename
    slide = openslide.open_slide(str(wsi_img_path))
    X, Y = slide.dimensions
    #print(f"size of slide: {slide.dimensions}")

    ratio = 16 # scaling between highest mag and the lowest mag

    tl = 4
    tf = 1
    color = (255, 0, 0)
    radius = 8

    dims = slide.level_dimensions
    dim_2 = dims[2]

    img = np.asarray(slide.read_region((0, 0), level=2, size=dim_2))

    for k, v in all_detections.items():
        # get the index of the patch in the WSI image at level 0
        f_n1_n2 = k.split('_')
        n1 = int(f_n1_n2[1])
        n2 = int(f_n1_n2[2])

        # get the top left corner (X, Y) of the patch
        # where there is a detection
        X_tl = n1 * patch_size
        Y_tl = n2 * patch_size

        # coordinate of the bbox top left within the patch
        x1 = int(v[0])
        y1 = int(v[1])

        # coordinate of the bbox bot right within the patch
        x2 = int(v[2])
        y2 = int(v[3])

        # coordinate of the bbox center within the patch
        x_c = int((x1+x2)/2)
        y_c = int((y1+y2)/2)

        # global coordinate of the bbox center at the highest mag
        x_c += X_tl
        y_c += Y_tl

        # global coordinate of the bbox center at the requested mag
        x_c = int(x_c/ratio)
        y_c = int(y_c/ratio)

        center = (x_c, y_c)
        

        cv2.circle(img, center, radius, color, thickness=tl)
        

    return img

def plot_all_detections_in_region(x, y, w, h , all_detections, wsi_img_path, patch_size, class_names):
    """
    x, y : coordinate of the top left corner of the region of interest in low mag
    """
    #slide_path = os.path.join(wsi_img_path)  #basepath + os.sep + filename
    imgs = {} # this is where we will store the zoomed in region at highest mag for each detection
    slide = openslide.open_slide(str(wsi_img_path))
    X,Y = slide.dimensions
    print(f"size of slide: {slide.dimensions}")

    ratio = 16

    # convert coordinate from lower level to level 0 (higher mag)
    X_tl = x*ratio
    Y_tl = y*ratio

    img = np.asarray(slide.read_region((X_tl, Y_tl ), level=0, size=(640, 640)))

    detections_inside_box = find_detections_within_box(X_tl, Y_tl, w, h, all_detections, patch_size)

    tl = 2
    tf =1 
    color = [0,0,255]

    for det_c, bbox_label in detections_inside_box.items():
        c1 = bbox_label[0]
        c2 = bbox_label[1]
        label_idx = int(bbox_label[2])
        label = class_names[label_idx].split('_')[-1]
        detect_prob = str(round(bbox_label[3],2))

        top_left_level0 = bbox_label[4]

        #print(c1)
        #print(c2)

        cv2.rectangle(img, c1, c2, color, thickness=tl)
        cv2.putText(img, label, (c1[0], c1[1]-5), 0, 0.5, [255, 0, 0], thickness=tf) #lineType=cv2.LINE_AA
        cv2.putText(img, f'P= {detect_prob}', (c2[0], c2[1]-5), 0, 0.5,
                    [255, 255, 255], thickness=tf)  # lineType=cv2.LINE_AA
        
        det_image = np.asarray(slide.read_region(
            top_left_level0, level=0, size=(80, 80)))
        
        imgs[det_c] = det_image

    return img, detections_inside_box, imgs


def plot_detections_in_zoomed_region(x, y, w, h, all_detections, wsi_img_path, patch_size, class_names):
    """
    x, y : coordinate of the top left corner of the region of interest in low mag
    """
    #slide_path = os.path.join(wsi_img_path)  #basepath + os.sep + filename
    slide = openslide.open_slide(str(wsi_img_path))
    X, Y = slide.dimensions
    print(f"size of slide: {slide.dimensions}")

    ratio = 16

    # convert coordinate from lower level to level 0 (higher mag)
    X_tl = x*ratio
    Y_tl = y*ratio

    img = np.asarray(slide.read_region((X_tl, Y_tl), level=0, size=(640, 640)))

    detections_inside_box = find_detections_within_box(
        X_tl, Y_tl, w, h, all_detections, patch_size)

    tl = 2
    tf = 1
    color = [0, 0, 255]
    shapes = []
    locs_labels = []

    for det_c, bbox_label in detections_inside_box.items():
        c1 = bbox_label[0]
        c2 = bbox_label[1]
        label_idx = int(bbox_label[2])
        label = class_names[label_idx].split('_')[-1]
        detect_prob = str(round(bbox_label[3], 2))

        #print(c1)
        #print(c2)
        shapes.append(dict(type="rect",
                           xref="x", yref="y",
                           x0=c1[0], y0=c1[1], x1=c2[0], y1=c2[1],
                           line_color='black'))
        
        locs_labels.append([c1, c2, label, detect_prob])
        #cv2.rectangle(img, c1, c2, color, thickness=tl)
        # lineType=cv2.LINE_AA
        #cv2.putText(img, label, (c1[0], c1[1]-5),
        #            0, 0.5, [255, 0, 0], thickness=tf)
        #cv2.putText(img, f'P= {detect_prob}', (c2[0], c2[1]-5), 0, 0.5,
        #            [255, 255, 255], thickness=tf)  # lineType=cv2.LINE_AA

    return img, shapes, locs_labels


def find_detections_within_box(x_tl, y_tl,w,h, all_detections, patch_size):

    detections = {}
    x_br = x_tl+w
    y_br = y_tl+h

    for k, v in all_detections.items():
        # get the index of the patch in the WSI image at level 0
        f_n1_n2 = k.split('_')
        n1 = int(f_n1_n2[1])
        n2 = int(f_n1_n2[2])
        
        # get the top left corner (X, Y) of the patch 
        # where there is a detection
        X_tl = n1 * patch_size
        Y_tl = n2 * patch_size

        #print(f"X_tl: {X_tl}, Y_tl: {Y_tl}, n1: {n1}, n2: {n2}")

        # coordinate of the bbox top left within the patch
        x1 = int(v[0])
        y1 = int(v[1])
        
        # coordinate of the bbox bot right within the patch
        x2 = int(v[2])
        y2 = int(v[3])
        #print(v, x1, y1, x2, y2)
        #print(n1, n2, patch_size, X_tl, Y_tl, x_tl, y_tl)

        # coordinate of the bbox center within the patch
        x_c = int((x1+x2)/2)
        y_c = int((y1+y2)/2)
        
        # Find the corodinate of the detection center in the WSI at level 0
        # wrt to the top left corner (x_tl, y_tl)
        x_c = x_c + X_tl - x_tl
        y_c = y_c + Y_tl - y_tl

        # Calculate the corordinates of top left and bottom right of the bbox
        # wrt to the top left corner (x_tl, y_tl)
        x1 = x1 + X_tl - x_tl
        y1 = y1 + Y_tl - y_tl

        x2 = x2 + X_tl - x_tl
        y2 = y2 + Y_tl - y_tl

        #print(v)
        #print(f"{(x1,y1)}  ::  {(x2,y2)}")

        # Compare if the center of detection lies within our region of interest
        if (x_c  >= 0) and (x_c  <= w ) and (y_c >= 0) and (y_c <= h): # we found a detection inside our specified box
            label = str(v[6])
            detections[k] = [(x1, y1), (x2, y2),label, round(v[4],3), (x1+x_tl, y1+y_tl)]

    return detections






