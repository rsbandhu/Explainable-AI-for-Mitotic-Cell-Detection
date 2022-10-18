import numpy as np
import pandas as pd
import os
from collections import defaultdict

from image_utils.plot_detections import convert_local_to_global_coordinates

def summarize_detections(detects, class_names):

    det_counts = defaultdict(int)
    for k, v in detects.items():
        det_class = class_names[v[6]]
        det_counts[det_class] += 1
    df = pd.DataFrame({}, index = ["COUNT"])

    for k,v  in sorted(det_counts.items(), key=lambda x:x[0]):
        new_key = k.split('_')[1]
        df[new_key] = [v]

    print(det_counts)
    

    return df


def detection_bbox_width_height(detects):

    det_widths = []
    det_heights = []

    for k, v in detects.items():
        w = int(v[2] - v[0])
        h = int(v[3] - v[1])

        det_widths.append(w)
        det_heights.append(h)

    return det_widths, det_heights

def generate_mitotic_count_10hpf(detects,dims1, dims2, patch_size = 320, microns_per_pixel=0.25):

    scaling_ratio = int(round(dims1[0]/ dims2[0])) # ratio between highest mag and the mag for which the count will be plotted

    area_10hpf = 2.37 # area of 10HPF in sq mm
    asp_ratio = 4/3 # ratio of X and Y size of 10HPF box
    area_1x1 = microns_per_pixel**2 # unit is sq um
    #pix_highmag_10hpf_y = 334*16
    #pix_highmag_10hpf_x = asp_ratio * pix_highmag_10hpf_y

    pix_lowmag_10hpf_y = np.sqrt(area_10hpf/asp_ratio)*(1000)/(scaling_ratio*microns_per_pixel)
    pix_lowmag_10hpf_x = asp_ratio*pix_lowmag_10hpf_y

    area_10hpf = (pix_lowmag_10hpf_y*pix_lowmag_10hpf_y)*(asp_ratio**2)*area_1x1/(10**6)
    #print(pix_lowmag_10hpf_x, pix_lowmag_10hpf_y, area_10hpf)

    # now round them to even integers
    pix_lowmag_10hpf_x = 2*int(round(pix_lowmag_10hpf_x/2))
    pix_lowmag_10hpf_y = 2*int(round(pix_lowmag_10hpf_y/2))

    w = int(pix_lowmag_10hpf_x/2)
    h = int(pix_lowmag_10hpf_y/2)

    heatmap_10hpf_lowmag = np.zeros(dims2)
    
    for filename, detection in detects.items():
        # get the coordinates in the full mag
        x1, y1, x2, y2, xc, yc = convert_local_to_global_coordinates(filename, detection, patch_size)

        
        #get detection center coordinates in low mag
        xc_low = int(xc/scaling_ratio)
        yc_low = int(yc/scaling_ratio)

        #print(xc_low, yc_low)
        heatmap_10hpf_lowmag[yc_low-h:yc_low+h, xc_low - w:xc_low+w] += 1

    return heatmap_10hpf_lowmag

