
def licenses():
    #license_dict = {}
    #license_dict["licenses"] =
    license = [
        {
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License"
        }
    ]
    return license
    

def info():
    info_dict = {}
    info_dict["info"] = {
    "description": "Canine Breast Cancer",
    "url": "https://doi.org/10.1038/s41597-020-00756-z",
    "version": "1.0",
    "year": 2020,
    "contributor": "R Klopfleisch",
    "date_created": "2020/09/01"}

    return info_dict

def categories():
    
    #catg_dict = {}
    #catg_dict["categories"] = 
    catgs = [
    {"supercategory": "cell","id": 1,"name": "mitotic_look_alike"},
    {"supercategory": "cell","id": 2,"name": "mitotic"},
    {"supercategory": "cell","id": 3,"name": "background"}
    ]

    return catgs

def image(data, height, width, img_id, img_filename):
    """
    image ids need to be unique (among other images), 
    but they do not necessarily need to match the file name 
    (unless the deep learning code you are using makes an assumption that they’ll be the same… 
    
    """
    image_dict = {}
    
    image_dict['height'] = height
    image_dict['width'] = width
    image_dict["id"] = img_id
    image_dict["file_name"] = filename
    image_dict["license"] = 1
    
    return image_dict
    

def category(label):
    cat_dict = {}
    cat_dict["supercategory"] = "cells"
    cat_dict["id"] = label[0]
    cat_dict["name"] = label[1]
    
    return cat_dict

def annotation(annot_uid, img_filename, bbox, class_label, area):
    """
    
    Is Crowd specifies whether the segmentation is for a single object or for a group/cluster of objects.
    image id corresponds to a specific image in the dataset.

    bbox: The COCO bounding box format is [top left x position, top left y position, width, height].
    The category id corresponds to a single category specified in the categories section.
    id (unique to all other annotations in the dataset).
    """
    annot_dict = {}
    
    annot_dict["segmentation"] = []
    annot_dict["iscrowd"] = 0
    annot_dict["area"] = area
    annot_dict["image_id"] = img_filename
    annot_dict["bbox"] = list(bbox)
    annot_dict["category_id"] = class_label  # self.getcatid(label)
    annot_dict["id"] = annot_uid
    
    return annot_dict