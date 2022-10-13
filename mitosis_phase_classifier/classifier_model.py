import numpy
import os

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset

from torchvision import datasets, models, transforms


class Mydataset(Dataset):
    '''
    Custom dataset for inference on cropped images to predict mitotic phases or brackground
    All image files are in one directory
    You need to specify the datadir where cropped images are stored
    '''
    def __init__(self, data_dir=None, transforms=None):
        
        #print(os.listdir(data_dir))
        if data_dir:
            self.data_dir = data_dir
        self.transforms = transforms
        
        self.img_types = ['jpg', 'jpeg', 'png']
        self.img_files = self.image_files_in_dir()
        self.img_count = len(self.img_files)
        print(self.img_count)
    
    def __getitem__(self, index):
        img_file = self.img_files[index]
        img = Image.open(os.path.join(self.data_dir, img_file))
        
        return img_file, self.transforms(img.convert("RGB"))
    
        #return img_file
    
    def __len__(self):
        return self.img_count
    
    def get_image_type(self, filename):
        img_type = filename.split('.')[-1]
        return img_type
    
    def image_files_in_dir(self):
        img_files = [x for x in os.listdir(self.data_dir) if self.get_image_type(x) in self.img_types]
        
        return img_files
        

# specify the transforms on the image
transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_image_dataloader(data_dir):
    #data_dir = 'yolov7/runs/detect/exp8/data_mitotic_detections_80X80'
    cropped_image_dataset = Mydataset(data_dir, transforms=transforms)

    dataloader = torch.utils.data.DataLoader(cropped_image_dataset, batch_size=4,
                                             shuffle=True, num_workers=4)

    return dataloader

def get_model(model_name):

    classifier_model = models.resnet50(pretrained=True)
    num_features = classifier_model.fc.in_features

    class_names = ['1_prophase', '2_metaphase', '3_anaphase', '5_background']
    # Change the final classification layer, to the number of classes 
    classifier_model.fc = nn.Linear(num_features, len(class_names))
    #model_path = os.path.join('pre_trained_models', model_name)
    model_name = 'mitosis_phase_classifier/resnet50_all_class.pt'
    classifier_model.load_state_dict(torch.load(model_name))

    return classifier_model


if __name__ == "__main__":

    data_dir = 'yolov7/runs/detect/exp8/data_mitotic_detections_80X80'
    dataloader = get_image_dataloader(data_dir)

    classifier_model = get_model()
    classifier_model.eval()

    with torch.no_grad():
        for i, inputs in enumerate(dataloader):
            imgs = inputs[0]
            inputs = inputs[1]
            
            #print(inputs.shape)
            outputs = classifier_model(inputs)
            _, preds = torch.max(outputs, 1)
            print(imgs)
            print(preds)
            if i == 1:
                break
