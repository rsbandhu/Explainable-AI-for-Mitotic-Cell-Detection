import time
import torch

from .classifier_model import get_image_dataloader
from .classifier_model import get_model

def detect_mitotic_phases(data_dir, model_clsf, all_mitotic_detections):

    ### perform mitotic phase and background classification of the detections that are mitotic

    #data_dir = 'runs/detect/exp8/data_mitotic_detections_80X80'
    # Create dataloader
    dataloader = get_image_dataloader(data_dir)

    # get classifier model and set it to eval mode
    mitotic_clsf = get_model(model_clsf)
    
    mitotic_clsf.eval()

    # iterate through the cropped images to perform classification
    #t0 = time.time()
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

    #print(f"time to classify mitotic detection areas : {time.time() - t0}")

    #for k, v in all_mitotic_detections.items():
    #    print(k, v)

    return all_mitotic_detections
