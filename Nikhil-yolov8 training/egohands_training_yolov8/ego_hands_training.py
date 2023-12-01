#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import torch.nn as nn

torch.cuda.set_device(0)
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

# Print memory reserved and allocated
print(torch.cuda.memory_reserved())
print(torch.cuda.memory_allocated())


# In[4]:


#To use DataParallel and both GPUs have enough memory
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")


# In[1]:


pip install -r ~/Documents/inseer_ML_Project/yolov8/requirements.txt


# In[2]:


pip install ultralytics


# In[5]:


from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from types import SimpleNamespace
import time
from sklearn.model_selection import train_test_split
from zipfile import ZipFile
import shutil
import glob


# In[6]:


#training model
def train_model(yaml_file_path, batch=1, imgsz=640):
    model = YOLO("yolov8m-seg.pt")
    model.train(data=yaml_file_path, epochs=80, batch=batch, imgsz=imgsz, project="Ego-Hands", name="Ego_Hands_Segmentation")
    model.val()


train_model('/home/nikhil/Documents/Datasets/ego_hands dataset/ego_hands_crops_yolov8/ego_hands_crops.yaml', batch=1)


# In[8]:


#Testing model
import os
import cv2
import numpy as np
from glob import glob
from pathlib import Path
from random import randint
import matplotlib as mpl
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

def plot_polygon_on_image(image, polygon):
    polygon = np.asarray(polygon)
    polygon = np.squeeze(polygon)
    image = cv2.polylines(image, [polygon.astype('int')], True, (randint(0, 255), randint(0, 255), randint(0, 255)),
                  2)  # Draw Poly Lines
    image = cv2.fillPoly(image, [polygon.astype('int')], (randint(0,255),randint(0,255),randint(0,255)), cv2.LINE_AA) # Draw area

    return image

def test_model(weights_path, image_dir, show=False):
    image_dir = Path(image_dir)
    save_dir = Path(image_dir).parent / "test_yolov8_segs"
    save_dir.mkdir(exist_ok=True, parents=True)
    assert image_dir.is_dir()
    # Load a model
    model = YOLO(weights_path)  # load pretrained segmentation model
    # predict with model
    for image_path in image_dir.iterdir():
        image_ext = image_path.suffix
        print("image name", image_path.name)
        if image_ext == ".npy":
            image = np.load(image_path.as_posix(), allow_pickle=True)
        else:
            image = cv2.imread(image_path.as_posix())

        results = model.predict(source=image, conf=0.5, iou=0.4)
        seg = results[0]
        if seg:
            all_segs = []
            for i in range(len(seg.masks.xy)):
                seg_xy = seg.masks.xy[i]
                all_segs.append(seg_xy)
                image = plot_polygon_on_image(image, seg_xy)
            seg_save_path = save_dir / image_path.with_suffix('.npy').name
            np.save(seg_save_path.as_posix(), np.array(all_segs, dtype=object), allow_pickle=True)

            if show:
                plt.imshow(image)
                plt.show()
        else:
            print(f"no results for image {image_path.name}!")



if __name__ == "__main__":
    weights_path = "/home/nikhil/Documents/inseer_ML_Project/yolov8/Nikhil-yolov8 training/egohands_training_yolov8/Ego-Hands/Ego_Hands_Segmentation/weights/best.pt"
    image_dir = "/home/nikhil/Documents/Datasets/ego_hands dataset/ego_hands_crops_yolov8/test/images"
    test_model(weights_path, image_dir, show=True)


# In[10]:


#Code to plot the segments(in .npy format) on the images
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2

def display_segments_on_images(image_dir, segmentation_dir, save_dir):
    image_dir = Path(image_dir)
    segmentation_dir = Path(segmentation_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)  # Ensure the save directory exists
    
    # Iterate over .npy files and display them on the corresponding images
    for seg_file in segmentation_dir.glob('*.npy'):
        image_file = image_dir / (seg_file.stem + '.jpg')
        if not image_file.exists():
            print(f"No corresponding image for segmentation {seg_file.stem}")
            continue
        
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        
        segmentations = np.load(seg_file, allow_pickle=True)
        
        plt.figure()
        plt.imshow(image)
        
        for polygon in segmentations:
            plt.plot(*np.array(polygon).T, marker='o', color='red')
        
        plt.axis('off')
        save_path = save_dir / f"{image_file.stem}_segmented.png"
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

# Define the paths to directories
images_dir = "/home/nikhil/Documents/Datasets/ego_hands dataset/ego_hands_crops_yolov8/test/images"


# Path to the directory where the .npy segmentations are stored
segs_dir = '/home/nikhil/Documents/Datasets/ego_hands dataset/ego_hands_crops_yolov8/test/test_yolov8_segs'
# Directory to save the plots
plots_save_dir = '/home/nikhil/Documents/Datasets/ego_hands dataset/ego_hands_crops_yolov8/test/saved_plotsonimages'

display_segments_on_images(images_dir, segs_dir, plots_save_dir)


# In[ ]:





# In[ ]:





# In[ ]:




