#!/usr/bin/env python
# coding: utf-8

# In[19]:


import torch
import torch.nn as nn

torch.cuda.set_device(0)
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

# Print memory reserved and allocated
print(torch.cuda.memory_reserved())
print(torch.cuda.memory_allocated())


# In[20]:


# Initialize your model
model = YOLO('yolov8m-seg.pt')

# Choose GPU 1 directly
device = 'cuda:1'

# Move model to GPU 1
model.to(device)

# To use DataParallel and both GPUs have enough memory
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)


# In[5]:


pip install -r ~/Documents/inseer_ML_Project/yolov8/requirements.txt


# In[6]:


pip install ultralytics


# In[7]:


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


# In[16]:


# Paths to the images and labels
images_path = "/home/nikhil/Documents/Datasets/GTEA dataset/GTEA_original_croppeddataset/Cropped_Images"
labels_path = "/home/nikhil/Documents/Datasets/GTEA dataset/GTEA_original_croppeddataset/Cropped_yoloLabels"


# In[61]:


# Function to split dataset into train and test
import os
import shutil
import random

# Set the seed for reproducibility
random.seed(42)

# Define the paths
source_images = "/home/nikhil/Documents/Datasets/GTEA dataset/GTEA_original_croppeddataset/Cropped_Images"
source_labels = "/home/nikhil/Documents/Datasets/GTEA dataset/GTEA_original_croppeddataset/Cropped_yoloLabels"
destination = "/home/nikhil/Documents/Datasets/GTEA dataset/GTEA_original_croppeddataset/GTEA_splitdata/"

# Create train and test directories
os.makedirs(destination + 'train/images', exist_ok=True)
os.makedirs(destination + 'train/labels', exist_ok=True)
os.makedirs(destination + 'test/images', exist_ok=True)
os.makedirs(destination + 'test/labels', exist_ok=True)

# List all images
all_images = [f for f in os.listdir(source_images) if os.path.isfile(os.path.join(source_images, f))]
# Shuffle the list
random.shuffle(all_images)

# Split images into train and test (90% train, 10% test)
split_idx = int(0.9 * len(all_images))
train_images = all_images[:split_idx]
test_images = all_images[split_idx:]

# Function to copy files
def copy_files(files, src_folder, dest_folder):
    for file in files:
        src_file = os.path.join(src_folder, file)
        dest_file = os.path.join(dest_folder, file)
        shutil.copy(src_file, dest_file)

# Copy train images and labels
copy_files(train_images, source_images, destination + 'train/images/')
copy_files([f.replace('.jpg', '.txt') for f in train_images], source_labels, destination + 'train/labels/')

# Copy test images and labels
copy_files(test_images, source_images, destination + 'test/images/')
copy_files([f.replace('.jpg', '.txt') for f in test_images], source_labels, destination + 'test/labels/')

print(f"Dataset split completed. Train set: {len(train_images)} images, Test set: {len(test_images)} images.")


# In[62]:


# Create YAML configuration for training
def create_yaml_configuration(train_path, test_path, val_path, num_classes=1):
    yaml_content = f"""
# YoloV8 training on GTEA dataset
train: /home/nikhil/Documents/Datasets/GTEA dataset/GTEA_original_croppeddataset/GTEA_splitdata/train
test: /home/nikhil/Documents/Datasets/GTEA dataset/GTEA_original_croppeddataset/GTEA_splitdata/test
val: /home/nikhil/Documents/Datasets/GTEA dataset/GTEA_original_croppeddataset/GTEA_splitdata/test

# Classes
nc: {num_classes}
names: ['hand']
    """
    yaml_file_path = '/home/nikhil/Documents/Datasets/GTEA dataset/GTEA_original_croppeddataset/GTEA_splitdata/GTEA_crop_hands.yaml'  # Path where the YAML file will be saved
    with open(yaml_file_path, 'w') as yaml_file:
        yaml_file.write(yaml_content)
    return yaml_file_path

# Adjusting the paths for train, test, and val
train_path = os.path.join(images_path, 'train')
test_path = os.path.join(images_path, 'test')
val_path = os.path.join(images_path, 'test')  # Assuming validation and test paths are the same

yaml_file_path = create_yaml_configuration(train_path, test_path, val_path)


# In[63]:


#training model
def train_model(yaml_file_path, batch=1, imgsz=640):
    model = YOLO("yolov8m-seg.pt")
    model.train(data=yaml_file_path, epochs=80, batch=batch, imgsz=imgsz, project="GTEA-Hand", name="GTEA_Hand_Segmentation")
    model.val()


train_model('/home/nikhil/Documents/Datasets/GTEA dataset/GTEA_original_croppeddataset/GTEA_splitdata/GTEA_crop_hands.yaml', batch=1)


# In[64]:


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
            np.save(seg_save_path.as_posix(), all_segs, allow_pickle=True)
            if show:
                plt.imshow(image)
                plt.show()
        else:
            print(f"no results for image {image_path.name}!")



if __name__ == "__main__":
    weights_path = "/home/nikhil/Documents/inseer_ML_Project/yolov8/Nikhil-yolov8 training/GTEAtraining_yolo/GTEA-Hand/GTEA_Hand_Segmentation/weights/best.pt"
    image_dir = "/home/nikhil/Documents/Datasets/GTEA dataset/GTEA_original_croppeddataset/GTEA_splitdata/test/images"

    test_model(weights_path, image_dir, show=True)


# In[65]:


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
image_dir = "/home/nikhil/Documents/Datasets/GTEA dataset/GTEA_original_croppeddataset/GTEA_splitdata/test/images"


# Path to the directory where the .npy segmentations are stored
segs_dir = '/home/nikhil/Documents/Datasets/GTEA dataset/GTEA_original_croppeddataset/GTEA_splitdata/test/test_yolov8_segs'
# Directory to save the plots
plots_save_dir = '/home/nikhil/Documents/Datasets/GTEA dataset/GTEA_original_croppeddataset/GTEA_splitdata/test/saved_plots'

display_segments_on_images(images_dir, segs_dir, plots_save_dir)


# In[66]:


#Testing model with 0.3 confidence
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
    save_dir = Path(image_dir).parent / "test_yolov8_segs_0.3conf"
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

        results = model.predict(source=image, conf=0.3, iou=0.4)
        seg = results[0]
        if seg:
            all_segs = []
            for i in range(len(seg.masks.xy)):
                seg_xy = seg.masks.xy[i]
                all_segs.append(seg_xy)
                image = plot_polygon_on_image(image, seg_xy)
            seg_save_path = save_dir / image_path.with_suffix('.npy').name
            np.save(seg_save_path.as_posix(), all_segs, allow_pickle=True)
            if show:
                plt.imshow(image)
                plt.show()
        else:
            print(f"no results for image {image_path.name}!")



if __name__ == "__main__":
    weights_path = "/home/nikhil/Documents/inseer_ML_Project/yolov8/Nikhil-yolov8 training/GTEAtraining_yolo/GTEA-Hand/GTEA_Hand_Segmentation/weights/best.pt"
    image_dir = "/home/nikhil/Documents/Datasets/GTEA dataset/GTEA_original_croppeddataset/GTEA_splitdata/test/images"
    test_model(weights_path, image_dir, show=True)


# In[67]:


#Code to plot the segments(in .npy format) on the images 0.3 confidence
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

image_dir = "/home/nikhil/Documents/Datasets/GTEA dataset/GTEA_original_croppeddataset/GTEA_splitdata/test/images"


# Path to the directory where the .npy segmentations are stored
segs_dir = '/home/nikhil/Documents/Datasets/GTEA dataset/GTEA_original_croppeddataset/GTEA_splitdata/test/test_yolov8_segs_0.3conf'
# Directory to save the plots
plots_save_dir = '/home/nikhil/Documents/Datasets/GTEA dataset/GTEA_original_croppeddataset/GTEA_splitdata/test/saved_plots_0.3conf'

display_segments_on_images(images_dir, segs_dir, plots_save_dir)



# In[68]:


#To count number of empty labels in test dataset (no hands in images)
from pathlib import Path

def count_empty_files(directory):
    # Convert string path to Path object
    path = Path(directory)

    # Check if directory exists
    if not path.exists() or not path.is_dir():
        return "Directory does not exist or is not a directory."

    # Count empty files
    empty_files = sum(1 for file in path.glob('*.txt') if not file.stat().st_size)
    return empty_files

# Path to the labels directory
labels_directory = "/home/nikhil/Documents/Datasets/GTEA dataset/GTEA_original_croppeddataset/GTEA_splitdata/test/labels"

# Call the function and print the result
empty_file_count = count_empty_files(labels_directory)
print(f"Number of empty txt files: {empty_file_count}")


# In[ ]:




