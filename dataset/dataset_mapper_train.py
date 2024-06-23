import copy
import cv2
import numpy as np
import torch
from PIL import Image
import albumentations as A

def read_image(file_name, format="RGB"):
    return np.array(Image.open(file_name).convert(format)).astype(np.uint8)

def dataset_mapper_train(dataset_dict, cfg):
    ret = dataset_dict | {"height": 224, "width": 224}
    return ret