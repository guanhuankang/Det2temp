import os
import json
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog

def loadJson(file):
    with open(file, "r") as f:
        data = json.load(f)
    return data

def prepare_dataset_list_of_dict(annotation_file, root):
    pass

def register_dataset(cfg):
    dataset_tuple = [
        ("rvsor_2019_train", "dataset_prepare/vsor_dataset_bbox_based/RVSOR_train.json", "2019_RVSOD/train"),
        ("rvsor_2019_test", "dataset_prepare/vsor_dataset_bbox_based/RVSOR_test.json", "2019_RVSOD/test"),
        ("rvsor_2019_val", "dataset_prepare/vsor_dataset_bbox_based/RVSOR_val.json", "2019_RVSOD/validation"),
        
        ("davsor_2024_train", "dataset_prepare/vsor_dataset_bbox_based/DAVSOR2024_train.json", "2024_DAVSOR/data"),
        ("davsor_2024_test", "dataset_prepare/vsor_dataset_bbox_based/DAVSOR2024_test.json", "2024_DAVSOR/data"),
        ("davsor_2024_val", "dataset_prepare/vsor_dataset_bbox_based/DAVSOR2024_val.json", "2024_DAVSOR/data"),
    ]
    
    for dataset_id, anno_file, path in zip(dataset_tuple):
        DatasetCatalog.register(dataset_id, lambda : prepare_dataset_list_of_dict(
            annotation_file=os.path.join(cfg.DATASETS.ROOT, anno_file), 
            root=os.pth.join(cfg.DATASETS.ROOT, path)
        ))
    
    