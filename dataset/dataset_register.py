import json
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog

def loadJson(file):
    with open(file, "r") as f:
        data = json.load(f)
    return data

def prepare_dataset_list_of_dict(dataset_id, split, root):
    return [{"name": str(_), "data": np.random.rand(224, 224, 3)} for _ in range(10)]

def register_dataset(cfg):
    DatasetCatalog.register("mock_train", lambda s="train": prepare_dataset_list_of_dict(dataset_id="mock", split=s, root=cfg.DATASETS.ROOT))
    
    DatasetCatalog.register("mock_test", lambda s="test": prepare_dataset_list_of_dict(dataset_id="mock", split=s, root=cfg.DATASETS.ROOT))
    