from detectron2.config import CfgNode as CN
from .toy import add_toy_config
# from .detr import add_detr_config

def add_project_config(cfg):
    cfg = add_global_config(cfg)
    
    cfg = add_toy_config(cfg)
    # cfg = add_detr_config(cfg)
    
    return cfg

def add_global_config(cfg):
    cfg.MODEL.WEIGHTS = ""
    
    cfg.DATASETS.ROOT = "assets"
    
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    cfg.SOLVER.IMS_PER_GPU = 32
    cfg.SOLVER.NUM_GPUS = 1
    
    cfg.TEST.AUG = CN()
    cfg.TEST.AUG.ENABLED = False
    
    cfg.INPUT.TRAIN_RESOLUTION = 800
    cfg.INPUT.TEST_RESOLUTION = 800
    return cfg