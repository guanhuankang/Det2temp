from detectron2.config import CfgNode as CN

def add_custom_config(cfg):
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
