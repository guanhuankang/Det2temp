from detectron2.config import CfgNode as CN

def add_toy_config(cfg):
    cfg.TOY = CN()
    return cfg