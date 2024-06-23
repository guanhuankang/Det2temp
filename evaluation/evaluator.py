import itertools
import detectron2.utils.comm as comm
from detectron2.evaluation import DatasetEvaluator

class Evaluator(DatasetEvaluator):
    def __init__(self, cfg, dataset_name) -> None:
        super().__init__()
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.results = []

    def reset(self):
        self.results = []

    def process(self, inputs, outputs):
        """
        Params:
            @inputs: model input
            @outputs: model output (it should be in numpy format)
        """
        self.results.append({})
    
    def evaluate(self):
        if self.cfg.SOLVER.NUM_GPUS > 1:
            comm.synchronize()
            results = comm.gather(self.results, dst=0)
            results = list(itertools.chain(*results))
            if not comm.is_main_process():
                return {}
        else:
            results = self.results

        ## Handle Results
        return {"num": len(results)}