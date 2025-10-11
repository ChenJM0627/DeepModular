import torch

from utils.registry import Registry
CONFIDENCE_REGISTRY = Registry('confidence')
PERFORMANCE_REGISTRY = Registry('performance')

class Evaluator:
    def __init__(self,config, visualizer):
        self.config = config
        self.visualizer = visualizer
        self.confidence = None
        self.performance = {}
        self._init_evaluator()
        self.results = []

    def _init_evaluator(self):
        name = self.config['evaluator']['confidence']
        self.confidence = CONFIDENCE_REGISTRY.get(name.lower())()

        for name in self.config['evaluator']['performance']:
            self.performance = {name:PERFORMANCE_REGISTRY.get(name.lower())()}


    def update(self, output, target):
        self.results.extend(self.confidence.evaluate(output, target))

    def compute(self,epoch):
        for name, metric in self.performance.items():
            res = metric.compute(self.results)
            self.visualizer.add_scalar(name, res, epoch)
        self.results.clear()

def register_confidence_evaluator(name:str):
    def wrapper(cls):
        CONFIDENCE_REGISTRY.register(name.lower(), cls)
        return cls
    return wrapper

def register_performance_evaluator(name:str):
    def wrapper(cls):
        PERFORMANCE_REGISTRY.register(name.lower(), cls)
        return cls
    return wrapper