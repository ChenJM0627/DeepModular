import torch

from core import register_confidence_evaluator,register_performance_evaluator
from .BaseEvaluator import *

@register_confidence_evaluator(name='cls_validate')
class cls_validate(BaseConfidenceEvaluator):
    def __init__(self):
        super().__init__()

    def evaluate(self, input, target):
        min_val = torch.min(input,dim=1,keepdim=True).values
        max_val = torch.max(input,dim=1,keepdim=True).values
        normalized = (input-min_val)/(max_val-min_val)
        preds = torch.argmax(normalized, dim=1).unsqueeze(1)
        if len(target.shape) == 1:
            target = target.unsqueeze(1)
            res = preds == target
            res = res.int()
        else:
            target = torch.argmax(target, dim=1).unsqueeze(1)
            res = preds == target
            res = res.int()
        return res.flatten().tolist()

@register_performance_evaluator(name='cls_accuracy')
class accuracy(BasePerformanceEvaluator):
    def __init__(self):
        super().__init__()

    def compute(self,results:list):
        ok = results.count(1)
        acc = ok/len(results)
        return acc