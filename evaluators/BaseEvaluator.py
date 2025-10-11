class BaseEvaluator:
    def __init__(self):
        pass

class BaseConfidenceEvaluator(BaseEvaluator):
    def __init__(self):
        super(BaseConfidenceEvaluator, self).__init__()

    def evaluate(self, output, target):
        raise NotImplementedError('未定义 Confidence 单个数据评估方式')

class BasePerformanceEvaluator(BaseEvaluator):
    def __init__(self):
        super(BasePerformanceEvaluator, self).__init__()

    def compute(self, result):
        raise NotImplementedError('未定义 Performance 评估方式')