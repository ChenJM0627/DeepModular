import torch.optim as optim
from core import register_optimizer
from core import register_scheduler

@register_optimizer('SGD')
class SGD(optim.SGD):
    def __init__(self,model,params:dict):
        super(SGD, self).__init__(model,lr=float(params['lr']), momentum=float(params['momentum']))

@register_optimizer('Adam')
class Adam(optim.Adam):
    def __init__(self,model,params:dict):
        super(Adam, self).__init__(model,lr=float(params['lr']), weight_decay=params['weight_decay'])

@register_scheduler('StepLR')
class StepLR(optim.lr_scheduler.StepLR):
    def __init__(self,optimizer,params:dict):
        super(StepLR, self).__init__(optimizer,step_size=int(params['step_size']),gamma=float(params['gamma']))

@register_scheduler('CosineAnnealingLR')
class CosineAnnealingLR(optim.lr_scheduler.CosineAnnealingWarmRestarts):
    def __init__(self,optimizer,params:dict):
        super(CosineAnnealingLR, self).__init__(optimizer,T_0=params['T_0'])