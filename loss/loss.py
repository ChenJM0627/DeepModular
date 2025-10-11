import torch.nn as nn
from core import register_loss_fn

@register_loss_fn(name='CELoss')
class CELoss(nn.CrossEntropyLoss):
    label_type = 'int'
    def __init__(self):
        super(CELoss, self).__init__()

@register_loss_fn(name='BCELoss')
class BCELoss(nn.BCEWithLogitsLoss):
    label_type = 'one-hot'
    def __init__(self):
        super(BCELoss, self).__init__()
