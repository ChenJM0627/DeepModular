from utils.registry import Registry
LOSS_REGISTRY = Registry('LOSS')

class LossFunction:

    @classmethod
    def from_config(cls, loss_fn_name:str, label_type):

        loss_fn = LOSS_REGISTRY.get(loss_fn_name.lower())
        if label_type == loss_fn.label_type:
            return loss_fn
        else:
            raise NotImplementedError(f'current label type:{label_type} loss label type {loss_fn.label_type} is not corrected')

def register_loss_fn(name:str):
    def wrapper(cls):
        LOSS_REGISTRY.register(name.lower(), cls)
        return cls

    return wrapper