from utils.registry import Registry
OPTIMIZER_REGISTRY = Registry('OPTIMIZER')
SCHEDULER_REGISTRY = Registry('SCHEDULER')

class OptimizerBuilder:

    @classmethod
    def from_config(cls, optimizer_config,scheduler_config=None):
        optimizer = OPTIMIZER_REGISTRY.get(optimizer_config['name'])
        if scheduler_config:
            scheduler = SCHEDULER_REGISTRY.get(scheduler_config['name'])
        else:
            scheduler = None
        return optimizer, scheduler

def register_optimizer(name:str):
    def wrapper(cls):
        OPTIMIZER_REGISTRY.register(name.lower(),cls)
        return cls
    return wrapper

def register_scheduler(name:str):
    def wrapper(cls):
        SCHEDULER_REGISTRY.register(name.lower(),cls)
        return cls
    return wrapper