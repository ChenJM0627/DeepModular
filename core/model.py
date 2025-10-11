from utils.registry import Registry

MODEL_REGISTRY = Registry('MODEL')

class Model:
    @classmethod
    def from_config(cls, model_name):
        return MODEL_REGISTRY.get(model_name.lower())

def register_model(name:str):
    def wrapper(cls):
        MODEL_REGISTRY.register(name.lower(),cls)
        return cls
    return wrapper
