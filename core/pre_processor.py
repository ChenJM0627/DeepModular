from utils.registry import Registry
PRE_PROCESSOR_REGISTRY = Registry('PRE_PROCESSOR')

class PreProcessorFunction:

    @classmethod
    def from_config(cls, pre_processor_name:str=None):
        if pre_processor_name is None:
            pre_processor = PRE_PROCESSOR_REGISTRY.get('default_pre_processor')
        else:
            pre_processor = PRE_PROCESSOR_REGISTRY.get(pre_processor_name.lower())
        return pre_processor

def register_pre_processor(name:str):
    def wrapper(cls):
        PRE_PROCESSOR_REGISTRY.register(name.lower(), cls)
        return cls

    return wrapper