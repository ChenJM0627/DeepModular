from utils.registry import Registry
DATASET_REGISTRY = Registry('DATASET')

class DataHandler:
    @classmethod
    def from_config(cls, config):
        task_name = config['task']
        if task_name.lower() == 'cls':
            name = config['data']['name']
        elif task_name.lower() == 'similar':
            name = config['data']['name']
        elif task_name.lower() == 'obj_detect':
            name = config['data']['name']
        else:
            raise NotImplementedError(f'{task_name} not implemented')
        return DATASET_REGISTRY.get(name)


def register_data_handler(name:str):
    def wrapper(cls):
        DATASET_REGISTRY.register(name.lower(),cls)
        return cls
    return wrapper