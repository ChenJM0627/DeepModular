class Registry:
    def __init__(self,name):
        self.name = name
        self._registry = {}

    def register(self, name, obj):
        if name in self._registry:
            raise Exception(f'{name} is already registered')
        self._registry[name] = obj

    def get(self, name:str):
        if name.lower() not in self._registry:
            raise Exception(f'{name} is not registered')
        return self._registry[name.lower()]

    def __contains__(self, name):
        return name in self._registry