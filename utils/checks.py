import torch
from packaging import version

torch_str = torch.__version__

def check_version(
    name: str,
    required: str,

) -> bool:
    if not required:
        return True
    if name == 'torch':
        if version.parse(torch_str) < version.parse(required):
            return False
        else:
            return True

    return True
