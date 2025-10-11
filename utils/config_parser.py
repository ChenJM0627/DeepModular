from typing import Dict,Any
import yaml
import os
from pathlib import Path

def parse_config(config_path)->Dict[str,Any]:
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f'配置文件不存在:{config_path}')

    if not config_path.suffix == '.yaml':
        raise ValueError(f'配置文件必须是YAML格')

    try:
        with open(config_path.absolute(),'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f'配置文件解析错误:{e}')

    required_sections = [
        'model',
        'data',
        'optimizer',
        'trainer',
    ]

    for section in required_sections:
        if section not in config:
            raise ValueError(f'配置文件缺少必要配置:{section}')

    return config



