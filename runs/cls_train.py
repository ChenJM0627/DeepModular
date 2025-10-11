from core import *
from utils.config_parser import *
import argparse

def cls_train():
    parser = argparse.ArgumentParser(description='DeepModular Training')
    parser.add_argument('--config_path', type=str, default=r'/home/unitx/CJM/DeepModular/configs/cls_default.yaml',required=False,help='Path to configuration file')
    args = parser.parse_args()
    config = parse_config(args.config_path)
    #创建训练器
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    cls_train()


