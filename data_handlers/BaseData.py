import glob
import os
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm", "heic"}

class BaseDataHandler:
    def __init__(self, config,pre_processor):
        self.config = config
        self.pre_processor = pre_processor
        self.train_dataset = self._gen_train_dataset()
        self.val_dataset = self._gen_val_dataset()
        self.train_data_loader = self._gen_train_data_loader()
        self.val_data_loader = self._gen_val_data_loader()

    def get_train_loader(self):
        return self.train_data_loader

    def get_val_loader(self):
        return self.val_data_loader

    def _gen_train_dataset(self):
        raise NotImplementedError

    def _gen_val_dataset(self):
        raise NotImplementedError

    def _gen_train_data_loader(self):
        return DataLoader(self.train_dataset, batch_size=int(self.config['batch_size']), shuffle=True, num_workers=int(self.config['num_workers']))

    def _gen_val_data_loader(self):
        return DataLoader(self.val_dataset, batch_size=int(self.config['batch_size']), shuffle=False, num_workers=int(self.config['num_workers']))

    def get_num_classes(self):
        return len(self.train_dataset.label_map)


class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()

    def read_img(self, path:str):
        with open(path, 'rb') as f:
            img_bytes = bytearray(f.read())
        parm = np.asarray(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(parm, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

class BaseClsDataset(BaseDataset):
    def __init__(self,config, pre_processor):
        super().__init__()
        self.config = config
        self.pre_processor = pre_processor
        self.data_dir = self.config.get('data_dir')
        self.im_files = []
        self.labels = []
        self.label_map = {}
        self.ni = len(self.labels)  # number of images
        self.get_img_files(self.data_dir)  # 获取指定地址的所有图像
        self.get_labels()

    def get_img_files(self, img_path: str | list[str]):
        try:
            f = []# image files
            l = []
            p = Path(img_path)
            assert p.is_dir(), f"{img_path} is not a directory"
            label_list = p.glob('*')
            for label in label_list:
                img_list = label.glob('*')
                for img in img_list:
                    if img.is_file() and str(img.absolute()).rpartition(".")[-1].lower() in IMG_FORMATS:
                        l.append(label.name)
                        f.append(img.absolute())
            self.im_files = f
            self.labels = l
            assert len(f) == len(l), f"{len(f)} != {len(l)}"
        except Exception as e:
            raise FileNotFoundError(f"Error loading data from {img_path}\n") from e

    def get_labels(self) -> None:
        classes = sorted(os.listdir(self.data_dir))
        label_type = self.config.get('label_type')
        self.label_map = {cls_name: i for i, cls_name in enumerate(classes)}
        if not label_type:
            raise ValueError('label_type is missing')

        if label_type == 'int':
            self.labels = [self.label_map[str(label)] for label in self.labels]
        if label_type == 'one-hot':
            self.labels = [self.to_onehot(self.label_map[str(label)]) for label in self.labels]

    def to_onehot(self,labels):
        onehot = np.zeros(len(self.label_map), dtype=np.float32)
        onehot[labels] = 1
        return onehot

class BaseSimilarDataset(BaseDataset):
    def __init__(self,config,pre_processor):
        super().__init__()
        self.config = config
        self.pre_processor = pre_processor
        self.data_dir = self.config.get('data_dir')
        self.same_data = []
        self.load_data()

    def load_data(self):
        data_path = list(Path(self.data_dir).iterdir())
        for folder in data_path:
            img_list = Path(folder).glob('*.png')
            img_path_list = []
            for img_path in img_list:
                img_path_list.append(str(img_path))
            self.same_data.append(img_path_list)


