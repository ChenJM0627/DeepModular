import cv2

from .BaseData import BaseClsDataset,BaseDataHandler
from core import register_data_handler


class ClsDataset(BaseClsDataset):
    def __init__(self,config,pre_processor):
        super().__init__(config,pre_processor)

    def __len__(self):
        return len(self.im_files)

    def __getitem__(self, index):
        img_path = self.im_files[index]
        label = self.labels[index]
        img = self.read_img(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.pre_processor.run(img)

        return img, label


@register_data_handler('cls_data')
class ClsDataHandler(BaseDataHandler):
    def __init__(self,config,pre_processor):
        super().__init__(config,pre_processor)

    def _gen_train_dataset(self):
        return ClsDataset(self.config,self.pre_processor)

    def _gen_val_dataset(self):
        return ClsDataset(self.config,self.pre_processor)

