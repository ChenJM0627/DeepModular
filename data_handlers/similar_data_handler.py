import random

from .BaseData import BaseSimilarDataset, BaseDataHandler
from core import register_data_handler
import torch

class SimilarDataset(BaseSimilarDataset):
    def __init__(self,config,pre_processor):
        super().__init__(config,pre_processor)
        self.data_size = config['data']['data_size']

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        if index % 2 == 0:
            target = torch.tensor(1, dtype=torch.float)
            index_1 = random.randint(0, len(self.same_data)-1)
            index_2 = random.sample(range(len(self.same_data[index_1])), 2)
            img_1 = torch.tensor(self.read_img(self.same_data[index_1][index_2[0]])).permute(2,0,1).float()
            img_2 = torch.tensor(self.read_img(self.same_data[index_1][index_2[1]])).permute(2,0,1).float()

        else:
            target = torch.tensor(0, dtype=torch.float)
            index_1 = random.sample(range(len(self.same_data)), 2)
            index_2 = random.randint(0, len(self.same_data[index_1[0]])-1)
            index_3 = random.randint(0, len(self.same_data[index_1[1]])-1)
            img_1 = torch.tensor(self.read_img(self.same_data[index_1[0]][index_2])).permute(2,0,1).float()
            img_2 = torch.tensor(self.read_img(self.same_data[index_1[1]][index_3])).permute(2,0,1).float()
        img_1 = self.pre_processor.run(img_1)
        img_2 = self.pre_processor.run(img_2)

        return img_1, img_2, target

@register_data_handler('similar_data')
class SimilarDataHandler(BaseDataHandler):
    def __init__(self,config,pre_processor):
        super().__init__(config,pre_processor)

    def _gen_train_dataset(self):
        return SimilarDataset(self.config,self.pre_processor)

    def _gen_val_dataset(self):
        return SimilarDataset(self.config,self.pre_processor)