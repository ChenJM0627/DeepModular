from core import register_pre_processor, pre_processor
from torchvision.transforms import transforms

@register_pre_processor('default_pre_processor')
class DefaultPreProcessor:
    def __init__(self, config):
        self.config = config
        self.trans = None
        self._build_transforms()

    def run(self, imgs):
        imgs = self.trans(imgs)
        return imgs

    def _build_transforms(self):
        if self.config['data'].get('img_size', None) is not None:
            img_sz = self.config['data']['img_sz']
            self.trans = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((img_sz[0], img_sz[1])),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]
            )
        else:
            self.trans = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]
            )

@register_pre_processor('pcba_pre_processor')
class PCBAPreProcessor:
    def __init__(self, config):
        self.config = config
        self.trans = None
        self._build_transforms()

    def run(self, imgs):
        imgs = self.trans(imgs)
        return imgs

    def _build_transforms(self):
        if self.config['data'].get('img_size', None) is not None:
            img_sz = self.config['data']['img_sz']
            self.trans = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((img_sz[0], img_sz[1])),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]
            )
        else:
            self.trans = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]
            )
