from pathlib import Path
import torch
import numpy as np
import cv2
from torchvision.transforms import transforms

class ProductionModel:

    def __init__(self,model_path=None,device='auto',cls_map=None):
        self.cls_map = cls_map
        self.model = None
        self.device = self._auto_select_device() if device == 'auto' else device
        self.img_sz = (2044,1508)
        if model_path is not None:
            self.load_model(model_path)

    def load_model(self,model_path):
        model_path = Path(model_path)
        if model_path.suffix == '.pt':
            if hasattr(torch.jit,'load') and self._is_torchscript_model(model_path):
                self.model = torch.jit.load(model_path)
        else:
            raise NotImplementedError('Only support Production model')

    def _is_torchscript_model(self,model_path):
        try:
            msg = torch.jit.load(model_path)
            return True
        except:
            print(msg.code)
            return False

    def _auto_select_device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def predict(self,image):
        output = self.preprocess(image)
        self.model.eval()
        with torch.no_grad():
            output = self.model(output)
        output = self.normalize_list(output)
        max_index = str(np.argmax(output))

        return self.cls_map[max_index]

    def normalize_list(self,input_list):
        if len(input_list) == 0:
            return input_list
        input_list = np.array(input_list.cpu())
        min_val = np.min(input_list)
        max_val = np.max(input_list)

        if min_val == max_val:
            return [0.5]*len(input_list)

        return [(x-min_val)/(max_val - min_val) for x in input_list]

    def set_img_sz(self, img, img_sz: tuple[int, int]):

        width, height = img.shape[:2]
        target_width = img_sz[0]
        target_height = img_sz[1]

        ratio = min(target_width / width, target_height / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)

        resized_img = cv2.resize(img, (new_height, new_width))
        new_img = np.zeros((img_sz[0], img_sz[1], 3), np.uint8)

        x_offset = int((target_width - new_width) // 2)
        y_offset = int((target_height - new_height) // 2)

        new_img[x_offset:x_offset + new_width, y_offset:y_offset + new_height, :] = resized_img

        return new_img

    def preprocess(self,image):
        if self.img_sz[0] != image.shape[0] or self.img_sz[1] != image.shape[1]:
            image = self.set_img_sz(image,self.img_sz)
        trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        image = trans(image)
        image = image.to(self.device).unsqueeze(0)
        return image

if __name__ == '__main__':
    cls_map = {"0":"ng","1":"ok"}
    model_path = '/home/unitx/CJM/DeepModular/runs/checkpoint/checkpoint_epoch_950_production.pt'
    img_path = '/home/unitx/CJM/DeepModular/data/PCBA/ok/0 (11).png'
    with open(img_path, 'rb') as f:
        img_bytes = bytearray(f.read())
    parm = np.asarray(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(parm, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    predictor = ProductionModel(model_path,cls_map=cls_map)
    output = predictor.predict(img)

    print(output)
