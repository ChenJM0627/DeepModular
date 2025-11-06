import numpy as np
import torch

from .data_handler import DataHandler
from .evaluator import Evaluator
from .loss import LossFunction
from .model import Model
from .optimizer import OptimizerBuilder
from .pre_processor import PreProcessorFunction
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        self.img_sz = None

        self.model = self._build_model()
        self.data_handler = self._build_data_handler()
        self.preprocessor = self._build_preprocessor()
        self.optimizer, self.scheduler = self._build_optimizer()
        self.loss_fn = self._build_loss()
        # self.postprocessor = self._build_postprocessor()
        self.evaluator = self._build_evaluator()
        run_time = datetime.now().strftime('%Y%m%d:%H:%M')
        self.visualizer = SummaryWriter(f'{self.config['log']['log_dir']}/{run_time}')

        self.current_epoch = 0

    def train(self):
        if self.config['task'] == 'cls':
            self.cls_train()
        elif self.config['task'] == 'similar':
            self.similarity_train()
        elif self.config['task'] == 'obj_detect':
            self.obj_train()

        self.visualizer.close()

    def _train_epoch(self):
        total_loss = 0
        for batch in self.data_handler.get_train_loader():
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            #前向传递
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            total_loss += loss.item()
            self.evaluator.update(outputs, targets)

            #反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return total_loss / len(self.data_handler.get_train_loader())


    def _build_model(self):
        return Model.from_config(self.config['model']['name'])

    def _build_data_handler(self):
        return DataHandler.from_config(self.config)

    def _build_preprocessor(self):
        return PreProcessorFunction.from_config(self.config.get('pre_processor',None))

    def _build_optimizer(self):
        return OptimizerBuilder.from_config(self.config['optimizer'],self.config['optimizer'].get('scheduler',None))

    def _build_loss(self):
        return LossFunction.from_config(self.config['loss']['name'],self.config['data']['label_type'])

    def _build_postprocessor(self):
        return None

    def _build_evaluator(self):
        return Evaluator

    def _save_checkpoint(self, epoch):
        img_sz = self.img_sz
        metadata = {
            "model_name": self.config['model']['name'],
            "create_time": datetime.now().strftime('%Y%m%d:%H:%M'),
            "input_size": img_sz
        }
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_architecture': str(self.model),
            'metadata': metadata,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
        }
        torch.save(checkpoint, Path(self.config['trainer']['checkpoint_path'])/f'checkpoint_epoch_{epoch}.pth')
        if self.config['trainer'].get('save_for_production',False):
            self.model.eval()
            example_input = torch.randn(self.config['data']['batch_size'], 3, img_sz[0], img_sz[1]).cuda()
            trace_script_module = torch.jit.trace(self.model, example_input)
            trace_script_module.save(Path(self.config['trainer']['checkpoint_path'])/f'checkpoint_epoch_{epoch}_production.pt')


    def cls_train(self):
        self.preprocessor = self.preprocessor(self.config)
        self.data_handler = self.data_handler(self.config['data'], self.preprocessor)
        self.model = self.model(num_classes=self.data_handler.get_num_classes()).cuda() if self.config[
                                                                                               'device'] == 'cuda' else self.model(
            num_classes=self.config['data']['num_classes'])
        self.optimizer = self.optimizer(self.model.parameters(), params=self.config['optimizer'])
        self.scheduler = self.scheduler(optimizer=self.optimizer, params=self.config['optimizer']['scheduler'])
        self.loss_fn = self.loss_fn()
        self.evaluator = self.evaluator(self.config, self.visualizer)

        self.model.train()
        if self.config['data'].get('img_sz',None) is not None:
            self.img_sz = self.config['data']['img_sz']
        else:
            data_iter = iter(self.data_handler.get_train_loader())
            batch = next(data_iter)
            inputs, targets = batch
            self.img_sz = inputs.shape[-2:]
        for epoch in range(self.current_epoch, self.config['trainer']['epoch']):
            avg_loss = self._train_epoch()
            self.visualizer.add_scalar('loss', avg_loss, global_step=epoch)
            self.visualizer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], global_step=epoch)

            if self.scheduler:
                self.scheduler.step()

            if epoch % self.config['trainer']['save_freq'] == 0:
                self._save_checkpoint(epoch)

            self.evaluator.compute(epoch)

    def similarity_train(self):
        pass

    def obj_train(self):
        self.preprocessor = self.preprocessor(self.config)
        self.data_handler = self.data_handler(self.config['data'], self.preprocessor)
        self.model = self.model(nc=self.data_handler.get_num_classes()).cuda() if self.config[
                                                                                               'device'] == 'cuda' else self.model(
            num_classes=self.config['data']['num_classes'])
        self.optimizer = self.optimizer(self.model.parameters(), params=self.config['optimizer'])
        self.scheduler = self.scheduler(optimizer=self.optimizer, params=self.config['optimizer']['scheduler'])
        self.loss_fn = self.loss_fn()
        self.evaluator = self.evaluator(self.config, self.visualizer)

        self.model.train()
        for epoch in range(self.current_epoch, self.config['trainer']['epoch']):
            avg_loss = self._train_epoch()
            self.visualizer.add_scalar('loss', avg_loss, global_step=epoch)
            self.visualizer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], global_step=epoch)

            if self.scheduler:
                self.scheduler.step()

            if epoch % self.config['trainer']['save_freq'] == 0:
                self._save_checkpoint(epoch)

            self.evaluator.compute(epoch)