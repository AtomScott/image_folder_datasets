# default_exp core


#hide
from nbdev.showdoc import *
from fastcore.test import *


# export 
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import warnings
import torchvision
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms import ToTensor, Resize, Compose, CenterCrop, Normalize
import pytorch_lightning as pl
# from pytorch_lightning.metrics.functional import classification, f1
from pytorch_lightning.loggers import TensorBoardLogger

import fastai.vision.augment
import fastai.vision.data
# from fastai.vision.data import ImageDataLoaders
# from fastai.vision.augment import Resize


#export
class ImageFolderDataModule(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size, transform):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform
#         Compose([
#             Resize(256, interpolation=2),
#             CenterCrop(224),
#             ToTensor(),
# #             TODO: check whether normalize is the same for imagenet and fractalDB
#             Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
#         ])

    def prepare_data(self, stage=None):
        pass
    
    def setup(self, stage=None):
        data_dir = self.data_dir
        transform = self.transform
        
        self.dls = fastai.vision.data.ImageDataLoaders.from_folder(data_dir, item_tfms=fastai.vision.augment.Resize(224))
        self.trainset = ImageFolder(os.path.join(data_dir, 'train'), transform)
        self.valset = ImageFolder(os.path.join(data_dir, 'valid'), transform)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        pass


data_dir = 'Datasets/cifar10'

transform = Compose([
        Resize(256, interpolation=2),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


dm = ImageFolderDataModule(data_dir, 128, transform)
dm.setup()


for x,y in dm.train_dataloader():
    test_eq(type(x), torch.Tensor) 
    test_eq(type(y), torch.Tensor) 
    break


#export
class CNNModule(pl.LightningModule):
    def __init__(self, model=None, pretrained=False, freeze_extractor=False, log_level=10, num_classes=None, weight_path=None):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_extractor = freeze_extractor

        assert model is not None, 'Select model from torchvision'
        assert num_classes is not None, 'Must configure number of classes with num_classes'
        
        if not model.startswith('resnet'):
            warnings.warn('models other than resnet variants may need different setup for finetuning to work.')
            
        # Prepare model for finetuning
        if weight_path is not None:
            param = torch.load(weight_path)
            backbone = eval(f'torchvision.models.{model}(pretrained={False})')     
            backbone.load_state_dict(param)
        else:
            backbone = eval(f'torchvision.models.{model}(pretrained={pretrained})')        
                    
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        
        self.feature_extractor = torch.nn.Sequential(*layers)
        self.classifier = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        if self.freeze_extractor:
            self.feature_extractor.eval()
            with torch.no_grad():
                representations = self.feature_extractor(x).flatten(1)
        else:
            representations = self.feature_extractor(x).flatten(1)

        y = self.classifier(representations)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        outputs = self.calculate_metrics(y_hat=y_hat, y=y)
        return outputs
    
    def training_epoch_end(self, outputs):
        avg_metrics = {}
        for metric in outputs[0].keys():
            val = torch.stack([x[metric] for x in outputs]).mean()
            self.logger.experiment.add_scalar(f"{metric}/train", val, self.current_epoch)
            avg_metrics[metric] = val

#         epoch_dictionary = {'loss': avg_metrics['loss']}
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        outputs = self.calculate_metrics(y_hat=y_hat, y=y)
        return outputs
    
    def validation_epoch_end(self, outputs):
        avg_metrics = {}
        for metric in outputs[0].keys():
            val = torch.stack([x[metric] for x in outputs]).mean()
            self.logger.experiment.add_scalar(f"{metric}/validation", val, self.current_epoch)
            avg_metrics[metric] = val

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02, weight_decay=1e-04)
#     >    return torch.optim.SGF(self.parameters(), lr=self.lr, aldsfk'a)
    
    
    def calculate_metrics(self, y, y_hat):
        loss = F.cross_entropy(y_hat, y)
        y_pred = y_hat.argmax(dim=1)
        acc = classification.accuracy(y_pred, y)
        f1_score = f1(y_pred, y, self.num_classes)
        return {
            "loss":loss,
            "acc": acc,
            "f1": f1_score
        }
    
    def on_sanity_check_start(self):
        self.logger.disable()

    def on_sanity_check_end(self):
        self.logger.enable() 



modelname = 'resnet18'
logger = TensorBoardLogger('tb_logs', name=modelname)
trainer = pl.Trainer(gpus=1, checkpoint_callback=False, logger=logger, fast_dev_run=5)
model = CNNModule(modelname, pretrained=True, num_classes=len(dm.trainset.classes))
test_eq(trainer.fit(model, dm), 1)


weight_path = 'FractalDB-1000_resnet50_epoch90.pth'
modelname = 'resnet50'
logger = TensorBoardLogger('tb_logs', name=modelname)
trainer = pl.Trainer(gpus=1, checkpoint_callback=False, logger=logger, fast_dev_run=5)
model = CNNModule(modelname, pretrained=True, num_classes=len(dm.trainset.classes), weight_path=weight_path)
test_eq(trainer.fit(model, dm), 1)




