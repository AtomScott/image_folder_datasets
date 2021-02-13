#hide
get_ipython().run_line_magic("load_ext", " autoreload")
get_ipython().run_line_magic("autoreload", " 2")


from image_folder_datasets.core import ImageFolderDataModule

data_dir = 'Datasets/cifar10'
dm = ImageFolderDataModule(data_dir, 128)
dm.setup()


# For ease of use, we also add a dataloader from the fastai library. This can be accessed from `dm.dls`.
# However it is not used for anything else.
dm.dls.show_batch()


import pytorch_lightning as pl
from image_folder_datasets.core import CNNModule

modelname = 'resnet50'
max_epochs = 50

logger = pl.loggers.TensorBoardLogger('tb_logs', name=modelname)
trainer = pl.Trainer(gpus=1, max_epochs=max_epochs, checkpoint_callback=False, logger=logger)
model = CNNModule(modelname, pretrained=False, freeze_extractor=False, num_classes=len(dm.trainset.classes))
trainer.fit(model, dm);


logger = pl.loggers.TensorBoardLogger('tb_logs', name=modelname+'_imagenet')
trainer = pl.Trainer(gpus=1, max_epochs=max_epochs, checkpoint_callback=False, logger=logger)

model = CNNModule(modelname, pretrained=True, freeze_extractor=True, num_classes=len(dm.trainset.classes))
trainer.fit(model, dm);


logger = pl.loggers.TensorBoardLogger('tb_logs', name=modelname+'_fractalDB')

weight_path = 'FractalDB-1000_resnet50_epoch90.pth'
trainer = pl.Trainer(gpus=1, max_epochs=max_epochs, checkpoint_callback=False, logger=logger)

model = CNNModule(modelname, pretrained=False, freeze_extractor=True, num_classes=len(dm.trainset.classes), weight_path=weight_path)
trainer.fit(model, dm);


from torchvision.transforms import ToTensor, Resize, Compose, CenterCrop, Normalize

transform = Compose([
        Resize(256, interpolation=2),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


dm = ImageFolderDataModule(data_dir, 128, transform)
dm.setup()

logger = pl.loggers.TensorBoardLogger('tb_logs', name=modelname+'_fractalDB_imagenet_nm')

weight_path = 'FractalDB-1000_resnet50_epoch90.pth'
trainer = pl.Trainer(gpus=1, max_epochs=max_epochs, checkpoint_callback=False, logger=logger)

model = CNNModule(modelname, freeze_extractor=False, num_classes=len(dm.trainset.classes), weight_path=weight_path)
trainer.fit(model, dm);


## DO NOT FREEZE EXTRACTOR
from torchvision.transforms import ToTensor, Resize, Compose, CenterCrop, Normalize

transform = Compose([
        Resize(256, interpolation=2),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.2, 0.2, 0.2],
                         std=[0.5, 0.5, 0.5])
])


dm = ImageFolderDataModule(data_dir, 128, transform)
dm.setup()

logger = pl.loggers.TensorBoardLogger('tb_logs', name=modelname+'_fractalDB')

weight_path = 'FractalDB-1000_resnet50_epoch90.pth'
trainer = pl.Trainer(gpus=1, max_epochs=max_epochs, checkpoint_callback=False, logger=logger)

model = CNNModule(modelname, freeze_extractor=False, num_classes=len(dm.trainset.classes), weight_path=weight_path)
trainer.fit(model, dm);



