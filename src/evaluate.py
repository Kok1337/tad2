import torch
import torchmetrics
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import albumentations as A

import yaml
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from prepare import FoodDataset

params = yaml.safe_load(open("params.yaml"))["evaluate"]



class LitModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(average='macro', num_classes=6)
        self.recall = torchmetrics.Recall(average='macro', num_classes=6)
        self.f1 = torchmetrics.F1Score(average='macro', num_classes=6)
        self.precision_metrics = torchmetrics.Precision(average='macro', num_classes=6)
        
    def forward(self, x):
        y_pred = self.model(x)
        return y_pred
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=0.008)
        return optimizer
    
    def get_metrics(self, y_pred, y, metrics_type="train"):
        predictions = y_pred.argmax(-1)
        y = y.type(torch.IntTensor)
        
        precision = self.precision_metrics(predictions, y)
        acc = self.accuracy(predictions, y)
        rec = self.recall(predictions, y)
        f1 = self.f1(predictions, y)

        self.log('acc/test', acc)
        self.log('rec/test', rec)
        self.log('f1/test', f1)
        self.log('prec/test', precision)

    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        y_pred = self.forward(x.float()).squeeze()
        # y_pred = torch.unsqueeze(y_pred, 0)
        loss = self.loss(y_pred, y)
        
        self.get_metrics(y_pred, y, "train")
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        y_pred = self.forward(x.float()).squeeze()
        # y_pred = torch.unsqueeze(y_pred, 0)
        loss = self.loss(y_pred, y)

        self.get_metrics(y_pred, y, "val")
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self.forward(x.float()).squeeze()
        # y_pred = torch.unsqueeze(y_pred, 0)
        loss = self.loss(y_pred, y)

        self.get_metrics(y_pred, y, "test")
        return loss

image_transformation = A.Compose([
                            A.RandomResizedCrop(256, 256),
                            ToTensorV2()
                        ])

test_dataset = FoodDataset('evaluation', image_transformation)
testloader = DataLoader(test_dataset, batch_size=32)

model = torch.load(params['model'], map_location=torch.device('cpu'))
model = LitModel(model)
trainer_model = pl.Trainer(max_epochs=20, gpus=0)
trainer_model.test(model, testloader)