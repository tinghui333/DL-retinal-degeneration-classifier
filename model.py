import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import resnet18


class ResNetClassifier(pl.LightningModule):
    def __init__(self, num_classes: int=1, learning_rate=1e-3, stack=None):
        super().__init__()
        self.resnet = resnet18(weights=None)
        # since the input is grayscale image, only 1 channel is needed
        if stack:
            self.resnet.conv1 = torch.nn.Conv2d(stack, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        else:
            self.resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, num_classes)
        if num_classes == 1:
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.learning_rate = learning_rate

    def forward(self, x):
        return self.resnet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True) 
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log('val_loss', loss, prog_bar=True) 

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log('test_loss', loss)
        return y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    