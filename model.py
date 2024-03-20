import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import resnet18


class ResNetClassifier(pl.LightningModule):
    def __init__(self, num_classes=5, learning_rate=1e-3):
        super().__init__()
        self.resnet = resnet18(weights=None)

        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, num_classes)
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.resnet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True) 
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        self.log('val_loss', loss, prog_bar=True) 

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        self.log('test_loss', loss)
        return y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer