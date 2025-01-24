import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import resnet18


class ResNetClassifier(pl.LightningModule):
    def __init__(self, num_classes: int=1, learning_rate=1e-3, stack=None, loss='crossentropy', data_type='OCT'):
        super().__init__()

        self.loss = loss

        # self.resnet = resnet18(pretrained=True)
        self.resnet = resnet18(pretrained=False)
        # since the input is grayscale image, only 1 channel is needed
        if stack:
            self.resnet.conv1 = nn.Conv2d(stack, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        elif data_type=='OCT':
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            pretrained_weights = resnet18(pretrained=True).conv1.weight.data
            self.resnet.conv1.weight.data = pretrained_weights.mean(dim=1, keepdim=True)

        if loss=='coor':
            self.resnet.fc = nn.Identity()
            self.type_head = nn.Linear(512, 1000 * 4) # Output for 1000 points, 4 classes (background, grafted, meaningful, NA)
            self.regression_head = nn.Linear(512, 1000)  # Output for 1000 regression scores (only meaningful regions)
        else:
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

        if loss == 'crossentropy':
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss == 'ordinalcrossentropy':
            self.loss_fn = OrdinalCrossEntropyLoss()
        elif loss == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss == 'mae':
            self.loss_fn = nn.L1Loss()
        elif loss == 'coor':
            self.loss_fn = MixedLoss()

        self.learning_rate = learning_rate

    def forward(self, x):
        if self.loss == 'coor':
            features = self.resnet(x)
            type_logits = self.type_head(features).view(-1, 1000, 4)
            degeneration_scores = self.regression_head(features)
            return type_logits, degeneration_scores
        else: 
            return self.resnet(x)
    
    def step(self, batch, kind: str, **kwargs):
        if self.loss == 'coor':
            x, y1, y2 = batch
            y_pred1, y_pred2 = self(x)
            loss = self.loss_fn(y_pred1, y_pred2, y1, y2)
        else:
            x, y = batch
            y_pred = self(x)
            loss = self.loss_fn(y_pred, y)
        self.log(f'{kind}_loss', loss, **kwargs)
        if kind == 'train':
            return loss
        elif kind == 'test':
            return y_pred

    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train', on_step=True, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.step(batch, 'val', prog_bar=True)

    def test_step(self, batch, batch_idx):
        return self.step(batch, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class MixedLoss(nn.Module):
    def __init__(self):
        super(MixedLoss, self).__init__()
        self.classification_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 1.0, 0.1]))
        self.regression_loss = nn.MSELoss()

    def forward(self, type_logits, degeneration_scores, type_labels, regression_labels):
        type_labels = type_labels.long()
        type_loss = self.classification_loss(type_logits.permute(0, 2, 1), type_labels)
        print("Type Loss:", type_loss.item())

        mask = (type_labels == 2)  # meaningful regions only
        if mask.sum() > 0: 
            regression_loss = self.regression_loss(degeneration_scores[mask], regression_labels[mask])
        else:
            regression_loss = torch.tensor(0.0, dtype=torch.float32, device=degeneration_scores.device)
        print("Regression Loss:", regression_loss.item())

        total_loss = 0.5 * type_loss + 0.5 * regression_loss
        return total_loss


class OrdinalCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(OrdinalCrossEntropyLoss, self).__init__()
        self.criterion = nn.BCELoss()

    def forward(self, outputs, targets):
        loss = 0
        for i in range(len(targets)):
            for j in range(targets[i].item() + 1):
                loss += self.criterion(torch.sigmoid(outputs[i, j]), torch.ones_like(outputs[i, j]))
            for j in range(targets[i].item() + 1, outputs.size(1)):
                loss += self.criterion(torch.sigmoid(outputs[i, j]), torch.zeros_like(outputs[i, j]))
        return loss / targets.size(0)

class ResNetFeatureExtractor(nn.Module):
    # Backbone feature extractor model
    def __init__(self, input_channels=3):
        super().__init__()
        self.resnet = resnet18(pretrained=True)
        # TODO: consider adding pretrain weights
        if input_channels != 3:
            self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            pretrained_weights = resnet18(pretrained=True).conv1.weight.data
            self.resnet.conv1.weight.data = pretrained_weights.mean(dim=1, keepdim=True)
        self.resnet.fc = nn.Identity()  # Remove the final fully connected layer

    def forward(self, x):
        return self.resnet(x)
    

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class SimCLR(pl.LightningModule):

    def __init__(self, lr=1e-3, weight_decay=1e-6, max_epochs=200):
        super().__init__()

        self.ct_extractor = ResNetFeatureExtractor(input_channels=1)
        self.histo_extractor = ResNetFeatureExtractor(input_channels=1)

        self.ct_pj = ProjectionHead(input_dim=512, hidden_dim=512, output_dim=128)
        self.histo_pj = ProjectionHead(input_dim=512, hidden_dim=512, output_dim=128)

        self.save_hyperparameters()
        # assert self.hparams.temperature > 0.0, 'The temperature must be a positive float!'

    def forward(self, ct_img, histo_img):
        ct_features = self.ct_extractor(ct_img)
        histo_features = self.histo_extractor(histo_img)
        ct_output = self.ct_pj(ct_features)
        histo_output = self.histo_pj(histo_features)
        return ct_output, histo_output

    def contrastive_loss(self, ct_proj, histo_proj, labels, margin=0.3):
        # Normalizing the output embeddings
        # TODO: not sure if this part is needed
        ct_proj = F.normalize(ct_proj, dim=1)
        histo_proj = F.normalize(histo_proj, dim=1)

        similarity = F.cosine_similarity(ct_proj, histo_proj)

        pos_mask = labels == 1
        neg_mask = labels == 0

        pos_loss = (1 - similarity[pos_mask]).pow(2).mean() if pos_mask.any() else 0
        neg_loss = (similarity[neg_mask] - margin).clamp(min=0).pow(2).mean() if neg_mask.any() else 0

        loss = pos_loss + neg_loss
        return loss
    
    def step(self, batch, kind: str, **kwargs):
        ct_img, histo_img, labels = batch
        ct_output, histo_output = self(ct_img, histo_img)
        loss = self.contrastive_loss(ct_output, histo_output, labels)

        self.log(f'{kind}_loss', loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, 'val')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                    lr=self.hparams.lr,
                                    weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=self.hparams.max_epochs,
                                                            eta_min=self.hparams.lr/50)
        return [optimizer], [lr_scheduler]


class Decoder(nn.Module):
    def __init__(self, input_shape=(224, 224), num_channels=1):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, num_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=input_shape, mode='bilinear', align_corners=False),  # Ensure output size matches input size
            nn.Sigmoid()  # Use Sigmoid to get output in the range [0, 1]
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

class AutoEncoder(pl.LightningModule):
    def __init__(self, input_shape=(224, 2224)):
        super(AutoEncoder, self).__init__()
        self.encoder = ResNetFeatureExtractor(input_channels=1)
        self.decoder = Decoder(num_channels=1, input_shape=input_shape)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), 512, 1, 1)  # Reshape to [batch_size, 512, 1, 1]
        x = self.decoder(x)
        return x
    
    def step(self, batch, kind: str, **kwargs):
        x, _ = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, x)
        self.log(f'{kind}_loss', loss, **kwargs)
        if kind == 'train':
            return loss
        elif kind == 'test':
            return y_pred

    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train', on_step=True, on_epoch=False, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.step(batch, 'val', prog_bar=True)

    def test_step(self, batch, batch_idx):
        return self.step(batch, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
