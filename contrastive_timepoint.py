import os
import pandas as pd
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from dataloader import get_data_transforms, PairDataset
from model import SimCLR


save_dir = './results/20240626-210615'
model = SimCLR.load_from_checkpoint(os.path.join(save_dir, 'bestmodel.ckpt'))
bs = 128

data_dir = './contrastive_learning'
train_df = pd.read_csv(os.path.join(data_dir, 'train_df.csv'))
valid_df = pd.read_csv(os.path.join(data_dir, 'valid_df.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'test_df.csv'))

data_transforms = get_data_transforms(input_shape=(225, 225), 
                                      preprocess=['crop'], data_mean=[0.3193], data_std=[0.1314])


train_loader = torch.utils.data.DataLoader(PairDataset(df=train_df, 
                                                       transform=data_transforms['train']), shuffle=False, 
                                                       batch_size=bs, num_workers=4, pin_memory=True)
valid_loader = torch.utils.data.DataLoader(PairDataset(df=valid_df, 
                                                       transform=data_transforms['val']), shuffle=False, 
                                                       batch_size=bs, num_workers=4, pin_memory=True)
test_loader = torch.utils.data.DataLoader(PairDataset(df=test_df, 
                                                       transform=data_transforms['val']), shuffle=False, 
                                                       batch_size=bs, num_workers=4, pin_memory=True)


model.eval()
feature1s = []
feature2s = []
for X, _, _ in tqdm.tqdm(train_loader):
    with torch.no_grad():
        feature1 = model.ct_extractor(X.cuda())
        feature2 = model.histo_extractor(X.cuda())
        feature1s.append(feature1.cpu())
        feature2s.append(feature2.cpu())

for X, _, _ in tqdm.tqdm(valid_loader):
    with torch.no_grad():
        feature1 = model.ct_extractor(X.cuda())
        feature2 = model.histo_extractor(X.cuda())
        feature1s.append(feature1.cpu())
        feature2s.append(feature2.cpu())

test_df['path2'] = './data/OCT-Tiff/P120_2_8-2024/AI006_OD_V_3x3_0_0000090   2-8-2024/AI006_OD_V_3x3_0_0000090_Structure_0001.TIFF'
test_df['same_date'] = 1
test_loader = torch.utils.data.DataLoader(PairDataset(df=test_df, 
                                                       transform=data_transforms['val']), shuffle=False, 
                                                       batch_size=bs, num_workers=4, pin_memory=True)
for X, _, _ in tqdm.tqdm(test_loader):
    with torch.no_grad():
        feature1 = model.ct_extractor(X.cuda())
        feature2 = model.histo_extractor(X.cuda())
        feature1s.append(feature1.cpu())
        feature2s.append(feature2.cpu())

feature1s = np.concatenate(feature1s)

train_feature = feature1s[:len(train_df)]
valid_feature = feature1s[len(train_df):(len(train_df)+len(valid_df))]
test_feature = feature1s[(len(train_df)+len(valid_df)):]
print(len(train_feature), len(valid_feature), len(test_feature))

class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features, df):
        self.features = features
        self.df = df
        self.class2index = {
            'P20': 0,
            'P40': 1, 
            'P60': 2,
            'P90': 3,
            'P120': 4,
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx])
        lbl = self.class2index[self.df.iloc[idx]['date']]
        lbl = torch.tensor(lbl)
        return feature, lbl
    
feature_train_loader = torch.utils.data.DataLoader(FeatureDataset(features=train_feature, df=train_df), shuffle=True, batch_size=bs)
feature_valid_loader = torch.utils.data.DataLoader(FeatureDataset(features=valid_feature, df=valid_df), shuffle=False, batch_size=bs)
feature_test_loader = torch.utils.data.DataLoader(FeatureDataset(features=test_feature, df=test_df), shuffle=False, batch_size=bs)

class ClassificationModel(pl.LightningModule):
    def __init__(self, input_dim, num_classes):
        super(ClassificationModel, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)  

    def forward(self, x):
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
input_dim = 512
num_classes = 5
classification_model = ClassificationModel(input_dim, num_classes)

trainer = pl.Trainer(max_epochs=50)
trainer.fit(classification_model, feature_train_loader, feature_valid_loader)