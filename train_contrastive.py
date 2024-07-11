import os
import torch
import argparse
import numpy as np
import time
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix

from dataloader import deserilizer as dataloader
from dataloader import get_data_transforms, PairDataset
from model import ResNetClassifier, SimCLR

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def main(args):
    save_dir = os.path.join(args.dir, f"{time.strftime('%Y%m%d-%H%M%S')}")
    logger = TensorBoardLogger(save_dir=save_dir)
    logger.log_hyperparams(args)

    print(f"building dataloader ...")
    
    data_dir = './contrastive_learning'
    train_df = pd.read_csv(os.path.join(data_dir, 'train_df.csv'))
    valid_df = pd.read_csv(os.path.join(data_dir, 'valid_df.csv'))

    data_transforms = get_data_transforms(input_shape=(args.input_size, args.input_size), 
                                          preprocess=args.preprocess, data_mean=[0.3193], data_std=[0.1314])
    

    train_loader = torch.utils.data.DataLoader(PairDataset(df=train_df, 
                                                           transform=data_transforms['train']), shuffle=True, 
                                                           batch_size=args.bs, num_workers=4, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(PairDataset(df=valid_df, 
                                                           transform=data_transforms['val']), shuffle=False, 
                                                           batch_size=args.bs, num_workers=4, pin_memory=True)



    print(f"building model ...")
    model = SimCLR(max_epochs=args.epoch)

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, 
        patience=args.patience, verbose=args.verbose, mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        every_n_train_steps=args.valid_steps,
        monitor="val_loss",
        save_top_k=3,
        filename="bestmodel",
        mode="min",
        verbose=True
    )

    callbacks = [checkpoint_callback, early_stop_callback]

    print(f"building trainer ...")
    trainer = Trainer(
        accelerator='cuda',
        max_epochs=args.epoch,
        enable_progress_bar=True,
        log_every_n_steps=10, 
        val_check_interval=args.valid_steps,
        callbacks=callbacks,
        logger=logger,
    )

    print(f"start training ...")
    trainer.fit(model, train_loader, valid_loader)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train diverse tasks in biomedicine')
    parser.add_argument('-i', '--input-size', type=int, default=225, help="image resize shape")
    parser.add_argument('--bs', type=int, default=128, help="batch size for training")
    parser.add_argument('-d', '--dir', type=str, default="results", help="folder to store training log")
    parser.add_argument('-e', '--epoch', type=int, default=10, help="train epoch")
    parser.add_argument('--valid-steps', type=int, default=50, help="number of train steps for running validation")
    parser.add_argument('-p', '--patience', type=int, default=10, help="patience for early stopping")
    parser.add_argument('-t', '--class-type', type=str, default="timepoint", help="timepoint or visual")
    parser.add_argument('--stack', type=int, default=None, help="stack")
    parser.add_argument('--stride', type=int, default=1, help="stride")
    parser.add_argument('--preprocess', nargs='+', help='preprocessing list')
    parser.add_argument('-l', '--loss', type=str, default=None, help='loss function')
    parser.add_argument('-v', '--verbose', action="store_true", help="print training info")
    args = parser.parse_args()
    main(args)