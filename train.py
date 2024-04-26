import os
import torch
import argparse
import numpy as np
import time
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix

from dataloader import deserilizer as dataloader
from model import ResNetClassifier


def main(args):
    save_dir = os.path.join(args.dir, f"{time.strftime('%Y%m%d-%H%M%S')}")
    logger = TensorBoardLogger(save_dir=save_dir)
    logger.log_hyperparams(args)

    print(f"building dataloader ...")
    train_loader, valid_loader, test_loader = dataloader(batch_size=args.bs, 
                                                         augmentation=True,
                                                         input_shape=(args.input_size, args.input_size), 
                                                         save_dir=save_dir if args.verbose else None,
                                                         class_type=args.class_type,
                                                         stack=args.stack, stride=args.stride,
                                                         preprocess=args.preprocess,
                                                         verbose=args.verbose)

    print(f"building model ...")
    if args.class_type == "timepoint":
        model = ResNetClassifier(num_classes=5)
    elif args.class_type == "visual":
        model = ResNetClassifier(stack=args.stack)

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

    print(f"start testing")
    model = model.load_from_checkpoint(os.path.join(save_dir, 'bestmodel.ckpt'), stack=args.stack)
    model.eval()
    predictions = []
    groundtrues = []
    for X, y in test_loader:
        with torch.no_grad():
            outputs = model(X.cuda())
            predictions.append(outputs.cpu())
            groundtrues.append(y)

    predictions = np.concatenate(predictions)
    groundtrues = np.concatenate(groundtrues)
    np.save(os.path.join(save_dir, 'pred.npy'), predictions)
    np.save(os.path.join(save_dir, 'gt.npy'), groundtrues)

    if args.class_type == "timepoint":
        predictions = np.argmax(predictions, axis=1)
        conf_matrix = confusion_matrix(groundtrues, predictions)
        print(f"Confusion Matrix:\n {conf_matrix}")
    elif args.class_type == "visual":
        mse_loss = np.mean((groundtrues - predictions) ** 2)
        mae_loss = np.mean(np.abs(groundtrues - predictions))
        print(f"MSE: {mse_loss} | MAE: {mae_loss}")
        if save_dir:
            with open(os.path.join(save_dir, 'result.txt'), 'w') as f:
                f.write(f"MSE: {mse_loss} | MAE: {mae_loss}")

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
    parser.add_argument('-v', '--verbose', action="store_true", help="print training info")
    args = parser.parse_args()
    main(args)