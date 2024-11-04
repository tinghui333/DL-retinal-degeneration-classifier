import os
from glob import glob
import pandas as pd
import torch
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix
import tqdm

from dataloader import OCT, get_data_transforms
from model import ResNetClassifier


def main(args):
    print(f"building dataloader ...")
    if args.df:
        df = pd.read_csv(args.df)
        save_dir = os.path.dirname(args.df)
    elif args.input_dir:
        total_img_list = glob(os.path.join(args.input_dir, '*', '*.tif'))
        total_img_list = sorted(total_img_list)
        df = pd.DataFrame(total_img_list, columns=['path'])

    data_transforms = get_data_transforms(input_shape=(args.input_size, args.input_size), preprocess=args.preprocess, data_mean=[0.3193], data_std=[0.1314])

    test_loader = torch.utils.data.DataLoader(OCT(df=df, class_type=args.class_type, 
                                                  stack=args.stack, stride=args.stride,
                                                  transform=data_transforms['val']), shuffle=False, 
                                                  batch_size=args.bs)


    print(f"building {args.class_type} model ...")
    # if args.class_type == "timepoint":
    #     model = ResNetClassifier(num_classes=5, loss='crossentropy')
    # elif args.class_type == "visual":
    #     model = ResNetClassifier(stack=args.stack, loss='mse')
        # model.to(dtype=torch.float32)
    model = ResNetClassifier()
    model = model.load_from_checkpoint(os.path.join(args.model_dir, 'bestmodel.ckpt'),
                                       num_classes=1, stack=args.stack, loss='mse')
    model.eval()

    predictions = []
    groundtrues = []
    print(f"start testing ...")
    for X, y in tqdm.tqdm(test_loader):
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

    # df['pred'] = predictions
    # df.to_csv('result.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train diverse tasks in biomedicine')
    parser.add_argument('-i', '--input-size', type=int, default=225, help="image resize shape")
    parser.add_argument('--bs', type=int, default=128, help="batch size for training")
    parser.add_argument('--model-dir', type=str, default='./results/20240321-195954', help="folder that stores the trained model")
    parser.add_argument('--input-dir', type=str, default='./data/OCT-Tiff/Grafted eyes', help="folder that stores the test data")
    parser.add_argument('-t', '--class-type', type=str, default="timepoint", help="timepoint or visual")
    parser.add_argument('--df', type=str, help="input data frame")
    parser.add_argument('--stack', type=int, default=None, help="stack")
    parser.add_argument('--stride', type=int, default=1, help="stride")
    parser.add_argument('--preprocess', nargs='+', help='preprocessing list')
    parser.add_argument('-v', '--verbose', action="store_true", help="print training info")
    args = parser.parse_args()
    main(args)