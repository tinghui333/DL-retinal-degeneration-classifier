import os
from glob import glob
import pandas as pd
import torch
import argparse
import numpy as np

from dataloader import OCT, get_data_transforms
from model import ResNetClassifier


def main(args):
    print(f"building dataloader ...")
    total_img_list = glob(os.path.join(args.input_dir, '*', '*', '*.TIFF'))
    total_img_list = sorted(total_img_list)
    df = pd.DataFrame(total_img_list, columns=['path'])

    data_transforms = get_data_transforms(input_shape=(args.input_size, args.input_size))
    test_loader = torch.utils.data.DataLoader(OCT(df=df, test_mode=True, transform=data_transforms['val']), shuffle=False, batch_size=args.bs)

    print(f"building model ...")
    model = ResNetClassifier(num_classes=5)
    model = model.load_from_checkpoint(os.path.join(args.model_dir, 'bestmodel.ckpt'))
    model.eval()

    predictions = []
    print(f"start testing ...")
    for X in test_loader:
        with torch.no_grad():
            outputs = model(X.cuda())
            predictions.append(outputs.cpu())

    predictions = np.concatenate(predictions)
    predictions = np.argmax(predictions, axis=1)

    df['pred'] = predictions
    df.to_csv('result.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train diverse tasks in biomedicine')
    parser.add_argument('-i', '--input-size', type=int, default=225, help="image resize shape")
    parser.add_argument('--bs', type=int, default=128, help="batch size for training")
    parser.add_argument('--model-dir', type=str, default='./results/20240321-195954', help="folder that stores the trained model")
    parser.add_argument('--input-dir', type=str, default='./data/OCT-Tiff/Grafted eyes', help="folder that stores the test data")
    parser.add_argument('-v', '--verbose', action="store_true", help="print training info")
    args = parser.parse_args()
    main(args)