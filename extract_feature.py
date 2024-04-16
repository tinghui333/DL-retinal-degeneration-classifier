import os
from glob import glob
import pandas as pd
import torch
import argparse
import numpy as np
from tqdm import tqdm

from dataloader import OCT, get_data_transforms
from model import ResNetClassifier


def main(args):
    print(f"building dataloader ...")
    # df = pd.DataFrame(total_img_list, columns=['path'])
    input_path = args.input_df
    df = pd.read_csv(input_path)

    data_transforms = get_data_transforms(input_shape=(args.input_size, args.input_size))
    test_loader = torch.utils.data.DataLoader(OCT(df=df, test_mode=True, transform=data_transforms['val']), shuffle=False, batch_size=args.bs)

    print(f"building model ...")
    model = ResNetClassifier(num_classes=5)
    model = model.load_from_checkpoint(os.path.join(args.model_dir, 'bestmodel.ckpt'))
    model.eval()

    def get_layer_output(module, input, output):
        layer_outputs.append(output.cpu().detach().numpy())

    target_layer = model.resnet.avgpool

    hook_handle = target_layer.register_forward_hook(get_layer_output)

    layer_outputs = []

    print(f"start testing ...")
    for X in tqdm(test_loader):
        with torch.no_grad():
            outputs = model(X.cuda())

    hook_handle.remove()

    layer_outputs_np = np.concatenate(layer_outputs)
    layer_outputs_np = np.squeeze(layer_outputs_np)

    print(np.shape(layer_outputs_np))
    np.save(f'{input_path.split("_")[0]}_features.npy', layer_outputs_np)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train diverse tasks in biomedicine')
    parser.add_argument('-i', '--input-size', type=int, default=225, help="image resize shape")
    parser.add_argument('--bs', type=int, default=128, help="batch size for training")
    parser.add_argument('--model-dir', type=str, default='./results/20240321-195954', help="folder that stores the trained model")
    parser.add_argument('--input-df', type=str, default='./test_df.csv', help="csv file that stores the data list")
    parser.add_argument('-v', '--verbose', action="store_true", help="print training info")
    args = parser.parse_args()
    main(args)