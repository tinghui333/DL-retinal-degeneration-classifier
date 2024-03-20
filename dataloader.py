import torch
import os
import pandas as pd
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms

def deserilizer(batch_size: int, augmentation: bool=False, input_shape=(225, 225),
                remove_same_case: bool=True, num_workers: int=0, verbose: bool=False):

    data_transforms = get_data_transforms(input_shape=input_shape)
    if not augmentation:
        data_transforms['train'] = data_transforms['val']

    df = data_collector(data_dir='./data/OCT-Tiff', remove_same_case=remove_same_case)

    train_df, valid_df, test_df = data_split(df, verbose=verbose)

    train_loader = torch.utils.data.DataLoader(OCT(df=train_df, transform=data_transforms['train']), shuffle=True, batch_size=batch_size, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(OCT(df=valid_df, transform=data_transforms['val']), shuffle=False, batch_size=batch_size, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(OCT(df=test_df, transform=data_transforms['val']), shuffle=False, batch_size=batch_size, num_workers=num_workers)

    return train_loader, valid_loader, test_loader


def data_collector(data_dir: str="./data/OCT-Tiff", remove_same_case=True):
    '''
    data_collector: find all the .TIFF images and summarize in a pd.DataFrame
    '''
    total_img_list = glob(os.path.join(data_dir, '*', '*', '*.TIFF'))
    date_list = [item.split('/')[3] for item in total_img_list]
    case_list = [item.split('/')[4] for item in total_img_list]
    id_list = [item.split('/')[4].split('_')[0] for item in total_img_list]
    df = pd.DataFrame(zip(total_img_list, date_list, case_list, id_list), 
                      columns=['path', 'date', 'case', 'id'])
    

    if remove_same_case:
        remove_list = []
        for date in set(date_list):
            for id in set(id_list):
                sub_df = df[(df['date'] == date) & (df['id'] == id)]
                total_case = list(set(list(sub_df['case'])))
                left = [item for item in total_case if item.split('_')[1] == 'OD']
                right = [item for item in total_case if item.split('_')[1] == 'OS']

                if len(left) > 1:
                    remove_list += left[1:]
                if len(right) > 1:
                    remove_list += right[1:]

        df = df[~df['case'].isin(remove_list)].reset_index()

    return df


def data_split(df: pd.DataFrame, random_seed: int=0, verbose: bool=True):
    '''
    data_split: split the data into train, valid, test set by id
    '''

    train_list, test_list = train_test_split(sorted(df['id'].unique()), test_size=0.2, random_state=random_seed)
    train_list, valid_list = train_test_split(train_list, test_size=0.25, random_state=random_seed)
    train_df = df[df['id'].isin(train_list)].reset_index()
    valid_df = df[df['id'].isin(valid_list)].reset_index()
    test_df = df[df['id'] .isin(test_list)].reset_index()

    if verbose:
        for datatype, df in zip(('train', 'valid', 'test'), (train_df, valid_df, test_df)):
            print(f"Get {datatype} list: {df['id'].unique()} \n" + 
                  f"with a total of {len(df['id'].unique())} ids, " + 
                  f"{len(df['case'].unique())} cases, and {len(df)} images")

    return train_df, valid_df, test_df


class OCT(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        """
        """
        self.df = df
        self.transform = transform
        self.class2index = {
            'P20_11_7_2023': 0,
            'P40_11_24_2023': 1, 
            'P60_12_13_2023': 2,
            'P90_1_10_2024': 3,
            'P120_2_8-2024': 4
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = Image.open(self.df.iloc[idx]['path'])

        if self.transform is not None:
            img = self.transform(img)
        else:
            return transforms.ToTensor()(img)
        lbl = self.class2index[self.df.iloc[idx]['date']]
        lbl = torch.tensor(lbl)
        return img, lbl


def get_data_transforms(input_shape: tuple[int, int]=(225, 225)):
    """
    data transforms: TBD
    """
    return {
        "train":
        transforms.Compose(transforms=[
            transforms.Resize(input_shape),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ]),
        "val":
        transforms.Compose(transforms=[
            transforms.Resize((1000, 1000)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
    }

