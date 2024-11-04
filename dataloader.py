import torch
import os
import pandas as pd
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2
from tqdm import tqdm

def deserilizer(batch_size: int, augmentation: bool=False, input_shape=(225, 225),
                data_type: str='OCT', remove_same_case: bool=True, num_workers: int=4, 
                save_dir: str=None, class_type: str="timepoint", 
                stack: int=None, stride: int=None,
                preprocess: list=None,
                verbose: bool=False):

    if data_type == 'OCT':
        df = data_collector(data_dir='./data/OCT-Tiff', remove_same_case=remove_same_case, class_type=class_type,
                            stack=stack, stride=stride)
    elif data_type == 'hist':
        df = data_collector_hist(data_dir="./data/Histology_Images_CV/patch_images")

    train_df, valid_df, test_df = data_split(df, verbose=verbose)
    if save_dir:
        train_df.to_csv(os.path.join(save_dir, 'train_df.csv'), index=False)
        valid_df.to_csv(os.path.join(save_dir, 'valid_df.csv'), index=False)
        test_df.to_csv(os.path.join(save_dir, 'test_df.csv'), index=False)

    # data_mean, data_std = online_mean_and_sd(torch.utils.data.DataLoader(OCT(df=pd.concat([train_df, valid_df]), transform=None), shuffle=False, batch_size=128))
    # data_mean, data_std = [0.3193], [0.1314]
    data_mean, data_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    # print(f"Getting mean {data_mean} and std {data_std}")

    data_transforms = get_data_transforms(input_shape=input_shape, preprocess=preprocess, data_mean=data_mean, data_std=data_std)
    print(data_transforms)
    if save_dir:
        with open(os.path.join(save_dir, 'preprocessing.txt'), 'w') as f:
            f.write(repr(data_transforms))
    if not augmentation:
        data_transforms['train'] = data_transforms['val']

    if data_type == 'OCT':
        train_loader = torch.utils.data.DataLoader(OCT(df=train_df, class_type=class_type, 
                                                    stack=stack, stride=stride,
                                                    transform=data_transforms['train']), shuffle=True, 
                                                    batch_size=batch_size, num_workers=num_workers,
                                                    pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(OCT(df=valid_df, class_type=class_type, 
                                                    stack=stack, stride=stride,
                                                    transform=data_transforms['val']), shuffle=False, 
                                                    batch_size=batch_size, num_workers=num_workers,
                                                    pin_memory=True)
        test_loader = torch.utils.data.DataLoader(OCT(df=test_df, class_type=class_type, 
                                                    stack=stack, stride=stride,
                                                    transform=data_transforms['val']), shuffle=False, 
                                                    batch_size=batch_size, num_workers=num_workers,
                                                    pin_memory=True)
    elif data_type == 'hist':
        train_loader = torch.utils.data.DataLoader(Hist(df=train_df, 
                                                    transform=data_transforms['train']), shuffle=True, 
                                                    batch_size=batch_size, num_workers=num_workers,
                                                    pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(Hist(df=valid_df, 
                                                    transform=data_transforms['val']), shuffle=False, 
                                                    batch_size=batch_size, num_workers=num_workers,
                                                    pin_memory=True)
        test_loader = torch.utils.data.DataLoader(Hist(df=test_df, 
                                                    transform=data_transforms['val']), shuffle=False, 
                                                    batch_size=batch_size, num_workers=num_workers,
                                                    pin_memory=True)

    return train_loader, valid_loader, test_loader


def data_collector(data_dir: str="./data/OCT-Tiff", remove_same_case: bool=True, 
                   class_type: str="timepoint", stack: int=None, stride: int=None):
    '''
    data_collector: find all the .TIFF images and summarize in a pd.DataFrame
    '''
    from utils import gen_position, extract_side, gen_next_date, remove_tail

    total_img_list = glob(os.path.join(data_dir, '*', '*', '*.TIFF'))
    total_img_list = sorted(total_img_list)
    date_list = [item.split('/')[3].split('_')[0] for item in total_img_list]
    case_list = [item.split('/')[4] for item in total_img_list]
    id_list = [item.split('/')[4].split('_')[0] for item in total_img_list]
    df = pd.DataFrame(zip(total_img_list, date_list, case_list, id_list), 
                      columns=['path', 'date', 'case', 'ID'])
    df['pos'] = df['path'].apply(gen_position)
    
    if remove_same_case:
        remove_list = []
        for date in set(date_list):
            for id in set(id_list):
                sub_df = df[(df['date'] == date) & (df['ID'] == id)]
                total_case = list(set(list(sub_df['case'])))
                left = [item for item in total_case if item.split('_')[1] == 'OD']
                right = [item for item in total_case if item.split('_')[1] == 'OS']

                if len(left) > 1:
                    remove_list += left[1:]
                if len(right) > 1:
                    remove_list += right[1:]

        df = df[~df['case'].isin(remove_list)].reset_index(drop=True)

    if stack:
        remove_n = (stack - 1) * stride
        df = df.groupby(['case']).apply(remove_tail, count=remove_n)

    if class_type in ("timepoint", "autoencoder"):
        # df = df[df['date'] != 'P150']
        # df = df[df['date'] != 'P158']
        return df
    
    elif class_type == "visual":

        df['side'] = df['case'].apply(extract_side)
        df['date'] = df['date'].apply(gen_next_date)

        okr = pd.read_csv('./AI_OKR_clean.csv')
        merged_df = pd.merge(df, okr, on=['date', 'ID', 'side'], how='inner')
        return merged_df
    
    
def data_collector_hist(data_dir: str="./data/Histology_Images_CV/patch_images"):
    '''
    data_collector_hist: find all the .png images and summarize in a pd.DataFrame
    '''

    total_img_list = glob(os.path.join(data_dir, '*', 'AI*.png'))
    total_img_list = sorted(total_img_list)
    date_list = [item.split('/')[4].split('_')[-1] for item in total_img_list]
    case_list = [item.split('/')[-1].split('_')[0] for item in total_img_list]
    id_list = [item[:-1] for item in case_list]
    side_list = [item[-1] for item in case_list]
    df = pd.DataFrame(zip(total_img_list, date_list, case_list, id_list, side_list), 
                      columns=['path', 'date', 'case', 'ID', 'side'])
    return df


def data_split(df: pd.DataFrame, random_seed: int=0, verbose: bool=True):
    '''
    data_split: split the data into train, valid, test set by id
    '''

    train_list, test_list = train_test_split(sorted(df['ID'].unique()), test_size=0.2, random_state=random_seed)
    train_list, valid_list = train_test_split(train_list, test_size=0.25, random_state=random_seed)
    train_df = df[df['ID'].isin(train_list)].reset_index(drop=True)
    valid_df = df[df['ID'].isin(valid_list)].reset_index(drop=True)
    test_df = df[df['ID'] .isin(test_list)].reset_index(drop=True)

    if verbose:
        for datatype, df in zip(('train', 'valid', 'test'), (train_df, valid_df, test_df)):
            print(f"Get {datatype} list: {df['ID'].unique()} \n" + 
                  f"with a total of {len(df['ID'].unique())} ids, " + 
                  f"{len(df['case'].unique())} cases, and {len(df)} images")

    return train_df, valid_df, test_df


class OCT(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, test_mode: bool=False, class_type: str="timepoint", transform=None,
                 stack: int=None, stride: int=None):
        """
        df: pd.Dataframe contains columns
            'path': image path
            'date': to transfer to label
        """
        self.df = df
        self.transform = transform
        self.class2index = {
            'P20': 0,
            'P40': 1, 
            'P60': 2,
            'P90': 3,
            'P120': 4,
            'P150': 4,
            'P158': 4
        }
        self.test = test_mode
        self.class_type = class_type
        self.stack = stack
        self.stride = stride
        self.basic_transform = v2.Compose(transforms=[
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Lambda(select_first_channel),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = Image.open(self.df.iloc[idx]['path'])
        img = self.basic_transform(img)

        if self.stack:
            init_pos = self.df.iloc[idx]['pos']
            init_path = self.df.iloc[idx]['path']
            stack_img = [img.squeeze()]
            for i in range(1, self.stack):
                curr_pos = init_pos + i * self.stride
                curr_path = init_path.replace(f'ture_{init_pos:04d}', f'ture_{curr_pos:04d}')
                curr_img = Image.open(curr_path)
                curr_img = self.basic_transform(curr_img)
                stack_img.append(curr_img.squeeze())

            img = torch.stack(stack_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.test:
            return img
        elif self.class_type in ("timepoint", "autoencoder"):
            lbl = self.class2index[self.df.iloc[idx]['date']]
            lbl = torch.tensor(lbl)
            return img, lbl
        elif self.class_type == "visual":
            lbl = self.df.iloc[idx]['value']
            lbl = torch.tensor(lbl).float()
            return img, lbl.unsqueeze(0)


class Hist(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, test_mode: bool=False, transform=None):
        """
        df: pd.Dataframe contains columns
            'path': image path
            'date': to transfer to label
        """
        self.df = df
        self.transform = transform
        self.class2index = {
            'P20': 0,
            'P40': 1, 
            'P60': 2,
            'P90': 3,
            'P120': 4,
            'P150': 4,
            'P158': 4
        }
        self.test = test_mode
        # self.basic_transform = v2.Compose(transforms=[
        #     v2.PILToTensor(),
        #     v2.ToDtype(torch.float32, scale=True),
        #     ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = Image.open(self.df.iloc[idx]['path']).convert("RGB")
        # img = self.basic_transform(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.test:
            return img
        else:
            lbl = self.class2index[self.df.iloc[idx]['date']]
            lbl = torch.tensor(lbl)
            return img, lbl


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        """
        df: pd.Dataframe contains columns
            'path': image path
            'date': to transfer to label
        """
        self.df = df
        self.transform = transform
        self.basic_transform = v2.Compose(transforms=[
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Lambda(select_first_channel),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img1 = Image.open(self.df.iloc[idx]['path'])
        img1 = self.basic_transform(img1)
        img2 = Image.open(self.df.iloc[idx]['path2'])
        img2 = self.basic_transform(img2)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        lbl = self.df.iloc[idx]['same_date']
        lbl = torch.tensor(lbl)
        return img1, img2, lbl
        

def get_data_transforms(input_shape: tuple[int, int]=(225, 225), preprocess: list=None, data_mean=None, data_std=None):
    """
    data transforms: TBD
    """
    train_list = []
    valid_list = []

    transfer_dict = {
        'crop': v2.RandomResizedCrop(size=input_shape, scale=(0.8, 1.0), antialias=True),
        'resize': v2.Resize(input_shape, antialias=True),
        'gaussian': v2.GaussianBlur(kernel_size=11),
        'rotate': v2.RandomRotation(degrees=5),
        'hor_flip': v2.RandomHorizontalFlip(),
        'ver_flip': v2.RandomVerticalFlip(),
        'color': v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        'normalize': v2.Normalize(mean=data_mean, std=data_std)
    }

    if preprocess is None:
        preprocess = ['resize']
    elif "all" in preprocess:
        index = preprocess.index('all')
        preprocess.pop(index)
        preprocess.append('rotate')
        preprocess.append('flip')
        preprocess.append('normalize')


    for step in preprocess:
        if step == 'flip':
            train_list.append(transfer_dict['hor_flip'])
            train_list.append(transfer_dict['ver_flip'])
        elif step == 'normalize':
            train_list.append(v2.ToTensor())
            train_list.append(transfer_dict[step])
        else:
            train_list.append(transfer_dict[step])
        if step == 'crop':
            valid_list.append(transfer_dict['resize'])
        elif step == 'normalize':
            valid_list.append(v2.ToTensor())
            valid_list.append(transfer_dict[step])
        elif step in ['gaussian', 'resize']:
            valid_list.append(transfer_dict[step])

    return {
        "train": v2.Compose(transforms=train_list),
        "val": v2.Compose(transforms=valid_list)
    }


def select_first_channel(img):
    return img[0, :, :].unsqueeze(0)


def online_mean_and_sd(
    loader: torch.utils.data.DataLoader
                        ) -> tuple[list[float], list[float]]:
    """
    From deep slide
    Computes the mean and standard deviation online.
        Var[x] = E[X^2] - (E[X])^2
    Args:
        loader: The PyTorch DataLoader containing the images to iterate over.
    Returns:
        A tuple containing the mean and standard deviation for the images
        over the channel, height, and width axes.
    """
    cnt = 0
    fst_moment = torch.empty(1)
    snd_moment = torch.empty(1)

    for data in tqdm(loader):
        b, __, h, w = data.shape
        nb_pixels = b * h * w
        fst_moment = (cnt * fst_moment +
                        torch.sum(data, dim=[0, 2, 3])) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + torch.sum(
            data**2, dim=[0, 2, 3])) / (cnt + nb_pixels)
        cnt += nb_pixels
    return fst_moment.tolist(), torch.sqrt(snd_moment -
                                            fst_moment**2).tolist()