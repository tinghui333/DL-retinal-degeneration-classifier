import os
import pandas as pd
from glob import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2

def deserilizer(batch_size: int, augmentation: bool=False, input_shape=(1000, 1000),
                data_type: str='OCT', remove_same_case: bool=True, num_workers: int=4, 
                save_dir: str=None, class_type: str="timepoint", 
                stack: int=None, stride: int=None,
                preprocess: list=None,
                verbose: bool=False):

    if class_type == 'coor':
        global type_lbl_dict
        global reg_lbl_dict
        df, type_lbl_dict, reg_lbl_dict = gen_x_coor_lbl()

    elif data_type == 'OCT':
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
    data_mean, data_std = [0.3193], [0.1314]
    # data_mean, data_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    # print(f"Getting mean {data_mean} and std {data_std}")

    data_transforms = get_data_transforms(input_shape=input_shape, preprocess=preprocess, data_mean=data_mean, data_std=data_std)
    if save_dir:
        with open(os.path.join(save_dir, 'preprocessing.txt'), 'w') as f:
            f.write(repr(data_transforms))
    if not augmentation:
        data_transforms['train'] = data_transforms['val']

    if data_type == 'OCT':
        train_loader = torch.utils.data.DataLoader(OCT(df=train_df, class_type=class_type, 
                                                    stack=stack, stride=stride,
                                                    transform=data_transforms['train'], data_type='train'), shuffle=True, 
                                                    batch_size=batch_size, num_workers=num_workers,
                                                    pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(OCT(df=valid_df, class_type=class_type, 
                                                    stack=stack, stride=stride,
                                                    transform=data_transforms['val'], data_type='valid'), shuffle=False, 
                                                    batch_size=batch_size, num_workers=num_workers,
                                                    pin_memory=True)
        test_loader = torch.utils.data.DataLoader(OCT(df=test_df, class_type=class_type, 
                                                    stack=stack, stride=stride,
                                                    transform=data_transforms['val'], data_type='test'), shuffle=False, 
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


def gen_x_coor_lbl(lbl_path: str='./data/lbl_x_coor.csv'):
    '''
    generate 2 1000x array for label
    '''

    type_transfer_dict = {
        'G': 1,
        'g': 1,
        'N': 3,
        'n': 3,
        'na': 3,
        '20': 2,
        '40': 2,
        '60': 2,
        '90': 2
    }

    reg_transfer_dict = {
        '20': 1, 
        '40': 2,
        '60': 3,
        '90': 4
    }

    df = pd.read_csv(lbl_path)

    type_lbl_dict = defaultdict()
    reg_lbl_dict = defaultdict()

    for image_path in set(df['path']):
        assert os.path.isfile(image_path), f"{image_path} doesn't exist"

        curr_df = df[df['path'] == image_path]
        type_lbl = np.zeros((1000, ))
        reg_lbl = np.zeros((1000, ))

        type_lbl[curr_df['x1'].min(): curr_df['x2'].max()] = 3
        for _, row in curr_df.iterrows():
            if not pd.isna(row['label']):
                type_lbl[row['x1']:row['x2']] = type_transfer_dict[row['label']]
            if row['label'] in reg_transfer_dict:
                reg_lbl[row['x1']:row['x2']] = reg_transfer_dict[row['label']]

        type_lbl_dict[image_path] = type_lbl
        reg_lbl_dict[image_path] = reg_lbl

    df_processed = df.drop(columns=['label', 'x1', 'x2']).drop_duplicates().reset_index()

    return df_processed, type_lbl_dict, reg_lbl_dict


class OCT(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, test_mode: bool=False, class_type: str="timepoint", transform=None,
                 stack: int=None, stride: int=None, data_type: str='test'):
        """
        df: pd.Dataframe contains columns
            'path': image path
            'date': to transfer to label
        """
        self.df = df
        self.transform = transform
        print(self.transform)
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
        self.data_type = data_type

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

        if self.transform is not None and self.class_type != 'coor':
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
        elif self.class_type == 'coor':
            lbl_type = torch.tensor(type_lbl_dict[self.df.iloc[idx]['path']])
            lbl_reg = torch.tensor(reg_lbl_dict[self.df.iloc[idx]['path']]).float()
            if self.data_type == 'train':
                img, lbl_type, lbl_reg = augment(img, lbl_type, lbl_reg)
            return img, lbl_type, lbl_reg


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


def horizontal_flip(image, label1, label2):
    flipped_image = torch.flip(image, dims=[2]) 
    flipped_label1 = torch.flip(label1, dims=[0])
    flipped_label2 = torch.flip(label2, dims=[0])
    return flipped_image, flipped_label1, flipped_label2


def rotate(image, type_label, reg_label):
    degrees = torch.randint(-10, 11, (1,)).item()
    h, w = image.shape[1], image.shape[2]
    center = (w // 2, h // 2)

    image = v2.functional.rotate(image, angle=degrees, expand=False, interpolation=v2.InterpolationMode.BILINEAR)

    x_indices = torch.arange(0, w)
    radians = torch.deg2rad(torch.tensor(degrees, dtype=torch.float32))
    x_shift = ((x_indices - center[0]) * torch.cos(radians) - 
                (h // 2) * torch.sin(radians)).round().long() + center[0]
    x_shift = torch.clamp(x_shift, 0, w - 1) 

    type_label = type_label[x_shift]
    reg_label = reg_label[x_shift]
    return image, type_label, reg_label


def shift(image, type_label, reg_label, max_shift=200):
    """
    Perform random shifting of the image in both horizontal (x) and vertical (y) directions,
    and adjust labels accordingly.

    Args:
        image (torch.Tensor): Input image, shape (C, H, W).
        type_label (torch.Tensor): Type label, shape (W,).
        reg_label (torch.Tensor): Regression label, shape (W,).
        max_shift (int): Maximum number of pixels to shift in any direction.

    Returns:
        torch.Tensor, torch.Tensor, torch.Tensor: Shifted image and labels.
    """
    _, height, width = image.shape

    shift_x = torch.randint(-max_shift, max_shift + 1, (1,)).item()
    shift_y = torch.randint(-max_shift, max_shift + 1, (1,)).item()

    shifted_image = torch.zeros_like(image)

    if shift_x > 0:
        x_start_src, x_end_src = 0, width - shift_x
        x_start_dst, x_end_dst = shift_x, width
    else:
        x_start_src, x_end_src = -shift_x, width
        x_start_dst, x_end_dst = 0, width + shift_x

    if shift_y > 0:
        y_start_src, y_end_src = 0, height - shift_y
        y_start_dst, y_end_dst = shift_y, height
    else:
        y_start_src, y_end_src = -shift_y, height
        y_start_dst, y_end_dst = 0, height + shift_y

    shifted_image[:, y_start_dst:y_end_dst, x_start_dst:x_end_dst] = \
        image[:, y_start_src:y_end_src, x_start_src:x_end_src]

    shifted_type_label = torch.zeros_like(type_label)
    shifted_reg_label = torch.zeros_like(reg_label)

    if shift_x > 0:
        shifted_type_label[shift_x:] = type_label[:-shift_x]
        shifted_reg_label[shift_x:] = reg_label[:-shift_x]
    elif shift_x < 0:
        shifted_type_label[:shift_x] = type_label[-shift_x:]
        shifted_reg_label[:shift_x] = reg_label[-shift_x:]
    else:
        shifted_type_label = type_label
        shifted_reg_label = reg_label

    return shifted_image, shifted_type_label, shifted_reg_label


def resize(image, type_label, reg_label):
    target_width = torch.randint(900, 1100, (1,)).item()
    image = F.interpolate(image.unsqueeze(0), size=(target_width, target_width), mode='bilinear', align_corners=False).squeeze(0)
    type_label = F.interpolate(type_label.unsqueeze(0).unsqueeze(0).float(), size=(target_width,), mode='nearest').squeeze(0).squeeze(0).long()
    reg_label = F.interpolate(reg_label.unsqueeze(0).unsqueeze(0).float(), size=(target_width,), mode='nearest').squeeze(0).squeeze(0)

    return image, type_label, reg_label


def augment(image, type_label, reg_label):
    """
    Augmentation logic specific to coor type.
    """

    if torch.rand(1).item() > 0.5:
        image, type_label, reg_label = resize(image, type_label, reg_label)

    if torch.rand(1).item() > 0.5:
        image, type_label, reg_label = rotate(image, type_label, reg_label)

    max_shift = 100
    if torch.rand(1).item() > 0.5:
        image, type_label, reg_label = shift(image, type_label, reg_label, max_shift)
        
    if torch.rand(1).item() > 0.5:
        image, type_label, reg_label = horizontal_flip(image, type_label, reg_label)

    original_width = 1000 
    image = F.interpolate(image.unsqueeze(0), size=(original_width, original_width), mode='bilinear', align_corners=False).squeeze(0)
    type_label = F.interpolate(type_label.unsqueeze(0).unsqueeze(0).float(), size=(original_width,), mode='nearest').squeeze(0).squeeze(0).long()
    reg_label = F.interpolate(reg_label.unsqueeze(0).unsqueeze(0).float(), size=(original_width,), mode='nearest').squeeze(0).squeeze(0)

    return image, type_label, reg_label


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