import torch
import collections
import os
from torch.utils.data import DataLoader, ConcatDataset
from models_resnet import Resnet18_md, Resnet50_md, ResnetModel
from data_loader import KittiLoader,KittiLoader_Eigen
from transforms import image_transforms
# 本模块主要包括三个函数

# def to_device(input,device).
# 考虑input的类型,将其赋予为的device类型. cpu/cuda:'0'/cuda:'1'/...

# def get_model(model,input_channels,pretrained)
# 选择合适的模型,提供了自定义的resnet_18,resnet_50和动态导入的resnet_

# def prepare_dataloader(data_directory, mode, augment_parameters,
#                        do_augmentation, batch_size, size, num_workers)
# 数据生成器,包括用于kitti数据集的KittiLoader().

def to_device(input, device):
    if torch.is_tensor(input):
        return input.to(device=device)
    elif isinstance(input, str):
        return input
    elif isinstance(input, collections.Mapping):
        return {k: to_device(sample, device=device) for k, sample in input.items()}
    elif isinstance(input, collections.Sequence):
        return [to_device(sample, device=device) for sample in input]
    else:
        raise TypeError("Input must contain tensor, dict or list, found {type(input)}")


def get_model(model, input_channels=3, pretrained=False):
    if model == 'resnet50_md':
        out_model = Resnet50_md(input_channels)
    elif model == 'resnet18_md':
        out_model = Resnet18_md(input_channels)
    else:
        out_model = ResnetModel(input_channels, encoder=model, pretrained=pretrained)
    return out_model


def prepare_dataloader(data_directory, mode, augment_parameters,
                       do_augmentation, batch_size, size, num_workers,split,filename):
    data_dirs = os.listdir(data_directory)
    data_transform = image_transforms(
        mode=mode,
        augment_parameters=augment_parameters,
        do_augmentation=do_augmentation,
        size = size)
    if split=='kitti':
        datasets = [KittiLoader(os.path.join(data_directory,
                                data_dir), mode, transform=data_transform)
                                for data_dir in data_dirs]
        # 考虑datasets是多个数据loader组成的list,通过ConcatDataset对其进行合并成一个整合loader
        dataset = ConcatDataset(datasets)
        n_img = len(dataset)
        print('KITTI: Use a dataset with', n_img, 'images')
    elif split=='eigen':
        dataset = KittiLoader_Eigen(root_dir=data_directory,root_filename = filename,
                                    mode = mode,transform=data_transform)
        n_img = len(dataset)
        print('EIGEN: Use a dataset with', n_img, 'images')
    else:
        print('Wrong split')
        pass
    if mode == 'train':
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers,
                            pin_memory=True)
    else:
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)
    return n_img, loader
