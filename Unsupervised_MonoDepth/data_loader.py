import os
from PIL import Image

from torch.utils.data import Dataset
# 在实现dataloader时,考虑的是不能够一口气把所有数据读到内存中,这样在大数据集下对硬件要求很高.
# 因此采用pytorch自带的Dataset, DataLoader, ConcatDataset来实现一个有效的数据生成器

class KittiLoader(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        left_dir = os.path.join(root_dir, 'image_02/data/')
        self.left_paths = sorted([os.path.join(left_dir, fname) for fname\
                           in os.listdir(left_dir)])
        if mode == 'train':
            right_dir = os.path.join(root_dir, 'image_03/data/')
            self.right_paths = sorted([os.path.join(right_dir, fname) for fname\
                                in os.listdir(right_dir)])
            assert len(self.right_paths) == len(self.left_paths)
        self.transform = transform
        self.mode = mode


    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        left_image = Image.open(self.left_paths[idx])
        if self.mode == 'train':
            right_image = Image.open(self.right_paths[idx])
            sample = {'left_image': left_image, 'right_image': right_image}

            if self.transform:
                sample = self.transform(sample)
                return sample
            else:
                return sample
        else:
            if self.transform:
                left_image = self.transform(left_image)
            return left_image

# 使用和Eigen相同的方法来分割训练/验证/测试集
class KittiLoader_Eigen(Dataset):
    def __init__(self, root_dir, root_filename, mode, transform=None):

        self.count_lines, self.all_lines = self.get_text_lines(root_filename)
        self.left_paths = []
        self.right_paths = []
        for line in self.all_lines:
            left_line = line.split(' ')[0][:63] + 'png'
            right_line = line.split(' ')[1][:63] + 'png'
            self.left_paths.append(os.path.join(root_dir, left_line))
            self.right_paths.append(os.path.join(root_dir, right_line))
        if mode == 'train':
            assert len(self.right_paths) == len(self.left_paths)
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.left_paths)

    def get_text_lines(self, file_path):
        f = open(file_path, 'r')
        lines = f.readlines()
        f.close()
        return len(lines), lines

    def __getitem__(self, idx):
        left_image = Image.open(self.left_paths[idx])
        if self.mode == 'train':
            right_image = Image.open(self.right_paths[idx])
            sample = {'left_image': left_image, 'right_image': right_image}

            if self.transform:
                sample = self.transform(sample)
                return sample
            else:
                return sample
        else:
            if self.transform:
                left_image = self.transform(left_image)
            return left_image