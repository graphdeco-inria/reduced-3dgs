from torch.utils.data import Dataset
from skimage import io
import numpy as np
import torch
import os
from random import randint

def crop_image(img, crop_coords):
    return img[crop_coords[2]:crop_coords[3], crop_coords[0]:crop_coords[1], :]

def crop_center(img, cropx, cropy):
    y,x,c = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2
    return img[starty:starty+cropy, startx:startx+cropx, :]

def get_random_crop_coords(img_dims, crop_size):
    x,y = img_dims
    x_crop, y_crop = crop_size
    assert(x >= x_crop and y >= y_crop)
    x0 = randint(0, x - x_crop)
    x1 = x0 + x_crop
    y0 = randint(0, y - y_crop)
    y1 = y0 + y_crop
    return (x0, x1, y0, y1)

class DatasetReaderV1(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.paths_of_views = [os.path.join(root_dir, folder) for folder in os.listdir(root_dir)]

    def __len__(self):
        return len(self.paths_of_views)

    def __getitem__(self, idx):
        batch_volume = None
        gt_mask = None  # 1 x CHANNELS x WIDTH x HEIGHT
        gt_render = None  # 1 x CHANNELS x WIDTH x HEIGHT
        image_volume = None  # IMAGES x CHANNELS x WIDTH x HEIGHT
        view_path = self.paths_of_views[idx]
        crop_coords = get_random_crop_coords((2256, 1504), (256, 256))
        for filename in os.listdir(view_path):
            image = io.imread(os.path.join(view_path, filename))
            image = crop_image(image, crop_coords) / 255
            image = np.transpose(image, (2, 0, 1))
            image = np.expand_dims(image, 0)
            if filename.startswith("title_cam"):
                if image_volume is None:
                    image_volume = np.array(image)
                else:
                    image_volume = np.concatenate((image_volume, image), axis=0)
            if filename.startswith("gt_"):
                gt_render = np.array(image)
            if filename.startswith("title_mask"):
                gt_mask = np.array(image)

        sample = {"image_volume": torch.from_numpy(image_volume), "gt_render":  torch.from_numpy(gt_render), "gt_mask": torch.from_numpy(gt_mask)}
        return sample
