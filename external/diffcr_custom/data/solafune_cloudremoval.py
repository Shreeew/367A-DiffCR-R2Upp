import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import rasterio


def load_image(image_path):
    with rasterio.open(image_path) as src:
        image = src.read()  # (12, H, W)
    image = np.transpose(image, (1, 2, 0))  # (H, W, 12)
    assert image.shape == (1024, 1024, 12)
    image = np.nan_to_num(image)
    return image.astype(np.float32)


def normalize_image(image):
    mean = np.array([
        285.819056, 327.220914, 552.930595, 392.157514,
        914.313880, 2346.118450, 2884.483170, 2886.442429,
        3176.750133, 3156.934442, 1727.194007, 848.573373
    ], dtype=np.float32).reshape(12, 1, 1)

    std = np.array([
        216.449756, 269.888025, 309.927908, 397.456556,
        400.220789, 630.326965, 789.800692, 810.477370,
        852.903143, 807.597620, 631.780811, 502.667887
    ], dtype=np.float32).reshape(12, 1, 1)

    return (image - mean) / std


class SolafuneCloudRemovalDataset(Dataset):
    def __init__(self, data_root, indices=None):
        all_paths = sorted(Path(data_root).joinpath("train_images").glob("train_*.tif"))
        if indices is not None:
            self.image_paths = [all_paths[i] for i in indices]
        else:
            self.image_paths = all_paths


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = load_image(self.image_paths[idx])         # (H, W, 12)
        img = img.transpose(2, 0, 1)                     # (12, H, W)
        clean = normalize_image(img)                    # (12, H, W)
        cloudy = self.add_synthetic_clouds(clean.copy())  # (12, H, W)

        return {
            'x': torch.from_numpy(cloudy).float(),   # cloudy input
            'y0': torch.from_numpy(clean).float()    # clean target
        }

    def add_synthetic_clouds(self, img):
        """Adds synthetic clouds by zeroing out a random 10% mask."""
        _, H, W = img.shape
        cloud_mask = np.random.rand(H, W) < 0.1  # 10% random occlusion
        img[:, cloud_mask] = 0
        return img
    
    
class SolafuneCloudyTestDataset(Dataset):
    def __init__(self, data_root):
        self.image_paths = sorted(Path(data_root).glob("*.tif"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = load_image(self.image_paths[idx])  # (H, W, 12)
        img = img.transpose(2, 0, 1)              # (12, H, W)
        img = normalize_image(img)               # normalize input

        return {
            'x': torch.from_numpy(img).float(),  # cloudy image
            'meta': {'name': self.image_paths[idx].stem}  # filename for saving
        }



