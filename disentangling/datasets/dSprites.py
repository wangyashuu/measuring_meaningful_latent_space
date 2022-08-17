from urllib import request
import os.path

import numpy as np
from torch.utils.data import Dataset, random_split
import torchvision.transforms as transforms


def download(remote_url, file="tmp"):
    if not os.path.exists(file):  # isfile(fname)
        request.urlretrieve(remote_url, file)


class DSprites(Dataset):
    def __init__(self, root="data", transform=None):
        """
        Args:
            root (string): Directory for saving the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        dataset_url = "https://raw.githubusercontent.com/deepmind/dsprites-dataset/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
        file_path = os.path.join(root, "dSprites.npz")
        download(dataset_url, file_path)
        # Data was saved originally using python2, so we need to set the encoding.
        data = np.load(file_path, encoding="latin1", allow_pickle=True)
        self.data = data
        self.images = data["imgs"] # [:, None] increase the channel dim
        self.latents = data["latents_values"]
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        latent = self.latents[idx]
        if self.transform:
            image = self.transform(image)
        return image, latent


def dSprites_sets(train_rate=0.8):
    dSprites = DSprites()
    size = len(dSprites)
    train_size = int(size * train_rate)
    val_size = size - train_size
    train_set, val_set = random_split(dSprites, [train_size, val_size])
    return train_set, val_set
