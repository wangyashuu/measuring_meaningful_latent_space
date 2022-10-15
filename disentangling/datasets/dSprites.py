from urllib import request
import os.path

import numpy as np
from torch.utils.data import Dataset, random_split
import torchvision.transforms as transforms


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
        if not os.path.exists(file_path):  # isfile(fname)
            request.urlretrieve(dataset_url, file_path)
        # Data was saved originally using python2, so we need to set the encoding.
        data = np.load(file_path, encoding="latin1", allow_pickle=True)
        self.data = data
        self.images = data["imgs"] * 255  # [:, None] increase the channel dim
        self.latents = data["latents_values"]
        self.latent_classes = data["latents_classes"]
        self.metadata = data["metadata"]
        self.transform = transform or transforms.ToTensor()

    @property
    def latents_sizes(self):
        return self.metatdata["latents_sizes"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        latent = self.latent_classes[idx]
        if self.transform:
            image = self.transform(image)
        return image, latent


def select_index_by_range(latents_sizes, includes, excludes):
    index_matrix = np.arange(latents_sizes).reshape(latents_sizes)
    all_indices = ()
    for i, s in enumerate(latents_sizes):
        if str(i) in includes:
            include_range = (np.array(includes[str(i)]) * s).astype(int)
            include_indices = slice(*include_range)
        elif str(i) in excludes:
            exclude_range = (np.array(excludes[str(i)]) * s).astype(int)
            include_indices = list(range(0, exclude_range[0])) + list(
                range(exclude_range[1], s)
            )
        else:
            include_indices = slice(
                None,
            )
        indices = indices + (include_indices,)
    selected = index_matrix[all_indices].flatten()
    return selected


def dSprites(train_rate=0.8, includes=None, excludes=None):
    dSprites = DSprites()
    size = len(dSprites)
    if includes is not None or excludes is not None:
        indices = select_index_by_range(
            dSprites.latents_sizes, includes, excludes
        )
        train_set = dSprites[indices]
        val_set = dSprites[np.delete(np.arange(size), indices)]
    else:
        train_size = int(size * train_rate)
        val_size = size - train_size
        train_set, val_set = random_split(dSprites, [train_size, val_size])
    return train_set, val_set
