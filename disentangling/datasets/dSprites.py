from urllib import request
import os.path

import numpy as np
import torch
from torch.utils.data import Dataset, random_split, Subset
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
        metadata = data["metadata"][()]
        self.images = data["imgs"] * 255
        self.metadata = metadata
        selected_latents = metadata["latents_sizes"] > 1
        self.latents = data["latents_values"][:, selected_latents]
        self.latent_classes = data["latents_classes"][:, selected_latents]
        self.transform = transform or transforms.ToTensor()
        self.latents_sizes = metadata["latents_sizes"][selected_latents]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        latent = self.latent_classes[idx]
        if self.transform:
            image = self.transform(image)
        return image, latent

    def latent2index(self, latents):
        latents_sizes = self.latents_sizes
        latents_bases = latents_sizes[::-1].cumprod()[::-1][1:]
        latents_bases = np.concatenate((latents_bases, np.array([1])))
        return np.dot(latents, latents_bases).astype(int)

    # def sample_latent(latent_ranges, size=1):
    #     samples = np.zeros((size, latent_ranges.shape[0]))
    #     for lat_i, lat_range in enumerate(latent_ranges):
    #         low, high = lat_range
    #         samples[:, lat_i] = np.random.randint(
    #             low=low, high=high, size=size
    #         )
    #     return samples

    # def get_latent_ranges(self):
    #     return self.latents


def select_index_by_range(latents_sizes, includes, excludes):
    index_matrix = np.arange(np.prod(latents_sizes)).reshape(latents_sizes)
    all_indices = ()
    for i, s in enumerate(latents_sizes):
        if includes is not None and i in includes:
            include_range = (np.array(includes[i]) * s).astype(int)
            include_indices = slice(*include_range)
        elif excludes is not None and i in excludes:
            exclude_range = (np.array(excludes[i]) * s).astype(int)
            include_indices = list(range(0, exclude_range[0])) + list(
                range(exclude_range[1], s)
            )
        else:
            include_indices = slice(None, None, None)
        all_indices = all_indices + (include_indices,)
    selected = index_matrix[all_indices].flatten()
    return selected


def dSprites(train_rate=0.8, random_state=None, includes=None, excludes=None):
    dSprites = DSprites()
    size = len(dSprites)
    if includes is not None or excludes is not None:
        indices = select_index_by_range(
            dSprites.latents_sizes, includes, excludes
        )
        train_set = Subset(dSprites, indices)
        val_set = Subset(dSprites, np.delete(np.arange(size), indices))
    else:
        train_size = int(size * train_rate)
        val_size = size - train_size
        if random_state is not None:
            generator = torch.Generator().manual_seed(random_state)
            train_set, val_set = random_split(
                dSprites, [train_size, val_size], generator=generator
            )
        else:
            train_set, val_set = random_split(dSprites, [train_size, val_size])
    return train_set, val_set
