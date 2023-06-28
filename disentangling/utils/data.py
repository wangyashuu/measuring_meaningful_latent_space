from typing import List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, random_split, Subset
import torchvision.transforms as transforms


class SamplableDataset(Dataset):
    """Samplable dataset

    Samplable means the dataset can generate the data sample from the random sampled factors. Unlike unsamplable dataset (such as CelebA), you can not find the corresponding data for a random factor representation.

    Attributes:
        discrete (List[bool]): indicate if each dimension of the factors is discrete

    """

    def __init__(
        self, data, labels, discrete, transform_x=None, transform_y=None
    ):
        """Create a samplable dataset.

        Args:
            transform_y (callable, optional): Optional transform to be applied
                on a sample.
            transform_x (callable, optional): Optional transform to be applied
                on a sample.
        """
        latent_values = [
            np.unique(labels[:, i]) for i in range(labels.shape[-1])
        ]
        self._latent_values = latent_values
        self._discrete = discrete
        self.data = data
        self.labels = labels
        self.transform_x = transform_x or transforms.ToTensor()
        self.transform_y = transform_y

    @property
    def discrete(self) -> List[bool]:
        return self._discrete

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.transform_x:
            x = self.transform_x(x)
        if self.transform_y:
            y = self.transform_y(y)
        return x, y

    def latent2index(self, latents: np.ndarray) -> np.ndarray:
        """Transform the factors into the indices in the dataset."""
        latent_bases = get_latent_bases(get_latent_sizes(self._latent_values))
        latent_classes = latent_to_class(self._latent_values, latents)
        return latent_class_to_index(latent_bases, latent_classes)

    def sample_latent(self, batch_size: int):
        """Sample a size of `batch_size` factor representation."""
        return sample_latent(self._latent_values, batch_size)


def class_to_latent(latent_values: List[np.ndarray], classes: np.ndarray):
    """Transform the class representation into latent representation.

    The class value start from 0, while the latent values can be any number.
    Example: a latent dimension has latent values [-1, 1], the corresponding class value is [0, 1]

    Args:
        latent_values (np.ndarray): A list where each element represents the possible latent value in this dimension.
        classes (np.ndarray): The class representation.

    Returns:
        latents (np.ndarray): The latent representation.
    """
    latents = []
    for i, vals in enumerate(latent_values):
        latents.append(vals[classes[:, i]])
    return np.stack(latents, axis=1)


def latent_to_class(latent_values, latents):
    """Transform the latent representation into class representation.

    The explaination of latent representation and class representation could refer to `class_to_latent` function.

    Args:
        latent_values (np.ndarray): A list where each element represents the possible latent value in this dimension.
        latents (np.ndarray): The latent representation.

    Returns:
        classes (np.ndarray): The class representation.
    """
    classes = []
    for i, vals in enumerate(latent_values):
        _, clas = np.where(latents[:, i].reshape(-1, 1) == vals)
        classes.append(clas)
    return np.stack(classes, axis=1)


def get_latent_sizes(latent_values):
    """Return the number of possibilities of each dimension in latent representation."""
    return np.array([len(v) for v in latent_values])


def get_latent_bases(latent_sizes):
    """Return the latent bases, the base means the max possible combinations from the first factor to the current factors.

    Example: get_latent_bases([2, 3, 4]) # => [12, 4, 1]
    """
    latent_bases = np.array(latent_sizes)[::-1].cumprod()[::-1][1:]
    latent_bases = np.concatenate((latent_bases, np.array([1])))
    return latent_bases


def latent_class_to_index(latent_bases, latent_classes):
    """Return the indices of the correponding class."""
    return np.dot(latent_classes, latent_bases).astype(int)


def index_to_latent_class(latent_bases, index):
    """Find the class reprentation of given indices."""
    classes = []
    left = index
    for base in latent_bases:
        classes.append(left // base)
        left = left % base
    return np.stack(classes, axis=1)


def sample_latent(latent_selections, batch_size):
    """Sample a size of `batch_size` latent representations.

    Args:
        latent_selections (np.ndarray): A list where each element represents the possible latent value in this dimension.
        batch_size (np.ndarray): The size to sample.

    Returns:
        latents (np.ndarray): The latent representation.
    """
    latent = [
        np.random.choice(selections, size=(batch_size, 1))
        for selections in latent_selections
    ]
    latent = np.hstack(latent)
    return latent


def select_index_by_range(latents_sizes, includes, excludes):
    """Return the indices of latents according to selected range."""
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


def train_test_split(dataset, train_rate=0.8, random_state=None):
    """Split train and test test according to given `random_state`"""
    params = dict(dataset=dataset, lengths=[train_rate, 1 - train_rate])
    if random_state is not None:
        params["generator"] = torch.Generator().manual_seed(random_state)
    return random_split(**params)


def get_selected_subsets(
    dataset: Dataset,
    train_rate: float = 0.8,
    random_state: Optional[float] = None,
    includes: Tuple[float, float] = None,
    excludes: Tuple[float, float] = None,
):
    """Return train and test dataset, where the indices of train set can be selected by includes or exclude.

    If no selected range provided, the dataset will be split into train and test set by random.
    If provided the selected range, the train set will include the selected indices and the test set will include the complement set of the selected range.

    Args:

        dataset (Dataset): The target dataset.
        train_rate (float, optional): The train size, ranging from 0 to 1. Default: 0.8.
        random_state (Optional[float], optional):
    """
    size = len(dataset)
    if includes is None and excludes is None:
        train_set, val_set = train_test_split(
            dataset, train_rate=train_rate, random_state=random_state
        )
    else:
        indices = select_index_by_range(
            dataset.latents_sizes, includes, excludes
        )
        train_set = Subset(dataset, indices)
        val_set = Subset(dataset, np.setdiff1d(np.arange(size), indices))
    return train_set, val_set
