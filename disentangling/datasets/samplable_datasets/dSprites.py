from pathlib import Path
from urllib import request
import numpy as np

from ...utils.data import SamplableDataset, get_selected_subsets


def dSprites(data_path="./data", **kwargs):
    """dSprites dataset from `dSprites - Disentanglement testing Sprites dataset <https://github.com/deepmind/dsprites-dataset>`

    Args:
        data_path (string): path to read data

    Returns:
        torch.utils.data.Dataset: samplable datasets for train
        torch.utils.data.Dataset: samplable datasets for evaluation
    """

    file_path = Path(data_path) / "dSprites.npz"

    if not file_path.exists():
        dataset_url = "https://raw.githubusercontent.com/deepmind/dsprites-dataset/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
        request.urlretrieve(dataset_url, file_path)

    # Data was saved originally using python2, so we need to set the encoding.
    file = np.load(file_path, encoding="latin1", allow_pickle=True)
    metadata = file["metadata"][()]
    selected_latents = metadata["latents_sizes"] > 1
    data = file["imgs"] * 255
    labels = file["latents_values"][:, selected_latents]
    discrete = [True, False, False, False, False]
    dataset = SamplableDataset(data, labels, discrete)
    return get_selected_subsets(dataset, **kwargs)


dSprites.shape = [1, 64, 64]
