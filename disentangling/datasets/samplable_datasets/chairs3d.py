from pathlib import Path
import re
import numpy as np
import scipy.io
from PIL import Image

from ...utils.data import SamplableDataset, get_selected_subsets


def read_image(p, size=64, center_crop=1):
    pic = Image.open(p)
    if center_crop < 1:
        width, height = pic.size
        negative = 0.5 - center_crop / 2
        positive = 0.5 - center_crop / 2
        left, top = width * negative, height * negative
        right, bottom = width * positive, height * positive
        pic = pic.crop((left, top, right, bottom))
    pic.thumbnail((size, size), Image.ANTIALIAS)
    return pic


def chairs3d(data_path="./data", **kwargs):
    """Chairs3d dataset from `Seeing 3D chairs: exemplar part-based 2D-3D alignment using a large dataset of CAD models <https://ieeexplore.ieee.org/document/6909876>`

    Args:
        data_path (string): path to read data

    Returns:
        torch.utils.data.Dataset: samplable datasets for train
        torch.utils.data.Dataset: samplable datasets for evaluation
    """

    # tar -xf nips2015-analogy-data.tar.gz
    # mv data cars3d
    file_path = Path(data_path) / "rendered_chairs"
    with open(file_path / "all_chair_names.mat", "rb") as f:
        mat = scipy.io.loadmat(f)
        folders = mat["folder_names"][0]
        instances = mat["instance_names"][0]

    sub_labels = [re.findall(r"[a-z](\d+)", item[0]) for item in instances]
    sub_labels = np.array(sub_labels).astype(int)[:, 0:2]

    x, y = [], []
    for f in folders:
        for i in instances:
            p = file_path / f[0] / "renders" / i[0]
            x.append(read_image(str(p)))
        y.append(np.hstack([np.ones((len(instances), 1)), sub_labels]))
    y = np.vstack(y)
    discrete = [True, True, False]
    dataset = SamplableDataset(x, y, discrete)
    return get_selected_subsets(dataset, **kwargs)


chairs3d.shape = [3, 64, 64]
