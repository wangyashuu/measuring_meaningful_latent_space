from pathlib import Path
import re
import numpy as np
import scipy.io
from PIL import Image

from ...utils.data import SamplableDataset, get_selected_subsets


def read_image(p, size=64):
    pic = Image.open(p)
    # width, height = pic.size   # Get dimensions
    # new_width, new_height = width * 0.8, height * 0.8
    # # Crop the center of the image
    # left = (width - new_width)/2
    # top = (height - new_height)/2
    # right = (width + new_width)/2
    # bottom = (height + new_height)/2
    # pic = pic.crop((left, top, right, bottom))
    pic.thumbnail((size, size), Image.ANTIALIAS)
    return pic


def chairs3d(data_path="./data", input_size=64, **kwargs):
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
