from pathlib import Path
import h5py
import numpy as np

from ...utils.data import SamplableDataset, get_selected_subsets


def shapes3d(data_path="./data", **kwargs):
    # data, labels, discrete,
    file_path = Path(data_path) / "3dshapes.h5"
    file = h5py.File(file_path, "r")
    data = np.asarray(file["images"])
    labels = np.asarray(file["labels"])
    discrete_factors = [False, False, False, False, True, False]
    dataset = SamplableDataset(data, labels, discrete_factors)
    return get_selected_subsets(dataset, **kwargs)


shapes3d.shape = [3, 64, 64]
