from pathlib import Path
import numpy as np
import scipy.io
from sklearn.utils import extmath

from ...utils.data import SamplableDataset, get_selected_subsets


def cars3d(data_path="./data", **kwargs):
    # tar -xf nips2015-analogy-data.tar.gz
    # mv data cars3d
    file_path = Path(data_path) / "cars3d" / "cars"
    with open(file_path / "list.txt") as f:
        fnames = f.read().splitlines()

    data = []
    for name in fnames:
        with open(file_path / f"{name}.mat", "rb") as f:
            d = scipy.io.loadmat(f)["im"]
            d = np.moveaxis(d, -1, 0)
            d = np.moveaxis(d, -1, 0)
            data.append(d)
    data = np.stack(data, 0)
    factor_sizes = data.shape[:3]
    data = np.reshape(data, (-1,) + data.shape[3:])
    labels = extmath.cartesian([np.arange(i) for i in factor_sizes])
    discrete_factors = [s != 24 for s in factor_sizes]
    dataset = SamplableDataset(data, labels, discrete=discrete_factors)
    return get_selected_subsets(dataset, **kwargs)


cars3d.shape = [3, 64, 64]
