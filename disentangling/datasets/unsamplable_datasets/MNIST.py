import torchvision.datasets as datasets
import torchvision.transforms as transforms


def MNIST(root="data", **kwargs):
    """MNIST dataset from `MNIST <http://yann.lecun.com/exdb/mnist/>`

    Args:
        data_path (string): path to read data

    Returns:
        torch.utils.data.Dataset: samplable datasets for train
        torch.utils.data.Dataset: samplable datasets for evaluation
    """

    params = dict(root=root, download=True, transform=transforms.ToTensor())
    train_set = datasets.MNIST(**params, train=True)
    test_set = datasets.MNIST(**params, train=False)
    return train_set, test_set


MNIST.shape = [1, 28, 28]
