import torchvision.datasets as datasets
import torchvision.transforms as transforms


def MNIST(root="data", **kwargs):
    params = dict(root=root, download=True, transform=transforms.ToTensor())
    train_set = datasets.MNIST(**params, train=True)
    test_set = datasets.MNIST(**params, train=False)
    return train_set, test_set


MNIST.shape = [1, 28, 28]
