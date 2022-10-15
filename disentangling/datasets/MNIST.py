import torchvision.datasets as datasets
import torchvision.transforms as transforms


def MNIST(root="data"):
    train_set = datasets.MNIST(
        root=root, train=True, download=True, transform=transforms.ToTensor()
    )
    test_set = datasets.MNIST(
        root=root, train=False, download=True, transform=transforms.ToTensor()
    )
    return train_set, test_set
