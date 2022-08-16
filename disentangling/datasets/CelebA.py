import torchvision.datasets as datasets
import torchvision.transforms as transforms


def CelebA_sets(input_size=64):
    trainset = datasets.CelebA(
        root="data",
        transform=transforms.Compose(
            [
                transforms.CenterCrop(148),
                transforms.Resize(input_size),
                transforms.ToTensor(),
            ]
        ),
        download=True,
        split="train",
    )
    testset = datasets.CelebA(
        root="data",
        transform=transforms.Compose(
            [
                transforms.CenterCrop(148),
                transforms.Resize(input_size),
                transforms.ToTensor(),
            ]
        ),
        download=True,
        split="test",
    )
    return trainset, testset
