import torchvision.datasets as datasets
import torchvision.transforms as transforms


def CelebA_sets(root="data", input_size=64):
    train_set = datasets.CelebA(
        root=root,
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
    test_set = datasets.CelebA(
        root=root,
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
    return train_set, test_set
