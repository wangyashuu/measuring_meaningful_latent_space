import torchvision.datasets as datasets
import torchvision.transforms as transforms


def CelebA(root="data", input_size=64):
    # gdown https://drive.google.com/uc?id=1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ
    # https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
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
