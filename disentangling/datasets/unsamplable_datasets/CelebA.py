import torchvision.datasets as datasets
import torchvision.transforms as transforms

from ...utils.data import train_test_split


def CelebA(data_path="./data", input_size=64, **kwargs):
    # gdown https://drive.google.com/uc?id=1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ
    # https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    transform = transforms.Compose(
        [
            transforms.CenterCrop(148),
            transforms.Resize(input_size),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.CelebA(
        root=data_path, transform=transform, download=True, split="all"
    )
    return train_test_split(dataset, **kwargs)


CelebA.shape = [3, 64, 64]
