# Measuring Meaningful Latent Space

## Implemented

-   Datasets: [cars3d](https://proceedings.neurips.cc//paper/5845-deep-visual-analogy-making), [chairs3d](https://ieeexplore.ieee.org/document/6909876), [dSprites](https://github.com/deepmind/dsprites-dataset), [shapes3d](https://github.com/deepmind/3d-shapes), [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [MNIST](http://yann.lecun.com/exdb/mnist/)
<!-- from .Generateds import generate -->
-   Metrics: [Kim metric](https://arxiv.org/abs/1802.05983), [SAP](https://arxiv.org/abs/1711.00848), [MIG](https://arxiv.org/abs/1802.04942), [MIG_sup](https://arxiv.org/abs/2002.10549), [DCIMIG](https://arxiv.org/abs/1910.05587), [DCI](https://openreview.net/forum?id=By-7dz-AZ), [Modularity](https://arxiv.org/abs/1802.05312), DCII, smoothness.

-   Models: AE (Autoencoder), [VAE (Variational Autoencoder)](https://arxiv.org/abs/1312.6114), [BetaVAE](https://openreview.net/forum?id=Sy2fzU9gl), [FactorVAE](https://arxiv.org/abs/1802.05983), [BetaTCVAE](https://arxiv.org/abs/1802.04942), [DIPVAE](https://arxiv.org/abs/1711.00848), [InfoVAE](https://arxiv.org/abs/1706.02262).

## Install

```shell
python3 -m venv .venv
source .venv/bin/activate
```

Install the requirements.

```shell
pip install torch torchvision numpy scipy scikit-learn xgboost pyyaml python-box pillow pytest h5py wandb tqdm
```

or install with the recommended version.

```shell
pip install requirements.txt
```

## Usage

### Prepare Datasets

1. Walk into the project folder, and create a folder name "data"
    ```
    cd measuring_meaningful_latent_space
    mkdir data
    ```
2. download the dataset from their official providing website, place them in the `data` folder and unzip.

### Reproducing experiments

-   Basic Tests with the metrics.

    ```shell
    python -m experiments.test_metric
    ```

-   Training: to run the scripts provided in the `experiments` folder, you can run the following command. You can also specify the dataset, model, model specific parameters, and seed.

    ```shell
    python -m experiments.run --dataset dSprites --model-name beta_vae --beta 2 --seed 0
    ```

-   Evaluating: to run the scripts provided in the `experiments` folder, you only need to add `--eval` command option comparing with training. Noted that it is required to run training first, since evaluation rely on the model saved in output path to evaluate.
    ```shell
    python -m experiments.run --dataset dSprites --model-name beta_vae --beta 2 --seed 0 --eval
    ```

### Run your own experiments

-   You can also run your own scripts using the library in `disentangling` folder.
