import torch
from sklearn.decomposition import PCA

# representation_functions


def m0c0(factors):
    latents = torch.zeros_like(factors)
    return latents


def m0c1(factors):
    batch_size, n_factors = factors.shape
    n_factors_per_latent = 2
    n_latents = n_factors // n_factors_per_latent
    latents = torch.zeros(
        (batch_size, n_latents), dtype=factors.dtype, device=factors.device
    )
    for i in range(n_latents):
        target_factors = factors[:, i * 2 : (i + 1) * 2].cpu()
        # print(a.shape, latents[:, i].shape)
        latents[:, i] = torch.from_numpy(
            PCA(n_components=1).fit_transform(target_factors)
        ).reshape(-1)
    return latents

def m0c1_duplicated_factors(factors):
    batch_size, n_factors = factors.shape
    n_factors_per_latent = 2
    n_latents = n_factors // n_factors_per_latent
    latents = factors[:, :n_latents]
    factors[:, n_latents:] = latents
    return latents

def m1c0(factors):
    x = torch.clone(factors)
    latents = torch.hstack([x, x])
    return latents


def m1c1(factors):
    latents = torch.clone(factors)
    return latents


def generate_factors(batch_size, n_factor_dims, mu=0, sigma=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    factors = (
        torch.normal(mu, sigma, size=(batch_size, n_factor_dims))
        .int()
        .to(device)
    )
    return factors


def run_metric(metric, representation_function, batch_size=12000):
    factors = generate_factors(batch_size, n_factor_dims=4)
    latents = representation_function(factors)
    score = metric(factors, latents)
    return score.cpu()
