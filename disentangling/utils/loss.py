from torch.nn import functional as F


def get_reconstruction_loss(output, input, distribution):
    batch_size = input.shape[0]
    if distribution == "bernoulli":
        loss = F.binary_cross_entropy_with_logits(
            output, input, reduction="sum"
        )
    elif distribution == "gaussian":
        loss = F.mse_loss(F.sigmoid(output), input, reduction='sum')
    else:
        raise NotImplementedError(
            f"get_reconstruction_loss distribution = {distribution}"
        )
    return loss / batch_size

def get_kld_loss(mu, logvar):
    kld_loss = -0.5 * (1 + logvar - mu**2 - logvar.exp()).sum(1).mean()
    return kld_loss