import torch

import disentangling.datasets
import disentangling.models
import disentangling.metrics

from disentangling.utils.nn import (
    get_fc_encoder,
    get_fc_decoder,
    get_cnn_decoder,
    get_cnn_encoder,
)


def create_datasets(conf):
    conf = conf.copy()
    name = conf.pop("name")
    conf.pop("input_shape")
    D = getattr(disentangling.datasets, name)
    if not hasattr(disentangling.datasets, name):
        raise Exception(f"dataset {name} not implemented")
    return D(**conf)


def create_model_core(input_shape, params):
    # TODO: refactor
    if isinstance(params, (list, tuple)):
        cnn_params, fc_params = params
        cnn_encoder, output_shape, output_paddings = get_cnn_encoder(
            input_shape=input_shape,
            hiddens=cnn_params["hiddens"],
        )
        fc_encoder = get_fc_encoder(
            input_shape=output_shape, hiddens=fc_params["hiddens"]
        )
        encoder_output_shape = (fc_params["hiddens"][-1],)
        encoder = torch.nn.Sequential(cnn_encoder, fc_encoder)
        fc_decoder = get_fc_decoder(
            hiddens=fc_params["hiddens"][::-1],
            output_shape=output_shape,
            is_output=False,
        )
        cnn_decoder = get_cnn_decoder(
            hiddens=cnn_params["hiddens"][::-1],
            output_paddings=output_paddings[::-1],
            output_shape=input_shape,
        )
        decoder = torch.nn.Sequential(fc_decoder, cnn_decoder)
    elif params["net_type"] == "fc":
        encoder = get_fc_encoder(
            input_shape=input_shape, hiddens=params["hiddens"]
        )
        decoder = get_fc_decoder(
            hiddens=params["hiddens"][::-1], output_shape=input_shape
        )
        encoder_output_shape = (params["hiddens"][-1],)
    elif params["net_type"] == "cnn":
        encoder, encoder_output_shape, output_paddings = get_cnn_encoder(
            input_shape=input_shape,
            hiddens=params["hiddens"],
        )
        decoder = get_cnn_decoder(
            hiddens=params["hiddens"][::-1],
            output_paddings=output_paddings[::-1],
            output_shape=input_shape,
        )
    return encoder, decoder, encoder_output_shape


model_cls_map = dict(
    ae="AE",
    beta_tcvae="BetaTCVAE",
    beta_vae="BetaVAE",
    dip_vae="DIPVAE",
    factor_vae="FactorVAE",
    info_vae="InfoVAE",
    vae="VAE",
    vq_vae="VQVAE",
)


def create_model(model_name, input_shape, net_params, latent_dim):
    model_cls_name = model_cls_map[model_name]
    encoder, decoder, output_shape = create_model_core(input_shape, net_params)

    if not hasattr(disentangling.models, model_name):
        raise Exception(f"model {model_name} not implemented")

    model_cls = getattr(disentangling.models, model_cls_name)
    if model_cls_name == "AE":
        return model_cls(encoder=encoder, decoder=decoder)
    return model_cls(
        encoder=encoder,
        decoder=decoder,
        encoder_output_shape=output_shape,
        decoder_input_shape=output_shape,
        latent_dim=latent_dim,
    )


def create_compute_loss(model_name, loss_kwargs):
    compute_loss = getattr(disentangling.models, f"compute_{model_name}_loss")

    def compute_loss_with_kwargs(*args, **kwargs):
        return compute_loss(*args, **kwargs, **loss_kwargs)

    return compute_loss_with_kwargs


def create_auxiliary_model(conf, latent_dim):
    conf = conf.copy()
    name = conf.pop("name")
    model_cls_name = name.title().replace("Vae", "VAE").replace("_", "")
    model = getattr(disentangling.models, model_cls_name)
    return model(latent_dim=latent_dim, **conf)


def create_optimizer(model, conf):
    optimizer_func = getattr(torch.optim, conf.pop("name", "Adam"))
    return optimizer_func(model.parameters(), **conf)


def create_scheduler(optimizer, conf):
    scheduler_func = getattr(
        torch.optim.lr_scheduler, conf.pop("name", "ExponentialLR")
    )
    scheduler = scheduler_func(optimizer, **conf)
    return scheduler


def create_metrics(confs):
    metric_funcs = [
        getattr(disentangling.metrics, metric_name) for metric_name in confs
    ]
    return metric_funcs
