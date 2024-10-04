import diffsci.models
import diffsci.models.nets.autoencoderldm3d

from . import embedder


def get_conditional_embedding(conditional_embedding=None,
                              conditional_embedding_args=None,
                              dembed=64):
    if conditional_embedding_args is None:
        conditional_embedding_args = {}
    if conditional_embedding is None:
        return None
    elif isinstance(conditional_embedding, str):
        return get_single_embedding(conditional_embedding,
                                    conditional_embedding_args,
                                    dembed)
    elif isinstance(conditional_embedding, list):
        embedders = []
        for embedding_type in conditional_embedding:
            args = conditional_embedding_args.get(embedding_type, {})
            embedders.append(get_single_embedding(embedding_type, args, dembed))
        return embedder.CompositeEmbedder(embedders)
    else:
        raise ValueError(f"Unsupported conditional_embedding type: {conditional_embedding}")


def get_single_embedding(embedding_type, embedding_kwargs, dembed):
    # Get the embedding class from its name
    # From the embedder module
    embedding_fn = getattr(embedder, f'get_{embedding_type}', None)
    if embedding_fn is None:
        raise ValueError(f"Embedding type {embedding_type} not found")

    # Create the embedding instance
    embed = embedding_fn(dembed, **embedding_kwargs)
    return embed


def get_model(cfg):
    model_type = cfg['type']
    items = dict()
    if model_type == 'PUNetG':
        # Create PUNetGConfig
        config_params = cfg.get('config', {})
        print(config_params)
        punetg_config = diffsci.models.PUNetGConfig(**config_params)

        # Create PUNetG
        model_params = cfg.get('params', {})
        conditional_embedding = model_params.pop('conditional_embedding', None)  # noqa: F841
        conditional_embedding_kwargs = model_params.pop('conditional_embedding_kwargs', None)  # noqa: F841
        channel_conditional_items = model_params.pop('channel_conditional_items', None)  # noqa: F841
        dembed = config_params.get('model_channels', 64)
        embed = get_conditional_embedding(conditional_embedding,
                                          conditional_embedding_kwargs,
                                          dembed)
        channel_conditional_items = model_params.pop('channel_conditional_items', None)

        if channel_conditional_items:
            raise NotImplementedError
            model = diffsci.models.PUNetGCond(punetg_config,
                                              conditional_embedding=embed,
                                              channel_conditional_items=channel_conditional_items,
                                              **model_params)
        else:
            model = diffsci.models.PUNetG(punetg_config,
                                          conditional_embedding=embed,
                                          **model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    items['model'] = model

    # Load autoencoder
    autoencoder_cfg = cfg.get('autoencoder', {})
    if autoencoder_cfg:
        autoencoder_type = autoencoder_cfg['type']
        if autoencoder_type == 'AutoencoderKL':
            checkpoint_path = autoencoder_cfg['checkpoint_path']
            lossconfig = diffsci.models.nets.autoencoderldm3d.lossconfig(
                kl_weight=autoencoder_cfg.get('kl_weight', 1e-4)
            )
            ddconfig = diffsci.models.nets.autoencoderldm3d.ddconfig(
                resolution=autoencoder_cfg['resolution'],
                has_mid_attn=autoencoder_cfg.get('has_mid_attn', False)
            )
            vae_module = diffsci.models.nets.autoencoderldm3d.AutoencoderKL.load_from_checkpoint(
                checkpoint_path,
                ddconfig=ddconfig,
                lossconfig=lossconfig
            )
            vae_module.eval()
        else:
            raise ValueError(f"Unsupported autoencoder type: {autoencoder_type}")
    else:
        vae_module = None

    items['autoencoder'] = vae_module
    return items
