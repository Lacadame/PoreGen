import diffsci.models
import diffsci.models.nets.autoencoderldm3d


def get_model(cfg):
    model_type = cfg['type']
    items = dict()
    if model_type == 'PUNetG':
        # Create PUNetGConfig
        config_params = cfg.get('config', {})
        punetg_config = diffsci.models.PUNetGConfig(**config_params)

        # Create PUNetG
        model_params = cfg.get('params', {})
        conditional_embedding = model_params.pop('conditional_embedding', None)
        channel_conditional_items = model_params.pop('channel_conditional_items', None)

        if channel_conditional_items:
            model = diffsci.models.PUNetGCond(punetg_config,
                                              conditional_embedding=conditional_embedding,
                                              channel_conditional_items=channel_conditional_items,
                                              **model_params)
        else:
            model = diffsci.models.PUNetG(punetg_config,
                                          conditional_embedding=conditional_embedding,
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
