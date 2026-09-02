from typing import Any

import torch
import diffsci.models
import diffsci.models.nets.autoencoderldm3d

from . import embedder

# Keys accepted by diffsci AutoencoderKL ddconfig. Unspecified keys keep
# the historical get_model defaults (has_mid_attn=False) or ddconfig() defaults.
_DDCONFIG_KEYS = (
    'double_z',
    'z_channels',
    'resolution',
    'in_channels',
    'out_ch',
    'ch',
    'ch_mult',
    'num_res_blocks',
    'attn_resolutions',
    'dropout',
    'has_mid_attn',
)
_CORE_VAE_PREFIXES = (
    'encoder.',
    'decoder.',
    'quant_conv.',
    'post_quant_conv.',
)


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


def get_model(cfg: dict[str, Any]) -> dict[str, Any]:
    """
        Returns a dict with keys 'model' and 'autoencoder'.
        'model' contains a PUNetG or PUNetGCond model
        'autoencoder' contains an autoencoder model or None
    """
    model_type = cfg['type']
    items = dict()
    if model_type == 'PUNetG':
        # Create PUNetGConfig
        config_params = cfg.get('config', {})
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
            raise NotImplementedError("Channel conditional items are not implemented in get_model")
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
    items['autoencoder'] = load_autoencoder_module(cfg.get('autoencoder', {}))
    return items


def build_ddconfig(autoencoder_cfg: dict[str, Any]):
    """Build a 3D AutoencoderKL ddconfig from a (possibly nested) yaml dict.

    Historical PoreGen LDM yamls only set ``resolution`` and ``has_mid_attn``.
    Geomodel VAEs also need ``ch``, ``ch_mult``, ``z_channels``, ``embed_dim``,
    and ``num_res_blocks`` or the checkpoint will not load.
    """
    nested = dict(autoencoder_cfg.get('config') or {})
    kwargs: dict[str, Any] = {}
    if 'resolution' not in autoencoder_cfg and 'resolution' not in nested:
        raise ValueError("autoencoder.resolution is required")
    # has_mid_attn defaulted to False in the old get_model path.
    kwargs['has_mid_attn'] = autoencoder_cfg.get(
        'has_mid_attn', nested.get('has_mid_attn', False))
    kwargs['resolution'] = autoencoder_cfg.get(
        'resolution', nested.get('resolution'))
    for key in _DDCONFIG_KEYS:
        if key in ('has_mid_attn', 'resolution'):
            continue
        if key in autoencoder_cfg:
            kwargs[key] = autoencoder_cfg[key]
        elif key in nested:
            kwargs[key] = nested[key]
    return diffsci.models.nets.autoencoderldm3d.ddconfig(**kwargs)


def _wrap_encode_decode_no_grad(vae_module):
    orig_encode = vae_module.encode
    orig_decode = vae_module.decode

    def encode(x):
        with torch.no_grad():
            return orig_encode(x)

    def decode(z):
        with torch.no_grad():
            return orig_decode(z)

    vae_module.encode = encode
    vae_module.decode = decode
    return vae_module


def load_autoencoder_module(autoencoder_cfg: dict[str, Any] | None):
    """Instantiate a frozen 3D AutoencoderKL from a checkpoint.

    Loads weights with ``strict=False`` so GeomodelAutoencoderKL extras
    (perceptual / well losses) are ignored. Core encoder/decoder keys must
    still match.
    """
    if not autoencoder_cfg:
        return None
    autoencoder_type = autoencoder_cfg['type']
    if autoencoder_type != 'AutoencoderKL':
        raise ValueError(f"Unsupported autoencoder type: {autoencoder_type}")

    checkpoint_path = autoencoder_cfg['checkpoint_path']
    lossconfig = diffsci.models.nets.autoencoderldm3d.lossconfig(
        kl_weight=autoencoder_cfg.get('kl_weight', 1e-4)
    )
    ddconfig = build_ddconfig(autoencoder_cfg)
    embed_dim = int(autoencoder_cfg.get('embed_dim', 4))
    vae_module = diffsci.models.nets.autoencoderldm3d.AutoencoderKL(
        ddconfig=ddconfig,
        lossconfig=lossconfig,
        embed_dim=embed_dim,
    )
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    incompatible = vae_module.load_state_dict(state, strict=False)
    core_missing = [
        k for k in incompatible.missing_keys
        if k.startswith(_CORE_VAE_PREFIXES)
    ]
    if core_missing:
        raise RuntimeError(
            "VAE checkpoint does not match autoencoder ddconfig/embed_dim. "
            f"Missing core keys (first 8): {core_missing[:8]}"
        )
    vae_module.eval()
    for param in vae_module.parameters():
        param.requires_grad_(False)
    return _wrap_encode_decode_no_grad(vae_module)


def get_autoencoder(config: dict[str, Any]):
    """
    Load an autoencoder model based on the provided configuration.

    Args:
        config: Configuration dictionary for the autoencoder

    Returns:
        dict: Dictionary containing the autoencoder model
    """
    return {'autoencoder': load_autoencoder_module(config)}
