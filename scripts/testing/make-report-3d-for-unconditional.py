import pathlib
import argparse
from typing import Optional

import torch
import lightning
import lightning.pytorch.callbacks as callbacks
import transformers
import numpy as np

import poregen.data
import poregen.models
import poregen.metrics

# Constants
CURRENT_PATH = pathlib.Path(__file__).parent.absolute()
MAIN_PATH = CURRENT_PATH.parent.parent
DATA_PATH = MAIN_PATH / "saveddata"
RAW_DATA_PATH = DATA_PATH / "raw"
MODELS_PATH = MAIN_PATH / "savedmodels" / "experimental"
OUTPUT_PATH = MAIN_PATH / "output"

class Config:
    def __init__(
        self,
        save_folder: str,
        model_path: str,
        integrator: str,
        sandstone: str,
        sample: bool = True,
        latent: bool = True,
        n_slices: int = 10,
        n_voxels: int = 100,
        n_steps: int = 50,
        p_split: float = 0.8,
        max_batch_size: int = 100,
        voxel_size: int = 64,
        filter_noise: bool = False
    ):
        self.save_folder = save_folder
        self.model_path = model_path
        self.integrator = integrator
        self.sandstone = sandstone
        self.sample = sample
        self.latent = latent
        self.n_slices = n_slices
        self.n_voxels = n_voxels
        self.n_steps = n_steps
        self.p_split = p_split
        self.max_batch_size = max_batch_size
        self.voxel_size = voxel_size
        self.filter_noise = filter_noise


def sample_data(module: poregen.models.KarrasModule, config: Config) -> torch.Tensor:
    if config.integrator == 'karras':
        module.config.noisescheduler.integrator = poregen.models.KarrasIntegrator()
        n_steps = 256
    elif config.integrator == 'ode':
        n_steps = config.n_steps
    elif config.integrator == 'sde':
        module.config.noisescheduler.integrator = poregen.models.EulerMaruyamaIntegrator()
        n_steps = 256
    else:
        raise ValueError(f"Invalid integrator: {config.integrator}. Choose 'ode', 'sde', or 'karras'.")

    sample = module.sample(
        config.n_voxels,
        [1, config.voxel_size, config.voxel_size, config.voxel_size],
        nsteps=n_steps,
        maximum_batch_size=config.max_batch_size
    )

    if not config.latent:
        sample = sample + 0.5

    return sample


def load_or_generate_sample(module: poregen.models.KarrasModule, config: Config) -> torch.Tensor:
    if config.sample:
        print('Sampling!')
        sample = sample_data(module, config)
        torch.save(sample, OUTPUT_PATH / 'bps-saved-samples' / config.save_folder)
    else:
        sample = torch.load(OUTPUT_PATH / 'bps-saved-samples' / config.save_folder)
    
    return sample


def generate_validation(config: Config) -> torch.Tensor:
    sandstone_path = RAW_DATA_PATH / 'eleven_sandstones' / f'{config.sandstone}_2d25um_binary.raw'
    dataset = poregen.data.VoxelToSubvoxelDataset(
        voxel=poregen.data.load_binary_from_eleven_sandstones(sandstone_path),
        subslice=config.voxel_size
    )
    voxels = torch.stack([dataset[i] for i in range(config.n_voxels)], axis=0)
    voxels_mean = voxels.mean(dim=tuple(range(1, voxels.ndim)), keepdims=True)
    voxels_bin = (voxels > voxels_mean).float()
    return voxels_bin


def create_metrics_report(sample: torch.Tensor, validation: torch.Tensor, config: Config):
    sample_mean = sample.mean(dim=tuple(range(1, sample.ndim)), keepdims=True)
    sample_bin = (sample > sample_mean).float()
    metric_creator = poregen.metrics.BinaryPorousImageMetrics()
    metric_creator.set_samples(sample_bin, validation)
    save_str = f'output/bps-reports/{config.save_folder}'
    metric_creator.make_report(MAIN_PATH, save_str, filter_noise=config.filter_noise, dimension=3)


def load_model(config: Config) -> poregen.models.KarrasModule:
    module_config = poregen.models.KarrasModuleConfig.from_edm()

    channels = 4 if config.latent else 1
    
    net_config = poregen.models.PUNetGConfig(
        input_channels=channels,
        output_channels=channels,
        dimension=3,
        number_resnet_attn_block=0,
        number_resnet_before_attn_block=3,
        number_resnet_after_attn_block=3
    )
    model = poregen.models.PUNetG(net_config)

    vae_module = None
    if config.latent:
        vae_checkpoint = MODELS_PATH / '[pore]-[2024-08-13]-[bps]-[64-noatt-autoencoder-1e-4-banderabrown]/sample-epoch=104-val/rec_loss=0.005341.ckpt'
        loss_config = poregen.models.nets.autoencoderldm3d.lossconfig(kl_weight=1e-4)
        dd_config = poregen.models.nets.autoencoderldm3d.ddconfig(resolution=config.voxel_size, has_mid_attn=False)
        vae_module = poregen.models.nets.autoencoderldm3d.AutoencoderKL.load_from_checkpoint(
            vae_checkpoint, ddconfig=dd_config, lossconfig=loss_config
        )
        vae_module.eval()

    diff_checkpoint = MAIN_PATH / config.model_path
    module = poregen.models.KarrasModule.load_from_checkpoint(
        diff_checkpoint,
        model=model,
        config=module_config,
        autoencoder=vae_module
    )
    module.norm = 20  # For normalized latent space

    return module


def main(config: Config):
    module = load_model(config)
    validation = generate_validation(config)
    sample = load_or_generate_sample(module, config)
    create_metrics_report(sample, validation, config)


def parse_arguments() -> Config:
    parser = argparse.ArgumentParser(description="poregen Script")
    parser.add_argument('--save-folder', type=str, required=True, help="Folder to save results")
    parser.add_argument('--model-path', type=str, required=True, help="Relative path of the model to be loaded")
    parser.add_argument('--integrator', type=str, default='ode', choices=['ode', 'sde', 'karras'], help="Integrator type")
    parser.add_argument('--sandstone', type=str, default='Berea', help="Sandstone type")
    parser.add_argument('--latent', type=str, default='yes', choices=['yes', 'no'], help="Use latent space")
    parser.add_argument('--sample', type=str, default='yes', choices=['yes', 'no'], help="Sample data")
    parser.add_argument('--n-voxels', type=int, default=500, help="Number of voxels")
    parser.add_argument('--filter-noise', type=str, default='no', choices=['yes', 'no'], help="Filter noise")
    parser.add_argument('--voxel-size', type=int, default=64, help="Voxel size")

    args = parser.parse_args()
    
    return Config(
        save_folder=args.save_folder,
        model_path=args.model_path,
        integrator=args.integrator,
        sandstone=args.sandstone,
        sample=args.sample == 'yes',
        latent=args.latent == 'yes',
        n_voxels=args.n_voxels,
        filter_noise=args.filter_noise == 'yes',
        voxel_size=args.voxel_size
    )

if __name__ == "__main__":
    config = parse_arguments()
    main(config)