import yaml
import pathlib
import json

import torch
import numpy as np

import poregen.data
import poregen.features
import poregen.models
from .pore_trainer import PoreTrainer


def pore_train(cfg_path, data_path=None, checkpoint_path=None, fast_dev_run=False):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    if data_path is None:
        data_path = cfg['data']['path']
    datamodule = poregen.data.get_binary_datamodule(data_path, cfg['data'])
    datamodule.setup()
    models = poregen.models.get_model(cfg['model'])
    trainer = PoreTrainer(
        models,
        cfg['training'],
        cfg['output'],
        load=checkpoint_path,
        fast_dev_run=fast_dev_run)
    trainer.train(datamodule)


def pore_load(cfg_path, checkpoint_path, load_data=False, data_path=None):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    res = dict()
    models = poregen.models.get_model(cfg['model'])
    trainer = PoreTrainer(
        models,
        cfg['training'],
        cfg['output'],
        load=checkpoint_path,
        data_config=cfg['data'])
    res['trainer'] = trainer
    if load_data:
        if data_path is None:
            data_path = cfg['data']['path']
        datamodule = poregen.data.get_binary_datamodule(data_path, cfg['data'])
        datamodule.setup()
        res['datamodule'] = datamodule
    else:
        res['datamodule'] = None
    return res


def pore_eval(cfg_path,
              checkpoint_path,
              nsamples: int = 64,
              maximum_batch_size: int = 16,
              integrator: str | None = None,
              extractors: str | list[str] = '3d'):
    trainer = poregen.trainers.pore_load(cfg_path,
                                         checkpoint_path,
                                         load_data=True)

    stats_folder = create_stats_folder_from_checkpoint(
        trainer['trainer'].checkpoint_path,
        nsamples,
        integrator
    )

    generated_samples = trainer['trainer'].sample(
        nsamples=nsamples,
        maximum_batch_size=maximum_batch_size,
        integrator=integrator
    )
    valid_samples = torch.stack([trainer['datamodule'].val_dataset[i] for i in range(nsamples)]).cpu().numpy()

    if isinstance(extractors, str):
        if extractors == '3d':
            extractors = ['porosimetry_from_voxel',
                          'two_point_correlation_from_voxel',
                          'permeability_from_pnm',
                          'porosity',
                          'surface_area_density_from_voxel']
        elif extractors == '2d':
            extractors = ['porosimetry_from_slice',
                          'two_point_correlation_from_slice',
                          'porosity',
                          'surface_area_density_from_slice']

    extractor = poregen.features.feature_extractors.make_composite_feature_extractor(
        extractors
    )

    generated_stats_all = []
    valid_stats_all = []
    for i, generated_sample in enumerate(generated_samples):
        generated_stats = extractor(torch.tensor(generated_sample))
        convert_dict_items_to_numpy(generated_stats)
        generated_stats_all.append(generated_stats)

    for i, valid_sample in enumerate(valid_samples):
        valid_stats = extractor(torch.tensor(valid_sample))
        convert_dict_items_to_numpy(valid_stats)
        valid_stats_all.append(valid_stats)

    # Create subfolders for generated and valid samples
    generated_folder = stats_folder / "generated_samples"
    valid_folder = stats_folder / "valid_samples"
    generated_folder.mkdir(exist_ok=True)
    valid_folder.mkdir(exist_ok=True)

    # Save generated samples
    for i, sample in enumerate(generated_samples):
        np.save(generated_folder / f"{i+1:05d}.npy", sample)

    # Save valid samples
    for i, sample in enumerate(valid_samples):
        np.save(valid_folder / f"{i+1:05d}.npy", sample)

    # Save generated stats
    generated_stats_dict = {f"{i+1:05d}": stats for i, stats in enumerate(generated_stats_all)}
    with open(stats_folder / "generated_stats.json", "w") as f:
        json.dump(generated_stats_dict, f, cls=NumpyEncoder)

    # Save valid stats
    valid_stats_dict = {f"{i+1:05d}": stats for i, stats in enumerate(valid_stats_all)}
    with open(stats_folder / "valid_stats.json", "w") as f:
        json.dump(valid_stats_dict, f, cls=NumpyEncoder)


# Auxiliary functions

def convert_dict_items_to_numpy(d):
    for k, v in d.items():
        if isinstance(v, dict):
            convert_dict_items_to_numpy(v)
        else:
            try:
                d[k] = np.array(v).astype(np.float32)
            except Exception:
                pass


def create_stats_folder_from_checkpoint(
        checkpoint_path,
        nsamples,
        integrator):
    # Convert the checkpoint path to a Path object
    if integrator is None:
        integrator = 'default'
    checkpoint_path = pathlib.Path(checkpoint_path)

    # Extract the checkpoint name (without the .ckpt extension)
    checkpoint_name = checkpoint_path.stem

    # Get the parent directory of 'checkpoints'
    parent_dir = checkpoint_path.parent.parent

    # Stats folder name
    folder_name = f'stats-{nsamples}-{integrator}'

    # Create the new stats folder path
    stats_folder = parent_dir / folder_name / checkpoint_name

    # Create the directory and all necessary parent directories
    stats_folder.mkdir(parents=True, exist_ok=True)

    return pathlib.Path(stats_folder)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)