from typing import Any
import copy
import shutil

import yaml
import pathlib
import json
import os

import torch
import numpy as np

import poregen.data
import poregen.features
import poregen.models
from .pore_trainer import PoreTrainer
from .pore_vae_trainer import PoreVAETrainer


KwargsType = dict[str, Any]
ConditionType = str | dict[str, torch.Tensor] | torch.Tensor


def pore_train(cfg_path, data_path=None, checkpoint_path=None, fast_dev_run=False):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    if data_path is None:
        data_path = cfg['data']['path']
    datamodule = poregen.data.get_binary_datamodule(data_path, cfg['data'])
    datamodule.setup()
    models = poregen.models.get_model(cfg['model'])

    filename = os.path.basename(cfg_path)
    # Remove yaml extension
    filename = filename.split('.')[0]
    folder = os.path.join('/home/danilo/repos/PoreGen/savedmodels/experimental', filename)
    cfg['output']['folder'] = folder

    trainer = PoreTrainer(
        models,
        cfg['training'],
        cfg['output'],
        load=checkpoint_path,
        fast_dev_run=fast_dev_run)
    trainer.train(datamodule)


def pore_vae_train(cfg_path, data_path=None, checkpoint_path=None, fast_dev_run=False):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    if data_path is None:
        data_path = cfg['data']['path']
    datamodule = poregen.data.get_binary_datamodule(data_path, cfg['data'])
    datamodule.setup()
    # TODO: Infinite loop to check RAM memory usage of datamodule
    filename = os.path.basename(cfg_path)
    # Remove yaml extension
    filename = filename.split('.')[0]
    folder = os.path.join('/home/danilo/repos/PoreGen/savedmodels/experimental', filename)
    cfg['output']['folder'] = folder

    trainer = PoreVAETrainer(
        cfg['model'],
        cfg['training'],
        cfg['output'],
        cfg['data'],
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


def pore_vae_load(cfg_path, checkpoint_path, load_data=False, data_path=None):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    res = dict()
    trainer = PoreVAETrainer(
        cfg['model'],
        cfg['training'],
        cfg['output'],
        cfg['data'],
        load=checkpoint_path)
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


def pore_eval(cfg_path,  # noqa: C901
              checkpoint_path,
              nsamples: int = 64,
              nsamples_valid: int | None = None,
              maximum_batch_size: int = 16,
              integrator: str | None = None,
              extractors: str | list[str] = '3d',
              extractor_kwargs: dict[str, KwargsType] = {},
              y: ConditionType = None,
              guided: bool = False,
              tag: None | int | str = None):
    loaded = poregen.trainers.pore_load(cfg_path,
                                        checkpoint_path,
                                        load_data=True)
    if nsamples_valid is None:
        nsamples_valid = nsamples
    voxel_size_um = loaded['datamodule'].voxel_size_um
    x_cond = None
    if y is not None:
        if isinstance(y, str):
            if guided:
                nconds = nsamples
            else:
                nconds = 1
            if y == "train":
                # I'll sample y from the training set
                x_cond, y = zip(*[loaded['datamodule'].train_dataset[i] for i in range(nconds)])
            elif y == "valid":
                # I'll sample y from the validation set
                x_cond, y = zip(*[loaded['datamodule'].val_dataset[i] for i in range(nconds)])
            else:
                raise ValueError("Invalid y argument should be either 'train' or 'valid'")
            x_cond = torch.stack(x_cond)
        else:
            pass  # Everything is fine, y is a tensor

    stats_folder = create_stats_folder_from_checkpoint(
        loaded['trainer'].checkpoint_path,
        nsamples,
        integrator,
        tag=tag
    )
    if guided:
        generated_samples = np.zeros(x_cond.shape)
        for i in range(nsamples):
            generated_sample = loaded['trainer'].sample(
                nsamples=1,
                maximum_batch_size=maximum_batch_size,
                integrator=integrator,
                y=y[i]
            )
            generated_samples[i] = generated_sample[0]
    else:
        if y is not None:
            y = y[0]
        print(y)
        generated_samples = loaded['trainer'].sample(
            nsamples=nsamples,
            maximum_batch_size=maximum_batch_size,
            integrator=integrator,
            y=y
            )

    # If x_valid exists, I'll add it to the validation samples
    if loaded['datamodule'].val_dataset.feature_extractor is not None:
        # FIXME: This is rather inneficient, because we are calculating feature_extractors twice for each sample.
        # I'm not sure if there is a better way to do this, except for artifically turning off the feature_extractor
        # and then turning it back on.
        # It would be something like
        # old_feature_extractor = loaded['datamodule'].val_dataset.feature_extractor
        # loaded['datamodule'].val_dataset.feature_extractor = None
        # valid_samples = torch.stack([loaded['datamodule'].val_dataset[i] for i in range(nsamples)]).cpu().numpy()
        # loaded['datamodule'].val_dataset.feature_extractor = old_feature_extractor
        # But I'm not sure if this is a good idea.
        valid_samples = [loaded['datamodule'].val_dataset[i][0]
                         for i in range(nsamples_valid)]
    else:
        valid_samples = [loaded['datamodule'].val_dataset[i]
                         for i in range(nsamples_valid)]
    valid_samples = torch.stack(valid_samples).cpu().numpy()

    if x_cond is not None:
        x_cond = x_cond[0].cpu().numpy()

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

    # A hack to put the voxel size for porosimetry_from_pnm
    if 'porosimetry_from_pnm' in extractors:
        if 'porosimetry_from_pnm' in extractor_kwargs:
            extractor_kwargs['porosimetry_from_pnm']['voxel_size'] = voxel_size_um*1e-6
        else:
            extractor_kwargs['porosimetry_from_pnm'] = {'voxel_size': voxel_size_um*1e-6}
    extractor = poregen.features.feature_extractors.make_composite_feature_extractor(
        extractors,
        extractor_kwargs=extractor_kwargs
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

    if x_cond is not None:
        if guided:
            cond_stats_all = []
            for x_cond_i in x_cond:
                cond_stats = extractor(torch.tensor(x_cond_i))
                convert_dict_items_to_numpy(cond_stats)
                cond_stats_all.append(cond_stats)
        else:
            cond_stats = extractor(torch.tensor(x_cond))
            convert_dict_items_to_numpy(cond_stats)
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

    # Save the cond sample
    if x_cond is not None:
        np.save(stats_folder / "xcond.npy", sample)

    # Save generated stats
    generated_stats_dict = {f"{i+1:05d}": stats for i, stats in enumerate(generated_stats_all)}
    if x_cond is not None:
        if not guided:          # TODO: figure out how to save the conditions for guided
            y = convert_condition_to_dict(y)
            generated_stats_dict['condition'] = y
    else:
        y = convert_condition_to_dict(y)
        generated_stats_dict['condition'] = y

    with open(stats_folder / "generated_stats.json", "w") as f:
        json.dump(generated_stats_dict, f, cls=NumpyEncoder)

    # Save valid stats
    valid_stats_dict = {f"{i+1:05d}": stats for i, stats in enumerate(valid_stats_all)}
    with open(stats_folder / "valid_stats.json", "w") as f:
        json.dump(valid_stats_dict, f, cls=NumpyEncoder)

    if x_cond is not None:
        if guided:
            x_cond_stats_dict = {f"{i+1:05d}": stats for i, stats in enumerate(cond_stats_all)}
            with open(stats_folder / "xcond_stats.json", "w") as f:
                json.dump(x_cond_stats_dict, f, cls=NumpyEncoder)
        else:
            print(cond_stats)
            with open(stats_folder / "xcond_stats.json", "w") as f:
                json.dump(cond_stats, f, cls=NumpyEncoder)

    # Save a copy of the config file
    shutil.copy(cfg_path, stats_folder / "config.yaml")


# for testing vae reconstruction
def pore_vae_eval(cfg_path,  # noqa: C901
                  checkpoint_path,
                  nsamples: int = 4,
                  x: str = 'valid',
                  tag: None | int | str = None):
    loaded = poregen.trainers.pore_vae_load(cfg_path,
                                            checkpoint_path,
                                            load_data=True)
    if isinstance(x, str):
        if x == "train":
            # I'll sample x from the training set
            x = [loaded['datamodule'].train_dataset[i] for i in range(nsamples)]
        elif x == "valid":
            # I'll sample x from the validation set
            x = [loaded['datamodule'].val_dataset[i] for i in range(nsamples)]
        else:
            raise ValueError("Invalid x argument should be either 'train' or 'valid'")
        x = torch.stack(x)
    else:
        pass  # Everything is fine, x is a tensor

    z = loaded['trainer'].encode(x)
    x_rec = loaded['trainer'].decode(z)

    # Binarize x_rec
    axes = list(range(1, len(x_rec.shape)))
    x_rec_bin = x_rec > x_rec.mean(axis=axes, keepdim=True)

    stats_folder = create_vae_stats_folder_from_checkpoint(
        loaded['trainer'].checkpoint_path,
        nsamples,
        tag=tag)

    input_folder = stats_folder / "input_samples"
    z_folder = stats_folder / "latent_samples"
    rec_folder = stats_folder / "reconstructed_samples"
    bin_rec_folder = stats_folder / "reconstructed_samples_bin"

    for folder in [input_folder, z_folder, rec_folder, bin_rec_folder]:
        folder.mkdir(exist_ok=True)

    # Save samples
    for i in range(nsamples):
        np.save(input_folder / f"{i:05d}_input.npy", x[i].cpu().numpy())
        np.save(z_folder / f"{i:05d}_z.npy", z[i].cpu().numpy())
        np.save(rec_folder / f"{i:05d}.npy", x_rec[i].cpu().numpy())
        np.save(bin_rec_folder / f"{i:05d}.npy", x_rec_bin[i].cpu().numpy())

    # Compute reconstruction errors
    l1_rec_error = torch.mean(torch.abs(x_rec - x))
    bin_rec_error = torch.mean(torch.abs(x_rec_bin - x))

    # Save reconstruction errors
    with open(stats_folder / "reconstruction_errors.json", "w") as f:
        json.dump({"l1_rec_error": l1_rec_error.item(), "bin_rec_error": bin_rec_error.item()}, f, cls=NumpyEncoder)

    # Save a copy of the config file
    shutil.copy(cfg_path, stats_folder / "config.yaml")


# Auxiliary functions

def convert_condition_to_dict(y):
    if y is None:
        return None
    elif isinstance(y, dict):
        y_copy = copy.deepcopy(y)
        convert_dict_items_to_numpy(y_copy)
        return y_copy
    else:
        y_copy = np.array(y).astype(np.float32)
        return y_copy


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
        integrator,
        tag: None | int = None):
    # Convert the checkpoint path to a Path object
    if integrator is None:
        integrator = 'default'
    checkpoint_path = pathlib.Path(checkpoint_path)

    # Extract the checkpoint name (without the .ckpt extension)
    checkpoint_name = checkpoint_path.stem

    # Get the parent directory of 'checkpoints'
    parent_dir = checkpoint_path.parent.parent

    stats_dir_name = 'stats'
    # Stats folder name
    folder_name = f'stats-{nsamples}-{integrator}'
    if tag is not None:
        if isinstance(tag, int):
            tag = f'{tag:06d}'
        folder_name += f'-{tag}'

    # Create the new stats folder path
    stats_folder = parent_dir / stats_dir_name / folder_name / checkpoint_name

    # Create the directory and all necessary parent directories
    stats_folder.mkdir(parents=True, exist_ok=True)

    return pathlib.Path(stats_folder)


def create_vae_stats_folder_from_checkpoint(
        checkpoint_path,
        nsamples,
        tag: None | int = None):

    return create_stats_folder_from_checkpoint(
        checkpoint_path,
        nsamples,
        integrator='vae',
        tag=tag)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
