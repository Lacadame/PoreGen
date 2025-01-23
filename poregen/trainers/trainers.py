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


def pore_train(cfg_path,
               data_path=None,
               checkpoint_path=None,
               fast_dev_run=False,
               load_on_fit=False):
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
    basepath = pathlib.Path(cfg_path).parent.parent.parent
    folder = basepath/'savedmodels/experimental'/filename
    cfg['output']['folder'] = folder

    trainer = PoreTrainer(
        models,
        cfg['training'],
        cfg['output'],
        load=checkpoint_path,
        fast_dev_run=fast_dev_run,
        load_on_fit=load_on_fit)
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
    basepath = pathlib.Path(cfg_path).parent.parent.parent
    folder = basepath/'savedmodels/experimental'/filename
    cfg['output']['folder'] = folder

    trainer = PoreVAETrainer(
        cfg['model'],
        cfg['training'],
        cfg['output'],
        cfg['data'],
        load=checkpoint_path,
        fast_dev_run=fast_dev_run)
    trainer.train(datamodule)


def pore_load(cfg_path, checkpoint_path, load_data=False, data_path=None, image_size: int | None = None):
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
        if image_size is not None:
            cfg['data']['image_size'] = image_size  # FIXME: Do a less ugly hack
        datamodule = poregen.data.get_binary_datamodule(data_path, cfg['data'])
        datamodule.setup()
        res['datamodule'] = datamodule
    else:
        res['datamodule'] = None
    return res


def pore_vae_load(cfg_path, checkpoint_path, load_data=False, data_path=None, image_size=None):
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
        if image_size is not None:
            datamodule.cfg['image_size'] = image_size
        datamodule.setup()
        res['datamodule'] = datamodule
    else:
        res['datamodule'] = None
    return res


def pore_eval_cached(cfg_path,
                    stats_folder_path: str | pathlib.Path,
                    extractors: str | list[str] = '3d',
                    extractor_kwargs: dict[str, KwargsType] = {},
                    device_id: int = 0,
                    which_stats: str = "both"):
    """Load cached samples and recalculate only missing properties.

    Args:
        cfg_path: Path to config file to get voxel size
        stats_folder_path: Path to folder containing saved samples and stats
        extractors: List of extractors to calculate
        extractor_kwargs: Kwargs for extractors
        device_id: GPU device id
        which_stats: Which stats to calculate - "both", "generated", or "valid"
    """
    stats_folder = pathlib.Path(stats_folder_path)
    generated_folder = stats_folder / "generated_samples"
    valid_folder = stats_folder / "valid_samples"

    if which_stats not in ["both", "generated", "valid"]:
        raise ValueError("which_stats must be one of: 'both', 'generated', 'valid'")

    # Check if folders exist
    if not generated_folder.exists() or not valid_folder.exists():
        raise FileNotFoundError("Sample folders not found. Run pore_eval first.")

    # Load config to get voxel size
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    voxel_size_um = cfg['data']['voxel_size_um']

    # Load existing stats if they exist
    generated_stats_path = stats_folder / "generated_stats.json"
    valid_stats_path = stats_folder / "valid_stats.json"
    existing_generated_stats = {}
    existing_valid_stats = {}

    if generated_stats_path.exists():
        with open(generated_stats_path, 'r') as f:
            existing_generated_stats = json.load(f)
    if valid_stats_path.exists():
        with open(valid_stats_path, 'r') as f:
            existing_valid_stats = json.load(f)

    # Load samples
    generated_samples = []
    valid_samples = []

    # Load generated samples if needed
    if which_stats in ["both", "generated"]:
        generated_files = sorted(generated_folder.glob("*.npy"))
        if not generated_files:
            raise FileNotFoundError("No generated samples found")
        for file in generated_files:
            generated_samples.append(np.load(file))
        generated_samples = np.stack(generated_samples)

    # Load valid samples if needed
    if which_stats in ["both", "valid"]:
        valid_files = sorted(valid_folder.glob("*.npy"))
        if not valid_files:
            raise FileNotFoundError("No validation samples found")
        for file in valid_files:
            valid_samples.append(np.load(file))
        valid_samples = np.stack(valid_samples)

    # Setup extractors
    if isinstance(extractors, str):
        if extractors == '3d':
            extractors = ['porosimetry_from_voxel',
                         'two_point_correlation_from_voxel',
                         'permeability_from_pnm',
                         'porosity',
                         'effective_porosity',
                         'surface_area_density_from_voxel']
        elif extractors == '2d':
            extractors = ['porosimetry_from_slice',
                         'two_point_correlation_from_slice',
                         'porosity',
                         'effective_porosity',
                         'surface_area_density_from_slice']

    # Add voxel size for permeability
    if 'permeability_from_pnm' in extractors:
        if 'permeability_from_pnm' in extractor_kwargs:
            extractor_kwargs['permeability_from_pnm']['voxel_length'] = voxel_size_um*1e-6
        else:
            extractor_kwargs['permeability_from_pnm'] = {'voxel_length': voxel_size_um*1e-6}

    # First determine which extractors are needed based on missing statistics
    needed_extractors = set()
    
    # Check generated samples for missing statistics
    if which_stats in ["both", "generated"]:
        for i in range(len(generated_samples)):
            sample_id = f"{i+1:05d}"
            if sample_id not in existing_generated_stats:
                needed_extractors.update(extractors)
                break
            for extractor_name in extractors:
                required_keys = poregen.features.feature_extractors.EXTRACTORS_RETURN_KEYS_MAP[extractor_name]
                if not all(key in existing_generated_stats[sample_id] for key in required_keys):
                    needed_extractors.add(extractor_name)

    # Check validation samples for missing statistics 
    if which_stats in ["both", "valid"]:
        for i in range(len(valid_samples)):
            sample_id = f"{i+1:05d}"
            if sample_id not in existing_valid_stats:
                needed_extractors.update(extractors)
                break
            for extractor_name in extractors:
                required_keys = poregen.features.feature_extractors.EXTRACTORS_RETURN_KEYS_MAP[extractor_name]
                if not all(key in existing_valid_stats[sample_id] for key in required_keys):
                    needed_extractors.add(extractor_name)

    # Only create extractors that are needed
    needed_extractors = list(needed_extractors)

    print("NEEDED EXTRACTORS", needed_extractors)
    if needed_extractors:
        needed_extractor_kwargs = {k: extractor_kwargs.get(k, {}) for k in needed_extractors}
        
        # Add voxel size for permeability if needed
        if 'permeability_from_pnm' in needed_extractors:
            if 'permeability_from_pnm' in needed_extractor_kwargs:
                needed_extractor_kwargs['permeability_from_pnm']['voxel_length'] = voxel_size_um*1e-6
            else:
                needed_extractor_kwargs['permeability_from_pnm'] = {'voxel_length': voxel_size_um*1e-6}

        extractor = poregen.features.feature_extractors.make_composite_feature_extractor(
            needed_extractors,
            extractor_kwargs=needed_extractor_kwargs
        )

    # Process generated samples
    if which_stats in ["both", "generated"]:
        generated_stats_all = []
        for i, generated_sample in enumerate(generated_samples):
            sample_id = f"{i+1:05d}"
            
            if sample_id in existing_generated_stats:
                stats = existing_generated_stats[sample_id].copy()
                # Only calculate missing properties
                if needed_extractors:
                    new_stats = extractor(torch.tensor(generated_sample))
                    convert_dict_items_to_numpy(new_stats)
                    stats.update(new_stats)
            else:
                # Calculate all required properties
                stats = extractor(torch.tensor(generated_sample))
                convert_dict_items_to_numpy(stats)
                
            generated_stats_all.append(stats)

        # Save generated stats
        generated_stats_dict = {f"{i+1:05d}": stats for i, stats in enumerate(generated_stats_all)}
        with open(stats_folder / "generated_stats.json", "w") as f:
            json.dump(generated_stats_dict, f, cls=NumpyEncoder)

    # Process validation samples
    if which_stats in ["both", "valid"]:
        valid_stats_all = []
        for i, valid_sample in enumerate(valid_samples):
            sample_id = f"{i+1:05d}"
            
            if sample_id in existing_valid_stats:
                stats = existing_valid_stats[sample_id].copy()
                # Only calculate missing properties
                if needed_extractors:
                    new_stats = extractor(torch.tensor(valid_sample))
                    convert_dict_items_to_numpy(new_stats)
                    stats.update(new_stats)
            else:
                # Calculate all required properties
                stats = extractor(torch.tensor(valid_sample))
                convert_dict_items_to_numpy(stats)
                
            valid_stats_all.append(stats)

        # Save valid stats
        valid_stats_dict = {f"{i+1:05d}": stats for i, stats in enumerate(valid_stats_all)}
        with open(stats_folder / "valid_stats.json", "w") as f:
            json.dump(valid_stats_dict, f, cls=NumpyEncoder)


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
              tag: None | int | str = None,
              device_id: int = 0,
              image_size: None | int = None,
              filter_spectra: bool = False,
              only_porosity: bool = False):
    # FIXME: change shape for be also possibly an int
    loaded = poregen.trainers.pore_load(cfg_path,
                                        checkpoint_path,
                                        load_data=True,
                                        image_size=image_size)
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
        elif isinstance(y, float):
            y = torch.tensor([y])
            # transform y into a dictionary
            y = ({'porosity': y},)
        else:
            pass  # Everything is fine, y is a tensor

    device = torch.device(f'cuda:{device_id}')
    module = loaded['trainer'].karras_module
    module.to(device)

    stats_folder = create_stats_folder_from_checkpoint(
        loaded['trainer'].checkpoint_path,
        nsamples,
        integrator,
        tag=tag
    )
    if guided:
        # I will change the logic a bit to consider
        # the possibility of spectra filtering
        # generated_samples = np.zeros(x_cond.shape)
        generated_samples = []
        for i in range(nsamples):
            generated_sample = loaded['trainer'].sample(
                nsamples=1,
                maximum_batch_size=maximum_batch_size,
                integrator=integrator,
                y=y[i],
                filter_spectra=filter_spectra
            )
            # generated_samples[i] = generated_sample[0]
            if len(generated_sample) > 0:
                generated_samples.append(np.squeeze(generated_sample, 0))
        generated_samples = np.stack(generated_samples, axis=0)
    else:
        if y is not None:
            print('Condition', y)
            y = y[0]
        print(y, 'CONDITION')
        generated_samples = loaded['trainer'].sample(
            nsamples=nsamples,
            maximum_batch_size=maximum_batch_size,
            integrator=integrator,
            y=y,
            filter_spectra=filter_spectra
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

    print(valid_samples.shape, 'VALID SHAPE')
    print(generated_samples.shape, 'GENERATED SHAPE')
    if x_cond is not None:
        x_cond = x_cond[0].cpu().numpy()

    if isinstance(extractors, str):
        if only_porosity:
            extractors = ['porosity']
        else:
            if extractors == '3d':
                extractors = ['porosimetry_from_voxel',
                              'two_point_correlation_from_voxel',
                              'permeability_from_pnm',
                              'porosity',
                              'effective_porosity',
                              'surface_area_density_from_voxel']
            elif extractors == '2d':
                extractors = ['porosimetry_from_slice',
                              'two_point_correlation_from_slice',
                              'porosity',
                              'effective_porosity',
                              'surface_area_density_from_slice']

    # A hack to put the voxel size for permeability_from_pnm
    if 'permeability_from_pnm' in extractors:
        if 'permeability_from_pnm' in extractor_kwargs:
            extractor_kwargs['permeability_from_pnm']['voxel_length'] = voxel_size_um*1e-6
        else:
            extractor_kwargs['permeability_from_pnm'] = {'voxel_length': voxel_size_um*1e-6}
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
                  data_path: str = None,
                  image_size: int = None,
                  tag: None | int | str = None,
                  device_id: int = 0,
                  maximum_batch_size: int = 1):
    loaded = poregen.trainers.pore_vae_load(cfg_path,
                                            checkpoint_path,
                                            load_data=True,
                                            data_path=data_path,
                                            image_size=image_size)
    if isinstance(x, str):
        if x == "train":
            # I'll sample x from the training set
            dataset = loaded['datamodule'].train_dataset
        elif x == "valid":
            # I'll sample x from the validation set
            dataset = loaded['datamodule'].val_dataset
        else:
            raise ValueError("Invalid x argument should be either 'train' or 'valid'")
        x = [dataset[i] for i in range(nsamples)]
        x = torch.stack(x)
    else:
        pass  # Everything is fine, x is a tensor

    device = torch.device(f'cuda:{device_id}')
    vae_module = loaded['trainer'].vae_module
    vae_module.to(device)
    x = x.to(device)

    # Split data into batches
    batches = torch.split(x, maximum_batch_size)
    z_list = []
    x_rec_list = []
    x_rec_bin_list = []

    for i, batch in enumerate(batches):
        batch = batch.to(vae_module.device)
        z_batch = loaded['trainer'].encode(batch)
        x_rec_batch = loaded['trainer'].decode(z_batch)
        z_list.append(z_batch.detach().cpu())
        x_rec_list.append(x_rec_batch.detach().cpu())

    # Binarize x_rec
    axes = list(range(1, len(x_rec_list[0].shape)))
    for i in range(nsamples):
        x_rec_bin_list.append(x_rec_list[i] > x_rec_list[i].mean(axis=axes, keepdim=True))
        x_rec_bin_list[i] = x_rec_bin_list[i].float()

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
        np.save(z_folder / f"{i:05d}_z.npy", z_list[i].detach().cpu().numpy())
        np.save(rec_folder / f"{i:05d}.npy", x_rec_list[i].detach().cpu().numpy())
        np.save(bin_rec_folder / f"{i:05d}.npy", x_rec_bin_list[i].detach().cpu().numpy())

    # Compute reconstruction errors
    l1_rec_error = torch.zeros(nsamples)
    bin_rec_error = torch.zeros(nsamples)

    for i in range(nsamples):
        l1_rec_error[i] = torch.mean(torch.abs(x_rec_list[i] - x.detach().cpu()[i]))
        bin_rec_error[i] = torch.mean(torch.abs(x_rec_bin_list[i] - x.detach().cpu()[i]))

    # Convert the tensors to Python lists before saving
    l1_rec_error_list = l1_rec_error.tolist()
    bin_rec_error_list = bin_rec_error.tolist()

    # Save reconstruction errors
    with open(stats_folder / "reconstruction_errors.json", "w") as f:
        json.dump({"l1_rec_error": l1_rec_error_list, "bin_rec_error": bin_rec_error_list}, f, cls=NumpyEncoder)

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
