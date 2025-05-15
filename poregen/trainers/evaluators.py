from typing import Any

import shutil
import copy
import yaml
import pathlib
import json
import warnings

import torch
import numpy as np

import poregen.data
import poregen.features
import poregen.models



KwargsType = dict[str, Any]
ConditionType = str | dict[str, torch.Tensor] | torch.Tensor


def pore_eval_cached(cfg_path: str | pathlib.Path,  # noqa: C901
                     stats_folder_path: str | pathlib.Path,
                     extractors: str | list[str] = '3d',
                     extractor_kwargs: dict[str, KwargsType] = {},
                     device_id: int = 0,
                     which_stats: str = "both",
                     default_voxel_size: float = 1.0,
                     max_samples: int = None):
    """Load cached samples and recalculate only missing properties.

    Args:
        cfg_path: Path to config file to get voxel size
        stats_folder_path: Path to folder containing saved samples and stats
        extractors: List of extractors to calculate
        extractor_kwargs: Kwargs for extractors
        device_id: GPU device id
        which_stats: Which stats to calculate - "both", "generated", or "valid"
    """
    # Load config to get voxel size
    try:
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        voxel_size_um = cfg['data']['voxel_size_um']
    except Exception:
        warnings.warn("No voxel size found, using default voxel size")
        voxel_size_um = default_voxel_size

    stats_folder, generated_folder, valid_folder = _validate_and_get_cached_folders(
        stats_folder_path, which_stats
    )

    existing_generated_stats, existing_valid_stats = _load_existing_stats(stats_folder)
    # Load samples

    generated_samples, valid_samples = _load_cached_samples(generated_folder, valid_folder, which_stats, max_samples)

    # Setup extractors

    extractors, extractor_kwargs = _get_extractors(extractors, voxel_size_um, extractor_kwargs)

    needed_extractors = _determine_needed_extractors(
        extractors, which_stats, generated_samples,
        valid_samples, existing_generated_stats,
        existing_valid_stats
    )

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
        _process_and_save_cached_stats(generated_samples, extractor, needed_extractors,
                                       existing_generated_stats, stats_folder, "generated")

    # Process validation samples
    if which_stats in ["both", "valid"]:
        _process_and_save_cached_stats(valid_samples, extractor, needed_extractors,
                                       existing_valid_stats, stats_folder, "valid")


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
    if only_porosity:
        warnings.warn("only_porosity is deprecated, use extractors=['porosity'] instead")
        extractors = ['porosity']
    # FIXME: change shape for be also possibly an int
    loaded = poregen.trainers.pore_load(cfg_path,
                                        checkpoint_path,
                                        load_data=True,
                                        image_size=image_size)
    try:
        voxel_size_um = loaded['datamodule'].voxel_size_um
    except Exception:
        warnings.warn("No voxel size found, using 1.0 um")
        voxel_size_um = 1.0

    if nsamples_valid is None:
        nsamples_valid = nsamples

    x_cond, y = _process_conditions(y, loaded, nsamples, guided)

    device = torch.device(f'cuda:{device_id}')
    module = loaded['trainer'].karras_module
    module.to(device)

    stats_folder = _create_stats_folder_from_checkpoint(
        loaded['trainer'].checkpoint_path,
        nsamples,
        integrator,
        tag=tag
    )

    generated_samples = _generate_samples(
        loaded,
        nsamples,
        maximum_batch_size,
        integrator,
        y,
        filter_spectra,
        guided)

    valid_samples = _get_validation_samples(loaded, nsamples_valid)

    print(valid_samples.shape, 'VALID SHAPE')
    print(generated_samples.shape, 'GENERATED SHAPE')

    extractor = _setup_extractor(extractors, voxel_size_um, extractor_kwargs)

    generated_stats_all, valid_stats_all, cond_stats, cond_stats_all = _calculate_statistics(
        generated_samples, valid_samples, extractor, guided, x_cond)

    _save_results(generated_samples, valid_samples, generated_stats_all, valid_stats_all,
                  cond_stats, cond_stats_all, stats_folder, x_cond, y, guided, cfg_path)


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

    x = _get_vae_input_samples(x, nsamples, loaded)

    device = torch.device(f'cuda:{device_id}')
    vae_module = loaded['trainer'].vae_module.to(device)
    x = x.to(device)

    z_list, x_rec_list, x_rec_bin_list = _process_vae_batches(
        x, vae_module, loaded, maximum_batch_size, nsamples)

    stats_folder = _create_vae_stats_folder_from_checkpoint(
        loaded['trainer'].checkpoint_path,
        nsamples,
        tag=tag)

    _save_vae_results(nsamples, x, z_list, x_rec_list, x_rec_bin_list,
                      stats_folder, cfg_path)


# Auxiliary functions
def _validate_and_get_cached_folders(stats_folder_path, which_stats):
    stats_folder = pathlib.Path(stats_folder_path)
    generated_folder = stats_folder / "generated_samples"
    valid_folder = stats_folder / "valid_samples"

    if which_stats not in ["both", "generated", "valid"]:
        raise ValueError("which_stats must be one of: 'both', 'generated', 'valid'")

    # Check if folders exist
    if not generated_folder.exists() or not valid_folder.exists():
        raise FileNotFoundError("Sample folders not found. Run pore_eval first.")

    return stats_folder, generated_folder, valid_folder


# Load existing stats if they exist
def _load_existing_stats(stats_folder):
    existing_generated_stats = {}
    existing_valid_stats = {}

    generated_stats_path = stats_folder / "generated_stats.json"
    valid_stats_path = stats_folder / "valid_stats.json"

    if generated_stats_path.exists():
        with open(generated_stats_path, 'r') as f:
            existing_generated_stats = json.load(f)
    if valid_stats_path.exists():
        with open(valid_stats_path, 'r') as f:
            existing_valid_stats = json.load(f)

    return existing_generated_stats, existing_valid_stats


def _load_cached_samples(generated_folder, valid_folder, which_stats, max_samples):
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

    if max_samples is not None:
        generated_samples = generated_samples[:max_samples]
        valid_samples = valid_samples[:max_samples]

    return generated_samples, valid_samples


def _determine_needed_extractors(  # noqa: C901
        extractors, which_stats, generated_samples,
        valid_samples, existing_generated_stats,
        existing_valid_stats):
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

    return needed_extractors


def _process_and_save_cached_stats(samples, extractor, needed_extractors,
                                   existing_stats, stats_folder, stats_type):

    stats_all = []
    for i, sample in enumerate(samples):
        sample_id = f"{i+1:05d}"

        if sample_id in existing_stats:
            stats = existing_stats[sample_id].copy()
            # Only calculate missing properties
            if needed_extractors:
                new_stats = extractor(torch.tensor(sample))
                _convert_dict_items_to_numpy(new_stats)
                stats.update(new_stats)
        else:
            # Calculate all required properties
            stats = extractor(torch.tensor(sample))
            _convert_dict_items_to_numpy(stats)

        stats_all.append(stats)

    # Save generated stats
    stats_dict = {f"{i+1:05d}": stats for i, stats in enumerate(stats_all)}
    with open(stats_folder / f"{stats_type}_stats.json", "w") as f:
        json.dump(stats_dict, f, cls=NumpyEncoder)


def _process_conditions(y, loaded, nsamples, guided):
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
    if x_cond is not None:
        x_cond = x_cond[0].cpu().numpy()
    return x_cond, y


def _generate_samples(
            loaded,
            nsamples,
            maximum_batch_size,
            integrator,
            y,
            filter_spectra,
            guided
        ):
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
    return generated_samples


def _get_validation_samples(loaded, nsamples_valid):
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
    return valid_samples


def _setup_extractor(extractors, voxel_size_um, extractor_kwargs):
    extractors, extractor_kwargs = _get_extractors(extractors, voxel_size_um, extractor_kwargs)
    extractor = poregen.features.feature_extractors.make_composite_feature_extractor(
        extractors,
        extractor_kwargs=extractor_kwargs
    )
    return extractor


def _get_extractors(extractors, voxel_size_um, extractor_kwargs):
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
        else:
            extractors = [extractors]
    # A hack to put the voxel size for permeability_from_pnm
    if 'permeability_from_pnm' in extractors:
        if 'permeability_from_pnm' in extractor_kwargs:
            extractor_kwargs['permeability_from_pnm']['voxel_length'] = voxel_size_um*1e-6
        else:
            extractor_kwargs['permeability_from_pnm'] = {'voxel_length': voxel_size_um*1e-6}
    return extractors, extractor_kwargs


def _calculate_statistics(generated_samples, valid_samples, extractor, guided, x_cond):
    generated_stats_all = []
    valid_stats_all = []
    for i, generated_sample in enumerate(generated_samples):
        generated_stats = extractor(torch.tensor(generated_sample))
        _convert_dict_items_to_numpy(generated_stats)
        generated_stats_all.append(generated_stats)
    for i, valid_sample in enumerate(valid_samples):
        valid_stats = extractor(torch.tensor(valid_sample))
        _convert_dict_items_to_numpy(valid_stats)
        valid_stats_all.append(valid_stats)

    cond_stats = None
    cond_stats_all = None
    if x_cond is not None:
        if guided:
            cond_stats_all = []
            for x_cond_i in x_cond:
                cond_stats = extractor(torch.tensor(x_cond_i))
                _convert_dict_items_to_numpy(cond_stats)
                cond_stats_all.append(cond_stats)
        else:
            cond_stats = extractor(torch.tensor(x_cond))
            _convert_dict_items_to_numpy(cond_stats)
    return generated_stats_all, valid_stats_all, cond_stats, cond_stats_all


def _save_results(generated_samples, valid_samples, generated_stats_all, valid_stats_all,
                  cond_stats, cond_stats_all, stats_folder, x_cond, y, guided, cfg_path):

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

    # print(x_cond)
    if x_cond is not None:
        if not guided:          # TODO: figure out how to save the conditions for guided
            y = _convert_condition_to_dict(y)
            generated_stats_dict['condition'] = y
        else:
            if isinstance(y, tuple) or isinstance(y, list):
                y = list(y)
                for i in range(len(y)):
                    y[i] = _convert_condition_to_dict(y[i])
            else:
                raise NotImplementedError("I do not know what to do in this case")
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
            with open(stats_folder / "xcond_stats.json", "w") as f:
                json.dump(cond_stats, f, cls=NumpyEncoder)

    # Save a copy of the config file
    shutil.copy(cfg_path, stats_folder / "config.yaml")


def _convert_condition_to_dict(y):
    if y is None:
        return None
    elif isinstance(y, dict):
        y_copy = copy.deepcopy(y)
        _convert_dict_items_to_numpy(y_copy)
        return y_copy
    else:
        print(y, 'CONDITION')
        y_copy = np.array(y).astype(np.float32)
        return y_copy


def _convert_dict_items_to_numpy(d):
    for k, v in d.items():
        if isinstance(v, dict):
            _convert_dict_items_to_numpy(v)
        else:
            try:
                d[k] = np.array(v).astype(np.float32)
            except Exception:
                pass


def _create_stats_folder_from_checkpoint(
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


def _get_vae_input_samples(x, nsamples, loaded):
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
        x  # Everything is fine, x is a tensor
    return x


def _process_vae_batches(x, vae_module, loaded, maximum_batch_size, nsamples):
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

    return z_list, x_rec_list, x_rec_bin_list


def _save_vae_results(nsamples, x, z_list, x_rec_list, x_rec_bin_list,
                      stats_folder, cfg_path):
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


def _create_vae_stats_folder_from_checkpoint(
        checkpoint_path,
        nsamples,
        tag: None | int = None):

    return _create_stats_folder_from_checkpoint(
        checkpoint_path,
        nsamples,
        integrator='vae',
        tag=tag)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def test_memorization(cfg_path,  # noqa: C901
                      stats_folder_path: str | pathlib.Path,
                      maximum_batch_size: int = 1,
                      mode: None | str = 'nearest-pixel',
                      checkpoint_path: str | pathlib.Path = None,
                      image_size: int | None = None,
                      stride: int = 1):
    # TODO: review this function
    loaded = poregen.trainers.pore_load(cfg_path,
                                        checkpoint_path,
                                        load_data=True,
                                        image_size=image_size)

    stats_folder = pathlib.Path(stats_folder_path)
    generated_folder = stats_folder / "generated_samples"
    valid_folder = stats_folder / "valid_samples"

    generated_samples, valid_samples = _load_cached_samples(generated_folder, valid_folder, which_stats, max_samples)

    print(valid_samples.shape, 'VALID SHAPE')
    print(generated_samples.shape, 'GENERATED SHAPE')

    if mode == 'nearest-pixel':
        metric = poregen.metrics.memorization_metrics.nearest_neighbour
        vae_model = None
    elif mode == 'nearest-latent':
        metric = poregen.metrics.memorization_metrics.nearest_neighbour
        vae_model = loaded['trainer'].vae_module
    else:
        raise NotImplementedError("Choose either 'nearest-pixel' or 'nearest-latent'.")

    target_dataset = poregen.data.VoxelToSubvoxelSequentialDataset(stride=stride)

    nearest = metric(generated_samples,
                     target_dataset,
                     vae_model=vae_model)

    # save nearest stats
    torch.save(nearest, stats_folder/f"nearest-{mode}.pt")
