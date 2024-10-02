import os
import pathlib
import json
import importlib

import numpy as np
import torch
import lightning
import transformers
import lightning.pytorch.callbacks as pl_callbacks
import lightning.pytorch.loggers as pl_loggers

import diffsci.models

import poregen.data
import poregen.features.feature_extractors


"""
DEPRECATED. Use pore_trainer.py instead
"""


class SinglelVoxelDataModule1(lightning.LightningDataModule):
    def __init__(self, data_dir: str | pathlib.Path,
                 batch_size: int = 8,
                 image_size: int | list[int] = 64,
                 training_dataset_size: int = 34560,
                 validation_dataset_size: int = 3840,
                 num_epochs: int = 1,
                 dimension: int = 3,
                 psplit: float = 0.8,
                 voxel_downscale_factor: int = 1,
                 feature_extractor: str | None = None,
                 center: bool = False,
                 invert: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_epochs = num_epochs
        self.psplit = psplit
        self.training_dataset_size = training_dataset_size
        self.validation_dataset_size = validation_dataset_size
        self.dimension = dimension
        self.voxel_downscale_factor = voxel_downscale_factor
        self.feature_extractor = feature_extractor
        self.center = center
        self.invert = invert

    def setup(self, stage: str):
        voxel = poregen.data.load_binary_from_eleven_sandstones(self.data_dir)
        split = int(voxel.shape[0]*self.psplit)
        train_voxel = voxel[:split, :, :]
        valid_voxel = voxel[split:, :, :]
        if self.dimension == 3:
            self.train_dataset = poregen.data.VoxelToSubvoxelDataset(
                train_voxel,
                subslice=[self.image_size, self.image_size, self.image_size],
                dataset_size=self.training_dataset_size,
                voxel_downscale_factor=self.voxel_downscale_factor,
                feature_extractor=self.get_extractor_fn(),
                center=self.center,
                invert=self.invert
            )
            self.valid_dataset = poregen.data.VoxelToSubvoxelDataset(
                valid_voxel,
                subslice=[self.image_size, self.image_size, self.image_size],
                dataset_size=self.validation_dataset_size,
                voxel_downscale_factor=self.voxel_downscale_factor,
                feature_extractor=self.get_extractor_fn(),
                center=self.center,
                invert=self.invert
            )
        elif self.dimension == 2:
            self.train_dataset = poregen.data.VoxelToSlicesDataset(
                train_voxel,
                image_size=[self.image_size, self.image_size],
                dataset_size=self.training_dataset_size,
                voxel_downscale_factor=self.voxel_downscale_factor,
                feature_extractor=self.get_extractor_fn(),
                center=self.center,
                invert=self.invert
            )
            self.valid_dataset = poregen.data.VoxelToSlicesDataset(
                valid_voxel,
                image_size=[self.image_size, self.image_size],
                dataset_size=self.validation_dataset_size,
                voxel_downscale_factor=self.voxel_downscale_factor,
                feature_extractor=self.get_extractor_fn(),
                center=self.center,
                invert=self.invert
            )

    def train_dataloader(self, num_workers='auto'):
        if num_workers == 'auto':
            num_workers = os.cpu_count() - 1
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    def val_dataloader(self, num_workers='auto'):
        if num_workers == 'auto':
            num_workers = os.cpu_count() - 1
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    def effective_batch_size(self, num_devices=1, num_nodes=1):
        return self.batch_size*num_devices*num_nodes

    def num_training_steps(self, num_devices=1, num_nodes=1):
        return (self.training_dataset_size * self.num_epochs //
                self.effective_batch_size(num_devices, num_nodes))

    def num_validation_steps(self, num_devices=1, num_nodes=1):
        return (self.validation_dataset_size //
                self.effective_batch_size(num_devices, num_nodes))

    def get_extractor_fn(self):
        if self.feature_extractor is None:
            return None
        elif isinstance(self.feature_extractor, str):
            return get_function_from_module(
                'poregen.features.feature_extractors',
                self.feature_extractor
            )
        else:
            feature_fns = []
            for feature in self.feature_extractor:
                feature_fn = get_function_from_module(
                    'poregen.features.feature_extractors',
                    feature
                )
                feature_fns.append(feature_fn)
            f = (poregen.
                 features.
                 feature_extractors.
                 make_composite_feature_extractor(feature_fns))
            return f


class SequenceOflVoxelDataModule1(lightning.LightningDataModule):
    def __init__(self,
                 data_dirs: list[int | pathlib.Path],
                 batch_size: int = 8,
                 image_size: int | list[int] = 64,
                 training_dataset_size: int = 34560,
                 validation_dataset_size: int = 3840,
                 num_epochs: int = 1,
                 dimension: int = 3,
                 psplit: float = 0.8,
                 voxel_downscale_factor: int = 1,
                 feature_extractor: str | list[str] | None = None,
                 center: bool = False,
                 invert: bool = False):

        super().__init__()
        self.data_dirs = data_dirs
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_epochs = num_epochs
        self.psplit = psplit
        self.training_dataset_size = training_dataset_size
        self.validation_dataset_size = validation_dataset_size
        self.dimension = dimension
        self.voxel_downscale_factor = voxel_downscale_factor
        self.feature_extractor = feature_extractor
        self.center = center
        self.invert = invert

    def setup(self, stage: str):
        voxels = [poregen.data.load_binary_from_eleven_sandstones(
                    data_dir)
                  for data_dir in self.data_dirs]
        train_voxels = []
        valid_voxels = []
        for voxel in voxels:
            split = int(voxel.shape[0]*self.psplit)
            train_voxel = voxel[:split, :, :]
            valid_voxel = voxel[split:, :, :]
            train_voxels.append(train_voxel)
            valid_voxels.append(valid_voxel)
        if self.dimension == 3:
            self.train_dataset = (
                poregen.data.SequenceOfVoxelsToSubvoxelDataset(
                    train_voxels,
                    subslice=[self.image_size,
                              self.image_size,
                              self.image_size],
                    dataset_size=self.training_dataset_size,
                    voxel_downscale_factor=self.voxel_downscale_factor,
                    feature_extractor=self.get_extractor_fn(),
                    center=self.center,
                    invert=self.invert
                )
            )
            self.valid_dataset = (
                poregen.data.SequenceOfVoxelsToSubvoxelDataset(
                    valid_voxels,
                    subslice=[self.image_size,
                              self.image_size,
                              self.image_size],
                    dataset_size=self.validation_dataset_size,
                    voxel_downscale_factor=self.voxel_downscale_factor,
                    feature_extractor=self.get_extractor_fn(),
                    center=self.center,
                    invert=self.invert
                )
            )
        elif self.dimension == 2:
            self.train_dataset = (
                poregen.data.SequenceOfVoxelsToSlicesDataset(
                    train_voxels,
                    image_size=[self.image_size, self.image_size],
                    dataset_size=self.training_dataset_size,
                    voxel_downscale_factor=self.voxel_downscale_factor,
                    feature_extractor=self.get_extractor_fn(),
                    center=self.center,
                    invert=self.invert
                )
            )
            self.valid_dataset = (
                poregen.data.SequenceOfVoxelsToSlicesDataset(
                    valid_voxels,
                    image_size=[self.image_size, self.image_size],
                    dataset_size=self.validation_dataset_size,
                    voxel_downscale_factor=self.voxel_downscale_factor,
                    feature_extractor=self.get_extractor_fn(),
                    center=self.center,
                    invert=self.invert
                )
            )

    def train_dataloader(self, num_workers='auto'):
        if num_workers == 'auto':
            num_workers = os.cpu_count() - 1
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    def val_dataloader(self, num_workers='auto'):
        if num_workers == 'auto':
            num_workers = os.cpu_count() - 1
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    def effective_batch_size(self, num_devices=1, num_nodes=1):
        return self.batch_size*num_devices*num_nodes

    def num_training_steps(self, num_devices=1, num_nodes=1):
        return (self.training_dataset_size * self.num_epochs //
                self.effective_batch_size(num_devices, num_nodes))

    def num_validation_steps(self, num_devices=1, num_nodes=1):
        return (self.validation_dataset_size //
                self.effective_batch_size(num_devices, num_nodes))

    def get_extractor_fn(self):
        if self.feature_extractor is None:
            return None
        elif isinstance(self.feature_extractor, str):
            return get_function_from_module(
                'poregen.features.feature_extractors',
                self.feature_extractor
            )
        else:
            feature_fns = []
            for feature in self.feature_extractor:
                feature_fn = get_function_from_module(
                    'poregen.features.feature_extractors',
                    feature
                )
                feature_fns.append(feature_fn)
            f = (poregen.
                 features.
                 feature_extractors.
                 make_composite_feature_extractor(feature_fns))
            return f


class BinaryVoxelTrainerConfig1(object):
    def __init__(self,
                 path: str | pathlib.Path | list[str | pathlib.Path],
                 savepath: str | pathlib.Path,
                 image_size: int = 64,
                 voxel_downscale_factor: int = 1,
                 mean_learning_rate: int = 2*1e-5,
                 batch_size: int = 8,
                 training_dataset_size: int = 34560,
                 validation_dataset_size: int = 3840,
                 num_epochs: int = 1,
                 warmup_step_percentage: int = 0.05,
                 val_check_interval: int | float = 1.0,
                 precision: int = 16,
                 gradient_clip_val: float = 0.5,
                 save_top_k: int = 3,
                 dimension: int = 3,
                 lr_scheduler_interval: str = "step",
                 num_lr_cycles: int = 1,
                 learning_scheduler: str = 'constant',
                 feature_extractor: str | list[str] | None = None,
                 psplit: float = 0.8,
                 center: bool = False,
                 invert: bool = False):
        self.path = path
        self.savepath = pathlib.Path(savepath)
        self.image_size = image_size
        self.voxel_downscale_factor = voxel_downscale_factor
        self.mean_learning_rate = mean_learning_rate
        self.batch_size = batch_size
        self.training_dataset_size = training_dataset_size
        self.validation_dataset_size = validation_dataset_size
        self.num_epochs = num_epochs
        self.warmup_step_percentage = warmup_step_percentage
        self.val_check_interval = val_check_interval
        self.precision = precision
        self.gradient_clip_val = gradient_clip_val
        self.save_top_k = save_top_k
        self.dimension = dimension
        self.lr_scheduler_interval = lr_scheduler_interval
        self.num_lr_cycles = num_lr_cycles
        self.learning_scheduler = learning_scheduler
        self.feature_extractor = feature_extractor
        self.psplit = psplit
        self.center = center
        self.invert = invert

    def export_description(self):
        return dict(
            name='UnconditionalBinaryVoxelTrainerConfig1',
            path=self.path,
            savepath=self.savepath,
            image_size=self.image_size,
            voxel_downscale_factor=self.voxel_downscale_factor,
            mean_learning_rate=self.mean_learning_rate,
            batch_size=self.batch_size,
            training_dataset_size=self.training_dataset_size,
            validation_dataset_size=self.validation_dataset_size,
            warmup_step_percentage=self.warmup_step_percentage,
            val_check_interval=self.val_check_interval,
            precision=self.precision,
            gradient_clip_val=self.gradient_clip_val,
            save_top_k=self.save_top_k,
            dimension=self.dimension,
            lr_scheduler_interval=self.lr_scheduler_interval,
            num_lr_cycles=self.num_lr_cycles,
            learning_scheduler=self.learning_scheduler,
            feature_extractor=self.feature_extractor,
            psplit=self.psplit,
            center=self.center,
            invert=self.invert
        )

    def is_sequence(self):
        return is_sequence(self.path)


def train_binary_voxel_1(
        model: torch.nn.Module,
        trainerconfig: BinaryVoxelTrainerConfig1,
        conditional: bool = False,
        moduleconfig: str | diffsci.models.KarrasModuleConfig = 'default',
        fast_dev_run: bool = False,
        autoencoder: None | torch.nn.Module = None,
        autoencoder_conditional: bool = False,
        strategy: str = "auto",
        accelerator: str = "auto",
        devices: str | int = "auto"):

    # Create the args dictionary
    args = {
        'name': 'train_unconditional_single_binary_voxel_1',
        'model': model.export_description(),
        'trainerconfig': trainerconfig.export_description(),
        'strategy': strategy,
    }

    # Load the module
    if moduleconfig == 'default':
        moduleconfig = diffsci.models.KarrasModuleConfig.from_edm()
    else:
        assert isinstance(moduleconfig, diffsci.models.KarrasModuleConfig)
    module = diffsci.models.KarrasModule(
        model,
        moduleconfig,
        conditional=conditional,
        autoencoder=autoencoder,
        autoencoder_conditional=autoencoder_conditional
    )
    args['module'] = module.export_description()

    # Load the data module
    if trainerconfig.is_sequence():
        datamodule = SequenceOflVoxelDataModule1(
            data_dirs=trainerconfig.path,
            image_size=trainerconfig.image_size,
            batch_size=trainerconfig.batch_size,
            training_dataset_size=trainerconfig.training_dataset_size,
            validation_dataset_size=trainerconfig.validation_dataset_size,
            dimension=trainerconfig.dimension,
            num_epochs=trainerconfig.num_epochs,
            voxel_downscale_factor=trainerconfig.voxel_downscale_factor,
            feature_extractor=trainerconfig.feature_extractor,
            psplit=trainerconfig.psplit,
            center=trainerconfig.center,
            invert=trainerconfig.invert
        )
    else:
        datamodule = SinglelVoxelDataModule1(
            data_dir=trainerconfig.path,
            image_size=trainerconfig.image_size,
            batch_size=trainerconfig.batch_size,
            training_dataset_size=trainerconfig.training_dataset_size,
            validation_dataset_size=trainerconfig.validation_dataset_size,
            dimension=trainerconfig.dimension,
            num_epochs=trainerconfig.num_epochs,
            voxel_downscale_factor=trainerconfig.voxel_downscale_factor,
            feature_extractor=trainerconfig.feature_extractor,
            psplit=trainerconfig.psplit,
            center=trainerconfig.center,
            invert=trainerconfig.invert
        )

    # Set the learning rate and optimizer
    assert (trainerconfig.learning_scheduler in
            ['cosine', 'constant'])
    if trainerconfig.learning_scheduler == 'cosine':
        lr = 2*trainerconfig.mean_learning_rate
    elif trainerconfig.learning_scheduler == 'constant':
        lr = trainerconfig.mean_learning_rate
    module.optimizer = torch.optim.AdamW(module.parameters(),
                                         lr=lr)
    device_count = torch.cuda.device_count() if devices == "auto" else devices
    nsteps = datamodule.num_training_steps(device_count)
    num_warmup_steps = int(trainerconfig.warmup_step_percentage*nsteps)

    if trainerconfig.learning_scheduler == 'cosine':
        module.lr_scheduler = (
            transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                module.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=nsteps,
                num_cycles=trainerconfig.num_lr_cycles)
        )
    elif trainerconfig.learning_scheduler == 'constant':
        module.lr_scheduler = (
            transformers.get_constant_schedule_with_warmup(
                module.optimizer,
                num_warmup_steps=num_warmup_steps)
        )
    module.lr_scheduler_interval = trainerconfig.lr_scheduler_interval

    # Part of the code to save the model structurally
    # Set the checkpoint to be named
    savepath = pathlib.Path(trainerconfig.savepath)
    savepath.mkdir(parents=True, exist_ok=True)

    # List the folders in the savepath numbered as 0000, 0001, ...,
    # and set the checkpoint_callback to save to the next folder.
    # If the savepath is empty, the checkpoint_callback will save to
    # the folder 0000.
    folders = [f for f in savepath.iterdir() if f.is_dir()]
    # Filter out the non-integer folders
    folders = [f for f in folders if f.name.isdigit()]
    # Sort the folders by their integer value
    folders = sorted(folders, key=lambda f: int(f.name))
    # Get the last folder
    if len(folders) == 0:
        next_folder = savepath / '0000'
    else:
        last_folder = folders[-1]
        # Get the next folder
        next_folder = savepath / f'{int(last_folder.name)+1:04d}'

    args = convert_dictionary_to_json_serializable(args)
    if not fast_dev_run:  # If fast_dev_run, don't create folders
        next_folder.mkdir(parents=True, exist_ok=True)

        # Save the file config_incomplete.json in the next folder
        with open(next_folder/'config_incomplete.json', 'w') as f:
            json.dump(args, f, indent=4)
    # End that part of the code

    checkpoint_callback = pl_callbacks.ModelCheckpoint(
        monitor='valid_loss',
        dirpath=trainerconfig.savepath/next_folder,
        filename='sample-{epoch:02d}-{step:010d}-{valid_loss:.6f}',
        save_top_k=trainerconfig.save_top_k,
        mode='min',
    )
    lr_callback = pl_callbacks.LearningRateMonitor(logging_interval='step')
    callbacks = [checkpoint_callback,
                 lr_callback]

    logger = pl_loggers.TensorBoardLogger(
        save_dir=trainerconfig.savepath,
        version=f'version_{next_folder.name}'
    )

    trainer = lightning.Trainer(
        max_epochs=trainerconfig.num_epochs,
        val_check_interval=trainerconfig.val_check_interval,
        callbacks=callbacks,
        precision=trainerconfig.precision,
        gradient_clip_val=trainerconfig.gradient_clip_val,
        fast_dev_run=fast_dev_run,
        logger=logger,
        strategy=strategy,
        accelerator=accelerator,
        devices=devices)
    trainer.fit(model=module,
                datamodule=datamodule)

    args['logged_metrics'] = trainer.logged_metrics
    args['checkpoint_callback'] = checkpoint_callback.best_model_path

    # Save the file config_complete.json in the next folder
    if not fast_dev_run:
        convert_dictionary_to_json_serializable(args)
        with open(next_folder/'config_complete.json', 'w') as f:
            json.dump(args, f, indent=4)
        # Delete the file config_incomplete.json
        # os.remove(next_folder/'config_incomplete.json')
    return args


# Helpers
def is_sequence(obj):
    t = type(obj)
    return hasattr(t, '__len__') and hasattr(t, '__getitem__')


def convert_dictionary_to_json_serializable(d):
    for k, v in d.items():
        if (isinstance(v, pathlib.Path) or
                isinstance(v, pathlib.PosixPath)):
            v = str(v)
            d[k] = v
        elif isinstance(v, torch.Tensor):
            d[k] = v.detach().cpu().numpy().tolist()
        elif isinstance(v, np.ndarray):
            d[k] = v.tolist()
        elif isinstance(v, dict):
            d[k] = convert_dictionary_to_json_serializable(v)
    return d


def get_function_from_module(module_name, function_name):
    # Import the module based on the module name string
    module = importlib.import_module(module_name)
    # Retrieve the function by its name
    func = getattr(module, function_name)
    return func
