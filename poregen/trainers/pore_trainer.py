import os
import glob
import warnings

import torch
import lightning
import lightning.pytorch.callbacks as pl_callbacks
import lightning.pytorch.loggers as pl_loggers
import transformers

import diffsci.models
from diffsci.models import KarrasModule, KarrasModuleConfig


class PoreTrainer:
    def __init__(self,
                 models,
                 train_config,
                 output_config,
                 data_config=None,
                 load=None,
                 training=True,
                 fast_dev_run=False,
                 load_on_fit=False):
        self.model = models['model']
        self.autoencoder = models.get('autoencoder', None)
        self.train_config = train_config
        self.output_config = output_config
        self.data_config = data_config
        self.load = load
        self.training = training
        self.fast_dev_run = fast_dev_run
        self.load_on_fit = load_on_fit

        self.checkpoint_path = self.get_checkpoint_path()

        # Create KarrasModuleConfig
        karras_config = self.create_karras_config()

        # Create or load KarrasModule
        self.karras_module = self.create_or_load_karras_module(karras_config)

        # TODO: Make this cleaner
        self.karras_module.norm = train_config.get('vae_norm', 1)

        if self.training:
            # Setup Lightning Trainer
            self.setup_lightning_trainer()
            # Set up optimizer
            self.setup_optimizer()

    def create_karras_config(self):
        karras_type = self.train_config.get('karras_type', 'edm')
        if karras_type == 'edm':
            return KarrasModuleConfig.from_edm(**self.train_config.get('karras_config', {}))
        elif karras_type == 'vp':
            return KarrasModuleConfig.from_vp(**self.train_config.get('karras_config', {}))
        elif karras_type == 've':
            return KarrasModuleConfig.from_ve(**self.train_config.get('karras_config', {}))
        else:
            raise ValueError(f"Unsupported Karras config type: {karras_type}")

    def create_or_load_karras_module(self, karras_config):
        if self.load is None or self.load_on_fit:
            # Create a new KarrasModule
            return KarrasModule(
                model=self.model,
                config=karras_config,
                conditional=self.train_config.get('conditional', False),
                masked=self.train_config.get('masked', False),
                autoencoder=self.autoencoder,
                autoencoder_conditional=False
            )
        else:
            # Load from checkpoint
            return KarrasModule.load_from_checkpoint(
                self.checkpoint_path,
                model=self.model,
                config=karras_config,
                conditional=self.train_config.get('conditional', False),
                masked=self.train_config.get('masked', False),
                autoencoder=self.autoencoder,
                autoencoder_conditional=False
            )

    def get_checkpoint_path(self):
        if self.load is None:
            return None
        elif self.load == "best":
            # Find the checkpoint with the lowest val_loss
            checkpoint_dir = os.path.join(self.output_config['folder'], 'checkpoints')
            checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
            # Remove last.ckpt
            # checkpoints = [ckpt for ckpt in checkpoints if not ckpt.endswith('last.ckpt')]
            checkpoints = [ckpt for ckpt in checkpoints if "last" not in ckpt]
            if not checkpoints:
                warnings.warn("No checkpoints found in the specified directory. Starting from scratch.")
                return None
            best_checkpoint = min(checkpoints, key=lambda x: float(x.split('val_loss=')[-1].split('.ckpt')[0]))
            return best_checkpoint
        elif self.load == "latest":
            # Find the latest checkpoint
            checkpoint_dir = os.path.join(self.output_config['folder'], 'checkpoints')
            checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
            if not checkpoints:
                warnings.warn("No checkpoints found in the specified directory. Starting from scratch.")
                return None
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            return latest_checkpoint
        elif self.load == "last":
            # Get the last.ckpt file
            checkpoint_dir = os.path.join(self.output_config['folder'], 'checkpoints')
            checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
            # Remove best.ckpt
            checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('last.ckpt')]
            if not checkpoints:
                warnings.warn("No last checkpoints found in the specified directory. Starting from scratch.")
                return None
            last_checkpoint = checkpoints[0]
            return last_checkpoint
        elif os.path.isfile(self.load):
            # Load the specified checkpoint
            return self.load
        else:
            raise ValueError(f"Invalid checkpoint specification: {self.load}")

    def setup_optimizer(self):
        # Create optimizer
        optimizer_config = self.train_config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adam')
        optimizer_lr = optimizer_config.get('lr', 2*1e-5)
        optimizer_cls = get_optimizer_cls(optimizer_type)
        optimizer_args = optimizer_config.get('args', {})
        optimizer = optimizer_cls(self.karras_module.parameters(),
                                  lr=optimizer_lr,
                                  **optimizer_args)
        # Create scheduler
        scheduler_config = self.train_config.get('scheduler', {})
        if scheduler_config:
            scheduler_type = scheduler_config.get('type', 'cosine')
            if scheduler_type == 'cosine':
                # num_training_steps = len(train_dataloader)*config.num_epochs
                scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                    optimizer=optimizer,
                    num_warmup_steps=scheduler_config.get('num_warmup_steps', 1000),
                    num_training_steps=scheduler_config.get('num_training_steps', 10000),
                    num_cycles=scheduler_config.get('num_cycles', 1)
                )
                self.karras_module.scheduler = scheduler
            elif scheduler_type == 'step':
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer=optimizer,
                    step_size=scheduler_config.get('step_size', 1000),
                    gamma=scheduler_config.get('gamma', 0.1)
                )
            elif scheduler_type == 'constant':
                scheduler = transformers.get_constant_schedule_with_warmup(
                    optimizer=optimizer,
                    num_warmup_steps=scheduler_config.get('num_warmup_steps', 1000)
                )
            else:
                raise NotImplementedError
        else:
            scheduler = None
        self.karras_module.set_optimizer_and_scheduler(optimizer, scheduler)

    def setup_lightning_trainer(self):
        # Callbacks
        output_dir = self.output_config['folder']
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_callback = pl_callbacks.ModelCheckpoint(
            dirpath=os.path.join(output_dir, 'checkpoints'),
            filename='model-{epoch:03d}-{val_loss:.6f}',
            save_top_k=self.train_config.get('save_top_k', 3),
            monitor='val_loss',
            mode='min',
            save_last=True
        )
        lr_monitor = pl_callbacks.LearningRateMonitor(logging_interval='step')

        # Logger
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=output_dir, name='logs')

        # Nan callback
        nan_callback = diffsci.models.callbacks.NanToZeroGradCallback()

        callbacks = [checkpoint_callback, lr_monitor, nan_callback]
        # Trainer
        self.trainer = lightning.Trainer(
            max_epochs=self.train_config.get('num_epochs', 100),
            callbacks=callbacks,
            logger=tb_logger,
            val_check_interval=self.train_config.get('val_check_interval', 1.0),
            precision=self.train_config.get('precision', 32),
            gradient_clip_val=self.train_config.get('gradient_clip_val', None),
            accumulate_grad_batches=self.train_config.get('accumulate_grad_batches', 1),
            strategy=self.train_config.get('strategy', 'auto'),
            accelerator=self.train_config.get('accelerator', 'auto'),
            devices=self.train_config.get('devices', 'auto'),
            fast_dev_run=self.fast_dev_run
        )

    def train(self, datamodule):
        if self.train:
            ckpt_path = self.checkpoint_path if self.load_on_fit else None
            self.trainer.fit(model=self.karras_module,
                             datamodule=datamodule,
                             ckpt_path=ckpt_path)
        else:
            print("Training is disabled. Use 'train=True' to enable training.")

    def test(self, test_loader):
        self.trainer.test(self.karras_module, test_loader)

    def predict(self, predict_loader):
        return self.trainer.predict(self.karras_module, predict_loader)

    def sample(self,
               nsamples,
               shape=None,
               y=None,
               guidance=1.0,
               nsteps=None,
               record_history=False,
               maximum_batch_size=None,
               integrator=None,
               binarize=True,
               return_numpy=True):
        self.karras_module.eval()
        if shape is None:
            shape = self.get_shape_from_data_config()
        if nsteps is None:
            if integrator == 'karras':
                nsteps = 256
            else:
                nsteps = 50
        print('nsteps:', nsteps)
        samples = self.karras_module.sample(
            nsamples,
            shape,
            y,
            guidance,
            nsteps,
            record_history,
            maximum_batch_size,
            integrator
        )
        if binarize:
            if record_history:
                axes = list(range(2, len(samples.shape)))
            else:
                axes = list(range(1, len(samples.shape)))
            samples = samples > samples.mean(axis=axes, keepdim=True)
        if return_numpy:
            samples = samples.cpu().detach().numpy()
        return samples

    def get_shape_from_data_config(self):
        if self.data_config is None:
            raise ValueError("Data config is None. Cannot infer shape.")
        image_size = self.data_config.get('image_size')
        dimension = self.data_config.get('dimension')
        if isinstance(image_size, int):
            base_shape = [image_size] * dimension
        elif hasattr(image_size, '__len__'):
            base_shape = image_size
            assert len(base_shape) == dimension
        shape = list([1] + base_shape)
        return shape


def get_scheduler_cls(scheduler_type):
    # lower the string
    scheduler_type = scheduler_type.lower()
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR
    else:
        raise NotImplementedError


def get_optimizer_cls(optimizer_type):
    # lower the string
    optimizer_type = optimizer_type.lower()
    if optimizer_type == 'adam':
        return torch.optim.Adam
    elif optimizer_type == 'sgd':
        return torch.optim.SGD
    elif optimizer_type == 'adamw':
        return torch.optim.AdamW
    else:
        raise NotImplementedError
