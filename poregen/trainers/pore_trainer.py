import os
import glob

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
                 load=None,
                 training=True,
                 fast_dev_run=False):
        self.model = models['model']
        self.autoencoder = models.get('autoencoder', None)
        self.train_config = train_config
        self.output_config = output_config
        self.load = load
        self.training = training
        self.fast_dev_run = fast_dev_run

        # Create KarrasModuleConfig
        karras_config = self.create_karras_config()

        # Create or load KarrasModule
        self.karras_module = self.create_or_load_karras_module(karras_config)

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
        if self.load is None:
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
            checkpoint_path = self.get_checkpoint_path()
            return KarrasModule.load_from_checkpoint(
                checkpoint_path,
                model=self.model,
                config=karras_config,
                conditional=self.train_config.get('conditional', False),
                masked=self.train_config.get('masked', False),
                autoencoder=self.autoencoder,
                autoencoder_conditional=False
            )

    def get_checkpoint_path(self):
        if self.load == "best":
            # Find the checkpoint with the lowest val_loss
            checkpoint_dir = os.path.join(self.output_config['folder'], 'checkpoints')
            checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
            if not checkpoints:
                raise ValueError("No checkpoints found in the specified directory.")
            best_checkpoint = min(checkpoints, key=lambda x: float(x.split('val_loss=')[-1].split('.ckpt')[0]))
            return best_checkpoint
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
            mode='min'
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
            strategy=self.train_config.get('strategy', 'auto'),
            accelerator=self.train_config.get('accelerator', 'auto'),
            devices=self.train_config.get('devices', 'auto'),
            fast_dev_run=self.fast_dev_run
        )

    def train(self, datamodule):
        if self.train:
            self.trainer.fit(model=self.karras_module, datamodule=datamodule)
        else:
            print("Training is disabled. Use 'train=True' to enable training.")

    def test(self, test_loader):
        self.trainer.test(self.karras_module, test_loader)

    def predict(self, predict_loader):
        return self.trainer.predict(self.karras_module, predict_loader)
    
    def sample(self,
               nsamples,
               shape,
               y=None,
               guidance=1.0,
               nsteps=100,
               record_history=False,
               maximum_batch_size=None,
               integrator=None,
               binarize=True,
               return_numpy=True):
        self.karras_module.eval()
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
            samples = samples > samples.mean(axis=[1, 2, 3, 4], keepdim=True)
        if return_numpy:
            samples = samples.cpu().detach().numpy()
        return samples


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