import os
import glob

import torch
import lightning
import lightning.pytorch.callbacks as pl_callbacks
import lightning.pytorch.loggers as pl_loggers
import transformers

import diffsci.models


class PoreVAETrainer:
    def __init__(self,
                 model,
                 train_config,
                 output_config,
                 data_config=None,
                 load=None,
                 training=True,
                 fast_dev_run=False):
        self.model = model
        self.train_config = train_config
        self.output_config = output_config
        self.data_config = data_config
        self.load = load
        self.training = training
        self.fast_dev_run = fast_dev_run
        self.checkpoint_path = None

        dim = data_config['dimension']
        model_type = model['type']
        assert model_type == 'ldm'

        # get configs
        param_dict = self.get_config_params()

        # Create or load vae_module
        self.vae_module = self.create_or_load_vae_module(param_dict, dim)

        if self.training:
            # Setup Lightning Trainer
            self.setup_lightning_trainer()
            # Set up optimizer
            self.setup_optimizer()

    def get_config_params(self):
        kl_weight = float(self.train_config.get('kl_weight', 1e-4))
        losstype = self.train_config.get(
            'target', diffsci.models.autoencoder.ldmlosses.LPIPSWithDiscriminator)
        ddconfig_params = self.model.get('config', {})
        param_dict = {
            'kl_weight': kl_weight,
            'target': losstype,
            'ddconfig_params': ddconfig_params,
            'embed_dim': int(self.model.get('embed_dim', 4)),
            'geomodel_kwargs': self._geomodel_loss_kwargs(),
        }
        return param_dict

    def _geomodel_loss_kwargs(self):
        extra = self.train_config.get('geomodel_losses') or {}
        if not extra:
            return None
        image_size = self.data_config.get('image_size', [128, 128, 32])
        if isinstance(image_size, int):
            volume_shape = [image_size] * 3
        else:
            volume_shape = [int(v) for v in image_size]
        from poregen.data.geomodel_datasets import parse_well_xy
        well_xy = parse_well_xy(
            self.data_config.get('wells', extra.get('wells', {})),
            index_base=int(self.data_config.get('well_index_base', 1)),
        )
        well_reduction = extra.get(
            'well_reduction', extra.get('well_loss', 'mse'))
        recon_loss = extra.get(
            'recon_loss', extra.get('objective', 'ldm'))
        return {
            'perceptual_weight': float(extra.get('perceptual_weight', 0.0)),
            'well_weight': float(extra.get('well_weight', 0.0)),
            'well_xy': well_xy,
            'volume_shape': volume_shape,
            'perceptual_network': extra.get(
                'perceptual_network', 'resnet18'),
            'slice_ratio': float(extra.get('slice_ratio', 0.2)),
            'well_reduction': well_reduction,
            'recon_loss': recon_loss,
        }

    def create_or_load_vae_module(self, param_dict, dim):
        ddconfig_params = param_dict['ddconfig_params']
        kl_weight = param_dict['kl_weight']
        losstype = param_dict['target']
        embed_dim = param_dict['embed_dim']
        extra = param_dict.get('geomodel_kwargs') or {}

        if dim == 2:
            nets = diffsci.models.nets.autoencoderldm2d
            vae_cls = nets.AutoencoderKL
        elif dim == 3:
            nets = diffsci.models.nets.autoencoderldm3d
            if extra:
                from poregen.models.geomodel_autoencoder import (
                    GeomodelAutoencoderKL)
                vae_cls = GeomodelAutoencoderKL
            else:
                vae_cls = nets.AutoencoderKL
        else:
            raise ValueError(f"Unsupported dimension: {dim}")

        vae_config = nets.ddconfig(**ddconfig_params)
        loss_config = nets.lossconfig(kl_weight=kl_weight, target=losstype)
        init_kwargs = {'embed_dim': embed_dim}
        if extra and dim == 3:
            init_kwargs.update(extra)
        # Training resume must go through Trainer.fit(ckpt_path=...) so that
        # epoch, optimizer, AMP scaler, and callback state are restored.
        # load_from_checkpoint only restores weights and would restart at 0.
        if self.training:
            self.checkpoint_path = self.resolve_fit_ckpt_path()
            return vae_cls(vae_config, loss_config, **init_kwargs)
        if self.load is None:
            return vae_cls(vae_config, loss_config, **init_kwargs)
        checkpoint_path = self.get_checkpoint_path()
        self.checkpoint_path = checkpoint_path
        return vae_cls.load_from_checkpoint(
            checkpoint_path,
            ddconfig=vae_config,
            lossconfig=loss_config,
            **init_kwargs
        )

    def _checkpoint_dir(self):
        return os.path.join(self.output_config['folder'], 'checkpoints')

    def _last_ckpt_path(self):
        return os.path.join(self._checkpoint_dir(), 'last.ckpt')

    def _best_checkpoint_path(self):
        checkpoint_dir = self._checkpoint_dir()
        checkpoints = [
            path for path in glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
            if os.path.basename(path) != 'last.ckpt'
            and 'val_loss=' in os.path.basename(path)
        ]
        if not checkpoints:
            raise ValueError(
                f"No val_loss checkpoints found in {checkpoint_dir}."
            )
        return min(
            checkpoints,
            key=lambda path: float(
                path.split('val_loss=')[-1].split('.ckpt')[0]
            ),
        )

    def resolve_fit_ckpt_path(self):
        """Path for Lightning fit resume, or None to start from scratch."""
        load = self.load
        if load in (None, False, "", "none"):
            return None
        if load is True:
            load = "last"
        if isinstance(load, str) and load.lower() in {"last", "auto"}:
            path = self._last_ckpt_path()
            return path if os.path.isfile(path) else None
        if isinstance(load, str) and load.lower() == "best":
            return self._best_checkpoint_path()
        if os.path.isfile(str(load)):
            return str(load)
        raise ValueError(f"Invalid checkpoint specification: {load}")

    def get_checkpoint_path(self):
        path = self.resolve_fit_ckpt_path()
        if path is None:
            raise ValueError(
                f"Invalid or missing checkpoint specification: {self.load}"
            )
        return path

    def setup_optimizer(self):
        # Create optimizer
        optimizer_config = self.train_config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adam')
        optimizer_lr = optimizer_config.get('lr', 2*1e-5)
        optimizer_cls = get_optimizer_cls(optimizer_type)
        optimizer_args = optimizer_config.get('args', {})
        optimizer = optimizer_cls(self.vae_module.parameters(),
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
                self.vae_module.scheduler = scheduler
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
        self.vae_module.set_optimizer_and_scheduler(optimizer, scheduler)

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
            save_last=True,
            every_n_epochs=self.train_config.get('every_n_epochs', 1),
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
            accumulate_grad_batches=self.train_config.get(
                'accumulate_grad_batches', 1),
            strategy=self.train_config.get('strategy', 'auto'),
            accelerator=self.train_config.get('accelerator', 'auto'),
            devices=self.train_config.get('devices', 'auto'),
            fast_dev_run=self.fast_dev_run
        )

    def train(self, datamodule):
        if self.training:
            if self.checkpoint_path:
                print(f"Resuming training from {self.checkpoint_path}")
            else:
                print("Starting training from scratch "
                      "(no last.ckpt to resume).")
            self.trainer.fit(
                model=self.vae_module,
                datamodule=datamodule,
                ckpt_path=self.checkpoint_path,
            )
        else:
            print("Training is disabled. Use 'train=True' to enable training.")

    def test(self, test_loader):
        self.trainer.test(self.vae_module, test_loader)

    def predict(self, predict_loader):
        return self.trainer.predict(self.vae_module, predict_loader)

    def encode(self, x):
        self.vae_module.eval()
        z = self.vae_module.encode(x)
        return z

    def decode(self, x):
        self.vae_module.eval()
        x_rec = self.vae_module.decode(x)
        return x_rec

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
