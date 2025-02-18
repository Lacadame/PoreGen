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
            'ddconfig_params': ddconfig_params
        }
        return param_dict

    def create_or_load_vae_module(self, param_dict, dim):
        ddconfig_params = param_dict['ddconfig_params']
        kl_weight = param_dict['kl_weight']
        losstype = param_dict['target']

        if dim == 2:
            vae_config = diffsci.models.nets.autoencoderldm2d.ddconfig(**ddconfig_params)
            loss_config = diffsci.models.nets.autoencoderldm2d.lossconfig(
                kl_weight=kl_weight, target=losstype)
            if self.load is None:
                # Create a new vae_module
                return diffsci.models.nets.autoencoderldm2d.AutoencoderKL(vae_config, loss_config)
            else:
                # Load from checkpoint
                checkpoint_path = self.get_checkpoint_path()
                self.checkpoint_path = checkpoint_path
                return diffsci.models.nets.autoencoderldm2d.AutoencoderKL.load_from_checkpoint(
                    checkpoint_path,
                    ddconfig=vae_config,
                    lossconfig=loss_config
                )

        elif dim == 3:
            vae_config = diffsci.models.nets.autoencoderldm3d.ddconfig(**ddconfig_params)
            loss_config = diffsci.models.nets.autoencoderldm3d.lossconfig(
                kl_weight=kl_weight, target=losstype
            )
            if self.load is None:
                # Create a new vae_module
                return diffsci.models.nets.autoencoderldm3d.AutoencoderKL(vae_config, loss_config)
            else:
                # Load from checkpoint
                checkpoint_path = self.get_checkpoint_path()
                self.checkpoint_path = checkpoint_path
                return diffsci.models.nets.autoencoderldm3d.AutoencoderKL.load_from_checkpoint(
                    checkpoint_path,
                    ddconfig=vae_config,
                    lossconfig=loss_config
                )
        else:
            raise ValueError(f"Unsupported dimension: {dim}")

    def get_checkpoint_path(self):
        if self.load == "best":
            # Find the checkpoint with the lowest val_loss
            checkpoint_dir = os.path.join(self.output_config['folder'], 'checkpoints')
            print(checkpoint_dir)
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
            self.trainer.fit(model=self.vae_module, datamodule=datamodule)
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
