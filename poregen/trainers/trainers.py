from typing import Any

import yaml
import pathlib
import os

import torch

import poregen.data
import poregen.features
import poregen.models
from .pore_trainer import PoreTrainer
from .pore_vae_trainer import PoreVAETrainer


KwargsType = dict[str, Any]
ConditionType = str | dict[str, torch.Tensor] | torch.Tensor


def pore_train(cfg_path: str | pathlib.Path,
               data_path: str | pathlib.Path | None = None,
               checkpoint_path: str | pathlib.Path | None = None,
               fast_dev_run: bool = False,
               load_on_fit: bool = False
               ) -> PoreTrainer:
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
