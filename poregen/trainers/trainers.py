from typing import Any

import yaml
import pathlib

import torch

import poregen.data
import poregen.features
import poregen.models
from .pore_trainer import PoreTrainer
from .pore_vae_trainer import PoreVAETrainer


KwargsType = dict[str, Any]
ConditionType = str | dict[str, torch.Tensor] | torch.Tensor


def _repo_root_from_cfg(cfg_path: str | pathlib.Path) -> pathlib.Path:
    cfg_path = pathlib.Path(cfg_path).resolve()
    for parent in [cfg_path.parent, *cfg_path.parents]:
        if (parent / 'poregen').is_dir() and (parent / 'scripts').is_dir():
            return parent
    return cfg_path.parent.parent.parent


def _experimental_output_folder(
        cfg: dict[str, Any],
        cfg_path: str | pathlib.Path) -> pathlib.Path:
    cfg.setdefault('output', {})
    existing = cfg['output'].get('folder')
    if existing:
        return pathlib.Path(existing)
    stem = pathlib.Path(cfg_path).stem
    return _repo_root_from_cfg(cfg_path) / 'savedmodels' / 'experimental' / stem


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
    datamodule = poregen.data.get_datamodule(data_path, cfg['data'])
    datamodule.setup()
    models = poregen.models.get_model(cfg['model'])

    cfg['output']['folder'] = _experimental_output_folder(cfg, cfg_path)
    if checkpoint_path is None:
        checkpoint_path = cfg['training'].get('resume_from_checkpoint')
    if not load_on_fit:
        load_on_fit = bool(cfg['training'].get('load_on_fit', False))

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
    datamodule = poregen.data.get_datamodule(data_path, cfg['data'])
    datamodule.setup()
    cfg['output']['folder'] = _experimental_output_folder(cfg, cfg_path)

    if checkpoint_path is None:
        checkpoint_path = cfg['training'].get('resume_from_checkpoint')
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
        datamodule = poregen.data.get_datamodule(data_path, cfg['data'])
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
        datamodule = poregen.data.get_datamodule(data_path, cfg['data'])
        if image_size is not None:
            datamodule.cfg['image_size'] = image_size
        datamodule.setup()
        res['datamodule'] = datamodule
    else:
        res['datamodule'] = None
    return res
