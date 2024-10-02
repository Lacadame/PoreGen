import yaml

import poregen.data
import poregen.features
import poregen.models
from .pore_trainer import PoreTrainer


def pore_train(cfg_path, data_path=None, checkpoint_path=None):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    if data_path is None:
        data_path = cfg['data']['path']
    datamodule = poregen.data.get_binary_datamodule(data_path, cfg['data'])
    datamodule.setup()
    models = poregen.models.get_model(cfg['model'])
    trainer = PoreTrainer(
        models,
        cfg['training'],
        cfg['output'],
        load=checkpoint_path)
    trainer.train(datamodule)


def pore_load(cfg_path, checkpoint_path, data_path=None):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    res = dict()
    models = poregen.models.get_model(cfg['model'])
    trainer = PoreTrainer(
        models,
        cfg['training'],
        cfg['output'],
        load=checkpoint_path)
    res['trainer'] = trainer
    if data_path:
        datamodule = poregen.data.get_binary_datamodule(data_path, cfg['data'])
        datamodule.setup()
        res['datamodule'] = datamodule
    else:
        res['datamodule'] = None
    return res
