import pathlib
import argparse

import torch
import lightning
import lightning.pytorch.callbacks as callbacks
import transformers

import poregen.data
import poregen.models
import poregen.metrics

import numpy as np


CURRENTPATH = pathlib.Path(__file__).parent.absolute()
MAINPATH = CURRENTPATH.parent.parent
DATAPATH = MAINPATH/"saveddata"  # This leads to the data folder
RAWDATAPATH = DATAPATH/"raw"  # This leads to the data folder
MODELSPATH = MAINPATH/"savedmodels"/"experimental"


class Config:
    def __init__(self,
                 savefolderstr: str,
                 validationpath1: str,
                 validationpath2: str,
                 type: str = 'slices',
                 nslices: int = 10,
                 voxel_size: int = 64):
        self.savefolderstr = savefolderstr
        self.validationpath1 = validationpath1
        self.validationpath2 = validationpath2
        self.type = type
        self.nslices = nslices
        self.voxel_size = voxel_size


def main(config):
    type=config.type
    nslices = config.nslices
    voxsize = config.voxel_size
    validation = torch.load(MAINPATH/config.validationpath1)
    validation2 = torch.load(MAINPATH/config.validationpath2)

    if type=='slices':
        # to slices
        indexes = np.random.randint(low=0, high=voxsize, size=nslices)
        validation2d = []
        validation2_2d = []
        for i in range(nslices):
            validation2_2d.append(validation2[..., indexes[i]])
            validation2d.append(validation[..., indexes[i]])
        
        validation2d = torch.cat([torch.tensor(x) for x in validation2d], dim=0)
        validation2_2d = torch.cat([torch.tensor(x) for x in validation2_2d], dim=0)
        print('2D validation shapes:', validation2d.shape, validation2_2d.shape)

        # make report
        metric_creator = poregen.metrics.BinaryPorousImageMetrics()
        metric_creator.set_samples(validation2_2d, validation2d)
        savestr = 'output/bps-reports/' + savefolderstr
        metric_creator.make_report(MAINPATH, savestr, filter_noise=False)
    
    elif type=='voxels':
        print('3D validation shapes:', validation.shape, validation2.shape)
        # make report
        metric_creator = poregen.metrics.BinaryPorousImageMetrics()
        metric_creator.set_samples(validation2, validation)
        savestr = 'output/bps-reports/' + savefolderstr
        metric_creator.make_report(MAINPATH, savestr, filter_noise=False,
                                   dimension=3)
    else:
        raise ValueError('--type argument should be either slices or voxels')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--savefolderstr', type=str, required=True)
    parser.add_argument('--validpath1', type=str, required=True)
    parser.add_argument('--validpath2', type=str, required=True)
    parser.add_argument('--type', type=str, default='slices')
    parser.add_argument('--voxelsize', type=int, default=64)
    args = parser.parse_args()
    savefolderstr = args.savefolderstr
    type = args.type
    voxel_size = args.voxelsize
    validationpath1 = args.validpath1
    validationpath2 = args.validpath2
    
    config = Config(savefolderstr,
                    validationpath1,
                    validationpath2,
                    type=type,
                    voxel_size=voxel_size)
    
    main(config)
