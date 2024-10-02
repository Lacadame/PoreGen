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


def main(volume_size, nsamples, dataname, savestr, psplit):

    dataname_path = RAWDATAPATH/dataname
    voxel = poregen.data.load_binary_from_eleven_sandstones(dataname_path)
    valid_voxel = voxel[int(1000 * psplit):, :, :]
    transform = poregen.data.get_standard_binary_transforms()
    dataset = poregen.data.VoxelToSubvoxelDataset(valid_voxel, transform=transform, subslice=volume_size)

    validation = dataset[0,:,:].unsqueeze(0)
    for i in range(nsamples-1):
        validation = torch.cat((validation, dataset[i+1,:,:].unsqueeze(0)))
    print(validation.shape)

    # save validation tensor
    savepath = MAINPATH/'output/bps-saved-samples'/savestr
    torch.save(validation, savepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--savestr', type=str, required=True)
    parser.add_argument('--sandstone', type=str, default='berea')
    parser.add_argument('--nvoxels', type=int, default=500)
    parser.add_argument('--voxelsize', type=int, default=64)

    args = parser.parse_args()
    sandstone = args.sandstone
    if sandstone=='berea':
        dataname = "eleven_sandstones/Berea_2d25um_binary.raw"
    elif sandstone=='banderabrown':
        dataname = "eleven_sandstones/BanderaBrown_2d25um_binary.raw"
    elif sandstone=='estaillades':
        dataname = "imperial_college/Estaillades_1000c_3p31136um.raw"
    else:
        raise ValueError('-sandstone argument should be either berea, banderabrown or estaillades')
    
    savestr = args.savestr
    volume_size = args.voxelsize
    nvoxels = args.nvoxels

    if volume_size==256:
        psplit = 0.7
    else:
        psplit = 0.8
    
    main(volume_size=volume_size,
         nsamples=nvoxels,
         dataname=dataname,
         savestr=savestr,
         psplit=psplit)