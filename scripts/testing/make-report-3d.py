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
                 integrator: str,
                 validationpath: str,
                 sample: bool = True,
                 datastr: str = None,
                 latent: bool = True,
                 type: str = 'slices',
                 nslices: int = 10,
                 nvoxels: int = 100,
                 nsteps: int = 50,
                 psplit: float = 0.8,
                 max_batch_size: int = 100,
                 voxel_size: int = 64,
                 filternoise: bool = False):
        self.savefolderstr = savefolderstr
        self.integrator = integrator
        self.validationpath = validationpath
        self.type = type
        self.sample = sample
        self.datastr = datastr
        self.latent = latent
        self.nslices = nslices
        self.nvoxels = nvoxels
        self.nsteps = nsteps
        self.psplit = psplit
        self.max_batch_size = max_batch_size
        self.voxel_size = voxel_size
        self.filternoise = filternoise


def main(config, module):
    nslices = config.nslices
    nvoxel_samples = config.nvoxels
    voxsize = config.voxel_size
    if config.sample:
        print('Sampling!')
        if config.integrator=='karras':
            module.config.noisescheduler.integrator = poregen.models.KarrasIntegrator()
            sample = module.sample(nvoxel_samples,
                                [1, voxsize, voxsize, voxsize],
                                nsteps=256,
                                maximum_batch_size=config.max_batch_size)
        elif config.integrator=='ode':
            sample = module.sample(nvoxel_samples,
                                [1, voxsize, voxsize, voxsize],
                                nsteps=config.nsteps,
                                maximum_batch_size=config.max_batch_size)
        elif config.integrator=='sde':
            module.config.noisescheduler.integrator = poregen.models.EulerMaruyamaIntegrator()
            sample = module.sample(nvoxel_samples,
                                [1, voxsize, voxsize, voxsize],
                                nsteps=256,
                                maximum_batch_size=config.max_batch_size)
        else:
            raise NameError('argument --integrator should be either ode, sde or karras')
        if not config.latent:
            sample = sample + 0.5
        
        torch.save(sample, MAINPATH/'output/bps-saved-samples'/config.savefolderstr)
    else:
        datastr = config.datastr
        if datastr is None:
            raise ValueError('--datastr argument should be passed if --sample=no')
        sample = torch.load(MAINPATH/datastr)
    
    sample_bin = (sample > 0.5).int()
    sample_bin = sample_bin.float()
    validation = torch.load(MAINPATH/config.validationpath)

    if config.type=='slices':
        # to slices
        indexes = np.random.randint(low=0, high=voxsize, size=nslices)
        validation2d = []
        samples2d = []
        for i in range(nslices):
            samples2d.append(sample_bin[..., indexes[i]])
            validation2d.append(validation[..., indexes[i]])
        
        validation2d = torch.cat([torch.tensor(x) for x in validation2d], dim=0)
        samples2d = torch.cat([torch.tensor(x) for x in samples2d], dim=0)

        # make report
        metric_creator = poregen.metrics.BinaryPorousImageMetrics()
        metric_creator.set_samples(samples2d, validation2d)
        savestr = 'output/bps-reports/' + savefolderstr
        metric_creator.make_report(MAINPATH, savestr, filter_noise=config.filternoise)
    
    elif config.type=='voxels':
        # make report
        metric_creator = poregen.metrics.BinaryPorousImageMetrics()
        metric_creator.set_samples(sample_bin, validation)
        savestr = 'output/bps-reports/' + savefolderstr
        metric_creator.make_report(MAINPATH, savestr, filter_noise=config.filternoise,
                                   dimension=3)
    else:
        raise ValueError('--type argument should be either slices or voxels')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--savefolderstr', type=str, required=True)
    parser.add_argument('--model', type=str, default='punetg')
    parser.add_argument('--integrator', type=str, required=True)
    parser.add_argument('--sandstone', type=str, default='berea')
    parser.add_argument('--latent', type=str, default='yes')
    parser.add_argument('--sample', type=str, default='yes')
    parser.add_argument('--diffcheckpoint', type=str, default=None)
    parser.add_argument('--modelchannels', type=int, default=64)
    parser.add_argument('--datastr', type=str, default=None)
    parser.add_argument('--type', type=str, default='slices')
    parser.add_argument('--nvoxels', type=int, default=500)
    parser.add_argument('--filternoise', type=str, default='no')
    parser.add_argument('--framework', type=str, default='edm',
                        choices=['edm', 'vp'])
    parser.add_argument('--voxelsize', type=int, default=64)
    
    args = parser.parse_args()
    sandstone = args.sandstone
    if sandstone=='berea':
        validationpath = 'output/bps-saved-samples/validation3d-500samples-berea64'
        vae_checkpoint = (
             MAINPATH/'savedmodels/production/[pore]-[2024-07-09]-[bps]-[64berea-noatt-autoencoder-1e-4]/sample-epoch=95-val/rec_loss=0.003954.ckpt')
    elif sandstone=='banderabrown':
        validationpath = 'output/bps-saved-samples/validation3d-500samples-banderabrown64'
        vae_checkpoint = (
             MAINPATH/'savedmodels/production/[pore]-[2024-08-13]-[bps]-[64banderabrown-noatt-autoencoder-1e-4]/sample-epoch=104-val/rec_loss=0.005341.ckpt')
    elif sandstone=='estaillades':
        validationpath = 'output/bps-saved-samples/validation3d-500samples-estaillades64'
        # vae_checkpoint = (
        #      MAINPATH/'savedmodels/experimental/[pore]-[2024-09-10]-[bps]-[64estaillades-noatt-autoencoder-1e-4]/sample-epoch=31-val/rec_loss=0.002858.ckpt')
        vae_checkpoint = (
             MAINPATH/'savedmodels/production/[pore]-[2024-07-09]-[bps]-[64berea-noatt-autoencoder-1e-4]/sample-epoch=95-val/rec_loss=0.003954.ckpt')
    else:
        raise ValueError('-sandstone argument should be either berea, banderabrown or estaillades')
    
    savefolderstr = args.savefolderstr
    net = args.model
    integrator = args.integrator
    voxel_size = args.voxelsize
    type = args.type
    datastr = args.datastr
    diffcheckpoint = args.diffcheckpoint
    modelchannels = args.modelchannels

    if args.sample=='yes':
        sample=True
    elif args.sample=='no':
        sample=False                 # argparse seems not to work with bool
    else:
        raise ValueError('--sample argument should be either yes or no')
    
    if args.latent=='yes':
        latent=True
    elif args.latent=='no':
        latent=False                 # argparse seems not to work with bool
    else:
        raise ValueError('--latent argument should be either yes or no')
    
    if args.filternoise=='yes':
        filternoise=True
    elif args.filternoise=='no':
        filternoise=False
    else:
        raise ValueError('--filternoise argument should be either yes or no')
    
    nvoxels = args.nvoxels
    # if nvoxels==100:
    #     validationpath = 'output/bps-saved-samples/validation3d-100samples-berea64'
    # elif nvoxels==500:
    #     validationpath = 'output/bps-saved-samples/validation3d-500samples-berea64'
    # else:
    #     raise ValueError('--nvoxels argument should be either 100 or 500')
    
    if args.framework == 'edm':
            moduleconfig = poregen.models.KarrasModuleConfig.from_edm()
    elif args.framework == 'vp':
        moduleconfig = poregen.models.KarrasModuleConfig.from_vp()

    if latent:
        channels = 4
        max_batch_size = 100
    else:
         channels = 1
         max_batch_size = 50
        
    if voxel_size==32:
        validationpath = 'output/bps-saved-samples/validation3d-1000samples-berea32'
        print('Validation: 1000 volumes of Berea 32!')

    if voxel_size==128:
        validationpath = 'output/bps-saved-samples/validation3d-500samples-berea128'
        max_batch_size = 10
        if sandstone=='banderabrown':
            validationpath = 'output/bps-saved-samples/validation3d-500samples-banderabrown128'

    if voxel_size==256:
        validationpath = 'output/bps-saved-samples/validation3d-100samples-berea256'
        max_batch_size = 1
        if sandstone=='banderabrown':
            validationpath = 'output/bps-saved-samples/validation3d-100samples-banderabrown256'

    if sample:
        if diffcheckpoint is None:
            raise ValueError('--diffcheckpoint argument should be passed if --sample=yes')
        diffcheckpoint = (MAINPATH/diffcheckpoint)
        if net=='punetg':
            netconfig = poregen.models.PUNetGConfig(input_channels=channels,
                                                output_channels=channels,
                                                model_channels=modelchannels,
                                                dimension=3,
                                                number_resnet_attn_block=0,
                                                number_resnet_before_attn_block=3,
                                                number_resnet_after_attn_block=3
                                                )
            model = poregen.models.PUNetG(netconfig)
        elif net=='edm2':
            model = poregen.models.UNet3D(img_resolution=voxel_size,
                                        img_channels=channels,
                                        label_dim=0,
                                        model_channels=32,
                                        dropout=0.1)
        else:
            raise NameError('argument --model should be either edm2 or punetg')
        
        if latent:
            lossconfig = poregen.models.nets.autoencoderldm3d.lossconfig(kl_weight=1e-4)
            ddconfig = poregen.models.nets.autoencoderldm3d.ddconfig(
                resolution=voxel_size,
                has_mid_attn=False)
            vae_module = poregen.models.nets.autoencoderldm3d.AutoencoderKL.load_from_checkpoint(
                vae_checkpoint, ddconfig=ddconfig, lossconfig=lossconfig
                )
            vae_module.eval()

            module = poregen.models.KarrasModule.load_from_checkpoint(
                diffcheckpoint,
                model=model,
                config=moduleconfig,
                autoencoder=vae_module)
            # for normalized latent space
            module.norm = 20
        else:
            module = poregen.models.KarrasModule.load_from_checkpoint(
                diffcheckpoint,
                model=model,
                config=moduleconfig)
    else:
        module = None
    

    config = Config(savefolderstr,
                    integrator,
                    validationpath,
                    sample=sample,
                    datastr=datastr,
                    latent=latent,
                    type=type,
                    voxel_size=voxel_size,
                    filternoise=filternoise,
                    max_batch_size=max_batch_size,
                    nvoxels=nvoxels)
    
    main(config, module)