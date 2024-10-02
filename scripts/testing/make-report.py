import pathlib
import argparse

import torch
import lightning
import lightning.pytorch.callbacks as callbacks
import transformers

import poregen.data
import poregen.models


CURRENTPATH = pathlib.Path(__file__).parent.absolute()
MAINPATH = CURRENTPATH.parent.parent
DATAPATH = MAINPATH/"saveddata"  # This leads to the data folder
RAWDATAPATH = DATAPATH/"raw"  # This leads to the data folder
MODELSPATH = MAINPATH/"savedmodels"/"experimental"


class Config:
    def __init__(self,
                 datastr: str,
                 kindofdata: str,
                 savefolderstr: str,
                 validationpath: str,
                 nsamples: int = 1000,
                 shape: int = [1, 192, 192],
                 nsteps: int = 50,
                 psplit: float = 0.8,
                 max_batch_size: int = 200,
                 voxel_size: int = 64):
        assert kindofdata in ['eleven', 'porespy']
        self.datastr = datastr
        self.kindofdata = kindofdata
        self.savefolderstr = savefolderstr
        self.validationpath = validationpath
        self.nsamples = nsamples
        self.shape = shape
        self.nsteps = nsteps
        self.psplit = psplit
        self.max_batch_size = max_batch_size
        self.voxel_size = voxel_size


def main(config, module):

    sample = module.sample(config.nsamples,
                           config.shape,
                           nsteps=config.nsteps,
                           maximum_batch_size=config.max_batch_size)

    torch.save(sample, module.savefolderstr)
    validation = torch.load(MAINPATH/module.validationpath)

    sample_bin = (sample > 0.5).int()
    sample_bin = sample_bin.float()

    metric_creator = poregen.metrics.BinaryPorousImageMetrics()
    metric_creator.set_samples(sample_bin, validation)
    metric_creator.make_report(MAINPATH, 'output/bps-reports/'/savefolderstr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--kindofdata', type=str,
                        choices=['eleven', 'porespy'],
                        default='eleven')
    parser.add_argument('--savefolderstr', type=str, required=True)
    parser.add_argument('--validationpath', type=str, 
                        default='output/bps-saved-samples/valid-set-2000samples-berea')
    parser.add_argument('--framework', type=str, default='edm',
                        choices=['edm', 'vp'])
    parser.add_argument('--voxelsize', type=int, default=64)
    # savefolderstr = '[pore]-[2024-05-06]-[bps]-[bereapunetb]'
    args = parser.parse_args()
    kindofdata = args.kindofdata
    savefolderstr = args.savefolderstr
    validationpath = args.validationpath
    voxel_size = args.voxelsize

    model = poregen.models.PUNetBUncond(64, dropout=0.2)
    if args.framework == 'edm':
        moduleconfig = poregen.models.KarrasModuleConfig.from_edm()
    elif args.framework == 'vp':
        moduleconfig = poregen.models.KarrasModuleConfig.from_vp()

    checkpoint_path=(MAINPATH/'savedmodels/experimental/[pore]-[2024-06-06]-[bps]-[ldmvae3d-64]/sample-epoch=130-val/rec_loss=0.008684.ckpt')

    lossconfig = poregen.models.nets.autoencoderldm3d.lossconfig(
        kl_weight=1.0)
    ddconfig = poregen.models.nets.autoencoderldm3d.ddconfig(
        ch=32,
        ch_mult=(1,2,2),
        resolution=voxel_size,
        dropout=0.1)

    vae_module = poregen.models.nets.autoencoderldm3d.AutoencoderKL.load_from_checkpoint(
        checkpoint_path, ddconfig=ddconfig, lossconfig=lossconfig
        )
    vae_module.eval()

    # checkpoint = (MAINPATH/'savedmodels/experimental/[pore]-[2024-05-20]-[bps]-[berealdm3d]/sample-epoch=29-valid_loss=7.609305.ckpt')

    # module = poregen.models.KarrasModule.load_from_checkpoint(
    #     checkpoint,
    #     model=model,
    #     config=moduleconfig,
    #     autoencoder=vae_module)

    module = poregen.models.KarrasModule(
        model=model,
        config=moduleconfig,
        autoencoder=vae_module)

    config = Config(datastr,
                    kindofdata,
                    savefolderstr,
                    voxel_size=voxel_size,
                    learning_rate=1e-3,
                    batch_size=100,
                    num_epochs=200)
    
    main(config, module)
