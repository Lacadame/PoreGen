import pathlib
import argparse

import torch
import lightning
from lightning.pytorch.tuner.tuning import Tuner
import lightning.pytorch.callbacks as callbacks
import transformers

import poregen.data
import poregen.models
import poregen.metrics


CURRENTPATH = pathlib.Path(__file__).parent.absolute()
MAINPATH = CURRENTPATH.parent.parent
DATAPATH = MAINPATH/"saveddata"  # This leads to the data folder
RAWDATAPATH = DATAPATH/"raw"  # This leads to the data folder
MODELSPATH = MAINPATH/"savedmodels"/"experimental"


class Config:
    def __init__(self,
                 savefolderstr: str,
                 validationstr: str,
                 nsamples = 1000,
                 shape = [1,128,128],
                 nsteps = 100,
                 ):
        self.savefolderstr = savefolderstr
        self.validationstr = validationstr
        self.nsamples = nsamples
        self.shape = shape
        self.nsteps = nsteps
        

def main(config, module):
    nsamples = config.nsamples
    shape = config.shape
    nsteps = config.nsteps
    savefolderstr = config.savefolderstr
    validationstr = config.validationstr
    
    # generate samples
    sample = module.sample(nsamples, shape, nsteps=nsteps, maximum_batch_size=500)
    torch.save(sample, MAINPATH/'output/bps-saved-samples'/savefolderstr)
    # sample = torch.load(MAINPATH/'output/bps-saved-samples/berea-punetb-mse-2000samples-100steps-edm')
    sample_bin = (sample > 0.5).int()
    sample_bin = sample_bin.float()
    validation = torch.load(MAINPATH/validationstr)

    # make report
    metric_creator = poregen.metrics.BinaryPorousImageMetrics()
    metric_creator.set_samples(sample_bin, validation)
    metric_creator.make_report(MAINPATH/'output/bps-reports', savefolderstr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--savefolderstr', type=str, required=True)
    parser.add_argument('--checkpointpath', type=str, required=True)
    parser.add_argument('--validationstr', type=str,
                        default='output/bps-saved-samples/valid-set-2000samples-berea')
    parser.add_argument('--nsamples', type=int, default=2000)
    parser.add_argument('--shape', type=int, default=[1,128,128])
    parser.add_argument('--nsteps', type=int, default=100)
    datastr = 'eleven_sandstones/Berea_2d25um_binary.raw'
    # savefolderstr = '[pore]-[2024-05-06]-[bps]-[bereapunetb]'
    args = parser.parse_args()
    savefolderstr = args.savefolderstr
    validationstr = args.validationstr
    nsamples = args.nsamples
    shape = args.shape
    nsteps = args.nsteps

    # get model
    checkpoint_path = MAINPATH/args.checkpointpath
    model = poregen.models.PUNetBUncond(64, dropout=0.2)
    moduleconfig = poregen.models.KarrasModuleConfig.from_edm()
    module = poregen.models.KarrasModule.load_from_checkpoint(checkpoint_path,
                                                              model=model,
                                                              config=moduleconfig)
    module.eval();

    config = Config(savefolderstr,
                    validationstr,
                    nsamples,
                    shape,
                    nsteps)
    
    main(config, module)
