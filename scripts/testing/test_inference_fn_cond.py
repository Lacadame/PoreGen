import poregen.data
import poregen.features
import poregen.models
import poregen.trainers
import torch


def main():

    cfgpath = "/home/ubuntu/repos/PoreGen/configs/bps/20251205-bps-ldm-bentheimer256-p_cond-interp-fixed.yaml"

    device_id = 7
    min_porosity = 0.15
    max_porosity = 0.3
    porosity_interval = 0.005
    samples_per_porosity = 16

    for porosity in torch.arange(min_porosity, max_porosity, porosity_interval):
        poregen.trainers.pore_eval(
            cfgpath,
            'best',
            y=porosity.item(),
            nsamples=samples_per_porosity,
            maximum_batch_size=1,
            device_id=device_id,
            tag=f'{porosity:.2f}',
            extractors=['porosity'])

    # nsamples = 10
    # poregen.trainers.pore_eval(
    #     cfgpath,
    #     'best',
    #     y=0.4,
    #     nsamples=nsamples,
    #     maximum_batch_size=1,
    #     device_id=6,
    #     tag='0.4-bias2',
    #     integrator='sde',
    #     only_porosity=True)

    # poregen.trainers.pore_eval(
    #     cfgpath,
    #     'best',
    #     y='valid',
    #     nsamples=nsamples,
    #     maximum_batch_size=1,
    #     device_id=6,
    #     tag='conditional')


if __name__ == "__main__":
    main()
