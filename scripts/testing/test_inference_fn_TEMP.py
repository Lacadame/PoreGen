import poregen.data
import poregen.features
import poregen.models
import poregen.trainers


def main():

    cfgpath1 = "/home/ubuntu/repos/PoreGen/configs/bps/20250108-bps-ldm-estaillades256-p_cond-aws.yaml"
    cfgpath2 = "/home/ubuntu/repos/PoreGen/configs/bps/20250108-bps-ldm-ketton256-p_cond-aws.yaml"

    for cfgpath in [cfgpath1, cfgpath2]:
        poregen.trainers.pore_eval(
            cfgpath,
            'best',
            nsamples=500,
            nsamples_valid=2,
            maximum_batch_size=1,
            device_id=6,
            integrator='heun',
            filter_spectra=True,
            guided=True,
            y='valid'
        )

    #     poregen.trainers.pore_eval(
    #         cfgpath,
    #         'best',
    #         nsamples=2,
    #         nsamples_valid=2,
    #         maximum_batch_size=1,
    #         device_id=6,
    #         integrator='heun',
    #         tag='TEST',
    #         filter_spectra=False,
    #         guided=True,
    #         y='valid'
    #     )

if __name__ == "__main__":
    main()
