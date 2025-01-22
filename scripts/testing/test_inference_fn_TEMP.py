import poregen.data
import poregen.features
import poregen.models
import poregen.trainers


def main():

    cfgpath =  "/home/ubuntu/repos/PoreGen/configs/bps/20250106-bps-ldm-estaillades256-aws.yaml"

    poregen.trainers.pore_eval(
        cfgpath,
        'best',
        nsamples=500,
        nsamples_valid=1,
        maximum_batch_size=1,
        device_id=6,
        integrator='heun',
        tag='100',
        filter_spectra=True
    )

    nsamples = 100
    poregen.trainers.pore_eval(
        cfgpath,
        'best',
        nsamples=100,
        nsamples_valid=1,
        maximum_batch_size=1,
        device_id=6,
        integrator='heun',
        tag='100',
        filter_spectra=False
    )

if __name__ == "__main__":
    main()
