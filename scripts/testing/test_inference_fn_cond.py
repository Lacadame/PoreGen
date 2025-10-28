import poregen.data
import poregen.features
import poregen.models
import poregen.trainers


def main():

    cfgpath = "/home/danilo/repos/PoreGen/configs/20250108-bps-ldm-estaillades256-p_cond-aws.yaml"

    nsamples = 10
    poregen.trainers.pore_eval(
        cfgpath,
        'best',
        y=0.4,
        nsamples=nsamples,
        maximum_batch_size=1,
        device_id=6,
        tag='0.4-bias2',
        integrator='sde',
        only_porosity=True)

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
