import poregen.data
import poregen.features
import poregen.models
import poregen.trainers


def main():

    cfgpath = "/home/danilo/repos/PoreGen/configs/bps/20241213-bps-ldm-ketton256-por-tpc_cond.yaml"

    nsamples = 100
    # poregen.trainers.pore_eval(
    #     cfgpath,
    #     'best',
    #     y=0.2,
    #     nsamples=nsamples,
    #     maximum_batch_size=1,
    #     device_id=6,
    #     tag='0.2',
    #     only_porosity=True)

    poregen.trainers.pore_eval(
        cfgpath,
        'best',
        y='valid',
        nsamples=nsamples,
        maximum_batch_size=1,
        device_id=6,
        tag='conditional')


if __name__ == "__main__":
    main()
