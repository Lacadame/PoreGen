import poregen.data
import poregen.features
import poregen.models
import poregen.trainers


def main():

    cfgpath = "/home/danilo/repos/PoreGen/configs/bps/20241205-bps-ldm-bentheimer256-por_cond.yaml"

    nsamples = 100
    poregen.trainers.pore_eval(
        cfgpath,
        'best',
        nsamples=nsamples,
        maximum_batch_size=1,
        device_id=6,
        y='train',
        guided=True)


if __name__ == "__main__":
    main()
