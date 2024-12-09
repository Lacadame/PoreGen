import poregen.data
import poregen.features
import poregen.models
import poregen.trainers


def main():

    cfgpath = "/home/danilo/repos/PoreGen/configs/bps/20241202-bps-ldm-bentheimer256.yaml"

    nsamples = 100
    poregen.trainers.pore_eval(
        cfgpath,
        'best',
        nsamples=nsamples,
        maximum_batch_size=1,
        device_id=6,
        integrator='karras',
        tag='256steps')


if __name__ == "__main__":
    main()
