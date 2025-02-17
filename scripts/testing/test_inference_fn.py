import poregen.data
import poregen.features
import poregen.models
import poregen.trainers


def main():

    cfgpath = "/home/ubuntu/repos/PoreGen/configs/bps/20250106-bps-ldm-doddington256-aws.yaml"

    nsamples = 100
    # poregen.trainers.pore_eval(
    #     cfgpath,
    #     'best',
    #     nsamples=nsamples,
    #     maximum_batch_size=1,
    #     device_id=7,
    #     tag='guided-train',
    #     guided=True,
    #     y='train')

    poregen.trainers.pore_eval(
        cfgpath,
        'best',
        nsamples=nsamples,
        maximum_batch_size=1,
        device_id=7)

    # for conditional
    # poregen.trainers.pore_eval(
    #     cfgpath,
    #     'best',
    #     nsamples=nsamples,
    #     maximum_batch_size=1,
    #     device_id=7,
    #     tag='conditional',
    #     y='valid')


if __name__ == "__main__":
    main()
