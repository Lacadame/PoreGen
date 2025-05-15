import poregen.data
import poregen.features
import poregen.models
import poregen.trainers


def main():

    cfgpath = "/home/danilo/repos/PoreGen/configs/bps/20250108-bps-ldm-bentheimer256-p_cond-aws.yaml"

    nsamples = 100
    # controlled
    poregen.trainers.pore_eval(
        cfgpath,
        'best',
        nsamples=nsamples,
        maximum_batch_size=1,
        device_id=5,
        tag='guided-train',
        guided=True,
        y='train')

    # poregen.trainers.pore_eval(
    #     cfgpath,
    #     'best',
    #     nsamples=nsamples,
    #     maximum_batch_size=1,
    #     device_id=7)

    # for conditional
    # poregen.trainers.pore_eval(
    #     cfgpath,
    #     'best',
    #     nsamples=nsamples,
    #     maximum_batch_size=8,
    #     device_id=7,
    #     tag='0.40',
    #     y=0.40,
    #     only_porosity=True)


if __name__ == "__main__":
    main()
