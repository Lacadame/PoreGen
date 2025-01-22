import poregen.data
import poregen.features
import poregen.models
import poregen.trainers


def main():

    cfgpath = "/home/ubuntu/repos/PoreGen/configs/bps/20250106-bps-ldm-estaillades256-aws.yaml"

    nsamples = 100
    # poregen.trainers.pore_eval(
    #     cfgpath,
    #     'best',
    #     nsamples=nsamples,
    #     maximum_batch_size=1,
    #     device_id=7,
    #     integrator='karras',
    #     tag='guided',
    #     guided=True,
    #     y='valid')

    poregen.trainers.pore_eval(
        cfgpath,
        'best',
        nsamples=nsamples,
        maximum_batch_size=1,
        device_id=7,
        integrator='karras',
        tag='256steps')


if __name__ == "__main__":
    main()
