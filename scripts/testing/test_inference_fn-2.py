import poregen.data
import poregen.features
import poregen.models
import poregen.trainers


def main():

    cfgpath = "/home/danilo/repos/PoreGen/configs/dfn/20250113-dfn-ldm-ketton64.yaml"

    nsamples = 500
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
        maximum_batch_size=64,
        device_id=6)


if __name__ == "__main__":
    main()
