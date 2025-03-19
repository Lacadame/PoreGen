import poregen.data
import poregen.features
import poregen.models
import poregen.trainers


def main():

    # poregen.trainers.pore_eval(
    #     '/home/danilo/repos/PoreGen/configs/dfn/20250113-dfn-ldm-doddington64.yaml',
    #     'best',
    #     nsamples=100,
    #     nsamples_valid=100,
    #     maximum_batch_size=1,
    #     device_id=6,
    #     integrator='heun',
    #     tag='100',
    # )

    poregen.trainers.pore_eval(
        '/home/danilo/repos/PoreGen/configs/dfn/20250113-dfn-doddington64.yaml',
        'best',
        nsamples=100,
        nsamples_valid=100,
        maximum_batch_size=1,
        device_id=6,
        integrator='heun',
        tag='100',
    )

if __name__ == "__main__":
    main()
