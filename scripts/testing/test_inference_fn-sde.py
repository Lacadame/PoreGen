import poregen.data
import poregen.features
import poregen.models
import poregen.trainers


def main():

    cfgpath = "/home/ubuntu/repos/PoreGen/configs/bps/20250611-bps-ldm-bentheimer64.yaml"

    nsamples = 500
    nsteps = 1000
    gammas = [0.02, 0.1, 0.3, 1, 2, 5, 10]
    device_id = 7

    for i, gamma in enumerate(gammas):
        print(f'Running for gamma = {gamma} ({i+1}/{len(gammas)})')
        poregen.trainers.pore_eval(
            cfgpath,
            '/home/ubuntu/repos/PoreGen/savedmodels/experimental/20250611-bps-ldm-bentheimer64/checkpoints/model-epoch=067-val_loss=0.096330.ckpt',
            nsamples=nsamples,
            nsteps=nsteps,
            maximum_batch_size=32,
            device_id=device_id,
            tag=f'gamma={gamma}-nsteps={nsteps}-e=67',
            integrator='sde',
            gamma=gamma,
            extractor_kwargs={
                'permeability_from_pnm': {
                    'disable_parallelization': True
                }
            })

    # poregen.trainers.pore_eval(
    #     cfgpath,
    #     'best',
    #     nsamples=nsamples,
    #     maximum_batch_size=64,
    #     device_id=6)

    # stats_folder_path = (
    #     "/home/ubuntu/repos/PoreGen/savedmodels/experimental/"
    #     "20250108-bps-ldm-bentheimer256-p_cond-aws/stats/"
    #     "stats-100-default-guided-train/model-epoch=066-val_loss=0.069545"
    # )
    # poregen.trainers.pore_eval_cached(cfgpath,
    #                                   stats_folder_path,
    #                                   device_id=6)


if __name__ == "__main__":
    main()
