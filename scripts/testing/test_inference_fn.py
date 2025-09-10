import poregen.data
import poregen.features
import poregen.models
import poregen.trainers

import diffsci.models


def main():

    cfgpath = "/home/ubuntu/repos/PoreGen/configs/20250108-bps-ldm-bentheimer256-p_cond-aws.yaml"

    # nsamples = 100
    # poregen.trainers.pore_eval(
    #     cfgpath,
    #     'best',
    #     nsamples=nsamples,
    #     maximum_batch_size=1,
    #     device_id=7,
    #     tag='guided-train',
    #     guided=True,
    #     integrator= diffsci.models.EulerMaruyamaIntegrator(),
    #     y='train')

    # cached
    stats_folder_path = "/home/ubuntu/repos/PoreGen/savedmodels/experimental/20250108-bps-ldm-bentheimer256-p_cond-aws/stats/stats-100-mixed-guided-train/model-epoch=066-val_loss=0.069545"
    poregen.trainers.pore_eval_cached(cfgpath,
                                      stats_folder_path,
                                      device_id=7)

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
    #     maximum_batch_size=1,
    #     device_id=7,
    #     tag='conditional',
    #     y='valid')


if __name__ == "__main__":
    main()
