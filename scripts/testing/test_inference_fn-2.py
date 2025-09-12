import poregen.data
import poregen.features
import poregen.models
import poregen.trainers


def main():

    cfgpath = "/home/ubuntu/repos/PoreGen/configs/20250108-bps-ldm-bentheimer256-p_cond-aws.yaml"

    # nsamples = 100
    # poregen.trainers.pore_eval(
    #     cfgpath,
    #     'best',
    #     nsamples=nsamples,
    #     maximum_batch_size=1,
    #     device_id=6,
    #     tag='guided-train',
    #     guided=True,
    #     integrator= 'sde',
    #     y='train')

    # poregen.trainers.pore_eval(
    #     cfgpath,
    #     'best',
    #     nsamples=nsamples,
    #     maximum_batch_size=64,
    #     device_id=6)

    stats_folder_path = "/home/ubuntu/repos/PoreGen/savedmodels/experimental/20250108-bps-ldm-bentheimer256-p_cond-aws/stats/stats-100-default-guided-train/model-epoch=066-val_loss=0.069545"
    poregen.trainers.pore_eval_cached(cfgpath,
                                      stats_folder_path,
                                      device_id=6)


if __name__ == "__main__":
    main()
