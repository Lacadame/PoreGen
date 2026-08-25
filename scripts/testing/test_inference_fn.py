import poregen.data
import poregen.features
import poregen.models
import poregen.trainers


def main():

    cfgpath = "/home/ubuntu/repos/PoreGen/configs/bps/20251205-bps-ldm-bentheimer256-p_cond-interp-fixed.yaml"

    nsamples = 100
    device_id = 6
    poregen.trainers.pore_eval(
        cfgpath,
        'best',
        nsamples=nsamples,
        maximum_batch_size=1,
        device_id=device_id,
        tag='guided-valid',
        guided=True,
        y='valid',
        val_hidden_interval_mode='full')

    # cached
    # stats_folder_path = "/home/ubuntu/repos/PoreGen/savedmodels/experimental/20250108-bps-ldm-bentheimer256-p_cond-aws/stats/stats-100-mixed-guided-train/model-epoch=066-val_loss=0.069545"
    # poregen.trainers.pore_eval_cached(cfgpath,
    #                                   stats_folder_path,
    #                                   device_id=device_id)

    # poregen.trainers.pore_eval(
    #     cfgpath,
    #     'best',
    #     nsamples=nsamples,
    #     maximum_batch_size=1,
    #     device_id=device_id,
    #     tag='uncond')

    # for conditional
    # poregen.trainers.pore_eval(
    #     cfgpath,
    #     'best',
    #     nsamples=nsamples,
    #     maximum_batch_size=1,
    #     device_id=device_id,
    #     tag='conditional',
    #     y='valid')


if __name__ == "__main__":
    main()
