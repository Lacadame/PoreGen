from poregen.trainers import evaluators


def main():

    cfg_path = '/home/ubuntu/repos/PoreGen/configs/20250108-bps-ldm-estaillades256-p_cond-aws.yaml'
    stats_folder_path = '/home/ubuntu/repos/PoreGen/savedmodels/experimental/20250108-bps-ldm-estaillades256-p_cond-aws/stats/stats-100-default-guided-train/model-epoch=062-val_loss=0.065075'

    nstride = 4
    max_samples = 100
    batch_size = 100
    evaluators.test_memorization(cfg_path, stats_folder_path, 256, stride=nstride, max_samples=max_samples,batch_size=batch_size)


if __name__ == "__main__":
    main()
