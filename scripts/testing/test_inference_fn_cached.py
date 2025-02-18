import poregen.data
import poregen.features
import poregen.models
import poregen.trainers


def main():

    stats_folder_paths = [
        # "/home/ubuntu/repos/PoreGen/savedmodels/experimental/20250106-bps-ldm-doddington256-aws/stats/stats-100-karras-256steps/model-epoch=049-val_loss=0.050241",
        # "/home/ubuntu/repos/PoreGen/savedmodels/experimental/20250106-bps-ldm-estaillades256-aws/stats/stats-100-karras-256steps/model-epoch=068-val_loss=0.064220", 
        # "/home/ubuntu/repos/PoreGen/savedmodels/experimental/20250106-bps-ldm-doddington256-aws/stats/stats-100-heun-100/model-epoch=049-val_loss=0.050241",
        # "/home/ubuntu/repos/PoreGen/savedmodels/experimental/20250106-bps-ldm-doddington256-aws/stats/stats-500-heun-100-filtered/model-epoch=049-val_loss=0.050241",
        # "/home/ubuntu/repos/PoreGen/savedmodels/experimental/20250106-bps-ldm-estaillades256-aws/stats/stats-100-heun-100/model-epoch=068-val_loss=0.064220",
        # "/home/ubuntu/repos/PoreGen/savedmodels/experimental/20250106-bps-ldm-estaillades256-aws/stats/stats-500-heun-100-fitered/model-epoch=068-val_loss=0.064220"
        # "/home/ubuntu/repos/PoreGen/savedmodels/experimental/20250108-bps-ldm-bentheimer256-p_cond-aws/stats/stats-100-default-guided/model-epoch=046-val_loss=0.072610",
        # "/home/ubuntu/repos/PoreGen/savedmodels/experimental/20250108-bps-ldm-doddington256-p_cond-aws/stats/stats-100-default-guided/model-epoch=069-val_loss=0.051829",
        # "/home/ubuntu/repos/PoreGen/savedmodels/experimental/20250108-bps-ldm-estaillades256-p_cond-aws/stats/stats-100-default-guided/model-epoch=062-val_loss=0.065075"
    ]

    for stats_folder_path in stats_folder_paths:
        cfgpath = f"{stats_folder_path}/config.yaml"
        print(f"Evaluating {cfgpath}")
        poregen.trainers.pore_eval_cached(
            cfgpath,
            stats_folder_path,
            extractors='3d',
            device_id=6,
            which_stats="both")

if __name__ == "__main__":
    main()
