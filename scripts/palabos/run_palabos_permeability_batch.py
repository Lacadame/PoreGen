import pathlib
import os


if __name__ == "__main__":
    folders = [
        # Bentheimer models
        # "/home/ubuntu/repos/PoreGen/savedmodels/experimental/20241107-bps-ldm-bentheimer256/stats/stats-100-default/model-epoch=049-val_loss=0.054713/valid_samples",
        # "/home/ubuntu/repos/PoreGen/savedmodels/experimental/20250108-bps-ldm-bentheimer256-p_cond-aws/stats/stats-100-default-guided-train/model-epoch=066-val_loss=0.069545/generated_samples",
        # "/home/ubuntu/repos/PoreGen/savedmodels/experimental/20250113-bps-ldm-bentheimer256-p_tpc_cond-aws/stats/stats-100-default-guided-train/model-epoch=063-val_loss=0.068560/generated_samples",

        # Doddington models
        # "/home/ubuntu/repos/PoreGen/savedmodels/experimental/20241107-bps-ldm-doddington256/stats/stats-100-default/model-epoch=057-val_loss=0.035366/valid_samples",
        # "/home/ubuntu/repos/PoreGen/savedmodels/experimental/20250108-bps-ldm-doddington256-p_cond-aws/stats/stats-100-default-guided-train/model-epoch=069-val_loss=0.051829/generated_samples",
        # "/home/ubuntu/repos/Po/reGen/savedmodels/experimental/20250113-bps-ldm-doddington256-p_tpc_cond-aws/stats/stats-100-default-guided-train/model-epoch=067-val_loss=0.051352/generated_samples",

        # Estaillades models
        # "/home/ubuntu/repos/PoreGen/savedmodels/experimental/20241107-bps-ldm-estaillades256/stats/stats-100-default/model-epoch=038-val_loss=0.053073/valid_samples",
        # "/home/ubuntu/repos/PoreGen/savedmodels/experimental/20250108-bps-ldm-estaillades256-p_cond-aws/stats/stats-100-default-guided-train/model-epoch=062-val_loss=0.065075/generated_samples",
        # "/home/ubuntu/repos/PoreGen/savedmodels/experimental/20250113-bps-ldm-estaillades256-p_tpc_cond-aws/stats/stats-100-default-guided-train/model-epoch=068-val_loss=0.065273/generated_samples",

        # Ketton models
        # "/home/ubuntu/repos/PoreGen/savedmodels/experimental/20241107-bps-ldm-ketton256/stats/stats-100-default/model-epoch=048-val_loss=0.025774/valid_samples",
        # "/home/ubuntu/repos/PoreGen/savedmodels/experimental/20250108-bps-ldm-ketton256-p_cond-aws/stats/stats-100-default-guided-train/model-epoch=065-val_loss=0.029266/generated_samples",
        # "/home/ubuntu/repos/PoreGen/savedmodels/experimental/20250113-bps-ldm-ketton256-p_tpc_cond-aws/stats/stats-100-default-guided-train/model-epoch=065-val_loss=0.029822/generated_samples"
    
        "/home/ubuntu/repos/PoreGen/notebooks/exploratory/dfn/0009-data-farhad/rocks256valid"
    ]


    # Check if all folders exist
    for folder in folders:
        folder_path = pathlib.Path(folder)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder does not exist: {folder}")
        if not folder_path.is_dir():
            raise NotADirectoryError(f"Path exists but is not a directory: {folder}")

    print("All folders exist. Proceeding with processing...")
    for folder in folders:
        for i, file in enumerate(sorted([f for f in os.listdir(folder) if f.endswith('.raw')])):
            if i >= 10:
                break
            filepath = pathlib.Path(folder) / file
            outdir = str(filepath).replace('.raw', '-out')
            command = (
                f"bash -c 'source $(conda info --base)/etc/profile.d/conda.sh && "
                f"conda activate ddpm_env && "
                f"python run_palabos_permeability.py "
                f"--datapath={filepath} "
                f"--outdir={outdir}'"
            )
            print(f"Running: {command}")
            os.system(command)