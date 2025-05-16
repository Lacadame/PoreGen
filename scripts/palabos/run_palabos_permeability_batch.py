import pathlib
import os


if __name__ == "__main__":
    folder = '/home/danilo/repos/PoreGen/savedmodels/experimental/20241202-bps-ldm-bentheimer256/stats/stats-100-default/model-epoch=042-val_loss=0.053127/valid_samples'
    for i, file in enumerate(sorted([f for f in os.listdir(folder) if f.endswith('.npy')])):
        if i >= 2:
            break
        filepath = pathlib.Path(folder) / file
        outdir = str(filepath).replace('.npy', '-out')
        command = (
            f"bash -c 'source $(conda info --base)/etc/profile.d/conda.sh && "
            f"conda activate ddpm_env && "
            f"python run_palabos_permeability.py "
            f"--datapath={filepath} "
            f"--outdir={outdir}'"
        )
        print(f"Running: {command}")
        os.system(command)