import os
import pathlib

import numpy as np
import torch

import poregen.metrics
import poregen.trainers


def main():

    datapath = '/home/ubuntu/repos/PoreGen/savedmodels/experimental/20250106-bps-ldm-estaillades256-aws/stats/stats-100-karras-256steps/model-epoch=068-val_loss=0.064220'
    
    datapath = pathlib.Path(datapath)

    cfg_path = f"{datapath}/config.yaml"
    checkpoint_path = "best"

    # np.load(pathlib.Path(datapath)/'generated_samples'/'00001.npy')

    loaded = poregen.trainers.pore_load(cfg_path,
                                        checkpoint_path,
                                        load_data=True)
    module = loaded['trainer'].karras_module
    module = module.to("cuda:7")


    generated_samples_folder = datapath/'generated_samples'

    spectrum_criteria_results = {}
    for sample_name in os.listdir(generated_samples_folder):
        if sample_name.endswith('.npy'):
            encoded_sample = module.encode(
                torch.tensor(
                    np.load(generated_samples_folder/sample_name), 
                    dtype=torch.float32
                ).unsqueeze(0).to("cuda:7")
            )
            spectrum_criteria_results[sample_name] = poregen.metrics.power_spectrum_criteria(encoded_sample).item()
    spectrum_criteria_results = {k.replace('.npy', ''): v for k, v in spectrum_criteria_results.items()}

    positives = {k: v for k, v in spectrum_criteria_results.items() if v == True}
    print(datapath, len(positives), len(spectrum_criteria_results))

    poregen.metrics.plot_unconditional_metrics(
            datapath=datapath,
            filter_dict=spectrum_criteria_results, plot_tag='_spectra_filtered',
            show_permeability=False)


if __name__ == "__main__":
    main()