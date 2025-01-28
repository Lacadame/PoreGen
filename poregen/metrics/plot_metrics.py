import json
import os
import pathlib
import yaml

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, interpolate
import scipy.integrate
import seaborn as sns


def plot_unconditional_metrics(datapath,
                               voxel_size_um=None,
                               min_porosity=-np.inf,
                               max_porosity=np.inf,
                               filter_dict=None,
                               plot_tag='',
                               show_permeability=True,
                               nbins=20,
                               use_log_properties=True,
                               convert_nan_to_zero=False):
    # voxel_size in um
    cfg_path = f"{datapath}/config.yaml"

    # TODO: Simplify this loop. It should begin with if voxel_size_um is None
    print(datapath)
    try:
        print(cfg_path)
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        assert voxel_size_um is not None, "voxel_size_um must be provided if config.yaml is not found"
    if voxel_size_um is None:
        try:
            voxel_size_um = cfg['data']['voxel_size_um']
        except KeyError:
            raise ValueError("voxel_size_um must be provided if config.yaml is not found")

    generated_stats_path = f"{datapath}/generated_stats.json"
    valid_stats_path = f"{datapath}/valid_stats.json"

    with open(generated_stats_path, "r") as f:
        generated_stats = json.load(f)
        if filter_dict is not None:
            print(generated_stats.keys())
            print(filter_dict.keys())
            generated_stats = {k: v for k, v in generated_stats.items() if filter_dict.get(k, False)}
    with open(valid_stats_path, "r") as f:
        valid_stats = json.load(f)

    generated_porosities, _ = extract_property(generated_stats, 'porosity')
    valid_porosities, _ = extract_property(valid_stats, 'porosity')
    generated_surface_area_densities, _ = extract_property(generated_stats, 'surface_area_density')
    valid_surface_area_densities, _ = extract_property(valid_stats, 'surface_area_density')
    generated_log_momenta, _ = extract_property(generated_stats, 'log_momenta')
    valid_log_momenta, _ = extract_property(valid_stats, 'log_momenta')

    # # print(generated_porosities)
    # # TODO: DELETE THAT
    ind = (generated_porosities > min_porosity).flatten() & (generated_porosities < max_porosity).flatten()

    print(ind.shape)
    print(generated_porosities.shape)
    if show_permeability:
        generated_permeabilities, _ = extract_property(generated_stats, 'permeability')
        valid_permeabilities, _ = extract_property(valid_stats, 'permeability')
        generated_permeabilities = generated_permeabilities[ind]
        generated_log_permeabilities = np.log10(np.prod(generated_permeabilities, axis=1)**(1/3))
        valid_log_permeabilities = np.log10(np.prod(valid_permeabilities, axis=1)**(1/3))
        generated_log_permeabilities[~np.isfinite(generated_log_permeabilities)] = 0
        valid_log_permeabilities[~np.isfinite(valid_log_permeabilities)] = 0

        if convert_nan_to_zero:
            generated_log_permeabilities[~np.isfinite(generated_log_permeabilities)] = 0
            valid_log_permeabilities[~np.isfinite(valid_log_permeabilities)] = 0
        else:
            generated_log_permeabilities = generated_log_permeabilities[np.isfinite(generated_log_permeabilities)]
            valid_log_permeabilities = valid_log_permeabilities[np.isfinite(valid_log_permeabilities)]
            print("HERE")
    generated_porosities = generated_porosities[ind]
    generated_surface_area_densities = generated_surface_area_densities[ind]
    generated_log_momenta = generated_log_momenta[ind]

    # Unit conversion
    # Permeability is already in Darcy
    generated_surface_area_densities = generated_surface_area_densities / voxel_size_um  # (1/um)
    valid_surface_area_densities = valid_surface_area_densities / voxel_size_um  # (1/um)

    log_momenta_conversion = np.log(np.array([voxel_size_um**(i+1) for i in range(generated_log_momenta.shape[1])]))
    generated_log_momenta = generated_log_momenta + log_momenta_conversion
    valid_log_momenta = valid_log_momenta + log_momenta_conversion

    generated_log_mean_pore_size = np.log10(np.exp(generated_log_momenta[:, 0]))
    valid_log_mean_pore_size = np.log10(np.exp(valid_log_momenta[:, 0]))

    if not use_log_properties:
        # Remove the logarithms, but keep the name "log" for backward compatibility
        generated_log_mean_pore_size = 10**generated_log_mean_pore_size
        valid_log_mean_pore_size = 10**valid_log_mean_pore_size
        generated_log_permeabilities = 10**generated_log_permeabilities
        valid_log_permeabilities = 10**valid_log_permeabilities
        print("NOT USING LOG PROPERTIES")
    # Example usage for both functions
    generated_data = [generated_porosities,
                      generated_surface_area_densities,
                      generated_log_mean_pore_size]
    valid_data = [valid_porosities, valid_surface_area_densities, valid_log_mean_pore_size]
    if use_log_properties:
        properties = ['porosity', 'surface_area_density', 'log_mean_pore_size']
        labels = ['Porosity', 'Surface Area Density', 'Log10 Mean Pore Size']
        units = [r"$\phi$", r"$1/\mu m$", r"$\log \,\mu m$"]
    else:
        properties = ['porosity', 'surface_area_density', 'mean_pore_size']
        labels = ['Porosity', 'Surface Area Density', 'Mean Pore Size']
        units = [r"$\phi$", r"$1/\mu m$", r"$\mu m$"]

    if show_permeability:
        generated_data.append(generated_log_permeabilities)
        valid_data.append(valid_log_permeabilities)
        if use_log_properties:
            properties.append('log_permeability')
            labels.append('Log10-Permeability')
            units.append(r"$\log \text{ Darcy}$")
        else:
            properties.append('permeability')
            labels.append('Permeability')
            units.append(r"$\text{Darcy}$")

    fig1 = plot_histograms(generated_data, valid_data, properties, labels, units, nbins)
    fig1_kde, divergences = plot_kde(generated_data, valid_data, properties, labels, units)

    fig2 = plot_boxplots(generated_data, valid_data, properties, labels, units)

    fig3, tpc_divergences = plot_two_point_correlation_comparison(generated_stats, valid_stats, voxel_size_um, ind)

    divergences['tpc_divergences'] = tpc_divergences

    # Example usage

    generated_psd_pdf, _ = extract_property(generated_stats, 'psd_pdf')
    generated_psd_cdf, _ = extract_property(generated_stats, 'psd_cdf')
    generated_psd_centers, _ = extract_property(generated_stats, 'psd_centers')

    valid_psd_pdf, _ = extract_property(valid_stats, 'psd_pdf')
    valid_psd_cdf, _ = extract_property(valid_stats, 'psd_cdf')
    valid_psd_centers, _ = extract_property(valid_stats, 'psd_centers')

    # # TODO: DELETE THAT
    generated_psd_pdf = generated_psd_pdf[ind]
    generated_psd_cdf = generated_psd_cdf[ind]
    generated_psd_centers = generated_psd_centers[ind]

    # The centers have units of um
    generated_psd_centers *= voxel_size_um
    valid_psd_centers *= voxel_size_um

    fig4, psd_divergences = plot_pore_size_distribution(generated_psd_pdf, generated_psd_cdf, generated_psd_centers,
                                       valid_psd_pdf, valid_psd_cdf, valid_psd_centers)
    # Add pore size distribution divergences to overall divergences
    divergences['pore_size_distribution'] = psd_divergences
    savefolder = pathlib.Path(f"{datapath}/figures")
    os.makedirs(savefolder, exist_ok=True)

    if plot_tag != '':
        plot_tag = f"_{plot_tag}"

    print('here')
    print(savefolder)
    fig1.savefig(savefolder / f"stats_histograms{plot_tag}.png", dpi=300)
    fig1_kde.savefig(savefolder / f"stats_kde{plot_tag}.png", dpi=300)
    fig2.savefig(savefolder / f"stats_boxplots{plot_tag}.png", dpi=300)
    fig3.savefig(savefolder / f"tpc_comparison{plot_tag}.png", dpi=300)
    fig4.savefig(savefolder / f"psd_comparison{plot_tag}.png", dpi=300)

    # Save KL divergences to JSON file
    divergences_path = savefolder / f"stats_divergences{plot_tag}.json"
    with open(divergences_path, "w") as f:
        json.dump(divergences, f, indent=4)



def plot_conditional_metrics(datapath, voxel_size_um=None, filter_dict=None, plot_tag=''):
    # TODO: Break this function into smaller functions!

    # voxel_size in um
    cfg_path = f"{datapath}/config.yaml"

    # TODO: Simplify this loop. It should begin with if voxel_size_um is None
    try:
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        assert voxel_size_um is not None, "voxel_size_um must be provided if config.yaml is not found"
    if voxel_size_um is None:
        try:
            voxel_size_um = cfg['data']['voxel_size_um']
        except KeyError:
            raise ValueError("voxel_size_um must be provided if config.yaml is not found")

    generated_stats_path = f"{datapath}/generated_stats.json"
    x_cond_stats_path = f"{datapath}/xcond_stats.json"
    valid_stats_path = f"{datapath}/valid_stats.json"

    with open(generated_stats_path, "r") as f:
        generated_stats = json.load(f)
        if filter_dict is not None:
            generated_stats = {k: v for k, v in generated_stats.items() if filter_dict.get(k, False)}

    with open(x_cond_stats_path, "r") as f:
        x_cond_stats = json.load(f)

    with open(valid_stats_path, "r") as f:
        valid_stats = json.load(f)

    conditions = cfg['data']['feature_extractor']
    fig1 = None
    fig2 = None
    fig3 = None
    fig4 = None
    fig5 = None

    if 'porosity' in conditions:
        gen, _ = extract_property(generated_stats, 'porosity')
        cond = x_cond_stats['porosity']
        gen = gen.flatten()
        cond = cond[0]

        fig1, ax = plt.subplots(1, 1, figsize=(6, 6))
        min_value = min(np.min(gen), cond)
        max_value = max(np.max(gen), cond)
        bins = np.linspace(min_value, max_value, 20)

        ax.hist(gen, bins=bins, density=True, color='red', alpha=0.5, label='Generated')
        ax.vlines(cond, 0, 10, color='black', label='Condition')

        ax.set_xlabel(r"$\phi$")
        ax.set_ylabel("Density")
        ax.set_title('Porosity')
        ax.legend()
        fig1.tight_layout()

    porosimetry_condition = (
        ('porosimetry_from_voxel' in conditions) or
        ('porosimetry_from_slice' in conditions) or
        ('porosimetry_from_voxel_slice' in conditions)
    )

    if porosimetry_condition:

        generated_psd_pdf, _ = extract_property(generated_stats, 'psd_pdf')
        generated_psd_cdf, _ = extract_property(generated_stats, 'psd_cdf')
        generated_psd_centers, _ = extract_property(generated_stats, 'psd_centers')

        valid_psd_pdf, _ = extract_property(valid_stats, 'psd_pdf')
        valid_psd_cdf, _ = extract_property(valid_stats, 'psd_cdf')
        valid_psd_centers, _ = extract_property(valid_stats, 'psd_centers')

        x_cond_psd_pdf = x_cond_stats['psd_pdf']
        x_cond_psd_pdf = np.array(x_cond_psd_pdf)
        x_cond_psd_cdf = x_cond_stats['psd_cdf']
        x_cond_psd_cdf = np.array(x_cond_psd_cdf)
        x_cond_psd_centers = x_cond_stats['psd_centers']
        x_cond_psd_centers = np.array(x_cond_psd_centers)

        # The centers have units of um
        generated_psd_centers *= voxel_size_um
        x_cond_psd_centers *= voxel_size_um
        valid_psd_centers *= voxel_size_um

        # Plot PDF
        fig2, axs = plt.subplots(1, 2, figsize=(12, 6))
        nplots = generated_psd_pdf.shape[1] - 1
        for i in range(nplots):
            axs[0].plot(generated_psd_centers[i], generated_psd_pdf[i], alpha=0.2, color='red')
            axs[1].plot(valid_psd_centers[i], valid_psd_pdf[i], alpha=0.2, color='red')
        axs[0].plot(generated_psd_centers[nplots], generated_psd_pdf[nplots], alpha=0.2, color='red',
                    label='Generated')
        axs[1].plot(valid_psd_centers[nplots], valid_psd_pdf[nplots], alpha=0.2, color='red', label='Valid')

        axs[0].plot(x_cond_psd_centers, x_cond_psd_pdf, color='black', label='Condition')
        axs[1].plot(x_cond_psd_centers, x_cond_psd_pdf, color='black', label='Condition')

        for ax in axs:
            ax.set_xlabel(r'Pore Size $(\mu m)$')
            ax.set_ylabel('Probability Density')
            ax.legend()
            ax.set_xscale('log')
        axs[0].set_title('Pore Size Distribution - PDF (Generated)')
        axs[1].set_title('Pore Size Distribution - PDF (Validation)')

        fig2.tight_layout()

        # Plot CDF
        fig3, axs = plt.subplots(1, 2, figsize=(12, 6))
        nplots = generated_psd_cdf.shape[1] - 1
        for i in range(nplots):
            axs[0].plot(generated_psd_centers[i], generated_psd_cdf[i], alpha=0.2, color='red')
            axs[1].plot(valid_psd_centers[i], valid_psd_cdf[i], alpha=0.2, color='red')
        axs[0].plot(generated_psd_centers[nplots], generated_psd_cdf[nplots], alpha=0.2, color='red',
                    label='Generated')
        axs[1].plot(valid_psd_centers[nplots], valid_psd_cdf[nplots], alpha=0.2, color='red', label='Valid')

        axs[0].plot(x_cond_psd_centers, x_cond_psd_cdf, color='black', label='Condition')
        axs[1].plot(x_cond_psd_centers, x_cond_psd_cdf, color='black', label='Condition')

        for ax in axs:
            ax.set_xlabel(r'Pore Size $(\mu m)$')
            ax.set_ylabel('Cumulative Probability')
            ax.set_title('Pore Size Distribution - CDF')
            ax.legend()
            ax.set_xscale('log')
        fig3.tight_layout()

        # boxplots for momenta
        generated_momenta, _ = extract_property(generated_stats, 'standardized_momenta')
        valid_momenta, _ = extract_property(valid_stats, 'standardized_momenta')
        x_cond_momenta = x_cond_stats['standardized_momenta']

        fig4, axs = plt.subplots(2, 4, figsize=(16, 8))
        for i in range(4):
            axs[0, i].boxplot([generated_momenta[:, i], x_cond_momenta[i]], labels=['Generated', 'Condition'])
            axs[1, i].boxplot([valid_momenta[:, i], x_cond_momenta[i]], labels=['Validation', 'Condition'])
            axs[0, i].set_ylabel(r'$\mu$')
            axs[0, i].set_title(f'Standardized Momenta {i+1}')
            axs[1, i].set_ylabel(r'$\mu$')
            axs[1, i].set_title(f'Standardized Momenta {i+1}')
        fig4.tight_layout()

    two_point_correlation_condition = (
        ('two_point_correlation_from_voxel' in conditions) or
        ('two_point_correlation_from_slice' in conditions) or
        ('two_point_correlation_from_voxel_slice' in conditions)
    )

    if two_point_correlation_condition:
        generated_tpc_dist, _ = extract_property(generated_stats, 'tpc_dist')
        valid_tpc_dist, _ = extract_property(valid_stats, 'tpc_dist')
        x_cond_tpc_dist = x_cond_stats['tpc_dist']
        x_cond_tpc_dist = np.array(x_cond_tpc_dist)

        # tpc_dists have units of um
        generated_tpc_dist *= voxel_size_um
        valid_tpc_dist *= voxel_size_um
        x_cond_tpc_dist *= voxel_size_um

        generated_tpc_prob, _ = extract_property(generated_stats, 'tpc_prob')
        valid_tpc_prob, _ = extract_property(valid_stats, 'tpc_prob')
        x_cond_tpc_prob = x_cond_stats['tpc_prob']

        # x_generated = generated_tpc_dist.mean(axis=0)
        # x_x_cond = x_cond_tpc_dist.mean(axis=0)

        fig5, axs = plt.subplots(1, 2, figsize=(12, 5))
        nplots = generated_tpc_dist.shape[0] - 1
        for i in range(nplots):
            axs[0].plot(generated_tpc_dist[i], generated_tpc_prob[i], alpha=0.2, color='red')
            axs[1].plot(valid_tpc_dist[i], valid_tpc_prob[i], alpha=0.2, color='red')
        axs[0].plot(generated_tpc_dist[nplots], generated_tpc_prob[nplots], alpha=0.2, color='red', label='Generated')
        axs[1].plot(valid_tpc_dist[nplots], valid_tpc_prob[nplots], alpha=0.2, color='red', label='Validation')
        print(len(x_cond_tpc_dist), len(x_cond_tpc_prob))
        print(x_cond_tpc_prob)
        print(x_cond_tpc_dist)
        for ax in axs:
            ax.plot(x_cond_tpc_dist, x_cond_tpc_prob, color='black', label='Condition')

            ax.set_xlabel(r"$r$ $(\mu m)$")
            ax.set_ylabel(r"$(s_2 - \phi^2)/(\phi - \phi^2)$")
            ax.legend()
            ax.set_title("Comparison of two point correlation (TPC)")

    savefolder = pathlib.Path(f"{datapath}/figures")
    os.makedirs(savefolder, exist_ok=True)
    if plot_tag != '':
        plot_tag = f"_{plot_tag}"
    if fig1 is not None:
        fig1.savefig(savefolder / f"porosity{plot_tag}.png")
    if fig2 is not None:
        fig2.savefig(savefolder / f"pore_size_distribution_pdf{plot_tag}.png")
    if fig3 is not None:
        fig3.savefig(savefolder / f"pore_size_distribution_cdf{plot_tag}.png")
    if fig4 is not None:
        fig4.savefig(savefolder / f"psd_momenta{plot_tag}.png")
    if fig5 is not None:
        fig5.savefig(savefolder / f"two_point_correlation{plot_tag}.png")


def plot_cond_porosity(datapaths, conditions, nsamples=100, bins=10, filter_dict=None, plot_tag=''):
    validation = []
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    for i, datapath in enumerate(datapaths):
        # Load porosities
        generated_stats_path = f"{datapath}/generated_stats.json"
        valid_stats_path = f"{datapath}/valid_stats.json"
        with open(generated_stats_path, "r") as f:
            generated_stats = json.load(f)
            if filter_dict is not None:
                generated_stats = {k: v for k, v in generated_stats.items() if filter_dict.get(k, False)}
        with open(valid_stats_path, "r") as f:
            valid_stats = json.load(f)
        gen, _ = extract_property(generated_stats, 'porosity')
        gen = gen.flatten()
        valid, _ = extract_property(valid_stats, 'porosity')
        valid = valid.flatten()
        validation.append(valid)
        # Plot the conditions
        ax.vlines(x=conditions[i], color='black', ymin=0, ymax=30, linewidth=3)
        # Plot the histograms
        ax.hist(gen, bins=bins, density=False, alpha=0.5, label=f'Condition = {conditions[i]}')
    validation = np.concatenate(validation)
    nconds = len(conditions)
    ax.hist(validation, bins=nconds*bins, density=False, label='Validation', histtype='step',
            color='grey', linewidth=2)
    ax.set_xlabel('Porosity values')
    # ax.set_ylabel('Density')
    ax.set_title(f'Porosity histograms - {nsamples} samples for each condition')
    ax.legend()
    fig.tight_layout()
    plt.show()

    # Save the figure in a parent folder
    savefolder = pathlib.Path(datapaths[0]).parent.parent / "figures"
    os.makedirs(savefolder, exist_ok=True)
    if plot_tag != '':
        plot_tag = f"_{plot_tag}"
    fig.savefig(savefolder / f"cond_porosity{plot_tag}.png")
    plt.close(fig)


def plot_vae_reconstruction(datapath, nsamples=1, tag=None):

    input_path = f"{datapath}/input_samples"
    bin_rec_path = f"{datapath}/reconstructed_samples_bin"
    input = np.stack([np.load(os.path.join(
        input_path, f"{i:05d}_input.npy")) for i in range(nsamples)])
    bin_rec = np.stack([np.load(os.path.join(
        bin_rec_path, f"{i:05d}.npy")) for i in range(nsamples)])

    savefolder = pathlib.Path(f"{datapath}/figures")
    os.makedirs(savefolder, exist_ok=True)

    for i in range(nsamples):
        x = 1 - input[i]
        x_rec = 1 - bin_rec[i]

        fig1 = plt.figure()
        ax = fig1.add_subplot(111, projection='3d')
        ax.voxels(x[0], facecolors='blue', alpha=0.2)
        fig1.tight_layout()
        ax.set_title(f"Input {i+1}")
        plt.show()
        fig1.savefig(savefolder / f"input{i+1}.png")

        fig2 = plt.figure()
        ax = fig2.add_subplot(111, projection='3d')
        ax.voxels(x_rec[0], facecolors='blue', alpha=0.2)
        fig2.tight_layout()
        ax.set_title(f"Reconstructed {i+1}")
        plt.show()
        fig2.savefig(savefolder / f"reconstructed{i+1}.png")


def plot_vae_reconstruction_errors(datapath: str | list[str],
                                   savefolder: str):
    if isinstance(datapath, str):
        datapath = [datapath]

    l1_rec_error = []
    bin_rec_error = []

    for i, path in enumerate(datapath):
        assert os.path.isdir(path), f"{path} is not a directory"

        # Load reconstruction errors
        with open(f"{path}/reconstruction_errors.json", "r") as f:
            rec_error = json.load(f)

        l1_rec_error.append(rec_error['l1_rec_error'])
        bin_rec_error.append(rec_error['bin_rec_error'])

    # Ensure savefolder exists
    savefolder = pathlib.Path(savefolder)
    os.makedirs(savefolder, exist_ok=True)

    # Plot reconstruction errors
    ndata = len(l1_rec_error)
    fig, axs = plt.subplots(ndata, 2, figsize=(12, 4 * ndata))  # Adjust height based on number of datasets

    for j in range(ndata):
        axs[j, 0].hist(l1_rec_error[j], bins=20, density=True, color='blue', alpha=0.7)
        axs[j, 0].set_title(f'L1 Reconstruction Error {j+1}')
        axs[j, 0].set_xlabel('L1 Reconstruction Error')
        axs[j, 0].set_ylabel('Density')

        axs[j, 1].hist(bin_rec_error[j], bins=20, density=True, color='green', alpha=0.7)
        axs[j, 1].set_title(f'Binary Reconstruction Error {j+1}')
        axs[j, 1].set_xlabel('Binary Reconstruction Error')
        axs[j, 1].set_ylabel('Density')

    fig.tight_layout()
    fig.savefig(savefolder / "reconstruction_errors.png")
    print(f"Reconstruction error plots saved to {savefolder / 'reconstruction_errors.png'}")


def extract_property(data, property_name):
    property_list = []
    input_list = []

    sorted_keys = sorted(data.keys())
    for k in sorted_keys:
        v = data[k]
        if k == 'condition':
            continue
        if isinstance(v, dict) and property_name in v:
            value = np.array(v[property_name])
            # Hack to check if the value is empty,
            # or if there is any non-finite value
            check = np.all(np.isfinite(value)) and len(value) > 0
            if not check:
                print("Empty or faulty value for key", k)
                continue
                # if len(value) == 0:
                    # continue
                # value = np.nan * np.ones_like(value)
            property_list.append(value)
            input_list.append(k)
    print(property_name, property_list)
    property_list = np.stack(property_list, axis=0)
    input_list = np.array(input_list)
    return property_list, input_list


def plot_boxplots(generated_data, valid_data, properties, labels, units):
    fig, ax = plt.subplots(1, len(properties), figsize=(4*len(properties), 4))

    for i, (gen, val, prop, label, unit) in enumerate(zip(generated_data, valid_data, properties, labels, units)):
        ax[i].boxplot([gen.flatten(), val.flatten()], labels=['Generated', 'Valid'])
        ax[i].set_ylabel(unit)
        ax[i].set_title(label)

    fig.tight_layout()
    return fig


def plot_two_point_correlation_comparison(generated_stats, valid_stats, voxel_size_um, ind=None):
    """Plot comparison of two point correlation between generated and validation data.
    
    Parameters
    ----------
    generated_stats : dict
        Dictionary containing generated statistics
    valid_stats : dict
        Dictionary containing validation statistics 
    voxel_size_um : float
        Voxel size in micrometers
    ind : array-like, optional
        Indices to select from generated data
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the two point correlation comparison plot
    divergences : dict
        Dictionary containing KL divergence, Hellinger distance and mean relative error
    """
    generated_tpc_dist, _ = extract_property(generated_stats, 'tpc_dist')
    valid_tpc_dist, _ = extract_property(valid_stats, 'tpc_dist')

    # tpc_dists have units of um
    generated_tpc_dist *= voxel_size_um
    valid_tpc_dist *= voxel_size_um

    generated_tpc_prob, _ = extract_property(generated_stats, 'tpc_prob')
    valid_tpc_prob, _ = extract_property(valid_stats, 'tpc_prob')

    if ind is not None:
        generated_tpc_dist = generated_tpc_dist[ind]
        generated_tpc_prob = generated_tpc_prob[ind]

    mean_generated_tpc_prob = np.mean(generated_tpc_prob, 0)
    std_generated_tpc_prob = np.std(generated_tpc_prob, 0)
    mean_valid_tpc_prob = np.mean(valid_tpc_prob, 0)
    std_valid_tpc_prob = np.std(valid_tpc_prob, 0)

    x_generated = generated_tpc_dist.mean(axis=0)
    x_valid = valid_tpc_dist.mean(axis=0)

    # Calculate divergences
    N = len(mean_valid_tpc_prob)
    kl_div = np.sum(mean_valid_tpc_prob * np.log(mean_valid_tpc_prob / mean_generated_tpc_prob)) / N
    hellinger_dist = np.sum(np.sqrt(mean_valid_tpc_prob * mean_generated_tpc_prob)) / N
    rel_error = np.mean(np.abs(mean_valid_tpc_prob - mean_generated_tpc_prob) / mean_valid_tpc_prob)

    divergences = {
        'kl_divergence': float(kl_div),
        'hellinger_distance': float(hellinger_dist),
        'mean_relative_error': float(rel_error)
    }

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.fill_between(x_generated,
                    y1=mean_generated_tpc_prob-2*std_generated_tpc_prob,
                    y2=mean_generated_tpc_prob+2*std_generated_tpc_prob,
                    color='red',
                    alpha=0.2)
    ax.fill_between(x_valid,
                    y1=mean_valid_tpc_prob-2*std_valid_tpc_prob,
                    y2=mean_valid_tpc_prob+2*std_valid_tpc_prob,
                    color='blue',
                    alpha=0.2)

    ax.plot(x_generated, mean_generated_tpc_prob, color='red', label='Generated')
    ax.plot(x_valid, mean_valid_tpc_prob, color='blue', label='Valid')
    ax.set_xlabel(r"$r$ $(\mu m)$")
    ax.set_ylabel(r"$(s_2 - \phi^2)/(\phi - \phi^2)$")
    ax.legend()
    ax.set_title("Comparison of two point correlation")
    
    return fig, divergences


def plot_histograms(generated_data, valid_data, properties, labels, units, nbins=20):
    fig, ax = plt.subplots(1, len(properties), figsize=(5*len(properties), 5))

    for i, (gen, val, prop, label, unit) in enumerate(zip(generated_data, valid_data, properties, labels, units)):
        min_value = min(np.min(gen), np.min(val))
        max_value = max(np.max(gen), np.max(val))
        bins = np.linspace(min_value, max_value, nbins)

        # Plot histograms
        ax[i].hist(gen.flatten(), bins=bins, density=True, color='red', alpha=0.5, label='Generated')
        ax[i].hist(val.flatten(), bins=bins, density=True, color='blue', alpha=0.5, label='Valid')

        ax[i].set_xlabel(unit)
        ax[i].set_ylabel("Density")
        ax[i].set_title(label)
        ax[i].legend()

    fig.tight_layout()
    return fig


def plot_kde(generated_data, valid_data, properties, labels, units):
    fig, ax = plt.subplots(1, len(properties), figsize=(5*len(properties), 5))
    divergences = {}

    for i, (gen, val, prop, label, unit) in enumerate(zip(generated_data, valid_data, properties, labels, units)):
        min_value = min(np.min(gen), np.min(val))
        max_value = max(np.max(gen), np.max(val))

        # Expand min and max values by 10%
        min_value = min_value - 0.2 * (max_value - min_value)
        max_value = max_value + 0.2 * (max_value - min_value)
        
        if label in ["Porosity", "Surface Area Density", "Mean Pore Size", "Permeability"]:
            min_value = max(min_value, 0.0)

        gen_kde = stats.gaussian_kde(gen.flatten())
        val_kde = stats.gaussian_kde(val.flatten())
        x_kde = np.linspace(min_value, max_value, 200)
        ax[i].fill_between(x_kde, 0, gen_kde(x_kde), color='red', alpha=0.4, label='Generated')
        ax[i].fill_between(x_kde, 0, val_kde(x_kde), color='blue', alpha=0.4, label='Valid')

        ax[i].set_xlabel(unit)
        ax[i].set_ylabel("Density")
        ax[i].set_title(label)
        ax[i].legend()

        # Calculate KL divergence and Hellinger distance between valid and generated distributions
        kl_div = kl_divergence_kde_mc(val.flatten(), gen.flatten(), n_samples=10000)
        hellinger_dist = hellinger_distance_kde_mc(val.flatten(), gen.flatten(), n_samples=10000)
        if label in ['Porosity', 'Surface Area Density', 'Mean Pore Size', 'Permeability']:
            rel_error = mean_relative_error(val.flatten(), gen.flatten())
        else:
            rel_error = np.nan
        divergences[prop] = {
            'kl_divergence': kl_div,
            'hellinger_distance': hellinger_dist,
            'mean_relative_error': rel_error
        }

    fig.tight_layout()
    return fig, divergences


def plot_pore_size_distribution(generated_psd_pdf, generated_psd_cdf, generated_psd_centers,
                                valid_psd_pdf, valid_psd_cdf, valid_psd_centers):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Create a common x-axis for interpolation
    x_interp = np.logspace(np.log10(min(np.min(generated_psd_centers), np.min(valid_psd_centers))),
                           np.log10(max(np.max(generated_psd_centers), np.max(valid_psd_centers))),
                           num=1000)

    # PDF Plot using direct KDE for each sample
    def direct_kde_plot(ax, data, centers, color, label):
        kde_estimates = []
        for sample, sample_centers in zip(data, centers):
            kde = stats.gaussian_kde(sample_centers, weights=sample)
            kde_estimates.append(kde(x_interp))

        kde_estimates = np.array(kde_estimates)
        kde_mean = np.mean(kde_estimates, axis=0)
        kde_std = np.std(kde_estimates, axis=0)

        ax.fill_between(x_interp, np.maximum(kde_mean - 2*kde_std, 0), kde_mean + 2*kde_std, alpha=0.3, color=color)
        ax.plot(x_interp, kde_mean, color=color, label=label)

        return kde_mean

    gen_mean = direct_kde_plot(ax1, generated_psd_pdf, generated_psd_centers, 'red', 'Generated')
    val_mean = direct_kde_plot(ax1, valid_psd_pdf, valid_psd_centers, 'blue', 'Valid')

    # Calculate divergence metrics for PDFs
    dx = np.diff(x_interp)[0]  # Approximate integration step
    
    # KL divergence integral
    eps = 1e-10
    kl_integrand = val_mean * np.log((val_mean + eps) / (gen_mean + eps))
    kl_div = np.trapz(kl_integrand, x_interp)
    
    # Hellinger distance integral
    hellinger_integrand = (np.sqrt(val_mean) - np.sqrt(gen_mean))**2
    hellinger_dist = np.sqrt(0.5 * np.trapz(hellinger_integrand, x_interp))
    
    # Mean relative error (using means of distributions)
    val_mean_integral = np.trapz(val_mean, x_interp)
    gen_mean_integral = np.trapz(gen_mean, x_interp)
    rel_error = np.abs(val_mean_integral - gen_mean_integral) / val_mean_integral

    divergences = {
        'kl_divergence': kl_div,
        'hellinger_distance': hellinger_dist,
        'mean_relative_error': rel_error
    }

    ax1.set_xlabel(r'Pore Size $(\mu m)$')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Pore Size Distribution - PDF')
    ax1.legend()
    ax1.set_xscale('log')

    # CDF Plot with interpolation (unchanged)
    def interpolate_cdf(centers, cdf):
        centers, unique_indices = np.unique(centers, return_index=True)
        cdf = cdf[unique_indices]
        return interpolate.interp1d(centers, cdf, kind='linear', fill_value='extrapolate')

    def cdf_plot(ax, data, centers, color, label):
        cdf_interp = []
        for sample_cdf, sample_centers in zip(data, centers):
            interp_func = interpolate_cdf(sample_centers, sample_cdf)
            cdf_interp.append(interp_func(x_interp))

        cdf_interp = np.array(cdf_interp)
        mean_cdf = np.mean(cdf_interp, axis=0)
        std_cdf = np.std(cdf_interp, axis=0)

        ax.fill_between(x_interp, np.maximum(mean_cdf - 2*std_cdf, 0),
                        np.minimum(mean_cdf + 2*std_cdf, 1), alpha=0.3, color=color)
        ax.plot(x_interp, mean_cdf, color=color, label=label)

    cdf_plot(ax2, generated_psd_cdf, generated_psd_centers, 'red', 'Generated')
    cdf_plot(ax2, valid_psd_cdf, valid_psd_centers, 'blue', 'Valid')

    ax2.set_xlabel('Pore Size')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Pore Size Distribution - CDF')
    ax2.legend()
    ax2.set_xscale('log')

    plt.tight_layout()
    return fig, divergences


def kl_divergence_kde_mc(sample1, sample2, n_samples=10000, seed=None):
    """
    Calculate KL divergence D_KL(P||Q) between two samples using Gaussian KDE
    and Monte Carlo integration with resampling.

    Parameters:
    -----------
    sample1 : array-like
        First sample (P distribution)
    sample2 : array-like
        Second sample (Q distribution)
    n_samples : int, optional
        Number of samples to use for Monte Carlo integration
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    float
        KL divergence D_KL(P||Q)
    """
    # Convert inputs to numpy arrays
    sample1 = np.asarray(sample1)
    sample2 = np.asarray(sample2)

    # Estimate PDFs using Gaussian KDE
    kde1 = stats.gaussian_kde(sample1)
    kde2 = stats.gaussian_kde(sample2)

    # Use resample method for Monte Carlo integration
    if seed is not None:
        np.random.seed(seed)

    # Sample from P distribution (kde1)
    mc_samples = kde1.resample(n_samples, seed=seed)

    # Evaluate both PDFs at the sampled points
    p_values = kde1.evaluate(mc_samples)
    q_values = kde2.evaluate(mc_samples)

    # Add small constant to prevent log(0)
    eps = 1e-10

    # Calculate KL divergence using Monte Carlo integration
    # KL = E_p[log(p/q)] ≈ (1/n) * Σ log(p/q)
    kl_div = np.mean(np.log((p_values + eps) / (q_values + eps)))

    return kl_div


def hellinger_distance_kde_mc(sample1, sample2, n_samples=10000, seed=None):
    """
    Calculate Hellinger distance H(P,Q) between two samples using Gaussian KDE
    and numerical integration. The Hellinger distance is defined as:
    H(P,Q) = sqrt(1/2 ∫(sqrt(p) - sqrt(q))^2 dx)

    Parameters:
    -----------
    sample1 : array-like
        First sample (P distribution)
    sample2 : array-like
        Second sample (Q distribution) 
    nsamples : int
        Dummy parameter to match the signature of kl_divergence_kde_mc
    seed : int, optional
        Dummy parameter to match the signature of kl_divergence_kde_mc

    Returns:
    --------
    float
        Hellinger distance H(P,Q)
    """
    # Convert inputs to numpy arrays
    sample1 = np.asarray(sample1)
    sample2 = np.asarray(sample2)

    # Estimate PDFs using Gaussian KDE
    kde1 = stats.gaussian_kde(sample1)
    kde2 = stats.gaussian_kde(sample2)

    # Define integration bounds as min/max of samples with some padding
    min_x = min(sample1.min(), sample2.min())
    max_x = max(sample1.max(), sample2.max())
    padding = 0.2 * (max_x - min_x)  # 20% padding on each side
    
    def integrand(x):
        # Add small constant to prevent numerical issues with sqrt
        eps = 1e-10
        p = kde1.evaluate(x) + eps
        q = kde2.evaluate(x) + eps
        return 0.5 * (np.sqrt(p) - np.sqrt(q))**2

    # Use scipy.integrate.quad for numerical integration
    integral, _ = scipy.integrate.quad(integrand, min_x - padding, max_x + padding)
    hellinger = np.sqrt(integral)

    return hellinger


def mean_relative_error(sample1, sample2):
    """
    Calculate mean relative error between two samples.
    
    Parameters:
    -----------
    sample1 : array-like
        First sample (generated distribution)
    sample2 : array-like
        Second sample (validation distribution)
        
    Returns:
    --------
    float
        Mean relative error |mean(gen) - mean(val)|/|mean(val)|
    """
    # Convert inputs to numpy arrays
    sample1 = np.asarray(sample1)
    sample2 = np.asarray(sample2)
    
    # Calculate means
    mean1 = np.mean(sample1)
    mean2 = np.mean(sample2)
    
    # Calculate relative error
    rel_error = np.abs(mean1 - mean2) / np.abs(mean2)
    
    return rel_error
