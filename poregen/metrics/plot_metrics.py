import json
import os
import pathlib
import yaml

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, interpolate


def plot_unconditional_metrics(datapath, voxel_size_um=None):
    # voxel_size in um
    cfg_path = f"{datapath}/config.yaml"
    try:
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        assert voxel_size_um is not None, "voxel_size must be provided if config.yaml is not found"
    if voxel_size_um is None:
        try:
            voxel_size_um = cfg['data']['voxel_size_um']
        except KeyError:
            raise ValueError("voxel_size_um must be provided if config.yaml is not found")

    generated_stats_path = f"{datapath}/generated_stats.json"
    valid_stats_path = f"{datapath}/valid_stats.json"

    with open(generated_stats_path, "r") as f:
        generated_stats = json.load(f)

    with open(valid_stats_path, "r") as f:
        valid_stats = json.load(f)

    generated_porosities, _ = extract_property(generated_stats, 'porosity')
    valid_porosities, _ = extract_property(valid_stats, 'porosity')
    generated_surface_area_densities, _ = extract_property(generated_stats, 'surface_area_density')
    valid_surface_area_densities, _ = extract_property(valid_stats, 'surface_area_density')
    generated_permeabilities, _ = extract_property(generated_stats, 'permeability')
    valid_permeabilities, _ = extract_property(valid_stats, 'permeability')
    generated_log_momenta, _ = extract_property(generated_stats, 'log_momenta')
    valid_log_momenta, _ = extract_property(valid_stats, 'log_momenta')

    # Unit conversion
    # Permeability is already in Darcy
    generated_surface_area_densities = generated_surface_area_densities / voxel_size_um  # (1/um)
    valid_surface_area_densities = valid_surface_area_densities / voxel_size_um  # (1/um)

    log_momenta_conversion = np.log(np.array([voxel_size_um**(i+1) for i in range(generated_log_momenta.shape[1])]))
    generated_log_momenta = generated_log_momenta + log_momenta_conversion
    valid_log_momenta = valid_log_momenta + log_momenta_conversion

    generated_log_permeabilities = np.log10(np.prod(generated_permeabilities, axis=1)**(1/3))
    valid_log_permeabilities = np.log10(np.prod(valid_permeabilities, axis=1)**(1/3))
    generated_log_mean_pore_size = np.log10(np.exp(generated_log_momenta[:, 0]))
    valid_log_mean_pore_size = np.log10(np.exp(valid_log_momenta[:, 0]))

    # Example usage for both functions
    generated_data = [generated_porosities,
                      generated_surface_area_densities,
                      generated_log_mean_pore_size,
                      generated_log_permeabilities]
    valid_data = [valid_porosities, valid_surface_area_densities, valid_log_mean_pore_size, valid_log_permeabilities]
    properties = ['porosity', 'surface_area_density', 'log_mean_pore_size', 'log_permeability']
    labels = ['Porosity', 'Surface Area Density', 'Log10 Mean Pore Size', 'Log10-Permeability']
    units = [r"$\phi$", r"$1/\mu m$", r"$\log \,\mu m$", r"$\log \text{ Darcy}$"]

    fig1 = plot_histograms(generated_data, valid_data, properties, labels, units)

    fig2 = plot_boxplots(generated_data, valid_data, properties, labels, units)

    generated_tpc_dist, _ = extract_property(generated_stats, 'tpc_dist')
    valid_tpc_dist, _ = extract_property(valid_stats, 'tpc_dist')

    # tpc_dists have units of um
    generated_tpc_dist *= voxel_size_um
    valid_tpc_dist *= voxel_size_um

    generated_tpc_prob, _ = extract_property(generated_stats, 'tpc_prob')
    valid_tpc_prob, _ = extract_property(valid_stats, 'tpc_prob')

    mean_generated_tpc_prob = np.mean(generated_tpc_prob, 0)
    std_generated_tpc_prob = np.std(generated_tpc_prob, 0)
    mean_valid_tpc_prob = np.mean(valid_tpc_prob, 0)
    std_valid_tpc_prob = np.std(valid_tpc_prob, 0)

    x_generated = generated_tpc_dist.mean(axis=0)
    x_valid = valid_tpc_dist.mean(axis=0)

    fig3, ax = plt.subplots(1, 1, figsize=(6, 6))

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

    ax.plot(x_generated, mean_generated_tpc_prob, color='red')
    ax.plot(x_valid, mean_valid_tpc_prob, color='blue')
    ax.set_xlabel(r"$r$ $(\mu m)$")
    ax.set_ylabel(r"$(s_2 - \phi^2)/(\phi - \phi^2)$")
    ax.legend()
    ax.set_title("Comparison of two point correlation")

    # Example usage

    generated_psd_pdf, _ = extract_property(generated_stats, 'psd_pdf')
    generated_psd_cdf, _ = extract_property(generated_stats, 'psd_cdf')
    generated_psd_centers, _ = extract_property(generated_stats, 'psd_centers')

    valid_psd_pdf, _ = extract_property(valid_stats, 'psd_pdf')
    valid_psd_cdf, _ = extract_property(valid_stats, 'psd_cdf')
    valid_psd_centers, _ = extract_property(valid_stats, 'psd_centers')

    # The centers have units of um
    generated_psd_centers *= voxel_size_um
    valid_psd_centers *= voxel_size_um

    fig4 = plot_pore_size_distribution(generated_psd_pdf, generated_psd_cdf, generated_psd_centers,
                                       valid_psd_pdf, valid_psd_cdf, valid_psd_centers)

    savefolder = pathlib.Path(f"{datapath}/figures")
    os.makedirs(savefolder, exist_ok=True)
    fig1.savefig(savefolder / "stats_histograms.png")
    fig2.savefig(savefolder / "stats_boxplots.png")
    fig3.savefig(savefolder / "tpc_comparison.png")
    fig4.savefig(savefolder / "psd_comparison.png")


def extract_property(data, property_name):
    property_list = []
    input_list = []

    sorted_keys = sorted(data.keys())
    for k in sorted_keys:
        v = data[k]
        if k == 'condition':
            continue
        if isinstance(v, dict) and property_name in v:
            property_list.append(v[property_name])
            input_list.append(k)
    property_list = np.array(property_list)
    input_list = np.array(input_list)
    return property_list, input_list


def plot_boxplots(generated_data, valid_data, properties, labels, units):
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))

    for i, (gen, val, prop, label, unit) in enumerate(zip(generated_data, valid_data, properties, labels, units)):
        ax[i].boxplot([gen.flatten(), val.flatten()], labels=['Generated', 'Valid'])
        ax[i].set_ylabel(unit)
        ax[i].set_title(label)

    fig.tight_layout()
    return fig


def plot_histograms(generated_data, valid_data, properties, labels, units):
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    for i, (gen, val, prop, label, unit) in enumerate(zip(generated_data, valid_data, properties, labels, units)):
        min_value = min(np.min(gen), np.min(val))
        max_value = max(np.max(gen), np.max(val))
        bins = np.linspace(min_value, max_value, 20)

        ax[i].hist(gen.flatten(), bins=bins, density=True, color='red', alpha=0.5, label='Generated')
        ax[i].hist(val.flatten(), bins=bins, density=True, color='blue', alpha=0.5, label='Valid')

        ax[i].set_xlabel(unit)
        ax[i].set_ylabel("Density")
        ax[i].set_title(label)
        ax[i].legend()

    fig.tight_layout()
    return fig


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

    direct_kde_plot(ax1, generated_psd_pdf, generated_psd_centers, 'red', 'Generated')
    direct_kde_plot(ax1, valid_psd_pdf, valid_psd_centers, 'blue', 'Valid')

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
    return fig
