import json
import os
import pathlib
import yaml

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy import stats
import scipy.integrate

# Default layout parameters
LAYOUT_PARAMS = {
    'figsize': (6, 6),
    'legend_loc': 'upper right',
    'title_pad': 10,
    'x_label_pad': 10,
    'y_label_pad': 10,
    'legend_bbox_to_anchor': (1.0, 1.0),
    'plot_margins': {'left': 0.20, 'right': 0.95,
                     'top': 0.92, 'bottom': 0.15}
}

# Default rcParams for plots
DEFAULT_RC_PARAMS = {
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10
}

VALIDATION_PLOT_COUNTOUR_COLOR = "black"
VALIDATION_PLOT_COLOR = 'black'  # '#1f77b4'
GENERATED_PLOT_COLORS = ['#ff7f0e',
                         '#2ca02c',
                         '#d62728',
                         '#9467bd',
                         '#8c564b',
                         '#e377c2',
                         '#7f7f7f',
                         '#bcbd22',
                         '#17becf']


def _get_rc_params(rcParams_dict=None):
    """Helper function to merge default and user-provided rcParams"""
    rc_params = DEFAULT_RC_PARAMS.copy()
    if rcParams_dict is not None:
        rc_params.update(rcParams_dict)
    return rc_params


def plot_unconditional_metrics(datapath,
                               voxel_size_um=None,
                               min_porosity=-np.inf,
                               max_porosity=np.inf,
                               filter_dict=None,
                               plot_tag='',
                               show_permeability=True,
                               show_psd=True,
                               nbins=20,
                               use_log_properties=True,
                               convert_nan_to_zero=False,
                               which_porosity='raw',
                               max_value_dict=None,
                               rcParams_dict=None,
                               has_new_properties=False):
    generated_datapaths = {'Generated': datapath}
    validation_datapath = datapath
    savepath = None
    plot_unconditional_metrics_group(generated_datapaths,
                                     validation_datapath,
                                     savepath,
                                     voxel_size_um,
                                     min_porosity,
                                     max_porosity,
                                     filter_dict,
                                     plot_tag,
                                     show_permeability,
                                     show_psd,
                                     nbins,
                                     use_log_properties,
                                     convert_nan_to_zero,
                                     which_porosity,
                                     max_value_dict,
                                     rcParams_dict,
                                     has_new_properties)


def plot_unconditional_metrics_group(generated_datapaths,  # noqa: C901
                                     validation_datapath,
                                     savepath=None,
                                     voxel_size_um=None,
                                     min_porosity=-np.inf,
                                     max_porosity=np.inf,
                                     filter_dict=None,
                                     plot_tag='',
                                     show_permeability=True,
                                     show_psd=True,
                                     nbins=20,
                                     use_log_properties=True,
                                     convert_nan_to_zero=False,
                                     which_porosity='raw',
                                     max_value_dict=None,
                                     rcParams_dict=None,
                                     has_new_properties=False):
    # Apply rcParams for this function scope
    with plt.rc_context(rc=_get_rc_params(rcParams_dict)):
        # voxel_size in um
        cfg_path = f"{validation_datapath}/config.yaml"

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

        valid_stats_path = f"{validation_datapath}/valid_stats.json"

        generated_stats_dict = {}
        for name, datapath in generated_datapaths.items():
            generated_stats_path = f"{datapath}/generated_stats.json"
            with open(generated_stats_path, "r") as f:
                generated_stats = json.load(f)
                if filter_dict is not None:
                    generated_stats = {k: v for k, v in generated_stats.items() if filter_dict.get(k, False)}
                generated_stats_dict[name] = generated_stats

        with open(valid_stats_path, "r") as f:
            valid_stats = json.load(f)

        if which_porosity == 'effective':
            generated_effective_porosities_dict = {
                name: extract_property(stats, 'effective_porosity')[0]
                for name, stats in generated_stats_dict.items()
            }
            valid_effective_porosities, _ = extract_property(valid_stats, 'effective_porosity')
            porosity_for_filter_dict = generated_effective_porosities_dict
        elif which_porosity == 'raw':
            generated_raw_porosities_dict = {
                name: extract_property(stats, 'porosity')[0]
                for name, stats in generated_stats_dict.items()
            }
            valid_raw_porosities, _ = extract_property(valid_stats, 'porosity')
            porosity_for_filter_dict = generated_raw_porosities_dict
        else:  # both
            generated_raw_porosities_dict = {
                name: extract_property(stats, 'porosity')[0]
                for name, stats in generated_stats_dict.items()
            }
            valid_raw_porosities, _ = extract_property(valid_stats, 'porosity')
            generated_effective_porosities_dict = {
                name: extract_property(stats, 'effective_porosity')[0]
                for name, stats in generated_stats_dict.items()
            }
            valid_effective_porosities, _ = extract_property(valid_stats, 'effective_porosity')
            porosity_for_filter_dict = generated_raw_porosities_dict

        generated_surface_area_densities_dict = {
            name: extract_property(stats, 'surface_area_density')[0]
            for name, stats in generated_stats_dict.items()
        }
        valid_surface_area_densities, _ = extract_property(valid_stats, 'surface_area_density')

        if has_new_properties:
            generated_mean_curvature_dict = {
                name: extract_property(stats, 'mean_curvature')[0]
                for name, stats in generated_stats_dict.items()
            }
            valid_mean_curvature, _ = extract_property(valid_stats, 'mean_curvature')

            generated_euler_number_densities_dict = {
                name: extract_property(stats, 'euler_number_density')[0]
                for name, stats in generated_stats_dict.items()
            }
            valid_euler_number_densities, _ = extract_property(valid_stats, 'euler_number_density')
            has_new_properties = True
        else:
            has_new_properties = False
            generated_mean_curvature_dict = {}
            valid_mean_curvature = None
            generated_euler_number_densities_dict = {}
            valid_euler_number_densities = None

        generated_log_momenta_dict = {
            name: extract_property(stats, 'log_momenta')[0]
            for name, stats in generated_stats_dict.items()
        }
        valid_log_momenta, _ = extract_property(valid_stats, 'log_momenta')

        ind_dict = {name: (porosity > min_porosity).flatten() & (porosity < max_porosity).flatten()
                    for name, porosity in porosity_for_filter_dict.items()}

        if show_permeability:
            generated_permeabilities_dict = {name: extract_property(stats, 'permeability')[0]
                                             for name, stats in generated_stats_dict.items()}
            valid_permeabilities, _ = extract_property(valid_stats, 'permeability')

            generated_log_permeabilities_dict = {}
            for name, permeabilities in generated_permeabilities_dict.items():
                permeabilities = permeabilities[ind_dict[name]]
                # log_permeabilities = np.log10(np.prod(permeabilities, axis=1)**(1/3))
                log_permeabilities = np.log10(permeabilities[:, 2])  # z-direction
                log_permeabilities[~np.isfinite(log_permeabilities)] = -np.inf
                generated_log_permeabilities_dict[name] = log_permeabilities

            # valid_log_permeabilities = np.log10(np.prod(valid_permeabilities, axis=1)**(1/3))
            valid_log_permeabilities = np.log10(valid_permeabilities[:, 2])  # z-direction
            valid_log_permeabilities[~np.isfinite(valid_log_permeabilities)] = -np.inf

            if convert_nan_to_zero:
                for name in generated_log_permeabilities_dict:
                    generated_log_permeabilities_dict[name][~np.isfinite(generated_log_permeabilities_dict[name])] = -np.inf
                valid_log_permeabilities[~np.isfinite(valid_log_permeabilities)] = -np.inf
            else:
                for name in generated_log_permeabilities_dict:
                    generated_log_permeabilities_dict[name] = generated_log_permeabilities_dict[name][
                        np.isfinite(generated_log_permeabilities_dict[name])]
                valid_log_permeabilities = valid_log_permeabilities[np.isfinite(valid_log_permeabilities)]

        # Apply filter to all properties
        if which_porosity == 'effective':
            generated_effective_porosities_dict = {name: porosities[ind_dict[name]]
                                                   for name, porosities in generated_effective_porosities_dict.items()}
        elif which_porosity == 'raw':
            generated_raw_porosities_dict = {name: porosities[ind_dict[name]]
                                             for name, porosities in generated_raw_porosities_dict.items()}
        else:  # both
            generated_raw_porosities_dict = {name: porosities[ind_dict[name]]
                                             for name, porosities in generated_raw_porosities_dict.items()}
            generated_effective_porosities_dict = {name: porosities[ind_dict[name]]
                                                   for name, porosities in generated_effective_porosities_dict.items()}

        generated_surface_area_densities_dict = {name: densities[ind_dict[name]]
                                                 for name, densities in generated_surface_area_densities_dict.items()}
        if has_new_properties:
            generated_mean_curvature_dict = {name: curvatures[ind_dict[name]]
                                             for name, curvatures in generated_mean_curvature_dict.items()}
            generated_euler_number_densities_dict = {name: densities[ind_dict[name]]
                                                     for name, densities in generated_euler_number_densities_dict.items()}
        generated_log_momenta_dict = {name: momenta[ind_dict[name]]
                                      for name, momenta in generated_log_momenta_dict.items()}

        # Unit conversion
        # Permeability is already in Darcy
        for name in generated_surface_area_densities_dict:
            generated_surface_area_densities_dict[name] = generated_surface_area_densities_dict[name] / voxel_size_um  # (1/um)
        valid_surface_area_densities = valid_surface_area_densities / voxel_size_um  # (1/um)
        print(has_new_properties)
        if has_new_properties:
            for name in generated_mean_curvature_dict:
                generated_mean_curvature_dict[name] = generated_mean_curvature_dict[name] / voxel_size_um  # (1/um)
            for name in generated_euler_number_densities_dict:
                print(name, generated_euler_number_densities_dict[name])
                generated_euler_number_densities_dict[name] = generated_euler_number_densities_dict[name] / voxel_size_um**3  # (1/um^3)
            valid_mean_curvature = valid_mean_curvature / voxel_size_um  # (1/um)
            print(valid_euler_number_densities)
            valid_euler_number_densities = valid_euler_number_densities / voxel_size_um**3  # (1/um^3)

        log_momenta_conversion = np.log(np.array([voxel_size_um**(i+1)
                                                for i in range(next(iter(generated_log_momenta_dict.values())).shape[1])]))

        for name in generated_log_momenta_dict:
            generated_log_momenta_dict[name] = generated_log_momenta_dict[name] + log_momenta_conversion
        valid_log_momenta = valid_log_momenta + log_momenta_conversion

        generated_log_mean_pore_size_dict = {name: np.log10(np.exp(momenta[:, 0]))
                                            for name, momenta in generated_log_momenta_dict.items()}
        valid_log_mean_pore_size = np.log10(np.exp(valid_log_momenta[:, 0]))

        if not use_log_properties:
            # Remove the logarithms, but keep the name "log" for backward compatibility
            generated_log_mean_pore_size_dict = {name: 10**size
                                               for name, size in generated_log_mean_pore_size_dict.items()}
            valid_log_mean_pore_size = 10**valid_log_mean_pore_size
            generated_log_permeabilities_dict = {name: 10**perms
                                               for name, perms in generated_log_permeabilities_dict.items()}
            valid_log_permeabilities = 10**valid_log_permeabilities

        # Initialize data lists based on which_porosity
        if which_porosity == 'effective':
            generated_data_dict = {name: [generated_effective_porosities_dict[name],
                                          generated_surface_area_densities_dict[name],
                                          generated_log_mean_pore_size_dict[name]]
                                 for name in generated_datapaths}
            valid_data = [valid_effective_porosities,
                         valid_surface_area_densities,
                         valid_log_mean_pore_size]
            properties = ['effective_porosity', 'surface_area_density']
            labels = ['Effective Porosity', 'Surface Area Density']
            units = [r"$\phi$", r"$1/\mu m$"]
        elif which_porosity == 'raw':
            generated_data_dict = {name: [generated_raw_porosities_dict[name],
                                        generated_surface_area_densities_dict[name],
                                        generated_log_mean_pore_size_dict[name]]
                                 for name in generated_datapaths}
            valid_data = [valid_raw_porosities,
                         valid_surface_area_densities,
                         valid_log_mean_pore_size]
            properties = ['porosity', 'surface_area_density']
            labels = ['Porosity', 'Surface Area Density']
            units = [r"$\phi$", r"$1/\mu m$"]
        else:  # both
            generated_data_dict = {name: [generated_raw_porosities_dict[name],
                                        generated_effective_porosities_dict[name],
                                        generated_surface_area_densities_dict[name],
                                        generated_log_mean_pore_size_dict[name]]
                                 for name in generated_datapaths}
            valid_data = [valid_raw_porosities,
                         valid_effective_porosities,
                         valid_surface_area_densities,
                         valid_log_mean_pore_size]
            properties = ['porosity', 'effective_porosity', 'surface_area_density']
            labels = ['Porosity', 'Effective Porosity', 'Surface Area Density']
            units = [r"$\phi$", r"$\phi$", r"$1/\mu m$"]
        if use_log_properties:
            properties.append('log_mean_pore_size')
            labels.append('Log10 Mean Pore Size')
            units.append(r"$\log \,\mu m$")
        else:
            properties.append('mean_pore_size')
            labels.append('Mean Pore Size')
            units.append(r"$\mu m$")
        if has_new_properties:
            for name in generated_data_dict:
                generated_data_dict[name].append(generated_mean_curvature_dict[name])
            valid_data.append(valid_mean_curvature)
            properties.append('mean_curvature')
            labels.append('Mean Curvature')
            units.append(r"$\mu m^{-1}$")
            for name in generated_data_dict:
                generated_data_dict[name].append(generated_euler_number_densities_dict[name])
            valid_data.append(valid_euler_number_densities)
            properties.append('euler_number_density')
            labels.append('Euler Number Density')
            units.append(r"$\mu m^{-3}$")

        if show_permeability:
            for name in generated_data_dict:
                generated_data_dict[name].append(generated_log_permeabilities_dict[name])
            valid_data.append(valid_log_permeabilities)
            if use_log_properties:
                properties.append('log_permeability')
                labels.append('Log10-Permeability')
                units.append(r"$\log \text{ Darcy}$")
            else:
                properties.append('permeability')
                labels.append('Permeability')
                units.append(r"$\text{Darcy}$")

        print("Plotting histograms")
        figs1 = plot_histograms(generated_data_dict, valid_data, properties, labels, units, nbins, max_value_dict=max_value_dict, layout_params=LAYOUT_PARAMS)
        print("Plotting KDEs")
        figs1_kde, divergences = plot_kde(generated_data_dict, valid_data, properties, labels, units, max_value_dict=max_value_dict, layout_params=LAYOUT_PARAMS)
        print("Plotting boxplots")
        fig2 = plot_boxplots(generated_data_dict, valid_data, properties, labels, units, layout_params=LAYOUT_PARAMS)
        print("Plotting TPC")
        fig3, tpc_divergences = plot_two_point_correlation_comparison(generated_stats_dict, valid_stats, voxel_size_um, layout_params=LAYOUT_PARAMS)
        divergences['tpc_divergences'] = tpc_divergences
        print("Plotting PSD")
        if show_psd:
            fig4, psd_divergences = plot_pore_size_distribution(generated_stats_dict, valid_stats, voxel_size_um)
            divergences['pore_size_distribution'] = psd_divergences

        if savepath is None:
            savepath = pathlib.Path(f"{validation_datapath}/figures")
        else:
            savepath = pathlib.Path(savepath)
        os.makedirs(savepath, exist_ok=True)

        if plot_tag != '':
            plot_tag = f"_{plot_tag}"

        # Save histogram and KDE plots for each property
        for i, prop in enumerate(properties):
            figs1[i].savefig(savepath / f"stats_histogram_{prop}{plot_tag}.png", dpi=300)
            figs1_kde[i].savefig(savepath / f"stats_kde_{prop}{plot_tag}.png", dpi=300)

        for i, prop in enumerate(properties):
            fig2[i].savefig(savepath / f"stats_boxplots_{prop}{plot_tag}.png", dpi=300)
        fig3.savefig(savepath / f"tpc_comparison{plot_tag}.png", dpi=300)
        if show_psd:
            fig4.savefig(savepath / f"psd_comparison{plot_tag}.png", dpi=300)

        # Save KL divergences to JSON file
        divergences_path = savepath / f"stats_divergences{plot_tag}.json"
        with open(divergences_path, "w") as f:
            json.dump(divergences, f, indent=4)


def plot_conditional_metrics(datapath, savepath=None, voxel_size_um=None, filter_dict=None, plot_tag='', rcParams_dict=None):
    # Apply rcParams for this function scope
    with plt.rc_context(rc=_get_rc_params(rcParams_dict)):
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
            ax.legend(loc='upper right')
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
                ax.legend(loc='upper right')
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
                ax.legend(loc='upper right')
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
            for ax in axs:
                ax.plot(x_cond_tpc_dist, x_cond_tpc_prob, color='black', label='Condition')

                ax.set_xlabel(r"$r$ $(\mu m)$")
                ax.set_ylabel(r"$s_2$")
                ax.legend(loc='upper right')
                ax.set_title("Two-point Correlation")
        if savepath is None:
            savefolder = pathlib.Path(f"{datapath}/figures")
        else:
            savefolder = pathlib.Path(savepath)
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


def plot_cond_porosity(datapaths, conditions, savepath=None, nsamples=100, bins=10, filter_dict=None, plot_tag='', legend_loc='upper left', rcParams_dict=None):
    # Apply rcParams for this function scope
    with plt.rc_context(rc=_get_rc_params(rcParams_dict)):
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
            ax.vlines(x=conditions[i], color='black', ymin=0, ymax=20, linewidth=3)
            # Plot the histograms
            ax.hist(gen, bins=bins, density=False, alpha=0.5, label=f'Condition = {conditions[i]}')
        validation = np.concatenate(validation)
        nconds = len(conditions)
        ax.hist(validation, bins=nconds*bins, density=False, label='Validation', histtype='step',
                color='grey', linewidth=2)
        ax.set_xlabel('Porosity values')
        # ax.set_ylabel('Density')
        ax.set_title(f'Porosity histograms - {nsamples} samples for each condition')
        ax.legend(loc=legend_loc)
        fig.tight_layout()
        plt.show()

        # Save the figure in a parent folder
        if savepath is None:
            savefolder = pathlib.Path(datapaths[0]).parent.parent / "figures"
        else:
            savefolder = pathlib.Path(savepath)
        os.makedirs(savefolder, exist_ok=True)
        if plot_tag != '':
            plot_tag = f"_{plot_tag}"
        fig.savefig(savefolder / f"cond_porosity{plot_tag}.png")
        plt.close(fig)


def plot_vae_reconstruction(datapath, nsamples=1, tag=None, rcParams_dict=None):
    # Apply rcParams for this function scope
    with plt.rc_context(rc=_get_rc_params(rcParams_dict)):
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


def plot_vae_reconstruction_errors(datapath: str | list[str], savefolder: str, rcParams_dict=None):
    # Apply rcParams for this function scope
    with plt.rc_context(rc=_get_rc_params(rcParams_dict)):
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
            # Hack: If value is a single number, convert to list for consistency
            try:
                len(value)
            except TypeError:
                value = np.array([value])
            # Hack to check if the value is empty,
            # or if there is any non-finite value
            check = np.all(np.isfinite(value)) and len(value) > 0
            if not check:
                print("Empty or faulty value for key", k)
                value = np.nan * np.ones_like(value)
            property_list.append(value)
            input_list.append(k)
    first_property_shape = property_list[0].shape
    # Replace any properties with different shapes with nan arrays
    for i in range(len(property_list)):
        if property_list[i].shape != first_property_shape:
            print("Empty or faulty value for key", k)
            property_list[i] = np.nan * np.ones(first_property_shape)
    # print([property_list[i].shape for i in range(len(property_list)) if property_list[i].shape != property_list[0].shape])
    # print(property_list)
    property_list = np.stack(property_list, axis=0)
    input_list = np.array(input_list)
    return property_list, input_list


def plot_boxplots(generated_data_dict, valid_data, properties, labels, units,
                 contour=False, layout_params=None, rcParams_dict=None):
    # Apply rcParams for this function scope
    with plt.rc_context(rc=_get_rc_params(rcParams_dict)):
        # Default layout parameters
        default_layout = {
            'figsize': (6, 6),
            'legend_loc': 'upper right',
            'title_pad': 10,
            'x_label_pad': 10,
            'y_label_pad': 10,
            'legend_bbox_to_anchor': (0.98, 0.98),
            'plot_margins': {'left': 0.15, 'right': 0.95,
                             'top': 0.92, 'bottom': 0.15}
        }

        # Update with any user-provided parameters
        if layout_params is not None:
            default_layout.update(layout_params)

        figs = []

        for i, (prop, label, unit) in enumerate(zip(properties, labels, units)):
            fig, ax = plt.subplots(1, 1, figsize=default_layout['figsize'])

            # Create list of data for boxplot
            boxplot_data = []
            boxplot_labels = []

            # Add validation data first
            # Remove nan values
            valid_data_i = valid_data[i][np.isfinite(valid_data[i])]
            boxplot_data.append(valid_data_i.flatten())
            boxplot_labels.append('Validation')

            for name, gen_data in generated_data_dict.items():
                # Remove nan values
                gen_data_i = gen_data[i][np.isfinite(gen_data[i])]
                boxplot_data.append(gen_data_i.flatten())
                boxplot_labels.append(name)

            colors = [VALIDATION_PLOT_COLOR] + GENERATED_PLOT_COLORS[:len(boxplot_data)-1]

            if contour:
                # Plot just the outlines
                bp = ax.boxplot(boxplot_data, labels=boxplot_labels, patch_artist=False)
            else:
                # Plot filled boxes
                bp = ax.boxplot(boxplot_data, labels=boxplot_labels, patch_artist=True)
                # Color the boxes
                for j, box in enumerate(bp['boxes']):
                    color = colors[j]
                    if j == 0:  # Validation box
                        box.set_facecolor(color)
                    else:  # Generated data boxes
                        box.set_facecolor(color)

            # Set labels and title with consistent padding
            ax.set_ylabel(unit, labelpad=default_layout['y_label_pad'])
            ax.set_title(label, pad=default_layout['title_pad'])

            # Adjust subplot parameters for consistent margins
            plt.subplots_adjust(**default_layout['plot_margins'])

            figs.append(fig)

        return figs


def plot_two_point_correlation_comparison(generated_stats_dict, valid_stats, voxel_size_um, ind=None, layout_params=None):
    """Plot comparison of two point correlation between generated and validation data.

    Parameters
    ----------
    generated_stats_dict : dict
        Dictionary containing generated statistics for each model
    valid_stats : dict
        Dictionary containing validation statistics
    voxel_size_um : float
        Voxel size in micrometers
    ind : array-like, optional
        Indices to select from generated data
    layout_params : dict, optional
        Dictionary of layout parameters to override defaults

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the two point correlation comparison plot
    divergences : dict
        Dictionary containing KL divergence, Hellinger distance and mean relative error
    """
    # Default layout parameters
    default_layout = {
        'figsize': (6, 6),
        'legend_loc': 'upper right',
        'title_pad': 10,
        'x_label_pad': 10,
        'y_label_pad': 10,
        'legend_bbox_to_anchor': (0.98, 0.98),
        'plot_margins': {'left': 0.15, 'right': 0.95,
                        'top': 0.92, 'bottom': 0.15}
    }

    # Update with any user-provided parameters
    if layout_params is not None:
        default_layout.update(layout_params)

    # Extract validation data
    valid_tpc_dist, _ = extract_property(valid_stats, 'tpc_dist')
    valid_tpc_prob, _ = extract_property(valid_stats, 'tpc_prob')
    valid_tpc_dist *= voxel_size_um  # Convert to um

    # Calculate validation statistics
    mean_valid_tpc_prob = np.mean(valid_tpc_prob, 0)
    std_valid_tpc_prob = np.std(valid_tpc_prob, 0)
    x_valid = valid_tpc_dist.mean(axis=0)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=default_layout['figsize'])

    # Plot validation data
    linewidth = 2
    ax.plot(x_valid, mean_valid_tpc_prob-2*std_valid_tpc_prob,
            color=VALIDATION_PLOT_COLOR, linestyle='--', linewidth=linewidth, alpha=0.3)
    ax.plot(x_valid, mean_valid_tpc_prob+2*std_valid_tpc_prob,
            color=VALIDATION_PLOT_COLOR, linestyle='--', linewidth=linewidth, alpha=0.3)

    ax.plot(x_valid, mean_valid_tpc_prob, color=VALIDATION_PLOT_COLOR,
            label='Validation', linewidth=2)

    # Initialize divergences dictionary
    divergences = {}

    # Process each generated dataset
    for i, (name, generated_stats) in enumerate(generated_stats_dict.items()):
        generated_tpc_dist, _ = extract_property(generated_stats, 'tpc_dist')
        generated_tpc_prob, _ = extract_property(generated_stats, 'tpc_prob')
        generated_tpc_dist *= voxel_size_um  # Convert to um

        if ind is not None:
            generated_tpc_dist = generated_tpc_dist[ind]
            generated_tpc_prob = generated_tpc_prob[ind]

        mean_generated_tpc_prob = np.mean(generated_tpc_prob, 0)
        std_generated_tpc_prob = np.std(generated_tpc_prob, 0)
        x_generated = generated_tpc_dist.mean(axis=0)

        # Calculate divergences
        kl_div = np.mean([hellinger_distance_kde_mc(generated_tpc_prob[:, i],
                                                   valid_tpc_prob[:, i])
                         for i in range(generated_tpc_prob.shape[1])])
        hellinger_dist = np.mean([hellinger_distance_kde_mc(generated_tpc_prob[:, i],
                                                          valid_tpc_prob[:, i])
                                for i in range(generated_tpc_prob.shape[1])])
        rel_error = np.mean(np.abs(mean_generated_tpc_prob - mean_valid_tpc_prob) /
                          mean_valid_tpc_prob)

        divergences[name] = {
            'kl_divergence': float(kl_div),
            'hellinger_distance': float(hellinger_dist),
            'mean_relative_error': float(rel_error)
        }

        # Plot generated data
        y1 = mean_generated_tpc_prob-2*std_generated_tpc_prob
        y2 = mean_generated_tpc_prob+2*std_generated_tpc_prob
        ax.fill_between(x_generated,
                       y1=y1,
                       y2=y2,
                       alpha=0.2,
                       color=GENERATED_PLOT_COLORS[i])
        ax.plot(x_generated, mean_generated_tpc_prob, label=name,
                color=GENERATED_PLOT_COLORS[i])

    ax.set_xlabel(r"$r$ $(\mu m)$", labelpad=default_layout['x_label_pad'])
    ax.set_ylabel(r"$s_2$", labelpad=default_layout['y_label_pad'])
    ax.set_ylim(bottom=0.0)
    # ax.legend(loc=default_layout['legend_loc'],
    #          bbox_to_anchor=default_layout['legend_bbox_to_anchor'])
    ax.set_title("Two-point correlation", pad=default_layout['title_pad'])

    plt.subplots_adjust(**default_layout['plot_margins'])

    return fig, divergences


def plot_histograms(generated_data_dict,
                    valid_data,
                    properties,
                    labels,
                    units,
                    nbins=20,
                    contour=False,
                    max_value_dict=None,
                    layout_params=None,
                    rcParams_dict=None):
    # Apply rcParams for this function scope
    with plt.rc_context(rc=_get_rc_params(rcParams_dict)):
        # Default layout parameters
        default_layout = {
            'figsize': (6, 6),
            'legend_loc': 'upper right',
            'title_pad': 10,
            'x_label_pad': 10,
            'y_label_pad': 10,
            'legend_bbox_to_anchor': (0.98, 0.98),
            'plot_margins': {'left': 0.15, 'right': 0.95,
                            'top': 0.92, 'bottom': 0.15}
        }

        # Update with any user-provided parameters
        if layout_params is not None:
            default_layout.update(layout_params)

        figs = []
        if max_value_dict is None:
            max_value_dict = {}

        for i, (val, prop, label, unit) in enumerate(zip(valid_data, properties, labels, units)):
            fig, ax = plt.subplots(1, 1, figsize=default_layout['figsize'])

            # Remove nan values from validation data
            val = val[np.isfinite(val)]
            min_value = np.min(val)
            max_value = np.max(val)

            # Find global min/max including all generated datasets
            for j, (name, gen_data) in enumerate(generated_data_dict.items()):
                gen = gen_data[i][np.isfinite(gen_data[i])]
                min_value = min(min_value, np.min(gen))
                max_value = max(max_value, np.max(gen))

            if prop in max_value_dict:
                max_value = max_value_dict[prop]

            bins = np.linspace(min_value, max_value, nbins)

            # Plot validation histogram
            ax.hist(val.flatten(), bins=bins, density=True, histtype='step',
                    color=VALIDATION_PLOT_COUNTOUR_COLOR, label='Validation')

            # Plot generated histograms
            for j, (name, gen_data) in enumerate(generated_data_dict.items()):
                gen = gen_data[i][np.isfinite(gen_data[i])]
                if not contour:
                    ax.hist(gen.flatten(), bins=bins, density=True, alpha=0.4, label=name,
                            color=GENERATED_PLOT_COLORS[j])
                else:
                    ax.hist(gen.flatten(), bins=bins, density=True, alpha=0.4, label=name,
                            color=GENERATED_PLOT_COLORS[j], histtype='step' if contour else 'bar')

            ax.set_xlabel(unit, labelpad=default_layout['x_label_pad'])
            ax.set_ylabel("Density", labelpad=default_layout['y_label_pad'])
            ax.set_title(label, pad=default_layout['title_pad'])
            if label == 'Porosity':
                ax.legend(loc=default_layout['legend_loc'],
                        bbox_to_anchor=default_layout['legend_bbox_to_anchor'])

            plt.subplots_adjust(**default_layout['plot_margins'])
            figs.append(fig)

        return figs


def plot_kde(generated_data_dict,
             valid_data,
             properties,
             labels,
             units,
             contour=False,
             max_value_dict=None,
             layout_params=None,
             rcParams_dict=None):
    # Apply rcParams for this function scope
    with plt.rc_context(rc=_get_rc_params(rcParams_dict)):
        # Default layout parameters
        default_layout = {
            'figsize': (6, 6),
            'legend_loc': 'upper right',
            'title_pad': 10,
            'x_label_pad': 10,
            'y_label_pad': 10,
            'legend_bbox_to_anchor': (0.98, 0.98),
            'plot_margins': {'left': 0.15, 'right': 0.95,
                            'top': 0.92, 'bottom': 0.15}
        }

        # Update with any user-provided parameters
        if layout_params is not None:
            default_layout.update(layout_params)

        figs = []
        divergences = {}

        if max_value_dict is None:
            max_value_dict = {}

        for i, (val, prop, label, unit) in enumerate(zip(valid_data, properties, labels, units)):
            fig, ax = plt.subplots(1, 1, figsize=default_layout['figsize'])

            # Remove nan values from validation data
            val = val[np.isfinite(val)]
            min_value = np.min(val)
            max_value = np.max(val)
            if prop in max_value_dict:
                max_value = max_value_dict[prop]

            # Find global min/max including all generated datasets
            for name, gen_data in generated_data_dict.items():
                gen = gen_data[i][np.isfinite(gen_data[i])]
                min_value = min(min_value, np.min(gen))
                max_value = max(max_value, np.max(gen))

            # Expand min and max values by 10%
            min_value = min_value - 0.2 * (max_value - min_value)
            max_value = max_value + 0.2 * (max_value - min_value)

            # In case of porosity only, expand 20% again
            if label == 'Porosity':
                min_value = min_value - 0.4 * (max_value - min_value)
                max_value = max_value + 0.4 * (max_value - min_value)

            if label in ["Porosity", "Surface Area Density", "Mean Pore Size", "Permeability"]:
                min_value = max(min_value, 0.0)

            if label in ["Porosity"]:
                max_value = min(max_value, 1.0)

            if prop in max_value_dict:
                max_value = max_value_dict[prop]

            # Plot validation KDE
            val_kde = stats.gaussian_kde(val.flatten())
            x_kde = np.linspace(min_value, max_value, 200)
            ax.plot(x_kde, val_kde(x_kde), color=VALIDATION_PLOT_COUNTOUR_COLOR,
                    label='Validation', linestyle='--', linewidth=2)

            # Plot generated KDEs and calculate divergences
            divergences[prop] = {}
            for j, (name, gen_data) in enumerate(generated_data_dict.items()):
                gen = gen_data[i][np.isfinite(gen_data[i])]
                gen_kde = stats.gaussian_kde(gen.flatten())

                if not contour:
                    ax.fill_between(x_kde, 0, gen_kde(x_kde), alpha=0.4,
                                  label=name, color=GENERATED_PLOT_COLORS[j])
                else:
                    ax.plot(x_kde, gen_kde(x_kde), label=name,
                          color=GENERATED_PLOT_COLORS[j])

                # Calculate divergence metrics
                kl_div = kl_divergence_kde_mc(val.flatten(), gen.flatten(),
                                            n_samples=10000)
                hellinger_dist = hellinger_distance_kde_mc(val.flatten(),
                                                        gen.flatten(),
                                                        n_samples=10000)
                if label in ['Porosity', 'Surface Area Density',
                            'Mean Pore Size', 'Permeability']:
                    rel_error = mean_relative_error(val.flatten(), gen.flatten())
                else:
                    rel_error = np.nan

                divergences[prop][name] = {
                    'kl_divergence': kl_div,
                    'hellinger_distance': hellinger_dist,
                    'mean_relative_error': rel_error
                }

            # Set labels and title with consistent padding
            ax.set_xlabel(unit, labelpad=default_layout['x_label_pad'])
            ax.set_ylabel("Density", labelpad=default_layout['y_label_pad'])
            ax.set_title(label, pad=default_layout['title_pad'])
            ax.set_ylim(bottom=0.0)

            # Set legend with consistent positioning
            if label == 'Porosity':
                ax.legend(loc=default_layout['legend_loc'],
                        bbox_to_anchor=default_layout['legend_bbox_to_anchor'],
                        borderaxespad=0)

            # Adjust subplot parameters for consistent margins
            plt.subplots_adjust(**default_layout['plot_margins'])

            figs.append(fig)

        return figs, divergences


def plot_pore_size_distribution(generated_stats_dict, valid_stats, voxel_size_um,
                               layout_params=None, rcParams_dict=None):
    # Apply rcParams for this function scope
    with plt.rc_context(rc=_get_rc_params(rcParams_dict)):
        # Default layout parameters
        default_layout = {
            'figsize': (6, 6),
            'legend_loc': 'upper right',
            'title_pad': 10,
            'x_label_pad': 10,
            'y_label_pad': 10,
            'legend_bbox_to_anchor': (0.98, 0.98),
            'plot_margins': {'left': 0.15, 'right': 0.95,
                            'top': 0.92, 'bottom': 0.15}
        }

        # Update with any user-provided parameters
        if layout_params is not None:
            default_layout.update(layout_params)

        # Extract validation data
        valid_psd_pdf, _ = extract_property(valid_stats, 'psd_pdf')
        valid_psd_centers, _ = extract_property(valid_stats, 'psd_centers')
        valid_psd_centers *= voxel_size_um  # Convert to um

        # Create a common x-axis for interpolation
        min_centers = np.inf
        max_centers = -np.inf

        # Extract generated data and find min/max centers
        generated_psd_pdf_dict = {}
        generated_psd_centers_dict = {}
        for name, generated_stats in generated_stats_dict.items():
            generated_psd_pdf, _ = extract_property(generated_stats, 'psd_pdf')
            generated_psd_centers, _ = extract_property(generated_stats, 'psd_centers')
            generated_psd_centers *= voxel_size_um  # Convert to um

            generated_psd_pdf_dict[name] = generated_psd_pdf
            generated_psd_centers_dict[name] = generated_psd_centers

            min_centers = min(min_centers, np.min(generated_psd_centers))
            max_centers = max(max_centers, np.max(generated_psd_centers))

        min_centers = min(min_centers, np.min(valid_psd_centers))
        max_centers = max(max_centers, np.max(valid_psd_centers))

        x_interp = np.logspace(np.log10(min_centers), np.log10(max_centers), num=100)

        # PDF Plot using direct KDE for each sample
        def direct_kde_plot(ax, data, centers, color, label, contour=True):
            # Remove nan values from the data
            kde_estimates = []
            for sample, sample_centers in zip(data, centers):
                # Remove nan values from the data
                sample_finite = sample[np.isfinite(sample) & np.isfinite(sample_centers)]
                sample_centers_finite = sample_centers[np.isfinite(sample_centers) & np.isfinite(sample)]
                if len(sample_finite) > 0 and len(sample_centers_finite) > 0:
                    # Count the number of non-zero values in sample_finite
                    non_zero_count = np.sum(sample_finite > 0)
                    if non_zero_count < 2:
                        continue
                    kde = stats.gaussian_kde(sample_centers_finite, weights=sample_finite)
                    kde_estimates.append(kde(x_interp))
            kde_estimates = np.array(kde_estimates)
            kde_mean = np.mean(kde_estimates, axis=0)
            kde_std = np.std(kde_estimates, axis=0)

            linewidth = 2
            if contour:
                ax.fill_between(x_interp, np.maximum(kde_mean - 2*kde_std, 0),
                               kde_mean + 2*kde_std, alpha=0.3, color=color)
            else:
                ax.plot(x_interp, kde_mean - 2*kde_std, color=color, linestyle='--',
                       linewidth=linewidth, alpha=0.3)
                ax.plot(x_interp, kde_mean + 2*kde_std, color=color, linestyle='--',
                       linewidth=linewidth, alpha=0.3)
            ax.plot(x_interp, kde_mean, color=color, label=label, linewidth=linewidth)

            return kde_estimates

        fig, ax = plt.subplots(1, 1, figsize=default_layout['figsize'])
        valid_kde_estimates = direct_kde_plot(ax, valid_psd_pdf, valid_psd_centers,
                                            VALIDATION_PLOT_COLOR, 'Validation', contour=False)

        divergences = {}
        for i, name in enumerate(generated_stats_dict):
            generated_kde_estimates = direct_kde_plot(ax, generated_psd_pdf_dict[name],
                                                    generated_psd_centers_dict[name],
                                                    GENERATED_PLOT_COLORS[i], name)

            divergences[name] = {
                'kl_divergence': np.nan,
                'hellinger_distance': np.nan,
                'mean_relative_error': np.nan
            }

        ax.set_xlabel(r'Pore Size $(\mu m)$', labelpad=default_layout['x_label_pad'])
        ax.set_ylabel('Probability Density', labelpad=default_layout['y_label_pad'])
        ax.set_title('Pore Size Distribution - PDF', pad=default_layout['title_pad'])
        ax.set_ylim(bottom=0.0)
        ax.set_xscale('log')
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-1,1))
        plt.subplots_adjust(**default_layout['plot_margins'])

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
