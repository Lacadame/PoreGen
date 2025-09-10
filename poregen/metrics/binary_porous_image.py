from typing import Any
import os

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import torch
import torchmetrics
from torch import Tensor
from jaxtyping import Float
import porespy

from poregen.utils import inverse_cdf_histogram

"""
Deprecated
"""


class BinaryPorousImageMetrics(object):
    def __init__(self):
        pass

    def set_samples(
            self,
            generated_samples: Float[Tensor, "batch 1 H W"],  # noqa: F722
            valid_samples: Float[Tensor, "batch 1 H W"]  # noqa: F722
            ):
        generated_samples = generated_samples.detach().squeeze(1).cpu().numpy()
        valid_samples = valid_samples.detach().squeeze(1).cpu().numpy()
        binary_generated_samples = mean_binarize(generated_samples)
        binary_valid_samples = mean_binarize(valid_samples)
        self.generated_samples = generated_samples
        self.valid_samples = valid_samples
        self.binary_generated_samples = binary_generated_samples
        self.binary_valid_samples = binary_valid_samples

    def make_two_point_correlations(self,
                                    filter_noise=False,
                                    filter_threshold=0.5,
                                    filter_ind=3,
                                    record_oiginal=False):
        generated_two_point_correlation = []
        for x in self.binary_generated_samples:
            tpc = porespy.metrics.two_point_correlation(~x)
            generated_two_point_correlation.append(tpc)
        generated_scaled_tpc = [get_scaled_two_point_correlation(tpc)
                                for tpc in generated_two_point_correlation]
        if filter_noise:
            is_noise = np.array([y[filter_ind] < filter_threshold
                                 for (x, y) in generated_scaled_tpc])
            generated_samples = self.generated_samples[~is_noise]
            binary_generated_samples = self.binary_generated_samples[~is_noise]
            generated_scaled_tpc = [tpc for (tpc, is_noise)
                                    in zip(generated_scaled_tpc, is_noise)
                                    if not is_noise]
            generated_two_point_correlation = [
                tpc for (tpc, is_noise)
                in zip(generated_two_point_correlation,
                       is_noise)
                if not is_noise]
            if record_oiginal:
                self.generated_samples_unfiltered = (self.
                                                     generated_samples.
                                                     copy())
            self.generated_samples = generated_samples
            self.binary_generated_samples = binary_generated_samples
            self.is_noise = is_noise

        self.generated_scaled_tpc = generated_scaled_tpc
        self.generated_two_point_correlation = generated_two_point_correlation

        self.valid_two_point_correlation = []
        for x in self.binary_valid_samples:
            tpc = porespy.metrics.two_point_correlation(~x)
            self.valid_two_point_correlation.append(tpc)
        self.generated_porosity = [tpc.probability_scaled[0]
                                   for tpc in
                                   self.generated_two_point_correlation]
        self.valid_porosity = [tpc.probability_scaled[0]
                               for tpc in self.valid_two_point_correlation]
        self.generated_scaled_tpc = [get_scaled_two_point_correlation(tpc)
                                     for tpc
                                     in self.generated_two_point_correlation]
        self.valid_scaled_tpc = [get_scaled_two_point_correlation(tpc)
                                 for tpc in self.valid_two_point_correlation]

    def make_porosity_profile_local_thickness(self,
                                              bins=100):
        self.generated_porosity_profile = []
        for x in self.binary_generated_samples:
            local_thickness = porespy.filters.local_thickness(~x)
            local_thickness = local_thickness.flatten()
            radii = local_thickness[local_thickness > 0]
            log_radii = np.log(radii)
            log_radii_quantile = inverse_cdf_histogram(log_radii)
            self.generated_porosity_profile.append(log_radii_quantile)
            # psd = porespy.metrics.pore_size_distribution(local_thickness,
            #                                              log=True,
            #                                              bins=bins)
            # self.generated_porosity_profile.append(psd)
        self.valid_porosity_profile = []
        for x in self.binary_valid_samples:
            local_thickness = porespy.filters.local_thickness(~x)
            local_thickness = local_thickness.flatten()
            radii = local_thickness[local_thickness > 0]
            log_radii = np.log(radii)
            log_radii_quantile = inverse_cdf_histogram(log_radii)
            self.valid_porosity_profile.append(log_radii_quantile)
            # psd = porespy.metrics.pore_size_distribution(local_thickness,
            #                                              log=True,
            #                                              bins=bins)
            # self.valid_porosity_profile.append(psd)

    def make_boxcount(self,
                      bins=10):
        self.generated_boxcount = []
        for x in self.binary_generated_samples:
            boxcount = porespy.metrics.boxcount(~x, bins=bins)
            self.generated_boxcount.append(boxcount)
        self.valid_boxcount = []
        for x in self.binary_valid_samples:
            boxcount = porespy.metrics.boxcount(~x, bins=bins)
            self.valid_boxcount.append(boxcount)

    def calculate_distances(self,
                            dimension=2):
        # Calculate frechet inception distance of images
        if dimension==2:
            fid = (torchmetrics.
                image.
                fid.
                FrechetInceptionDistance(normalize=True))
            generated_frechet = (torch.tensor(self.binary_generated_samples,
                                            dtype=torch.float).
                                unsqueeze(1).
                                expand(-1, 3, -1, -1))
            valid_frechet = (torch.tensor(self.binary_valid_samples,
                                        dtype=torch.float).
                            unsqueeze(1).
                            expand(-1, 3, -1, -1))
            fid.update(generated_frechet, real=True)
            fid.update(valid_frechet, real=False)
            fid = fid.compute()
        else:
            fid = np.nan

        # Calculate frechet inception distance of porosity

        x_generated = np.stack(self.generated_porosity, axis=0)[..., None]
        x_valid = np.stack(self.valid_porosity, axis=0)[..., None]
        try:
            fid_porosity = gaussianized_wasserstein_distance(x_generated,
                                                             x_valid)
        except ValueError:
            fid_porosity = np.nan

        # Calculate hellinger distance of porosity

        # x_generated_round = np.round(x_generated, 4).squeeze(-1)
        # x_valid_round = np.round(x_valid, 4).squeeze(-1)

        # hellinger_porosity = hellinger_distance1D(x_generated,
        #                                           x_valid)
        hellinger_porosity = 0

        # Calculate frechet inception distance of two point correlation
        x_generated = np.stack([scaled_tpc[1]
                                for scaled_tpc in self.generated_scaled_tpc],
                               axis=0)
        x_valid = np.stack([scaled_tpc[1]
                            for scaled_tpc in self.valid_scaled_tpc],
                           axis=0)
        try:
            fid_tpc = gaussianized_wasserstein_distance(x_generated, x_valid)
        except ValueError:
            fid_tpc = np.nan
        return {'fid': fid,
                'fd_porosity': fid_porosity,
                'fd_tpc': fid_tpc,
                'hellinger_porosity': hellinger_porosity}

    def plot_porosity_profile_local_thickness(self):

        x_generated = np.linspace(0, 1)
        y_generated = np.stack([quantile(x_generated)
                                for quantile
                                in self.generated_porosity_profile], axis=0)
        mean_y_generated = np.mean(y_generated, axis=0)
        std_y_generated = np.std(y_generated, axis=0)

        x_valid = np.linspace(0, 1)
        y_valid = np.stack([quantile(x_valid)
                            for quantile
                            in self.valid_porosity_profile], axis=0)
        mean_y_valid = np.mean(y_valid, axis=0)
        std_y_valid = np.std(y_valid, axis=0)

        # Creating the plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.fill_between(x_generated,
                        y1=mean_y_generated-2*std_y_generated,
                        y2=mean_y_generated+2*std_y_generated,
                        color='red', alpha=0.2)

        ax.fill_between(x_valid,
                        y1=mean_y_valid-2*std_y_valid,
                        y2=mean_y_valid+2*std_y_valid,
                        color='blue', alpha=0.2)

        # Plotting the mean profiles
        ax.plot(x_generated, mean_y_generated, color='red', label='generated')
        ax.plot(x_valid, mean_y_valid, color='blue', label='valid')

        # Setting labels, title, and legend
        # ax.set_xlabel(r"$$")
        ax.set_xlabel(r"$p$")
        ax.set_ylabel(r"$\log(R)$")
        ax.set_title("Porosity Profile Local Thickness")
        ax.legend()

        fig.tight_layout()
        return fig

    def plot_boxcount(self):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        # Calculating log values for generated boxcount
        log_y_generated = np.log10(np.stack([boxcount.count
                                            for boxcount
                                            in self.generated_boxcount],
                                            axis=0))
        mean_log_y_generated = np.mean(log_y_generated, axis=0)
        std_log_y_generated = np.std(log_y_generated, axis=0)
        log_x_generated = np.log10(self.generated_boxcount[0].size)

        # Calculating log values for valid boxcount
        log_y_valid = np.log10(np.stack([boxcount.count
                                        for boxcount
                                        in self.valid_boxcount], axis=0))
        mean_log_y_valid = np.mean(log_y_valid, axis=0)
        std_log_y_valid = np.std(log_y_valid, axis=0)
        log_x_valid = np.log10(self.valid_boxcount[0].size)

        # Plotting with error bars for generated boxcount
        ax.fill_between(log_x_generated,
                        y1=mean_log_y_generated - 2 * std_log_y_generated,
                        y2=mean_log_y_generated + 2 * std_log_y_generated,
                        color='red', alpha=0.2)
        ax.plot(log_x_generated, mean_log_y_generated, 'r-', label='generated')

        # Plotting with error bars for valid boxcount
        ax.fill_between(log_x_valid,
                        y1=mean_log_y_valid - 2 * std_log_y_valid,
                        y2=mean_log_y_valid + 2 * std_log_y_valid,
                        color='blue', alpha=0.2)
        ax.plot(log_x_valid, mean_log_y_valid, 'b-', label='valid')

        ax.set_xlabel("Log(Size)")
        ax.set_ylabel("Log(Count)")
        ax.set_title("Boxcount Comparison")
        ax.legend()

        fig.tight_layout()
        return fig

    def plot_images(self,
                    num_maximum_images=None,
                    num_columns=4,
                    window_size=2,
                    which="generated",
                    binarized=True):
        if num_maximum_images is None:
            num_images = len(self.generated_samples)
        else:
            num_images = min(num_maximum_images, len(self.generated_samples))
        num_rows = int(np.ceil(num_images/num_columns))
        fig_width = num_columns*window_size
        fig_height = num_rows*window_size
        fig, axes = plt.subplots(num_rows,
                                 num_columns,
                                 figsize=(fig_width, fig_height))
        if which == "generated":
            if binarized:
                images = self.binary_generated_samples
            else:
                images = self.generated_samples
        elif which == "valid":
            if binarized:
                images = self.binary_valid_samples
            else:
                images = self.valid_samples
        else:
            raise ValueError("which must be 'generated' or 'valid'")
        for i, ax in enumerate(axes.flat):
            if i < num_images:
                img = images[i]
                ax.matshow(img, cmap='gray')
            # ax.set_xticks([])
            # ax.set_yticks([])
        fig.suptitle(which.capitalize())
        fig.tight_layout()
        return fig

    def plot_samples_histograms(self,
                                num_maximum_images=None,
                                num_columns=4,
                                window_size=2,
                                bins=100):
        if num_maximum_images is None:
            num_images = len(self.generated_samples)
        else:
            num_images = min(num_maximum_images, len(self.generated_samples))
        num_rows = int(np.ceil(num_images/num_columns))
        fig_width = num_columns*window_size
        fig_height = num_rows*window_size
        fig, axes = plt.subplots(num_rows,
                                 num_columns,
                                 figsize=(fig_width, fig_height))
        images = self.generated_samples

        for i, ax in enumerate(axes.flat):
            if i < num_images:
                img = images[i]
                ax.hist(img.flatten(), bins=bins, density=True)
        fig.tight_layout()
        return fig

    def plot_two_point_correlations(self):
        y_generated = np.stack([scaled_tpc[1]
                                for scaled_tpc in self.generated_scaled_tpc],
                               axis=0)
        y_valid = np.stack([scaled_tpc[1]
                            for scaled_tpc in self.valid_scaled_tpc],
                           axis=0)
        mean_y_generated = np.mean(y_generated, 0)
        std_y_generated = np.std(y_generated, 0)
        mean_y_valid = np.mean(y_valid, 0)
        std_y_valid = np.std(y_valid, 0)
        x_generated = self.generated_scaled_tpc[0][0]
        x_valid = self.valid_scaled_tpc[0][0]

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        ax.fill_between(x_generated,
                        y1=mean_y_generated-2*std_y_generated,
                        y2=mean_y_generated+2*std_y_generated,
                        color='red',
                        alpha=0.2)
        ax.fill_between(x_valid,
                        y1=mean_y_valid-2*std_y_valid,
                        y2=mean_y_valid+2*std_y_valid,
                        color='blue',
                        alpha=0.2)
        ax.plot(x_generated, mean_y_generated,
                color='red',
                label='generated')
        ax.plot(x_valid, mean_y_valid, color='blue', label='valid')
        ax.set_xlabel(r"$r$")
        ax.set_xlabel(r"$(s_2 - \phi^2)/(\phi - \phi^2)$")
        ax.legend()
        ax.set_title("Comparison of two point correlation")

        fig.tight_layout()
        return fig

    def plot_porosity_histograms(self, bins=10):
        fig, ax = plt.subplots(figsize=(4, 4))

        min_value = min(min(self.valid_porosity),
                        min(self.generated_porosity))
        # max_value = max(max(self.valid_porosity),
        #                 max(self.generated_porosity))

        bins = np.linspace(min_value, 0.5, bins)

        ax.hist(self.generated_porosity,
                bins=bins, density=True,
                color='red',
                alpha=0.5,
                label='generated')
        ax.hist(self.valid_porosity,
                bins=bins, density=True,
                color='blue',
                alpha=0.5,
                label='valid')
        ax.set_xlabel(r"$\phi$")
        ax.set_ylabel(r"Frequency")
        ax.set_title("Porosity histogram")
        ax.legend()
        return fig

    def get_rejection_rate(self):
        if not hasattr(self, 'is_noise'):
            return 0.0
        else:
            return np.mean(self.is_noise)

    def make_report(self, mainpath, folder, filter_noise=True, dimension=2):
        try:
            os.mkdir(mainpath/folder)
        except FileExistsError:
            pass
        self.make_two_point_correlations(filter_noise=filter_noise)
        # self.make_porosity_profile_local_thickness(bins=50)
        distances = self.calculate_distances(dimension=dimension)
        rejection_rate = self.get_rejection_rate()
        if dimension==2:        # TODO: create 3D image plot
            fig = self.plot_images(num_maximum_images=12,
                                num_columns=4,
                                window_size=2,
                                which="generated",
                                binarized=True)
            fig.savefig(mainpath/folder/"generated_images.png")
            plt.close(fig)
            fig = self.plot_images(num_maximum_images=12,
                                num_columns=4,
                                window_size=2,
                                which="valid",
                                binarized=True)
            fig.savefig(mainpath/folder/"valid_images.png")
            plt.close(fig)

        fig = self.plot_two_point_correlations()
        fig.savefig(mainpath/folder/"two_point_correlations.png")
        plt.close(fig)
        fig = self.plot_porosity_histograms(bins=20)
        fig.savefig(mainpath/folder/"porosity_histogram.png")
        plt.close(fig)
        data = {'fid': distances['fid'],
                'fd_porosity': distances['fd_porosity'],
                'hellinger_porosity': distances['hellinger_porosity'],
                'fd_tpc': distances['fd_tpc'],
                'rejection_rate': rejection_rate}
        np.savez(mainpath/folder/"metrics.npz", **data)
        return


def mean_binarize(x: Float[np.ndarray, "batch *shape"]  # noqa: F821
                  ) -> Float[np.ndarray, "batch *shape"]:  # noqa: F821
    dims = tuple(range(1, len(x.shape)))
    return x > (x.mean(axis=dims, keepdims=True))


def get_scaled_two_point_correlation(
        tpc: Any,  # TODO: Actual type here
        ) -> tuple[np.ndarray, np.ndarray]:
    x = tpc.distance
    porosity = tpc.probability_scaled[0]
    if (porosity - porosity**2) == 0:
        return x, np.zeros_like(x)          # avoid division by zero
    ub = porosity
    lb = porosity**2
    y0 = tpc.probability_scaled
    y = (y0 - lb)/(ub - lb)
    return x, y


def gaussianized_wasserstein_distance(x1, x2):
    mean1, mean2 = x1.mean(axis=0), x2.mean(axis=0)
    cov1, cov2 = np.cov(x1, rowvar=False), np.cov(x2, rowvar=False)
    cov1, cov2 = np.atleast_2d(cov1), np.atleast_2d(cov2)
    mean_term = np.linalg.norm(mean1 - mean2)**2
    covprod = cov1 @ cov2
    covprod = 0.5*(covprod + covprod.T)
    cov_term = np.trace(cov1 + cov2 - 2 * np.real(scipy.linalg.sqrtm(covprod)))
    return np.sqrt(mean_term + cov_term)


# Approximate Hellinger estimator: small noise is added
# to make the distribution continuous. Then, any repeated value is removed.

def approx_affinity(x1: Float[np.ndarray, "batch"],     # noqa: F821
                    x2: Float[np.ndarray, "batch"],     # noqa: F821
                    noise=1e-5     # noqa: F821
                    ):
    x1 = x1 + np.random.randn(x1.shape)*noise
    x2 = x2 + np.random.randn(x2.shape)*noise
    x1 = np.unique(x1)
    x2 = np.unique(x2)
    n1 = len(x1)
    n2 = len(x2)
    min = 0
    i = 0
    P2 = np.zeros(n1-1)
    P1inv = np.diff(x1)
    P2inv = np.diff(x2)
    while i < (n1-1) and min < (n2-1):
        if x1[i] > x2[min]:
            if x1[i] <= x2[min+1]:
                P2[i] = 1/P2inv[min]
                i += 1
            else:
                min += 1
        else:
            i += 1
    P1inv = P1inv.squeeze()
    sqrt = np.sqrt((P2/n2)*P1inv*n1)
    return np.mean(sqrt)


def hellinger_distance1D(x1, x2):
    A_sym = (approx_affinity(x1, x2) + approx_affinity(x2, x1))/2
    H2 = 1 - (4/np.pi)*A_sym
    return np.sqrt(max(0, H2))
