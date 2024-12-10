from collections import defaultdict
import os
import pathlib

import joblib
import torch
import numpy as np
from scipy import fftpack
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


def calculate_3d_radial_spectrum(cubes, window_ratio=0.95, sigma=2.0):
    """
    Calculate the radial Fourier spectrum of 3D data cubes while handling border effects.

    Parameters:
    -----------
    cubes : list of numpy.ndarray or numpy.ndarray
        Input 3D data cube(s). If a single cube is provided, it will be converted to a list
    window_ratio : float, optional
        Ratio of the cube to use (excluding borders), default is 0.95
    sigma : float, optional
        Standard deviation for Gaussian window, default is 2.0

    Returns:
    --------
    freqs : numpy.ndarray
        Radial frequencies
    spectra : numpy.ndarray
        Radial power spectra for each cube (2D array: n_cubes × n_frequencies)
    """
    if isinstance(cubes, torch.Tensor):
        cubes = cubes.detach().cpu().numpy()
    elif isinstance(cubes[0], torch.Tensor):
        cubes = [cube.detach().cpu().numpy() for cube in cubes]
    # Convert single cube to list
    if isinstance(cubes, np.ndarray) and cubes.ndim == 3:
        cubes = [cubes]

    # Input validation
    for i, cube in enumerate(cubes):
        if not isinstance(cube, np.ndarray) or cube.ndim != 3:
            raise ValueError(f"Input cube {i} must be a 3D numpy array")

    # Process each cube
    all_spectra = []
    freqs = None

    for _, cube in enumerate(cubes):
        depth, rows, cols = cube.shape
        depth_center = depth // 2
        row_center = rows // 2
        col_center = cols // 2

        # Create 3D Gaussian window
        z, y, x = np.ogrid[-depth_center:depth_center,
                           -row_center:row_center,
                           -col_center:col_center]
        gaussian_window = np.exp(-(x*x + y*y + z*z) / (2*sigma*sigma))

        # Apply window to cube
        windowed_cube = cube * gaussian_window

        # Compute 3D FFT
        fft3 = fftpack.fftn(windowed_cube)
        fft3_shifted = fftpack.fftshift(fft3)

        # Calculate power spectrum
        power_spectrum = np.abs(fft3_shifted)**2

        # Create frequency grid
        freq_z = fftpack.fftfreq(depth, d=1)
        freq_y = fftpack.fftfreq(rows, d=1)
        freq_x = fftpack.fftfreq(cols, d=1)

        freq_z_shifted = fftpack.fftshift(freq_z)
        freq_y_shifted = fftpack.fftshift(freq_y)
        freq_x_shifted = fftpack.fftshift(freq_x)

        # Create meshgrid for radial calculation
        fz, fy, fx = np.meshgrid(freq_z_shifted,
                                 freq_y_shifted,
                                 freq_x_shifted,
                                 indexing='ij')
        radial_freq = np.sqrt(fx*fx + fy*fy + fz*fz)

        # Calculate radial average
        max_freq = min(depth, rows, cols) // 2
        freq_bins = np.linspace(0, np.max(radial_freq), max_freq)
        spectrum = np.zeros_like(freq_bins)

        for i in range(len(freq_bins)-1):
            freq_mask = (radial_freq >= freq_bins[i]) & (radial_freq < freq_bins[i+1])
            if freq_mask.any():
                spectrum[i] = np.mean(power_spectrum[freq_mask])

        all_spectra.append(spectrum[:-1])
        if freqs is None:
            freqs = freq_bins[:-1]

    return freqs, np.array(all_spectra)


def cluster_spectra(spectra_data, n_clusters=2, random_state=42):
    """
    Perform time series clustering on spectral data using tslearn.

    Parameters:
    -----------
    spectra_data : dict
        Nested dictionary containing spectral data organized by stone and subpath
    n_clusters : int
        Number of clusters to create
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    dict
        Contains clustering results and analysis
    """
    # Prepare data for clustering
    # We'll stack all spectra into a 3D array: [n_samples, n_timestamps, n_features]
    all_spectra = []
    stone_subpath_map = []  # Keep track of which spectrum belongs to which stone/subpath

    # First, verify all spectra have the same shape
    shapes = set()
    for stone, subpaths in spectra_data.items():
        for subpath, data in subpaths.items():
            shapes.add(data['spectra'].shape)
    if len(shapes) > 1:
        raise ValueError(f"Found inconsistent spectra shapes: {shapes}")

    # Stack the data
    for stone, subpaths in spectra_data.items():
        for subpath, data in subpaths.items():
            # Reshape spectra to [n_timestamps, n_features]
            spectra = data['spectra'].T  # Transpose to make time series the first dimension
            all_spectra.append(spectra)
            stone_subpath_map.append((stone, subpath))

    # Convert to numpy array
    X = np.array(all_spectra)

    # Normalize the time series
    scaler = TimeSeriesScalerMeanVariance()
    X_scaled = scaler.fit_transform(X)

    # Perform clustering
    km = TimeSeriesKMeans(n_clusters=n_clusters,
                          metric="dtw",  # Dynamic Time Warping
                          random_state=random_state)
    labels = km.fit_predict(X_scaled)

    # Organize results
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        stone, subpath = stone_subpath_map[idx]
        clusters[int(label)].append({
            'stone': stone,
            'subpath': subpath,
            'spectra': all_spectra[idx]
        })

    # Calculate cluster statistics
    cluster_stats = {}
    for label, members in clusters.items():
        cluster_spectra = np.array([m['spectra'] for m in members])
        cluster_stats[label] = {
            'size': len(members),
            'mean_spectrum': np.mean(cluster_spectra, axis=0),
            'std_spectrum': np.std(cluster_spectra, axis=0),
            'members': members
        }

    return {
        'clusters': dict(clusters),
        'cluster_stats': cluster_stats,
        'labels': labels,
        'model': km,
        'stone_subpath_map': stone_subpath_map,
        'scaler': scaler
    }


def predict_cluster_spectra(model, scaler, new_data):
    """
    Predict cluster for new spectral data.

    Parameters:
    -----------
    model : TimeSeriesKMeans
        Fitted clustering model
    scaler : TimeSeriesScalerMeanVariance
        Fitted scaler used during training
    new_data : numpy.ndarray
        New spectral data to classify. Should be in the same format as training data

    Returns:
    --------
    int
        Predicted cluster label
    """
    # Reshape if necessary
    if len(new_data.shape) == 2:
        new_data = new_data[np.newaxis, :]  # Add sample dimension

    # Scale the new data using the same scaler
    new_data_scaled = scaler.transform(new_data)

    # Predict cluster
    return model.predict(new_data_scaled)[0]


def save_cluster_model(results, save_dir):
    """
    Save the clustering model and scaler to disk.

    Parameters:
    -----------
    results : dict
        Output from cluster_spectra function
    save_dir : str or Path
        Directory to save the model files
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save model and scaler
    joblib.dump(results['model'], os.path.join(save_dir, 'cluster_model.joblib'))
    joblib.dump(results['scaler'], os.path.join(save_dir, 'scaler.joblib'))

    # Save cluster statistics as numpy arrays
    np.save(os.path.join(save_dir, 'cluster_stats.npy'), {
        label: {
            'mean_spectrum': stats['mean_spectrum'],
            'std_spectrum': stats['std_spectrum'],
            'size': stats['size']
        }
        for label, stats in results['cluster_stats'].items()
    }, allow_pickle=True)


def load_cluster_model(load_dir):
    """
    Load the saved clustering model and scaler.

    Parameters:
    -----------
    load_dir : str or Path
        Directory containing the saved model files

    Returns:
    --------
    dict
        Contains loaded model, scaler, and cluster statistics
    """
    model = joblib.load(os.path.join(load_dir, 'cluster_model.joblib'))
    scaler = joblib.load(os.path.join(load_dir, 'scaler.joblib'))
    cluster_stats = np.load(os.path.join(load_dir, 'cluster_stats.npy'),
                            allow_pickle=True).item()

    return {
        'model': model,
        'scaler': scaler,
        'cluster_stats': cluster_stats
    }


def get_model_load_dir(model: str):
    if model == 'CLUSTER-256-1':
        return pathlib.Path(__file__).parent / 'cluster_data' / 'clusterer256.joblib'
    else:
        raise ValueError(f'Unknown model: {model}')


def power_spectrum_criteria(samples: torch.Tensor, model='CLUSTER-256-1'):
    # sample: torch.Tensor of shape (N, 4, H, W, D)

    results = []
    for sample in samples:
        sample = sample.detach().cpu().numpy()[None, ...]
        spectra = np.concatenate([
            calculate_3d_radial_spectrum(sample[0, i, ...])[1]
            for i in range(sample.shape[1])
        ])
        spectra = np.log(spectra)
        load_dir = get_model_load_dir(model)
        model = load_cluster_model(load_dir)
        res = predict_cluster_spectra(model['model'], model['scaler'], spectra.T)
        results.append(res)
    return torch.tensor(results, dtype=torch.bool)
