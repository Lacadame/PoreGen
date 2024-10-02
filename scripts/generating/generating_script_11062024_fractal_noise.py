import pathlib
import os

import numpy as np
import porespy
import pytictoc


CURRENTPATH = pathlib.Path(__file__).parent.absolute()
MAINPATH = CURRENTPATH.parent.parent
SYNTHETICDATAPATH = MAINPATH/"saveddata"/"raw"/"synthetic"


def generate_fractal_image(size, porosity, frequency, octaves, gain):
    image = porespy.generators.fractal_noise([size, size, size],
                                             frequency=frequency,
                                             octaves=octaves,
                                             gain=gain)
    image = image > (1 - porosity)
    return image


def fractal_image_parameters_sampler():
    porosity = np.random.uniform(0.05, 0.5)
    frequency = np.random.exponential(0.05) + 0.01
    octaves = np.random.randint(1, 6)
    gain = np.random.uniform(0.1, 0.5)
    return porosity, frequency, octaves, gain


def two_point_correlation_estimate(image):
    size = image.shape[-1]
    data2d = porespy.metrics.two_point_correlation(image[:, :, size//2])
    distance = data2d.distance
    probscaled = data2d.probability_scaled
    return distance, probscaled


def generate_and_save_sample(index, run_id):
    porosity, frequency, octaves, gain = fractal_image_parameters_sampler()
    image = generate_fractal_image(128, porosity, frequency, octaves, gain)
    x, y = two_point_correlation_estimate(image)
    basename = (
        f"sample_{str(index).zfill(8)}"
        f"_p{porosity:.4f}"
        f"_f{frequency:.4f}"
        f"_o{octaves}_g{gain:.4f}"
        f".npz"
    )
    savedir = SYNTHETICDATAPATH/f"run_{str(run_id).zfill(3)}"
    try:
        os.makedirs(savedir)
    except FileExistsError:
        pass
    np.savez(savedir/basename, image=image, x=x, y=y)


if __name__ == "__main__":
    tictoc = pytictoc.TicToc()
    tictoc.tic()
    run_id = 2
    for i in range(int(5*1e5)):
        generate_and_save_sample(i, 2)
    tictoc.toc()
