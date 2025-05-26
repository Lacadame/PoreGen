import pathlib
import os
import time

import click
import porespy
import numpy as np

import poregen.data

@click.command()
@click.option('--datapath', type=click.Path(exists=True), required=True,
              help='Path to the data file (e.g., Estaillades_1000c_3p31136um.raw)')
@click.option('--outdir', type=click.Path(exists=False), required=True,
              help='Path to the output directory')
@click.option('--scriptpath', type=click.Path(exists=True),
              default=os.path.join(os.path.expanduser('~'), 'repos/palabos/examples/tutorial/permeability/permeability'),
              help='Path to the Palabos script')
@click.option('--palabos_env', type=str, default='palabos_env',
              help='Name of the Palabos environment')
@click.option('--deltap', type=float, default=5e-6,
              help='Pressure drop for permeability calculation')
@click.option('--np_val', type=int, default=48,
              help='Number of processors')
@click.option('--use-cached', type=bool, default=True,
              help='Use cached permeability results')
def run_palabos_permeability(datapath, outdir, scriptpath, palabos_env, deltap, np_val, use_cached):
    """
    Run Palabos permeability calculation.
    """
    datapath = pathlib.Path(datapath)
    outdir = pathlib.Path(outdir)
    if str(datapath).endswith('.raw'):
        rock = poregen.data.load_binary_from_eleven_sandstones(
            datapath
        )
        normalizer = 255
    else:
        rock = np.load(datapath).astype(int)[0]
        normalizer = 1

    xshape, yshape, zshape = rock.shape

    os.makedirs(outdir, exist_ok=True)
    palabos_fname = outdir / 'rock.dat'
    print("Checking normalizer", normalizer, rock.mean())
    porespy.io.to_palabos(rock.transpose(2, 0, 1) // normalizer, palabos_fname, solid=1)   # writes ASCII grid

    output_file = outdir / 'output'
    if use_cached:
        print(f"Checking for cached permeability results in {output_file}")
        if output_file.exists():
            print(f"Using cached permeability results from {output_file}")
            return

    run_command = (
        f"bash -c 'source $(conda info --base)/etc/profile.d/conda.sh && "
        f"conda activate {palabos_env} && "
        f"time mpirun -np {np_val} {scriptpath} {palabos_fname} {outdir}/ "
        f"{xshape} {yshape} {zshape} {deltap} > {output_file} 2>&1'"
    )
    print(f"Running command: {run_command}")
    start_time = time.time()
    os.system(run_command)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Permeability calculation completed in {elapsed_time:.2f} seconds. Results saved to {output_file}")


if __name__ == '__main__':
    run_palabos_permeability()
