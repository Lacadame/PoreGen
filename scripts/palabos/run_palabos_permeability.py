import pathlib
import os

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
              default=os.path.join(os.path.expanduser('~'), 'local/palabos/examples/tutorial/permeability/permeability'),
              help='Path to the Palabos script')
@click.option('--palabos_env', type=str, default='palabos_env',
              help='Name of the Palabos environment')
@click.option('--deltap', type=float, default=5e-6,
              help='Pressure drop for permeability calculation')
@click.option('--np_val', type=int, default=48,
              help='Number of processors')
def run_palabos_permeability(datapath, outdir, scriptpath, palabos_env, deltap, np_val):
    """
    Run Palabos permeability calculation.
    """
    datapath = pathlib.Path(datapath)
    outdir = pathlib.Path(outdir)
    if str(datapath).endswith('.raw'):
        rock = poregen.data.load_binary_from_eleven_sandstones(
            datapath
        )
    else:
        rock = np.load(datapath).astype(int)[0]

    xshape, yshape, zshape = rock.shape

    os.makedirs(outdir, exist_ok=True)
    palabos_fname = outdir / 'rock.dat'
    porespy.io.to_palabos(rock.transpose(2, 0, 1) // 255, palabos_fname, solid=1)   # writes ASCII grid

    output_file = outdir / 'output'
    run_command = (
        f"bash -c 'source $(conda info --base)/etc/profile.d/conda.sh && "
        f"conda activate {palabos_env} && "
        f"mpirun -np {np_val} {scriptpath} {palabos_fname} {outdir}/ "
        f"{xshape} {yshape} {zshape} {deltap} > {output_file}'"
    )
    os.system(run_command)
    print(f"Permeability calculation completed. Results saved to {output_file}")


if __name__ == '__main__':
    run_palabos_permeability()
