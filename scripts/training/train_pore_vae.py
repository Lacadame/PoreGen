import pathlib
import click

import poregen.data
import poregen.features
import poregen.models
import poregen.trainers


@click.command()
@click.option('--cfgpath', type=click.Path(exists=True), required=True,
              help='Path to the YAML configuration file')
@click.option('--datapath', type=click.Path(exists=True), default=None,
              help='Path to the data file (e.g., Estaillades_1000c_3p31136um.raw)'
                   'If not provided, the data will be loaded from the configuration file.')
@click.option('--checkpoint_path', type=str, default=None,
              help='Resume from a checkpoint. Use "last" (default in the '
                   'geomodel yaml) for last.ckpt after a crash, "best" for '
                   'lowest val_loss, or a .ckpt path. If omitted, uses '
                   'training.resume_from_checkpoint in the yaml. Missing '
                   'last.ckpt starts a new run.')
@click.option('--fast_dev_run', is_flag=True, default=False,
              help='If set, disables saving and trains only a few steps.')
def train(datapath, cfgpath, checkpoint_path, fast_dev_run):
    """
    Train a pore generation model using the specified data and configuration.
    """
    if datapath is not None:
        datapath = pathlib.Path(datapath)
    cfgpath = pathlib.Path(cfgpath)

    # Run training
    poregen.trainers.pore_vae_train(cfgpath,
                                    datapath,
                                    checkpoint_path=checkpoint_path,
                                    fast_dev_run=fast_dev_run)

    click.echo("Training completed.")


if __name__ == '__main__':
    train()
