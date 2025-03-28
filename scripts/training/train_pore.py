import pathlib
import click
import sys
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
              help='Path to checkpoint for resuming training. '
                   'Use "best" for the best checkpoint, or provide a specific path. Default is None.')
@click.option('--fast_dev_run', is_flag=True, default=False,
              help='If set, disables saving and trains only a few steps.')
@click.option('--load_on_fit', is_flag=True, default=False,
              help='If set, loads the checkpoint only on fit, not on initialization.')
def train(datapath, cfgpath, checkpoint_path, fast_dev_run, load_on_fit):
    """
    Train a pore generation model using the specified data and configuration.
    """
    try:
        if datapath is not None:
            datapath = pathlib.Path(datapath)
        cfgpath = pathlib.Path(cfgpath)

        # Run training
        poregen.trainers.pore_train(cfgpath,
                                    datapath,
                                    checkpoint_path=checkpoint_path,
                                    fast_dev_run=fast_dev_run,
                                    load_on_fit=load_on_fit)

        click.echo("Training completed.")
        sys.exit(0)  # Successful completion

    except KeyboardInterrupt:
        click.echo("Training interrupted by user.")
        sys.exit(1)  # Interrupted
#    except Exception as e:
#        click.echo(f"Training failed with error: {str(e)}")
#        sys.exit(2)  # Other error


if __name__ == '__main__':
    train()