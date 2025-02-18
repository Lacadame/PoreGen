import click
import poregen.metrics

@click.command()
@click.option('--datapath', type=click.Path(exists=True), required=True,
              help='Path to the folder containing the stats files')
def main(datapath):
    poregen.metrics.plot_unconditional_metrics(datapath=datapath, nbins=20, show_permeability=False)

    
if __name__ == "__main__":
    main()