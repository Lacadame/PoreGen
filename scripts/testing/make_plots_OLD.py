import poregen.metrics

def main():
    datapaths = [
        '/home/danilo/repos/PoreGen/savedmodels/experimental/20250113-dfn-ldm-doddington64/stats/stats-100-heun-100/model-epoch=031-val_loss=0.074421',
        '/home/danilo/repos/PoreGen/savedmodels/experimental/20250113-dfn-doddington64/stats/stats-100-heun-100/model-epoch=074-val_loss=0.007460'
    ]
    for datapath in datapaths:
        poregen.metrics.plot_unconditional_metrics(datapath=datapath, show_permeability=False)


if __name__ == "__main__":
    main()
